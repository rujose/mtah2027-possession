"""Lab candidate `segments`: possession-level clean-up, wrong team and spurious possessions (docs/lab/segments.md).

    python -m bench.lab_segments [--set all|idsse]   # ~30 s: leave one match out over the grid, judged
    python -m bench.lab_segments --oracle            # ~10 s, exploration: the same actions with the truth's answers
    python -m bench.lab_segments --nested [--set ..] # ~10 min, after the run: settings picked with no model trained on the held-out match

Each predicted possession (a maximal run of H or A inside a half) is one object. Two classifiers,
trained on the other bench matches only, read it from prediction-time inputs (its length, the ball,
each team's nearest player, touches and passes by each team inside it, PathCRF, the decoder's costs,
P(dead) inside and around it, its neighbours, where the ball is) and give

- P(wrong): the possession's team isn't the truth's majority team over its frames;
- P(spurious): no true possession of its team overlaps it at IoU >= 0.3 (the metric's lowest bar).

Then, with thresholds chosen on the bench:

- **flip**: where P(wrong) > FLIP, the possession becomes the other team's (so it joins any
  neighbour of that team: a blip inside the other team's possession disappears);
- **merge**: where P(spurious) > MERGE and it wasn't flipped, its frames go to its neighbours: to the
  other team if a neighbour is the other team's (mode "neighbour"; mode "dead" only when both are),
  otherwise to D.

A post stage: `fit(matches, letters_of, **setting)` and `apply(match, letters, model, **setting)` work
on any input letters (H/A/D/X per frame) and read only the match's damaged tracking and the
out-of-sample outputs on it, never its truth (truth is read by `targets`, for training on the other
matches and for scoring).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from bench import lab
from possession import touches, v9, v11
from possession.data import FPS, ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params (by reasoning, fixed before any held-out run; docs/lab/segments.md gives the why of each)
STAGE = "post"
NAME = "segments"
DV = 3.0                    # m/s: v11's touch ΔV (both sets), so a "touch" means what it means there
REACH = touches.REACH       # 2 m: "on the ball"
FAR = 30.0                  # m: cap on a nearest distance (no player of the team seen near)
AROUND = int(2 * FPS)       # frames before and after a possession where P(dead) is read
START = int(0.5 * FPS)      # frames either side of its first frame where a touch by its team counts as "at the start"
IOU = 0.3                   # spurious: no true possession of its team at this IoU (the lowest the metric counts)
OFF = 1.0                   # a threshold no probability exceeds: the action is off
CLASSIFIER = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 8, "min_samples_leaf": 40,
              "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}
OTHER = {"H": "A", "A": "H"}
SIDE = {"H": 0, "A": 1}

COMMON = ["log_len", "seen", "near_share", "reach_t", "reach_o", "dist_t", "dist_o",
          "touch_t", "touch_o", "pass_t", "pass_o", "touch_start",
          "pdead_in", "pdead_before", "pdead_after",
          "left_d", "left_o", "right_d", "right_o", "left_len", "right_len", "sandwich",
          "x_toward", "y_off", "restart_t"]
FEATURES = {
    # PathCRF's P(team) and the v7 decoder's costs (which include PathCRF's clue)
    "all": COMMON + ["p_mean", "p_share", "p_cover", "margin_o_mean", "margin_o_sum", "margin_d_mean"],
    # neither: PathCRF was pretrained on 6 of the 7 IDSSE matches, so it's too good there; v6's costs instead
    "no_pathcrf": COMMON + ["margin6_o_mean", "margin6_o_sum", "margin6_d_mean"],
}

# The grid, simplest first: the no-op (v11 unchanged), then flips alone, merges alone, both; for each
# feature set. FLIP near 0.5 (a wrong team flipped gains about what a right one flipped loses); MERGE
# higher (at spellF1 ~0.57 removing a true match costs ~2.5 times what removing a spurious one gains,
# so the break-even is near P = 0.7).
FLIPS = (OFF, 0.5, 0.6, 0.7, 0.8)
MERGES = (OFF, 0.6, 0.7, 0.8, 0.9)
MODES = ("neighbour", "dead")
COLUMNS = ["features", "flip", "merge", "mode"]


def _grid():
    out = [{"features": "all", "flip": OFF, "merge": OFF, "mode": "neighbour"}]
    for features in FEATURES:
        settings = [(f, OFF, "neighbour") for f in FLIPS if f < OFF]
        settings += [(OFF, g, mode) for g in MERGES if g < OFF for mode in MODES]
        settings += [(f, g, mode) for f in FLIPS if f < OFF for g in MERGES if g < OFF for mode in MODES]
        out += [{"features": features, "flip": f, "merge": g, "mode": mode} for f, g, mode in settings]
    return out


GRID = _grid()


# --------------------------------------------------------------------------------------------
# The possessions and their features (prediction-time inputs only)
# --------------------------------------------------------------------------------------------

def filled(letters, halves):
    """Letters with X forward-filled inside each half (back-filled at its head), as the scorer reads them."""
    out = np.asarray(letters).astype("<U1").copy()
    for _, first, last in halves:
        w = out[first:last + 1]
        known = np.flatnonzero(w != "X")
        if len(known) == 0 or len(known) == len(w):
            continue
        carry = np.maximum.accumulate(np.where(w != "X", np.arange(len(w)), -1))
        carry[carry < 0] = known[0]
        out[first:last + 1] = w[carry]
    return out


def possessions(letters, halves):
    """[(start, stop, team, left, right, left length, right length, first, last)]: each maximal H or A
    run [start, stop) inside a half [first, last], with its neighbours' labels ("E" at the half's edge)."""
    out = []
    for _, first, last in halves:
        w = letters[first:last + 1]
        bounds = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1, len(w)]
        runs = [(first + int(bounds[i]), first + int(bounds[i + 1]), str(w[bounds[i]])) for i in range(len(bounds) - 1)]
        for i, (a, b, k) in enumerate(runs):
            if k not in SIDE:
                continue
            left = runs[i - 1] if i > 0 else (a, a, "E")
            right = runs[i + 1] if i + 1 < len(runs) else (b, b, "E")
            out.append((a, b, k, left[2], right[2], left[1] - left[0], right[1] - right[0], first, last))
    return out


def signals(match):
    """Per-frame inputs every possession reads, independent of the letters (computed once per match)."""
    cached = getattr(match, "_segments_signals", None)
    if cached is not None:
        return cached
    tr = match.track
    ball, players, team = tr["ball"], tr["players"], tr["team"]
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    near = np.minimum(np.stack([touches.nearest(players, team, ball, s) for s in (0, 1)], axis=1), FAR)
    found = v11.seen_touches(tr, DV)
    passed = v9.seen_passes(tr)
    # Attacking direction per half: a team attacks toward +x when its players' mean x is below the other's
    # (its keeper and back line sit in its own half).
    plus = np.zeros((n, 2), bool)
    for _, first, last in match.halves:
        mean_x = [np.nanmean(players[first:last + 1, team == s, 0]) for s in (0, 1)]
        plus[first:last + 1, 0], plus[first:last + 1, 1] = mean_x[0] < mean_x[1], mean_x[1] < mean_x[0]
    x_toward = np.where(plus, ball[:, [0]], match.pitch[0] - ball[:, [0]])
    out = {"seen": seen, "near": near, "touch": np.stack([found[0], found[1]], axis=1),
           "pass": np.stack([passed[0], passed[1]], axis=1), "x_toward": x_toward,
           "y_off": np.abs(ball[:, 1] - match.pitch[1] / 2), "p_home": match.pred["p_pathcrf"],
           "cost7": match.pred["v7_cost"].astype(float), "cost6": match.pred["v6_cost"].astype(float),
           "p_dead": match.pred["p_dead"], "p_start": match.pred["p_start"]}
    match._segments_signals = out
    return out


def _mean(x):
    return float(np.mean(x)) if len(x) else np.nan


def table(match, letters):
    """One row per predicted possession in `letters`: where it is and its features."""
    sig = signals(match)
    letters = filled(letters, match.halves)
    rows = []
    for a, b, k, left, right, left_len, right_len, first, last in possessions(letters, match.halves):
        t, o = SIDE[k], 1 - SIDE[k]
        n = b - a
        s = sig["seen"][a:b]
        d_t, d_o = sig["near"][a:b, t][s], sig["near"][a:b, o][s]
        p_t = sig["p_home"][a:b] if k == "H" else 1 - sig["p_home"][a:b]
        p_t = p_t[~np.isnan(p_t)]
        before, after = sig["p_dead"][max(a - AROUND, first):a], sig["p_dead"][b:min(b + AROUND, last + 1)]
        row = {"start": a, "stop": b, "team": k, "first": first, "last": last,
               "log_len": np.log(n), "seen": s.mean(),
               "near_share": _mean(d_t < d_o), "reach_t": _mean(d_t < REACH), "reach_o": _mean(d_o < REACH),
               "dist_t": float(np.median(d_t)) if len(d_t) else np.nan,
               "dist_o": float(np.median(d_o)) if len(d_o) else np.nan,
               "touch_t": int(sig["touch"][a:b, t].sum()), "touch_o": int(sig["touch"][a:b, o].sum()),
               "pass_t": int(sig["pass"][a:b, t].sum()), "pass_o": int(sig["pass"][a:b, o].sum()),
               "touch_start": float(sig["touch"][max(a - START, first):a + START + 1, t].any()),
               "pdead_in": float(sig["p_dead"][a:b].mean()), "pdead_before": _mean(before), "pdead_after": _mean(after),
               "left_d": float(left == "D"), "left_o": float(left == OTHER[k]),
               "right_d": float(right == "D"), "right_o": float(right == OTHER[k]),
               "left_len": np.log1p(left_len), "right_len": np.log1p(right_len),
               "sandwich": float(left == OTHER[k] and right == OTHER[k]),
               "x_toward": _mean(sig["x_toward"][a:b, t][s]), "y_off": _mean(sig["y_off"][a:b][s]),
               "restart_t": float(sig["p_start"][a] if k == "H" else 1 - sig["p_start"][a]),
               "p_mean": _mean(p_t), "p_share": _mean(p_t > 0.5), "p_cover": len(p_t) / n}
        for name, cost in (("margin", sig["cost7"][a:b]), ("margin6", sig["cost6"][a:b])):
            gap = cost[:, o] - cost[:, t]
            row[name + "_o_mean"], row[name + "_o_sum"] = float(gap.mean()), float(gap.sum())
            row[name + "_d_mean"] = float((cost[:, 2] - cost[:, t]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def targets(match, tab):
    """The truth's answers per possession of `tab` (training on other matches, and reporting only):
    wrong (the truth's majority team over its H/A frames is the other one), spurious (no true
    possession of its team at IoU >= IOU), and the best IoU."""
    truth = match.truth_letters
    spells = [(a, b, lab.LETTERS[k]) for a, b, k in pm.possession_spells(match.truth, pm.valid_runs(match.truth))]
    starts = np.array([s[0] for s in spells])
    stops = np.array([s[1] for s in spells])
    teams = np.array([s[2] for s in spells])
    wrong, spurious, best = [], [], []
    for a, b, k in zip(tab.start, tab.stop, tab.team):
        w = truth[a:b]
        n_t, n_o = int((w == k).sum()), int((w == OTHER[k]).sum())
        wrong.append(n_o > n_t)
        hit = (teams == k) & (starts < b) & (stops > a)
        iou = (np.minimum(stops[hit], b) - np.maximum(starts[hit], a)) / (np.maximum(stops[hit], b) - np.minimum(starts[hit], a))
        best.append(float(iou.max()) if len(iou) else 0.0)
        spurious.append(best[-1] < IOU)
    return pd.DataFrame({"wrong": wrong, "spurious": spurious, "best_iou": best}, index=tab.index)


# --------------------------------------------------------------------------------------------
# The interface: fit on some matches, apply to another
# --------------------------------------------------------------------------------------------

def train(tables, features):
    """The two classifiers on the rows of `tables` (each a table() with targets() joined)."""
    rows = pd.concat(tables, ignore_index=True)
    x = rows[FEATURES[features]].to_numpy(float)
    return {"features": features,
            "wrong": HistGradientBoostingClassifier(**CLASSIFIER).fit(x, rows["wrong"].to_numpy(int)),
            "spurious": HistGradientBoostingClassifier(**CLASSIFIER).fit(x, rows["spurious"].to_numpy(int))}


def fit(matches, letters_of=None, features="all", **_):
    """The model, trained on `matches` (any iterable; each read once) with their input letters
    `letters_of(match)` (default match.v11). Only `features` matters: the thresholds are apply's."""
    letters_of = letters_of or (lambda m: m.v11)
    tables = []
    for match in matches:
        tab = table(match, letters_of(match))
        tables.append(pd.concat([tab, targets(match, tab)], axis=1))
    return train(tables, features)


def probabilities(tab, model):
    """(P(wrong), P(spurious)) per row of `tab`."""
    if len(tab) == 0:
        return np.zeros(0), np.zeros(0)
    x = tab[FEATURES[model["features"]]].to_numpy(float)
    return model["wrong"].predict_proba(x)[:, 1], model["spurious"].predict_proba(x)[:, 1]


def act(letters, tab, p_wrong, p_spurious, flip=OFF, merge=OFF, mode="neighbour"):
    """(letters after the actions, [flipped rows], [merged rows]). Flips first (all at once), then
    merges from the most likely spurious down, each reading its neighbours in the labels as they
    are by then; a possession that a flipped neighbour has joined is left alone."""
    out = np.asarray(letters).astype("<U1").copy()
    flipped = [i for i in range(len(tab)) if p_wrong[i] > flip]
    for i in flipped:
        out[tab.start.iat[i]:tab.stop.iat[i]] = OTHER[tab.team.iat[i]]
    merged, skip = [], set(flipped)
    for i in np.argsort(-p_spurious, kind="stable"):
        if p_spurious[i] <= merge:
            break
        if i in skip:
            continue
        a, b, k, first, last = tab.start.iat[i], tab.stop.iat[i], tab.team.iat[i], tab.first.iat[i], tab.last.iat[i]
        left = out[a - 1] if a > first else "E"
        right = out[b] if b <= last else "E"
        if left == k or right == k:
            continue
        o = OTHER[k]
        if left == o and right == o:
            fill = o                       # a blip inside the other team's possession
        elif mode == "neighbour" and (left == o or right == o):
            fill = o                       # the other team's possession next to it takes it
        else:
            fill = "D"
        out[a:b] = fill
        merged.append(int(i))
    return out, flipped, merged


def apply(match, letters, model, flip=OFF, merge=OFF, mode="neighbour", **_):
    """Letters for every frame of `match`: `letters` with the possessions flipped or merged away."""
    letters = np.asarray(letters).astype("<U1")
    tab = table(match, letters)
    p_wrong, p_spurious = probabilities(tab, model)
    return act(letters, tab, p_wrong, p_spurious, flip, merge, mode)[0]


# --------------------------------------------------------------------------------------------
# The bench
# --------------------------------------------------------------------------------------------

def tables_of(which):
    """{match: its v11 table with the truth's answers joined}, one match loaded at a time."""
    out = {}
    for match in lab.each(which):
        tab = table(match, match.v11)
        out[match.id] = pd.concat([tab, targets(match, tab)], axis=1)
        print("{}: {} possessions, {} wrong team, {} spurious".format(
            match.id, len(tab), int(out[match.id].wrong.sum()), int(out[match.id].spurious.sum())), flush=True)
    return out


def oracle(which, tables):
    """Exploration: the actions with the truth's own answers (P = 0 or 1), on v11."""
    rows = []
    for match in lab.each(which):
        tab = tables[match.id]
        base = lab.score(match, match.v11)["spell_f1"]
        wrong, spurious = tab.wrong.to_numpy(float), tab.spurious.to_numpy(float)
        none = np.zeros(len(tab))
        for name, pw, ps, mode in (("flip", wrong, none, "neighbour"), ("merge, neighbour", none, spurious, "neighbour"),
                                   ("merge, dead", none, spurious, "dead"), ("both, neighbour", wrong, spurious, "neighbour"),
                                   ("both, dead", wrong, spurious, "dead")):
            letters = act(match.v11, tab, pw, ps, 0.5, 0.5, mode)[0]
            s = lab.score(match, letters)
            rows.append({"match": match.id, "oracle": name, "spell_f1": s["spell_f1"], "base": base,
                         "possessions": s["possessions"], "true": s["true_possessions"]})
    f = pd.DataFrame(rows)
    f["gain"] = f.spell_f1 - f.base
    print(f.groupby("oracle", sort=False).agg(spell_f1=("spell_f1", "mean"), gain=("gain", "mean"),
                                              better=("gain", lambda g: int((g > 0).sum())),
                                              possessions=("possessions", "mean"), true=("true", "mean")).round(4))


def nested(which, tables):
    """Robustness check of the same grid: lab.judge picks match m's setting from the other matches'
    scores, but their models were trained with m's truth in. Here every score that picks m's setting
    comes from models trained without m too (for each other match j: trained without m and j)."""
    grid = pd.read_csv(os.path.join(lab.RESULTS, "lab_{}_grid{}.csv".format(NAME, "" if which == "all" else "_" + which)))
    ms = list(tables)
    rows = []
    for match in lab.each(which):
        j, tab = match.id, tables[match.id]
        for m in ms:
            if m == j:
                continue
            models = {f: train([tables[x] for x in ms if x not in (m, j)], f) for f in FEATURES}
            probs = {f: probabilities(tab, models[f]) for f in FEATURES}
            for order, setting in enumerate(GRID):
                pw, ps = probs[setting["features"]]
                letters = act(match.v11, tab, pw, ps, setting["flip"], setting["merge"], setting["mode"])[0]
                rows.append({"without": m, "match": j, "order": order, "spell_f1": lab.score(match, letters)["spell_f1"]})
        print("{}: scored without each other match".format(j), flush=True)
    inner = pd.DataFrame(rows)
    base = lab.baseline(which)
    folds = []
    for m in ms:
        means = inner[inner.without == m].groupby("order").spell_f1.mean()
        order = int(means.round(6).idxmax())                 # ties: the first in grid order
        setting = GRID[order]
        row = grid[(grid.match == m) & np.all([grid[c] == setting[c] for c in COLUMNS], axis=0)].iloc[0]
        folds.append({"match": m, "source": base[m]["source"], "unseen": base[m]["unseen"], "picked": setting,
                      "held_out": float(row.spell_f1), "base": base[m]["spell_f1"]})
    f = pd.DataFrame(folds)
    for name, part in [("all", f)] + [(s, f[f.source == s]) for s in sorted(f.source.unique())] + [("PathCRF-unseen", f[f.unseen])]:
        print("nested {}: {:.4f} vs {:.4f} ({:+.4f}), better on {}/{}".format(
            name, part.held_out.mean(), part.base.mean(), part.held_out.mean() - part.base.mean(),
            int((part.held_out > part.base).sum()), len(part)))
    print(f[["match", "held_out", "base", "picked"]].to_string(index=False))


def sub_report(grid, keep, which):
    """lab.judge over the no-op plus the settings `keep` picks (flips alone, merges alone)."""
    rows = grid[((grid["flip"] == OFF) & (grid["merge"] == OFF)) | keep]
    r = lab.judge(rows.reset_index(drop=True), COLUMNS, which)
    return {k: r[k] for k in ("all", "by_source", "unseen", "macro", "chosen", "checks", "passes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--oracle", action="store_true", help="exploration: the actions with the truth's answers")
    parser.add_argument("--nested", action="store_true", help="robustness: settings picked with no model trained on the held-out match (after the main run)")
    args = parser.parse_args()
    which = args.which
    tables = tables_of(which)
    if args.oracle:
        oracle(which, tables)
        return
    if args.nested:
        nested(which, tables)
        return
    rows, quality = [], []
    for match in lab.each(which):
        tab = tables[match.id]
        models = {f: train([tables[m] for m in tables if m != match.id], f) for f in FEATURES}
        probs = {f: probabilities(tab, models[f]) for f in FEATURES}
        for f, (pw, ps) in probs.items():
            quality.append({"match": match.id, "features": f,
                            "auc_wrong": roc_auc_score(tab.wrong, pw) if tab.wrong.nunique() > 1 else np.nan,
                            "auc_spurious": roc_auc_score(tab.spurious, ps) if tab.spurious.nunique() > 1 else np.nan})
        for setting in GRID:
            pw, ps = probs[setting["features"]]
            letters, flipped, merged = act(match.v11, tab, pw, ps, setting["flip"], setting["merge"], setting["mode"])
            s = lab.score(match, letters)
            rows.append({"match": match.id, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                         "possessions": s["possessions"], "true_possessions": s["true_possessions"],
                         "flips": len(flipped), "flips_right": int(tab.wrong.iloc[flipped].sum()),
                         "merges": len(merged), "merges_right": int(tab.spurious.iloc[merged].sum())})
        print("{}: {} settings scored".format(match.id, len(GRID)), flush=True)
    grid = pd.DataFrame(rows)
    report = lab.judge(grid, COLUMNS, which)
    print(lab.describe(report))

    # What the held-out folds did, each with the setting picked without it
    picked = []
    for fold in report["folds"]:
        p = fold["picked"]
        row = grid[(grid.match == fold["match"]) & np.all([grid[c] == p[c] for c in COLUMNS], axis=0)].iloc[0]
        picked.append(row)
    picked = pd.DataFrame(picked)
    print(picked[["match"] + COLUMNS + ["spell_f1", "possessions", "true_possessions", "flips", "flips_right",
                                        "merges", "merges_right"]].to_string(index=False))
    q = pd.DataFrame(quality)
    print("held-out AUC by feature set:\n", q.groupby("features")[["auc_wrong", "auc_spurious"]].mean().round(3))
    print(q.round(3).to_string(index=False))

    # Flips alone and merges alone, judged the same way (diagnostic; the verdict is the full grid's)
    alone = {"flip": sub_report(grid, (grid["merge"] == OFF) & (grid["flip"] < OFF), which),
             "merge": sub_report(grid, (grid["flip"] == OFF) & (grid["merge"] < OFF), which)}
    for name, r in alone.items():
        a = r["all"]
        print("{} alone: held out {:.4f} vs {:.4f} ({:+.4f}), better {}/{}; unseen {:+.4f}; chosen {}; passes {}".format(
            name, a["held_out"], a["base"], a["held_out"] - a["base"], a["better"], a["matches"],
            r["unseen"]["held_out"] - r["unseen"]["base"] if r["unseen"] else float("nan"), r["chosen"], r["passes"]))
    report["alone"] = alone
    report["held_out_auc"] = q.to_dict("records")
    report["held_out_actions"] = picked[["match", "possessions", "true_possessions", "flips", "flips_right",
                                         "merges", "merges_right"]].to_dict("records")
    lab.save(NAME, grid, report, which)


if __name__ == "__main__":
    main()
