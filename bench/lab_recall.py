"""Lab round 2 candidate `recall`: missed possessions proposed broadly, each written in only if a
classifier trained on the other matches says it's a true one (docs/lab/recall.md).

    python -m bench.lab_recall [--set all|idsse]            # ~1 min: leave one match out over the grid, judged
    python -m bench.lab_recall --nested [--set all|idsse]   # ~1.5 min, after the run: settings picked with no
                                                            # model trained on the held-out match

The mix's biggest error left is missed possessions (+0.252 if all were found). Inside each predicted
possession of team O, three generators propose a stretch [k1, k2) for the other team N:
- "ctl": N "on the ball" (its nearest player within CTL of the seen ball and nearer than O's) for
  one or more frames, from N's first such frame to O's next one (O-N-O); or to the stoppage when
  O never comes back before it (O-N-D); or from the possession's start when it follows a stoppage
  and N is on the ball first (D-N-O: N restarts, O wins it);
- "touch": round 1's `short` candidates (a group of N touches after an O touch, to O's next touch or
  the stoppage), loose: any length 0.2-8 s, no clarity check;
- "pc": a stretch where PathCRF gives N more than half (feature set "all" only).
Each proposal is described by prediction-time evidence (how long and how close N is on the ball,
O's presence, touches and passes by each team, the ball's speed, PathCRF, the decoder's costs,
P(dead), where it sits in O's possession) and a classifier, trained leave one match out on the
other matches' proposals and the truth, gives P(a true possession of N matches it at IoU >= 0.3).
From the most likely down, each proposal at or above `p_min` that doesn't touch one already
written is written in.

A post stage: `fit(matches, letters_of, **setting)` and `apply(match, letters, model, **setting)`
work on any input letters and read only the match's damaged tracking and the out-of-sample outputs
on it (`match.pred`), never its truth (read by `targets`, for training on the other matches and
for reporting).
"""

import argparse
import json
import os
import types

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bench import chain, lab, lab_budget, lab_mix, lab_short
from possession import v9
from possession.data import FPS

# Params (by reasoning and the exploration in docs/lab/recall.md, fixed before any held-out run)
STAGE = "post"
NAME = "recall"
CTL = 1.0                      # m: "on the ball" for a proposal (short.md: within 1 m the truth agrees 61-70 %, at 1.5-2 m 6 %)
TIGHT = 0.5                    # m: "at the feet", a feature
MIN_LEN = int(0.2 * FPS)       # frames: shorter than this is under the touch detector's own look either side
MAX_LEN = int(8 * FPS)         # frames: 90 % of the missed possessions are under 4.8 s; past 8 s it isn't "missed short"
PC_MIN = int(0.3 * FPS)        # frames: a PathCRF stretch shorter than this is flicker
TOUCH_LENGTH = (0.2, 8.0)      # s: short's candidates, loose (short itself kept 0.5-5 s with its clarity check)
AFTER = FPS                    # frames after a proposal where P(dead) is read (a stoppage follows an O-N-D)
IOU = 0.3                      # the label: a true possession of N matches the proposal at the metric's lowest IoU
FAR = 30.0                     # m: cap on a nearest distance
GAP = 1                        # frames: proposals written must be this far apart (a 1-frame O between two N's is a blip)
OTHER = {"H": "A", "A": "H"}
SIDE = {"H": 0, "A": 1}
HGB = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 8, "min_samples_leaf": 40,
       "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}      # segments' settings, untuned
C = 1.0                        # the logistic regression's (sklearn's default, as short)

COMMON = ["log_len", "head", "tail", "src_ctl", "src_touch", "left_rem", "right_rem", "log_run",
          "ctl_n", "ctl_o", "ctl_share", "tight_n", "min_n", "med_n", "med_o", "near_share", "seen",
          "touch_n", "touch_o", "touch_first", "dv", "v_after", "cos", "pass_n", "pass_o", "speed",
          "pdead_in", "pdead_after", "edge_dist"]
FEATURES = {
    # PathCRF (a generator and features) and v7's decoder costs, which include PathCRF's clue
    "all": COMMON + ["src_pc", "pc_mean", "pc_share", "margin_mean", "margin_sum"],
    # neither: PathCRF was pretrained on 6 of the 7 IDSSE matches; v6's costs instead (no PathCRF clue)
    "no_pathcrf": COMMON + ["margin_mean", "margin_sum"],
}
MODELS = ("logistic", "hgb")
P_MINS = (0.4, 0.5, 0.6, 0.7, 0.8)
COLUMNS = ["features", "model", "p_min"]


def _grid():
    """Simplest first, the no-op first (a tie keeps the mix)."""
    out = [{"features": "all", "model": "none", "p_min": 1.0}]
    for features in ("no_pathcrf", "all"):
        for model in MODELS:
            for p_min in sorted(P_MINS, reverse=True):
                out.append({"features": features, "model": model, "p_min": p_min})
    return out


GRID = _grid()
# Chosen on all 10 matches, and by every fold (bench/results/lab_recall.json); also on `--set idsse`
CHOSEN = {"features": "all", "model": "hgb", "p_min": 0.4}


# --------------------------------------------------------------------------------------------
# Evidence (per match, independent of the letters) and proposals (on the letters)
# --------------------------------------------------------------------------------------------

def evidence(match):
    """Per-frame signals every proposal reads, computed once per match (cached on it)."""
    cached = getattr(match, "_recall_evidence", None)
    if cached is not None:
        return cached
    ev = dict(lab_short.evidence(match))         # near, touch (ΔV 3 m/s), before, after, seen, p_home
    seen = ev["seen"]
    with np.errstate(invalid="ignore"):
        ev["ctl"] = {s: seen & (ev["near"][s] < CTL) & (ev["near"][s] < ev["near"][OTHER[s]]) for s in SIDE}
    passed = v9.seen_passes(match.track)
    ev["pass"] = {s: passed[i] for s, i in SIDE.items()}
    ball = match.track["ball"]
    ev["speed"] = np.hypot(*ev["after"].T)
    length, width = match.pitch
    ev["edge"] = np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]])
    ev["cost7"], ev["cost6"] = match.pred["v7_cost"].astype(float), match.pred["v6_cost"].astype(float)
    ev["p_dead"] = match.pred["p_dead"]
    match._recall_evidence = ev
    return ev


def runs_in(letters, first, last):
    seg = letters[first:last + 1]
    bounds = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)] + first
    return [(int(bounds[i]), int(bounds[i + 1]), str(seg[bounds[i] - first])) for i in range(len(bounds) - 1)]


def _shape(a, b, start, stop, prev, nxt):
    """Whether [start, stop) inside O's run [a, b) is a usable proposal, and if it's at the head or tail.
    At the head only after a stoppage (D-N-O), at the tail only before one (O-N-D): at a change of
    team it's the change's timing, not a missed possession; the whole run is a wrong team."""
    head, tail = start == a, stop == b
    if (head and prev != "D") or (tail and nxt != "D") or (head and tail):
        return None
    if not MIN_LEN <= stop - start <= MAX_LEN:
        return None
    return head, tail


def ctl_proposals(letters, halves, ev):
    """[(start, stop, N, head, tail)]: N on the ball inside O's run, to O's next time on it."""
    out = []
    for _, first, last in halves:
        rs = runs_in(letters, first, last)
        for r, (a, b, own) in enumerate(rs):
            if own not in OTHER:
                continue
            new = OTHER[own]
            prev = rs[r - 1][2] if r else "E"
            nxt = rs[r + 1][2] if r + 1 < len(rs) else "E"
            cn, co = ev["ctl"][new][a:b], ev["ctl"][own][a:b]
            events = np.flatnonzero(cn | co)
            if not len(events):
                continue
            mine = cn[events].astype(np.int8)
            edges = np.flatnonzero(np.diff(np.r_[0, mine, 0]))
            for s, e in zip(edges[::2], edges[1::2]):
                start = a if s == 0 else a + int(events[s])
                stop = b if e == len(events) else a + int(events[e])
                shape = _shape(a, b, start, stop, prev, nxt)
                if shape:
                    out.append((start, stop, new) + shape)
    return out


def pc_proposals(letters, halves, ev):
    """[(start, stop, N, head, tail)]: PathCRF gives N more than half, inside O's run."""
    out = []
    p_home = np.nan_to_num(ev["p_home"], nan=0.5)
    for _, first, last in halves:
        rs = runs_in(letters, first, last)
        for r, (a, b, own) in enumerate(rs):
            if own not in OTHER:
                continue
            new = OTHER[own]
            prev = rs[r - 1][2] if r else "E"
            nxt = rs[r + 1][2] if r + 1 < len(rs) else "E"
            p = p_home[a:b] if new == "H" else 1 - p_home[a:b]
            edges = np.flatnonzero(np.diff(np.r_[0, (p > 0.5).astype(np.int8), 0]))
            for s, e in zip(edges[::2], edges[1::2]):
                if e - s < PC_MIN:
                    continue
                shape = _shape(a, b, a + int(s), a + int(e), prev, nxt)
                if shape:
                    out.append((a + int(s), a + int(e), new) + shape)
    return out


def proposals(letters, halves, ev, features="all"):
    """[(start, stop, N, head, tail, sources)]: every generator's proposals, the same stretch once."""
    letters = np.asarray(letters).astype("<U1")
    found = {}

    def add(items, source):
        for start, stop, new, head, tail in items:
            key = (start, stop, new)
            if key not in found:
                found[key] = {"head": head, "tail": tail, "sources": set()}
            found[key]["sources"].add(source)

    add(ctl_proposals(letters, halves, ev), "ctl")
    touch = [(k1, k2, new, False, bool(to_dead))          # after an O touch, so never at the head
             for k1, k2, new, to_dead in lab_short.candidates(letters, halves, ev, TOUCH_LENGTH, True)]
    add([t for t in touch if MIN_LEN <= t[1] - t[0] <= MAX_LEN], "touch")
    if features == "all":
        add(pc_proposals(letters, halves, ev), "pc")
    return [(s, e, n, v["head"], v["tail"], v["sources"]) for (s, e, n), v in sorted(found.items())]


def _mean(x):
    return float(np.mean(x)) if len(x) else np.nan


def table(match, letters, features="all"):
    """One row per proposal on `letters`: where it is and its features (prediction-time inputs only)."""
    ev = evidence(match)
    letters = np.asarray(letters).astype("<U1")
    cost = ev["cost7"] if features == "all" else ev["cost6"]
    run_of = {}
    for _, first, last in match.halves:
        for a, b, k in runs_in(letters, first, last):
            run_of[a] = (a, b, first, last)
    starts = np.array(sorted(run_of))
    rows = []
    for k1, k2, new, head, tail, sources in proposals(letters, match.halves, ev, features):
        own = OTHER[new]
        a, b, first, last = run_of[int(starts[np.searchsorted(starts, k1, side="right") - 1])]
        s = ev["seen"][k1:k2]
        dn, do = np.minimum(ev["near"][new][k1:k2], FAR), np.minimum(ev["near"][own][k1:k2], FAR)
        dn_s, do_s = dn[s], do[s]
        tn = np.flatnonzero(ev["touch"][new][max(k1 - lab_short.LOOK, first):k2]) + max(k1 - lab_short.LOOK, first)
        if len(tn):
            k = tn[0]
            vb, va = ev["before"][k], ev["after"][k]
            dv, v_after = float(np.hypot(*(va - vb))), float(np.linalg.norm(va))
            cos = float(np.dot(vb, va) / (np.linalg.norm(vb) * np.linalg.norm(va) + 1e-9))
        else:
            dv, v_after, cos = np.nan, np.nan, np.nan
        p = ev["p_home"][k1:k2] if new == "H" else 1 - ev["p_home"][k1:k2]
        p = p[~np.isnan(p)]
        gap = cost[k1:k2, SIDE[own]] - cost[k1:k2, SIDE[new]]       # > 0: the decoder found N cheaper
        edge = ev["edge"][k1:k2][s]
        n_ctl = int(ev["ctl"][new][k1:k2].sum())
        o_ctl = int(ev["ctl"][own][k1:k2].sum())
        rows.append({
            "start": k1, "stop": k2, "team": new, "run_start": a, "run_stop": b,
            "log_len": np.log((k2 - k1) / FPS), "head": float(head), "tail": float(tail),
            "src_ctl": float("ctl" in sources), "src_touch": float("touch" in sources), "src_pc": float("pc" in sources),
            "left_rem": (k1 - a) / FPS, "right_rem": (b - k2) / FPS, "log_run": np.log((b - a) / FPS),
            "ctl_n": n_ctl, "ctl_o": o_ctl, "ctl_share": n_ctl / max(int(s.sum()), 1),
            "tight_n": int(np.sum(dn_s < TIGHT)),
            "min_n": float(dn_s.min()) if len(dn_s) else FAR, "med_n": float(np.median(dn_s)) if len(dn_s) else FAR,
            "med_o": float(np.median(do_s)) if len(do_s) else FAR, "near_share": _mean(dn_s < do_s), "seen": float(s.mean()),
            "touch_n": len(tn), "touch_o": int(ev["touch"][own][k1:k2].sum()),
            "touch_first": float(len(tn) > 0 and tn[0] <= k1 + lab_short.LOOK),
            "dv": dv, "v_after": v_after, "cos": cos,
            "pass_n": int(ev["pass"][new][k1:k2].sum()), "pass_o": int(ev["pass"][own][k1:k2].sum()),
            "speed": _mean(ev["speed"][k1:k2][~np.isnan(ev["speed"][k1:k2])]),
            "pdead_in": float(np.mean(ev["p_dead"][k1:k2])), "pdead_after": _mean(ev["p_dead"][k2:min(k2 + AFTER, last + 1)]),
            "edge_dist": float(edge[-1]) if len(edge) else np.nan,
            "pc_mean": _mean(p), "pc_share": _mean(p > 0.5),
            "margin_mean": float(gap.mean()), "margin_sum": float(gap.sum())})
    columns = ["start", "stop", "team", "run_start", "run_stop"] + FEATURES["all"]
    return pd.DataFrame(rows, columns=columns)


def targets(match, tab):
    """The truth's answer per proposal (training on other matches, and reporting only): a true
    possession of its team matches it at IoU >= IOU, and the best IoU."""
    spells = lab_budget.spells(match.truth_letters.astype("<U1"))
    best = np.array([lab_budget.best_iou((a, b, k), spells) for a, b, k in zip(tab.start, tab.stop, tab.team)])
    return pd.DataFrame({"right": best >= IOU, "best_iou": best}, index=tab.index)


# --------------------------------------------------------------------------------------------
# The interface: fit on some matches, apply to another
# --------------------------------------------------------------------------------------------

def design(tab, features):
    """The feature matrix (NaN where a value doesn't exist: HGB handles it, the regression imputes)."""
    return tab[FEATURES[features]].to_numpy(float)


def train(tables, features, model):
    """The classifier on the rows of `tables` (each a table() with targets() joined)."""
    rows = pd.concat(tables, ignore_index=True)
    x, y = design(rows, features), rows["right"].to_numpy(int)
    if model == "hgb":
        clf = HistGradientBoostingClassifier(**HGB).fit(x, y)
    elif model == "logistic":
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=C, max_iter=2000)).fit(x, y)
    else:
        raise ValueError(model)
    return {"features": features, "model": model, "clf": clf}


def fit(matches, letters_of=None, features="all", model="hgb", **_):
    """The model, trained on `matches` (any iterable; each read once, never the one it will predict)
    with the letters this stage will see on them, `letters_of(match)` (default: the mix's out of fold)."""
    if model == "none":
        return None
    letters_of = letters_of or (lambda m: lab_mix.kept_letters(m.id))
    tables = []
    for match in matches:
        tab = table(match, letters_of(match), features)
        tables.append(pd.concat([tab, targets(match, tab)], axis=1))
    return train(tables, features, model)


def probabilities(tab, model):
    if len(tab) == 0:
        return np.zeros(0)
    return model["clf"].predict_proba(design(tab, model["features"]))[:, 1]


def act(letters, tab, p, p_min):
    """(letters with the proposals written in, [rows written]): from the most likely down, each at or
    above p_min whose frames are all still the other team's and that keeps GAP frames from one written."""
    out = np.asarray(letters).astype("<U1").copy()
    used = np.zeros(len(out) + 1, bool)
    written = []
    for i in np.argsort(-p, kind="stable"):
        if p[i] < p_min:
            break
        k1, k2, new = int(tab.start.iat[i]), int(tab.stop.iat[i]), tab.team.iat[i]
        if used[max(k1 - GAP, 0):k2 + GAP].any() or np.any(out[k1:k2] != OTHER[new]):
            continue
        out[k1:k2] = new
        used[k1:k2] = True
        written.append(int(i))
    return out, written


def apply(match, letters, model, features="all", p_min=1.0, **_):
    """Letters for every frame of `match`: `letters` with the likely missed possessions written in."""
    letters = np.asarray(letters).astype("<U1")
    if model is None:
        return letters.copy()
    tab = table(match, letters, model["features"])
    return act(letters, tab, probabilities(tab, model), p_min)[0]


# --------------------------------------------------------------------------------------------
# The bench
# --------------------------------------------------------------------------------------------

def tables_of(which):
    """{match: (input letters, a scoring stub, {features: table with targets})}, one match loaded at a time."""
    out = {}
    for match in lab.each(which):
        letters = np.asarray(lab_mix.kept_letters(match.id, which)).astype("<U1")
        tabs = {}
        for f in FEATURES:
            tab = table(match, letters, f)
            tabs[f] = pd.concat([tab, targets(match, tab)], axis=1)
        stub = types.SimpleNamespace(id=match.id, truth=match.truth, source=match.source, unseen=match.unseen)
        out[match.id] = (letters, stub, tabs)
        t = tabs["all"]
        print("{}: {} proposals ({} right), no_pathcrf {} ({} right)".format(
            match.id, len(t), int(t.right.sum()), len(tabs["no_pathcrf"]), int(tabs["no_pathcrf"].right.sum())), flush=True)
    return out


def score_setting(stub, letters, tab, probs, setting):
    if setting["model"] == "none":
        out, written = letters.copy(), []
    else:
        out, written = act(letters, tab, probs[(setting["features"], setting["model"])], setting["p_min"])
    s = lab.score(stub, out)
    right = tab.right.iloc[written] if written else pd.Series([], dtype=bool)
    return {"spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "possessions": s["possessions"],
            "true_possessions": s["true_possessions"], "written": len(written), "written_right": int(right.sum()),
            "written_frames": int(np.sum(out != letters))}


def models_without(tables, out, features_models=None):
    """{(features, model): trained on every match in `tables` except those in `out`}."""
    keys = features_models or [(f, m) for f in FEATURES for m in MODELS]
    return {(f, m): train([tables[x][2][f] for x in tables if x not in out], f, m) for f, m in keys}


def nested(which, tables):
    """Robustness: lab.judge picks match m's setting from the other matches' scores, whose models
    were trained with m in. Here every score that picks m's setting comes from models trained
    without m too (for each other match j: trained without m and j)."""
    grid = pd.read_csv(os.path.join(lab.RESULTS, "lab_{}_grid{}.csv".format(NAME, chain.suffix(which))))
    base = lab_mix.kept_base(which)
    ms = list(tables)
    rows = []
    for j in ms:
        letters, stub, tabs = tables[j]
        for m in ms:
            if m == j:
                continue
            models = models_without(tables, (m, j))
            probs = {key: probabilities(tabs[key[0]], mod) for key, mod in models.items()}
            for order, setting in enumerate(GRID):
                tab = tabs[setting["features"]]
                rows.append({"without": m, "match": j, "order": order,
                             "spell_f1": score_setting(stub, letters, tab, probs, setting)["spell_f1"]})
        print("{}: scored without each other match".format(j), flush=True)
    inner = pd.DataFrame(rows)
    folds = []
    for m in ms:
        means = inner[inner.without == m].groupby("order").spell_f1.mean()
        setting = GRID[int(means.round(6).idxmax())]             # ties: the first in grid order
        row = grid[(grid.match == m) & np.all([grid[c].astype(str) == str(setting[c]) for c in COLUMNS], axis=0)].iloc[0]
        folds.append({"match": m, "source": base[m]["source"], "unseen": base[m]["unseen"], "picked": setting,
                      "held_out": float(row.spell_f1), "base": base[m]["spell_f1"],
                      "macro_held_out": float(row.macro_f1), "macro_base": base[m]["macro_f1"]})
    f = pd.DataFrame(folds)
    out = {"folds": folds}
    for name, part in [("all", f)] + [(s, f[f.source == s]) for s in sorted(f.source.unique())] + [("PathCRF-unseen", f[f.unseen])]:
        out[name] = {"held_out": float(part.held_out.mean()), "base": float(part.base.mean()),
                     "better": int((part.held_out > part.base).sum()), "matches": len(part)}
        print("nested {}: {:.4f} vs the mix {:.4f} ({:+.4f}), better on {}/{}".format(
            name, part.held_out.mean(), part.base.mean(), part.held_out.mean() - part.base.mean(),
            int((part.held_out > part.base).sum()), len(part)))
    print("  macroF1 {:.4f} vs {:.4f}".format(f.macro_held_out.mean(), f.macro_base.mean()))
    print(f[["match", "held_out", "base", "picked"]].to_string(index=False))
    with open(os.path.join(lab.RESULTS, "lab_{}_nested{}.json".format(NAME, chain.suffix(which))), "w") as handle:
        json.dump(out, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=chain.SETS, default="all", dest="which")
    parser.add_argument("--nested", action="store_true", help="robustness: settings picked with no model trained on the held-out match (after the main run)")
    args = parser.parse_args()
    which = args.which
    tables = tables_of(which)
    if args.nested:
        nested(which, tables)
        return
    rows, quality = [], []
    for m, (letters, stub, tabs) in tables.items():
        models = models_without(tables, (m,))
        probs = {key: probabilities(tabs[key[0]], mod) for key, mod in models.items()}
        for (f, mod), p in probs.items():
            y = tabs[f].right
            quality.append({"match": m, "features": f, "model": mod,
                            "auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan})
        for setting in GRID:
            tab = tabs[setting["features"]]
            rows.append({"match": m, **setting, **score_setting(stub, letters, tab, probs, setting),
                         "proposals": len(tab), "proposals_right": int(tab.right.sum())})
        print("{}: {} settings scored".format(m, len(GRID)), flush=True)
    grid = pd.DataFrame(rows)
    report = lab.judge(grid, COLUMNS, which, base=lab_mix.kept_base(which))
    print(lab.describe(report).replace("vs v11", "vs the mix"))

    picked = []
    for fold in report["folds"]:
        p = fold["picked"]
        picked.append(grid[(grid.match == fold["match"]) & np.all([grid[c] == p[c] for c in COLUMNS], axis=0)].iloc[0])
    picked = pd.DataFrame(picked)
    cols = ["match"] + COLUMNS + ["spell_f1", "macro_f1", "possessions", "true_possessions", "written", "written_right",
                                  "written_frames", "proposals", "proposals_right"]
    print(picked[cols].to_string(index=False))
    q = pd.DataFrame(quality)
    print("held-out AUC:\n", q.groupby(["features", "model"]).auc.mean().round(3))
    summary = grid.groupby(COLUMNS, sort=False).agg(spell_f1=("spell_f1", "mean"), written=("written", "mean"),
                                                    right=("written_right", "mean")).reset_index()
    base = lab_mix.kept_base(which)
    summary["gain"] = summary.spell_f1 - np.mean([b["spell_f1"] for b in base.values()])
    summary["better"] = [int(sum(grid[(grid.match == mm) & np.all([grid[c] == r[c] for c in COLUMNS], axis=0)].spell_f1.iloc[0]
                                 > base[mm]["spell_f1"] for mm in base)) for _, r in summary.iterrows()]
    print("the whole grid, in sample:\n", summary.round(4).to_string(index=False))
    report["possessions"] = float(picked.possessions.mean())
    report["true_possessions"] = float(picked.true_possessions.mean())
    report["held_out_actions"] = picked[cols].to_dict("records")
    report["held_out_auc"] = q.to_dict("records")
    lab.save(NAME, grid, report, which)


if __name__ == "__main__":
    main()
