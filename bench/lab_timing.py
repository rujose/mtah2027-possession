"""Lab candidate `timing`: each change of team moved to the frame a learned ranker picks (docs/lab/timing.md).

    python -m bench.lab_timing [--set all|idsse]     # ~1 min: leave one match out over the grid, judged against the mix
    python -m bench.lab_timing --nested [--set ..]   # ~4 min, after the run: settings picked with no model trained on the held-out match

Round 2, on round 1's mix (bench/lab_mix.py, `kept_letters`). v11 moves each H <-> A change to the
nearest touch by the new team within 1 s; what's left is the changes that sit on the wrong touch:
of the mix's changes that match a true one, 58 % are within 0.2 s, but a quarter are more than
0.8 s off, and those carry most of the team-change-timing budget (+0.048).

For each predicted change from team O to team N at frame g, the candidates are the frames from
MAX_WINDOW before to MAX_WINDOW after g, inside the two possessions it separates (O keeps at least
its first frame, N its last), where
- either team touches the ball (a loose touch: a player within LOOSE_REACH, the ball's velocity
  changing by at least LOOSE_DV),
- N's nearest player comes within touches.REACH of the ball,
- N's nearest player becomes nearer than O's,
plus g itself (staying). A gradient-boosted classifier, trained on the other bench matches'
candidates, gives each one P(a true O -> N change is within TOL of it), from its offset from g, the
ball (seen, gaps, speed and direction before and after, its change), each team's nearest player,
the touches by each team before, after and between, PathCRF and the decoder's costs before and after
(or v6's costs only: `no_pathcrf`), P(dead), and the two possessions' lengths. The change moves to
the most likely candidate when it isn't g, beats g, and is at least P_MIN, within the setting's
window. Moves follow v11's guard: a move never crosses another change and never swallows a
possession, so the teams and the number of possessions never change.

A post stage: `fit(matches, letters_of, **setting)` and `apply(match, letters, model, **setting)`
work on any input letters (H/A/D/X per frame) and read only the match's damaged tracking and the
out-of-sample outputs on it, never its truth (truth is read by `targets`, for training on the other
matches, and for scoring).
"""

import argparse
import json
import os
import types

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from bench import chain, lab, lab_mix
from bench.run import score as bench_score
from possession import touches, v11
from possession.data import FPS

# Params (by reasoning, fixed before any held-out run; docs/lab/timing.md gives the why of each)
STAGE = "post"
NAME = "timing"
REACH = touches.REACH        # 2 m: "on the ball" (v1, v11)
LOOSE_REACH = 3.0            # m: a candidate touch may be a little further (the damaged ball and players jitter)
LOOSE_DV = 1.0               # m/s: a candidate touch needs only a small change (the damaged ball's jitter is ~2 m/s;
                             # the ranker reads the size of the change as a feature)
LOOK = touches.LOOK          # 5 frames: the touch detector's look either side, for speed and direction
MAX_WINDOW = 3.0             # s: candidates looked for up to this far from the change (the widest window in the grid)
TOL = int(0.2 * FPS)         # 5 frames: a candidate is right when a true change of the same kind is this close
NEAR = 2 * FPS               # a predicted change and a true one this close (same kind) are "the same change" (lab_budget)
CONTEXT = 2 * FPS            # frames before / after a candidate where each team's touches are counted
SIDE_LEN = FPS               # frames before / after a candidate where PathCRF and the costs are averaged
SOON = int(0.5 * FPS)        # frames after (before) a candidate where N's (O's) nearest distance is read
GAP_CAP = 2 * FPS            # frames: cap on "frames since the ball was last seen"
FAR = 30.0                   # m: cap on a nearest distance (no player of the team seen near)
RUN_CAP = 30 * FPS           # frames: cap on a possession's length (as a feature)
CLASSIFIER = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 8, "min_samples_leaf": 40,
              "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}     # as lab_segments
RULE_DV = 3.0                # m/s: the "nearest" rule is v11's move again (v11's ΔV), with a wider window
OTHER = {"H": "A", "A": "H"}
SIDE = {"H": 0, "A": 1}

COMMON = ["dk", "adk", "is_g", "tl_n", "tl_o", "tv_n", "tv_o", "rs_n", "ns_n",
          "dv_k", "sp_b", "sp_a", "cos_k", "dn_k", "do_k", "margin_k", "dn_soon", "do_recent",
          "seen_k", "gap_b", "gap_a", "nt_before", "ot_before", "nt_after", "ot_after",
          "last_touch", "next_touch", "closer_n", "between_n", "between_o",
          "pdead_k", "run_o", "run_n", "g_on_touch", "g_dv"]
FEATURES = {
    # PathCRF's P(team) and the v7 decoder's costs (which include PathCRF's clue)
    "all": COMMON + ["pn_k", "pn_before", "pn_after", "pn_jump", "m7_before", "m7_after", "m7_jump"],
    # neither: PathCRF was pretrained on 6 of the 7 IDSSE matches, so it's too good there; v6's costs instead
    "no_pathcrf": COMMON + ["m6_before", "m6_after", "m6_jump"],
}

# The grid, simplest first: the no-op (the mix unchanged); v11's nearest-touch move again with a wider
# window; then the ranker, for each feature set, window and minimum probability.
WINDOWS = (1.0, 2.0, 3.0)
P_MINS = (0.5, 0.35, 0.65)
COLUMNS = ["rule", "features", "window", "p_min"]


def _grid():
    out = [{"rule": "none", "features": "-", "window": 0.0, "p_min": 0.0}]
    out += [{"rule": "nearest", "features": "-", "window": w, "p_min": 0.0} for w in (1.0, 2.0)]
    out += [{"rule": "learned", "features": f, "window": w, "p_min": p}
            for f in FEATURES for w in WINDOWS for p in P_MINS]
    return out


GRID = _grid()


# --------------------------------------------------------------------------------------------
# Per-frame signals (prediction-time inputs only; computed once per match)
# --------------------------------------------------------------------------------------------

def peaks(holds, value):
    """bool: the frame of each run of `holds` where `value` is largest (as touches.touches)."""
    out = np.zeros(len(holds), bool)
    edges = np.flatnonzero(np.diff(np.r_[0, holds.astype(np.int8), 0]))
    for first, end in zip(edges[::2], edges[1::2]):
        v = value[first:end]
        out[first + (int(np.nanargmax(v)) if np.any(~np.isnan(v)) else 0)] = True
    return out


def run_starts(holds):
    """bool: the first frame of each run of `holds`."""
    out = np.zeros(len(holds), bool)
    out[np.flatnonzero(np.diff(np.r_[0, holds.astype(np.int8)]) == 1)] = True
    return out


def since_seen(seen):
    """int: frames since the ball was last seen (0 where seen), capped at GAP_CAP."""
    idx = np.where(seen, np.arange(len(seen)), -GAP_CAP - 1)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(seen)) - last, GAP_CAP)


def cumulative(x):
    """(cumsum of x with NaN as 0, cumsum of the non-NaN count), each with a leading 0."""
    ok = ~np.isnan(x)
    return np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))], np.r_[0, np.cumsum(ok)]


def window_mean(cs, a, b):
    """Mean over frames [a, b) from cumulative() (NaN if none)."""
    s, c = cs
    a, b = max(a, 0), min(b, len(c) - 1)
    n = c[b] - c[a] if b > a else 0
    return (s[b] - s[a]) / n if n else np.nan


def signals(match):
    """Everything a candidate reads, independent of the letters."""
    cached = getattr(match, "_timing_signals", None)
    if cached is not None:
        return cached
    tr = match.track
    ball, players, team = tr["ball"], tr["players"], tr["team"]
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    change = touches.velocity_change(ball)
    near = np.stack([touches.nearest(players, team, ball, s) for s in (0, 1)], axis=1)
    near = np.where(np.isfinite(near), np.minimum(near, FAR), FAR)
    v_before, v_after = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    v_before[LOOK:] = (ball[LOOK:] - ball[:-LOOK]) / (LOOK / FPS)
    v_after[:-LOOK] = (ball[LOOK:] - ball[:-LOOK]) / (LOOK / FPS)
    sp_b, sp_a = np.hypot(*v_before.T), np.hypot(*v_after.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v_before * v_after, axis=1) / (sp_b * sp_a)
        loose = np.stack([peaks((near[:, s] < LOOSE_REACH) & (change >= LOOSE_DV), change) for s in (0, 1)], axis=1)
        strict = np.stack([peaks((near[:, s] < REACH) & (change >= RULE_DV), change) for s in (0, 1)], axis=1)
        reach = np.stack([run_starts(near[:, s] < REACH) for s in (0, 1)], axis=1)
        nearest = np.stack([run_starts(seen & (near[:, s] < near[:, 1 - s])) for s in (0, 1)], axis=1)
    p_home = match.pred["p_pathcrf"].astype(float)
    cost7, cost6 = match.pred["v7_cost"].astype(float), match.pred["v6_cost"].astype(float)
    out = {"seen": seen, "change": change, "near": near, "sp_b": sp_b, "sp_a": sp_a, "cos": cos,
           "loose": loose, "strict": strict, "reach": reach, "nearest": nearest,
           "loose_cs": [np.r_[0, np.cumsum(loose[:, s])] for s in (0, 1)],
           "gap_b": since_seen(seen), "gap_a": since_seen(seen[::-1])[::-1],
           "p": [cumulative(p_home), cumulative(1 - p_home)], "p_home": p_home,
           # cost margin for team s: cost of the other team minus s's (positive: s is favoured)
           "m7": [cumulative(cost7[:, 1 - s] - cost7[:, s]) for s in (0, 1)],
           "m6": [cumulative(cost6[:, 1 - s] - cost6[:, s]) for s in (0, 1)],
           "pdead": cumulative(match.pred["p_dead"].astype(float))}
    match._timing_signals = out
    return out


# --------------------------------------------------------------------------------------------
# The changes, their candidates and features
# --------------------------------------------------------------------------------------------

def changes(letters, halves):
    """[(g, old, new, lo, hi, first, last)]: each H <-> A change inside a half, with the frames a new
    change may sit on (lo..hi: O keeps at least its first frame, N its last)."""
    out = []
    for _, first, last in halves:
        for g in v11.team_changes(letters, first, last):
            old, new = letters[g - 1], letters[g]
            a = g - 1
            while a > first and letters[a - 1] == old:
                a -= 1
            b = g
            while b < last and letters[b + 1] == new:
                b += 1
            out.append((int(g), str(old), str(new), a + 1, b, a, b))
    return out


def count(cs, a, b):
    a, b = max(a, 0), min(b, len(cs) - 1)
    return int(cs[b] - cs[a]) if b > a else 0


def table(match, letters, max_window=MAX_WINDOW):
    """One row per candidate of each change in `letters`: where it is and its features."""
    sig = signals(match)
    letters = np.asarray(letters).astype("<U1")
    w = int(max_window * FPS)
    rows = []
    for ci, (g, old, new, lo, hi, run_a, run_b) in enumerate(changes(letters, match.halves)):
        n, o = SIDE[new], SIDE[old]
        a, b = max(lo, g - w), min(hi, g + w)
        span = np.arange(a, b + 1)
        hit = sig["loose"][a:b + 1].any(axis=1) | sig["reach"][a:b + 1, n] | sig["nearest"][a:b + 1, n]
        ks = set(span[hit].tolist()) | {g}
        loose_n = np.flatnonzero(sig["loose"][a:b + 1, n]) + a
        g_on = bool(sig["strict"][max(g - 1, 0):g + 2, n].any())
        g_dv = sig["change"][g]
        for k in sorted(ks):
            dn, do = sig["near"][k, n], sig["near"][k, o]
            before = [j for j in range(k - 1, max(k - CONTEXT, 0) - 1, -1) if sig["loose"][j].any()]
            after = [j for j in range(k + 1, min(k + CONTEXT, len(letters) - 1) + 1) if sig["loose"][j].any()]

            def team_of(j):
                return 1 if sig["loose"][j, n] and not sig["loose"][j, o] else (-1 if sig["loose"][j, o] and not sig["loose"][j, n] else 0)
            lo_k, hi_k = min(g, k), max(g, k)
            row = {"change": ci, "g": g, "k": k, "old": old, "new": new, "first": run_a, "last": run_b,
                   "dk": (k - g) / FPS, "adk": abs(k - g) / FPS, "is_g": float(k == g),
                   "tl_n": float(sig["loose"][k, n]), "tl_o": float(sig["loose"][k, o]),
                   "tv_n": float(sig["strict"][k, n]), "tv_o": float(sig["strict"][k, o]),
                   "rs_n": float(sig["reach"][k, n]), "ns_n": float(sig["nearest"][k, n]),
                   "dv_k": sig["change"][k], "sp_b": sig["sp_b"][k], "sp_a": sig["sp_a"][k], "cos_k": sig["cos"][k],
                   "dn_k": dn, "do_k": do, "margin_k": do - dn,
                   "dn_soon": float(sig["near"][k:k + SOON + 1, n].min()),
                   "do_recent": float(sig["near"][max(k - SOON, 0):k + 1, o].min()),
                   "seen_k": float(sig["seen"][k]), "gap_b": float(sig["gap_b"][k]), "gap_a": float(sig["gap_a"][k]),
                   "nt_before": count(sig["loose_cs"][n], k - CONTEXT, k), "ot_before": count(sig["loose_cs"][o], k - CONTEXT, k),
                   "nt_after": count(sig["loose_cs"][n], k + 1, k + CONTEXT + 1),
                   "ot_after": count(sig["loose_cs"][o], k + 1, k + CONTEXT + 1),
                   "last_touch": float(team_of(before[0])) if before else 0.0,
                   "next_touch": float(team_of(after[0])) if after else 0.0,
                   "closer_n": float(np.sum(np.abs(loose_n - g) < abs(k - g))),
                   "between_n": count(sig["loose_cs"][n], lo_k + 1, hi_k), "between_o": count(sig["loose_cs"][o], lo_k + 1, hi_k),
                   "pdead_k": window_mean(sig["pdead"], k - SIDE_LEN, k + SIDE_LEN),
                   "run_o": float(np.log(min(g - run_a, RUN_CAP))), "run_n": float(np.log(min(run_b - g + 1, RUN_CAP))),
                   "g_on_touch": float(g_on), "g_dv": g_dv}
            p_k = sig["p_home"][k]
            row["pn_k"] = p_k if n == 0 else 1 - p_k
            row["pn_before"] = window_mean(sig["p"][n], k - SIDE_LEN, k)
            row["pn_after"] = window_mean(sig["p"][n], k, k + SIDE_LEN)
            row["pn_jump"] = row["pn_after"] - row["pn_before"]
            for name in ("m7", "m6"):
                row[name + "_before"] = window_mean(sig[name][n], k - SIDE_LEN, k)
                row[name + "_after"] = window_mean(sig[name][n], k, k + SIDE_LEN)
                row[name + "_jump"] = row[name + "_after"] - row[name + "_before"]
            rows.append(row)
    return pd.DataFrame(rows)


def true_changes(truth_letters, halves):
    """{(old, new): sorted frames} of the truth's H <-> A changes."""
    out = {("H", "A"): [], ("A", "H"): []}
    for _, first, last in halves:
        for f in v11.team_changes(truth_letters, first, last):
            out[(str(truth_letters[f - 1]), str(truth_letters[f]))].append(int(f))
    return {key: np.array(sorted(v), dtype=int) for key, v in out.items()}


def targets(match, tab):
    """The truth's answer per candidate (training on other matches, and reporting only): a true change
    of the same kind within TOL of it."""
    found = true_changes(match.truth_letters.astype("<U1"), match.halves)
    right = np.zeros(len(tab), bool)
    for key, frames in found.items():
        sel = ((tab.old == key[0]) & (tab.new == key[1])).to_numpy()
        if not sel.any() or len(frames) == 0:
            continue
        k = tab.k.to_numpy()[sel]
        i = np.clip(np.searchsorted(frames, k), 1, len(frames) - 1) if len(frames) > 1 else np.zeros(len(k), int)
        d = np.minimum(np.abs(frames[i] - k), np.abs(frames[np.maximum(i - 1, 0)] - k))
        right[sel] = d <= TOL
    return pd.Series(right, index=tab.index, name="right")


# --------------------------------------------------------------------------------------------
# The interface: fit on some matches, apply to another
# --------------------------------------------------------------------------------------------

def train(tables, features):
    """The ranker on the rows of `tables` (each a table() with targets() joined as "right")."""
    rows = pd.concat(tables, ignore_index=True)
    x = rows[FEATURES[features]].to_numpy(float)
    return {"features": features, "clf": HistGradientBoostingClassifier(**CLASSIFIER).fit(x, rows["right"].to_numpy(int))}


def fit(matches, letters_of=None, rule="learned", features="all", **_):
    """The model, trained on `matches` (each read once) with their input letters `letters_of(match)`
    (default: the mix's kept letters). None for the rules that don't learn."""
    if rule != "learned":
        return None
    letters_of = letters_of or (lambda m: lab_mix.kept_letters(m.id))
    tables = []
    for match in matches:
        tab = table(match, letters_of(match))
        tables.append(tab.assign(right=targets(match, tab)))
    return train(tables, features)


def probabilities(tab, model):
    if len(tab) == 0:
        return np.zeros(0)
    return model["clf"].predict_proba(tab[FEATURES[model["features"]]].to_numpy(float))[:, 1]


def guarded_move(out, g, k, old, new, first):
    """v11's move of the change at g to k, only if it never crosses another change nor swallows a possession."""
    if k < g and k > first and np.all(out[k - 1:g] == old):
        out[k:g] = new                 # the old team keeps at least the frame before
        return True
    if k > g and np.all(out[g:k + 1] == new):
        out[g:k] = old                 # the touch is inside the new team's possession
        return True
    return False


def act(letters, tab, p, window, p_min):
    """(letters with the changes moved, [(g, k)]). Each change, in time order, moves to its most likely
    candidate within `window` s when that isn't g, beats g and reaches `p_min`."""
    out = np.asarray(letters).astype("<U1").copy()
    moves = []
    if len(tab) == 0:
        return out, moves
    inside = (tab.adk.to_numpy() * FPS <= window * FPS + 1e-9)
    change, p, adk = tab.change.to_numpy()[inside], np.asarray(p)[inside], tab.adk.to_numpy()[inside]
    is_g = tab.is_g.to_numpy()[inside] == 1
    order = np.lexsort((adk, -p, change))                   # per change: most likely first; ties: nearest g
    best = order[np.r_[True, change[order][1:] != change[order][:-1]]]
    p_g = dict(zip(change[is_g], p[is_g]))
    g_all, k_all = tab.g.to_numpy()[inside], tab.k.to_numpy()[inside]
    old_all, new_all, first_all = tab.old.to_numpy()[inside], tab.new.to_numpy()[inside], tab["first"].to_numpy()[inside]
    for i in best[np.argsort(g_all[best], kind="stable")]:     # in time order
        if is_g[i] or p[i] < p_min or p[i] <= p_g.get(change[i], 0.0):
            continue
        g, k = int(g_all[i]), int(k_all[i])
        if guarded_move(out, g, k, old_all[i], new_all[i], int(first_all[i])):
            moves.append((g, k))
    return out, moves


def apply(match, letters, model=None, rule="none", features="-", window=0.0, p_min=0.0, **_):
    """Letters for every frame of `match`: `letters` with its H <-> A changes moved."""
    letters = np.asarray(letters).astype("<U1")
    if rule == "none" or window <= 0:
        return letters.copy()
    if rule == "nearest":
        moved, _ = v11.move(letters, match.track, match.halves, RULE_DV, window, window)
        return moved
    tab = table(match, letters, max(window, 0.0))
    return act(letters, tab, probabilities(tab, model), window, p_min)[0]


# --------------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------------

def offsets(letters, truth_letters, halves):
    """Signed distance (s) from each predicted H <-> A change to the nearest true change of the same
    kind within NEAR (unmatched changes left out)."""
    found = true_changes(truth_letters, halves)
    out = []
    for g, old, new, *_ in changes(np.asarray(letters).astype("<U1"), halves):
        frames = found[(old, new)]
        near = frames[np.abs(frames - g) <= NEAR]
        if len(near):
            out.append((g - near[np.argmin(np.abs(near - g))]) / FPS)
    return np.array(out)


def timing_summary(values):
    a = np.abs(values)
    return {"matched": int(len(a)), "within_02": float(np.mean(a <= 0.2)), "within_004": float(np.mean(a <= 0.04)),
            "median": float(np.median(a)), "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
            "over_1s": float(np.mean(a > 1.0))}


def ceiling(match, letters, tab, right):
    """Exploration (truth read): spellF1 with every matched change moved onto its true change, and onto
    the right candidate nearest it (what a perfect ranker over these candidates reaches)."""
    truth = match.truth_letters.astype("<U1")
    found = true_changes(truth, match.halves)
    perfect, reach = letters.copy(), letters.copy()
    tab = tab.assign(right=right)
    reachable = matched = 0
    for (g, old, new, lo, hi, first, last) in changes(letters, match.halves):
        frames = found[(old, new)]
        near = frames[np.abs(frames - g) <= NEAR]
        if len(near) == 0:
            continue
        matched += 1
        f = int(near[np.argmin(np.abs(near - g))])
        guarded_move(perfect, g, f, old, new, first)
        rows = tab[(tab.g == g) & tab.right]
        if len(rows):
            reachable += 1
            k = int(rows.k.to_numpy()[np.argmin(np.abs(rows.k.to_numpy() - f))])
            guarded_move(reach, g, k, old, new, first)
    return {"perfect": bench_score(match.truth, perfect)["spell_f1"], "reachable": bench_score(match.truth, reach)["spell_f1"],
            "matched": matched, "with_right_candidate": reachable}


# --------------------------------------------------------------------------------------------
# The bench
# --------------------------------------------------------------------------------------------

def prepare(which):
    """{match: namespace with its table (truth's answers joined), letters, truth, rule outputs, ceiling},
    one match loaded at a time."""
    out = {}
    for match in lab.each(which):
        letters = np.asarray(lab_mix.kept_letters(match.id, which)).astype("<U1")
        tab = table(match, letters)
        right = targets(match, tab)
        rules = {s["window"]: apply(match, letters, None, **s) for s in GRID if s["rule"] == "nearest"}
        c = ceiling(match, letters, tab, right)
        out[match.id] = types.SimpleNamespace(
            id=match.id, source=match.source, unseen=match.unseen, halves=match.halves, truth=match.truth,
            truth_letters=match.truth_letters.astype("<U1"), letters=letters, tab=tab.assign(right=right),
            rules=rules, ceiling=c)
        print("{}: {} changes, {} candidates ({} right); {} of {} matched changes have a right candidate".format(
            match.id, tab.change.nunique(), len(tab), int(right.sum()), c["with_right_candidate"], c["matched"]), flush=True)
    return out


def scores_for(m, models, probs):
    """Every setting of GRID on match m: [(setting, letters, moves)]."""
    out = []
    for setting in GRID:
        if setting["rule"] == "none":
            out.append((setting, m.letters, []))
        elif setting["rule"] == "nearest":
            out.append((setting, m.rules[setting["window"]], None))
        else:
            letters, moves = act(m.letters, m.tab, probs[setting["features"]], setting["window"], setting["p_min"])
            out.append((setting, letters, moves))
    return out


def nested(which, prepared):
    """Robustness check of the same grid: lab.judge picks match m's setting from the other matches'
    scores, but their models were trained with m's truth in. Here every score that picks m's setting
    comes from models trained without m too (for each other match j: trained without m and j)."""
    grid = pd.read_csv(os.path.join(lab.RESULTS, "lab_{}_grid{}.csv".format(NAME, chain.suffix(which))))
    ms = list(prepared)
    base = lab_mix.kept_base(which)
    rows = []
    for j in ms:
        pj = prepared[j]
        for m in ms:
            if m == j:
                continue
            models = {f: train([prepared[x].tab for x in ms if x not in (m, j)], f) for f in FEATURES}
            probs = {f: probabilities(pj.tab, models[f]) for f in FEATURES}
            for order, (setting, letters, _) in enumerate(scores_for(pj, models, probs)):
                rows.append({"without": m, "match": j, "order": order, "spell_f1": bench_score(pj.truth, letters)["spell_f1"]})
        print("{}: scored without each other match".format(j), flush=True)
    inner = pd.DataFrame(rows)
    folds = []
    for m in ms:
        means = inner[inner.without == m].groupby("order").spell_f1.mean()
        order = int(means.round(6).idxmax())                 # ties: the first in grid order
        setting = GRID[order]
        sel = np.all([grid[c].astype(str) == str(setting[c]) if c in ("rule", "features")
                      else np.isclose(grid[c].astype(float), float(setting[c])) for c in COLUMNS], axis=0)
        row = grid[(grid.match == m) & sel].iloc[0]
        folds.append({"match": m, "source": base[m]["source"], "unseen": base[m]["unseen"], "picked": setting,
                      "held_out": float(row.spell_f1), "base": base[m]["spell_f1"]})
    f = pd.DataFrame(folds)
    summary = {}
    for name, part in [("all", f)] + [(s, f[f.source == s]) for s in sorted(f.source.unique())] + [("PathCRF-unseen", f[f.unseen])]:
        summary[name] = {"held_out": float(part.held_out.mean()), "base": float(part.base.mean()),
                         "better": int((part.held_out > part.base).sum()), "matches": len(part)}
        print("nested {}: {:.4f} vs the mix {:.4f} ({:+.4f}), better on {}/{}".format(
            name, part.held_out.mean(), part.base.mean(), part.held_out.mean() - part.base.mean(),
            int((part.held_out > part.base).sum()), len(part)))
    print(f[["match", "held_out", "base", "picked"]].to_string(index=False))
    with open(os.path.join(lab.RESULTS, "lab_{}_nested{}.json".format(NAME, chain.suffix(which))), "w") as handle:
        json.dump({"folds": folds, "summary": summary}, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--nested", action="store_true", help="robustness: settings picked with no model trained on the held-out match (after the main run)")
    args = parser.parse_args()
    which = args.which
    prepared = prepare(which)
    if args.nested:
        nested(which, prepared)
        return
    base = lab_mix.kept_base(which)
    rows, quality, outputs = [], [], {}
    for m in prepared.values():
        models = {f: train([p.tab for p in prepared.values() if p.id != m.id], f) for f in FEATURES}
        probs = {f: probabilities(m.tab, models[f]) for f in FEATURES}
        for f, p in probs.items():
            tab = m.tab.assign(p=p)
            top = tab.loc[tab.groupby("change").p.idxmax()]
            has = tab.groupby("change").right.any()
            quality.append({"match": m.id, "features": f, "auc": roc_auc_score(tab.right, p) if tab.right.nunique() > 1 else np.nan,
                            "top_right": float(top.right.mean()), "top_right_when_reachable": float(top.set_index("change").right[has].mean())})
        for order, (setting, letters, moves) in enumerate(scores_for(m, models, probs)):
            s = bench_score(m.truth, letters)
            outputs[(m.id, order)] = letters
            rows.append({"match": m.id, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                         "possessions": s["possessions"], "moves": len(moves) if moves is not None else -1})
        print("{}: {} settings scored".format(m.id, len(GRID)), flush=True)
    grid = pd.DataFrame(rows)
    report = lab.judge(grid, COLUMNS, which, base=base)
    print(lab.describe(report).replace("vs v11", "vs the mix"))

    # The held-out folds' timing, each with the setting picked without it
    before, after, per_match = [], [], []
    for fold in report["folds"]:
        m = prepared[fold["match"]]
        order = next(i for i, s in enumerate(GRID) if all(str(s[c]) == str(fold["picked"][c]) for c in COLUMNS))
        b, a = offsets(m.letters, m.truth_letters, m.halves), offsets(outputs[(m.id, order)], m.truth_letters, m.halves)
        before.append(b)
        after.append(a)
        row = grid[(grid.match == m.id)].iloc[order]
        per_match.append({"match": m.id, "picked": GRID[order], "held_out": fold["held_out"], "base": fold["base"],
                          "moves": int(row.moves), "within_02_before": float(np.mean(np.abs(b) <= 0.2)),
                          "within_02_after": float(np.mean(np.abs(a) <= 0.2))})
    timing = {"before": timing_summary(np.concatenate(before)), "after": timing_summary(np.concatenate(after))}
    print("timing of the changes matching a true one (held-out picks): within 0.2 s {:.1%} -> {:.1%}; median {:.2f} -> {:.2f} s; "
          "quartiles {:.2f}/{:.2f} -> {:.2f}/{:.2f} s; over 1 s {:.1%} -> {:.1%}".format(
              timing["before"]["within_02"], timing["after"]["within_02"], timing["before"]["median"], timing["after"]["median"],
              timing["before"]["q25"], timing["before"]["q75"], timing["after"]["q25"], timing["after"]["q75"],
              timing["before"]["over_1s"], timing["after"]["over_1s"]))
    print(pd.DataFrame(per_match).drop(columns="picked").round(4).to_string(index=False))
    q = pd.DataFrame(quality)
    print("held-out ranker quality by feature set:\n", q.groupby("features")[["auc", "top_right", "top_right_when_reachable"]].mean().round(3))
    c = pd.DataFrame([{"match": m.id, "base": base[m.id]["spell_f1"], **m.ceiling} for m in prepared.values()])
    print("ceilings (truth read; exploration): every matched change on its true one {:+.4f}; on the right candidate "
          "nearest it {:+.4f}; matched changes with a right candidate {:.0%}".format(
              (c.perfect - c.base).mean(), (c.reachable - c.base).mean(), c.with_right_candidate.sum() / c.matched.sum()))
    means = grid.groupby(COLUMNS, sort=False)[["spell_f1", "macro_f1", "moves"]].mean().reset_index()
    print("means over the matches, every setting (not held out):\n" + means.round(4).to_string(index=False))
    report["timing"] = timing
    report["held_out_per_match"] = per_match
    report["ranker_quality"] = q.to_dict("records")
    report["ceilings"] = c.to_dict("records")
    lab.save(NAME, grid, report, which)


if __name__ == "__main__":
    main()
