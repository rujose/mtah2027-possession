"""v11's settings on the bench: move each change of team onto the new team's touch
(docs/models/v11-touch-line.md).

    python -m bench.touch_timing [--set all|idsse]   # ~5 min; after `python -m bench.restart_timing`;
                                                      # writes bench/results/touch_timing_*

1. The truth's convention: at every true change of team, is a player of the new team within reach of
   the ball, and was the ball coming toward them?
2. The touch detector on the damaged tracking: how many true changes have a touch by the new team
   close to them, and how many touches a minute it sees, for each ΔV.
3. The input: v9's labels on the bench (v7's out-of-sample predictions with the restart snap).
4. Every (ΔV, window), every match: spellF1 and macroF1 with the move.
5. Leave one match out: each match scored with the setting best on the others.
6. Adopted only if the held-out mean beats v9's and at least MIN_BETTER of the matches improve (fixed
   on the page before the run).
7. For the chosen setting: the moved changes' distance to the true ones; the move without v9's snap
   (v10's RESTART_SNAP = False); and the detector reading tracking damaged at 30 % and 55 %.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd

from bench import chain, degrade, sets
from bench.run import score
from bench.train_restart import prepare_level
from possession import touches, v9, v11
from possession.data import AWAY, FPS, HOME, ROOT

# Params
DVS = (3.0, 5.0)                          # m/s
WINDOWS = ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0))   # (s before, s after) the change
LEVEL = 41
OTHER_LEVELS = (30, 55)
MIN_BETTER = 0.7                          # share of the matches, held out (7 of 10, 5 of 7)
NEAR = 2 * FPS                            # a predicted change and a true one this close (same new team) are the same
ARRIVING = int(0.4 * FPS)                 # frames before a true change to see the ball coming
RESULTS = os.path.join(ROOT, "bench", "results")
LETTERS = np.array(["H", "A", "D", "X"])
SIDE = {"H": HOME, "A": AWAY}


def all_changes(letters):
    """Frames where the labels go from one team straight to the other, over the whole match."""
    return v11.team_changes(letters, 0, len(letters) - 1)


def truth_convention(data, matches):
    """At true changes of team, with the clean ball (IDSSE only: Metrica's ball is blanked while dead)."""
    counts = {"changes": 0, "near_new": 0, "arriving": 0}
    for m in matches:
        if sets.source(m) == "metrica":
            continue
        truth = LETTERS[sets.truth(m, data[m])]
        players, team, ball = data[m]["players"], data[m]["team"], data[m]["ball"]
        for f in all_changes(truth):
            if f < ARRIVING:
                continue
            d = touches.nearest(players[f - ARRIVING:f + 1], team, ball[f - ARRIVING:f + 1], SIDE[truth[f]])
            counts["changes"] += 1
            counts["near_new"] += d[-1] < touches.REACH
            counts["arriving"] += d[-1] < d[0] - 1
    return {k: (v / counts["changes"] if k != "changes" else v) for k, v in counts.items()}


def detector_recall(truth, found, graded):
    """(true changes, with a new-team touch within 0.2 s, within 0.5 s, touches seen, graded minutes)."""
    near_02 = near_05 = 0
    changes = all_changes(truth)
    for f in changes:
        t = found[SIDE[truth[f]]]
        near_02 += t[max(f - int(0.2 * FPS), 0):f + int(0.2 * FPS) + 1].any()
        near_05 += t[max(f - int(0.5 * FPS), 0):f + int(0.5 * FPS) + 1].any()
    return np.array([len(changes), near_02, near_05, sum(int(t[graded].sum()) for t in found.values()),
                     graded.sum() / FPS / 60])


def timing(moves, before_labels, truth):
    """For each (change, touch) the move made that matches a true change into the same team within
    NEAR: distance (s) to it before and after the move."""
    true_changes = all_changes(truth)
    before, after = [], []
    for g, k in moves:
        near = true_changes[(np.abs(true_changes - g) <= NEAR) & (truth[true_changes] == before_labels[g])]
        if len(near) == 0:
            continue
        f = near[np.argmin(np.abs(near - g))]
        before.append(abs(g - f) / FPS)
        after.append(abs(k - f) / FPS)
    return before, after


def tie_break(setting):
    """Tie-break: the smaller window, then the larger ΔV (fewer touches)."""
    dv, back, ahead = setting
    return -(back + ahead), dv


def main():
    which = chain.which_set("v11's settings on the bench")
    tag = chain.suffix(which)
    stats = degrade.load_stats()
    matches = chain.matches(which)
    min_better = int(np.ceil(MIN_BETTER * len(matches) - 1e-9))
    back9, ahead9 = chain.restart_snap(which)
    data = {m: sets.load(m) for m in matches}
    convention = truth_convention(data, matches)
    print("truth (IDSSE, clean ball), at {} team changes: a new-team player within {} m {:.0%}, ball coming toward them {:.0%}".format(
        convention["changes"], touches.REACH, convention["near_new"], convention["arriving"]))

    prepared, scale = prepare_level(LEVEL, matches, data, stats)
    rows, inputs, recall = [], {}, {dv: np.zeros(5) for dv in DVS}
    for m in matches:
        path = os.path.join(chain.predictions(which), "{}.npz".format(m))
        if not os.path.exists(path):
            raise FileNotFoundError("{}: run `python -m bench.eval_pathcrf --set {}` first".format(path, which))
        with np.load(path) as z:
            v7 = z["v7"].astype(str)
        codes = sets.truth(m, data[m])
        truth = LETTERS[codes]
        halves = data[m]["halves"]
        track = degrade.tracking(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        base, _ = v9.snap(v7, track, halves, back9, ahead9)
        found = {dv: v11.seen_touches(track, dv) for dv in DVS}
        inputs[m] = (v7, base, codes, halves)
        s = score(codes, base)
        rows.append({"match": m, "dv": 0, "back": 0, "ahead": 0, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "moved": 0})
        for dv in DVS:
            recall[dv] += detector_recall(truth, found[dv], np.isin(truth, ["H", "A", "D"]))
        for dv, (back, ahead) in itertools.product(DVS, WINDOWS):
            moved, moves = v11.move(base, track, halves, dv, back, ahead, found[dv])
            s = score(codes, moved)
            rows.append({"match": m, "dv": dv, "back": back, "ahead": ahead, "spell_f1": s["spell_f1"],
                         "macro_f1": s["macro_f1"], "moved": len(moves) / max(len(all_changes(base)), 1)})
        print("{}: grid scored".format(m), flush=True)
    for dv, (n, near_02, near_05, seen, minutes) in recall.items():
        print("touch detector, ΔV {:g} m/s: a new-team touch within 0.2 s of {:.0%} of true changes, within 0.5 s {:.0%}; "
              "{:.1f} touches a minute".format(dv, near_02 / n, near_05 / n, seen / minutes))
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(RESULTS, "touch_timing_grid{}.csv".format(tag)), index=False, float_format="%.5f")

    moves_grid = grid[grid.dv > 0]
    v9_rows = grid[grid.dv == 0].set_index("match")
    columns = ["dv", "back", "ahead"]
    folds = []
    for m in matches:
        others = moves_grid[moves_grid.match != m].groupby(columns).spell_f1.mean()
        best = max(others.index, key=lambda s: (round(others[s], 6), tie_break(s)))
        row = moves_grid[(moves_grid.match == m) & (moves_grid.dv == best[0]) & (moves_grid.back == best[1])
                         & (moves_grid.ahead == best[2])].iloc[0]
        folds.append({"match": m, "picked": [float(b) for b in best], "held_out": row.spell_f1,
                      "v9": v9_rows.loc[m, "spell_f1"], "macro_held_out": row.macro_f1, "macro_v9": v9_rows.loc[m, "macro_f1"]})
        print("fold {}: picked ΔV {:g} m/s, {:g} s back / {:g} s ahead -> {:.4f} (v9 {:.4f})".format(
            m, *best, row.spell_f1, v9_rows.loc[m, "spell_f1"]))
    held_mean = np.mean([f["held_out"] for f in folds])
    v9_mean = np.mean([f["v9"] for f in folds])
    better = sum(f["held_out"] > f["v9"] for f in folds)
    adopted = bool(held_mean > v9_mean and better >= min_better)
    overall = moves_grid.groupby(columns).spell_f1.mean()
    chosen = max(overall.index, key=lambda s: (round(overall[s], 6), tie_break(s)))
    print("held out: {:.4f} against v9 {:.4f} ({:+.4f}), better on {}/{}; macroF1 {:.4f} against {:.4f}".format(
        held_mean, v9_mean, held_mean - v9_mean, better, len(matches),
        np.mean([f["macro_held_out"] for f in folds]), np.mean([f["macro_v9"] for f in folds])))
    print("chosen on all {}: ΔV {:g} m/s, {:g} s back / {:g} s ahead; adopted: {}".format(len(matches), *chosen, adopted))

    before, after, shifts, no_snap = [], [], [], []
    for m in matches:
        v7, base, codes, halves = inputs[m]
        track = degrade.tracking(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        found = v11.seen_touches(track, chosen[0])
        _, moves = v11.move(base, track, halves, *chosen, found)
        b, a = timing(moves, base, LETTERS[codes])
        before += b
        after += a
        shifts += [(k - g) / FPS for g, k in moves]
        plain, _ = v11.move(v7, track, halves, *chosen, found)
        no_snap.append((score(codes, plain)["spell_f1"], score(codes, v7)["spell_f1"]))
    before, after, shifts, no_snap = np.array(before), np.array(after), np.array(shifts), np.array(no_snap)
    print("moved changes: {}, earlier {}, later {}, shift quartiles {:+.2f} / {:+.2f} s".format(
        len(shifts), np.sum(shifts < 0), np.sum(shifts > 0), *np.percentile(shifts, [25, 75])))
    print("timing of the {} moved changes that match a true one: within 0.2 s {:.0%} -> {:.0%}; median {:.2f} s -> {:.2f} s".format(
        len(before), np.mean(before <= 0.2), np.mean(after <= 0.2), np.median(before), np.median(after)))
    print("without v9's snap (v10 with RESTART_SNAP = False): {:.4f} against v7 {:.4f}, better on {}/{}".format(
        *no_snap.mean(axis=0), int((no_snap[:, 0] > no_snap[:, 1]).sum()), len(matches)))

    sensitivity = {}
    for level in OTHER_LEVELS:
        prepared_l, scale_l = prepare_level(level, matches, data, stats)
        scores = []
        for m in matches:
            _, base, codes, halves = inputs[m]
            track = degrade.tracking(m, data[m], level, prepared_l[m], scale_l[sets.source(m)], stats)
            moved, _ = v11.move(base, track, halves, *chosen)
            scores.append((score(codes, moved)["spell_f1"], v9_rows.loc[m, "spell_f1"]))
        scores = np.array(scores)
        sensitivity[str(level)] = {"move": scores[:, 0].mean(), "v9": scores[:, 1].mean(),
                                   "better": int((scores[:, 0] > scores[:, 1]).sum())}
        print("touch detector on tracking damaged at {} %: {:.4f} against v9 {:.4f}, better on {}/{}".format(
            level, *scores.mean(axis=0), sensitivity[str(level)]["better"], len(matches)))

    with open(os.path.join(RESULTS, "touch_timing_choice{}.json".format(tag)), "w") as handle:
        json.dump({"truth_convention": convention,
                   "detector": {str(dv): {"within_02": r[1] / r[0], "within_05": r[2] / r[0], "per_minute": r[3] / r[4]}
                                for dv, r in recall.items()},
                   "folds": folds, "held_out": held_mean, "v9": v9_mean, "better": better, "min_better": min_better,
                   "adopted": adopted, "chosen": {"dv": float(chosen[0]), "back": float(chosen[1]), "ahead": float(chosen[2])},
                   "moved": len(shifts), "earlier": int(np.sum(shifts < 0)), "later": int(np.sum(shifts > 0)),
                   "timing": {"matched": len(before), "within_02_before": float(np.mean(before <= 0.2)),
                              "within_02_after": float(np.mean(after <= 0.2)),
                              "median_before": float(np.median(before)), "median_after": float(np.median(after))},
                   "without_snap": {"move": float(no_snap[:, 0].mean()), "v7": float(no_snap[:, 1].mean()),
                                    "better": int((no_snap[:, 0] > no_snap[:, 1]).sum())},
                   "sensitivity": sensitivity}, handle, indent=1, default=float)
    print("wrote bench/results/touch_timing_grid{0}.csv and touch_timing_choice{0}.json".format(tag))


if __name__ == "__main__":
    main()
