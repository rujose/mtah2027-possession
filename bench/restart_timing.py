"""v9's settings on the bench: where to look for the restart pass (docs/models/v9-restart-pass.md).

    python -m bench.restart_timing     # ~6 min; after `python -m bench.eval_pathcrf`;
                                       # writes bench/results/restart_timing_*

1. The truth's convention: at every true dead -> team change, is the ball leaving or arriving?
2. The input: v7's out-of-sample predictions on the 10 bench matches at 41 % damage (v8 equals v7
   there: the keeper repair never fires on the bench).
3. Every (back, ahead) window, every match: spellF1 and macroF1 with the snap.
4. Leave one match out: each match scored with the window best on the other 9.
5. Adopted only if the held-out mean beats v7's and at least MIN_BETTER of the 10 matches improve
   (fixed on the page before the run).
6. For the chosen window: the snapped starts' distance to the true ones, and a sensitivity check
   with the pass detector reading tracking damaged at 30 % and 55 % (v7's predictions stay those
   made at 41 %).
"""

import itertools
import json
import os

import numpy as np
import pandas as pd

from bench import chain, degrade, sets
from bench.run import score
from bench.train_restart import prepare_level
from possession import v9
from possession.data import FPS, ROOT

# Params
BACKS = (1, 2, 3, 5)            # s
AHEADS = (1, 2, 3)              # s
LEVEL = 41
OTHER_LEVELS = (30, 55)
MIN_BETTER = 0.7                # share of the matches, held out (7 of 10)
NEAR = 10 * FPS                 # a predicted change and a true one this close are the same restart
RESULTS = os.path.join(ROOT, "bench", "results")
LETTERS = np.array(["H", "A", "D", "X"])


def changes(letters):
    """Frames where the labels go from D to a team."""
    return np.flatnonzero((letters[:-1] == "D") & np.isin(letters[1:], ["H", "A"])) + 1


def truth_convention(data, matches):
    """At true starts, with the clean ball (IDSSE only: Metrica's ball is blanked while dead)."""
    counts = {"starts": 0, "leaving": 0, "arriving": 0, "fast_before": 0}
    for m in matches:
        if sets.source(m) == "metrica":
            continue
        b = v9.ball_at_starts(LETTERS[sets.truth(m, data[m])], data[m]["ball"])
        counts["starts"] += b["seen"]
        for k in ("leaving", "arriving", "fast_before"):
            counts[k] += b[k] * b["seen"]
    return {k: (v / counts["starts"] if k != "starts" else v) for k, v in counts.items()}


def timing(moves, truth):
    """For each (change, pass) the snap made that matches a true start of the same team within
    NEAR: distance (s) to it before and after the move."""
    true_starts, before, after = changes(truth), [], []
    for g, k in moves:
        near = true_starts[np.abs(true_starts - g) <= NEAR]
        if len(near) == 0:
            continue
        f = near[np.argmin(np.abs(near - g))]
        before.append(abs(g - f) / FPS)
        after.append(abs(k - f) / FPS)
    return before, after


def main():
    which = chain.which_set("v9's settings on the bench")
    tag = chain.suffix(which)
    stats = degrade.load_stats()
    matches = chain.matches(which)
    min_better = int(np.ceil(MIN_BETTER * len(matches) - 1e-9))
    data = {m: sets.load(m) for m in matches}
    convention = truth_convention(data, matches)
    print("truth (IDSSE, clean ball), at {} starts: leaving {:.0%}, arriving {:.0%}, above 8 m/s before {:.0%}".format(
        convention["starts"], convention["leaving"], convention["arriving"], convention["fast_before"]))

    prepared, scale = prepare_level(LEVEL, matches, data, stats)
    rows, inputs = [], {}
    for m in matches:
        path = os.path.join(chain.predictions(which), "{}.npz".format(m))
        if not os.path.exists(path):
            raise FileNotFoundError("{}: run `python -m bench.eval_pathcrf --set {}` first".format(path, which))
        with np.load(path) as z:
            pred = z["v7"].astype(str)
        codes = sets.truth(m, data[m])
        track = degrade.tracking(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        starts = v9.seen_passes(track)
        inputs[m] = (pred, codes, data[m]["halves"])
        base = score(codes, pred)
        rows.append({"match": m, "back": 0, "ahead": 0, "spell_f1": base["spell_f1"], "macro_f1": base["macro_f1"], "moved": 0})
        for back, ahead in itertools.product(BACKS, AHEADS):
            snapped, moves = v9.snap(pred, track, data[m]["halves"], back, ahead, starts)
            s = score(codes, snapped)
            rows.append({"match": m, "back": back, "ahead": ahead, "spell_f1": s["spell_f1"],
                         "macro_f1": s["macro_f1"], "moved": len(moves) / max(len(changes(pred)), 1)})
        print("{}: grid scored".format(m), flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(RESULTS, "restart_timing_grid{}.csv".format(tag)), index=False, float_format="%.5f")

    snaps = grid[grid.back > 0]
    v7 = grid[grid.back == 0].set_index("match")
    folds, held = [], {}
    for m in matches:
        others = snaps[snaps.match != m].groupby(["back", "ahead"]).spell_f1.mean()
        best = max(others.index, key=lambda s: (round(others[s], 6), -(s[0] + s[1])))   # tie: the smaller window
        row = snaps[(snaps.match == m) & (snaps.back == best[0]) & (snaps.ahead == best[1])].iloc[0]
        held[m] = row
        folds.append({"match": m, "picked": [int(best[0]), int(best[1])], "held_out": row.spell_f1,
                      "v7": v7.loc[m, "spell_f1"], "macro_held_out": row.macro_f1, "macro_v7": v7.loc[m, "macro_f1"]})
        print("fold {}: picked back {} s / ahead {} s -> {:.4f} (v7 {:.4f})".format(m, best[0], best[1], row.spell_f1, v7.loc[m, "spell_f1"]))
    held_mean = np.mean([f["held_out"] for f in folds])
    v7_mean = np.mean([f["v7"] for f in folds])
    better = sum(f["held_out"] > f["v7"] for f in folds)
    adopted = bool(held_mean > v7_mean and better >= min_better)
    overall = snaps.groupby(["back", "ahead"]).spell_f1.mean()
    chosen = max(overall.index, key=lambda s: (round(overall[s], 6), -(s[0] + s[1])))
    print("held out: {:.4f} against v7 {:.4f} ({:+.4f}), better on {}/{}; macroF1 {:.4f} against {:.4f}".format(
        held_mean, v7_mean, held_mean - v7_mean, better, len(matches),
        np.mean([f["macro_held_out"] for f in folds]), np.mean([f["macro_v7"] for f in folds])))
    print("chosen on all {}: back {} s / ahead {} s; adopted: {}".format(len(matches), chosen[0], chosen[1], adopted))

    before, after = [], []
    for m in matches:
        pred, codes, halves = inputs[m]
        track = degrade.tracking(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        _, moves = v9.snap(pred, track, halves, *chosen)
        b, a = timing(moves, LETTERS[codes])
        before += b
        after += a
    before, after = np.array(before), np.array(after)
    print("timing of the {} moved starts that match a true one: within 0.2 s {:.0%} -> {:.0%}; median {:.2f} s -> {:.2f} s".format(
        len(before), np.mean(before <= 0.2), np.mean(after <= 0.2), np.median(before), np.median(after)))

    sensitivity = {}
    for level in OTHER_LEVELS:
        prepared_l, scale_l = prepare_level(level, matches, data, stats)
        scores = []
        for m in matches:
            pred, codes, halves = inputs[m]
            track = degrade.tracking(m, data[m], level, prepared_l[m], scale_l[sets.source(m)], stats)
            s, _ = v9.snap(pred, track, halves, *chosen)
            scores.append((score(codes, s)["spell_f1"], v7.loc[m, "spell_f1"]))
        scores = np.array(scores)
        sensitivity[str(level)] = {"snap": scores[:, 0].mean(), "v7": scores[:, 1].mean(),
                                   "better": int((scores[:, 0] > scores[:, 1]).sum())}
        print("pass detector on tracking damaged at {} %: {:.4f} against v7 {:.4f}, better on {}/{}".format(
            level, *scores.mean(axis=0), sensitivity[str(level)]["better"], len(matches)))

    with open(os.path.join(RESULTS, "restart_timing_choice{}.json".format(tag)), "w") as handle:
        json.dump({"truth_convention": convention, "folds": folds, "held_out": held_mean, "v7": v7_mean,
                   "better": better, "min_better": min_better, "adopted": adopted,
                   "chosen": {"back": int(chosen[0]), "ahead": int(chosen[1])},
                   "timing": {"moved": len(before), "within_02_before": float(np.mean(before <= 0.2)),
                              "within_02_after": float(np.mean(after <= 0.2)),
                              "median_before": float(np.median(before)), "median_after": float(np.median(after))},
                   "sensitivity": sensitivity}, handle, indent=1, default=float)
    print("wrote bench/results/restart_timing_grid{0}.csv and restart_timing_choice{0}.json".format(tag))


if __name__ == "__main__":
    main()
