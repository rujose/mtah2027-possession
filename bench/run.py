"""Run models on the 7 IDSSE matches at each damage level and score them with the organisers'
own evaluate() (docs/bench.md, "Run and score").

    python -m bench.run                                  # every model, every level, both label sets
    python -m bench.run --models v1 --levels 41          # a subset

Writes bench/results/scores.csv (one row per model, level, label set and match: scores only, no
frames) and prints the mean and spread across matches.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from bench import degrade, labels, sets
from possession import v1, v3
from possession.data import ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from starter.baseline import AWAY as B_AWAY, HOME as B_HOME, NOBODY, smooth  # noqa: E402

# Params
RESULTS = os.path.join(ROOT, "bench", "results", "scores.csv")
CODE = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}


# --------------------------------------------------------------------------------------------
# Models, all taking (tracking, halves, pitch) and returning letters for every frame
# --------------------------------------------------------------------------------------------

def nearest_team(tracking):
    """The organisers' step 1 on arrays: team of the nearest player wherever the ball is seen."""
    players, ball, team = tracking["players"], tracking["ball"], tracking["team"]
    distance = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))
    seen = ~np.isnan(ball[:, 0]) & ~np.isnan(distance).all(axis=1)
    out = np.full(len(ball), NOBODY, dtype=np.int8)
    nearest = np.nanargmin(np.where(np.isnan(distance[seen]), np.inf, distance[seen]), axis=1)
    out[seen] = np.where(team[nearest] == v1.HOME, B_HOME, B_AWAY)
    return out


def baseline(window):
    """The organisers' baseline (starter/baseline.py): nearest player, then a majority vote over
    +/- window frames, smoothed within each half so no vote reaches across half-time."""
    def predict(tracking, halves, pitch):
        codes = nearest_team(tracking)
        out = np.full(len(codes), "X", dtype="<U1")
        for _, first, last in halves:
            out[first:last + 1] = smooth(codes[first:last + 1], window)
        return out
    return predict


def v1_without_tags(tracking, halves, pitch):
    """v1 exactly, minus the clues the bench cannot feed: no Smart Tagging exists for these matches."""
    return v1.predict(tracking, [], halves, pitch)[0]


def v3_without_tags(tracking, halves, pitch):
    """v3 exactly, minus Smart Tagging and the whistle (no audio either)."""
    return v3.predict(tracking, [], [], halves, pitch)[0]


MODELS = {
    "baseline_raw": baseline(0),
    "baseline_smooth25": baseline(25),
    "baseline_smooth50": baseline(50),
    "baseline_smooth125": baseline(125),
    "v1": v1_without_tags,
    "v3": v3_without_tags,
}


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------

def score(gt, letters):
    pred = np.array([CODE[x] for x in letters.tolist()], dtype=np.int8)
    result = pm.evaluate(gt, pred)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    return {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"],
            "d_f1": pm.per_class_prf(cm3, pm.D)[2],
            "team_acc_alive": result["team_acc_both_alive"],
            "possessions": result["n_pred_spells"], "true_possessions": result["n_gt_spells"],
            "mean_possession_s": result["mean_spell_duration_s"],
            "true_mean_possession_s": result["gt_mean_spell_duration_s"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--levels", nargs="+", type=int, default=list(degrade.LEVELS))
    parser.add_argument("--labels", choices=("raw", "merged", "both"), default="both")
    parser.add_argument("--set", choices=sets.SETS, default="idsse", dest="which")
    args = parser.parse_args()
    variants = ("raw", "merged") if args.labels == "both" else (args.labels,)

    stats = degrade.load_stats()
    matches = sets.matches(args.which)
    data = {m: sets.load(m) for m in matches}
    truth = {m: {"raw": sets.truth(m, data[m])} for m in matches}
    for m in matches:
        truth[m]["merged"] = labels.merge_short(truth[m]["raw"])

    rows = []
    for level in args.levels:
        started = time.time()
        prepared = {m: degrade.prepare(m, data[m], level, stats) for m in matches}
        # each source calibrated on its own, so adding one never moves the other's damage
        scale = {}
        for name in sorted({sets.source(m) for m in matches}):
            own = [prepared[m] for m in matches if sets.source(m) == name]
            scale[name] = degrade.calibrate(own, level) if level else None
        print("level {}%: losing-chance scale {}  ({:.0f} s to prepare)".format(
            level, {k: "-" if v is None else round(v, 3) for k, v in scale.items()},
            time.time() - started), flush=True)
        for m in matches:
            track = degrade.tracking(m, data[m], level, prepared[m], scale[sets.source(m)], stats)
            halves, pitch = data[m]["halves"], data[m]["pitch"]
            in_halves = np.concatenate([np.arange(a, b + 1) for _, a, b in halves])
            missing = float(np.isnan(track["ball"][in_halves, 0]).mean())
            for name in args.models:
                letters = MODELS[name](track, halves, pitch)
                for variant in variants:
                    rows.append({"model": name, "level": level, "labels": variant, "match": m,
                                 "ball_missing": missing, **score(truth[m][variant], letters)})

    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    if os.path.exists(RESULTS):
        old = pd.read_csv(RESULTS)
        key = ["model", "level", "labels", "match"]
        old = old[~old.set_index(key).index.isin(frame.set_index(key).index)]
        frame = pd.concat([old, frame], ignore_index=True)
    frame.sort_values(["labels", "level", "model", "match"]).to_csv(
        RESULTS, index=False, float_format="%.4f")

    shown = frame[frame["model"].isin(args.models) & frame["level"].isin(args.levels)
                  & frame["match"].isin(matches)]
    summary = shown.groupby(["labels", "level", "model"]).agg(
        spell_f1=("spell_f1", "mean"), spell_sd=("spell_f1", "std"),
        macro_f1=("macro_f1", "mean"), d_f1=("d_f1", "mean"), team_acc=("team_acc_alive", "mean"),
        possessions=("possessions", "mean"), true_possessions=("true_possessions", "mean"),
        ball_missing=("ball_missing", "mean"))
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(summary)
    print("wrote {}".format(os.path.relpath(RESULTS, ROOT)))


if __name__ == "__main__":
    main()
