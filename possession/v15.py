"""Model v15: v14 with a new branch B, the chain the stoppage switch runs where the ball stays tracked and moving
through stoppages (docs/models/v15-branch-b.md, lab round 8: docs/lab/branch-b.md).

    python -m possession.v15           # writes evaluation/results/v15.csv; prints only its path and SHA-256

v14 (possession/v14.py) runs branch A unless the switch's signal (bench/lab_gate.py) is above its threshold; then it
runs v13's branch B: `augment` (robust's trees retrained with SkillCorner-like stoppage balls), decoded with v12's
setting, snapped and moved, then v12's post stages trained on augment's letters on the NORMAL bench. v15 keeps branch
A, the switch and its threshold, and changes branch B (bench set `all` only):

- **the start** mixes `augment` with `gru_b`, lab_temporal's network trained the way augment was (each match as it is
  and with SkillCorner-like stoppage balls), weighted by the bundle's `w_gru`; P(dead) scaled by `f_dead`;
- **the decoder, snap and move, each post stage on/off and its thresholds** are the bundle's (bench/search_b.py);
- **the post stages are trained on the bundle's data** (the bench as it is, under augment's training stress, or both).

Everything B reads is in one file, possession/models/v15_b_all.joblib (`python -m bench.train_v15`).
`python -m bench.check_v15` shows the port gives the lab's letters.
"""

import argparse
import hashlib
import os

import joblib
import numpy as np

from bench import lab_gate
from possession import v1, v10, v12, v13, v14
from possession.data import PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v15.csv")
BRANCH_B = os.path.join(ROOT, "possession", "models", "v15_b_{}.joblib")
SETS = ("all",)            # v15's branch B exists for this bench set; on any other, v15 is v14 (which is v13 there)
L = v13.L


def load_branch_b(which):
    path = BRANCH_B.format(which)
    if not os.path.exists(path):
        raise FileNotFoundError("{} is missing: run `python -m bench.train_v15`".format(path))
    bundle = joblib.load(path)
    if bundle["which"] != which or tuple(bundle["chain"]) != v13.CHAIN[which]:
        raise ValueError("{} was trained for another set or chain".format(path))
    if bundle["threshold"] != lab_gate.THRESHOLD:
        raise ValueError("{}'s threshold {} differs from bench/lab_gate.py's {}".format(path, bundle["threshold"], lab_gate.THRESHOLD))
    return bundle


def b_start_proba(match, bundle, which, dtype=None, features=None):
    """Branch B's start: augment's P mixed with gru_b's (v14.mix, both through float32; w 0 is augment alone, through
    float32 as the lab read its saved P), P(dead) scaled by f_dead and the rows renormalised, clipped and rounded."""
    features = features or v13.blanked_features(match, which)
    aug = v13.augment_proba(match, bundle, which, features)
    w = bundle["w_gru"]
    if w == 0:
        p = aug.astype(np.float32).astype(np.float64)
    else:
        p = v14.mix(v13.gru_proba(match, bundle["gru"], which, dtype, features), aug, w)
    if bundle["f_dead"] != 1.0:
        p = p.copy()
        p[:, 2] *= bundle["f_dead"]
        p = v13.clipped(p / p.sum(axis=1, keepdims=True))
    return p


def b_chain_from(match, p, bundle, which):
    """Letters for every frame from B's start: decoded with the bundle's setting, then (if on) v9's snap and v11's
    move, then each of v12's post stages the bundle keeps, with its model and setting."""
    settings = bundle["settings"]
    letters = L.decode_cost(match, -np.log(p), **settings["start"])
    if bundle["snap"]:
        letters = v13.lab_robust.finish(match, letters, which)
    letters = np.asarray(letters).astype("<U1")
    for stage in v13.CHAIN[which]:
        if stage not in bundle["stages"]:
            continue
        s = settings[stage]
        if stage == "short":
            letters = v13.lab_short.apply(match, letters, model=bundle[stage], **s)
        elif stage == "segments":
            letters = v13.lab_segments.apply(match, letters, bundle[stage], **s)
        elif stage == "timing":
            letters = v13.lab_timing.apply(match, letters, bundle[stage], **s)
        else:
            raise ValueError(stage)
        letters = np.asarray(letters).astype("<U1")
    return letters


def predict(tracking, tags=None, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH), bench_set=None):
    """(labels for every video frame, {"pred", "p_start", "branch"}): v14's predict with v15's branch B."""
    halves = halves or graded_halves()
    which = bench_set or v10.BENCH_SET
    if which not in SETS:
        return v14.predict(tracking, tags, halves, pitch, which)
    fixed, pred = v12.inputs(tracking, halves, pitch, which)
    match = v12.as_match(fixed, pred, halves, pitch)
    chosen = v13.branch(fixed, pred, halves, pitch, which)
    if chosen == "B":
        bundle = load_branch_b(which)
        p = b_start_proba(match, bundle, which)
        labels = b_chain_from(match, p, bundle, which)
    else:                                                   # v14's branch A, unchanged
        models = v14.load_models(which)
        p = v14.start_proba(match, models, which)
        labels = v14.run_chain_from(match, p, models, which)
    labels[v12.before_kickoff(tags, len(labels), halves)] = "D"
    return labels, {"pred": pred, "p_start": p, "branch": chosen}


def main():
    """Only what Jose needs to check the file, as v10-v14: no statistics about the match. --show-branch adds the
    branch the switch took (Jose's to read; Claude doesn't run it on our match)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--show-branch", action="store_true", help="also print the branch the stoppage switch took")
    args = parser.parse_args()
    labels, info = predict(load_tracking(), load_tags())
    v1.write_submission(OUTPUT, labels)
    with open(OUTPUT, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    path = os.path.relpath(OUTPUT, ROOT)
    print("wrote {} (bench set {})".format(path, v10.BENCH_SET))
    print("SHA-256 {}".format(digest))
    if args.show_branch:
        print("stoppage switch: branch {}".format(info["branch"]))
    print("format check: .venv/bin/python evaluation/validate_submission.py {}".format(path))


if __name__ == "__main__":
    main()
