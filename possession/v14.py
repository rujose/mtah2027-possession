"""Model v14: v13 with two settings found by a nested search on Metrica's convention (docs/models/v14-search.md).

    python -m possession.v14           # writes evaluation/results/v14.csv; prints only its path and SHA-256

v13 (possession/v13.py) runs, on bench set `all`, branch A: the mean of a GRU's P and robust's P, decoded, v9's
snap and v11's move, then v12's post stages; and a stoppage switch to branch B (`augment`'s chain). v14 changes
branch A only, on `all` only:

- **the start weighs the GRU 0.8 and robust 0.2** (v13: 0.5 each), and
- **v9's snap and v11's move are off**: the decoded letters go straight to the post stages, refitted on this
  start's letters (`python -m bench.train_v14`, possession/models/v14_all.joblib).

Why: across 2,000 settings drawn at random and scored on Metrica's 3 games (whose labels follow the answer key's
convention, docs/owner-rule.md), those two changes helped every game; confirmed the lab's way (every stage
refitted out of fold): 0.7241 against v13's 0.7162 on `all` (8/10), Metrica +0.016 (3/3), DFL +0.005 (5/7)
(docs/search-metrica.md). Branch B, the switch and its threshold, and the IDSSE-only set (where the change isn't a
gain) are v13's, unchanged. `python -m bench.check_v14` shows the port gives the lab's letters.
"""

import argparse
import hashlib
import os

import joblib
import numpy as np

from possession import v1, v10, v12, v13
from possession.data import PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v14.csv")
MODELS = os.path.join(ROOT, "possession", "models", "v14_{}.joblib")
SETS = ("all",)            # v14's branch A exists for this bench set; on any other, v14 is v13
W_GRU = 0.8                # the GRU's weight in the start (v13: 0.5)
SNAP = False               # v9's snap and v11's move (v13: on)


def mix(p_gru, p_robust, w=W_GRU):
    """The start's P: w * GRU + (1 - w) * robust, clipped and rounded; both through float32 first, as v13.average
    (w = 0.5 is v13.average exactly)."""
    if w == 0.5:
        return v13.average(p_gru, p_robust)
    g = p_gru.astype(np.float32).astype(np.float64)
    r = p_robust.astype(np.float32).astype(np.float64)
    return v13.clipped(w * g + (1 - w) * r)


def start_proba(match, models, which, dtype=None):
    """P(H, A, D) per frame from v14's start: the GRU (in float64 unless dtype says otherwise) and v12's robust."""
    return mix(v13.gru_proba(match, models["gru"], which, dtype), v12.robust_proba(match, models["robust"], which), models["w_gru"])


def run_chain_from(match, p, models, which, settings=None):
    """Letters for every frame from a start's P: -log P decoded with the start's setting, then (if the bundle says so)
    v9's snap and v11's move, then v12's post stages of the set."""
    settings = settings or models["settings"]
    letters = v13.L.decode_cost(match, -np.log(p), **settings["start"])
    if models["snap"]:
        letters = v13.lab_robust.finish(match, letters, which)
    return v13.post_stages(match, np.asarray(letters).astype("<U1"), models, which, settings)


def load_models(which):
    path = MODELS.format(which)
    if not os.path.exists(path):
        raise FileNotFoundError("{} is missing: run `python -m bench.train_v14`".format(path))
    models = joblib.load(path)
    if models["which"] != which or tuple(models["chain"]) != v13.CHAIN[which]:
        raise ValueError("{} was trained for another set or chain".format(path))
    return models


def predict(tracking, tags=None, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH), bench_set=None):
    """(labels for every video frame, {"pred", "p_start", "branch"}): v13's predict with v14's branch A."""
    halves = halves or graded_halves()
    which = bench_set or v10.BENCH_SET
    if which not in SETS:
        return v13.predict(tracking, tags, halves, pitch, which)
    fixed, pred = v12.inputs(tracking, halves, pitch, which)
    match = v12.as_match(fixed, pred, halves, pitch)
    chosen = v13.branch(fixed, pred, halves, pitch, which)
    if chosen == "B":                                       # v13's branch B, unchanged
        models = v13.load_branch_b(which)
        p = v13.augment_proba(match, models, which)
        labels = v13.run_chain_from(match, p, models, which)
    else:
        models = load_models(which)
        p = start_proba(match, models, which)
        labels = run_chain_from(match, p, models, which)
    labels[v12.before_kickoff(tags, len(labels), halves)] = "D"
    return labels, {"pred": pred, "p_start": p, "branch": chosen}


def main():
    """Only what Jose needs to check the file, as v10-v13: no statistics about the match. --show-branch adds the
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
