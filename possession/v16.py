"""Model v16: v15 with branch B's network replaced by `netmet5`, four GRUs trained with Metrica's sample games too,
their P averaged (docs/models/v16-netmet.md; the night's builder "network": docs/lab/night/network.md).

    python -m possession.v16           # writes evaluation/results/v16.csv; prints only its path and SHA-256

v15 (possession/v15.py) runs v14's branch A unless the stoppage switch's signal (bench/lab_gate.py) is above its
threshold; then its branch B: 0.915 x gru_b's P + 0.085 x augment's (trial 1015 of bench/search_b.py), P(dead) x 0.909,
decoded with the trial's setting, snap and move off, then the trial's post stages. v16 changes one thing, B's GRU P:

- **the GRU P is the mean of several networks' P** (the bundle's `grus`; netmet5: bench/lab_netmet.py's `all` nets,
  seeds 0-3, gru_b's recipe trained on all 10 bench matches, Metrica's 3 included), each clipped and rounded as v13's,
  through float32, averaged, clipped and rounded again (bench/lab_netmirror.average, the lab's ens4 and its
  SkillCorner run); then mixed with augment's exactly as v15 (v14.mix, the bundle's w_gru) and P(dead) scaled as v15;
- **the post stages are trial 1015's, trained on that chain's letters** (the averaged P's out-of-fold chain on the
  bench's training-stress copies; bench/train_v16.py).

Branch A, the switch and its threshold, the kick-off rule, the decoder's and post stages' settings are v15's.
Everything B reads is in one file, possession/models/v16_b_all.joblib (`python -m bench.train_v16`).
`python -m bench.check_v16` shows the port gives the lab's letters and the file the lab's SkillCorner scores.
"""

import argparse
import hashlib
import os

import joblib
import numpy as np

from bench import lab_gate
from possession import v1, v10, v12, v13, v14, v15
from possession.data import PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v16.csv")
BRANCH_B = os.path.join(ROOT, "possession", "models", "v16_b_{}.joblib")
SETS = ("all",)            # v16's branch B exists for this bench set; on any other, v16 is v14 (which is v13 there)


def load_branch_b(which):
    path = BRANCH_B.format(which)
    if not os.path.exists(path):
        raise FileNotFoundError("{} is missing: run `python -m bench.train_v16`".format(path))
    bundle = joblib.load(path)
    if bundle["which"] != which or tuple(bundle["chain"]) != v13.CHAIN[which]:
        raise ValueError("{} was trained for another set or chain".format(path))
    if bundle["threshold"] != lab_gate.THRESHOLD:
        raise ValueError("{}'s threshold {} differs from bench/lab_gate.py's {}".format(path, bundle["threshold"], lab_gate.THRESHOLD))
    if not bundle["grus"]:
        raise ValueError("{} holds no network".format(path))
    return bundle


def average(ps):
    """The networks' P averaged as the lab averaged its seeds (bench/lab_netmirror.average): each member through
    float32, the mean in float64, clipped and rounded. One member is its own P (as the lab's SkillCorner run)."""
    if len(ps) == 1:
        return ps[0]
    return v13.clipped(np.mean([np.asarray(p, np.float32).astype(np.float64) for p in ps], axis=0))


def gru_proba(match, grus, which, dtype=None, features=None):
    """B's GRU P: each network's P (v13.gru_proba, float64 unless dtype says otherwise) on the same features, averaged."""
    features = features or v13.blanked_features(match, which)
    return average([v13.gru_proba(match, g, which, dtype, features) for g in grus])


def b_start_proba(match, bundle, which, dtype=None, features=None):
    """Branch B's start: as v15.b_start_proba, with the networks' averaged P in place of gru_b's."""
    features = features or v13.blanked_features(match, which)
    aug = v13.augment_proba(match, bundle, which, features)
    w = bundle["w_gru"]
    if w == 0:
        p = aug.astype(np.float32).astype(np.float64)
    else:
        p = v14.mix(gru_proba(match, bundle["grus"], which, dtype, features), aug, w)
    if bundle["f_dead"] != 1.0:
        p = p.copy()
        p[:, 2] *= bundle["f_dead"]
        p = v13.clipped(p / p.sum(axis=1, keepdims=True))
    return p


def b_chain_from(match, p, bundle, which):
    """Letters for every frame from B's start: v15's chain (the bundle's decoder setting, snap and move, post stages)."""
    return v15.b_chain_from(match, p, bundle, which)


def predict(tracking, tags=None, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH), bench_set=None):
    """(labels for every video frame, {"pred", "p_start", "branch"}): v15's predict with v16's branch B."""
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
    """Only what Jose needs to check the file, as v10-v15: no statistics about the match. --show-branch adds the
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
