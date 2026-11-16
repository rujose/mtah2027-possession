"""Night 2026-11-14, idea `netavg` (docs/lab/night/network.md): B's GRU P = the plain mean of gru_b's and A's GRU's.

    python -m bench.lab_netavg prepare      # A's GRU on each match's training-stress copy (inference only), then the mean

No training. A's GRU is lab_temporal's network (trained on the normal DFL matches, its own folds: lab_gate.gru_file);
its held-out P on the bench variants already exists (lab_temporal's normal P, lab_gate's p10 / p25 / p50 / stressed),
read only. Its P on the training-stress copies (the letters B's post stages learn from) is computed here with the same
fold nets and normalisers (lab_gate.gru_model / gru_proba), into data/external/lab_netavg/a_gru/. The mean
(round(clip(mean, 0.01, 0.99), 3), each member through float32) goes to data/external/lab_netavg/main/.
SkillCorner: lab_netmirror.skillcorner with both deployment nets (each on all 7 DFL).

Nothing here reads our match.
"""

import argparse
import os

import numpy as np

from bench import lab, lab_branch_b, lab_carried, lab_gate, lab_netmirror, lab_temporal, sets

# Params
B = lab_branch_b
N = lab_netmirror
C = lab_carried
NAME = "netavg"
WHICH = "all"


def root(sub):
    return N.root_of(NAME, sub)


def a_used(m):
    return tuple(o for o in lab.matches(WHICH) if o != m and sets.source(o) in B.L.TRAIN_SOURCES)


def a_p(variant, m):
    """A's GRU's held-out P on a variant (shared caches, read only), or on the training-stress copy (computed here)."""
    if variant == "train":
        return np.load(N.p_file(root("a_gru"), "train", m))
    return lab_gate.held_out_p("gru", variant, m, WHICH)


def deployment_nets():
    return [os.path.join(N.SHARED_B, "net_{}.pt".format(B.held_list()[-1])), lab_gate.gru_file("metrica1", WHICH)]


def prepare():
    for m in lab.matches(WHICH):
        path = N.p_file(root("a_gru"), "train", m)
        if not os.path.exists(path):
            model = lab_gate.gru_model(a_used(m), WHICH, lab_gate.gru_file(m, WHICH))
            halves = lab_temporal.halves_of(m, WHICH)
            if m == lab.matches(WHICH)[0]:      # the fold net and normaliser give A's saved P (lab_gate's own check)
                q = lab_gate.gru_proba(model, lab_gate.variant_features(m, WHICH, "stressed"), halves)
                print("{}: A's GRU recomputed on stressed against lab_gate's saved P, max |diff| {:.2e}".format(
                    m, float(np.abs(q - a_p("stressed", m)).max())), flush=True)
            p = lab_gate.gru_proba(model, C.stressed_features(m, WHICH, C.TRAIN_SEED), halves)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, p.astype(np.float32))
            print("{}: A's GRU on the training-stress copy".format(m), flush=True)
        for variant in N.BENCH_VARIANTS + ("train",):
            p = N.average([np.load(N.shared_p_file(variant, m)), a_p(variant, m)])
            out = N.p_file(root("main"), variant, m)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, p.astype(np.float32))
        print("{}: the mean written".format(m), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("prepare", "nets"))
    args = parser.parse_args()
    if args.step == "nets":
        return print(" ".join(deployment_nets()))
    return prepare()


if __name__ == "__main__":
    main()
