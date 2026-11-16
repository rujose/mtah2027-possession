"""Night 2026-11-14, idea `netens` (docs/lab/night/network.md): five seeds of gru_b, P averaged.

    python -m bench.lab_netens train --seed S --fold HELD     # one fold of seed S (~8 min, < 1.5 GB), then its held-out P
    python -m bench.lab_netens check                          # predict_into with gru_b's own nets reproduces its P (one match)
    python -m bench.lab_netens combine --seeds 0 1 2 3 4 --sub ens5   # the members' mean P into a mirror
    python -m bench.lab_netens frames                         # held-out log loss of every seed and ensemble (information)

Seed S (1-4) is gru_b's recipe exactly (lab_branch_b.fit, which reads lab_temporal.SEED for torch and numpy), the same
folds (lab_branch_b.held_list: each DFL match, and metrica1's fold on all 7 DFL for Metrica's 3), the same settings.
Seed 0 is gru_b itself: its nets and P are read from the shared caches, never retrained or written.
The ensemble's P = round(clip(mean of the members' P, 0.01, 0.99), 3) on every variant and the training-stress copies.
Each seed's files: data/external/lab_netens/seedS/ (nets/, lab_branch_b/, search_b/); the ensemble's: lab_netens/ensK/.

Nothing here reads our match.
"""

import argparse
import json
import os

import numpy as np

from bench import lab, lab_branch_b, lab_netmirror, sets

# Params
B = lab_branch_b
N = lab_netmirror
NAME = "netens"
SEEDS = (0, 1, 2, 3, 4)
WHICH = "all"


def root(sub):
    return N.root_of(NAME, sub)


def net_path(seed, held):
    if seed == 0:
        return os.path.join(N.SHARED_B, "net_{}.pt".format(held))
    return os.path.join(root("seed{}".format(seed)), "nets", "net_{}.pt".format(held))


def fold_matches(held):
    """The matches whose held-out P comes from this fold's net."""
    if sets.source(held) in B.L.TRAIN_SOURCES:
        return [held]
    return [m for m in lab.matches(WHICH) if sets.source(m) not in B.L.TRAIN_SOURCES]


def train(seed, held):
    if seed == 0:
        raise ValueError("seed 0 is gru_b itself: never retrained here")
    path = net_path(seed, held)
    used = B.used_for(held)
    if not os.path.exists(path):
        B.T.SEED = seed                  # read by lab_branch_b.fit at call time (torch and numpy)
        print("seed {} fold {}: training on {} (as they are and stressed)".format(seed, held, " ".join(used)), flush=True)
        model = B.fit(used, log=lambda s: print(s, flush=True))
        model.update({"norm_on": list(used), "seed": seed, "history_head": None})
        N.save_net(model, path)
        print("seed {} fold {}: saved ({:.0f} s)".format(seed, held, model["seconds"]), flush=True)
    for m in fold_matches(held):
        N.predict_into(root("seed{}".format(seed)), m, path, used)


def check(m="J03WMX"):
    """N.predict_into with gru_b's own net must give gru_b's saved P (normal, stressed, training-stress copy)."""
    tmp = root("check")
    held = m if sets.source(m) in B.L.TRAIN_SOURCES else B.held_list()[-1]
    N.predict_into(tmp, m, net_path(0, held), B.used_for(held))
    out = {}
    for variant in N.BENCH_VARIANTS + ("train",):
        a = np.load(N.p_file(tmp, variant, m))
        b = np.load(N.shared_p_file(variant, m))
        out[variant] = float(np.abs(a.astype(np.float64) - b).max())
    print("predict check on {}: max |diff| per variant {}".format(m, out), flush=True)
    with open(os.path.join(tmp, "check.json"), "w") as handle:
        json.dump(out, handle)
    return out


def member_p(seed, variant, m):
    path = N.shared_p_file(variant, m) if seed == 0 else N.p_file(root("seed{}".format(seed)), variant, m)
    return np.load(path)


def combine(seeds, sub):
    out = root(sub)
    for m in lab.matches(WHICH):
        for variant in N.BENCH_VARIANTS + ("train",):
            p = N.average([member_p(s, variant, m) for s in seeds])
            path = N.p_file(out, variant, m)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, p.astype(np.float32))
    with open(os.path.join(out, "members.json"), "w") as handle:
        json.dump({"seeds": list(seeds)}, handle)
    print("{}: the mean of seeds {} written".format(sub, list(seeds)), flush=True)


def seed_root(seed):
    return root("seed{}".format(seed))


def make_seed_mirror(seed):
    """A seed alone as B's GRU (the noise yardstick): its P files are already in its own mirror."""
    missing = N.missing_inputs(seed_root(seed))
    print("seed {}: {} files missing".format(seed, len(missing)), flush=True)
    return missing


def frames():
    roots = {"seed0": None}
    for s in SEEDS[1:]:
        if os.path.isdir(seed_root(s)):
            roots["seed{}".format(s)] = seed_root(s)
    for sub in sorted(os.listdir(N.root_of(NAME, ""))):
        if sub.startswith("ens"):
            roots[sub] = root(sub)
    df = N.frame_table(roots)
    os.makedirs(N.results_of(NAME, ""), exist_ok=True)
    df.to_csv(os.path.join(N.results_of(NAME, ""), "frames.csv"), index=False, float_format="%.5f")
    print(df.pivot_table(index="source", columns="variant", values="log_loss").round(4).to_string())
    print(df.pivot_table(index="source", columns="variant", values="acc").round(4).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("train", "check", "combine", "frames"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--sub", default="ens5")
    args = parser.parse_args()
    if args.step == "train":
        return train(args.seed, args.fold)
    if args.step == "check":
        return check()
    if args.step == "combine":
        return combine(args.seeds, args.sub)
    return frames()


if __name__ == "__main__":
    main()
