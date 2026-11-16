"""Night 2026-11-14, idea `netmet` (docs/lab/night/network.md): gru_b trained with Metrica's games too.

    python -m bench.lab_netmet train --fold HELD     # HELD one of the 10 bench matches, or `all` (the deployment net)
    python -m bench.lab_netmet report                # Metrica's 3 and DFL's 7 apart, and the long live->dead runs

gru_b's recipe (seed 0, 20 epochs, both orientations, each training match as it is and under augment's training
stress), with leave one match out over all 10: a DFL fold trains on the other 6 DFL + Metrica's 3, a Metrica fold on
the 7 DFL + the other 2 Metrica; `all` trains on the 10 (SkillCorner's deployment net). The normaliser and missing
flags stay the fold's IDSSE matches' (lab_temporal.normaliser), so the inputs read as gru_b's: a DFL fold's is gru_b's
own fold normaliser, a Metrica fold's (and `all`'s) the 7 DFL matches'.
Files: data/external/lab_netmet/main/ (nets/, lab_branch_b/, search_b/).

Nothing here reads our match.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from bench import lab, lab_branch_b, lab_netmirror, sets

# Params
B = lab_branch_b
N = lab_netmirror
NAME = "netmet"
SUB = "main"
WHICH = "all"
SEED = 0


def root(seed=SEED):
    return N.root_of(NAME, SUB if seed == SEED else "seed{}".format(seed))


def used_for(held):
    return tuple(m for m in lab.matches(WHICH) if m != held)


def norm_for(held):
    return tuple(m for m in used_for(held) if sets.source(m) in B.L.TRAIN_SOURCES)


def net_path(held, seed=SEED):
    return os.path.join(root(seed), "nets", "net_{}.pt".format(held))


def train(held, seed=SEED):
    path = net_path(held, seed)
    used, norm_used = used_for(held), norm_for(held)
    if not os.path.exists(path):
        print("netmet fold {}: training on {} (as they are and stressed); normaliser on {}".format(
            held, " ".join(used), " ".join(norm_used)), flush=True)
        model = N.fit(used, norm_used=norm_used, seed=seed, log=lambda s: print(s, flush=True))
        N.save_net(model, path)
        print("netmet fold {}: saved ({:.0f} s)".format(held, model["seconds"]), flush=True)
    if held != "all":
        N.predict_into(root(seed), held, path, norm_used)


def combine(seeds, sub):
    """The mean of several seeds' held-out P (round(clip(mean, 0.01, 0.99), 3)) into lab_netmet/<sub>/."""
    import json
    out = N.root_of(NAME, sub)
    for m in lab.matches(WHICH):
        for variant in N.BENCH_VARIANTS + ("train",):
            p = N.average([np.load(N.p_file(root(s), variant, m)) for s in seeds])
            path = N.p_file(out, variant, m)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, p.astype(np.float32))
    with open(os.path.join(out, "members.json"), "w") as handle:
        json.dump({"seeds": list(seeds), "deployment": [net_path("all", s) for s in seeds]}, handle)
    print("{}: the mean of netmet seeds {} written".format(sub, list(seeds)), flush=True)


def report():
    """Per source; the frame errors F4 looked at (live called dead, in runs of 3 s or more), before and after."""
    rows = []
    for m in lab.matches(WHICH):
        for variant in ("p25", "p50", "stressed", "normal"):
            c = N.lab_gate.variant_features(m, WHICH, variant)
            truth = np.asarray(c["truth"])
            del c
            for name, path in (("gru_b", N.shared_p_file(variant, m)), ("netmet", N.p_file(root(), variant, m))):
                if not os.path.exists(path):
                    continue
                p = np.load(path).astype(np.float64)
                f = B.T.frame_check(p, truth)
                live = np.isin(truth, (0, 1))
                called_dead = p.argmax(axis=1) == 2
                bad = live & called_dead
                # runs of 3 s or more (75 frames) of live called dead
                long = 0
                edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
                for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    if b - a >= 75:
                        long += b - a
                graded = np.isin(truth, (0, 1, 2))
                rows.append({"source": name, "match": m, "provider": sets.source(m), "variant": variant, **f,
                             "live_called_dead": float(bad.sum() / graded.sum()),
                             "live_called_dead_long": float(long / graded.sum())})
    df = pd.DataFrame(rows)
    os.makedirs(N.results_of(NAME, SUB), exist_ok=True)
    df.to_csv(os.path.join(N.results_of(NAME, SUB), "frames.csv"), index=False, float_format="%.5f")
    print(df.groupby(["provider", "variant", "source"])[["log_loss", "acc", "live_called_dead", "live_called_dead_long"]]
          .mean().round(4).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("train", "report", "combine"))
    parser.add_argument("--fold")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sub", default="ens5")
    parser.add_argument("--seed", type=int, default=SEED, help="train: the seed (not 0: files under seedS/, the noise rule)")
    args = parser.parse_args()
    if args.step == "train":
        return train(args.fold, args.seed)
    if args.step == "combine":
        return combine(args.seeds, args.sub)
    return report()


if __name__ == "__main__":
    main()
