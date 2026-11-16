"""Train v16's branch B (docs/models/v16-netmet.md): one file, possession/models/v16_b_all.joblib.

    OMP_NUM_THREADS=2 python -m bench.train_v16 nets --seeds 0 1   # ~15 min a seed: deployment nets, each against the lab's
    OMP_NUM_THREADS=2 python -m bench.train_v16 build              # any net not yet trained, then the post stages; the file
    OMP_NUM_THREADS=2 python -m bench.train_v16 tables             # (check) trial 1015's stage tables rebuilt from the P

v16 is v15 (possession/v15.py) with branch B's GRU replaced by a RECIPE's networks, one per seed, their P averaged
(lab_netmirror.average: each through float32, the mean, clipped and rounded). Everything else in B is v15's search
trial 1015. What goes in, for the recipe (default `netmet5`, the night's ens4: docs/lab/night/network.md):
1. `augment`: v13's branch B classifier (lab_carried.deployment_augment), unchanged from v15.
2. The recipe's deployment networks: per seed, the recipe's `all` net (netmet: gru_b's recipe trained on all 10 bench
   matches, as they are and under augment's training stress; normaliser on the 7 DFL matches). Each is compared with
   the lab's `all` net of that seed, which it must equal bit for bit, or nothing is written. Trained nets are kept
   in data/external/train_v16/<recipe>/ so the seeds can train in separate processes (`nets`) before `build`.
3. Trial 1015's post stages, trained on all 10 bench matches' out-of-fold letters of trial 1015's chain started from
   the recipe's averaged held-out P on the training-stress copies (search_b.trial_models, pointed at the recipe's lab
   mirror by lab_netmirror.use: what the lab's confirm of the recipe used), and trial 1015's settings (v15's, checked).
4. The switch's threshold (bench/lab_gate.py), checked on load.

A recipe (RECIPES) is: its seeds; how its `all` net is trained for a seed (`fit(seed, log)` -> lab_netmirror.fit's
dict); where the lab saved that net (the bit-for-bit reference; None: no reference, written only with
--no-reference); its members' lab folders and the lab mirror holding their averaged out-of-fold P and trial 1015's
stage tables; its fold nets and their normalisers' matches (for bench/check_v16.py's port). Another recipe with
lab_netmet's interface (track 2, docs/lab/push/netweight.md) is one more entry.
"""

import argparse
import json
import os
import time

import joblib
import numpy as np

from bench import lab, lab_carried, lab_gate, lab_netmet, lab_netmirror, lab_robust, lab_temporal, search_b
from possession import v12, v13, v15, v16

# Params
WHICH = "all"
TRIAL = 1015             # v15's search trial: v16 changes only the network behind B's GRU P
RECIPE = "netmet5"       # docs/lab/push.md, track 1 (Jose's "Yes, all three", 2026-11-15)
NETS = os.path.join(lab.ROOT, "data", "external", "train_v16")      # gitignored (data/*)
RECIPES = {
    "netmet5": {
        "about": "bench/lab_netmet.py: gru_b's recipe (seed S, 20 epochs, each match as it is and under augment's "
                 "training stress) trained with Metrica's 3 games too; the `all` net on all 10 bench matches, "
                 "normaliser on the 7 DFL; seeds 0-3, P averaged (the night's ens4)",
        "seeds": (0, 1, 2, 3),
        "fit": lambda seed, log: lab_netmirror.fit(lab_netmet.used_for("all"), norm_used=lab_netmet.norm_for("all"),
                                                   seed=seed, log=log),
        "lab_net": lambda seed: lab_netmet.net_path("all", seed),
        "member": lambda seed: lab_netmet.root(seed),
        "mirror": ("netmet", "ens4"),
        # the lab's fold that never saw a bench match, and its normaliser's matches (bench/check_v16.py's port)
        "fold_net": lambda held, seed: lab_netmet.net_path(held, seed),
        "fold_norm": lambda held: lab_netmet.norm_for(held),
    },
}


def net_cache(recipe, seed):
    return os.path.join(NETS, recipe, "gru_all_seed{}.joblib".format(seed))


def train_net(recipe, seed, reference=True):
    """One deployment net: trained, compared bit for bit with the lab's, kept as a v13 GRU bundle."""
    path = net_cache(recipe, seed)
    if os.path.exists(path):
        return joblib.load(path)
    r = RECIPES[recipe]
    print("{} seed {}: training".format(recipe, seed), flush=True)
    model = r["fit"](seed, lambda s: print(s, flush=True))
    state = {k: v.detach().cpu().numpy() for k, v in model["net"].state_dict().items()}
    ref_path = r["lab_net"](seed)
    identical = None
    if ref_path is not None and os.path.exists(ref_path):
        torch = lab_temporal.torch_ready()
        ref = {k: v.numpy() for k, v in torch.load(ref_path).items()}
        identical = list(ref) == list(state) and all(np.array_equal(ref[k], state[k]) for k in ref)
        print("{} seed {} against the lab's {}: bit-identical {}".format(recipe, seed, os.path.relpath(ref_path, lab.ROOT),
                                                                        identical), flush=True)
        if not identical:
            raise SystemExit("seed {}: the deployment net differs from the lab's: not written".format(seed))
    elif reference:
        raise SystemExit("seed {}: the lab has no `all` net to compare with ({}); --no-reference to write it anyway".format(
            seed, ref_path))
    names = lab_robust.cached(model["trained_on"][0], WHICH)["names"]
    gru = v13.gru_bundle(model["net"].state_dict(), model["norm"], names, trained_on=list(model["trained_on"]),
                         norm_on=list(model["norm_on"]), seed=int(seed), history=model["history"],
                         seconds=model["seconds"], which=WHICH, recipe=r["about"],
                         lab_net=os.path.relpath(ref_path, lab.ROOT) if ref_path else None, bit_identical=identical)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(gru, path + ".part")
    os.replace(path + ".part", path)
    print("{} seed {}: kept in {} ({:.0f} s)".format(recipe, seed, os.path.relpath(path, lab.ROOT), model["seconds"]),
          flush=True)
    return gru


def check_mirror(recipe):
    """The lab mirror's P on the training-stress copies (what the post stages learn from) is the mean of the recipe's
    members' P, as v16 averages (lab_netmirror.average); the mirror's trial 1015 tables exist."""
    r = RECIPES[recipe]
    root = lab_netmirror.root_of(*r["mirror"])
    worst = 0.0
    for m in lab.matches(WHICH):
        got = np.load(lab_netmirror.p_file(root, "train", m))
        want = lab_netmirror.average([np.load(lab_netmirror.p_file(r["member"](s), "train", m)) for s in r["seeds"]])
        worst = max(worst, float(np.abs(got.astype(np.float64) - want.astype(np.float32).astype(np.float64)).max()))
    print("{}: the mirror's training-copy P against the mean of seeds {}: max |difference| {}".format(
        recipe, list(r["seeds"]), worst), flush=True)
    if worst != 0:
        raise SystemExit("the mirror's P is not the mean of the recipe's members")
    if not os.path.exists(os.path.join(root, "search_b", "tables_trial{}.joblib".format(TRIAL))):
        print("  (the mirror has no trial {} tables: search_b computes them into it)".format(TRIAL), flush=True)


def trial_and_check():
    """Trial 1015's settings, which must be the ones v15's file deploys."""
    t = search_b.every_trial()[TRIAL]
    built = v15.load_branch_b(WHICH)["trial"]
    if dict(t) != dict(built):
        raise SystemExit("trial {} differs from v15's file: {} vs {}".format(TRIAL, t, built))
    return t


def build(recipe, seeds, reference=True):
    started = time.time()
    t = trial_and_check()
    print("trial {} (v15's): {}".format(TRIAL, t), flush=True)
    names = v12.load_models(WHICH)["robust"]["names"]
    augment = lab_carried.deployment_augment()
    if augment.n_features_in_ != len(names):
        raise ValueError("augment's columns are not robust's 97 cached features")
    grus = [train_net(recipe, s, reference) for s in seeds]
    if any(list(g["names"]) != list(names) for g in grus):
        raise ValueError("a network's features are not robust's 97")
    check_mirror(recipe)
    lab_netmirror.use_named(*RECIPES[recipe]["mirror"])              # search_b's cache: the recipe's P and tables
    stages = search_b.trial_models(t)
    settings = {"start": {"switch_team": t["switch_team"], "switch_dead": t["switch_dead"],
                          "min_possession": int(t["min_possession"])},
                **{stage: search_b.trial_setting(t, stage) for stage in search_b.CHAIN}}
    bundle = {"augment": augment, "names": names, "grus": grus, "recipe": recipe, "seeds": [int(s) for s in seeds],
              "w_gru": float(t["w_gru"]), "f_dead": float(t["f_dead"]), "snap": bool(t["snap"]),
              "stages": tuple(s for s in search_b.CHAIN if t[s]), "settings": settings, **stages, "post_data": t["post"],
              "post_tables": os.path.relpath(os.path.join(search_b.CACHE, "tables_trial{}.joblib".format(TRIAL)), lab.ROOT),
              "trial": dict(t), "chain": v13.CHAIN[WHICH], "which": WHICH, "threshold": lab_gate.THRESHOLD}
    v15_b = v15.load_branch_b(WHICH)
    same = {k: joblib.hash(bundle[k]) == joblib.hash(v15_b[k])
            for k in ("augment", "names", "w_gru", "f_dead", "snap", "stages", "settings", "post_data", "chain", "threshold")}
    print("as v15's file: {}".format(same), flush=True)
    if not all(same.values()):
        raise SystemExit("v16's B differs from v15's beyond the network and the post stages' models")
    path = v16.BRANCH_B.format(WHICH)
    joblib.dump(bundle, path, compress=3)
    print("wrote {} ({:.2f} MB; {} networks) in {:.0f} s".format(os.path.relpath(path, lab.ROOT),
                                                                  os.path.getsize(path) / 1e6, len(grus),
                                                                  time.time() - started))


def tables(recipe):
    """Trial 1015's out-of-fold stage tables rebuilt in memory from the recipe's averaged P (search_b.oof_tables on
    the training-stress copies, as trial_tables builds them), against the lab mirror's cached file. Writes nothing."""
    import pandas as pd
    started = time.time()
    t = trial_and_check()
    lab_netmirror.use_named(*RECIPES[recipe]["mirror"])
    cached = joblib.load(os.path.join(search_b.CACHE, "tables_trial{}.joblib".format(TRIAL)))
    kinds = {"normal": ("normal",), "stressed": ("train",), "both": ("normal", "train")}[t["post"]]
    rebuilt = {}
    for data in kinds:
        matches = [lab.load(m, WHICH) if data == "normal" else search_b.train_copy(m) for m in lab.matches(WHICH)]
        for stage, tabs in search_b.oof_tables(matches, data, t).items():
            rebuilt.setdefault(stage, {}).update(tabs)
        del matches
    def equal(a, b):
        """A stage table: a DataFrame (segments, timing) or a (DataFrame, labels) tuple (short)."""
        if isinstance(b, tuple):
            return isinstance(a, tuple) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(b, pd.DataFrame):
            return isinstance(a, pd.DataFrame) and a.equals(b)
        return np.array_equal(np.asarray(a), np.asarray(b))
    same = {stage: set(rebuilt.get(stage, {})) == set(cached[stage])
            and all(equal(rebuilt[stage][i], cached[stage][i]) for i in cached[stage]) for stage in cached}
    out = {"recipe": recipe, "tables": os.path.relpath(os.path.join(search_b.CACHE, "tables_trial{}.joblib".format(TRIAL)),
                                                       lab.ROOT), "identical": same, "seconds": time.time() - started}
    print(json.dumps(out, indent=1), flush=True)
    os.makedirs(os.path.join(lab.RESULTS, "train_v16"), exist_ok=True)
    with open(os.path.join(lab.RESULTS, "train_v16", "tables_{}.json".format(recipe)), "w") as handle:
        json.dump(out, handle, indent=1)
    if not all(same.values()):
        raise SystemExit("the rebuilt tables differ from the lab's")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", nargs="?", default="build", choices=("nets", "build", "tables"))
    parser.add_argument("--recipe", default=RECIPE, choices=sorted(RECIPES))
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="default: the recipe's")
    parser.add_argument("--no-reference", action="store_true", help="write a net the lab has no `all` net for")
    args = parser.parse_args()
    seeds = tuple(args.seeds) if args.seeds is not None else RECIPES[args.recipe]["seeds"]
    if args.step == "nets":
        for s in seeds:
            train_net(args.recipe, s, not args.no_reference)
        return None
    if args.step == "tables":
        return tables(args.recipe)
    if seeds != RECIPES[args.recipe]["seeds"]:
        raise SystemExit("build takes the recipe's seeds {} (its mirror's P and tables are their mean)".format(
            RECIPES[args.recipe]["seeds"]))
    return build(args.recipe, seeds, not args.no_reference)


if __name__ == "__main__":
    main()
