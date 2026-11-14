"""Train v15's branch B (docs/models/v15-branch-b.md): one file, possession/models/v15_b_all.joblib.

    OMP_NUM_THREADS=2 python -m bench.train_v15 [--trial N]   # ~10 min (the network) + ~5 min (the post stages)

What goes in, for the search trial picked in docs/lab/branch-b.md (TRIAL; bench/search_b.py's seeded draw):
1. `augment`: v13's branch B classifier (lab_carried.deployment_augment: the 7 DFL matches as they are and under
   augment's training stress), unchanged.
2. `gru_b`: bench/lab_branch_b.py's network trained on the same 7 matches the same way (seed 0, deterministic); its
   weights are compared with the lab's fold that never saw Metrica's games (net_metrica1.pt), which they must equal.
3. The trial's post stages, trained on all 10 bench matches' out-of-fold letters of the trial's own chain on the
   trial's data (search_b.trial_models), and its settings: the GRU's weight, the factor on P(dead), the decoder's
   setting, snap and move on/off, which post stages run and their thresholds.
4. The switch's threshold (bench/lab_gate.py), checked on load.
"""

import argparse
import os
import time

import joblib
import numpy as np

from bench import lab, lab_branch_b, lab_carried, lab_gate, lab_robust, lab_temporal, search_b
from possession import v12, v13, v15

# Params
WHICH = "all"
TRIAL = 1015             # the search trial v15's branch B is built from (docs/lab/branch-b.md's verdict, adopted 2026-11-14)


def network():
    """gru_b's deployment network, as a v13 GRU bundle, with its comparison against the lab's fold."""
    used = lab_branch_b.used_for(lab_branch_b.held_list()[-1])
    print("gru_b: training on {} (as they are and stressed)".format(" ".join(used)), flush=True)
    model = lab_branch_b.fit(used, log=lambda s: print(s, flush=True))
    torch = lab_temporal.torch_ready()
    ref = {k: v.numpy() for k, v in torch.load(lab_branch_b.net_path(lab_branch_b.held_list()[-1])).items()}
    state = {k: v.detach().cpu().numpy() for k, v in model["net"].state_dict().items()}
    identical = list(ref) == list(state) and all(np.array_equal(ref[k], state[k]) for k in ref)
    print("gru_b against the lab's {}: bit-identical {}".format(os.path.basename(lab_branch_b.net_path("metrica1")), identical),
          flush=True)
    if not identical:
        raise SystemExit("the deployment network differs from the lab's fold: not written")
    names = lab_robust.cached(used[0], WHICH)["names"]
    return v13.gru_bundle(model["net"].state_dict(), model["norm"], names, trained_on=list(used), history=model["history"],
                          seconds=model["seconds"], which=WHICH, recipe="bench/lab_branch_b.py: as is + augment's training stress")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trial", type=int, default=TRIAL)
    args = parser.parse_args()
    if args.trial is None:
        raise SystemExit("no trial: set TRIAL (or pass --trial) from docs/lab/branch-b.md's verdict")
    started = time.time()
    t = search_b.every_trial()[args.trial]
    print("trial {}: {}".format(args.trial, t), flush=True)
    names = v12.load_models(WHICH)["robust"]["names"]
    augment = lab_carried.deployment_augment()
    if augment.n_features_in_ != len(names):
        raise ValueError("augment's columns are not robust's 97 cached features")
    lab_search = search_b.trial_models(t)
    gru = network() if t["w_gru"] > 0 else None
    settings = {"start": {"switch_team": t["switch_team"], "switch_dead": t["switch_dead"],
                          "min_possession": int(t["min_possession"])},
                **{stage: search_b.trial_setting(t, stage) for stage in search_b.CHAIN}}
    bundle = {"augment": augment, "names": names, "gru": gru, "w_gru": float(t["w_gru"]), "f_dead": float(t["f_dead"]),
              "snap": bool(t["snap"]), "stages": tuple(s for s in search_b.CHAIN if t[s]), "settings": settings,
              **lab_search, "post_data": t["post"], "trial": dict(t), "chain": v13.CHAIN[WHICH], "which": WHICH,
              "threshold": lab_gate.THRESHOLD}
    path = v15.BRANCH_B.format(WHICH)
    joblib.dump(bundle, path, compress=3)
    print("wrote {} ({:.1f} MB) in {:.0f} s".format(os.path.relpath(path, lab.ROOT), os.path.getsize(path) / 1e6,
                                                     time.time() - started))


if __name__ == "__main__":
    main()
