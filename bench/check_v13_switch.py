"""v13's stoppage switch against the lab, on SkillCorner's 20 matches (never our match).

    python -m bench.check_v13_switch           # ~2 min: the signal and branch B
    python -m bench.check_v13_switch --with-a  # ~30 min more: branch A too, so v13 with and without the switch

On each match, from lab_carried's cache of v12's inputs and robust's features (bench/lab_carried.py sc_prepare):
1. the signal v13 computes (lab_gate.signal on the repaired tracking and v12's inputs) against the one lab_gate
   recorded (bench/results/lab_gate_skillcorner_signals.csv), and the branch v13 takes;
2. branch B through v13's code (possession/models/v13_augment_all.joblib: v13.augment_proba, v13.run_chain_from)
   against lab_carried's deployment path on the same models (deploy_chain with augment's retrained post stages):
   frames that differ must be 0;
3. with --with-a, branch A through v13's code (possession/models/v13_all.joblib, the GRU in float64), so v13's
   score there with the switch can be set beside it without.
SkillCorner's labels are external (MIT); this reads nothing of our match. Writes
bench/results/check_v13_switch.csv.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from bench import lab, lab_carried, lab_gate, lab_skillcorner, skillcorner
from possession import v12, v13

# Params
WHICH = "all"
OUT = os.path.join(lab.RESULTS, "check_v13_switch.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--with-a", action="store_true", help="also run branch A (the GRU; ~1.5 min a match)")
    args = parser.parse_args()
    recorded = pd.read_csv(os.path.join(lab.RESULTS, "lab_gate_skillcorner_signals.csv"), dtype={"match": str})
    recorded = dict(zip(recorded["match"], recorded["signal"]))
    bundle = v13.load_branch_b(WHICH)
    v12_models = v12.load_models(WHICH)
    post = joblib.load(os.path.join(lab_carried.CACHE, "post_augment_{}.joblib".format(WHICH)))
    models_a = v13.load_models(WHICH) if args.with_a else None
    rows = []
    for m in skillcorner.match_ids():
        data = skillcorner.load(m)
        gt = data["labels"]
        match, X, Xs = lab_carried.sc_prepared(m)
        names = list(bundle["names"])
        value = lab_gate.signal(match.track, match.pred, match.halves, match.pitch)
        picked = v13.branch(match.track, match.pred, match.halves, match.pitch, WHICH)
        p_b = v13.augment_proba(match, bundle, WHICH, features=(names, X, Xs))
        b = v13.run_chain_from(match, p_b, bundle, WHICH)
        ref = lab_carried.deploy_chain(match, lab_carried.proba(lab_carried.deployment_augment(), {"X": X, "X_swapped": Xs}),
                                       v12_models, WHICH, post)["timing"]
        with np.load(lab_skillcorner.pred_path(m)) as z:
            v12_letters = z["v12"]
        row = {"match": m, "signal": value, "recorded_signal": recorded.get(m, np.nan), "branch": picked,
               "b_frames_differing": int(np.sum(b != ref)), "v12": lab_skillcorner.score(gt, v12_letters)["spell_f1"],
               "B": lab_skillcorner.score(gt, b)["spell_f1"]}
        if models_a is not None:
            a = v13.run_chain_from(match, v13.start_proba(match, models_a, WHICH), models_a, WHICH)
            row["A"] = lab_skillcorner.score(gt, a)["spell_f1"]
        row["v13"] = row["B"] if picked == "B" else row.get("A", np.nan)
        rows.append(row)
        print("{}: signal {:.4f} (lab_gate {:.4f}), branch {}, B frames differing from lab_carried {}, spellF1 v12 {:.4f} "
              "B {:.4f}{}".format(m, value, row["recorded_signal"], picked, row["b_frames_differing"], row["v12"], row["B"],
                                 " A {:.4f}".format(row["A"]) if "A" in row else ""), flush=True)
        del data, match, X, Xs
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False, float_format="%.5f")
    print("\nsignal: largest difference from lab_gate's {:.1e}; branch B on {} of {}; B frames differing in all: {}".format(
        float(np.nanmax(np.abs(frame["signal"] - frame["recorded_signal"]))), int((frame["branch"] == "B").sum()), len(frame),
        int(frame["b_frames_differing"].sum())))
    means = frame[[c for c in ("v12", "A", "B", "v13") if c in frame]].mean()
    print("spellF1 means: " + ", ".join("{} {:.4f}".format(k, v) for k, v in means.items()))
    if "A" in frame:
        print("v13 (with the switch) better than A on {} of {}".format(int((frame["v13"] > frame["A"]).sum()), len(frame)))
    print("wrote {}".format(os.path.relpath(OUT, lab.ROOT)))


if __name__ == "__main__":
    main()
