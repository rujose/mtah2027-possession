"""v15's branch B against the lab (docs/models/v15-branch-b.md); never our match.

    python -m bench.check_v15 port          # ~5 min: the port with fold models on bench units, against bench/search_b.py
    python -m bench.check_v15 skillcorner   # ~10 min: the built file on SkillCorner's 20, against search_b's deployment run

port: on bench units (one match on one stressed variant), branch B through v15's code with the lab's fold models
(augment's and gru_b's that never saw the match, the trial's stage models trained without it), its features
computed from the variant's tracking, in float32 and float64, against search_b.letters_of with the lab's saved P.
skillcorner: possession/models/v15_b_all.joblib through v15's code on each SkillCorner match (lab_carried's cached
inputs), against search_b.skillcorner's scores for the same trial; and the branch the switch takes there.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from bench import lab, lab_branch_b, lab_carried, lab_gate, lab_temporal, search_b
from bench.run import score as bench_score
from possession import v13, v15

# Params
WHICH = "all"
UNITS = (("J03WR9", "stressed"), ("metrica1", "p50"), ("J03WOH", "p25"))


def fold_bundle(m, t, built):
    """The built bundle with the lab's fold models for match m in place of the deployment ones."""
    held = m if m in lab_branch_b.held_list()[:-1] else lab_branch_b.held_list()[-1]
    folds = joblib.load(os.path.join(lab_gate.folder(WHICH), "folds_{}.joblib".format(held)))
    gru = None
    if t["w_gru"] > 0:
        torch = lab_temporal.torch_ready()
        used = lab_branch_b.used_for(held)
        gru = v13.gru_bundle(torch.load(lab_branch_b.net_path(held)), lab_temporal.normaliser(used, WHICH), built["names"])
    return {**built, "augment": folds["augment"], "gru": gru, **search_b.trial_models(t, without=m)}


def port():
    import torch  # noqa: F401  (lab_temporal sets its threads)
    built = v15.load_branch_b(WHICH)
    t = built["trial"]
    lab_search_trial = {**t, "post": "own"}
    rows = []
    for m, variant in UNITS:
        match = lab_gate.variant_match(m, WHICH, variant)
        bundle = fold_bundle(m, t, built)
        ref = search_b.letters_of(match, variant, lab_search_trial, {"own": {k: bundle[k] for k in bundle["stages"]}})
        p_lab = search_b.start_p(m, variant, t["w_gru"], t["f_dead"])
        features = v13.blanked_features(match, WHICH)
        for dtype in (np.float32, np.float64):
            p = v15.b_start_proba(match, bundle, WHICH, dtype, features)
            letters = v15.b_chain_from(match, p, bundle, WHICH)
            rows.append({"match": m, "variant": variant, "dtype": np.dtype(dtype).name,
                         "p_max_abs_diff": float(np.abs(p - p_lab).max()), "frames_differing": int(np.sum(letters != ref)),
                         "spell_f1": bench_score(match.truth, letters)["spell_f1"],
                         "lab_spell_f1": bench_score(match.truth, ref)["spell_f1"]})
            print(rows[-1], flush=True)
        del match, features
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(lab.RESULTS, "check_v15_port.csv"), index=False, float_format="%.6f")
    print("frames differing in all: {}".format(int(df.frames_differing.sum())))


def skillcorner():
    from bench import lab_skillcorner
    from bench import skillcorner as sc
    built = v15.load_branch_b(WHICH)
    t = built["trial"]
    folder = os.path.join(lab.RESULTS, "search_b")
    names = [f for f in sorted(os.listdir(folder)) if f.startswith("skillcorner_") and f.endswith(".csv")
             and str(t["trial"]) in f[len("skillcorner_"):-len(".csv")].split("_")]
    ref = pd.read_csv(os.path.join(folder, names[0]), dtype={"match": str}) if names else None
    rows = []
    for m in sc.match_ids():
        gt = sc.load(m)["labels"]
        match, X, Xs = lab_carried.sc_prepared(m)
        features = (list(built["names"]), X, Xs)
        letters = v15.b_chain_from(match, v15.b_start_proba(match, built, WHICH, features=features), built, WHICH)
        f1 = lab_skillcorner.score(gt, letters)["spell_f1"]
        lab_f1 = float(ref[(ref.match == m) & (ref.trial == t["trial"])].spell_f1.iloc[0]) if ref is not None else np.nan
        rows.append({"match": m, "branch": v13.branch(match.track, match.pred, match.halves, match.pitch, WHICH),
                     "v15_b": f1, "search_b": lab_f1})
        print(rows[-1], flush=True)
        del match, X, Xs
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(lab.RESULTS, "check_v15_skillcorner.csv"), index=False, float_format="%.5f")
    print("v15's B {:.4f} (search_b {:.4f}, largest difference {:.1e}); branch B on {} of 20".format(
        df.v15_b.mean(), df.search_b.mean(), float((df.v15_b - df.search_b).abs().max()), int((df.branch == "B").sum())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("port", "skillcorner"))
    args = parser.parse_args()
    return port() if args.step == "port" else skillcorner()


if __name__ == "__main__":
    main()
