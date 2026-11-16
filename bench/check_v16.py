"""v16's branch B against the lab (docs/models/v16-netmet.md); never our match.

    python -m bench.check_v16 port              # ~4 min: the port with fold models on bench units, against the lab's netmet5
    python -m bench.check_v16 skillcorner       # ~17 min: the built file on SkillCorner's 20, against the lab's netmet5 run
    python -m bench.check_v16 switch            # ~1 min: the switch's check with netmet5's B (information; the night ran it)

port: on bench units (one match on one stressed variant), branch B through v16's code with the lab's fold models
(augment's fold, the recipe's four fold nets that never saw the match with their fold normaliser, trial 1015's stage
models trained without it on the recipe's tables), its features computed from the variant's tracking, in float32
(the lab's arithmetic) and float64 (as deployed), against search_b.letters_of with the lab mirror's saved P
(bench/lab_netmirror.py's `netmet/ens4`, the P the lab's netmet5 confirm scored).
skillcorner: possession/models/v16_b_all.joblib through v16's code on each SkillCorner match (lab_carried's cached
inputs), against lab_netmirror.skillcorner's scores for netmet5 (the four `all` nets, P averaged); and the branch
the switch takes there.
switch: bench/check_v15_switch.py's four parts with netmet5's B, read from the lab's confirm and SkillCorner files
(the lab's fold models, not the built file), compared with the night's report.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from bench import lab, lab_branch_b, lab_carried, lab_gate, lab_netmirror, lab_temporal, search_b, train_v16
from bench.run import score as bench_score
from possession import v13, v16

# Params
WHICH = "all"
UNITS = (("J03WR9", "stressed"), ("metrica1", "p50"), ("J03WOH", "p25"))     # check_v15's units
TOLERANCE = 5e-6         # the lab's CSVs keep 5 decimals
OUT = os.path.join(lab.RESULTS, "check_v16")


def recipe_of(built):
    r = train_v16.RECIPES[built["recipe"]]
    if tuple(built["seeds"]) != tuple(r["seeds"]):
        raise ValueError("the file's seeds {} are not the recipe's {}".format(built["seeds"], r["seeds"]))
    return r


def lab_results(r):
    return lab_netmirror.results_of(*r["mirror"])


def fold_bundle(m, t, built, r):
    """The built bundle with the lab's fold models for match m in place of the deployment ones."""
    held = m if m in lab_branch_b.held_list()[:-1] else lab_branch_b.held_list()[-1]      # augment's fold, as check_v15
    folds = joblib.load(os.path.join(lab_gate.folder(WHICH), "folds_{}.joblib".format(held)))
    torch = lab_temporal.torch_ready()
    norm = lab_temporal.normaliser(tuple(r["fold_norm"](m)), WHICH)
    grus = [v13.gru_bundle(torch.load(r["fold_net"](m, s)), norm, built["names"]) for s in built["seeds"]]
    return {**built, "augment": folds["augment"], "grus": grus, **search_b.trial_models(t, without=m)}


def start_and_gru(match, bundle, dtype, features):
    """v16.b_start_proba's P, and the averaged GRU P it mixed (recorded on the way, so the GRUs run once)."""
    seen, original = {}, v16.gru_proba

    def recording(*args, **kwargs):
        seen["p"] = original(*args, **kwargs)
        return seen["p"]
    v16.gru_proba = recording
    try:
        p = v16.b_start_proba(match, bundle, WHICH, dtype, features)
    finally:
        v16.gru_proba = original
    return p, seen["p"]


def port():
    built = v16.load_branch_b(WHICH)
    r = recipe_of(built)
    lab_netmirror.use_named(*r["mirror"])                   # search_b and lab_branch_b read the recipe's P and tables
    t = built["trial"]
    confirm = pd.read_csv(os.path.join(lab_results(r), "confirm_{}.csv".format(t["trial"])))
    rows = []
    for m, variant in UNITS:
        match = lab_gate.variant_match(m, WHICH, variant)
        bundle = fold_bundle(m, t, built, r)
        ref = search_b.letters_of(match, variant, {**t, "post": "own"}, {"own": {k: bundle[k] for k in bundle["stages"]}})
        p_lab = search_b.start_p(m, variant, t["w_gru"], t["f_dead"])
        gru_lab = lab_branch_b.gru_b(variant, m)
        features = v13.blanked_features(match, WHICH)
        lab_f1 = bench_score(match.truth, ref)["spell_f1"]
        csv_f1 = float(confirm[(confirm.trial == t["trial"]) & (confirm.match == m) & (confirm.variant == variant)].spell_f1.iloc[0])
        for dtype in (np.float32, np.float64):
            p, g = start_and_gru(match, bundle, dtype, features)
            letters = v16.b_chain_from(match, p, bundle, WHICH)
            g32 = np.abs(g.astype(np.float32).astype(np.float64) - gru_lab)       # the lab stored its mean as float32
            rows.append({"match": m, "variant": variant, "dtype": np.dtype(dtype).name,
                         "gru_p_max_abs_diff": float(g32.max()), "gru_p_frames_differing": int(np.any(g32 > 0, axis=1).sum()),
                         "p_max_abs_diff": float(np.abs(p - p_lab).max()), "frames_differing": int(np.sum(letters != ref)),
                         "spell_f1": bench_score(match.truth, letters)["spell_f1"], "lab_spell_f1": lab_f1,
                         "confirm_csv": csv_f1})
            print(rows[-1], flush=True)
        del match, features
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "port.csv"), index=False, float_format="%.6f")
    for dtype, d in df.groupby("dtype", sort=False):
        print("{}: frames differing in all {}; largest |P - lab's P| {:.1e}; the lab's letters score as its confirm CSV "
              "on every unit: {}".format(dtype, int(d.frames_differing.sum()), d.p_max_abs_diff.max(),
                                         bool((d.lab_spell_f1 - d.confirm_csv).abs().max() <= TOLERANCE)))


def skillcorner(dtype=None):
    from bench import lab_skillcorner
    from bench import skillcorner as sc
    built = v16.load_branch_b(WHICH)
    r = recipe_of(built)
    t = built["trial"]
    ref = pd.read_csv(os.path.join(lab_results(r), "skillcorner_{}.csv".format(t["trial"])), dtype={"match": str})
    ref = ref[ref.trial == t["trial"]].set_index("match").spell_f1
    name = "float32" if dtype == np.float32 else "float64"
    rows = []
    for m in sc.match_ids():
        gt = sc.load(m)["labels"]
        match, X, Xs = lab_carried.sc_prepared(m)
        features = (list(built["names"]), X, Xs)
        letters = v16.b_chain_from(match, v16.b_start_proba(match, built, WHICH, dtype, features=features), built, WHICH)
        f1 = lab_skillcorner.score(gt, letters)["spell_f1"]
        rows.append({"match": m, "branch": v13.branch(match.track, match.pred, match.halves, match.pitch, WHICH),
                     "v16_b": f1, "lab": float(ref.get(m, np.nan))})
        print(rows[-1], flush=True)
        del match, X, Xs
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "skillcorner_{}.csv".format(name)), index=False, float_format="%.6f")
    diff = (df.v16_b - df.lab).abs()
    print("GRU in {}: v16's B {:.4f} (the lab's netmet5 {:.4f}); largest per-match difference {:.1e}, {} of 20 within {}; "
          "branch B on {} of 20".format(name, df.v16_b.mean(), df.lab.mean(), float(diff.max()),
                                        int((diff <= TOLERANCE).sum()), TOLERANCE, int((df.branch == "B").sum())))


def switch():
    """check_v15_switch with the recipe's B (the lab's confirm and SkillCorner files), into bench/results/check_v16/."""
    from bench import check_v15_switch as CS
    r = train_v16.RECIPES[train_v16.RECIPE]
    trial = train_v16.TRIAL
    CS.SEARCH = lab_results(r)
    CS.RESULTS = os.path.join(OUT, "switch")
    argv, sys.argv = sys.argv, ["check_v15_switch", "--trial", str(trial)]
    try:
        CS.main()
    finally:
        sys.argv = argv
    with open(os.path.join(CS.RESULTS, "report_{}.json".format(trial))) as handle:
        got = json.load(handle)
    with open(os.path.join(lab_results(r), "switch", "report_{}.json".format(trial))) as handle:
        night = json.load(handle)
    print("the same as the night's report ({}): {}".format(
        os.path.relpath(os.path.join(lab_results(r), "switch"), lab.ROOT), got == night))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("port", "skillcorner", "switch"))
    parser.add_argument("--float32", action="store_true", help="skillcorner: the GRUs in float32 (the lab's arithmetic)")
    args = parser.parse_args()
    if args.step == "port":
        return port()
    if args.step == "switch":
        return switch()
    return skillcorner(np.float32 if args.float32 else None)


if __name__ == "__main__":
    main()
