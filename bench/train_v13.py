"""Train v13's models on every bench match of a set (docs/models/v13-temporal.md).

    OMP_NUM_THREADS=2 python -m bench.train_v13 [--set all|idsse]   # ~6 min, 1.2 GB; writes possession/models/v13_<set>.joblib
    python -m bench.train_v13 --switch                               # branch B: possession/models/v13_augment_all.joblib

As bench/train_v12.py, with the lab's temporal `average` start (bench/lab_temporal.py) in place of robust:

1. The start's out-of-fold letters on every bench match: lab_temporal's held-out P of `average` (the
   GRU trained without the match, averaged with robust's held-out P), decoded with `average`'s setting
   chosen on all the set's matches, then v9's snap and v11's move.
2. Each post stage of v12's chain out of fold (lab_mix.add_stage), on what the stages before it wrote,
   with the setting chosen on all the set's matches (train_v12.chosen, as lab_temporal.chains judged
   it); the chain's out-of-fold spellF1 is printed. Then each post stage fitted on every match's
   letters before it (train_v12.fit_stage).
3. The deployment GRU: lab_temporal.fit on the set's IDSSE matches, in lab.matches order (seed 0,
   deterministic). On `all` it is the lab's metrica1 fold (trained on the same 7 matches), and its
   weights are compared with the lab's net_metrica1.pt.
4. Robust: v12's deployment classifier (possession/models/v12_<set>.joblib), the IDSSE matches' model
   that the lab averaged on Metrica's folds.
"""

import argparse
import json
import os
import time

import joblib
import numpy as np

from bench import chain, lab, lab_mix, lab_robust, lab_temporal, sets, train_v12
from possession import v12, v13


def start_setting(which):
    """`average`'s decoder setting, chosen on all the set's matches (bench/results/lab_temporal_average*.json)."""
    with open(os.path.join(lab.RESULTS, "lab_temporal_average{}.json".format(chain.suffix(which)))) as handle:
        report = json.load(handle)
    return {k: lab_mix.literal(v) for k, v in report["chosen"].items()}


def start_folds(which):
    """{match: setting}: `average`'s pick for each match, chosen without it (for bench/check_v13.py)."""
    with open(os.path.join(lab.RESULTS, "lab_temporal_average{}.json".format(chain.suffix(which)))) as handle:
        return {f["match"]: {k: lab_mix.literal(v) for k, v in f["picked"].items()} for f in json.load(handle)["folds"]}


def chosen(which):
    """{stage: setting}: the start's and each of v12's post stages' setting chosen on all the set's matches."""
    post = train_v12.chosen(which)
    return {"start": start_setting(which), **{stage: post[stage] for stage in v13.CHAIN[which]}}


def start_letters(match, which, setting):
    """`average`'s held-out P decoded, then v9's snap and v11's move (lab_temporal.letters_of)."""
    return lab_temporal.letters_of(match, which, "average", setting)


def out_of_fold(matches, which, settings):
    """{stage: {match: letters}} after the start and after each post stage, each match's from models
    trained without it. settings: {stage: {match: setting}}."""
    letters = {m.id: start_letters(m, which, settings["start"][m.id]) for m in matches}
    steps = {"start": letters}
    for stage in v13.CHAIN[which]:
        letters = lab_mix.add_stage(stage, letters, matches, settings)
        steps[stage] = letters
        print("  {} out of fold: done".format(stage), flush=True)
    return steps


def fit_post(matches, which, steps, settings):
    """{stage: model}: each post stage fitted on `matches`, on the letters of the stage before it."""
    models, before = {}, "start"
    for stage in v13.CHAIN[which]:
        models[stage] = train_v12.fit_stage(stage, matches, steps[before], settings[stage])
        before = stage
    return models


def gru_matches(which):
    """The set's IDSSE matches, in lab.matches order (as the lab's metrica1 fold)."""
    return tuple(m for m in lab.matches(which) if sets.source(m) in lab_temporal.L.TRAIN_SOURCES)


def fit_gru(which):
    """The deployment network, as a v13 bundle."""
    used = gru_matches(which)
    print("GRU: training on {}".format(" ".join(used)), flush=True)
    model = lab_temporal.fit(used, which)
    names = lab_robust.cached(used[0], which)["names"]
    return v13.gru_bundle(model["net"].state_dict(), model["norm"], names, trained_on=list(used),
                          history=model["history"], seconds=model["seconds"], which=which)


def compare_with_lab(gru, which):
    """The deployment network's weights against the lab's fold trained on the same matches (net_metrica1.pt on
    `all`; the IDSSE set has no such fold: each of its folds left one IDSSE match out)."""
    path = os.path.join(lab_temporal.folder(which), "net_metrica1.pt")
    if which != "all" or not os.path.exists(path):
        print("GRU: no lab fold trained on the same matches to compare with", flush=True)
        return None
    torch = lab_temporal.torch_ready()
    ref = {k: v.numpy() for k, v in torch.load(path).items()}
    same_keys = list(ref) == list(gru["state"])
    identical = same_keys and all(np.array_equal(ref[k], gru["state"][k]) for k in ref)
    worst = max(float(np.max(np.abs(ref[k] - gru["state"][k]))) for k in ref) if same_keys else float("nan")
    print("GRU against the lab's {}: {} tensors, bit-identical: {} (largest difference {:.1e})".format(
        os.path.relpath(path, lab.ROOT), len(ref), identical, worst), flush=True)
    return {"identical": identical, "largest_difference": worst}


def branch_b(which):
    """Branch B of v13's stoppage switch (docs/lab/gate.md), as one committed file: `augment` (robust's classifier
    retrained on the set's IDSSE matches as they are and with SkillCorner-like balls in their stoppages,
    lab_carried.deployment_augment) and v12's post stages refitted on its out-of-fold letters
    (lab_carried.deploy_post), with v12's deployment settings, as lab_carried judged it on SkillCorner. The switch's
    signal and threshold are bench/lab_gate.py's; the threshold is saved too and checked on load."""
    from bench import lab_carried, lab_gate
    if which not in v13.SWITCH_SETS:
        raise SystemExit("branch B is built only for the bench set(s) {}".format(", ".join(v13.SWITCH_SETS)))
    robust = v12.load_models(which)
    augment = lab_carried.deployment_augment()
    post = lab_carried.deploy_post("augment", which)
    names = robust["robust"]["names"]
    if augment.n_features_in_ != len(names) or list(robust["robust"]["cols"]) != list(range(len(names))):
        raise ValueError("augment's columns are not robust's 97 cached features")
    settings = {"start": robust["settings"]["robust"], **{stage: robust["settings"][stage] for stage in v13.CHAIN[which]}}
    bundle = {"augment": augment, "names": names, **{stage: post[stage] for stage in v13.CHAIN[which]},
              "settings": settings, "chain": v13.CHAIN[which], "which": which, "threshold": lab_gate.THRESHOLD}
    path = v13.AUGMENT.format(which)
    joblib.dump(bundle, path, compress=3)
    print("wrote {} ({:.1f} MB); threshold {}".format(os.path.relpath(path, lab.ROOT), os.path.getsize(path) / 1e6,
                                                     lab_gate.THRESHOLD))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--switch", action="store_true", help="only branch B's models (possession/models/v13_augment_<set>.joblib)")
    args = parser.parse_args()
    which = args.which
    if args.switch:
        return branch_b(which)
    started = time.time()
    lab_mix.cache_short_evidence()
    matches = list(lab.each(which))
    setting = chosen(which)
    print("settings: {}".format(setting), flush=True)
    per_match = {stage: {m.id: s for m in matches} for stage, s in setting.items()}
    steps = out_of_fold(matches, which, per_match)
    v12_oof = train_v12.OOF + chain.suffix(which)
    for stage, letters in steps.items():
        mean = np.mean([lab.score(m, letters[m.id])["spell_f1"] for m in matches])
        line = "the chain out of fold to {}, with the chosen settings: spellF1 {:.4f}".format(stage, mean)
        reference = os.path.join(v12_oof, "robust" if stage == "start" else stage)
        if os.path.isdir(reference):    # v12's own out-of-fold letters, from `python -m bench.train_v12`
            line += " (v12 {:.4f})".format(np.mean([lab.score(m, np.load(os.path.join(reference, m.id + ".npy")))["spell_f1"]
                                                    for m in matches]))
        print(line, flush=True)
    final = steps[v13.CHAIN[which][-1]]
    for m in matches:
        print("  {}: {:.4f}".format(m.id, lab.score(m, final[m.id])["spell_f1"]), flush=True)
    post = fit_post(matches, which, steps, setting)
    print("post stages fitted on all {} matches ({:.0f} s so far)".format(len(matches), time.time() - started), flush=True)
    del matches
    gru = fit_gru(which)
    compare_with_lab(gru, which)
    robust = v12.load_models(which)["robust"]
    models = {"robust": robust, "gru": gru, **post, "settings": setting, "chain": v13.CHAIN[which], "which": which}
    path = v13.MODELS.format(which)
    joblib.dump(models, path, compress=3)
    print("wrote {} ({:.1f} MB) in {:.0f} s".format(os.path.relpath(path, lab.ROOT), os.path.getsize(path) / 1e6,
                                                     time.time() - started))


if __name__ == "__main__":
    main()
