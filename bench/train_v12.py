"""Train v12's models on every bench match of a set (docs/models/v12-lab-chain.md).

    python -m bench.train_v12 [--set all|idsse]   # ~15 min; writes possession/models/v12_<set>.joblib
    python -m bench.train_v12 --oof-only          # only the chain's out-of-fold letters (for the viewer)

The robust classifier learns from the set's IDSSE matches (as in the lab). Each post stage learns
from what the stages before it write on the bench matches out of fold (each match's letters from
models trained without it, as bench/lab_mix.py does), so its training input looks like what it will
see on our match. Settings: each stage's choice on all the set's matches (bench/results/lab_*.json).
The chain's out-of-fold letters on each bench match are saved too (data/external/v12_oof[_idsse]/,
gitignored), for the viewer's bench matches (viewer/build_bench_cache.py).
"""

import argparse
import json
import os

import joblib
import numpy as np

from bench import chain, lab, lab_dead, lab_mix, lab_recall, lab_robust, lab_segments, lab_short, lab_timing
from possession import v12

# Params
OOF = os.path.join(lab.ROOT, "data", "external", "v12_oof")
REPORTS = {"robust": "lab_robust", "short": "lab_short", "segments": "lab_segments", "timing": "lab_timing",
           "recall": "lab_recall", "dead": "lab_dead"}


def chosen(which):
    """{stage: the setting chosen on all the set's matches}."""
    out = {}
    for stage, name in REPORTS.items():
        with open(os.path.join(lab.RESULTS, "{}{}.json".format(name, chain.suffix(which)))) as handle:
            out[stage] = {k: lab_mix.literal(v) for k, v in json.load(handle)["chosen"].items()}
    return out


def robust_letters(match, which, setting, p=None):
    """robust's held-out P (or `p`) decoded, then v9's snap and v11's move."""
    p = lab_robust.held_out_proba(match, which, setting["variant"]) if p is None else p
    return lab_robust.finish(match, lab_robust.L.decode_cost(match, -np.log(p), **setting), which)


def out_of_fold(matches, which, settings):
    """{stage: {match: letters}} after robust and after each post stage of the set's chain, each
    match's from models trained without it. settings: {stage: {match: setting}}."""
    letters = {m.id: robust_letters(m, which, settings["robust"][m.id]) for m in matches}
    steps = {"robust": letters}
    for stage in v12.CHAIN[which]:
        letters = lab_mix.add_stage(stage, letters, matches, settings)
        steps[stage] = letters
        print("  {} out of fold: done".format(stage), flush=True)
    return steps


def fit_stage(stage, matches, letters, setting):
    """One post stage's model on `matches`, given the letters it sees on each."""
    if stage == "short":
        return lab_short.fit(matches, letters=lambda o: letters[o.id], **setting)
    if stage == "segments":
        return lab_segments.fit(matches, lambda o: letters[o.id], features=setting["features"])
    if stage == "timing":
        return lab_timing.fit(matches, letters_of=lambda o: letters[o.id], **setting)
    if stage == "recall":
        return lab_recall.fit(matches, letters_of=lambda o: letters[o.id], **setting)
    if stage == "dead":
        return lab_dead.fit(matches, letters_of=lambda o: letters[o.id], features=setting["features"])
    raise ValueError(stage)


def fit_chain(matches, which, steps, settings):
    """{stage: model} trained on `matches`; each post stage on the letters of the stage before it."""
    robust = lab_robust.fit(matches, variant=settings["robust"]["variant"], which=which)
    robust.pop("_p", None)
    models = {"robust": robust}
    before = "robust"
    for stage in v12.CHAIN[which]:
        models[stage] = fit_stage(stage, matches, steps[before], settings[stage])
        before = stage
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--oof-only", action="store_true", help="save the out-of-fold letters, train nothing")
    args = parser.parse_args()
    which = args.which
    lab_mix.cache_short_evidence()
    matches = list(lab.each(which))
    setting = chosen(which)
    per_match = {stage: {m.id: s for m in matches} for stage, s in setting.items()}
    steps = out_of_fold(matches, which, per_match)
    final = steps[v12.CHAIN[which][-1]]
    print("the chain out of fold, with the chosen settings: spellF1 {:.4f} (v11 {:.4f})".format(
        np.mean([lab.score(m, final[m.id])["spell_f1"] for m in matches]),
        np.mean([b["spell_f1"] for b in lab.baseline(which).values()])), flush=True)
    folder = OOF + chain.suffix(which)
    os.makedirs(folder, exist_ok=True)
    for m in matches:
        np.save(os.path.join(folder, "{}.npy".format(m.id)), final[m.id])
    print("wrote the out-of-fold letters to {}".format(os.path.relpath(folder, lab.ROOT)), flush=True)
    if args.oof_only:
        return
    models = fit_chain(matches, which, steps, setting)
    models.update({"which": which, "chain": v12.CHAIN[which], "settings": setting})
    path = v12.MODELS.format(which)
    joblib.dump(models, path, compress=3)
    print("wrote {} ({:.1f} MB)".format(os.path.relpath(path, lab.ROOT), os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()
