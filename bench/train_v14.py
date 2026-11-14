"""Train v14's models on every bench match of `all` (docs/models/v14-search.md).

    OMP_NUM_THREADS=2 python -m bench.train_v14     # ~6 min; writes possession/models/v14_all.joblib

As bench/train_v13.py, with v14's start (possession/v14.py: the GRU weighted 0.8, no snap or move):

1. The start's out-of-fold letters on every bench match: the lab's held-out GRU and robust P mixed 0.8 / 0.2
   (bench/search_metrica.held_out_start), decoded with `average`'s setting chosen on all the set's matches, no snap
   or move.
2. Each of v12's post stages out of fold (lab_mix.add_stage), with the setting chosen on all the matches
   (train_v12.chosen, as for v13); the chain's out-of-fold spellF1 is printed beside v13's. Then each post stage
   fitted on every match's letters before it.
3. The GRU and robust: v13's deployment models, unchanged (possession/models/v13_all.joblib).
"""

import os
import time

import joblib
import numpy as np

from bench import lab, lab_mix, search_metrica, train_v13
from possession import v13, v14

# Params
WHICH = "all"


def start_letters(match, setting):
    p = search_metrica.held_out_start(match, WHICH, v14.W_GRU)
    letters = v13.L.decode_cost(match, -np.log(p), **setting)
    if v14.SNAP:
        letters = v13.lab_robust.finish(match, letters, WHICH)
    return np.asarray(letters).astype("<U1")


def out_of_fold(matches, settings):
    """{stage: {match: letters}} after the start and each post stage, each match's from models trained without it.
    settings: {stage: {match: setting}}."""
    letters = {m.id: start_letters(m, settings["start"][m.id]) for m in matches}
    steps = {"start": letters}
    for stage in v13.CHAIN[WHICH]:
        letters = lab_mix.add_stage(stage, letters, matches, settings)
        steps[stage] = letters
        print("  {} out of fold: done".format(stage), flush=True)
    return steps


def main():
    started = time.time()
    lab_mix.cache_short_evidence()
    matches = list(lab.each(WHICH))
    setting = train_v13.chosen(WHICH)
    per_match = {stage: {m.id: s for m in matches} for stage, s in setting.items()}
    steps = out_of_fold(matches, per_match)
    for stage, letters in steps.items():
        print("the chain out of fold to {}, with the chosen settings: spellF1 {:.4f}".format(
            stage, np.mean([lab.score(m, letters[m.id])["spell_f1"] for m in matches])), flush=True)
    final = steps[v13.CHAIN[WHICH][-1]]
    for m in matches:
        print("  {}: {:.4f}".format(m.id, lab.score(m, final[m.id])["spell_f1"]), flush=True)
    post = train_v13.fit_post(matches, WHICH, steps, setting)
    base = v13.load_models(WHICH)
    models = {"robust": base["robust"], "gru": base["gru"], **post, "settings": setting, "chain": v13.CHAIN[WHICH],
              "which": WHICH, "w_gru": v14.W_GRU, "snap": v14.SNAP}
    path = v14.MODELS.format(WHICH)
    joblib.dump(models, path, compress=3)
    print("wrote {} ({:.1f} MB) in {:.0f} s".format(os.path.relpath(path, lab.ROOT), os.path.getsize(path) / 1e6, time.time() - started))


if __name__ == "__main__":
    main()
