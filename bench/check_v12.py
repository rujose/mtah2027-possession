"""v12's port against the lab: on bench matches, v12's own code gives the lab's letters.

    python -m bench.check_v12          # ~20 min; after `python -m bench.train_v12`

1. The port. For J03WR9 and metrica1: the lab's chain (round 4: robust -> short -> segments -> timing,
   each stage on its setting for that match, models trained without it), against v12.run_chain on
   the same match wrapped as v12 wraps ours, with the same models; robust's P recomputed from its
   features (v12's path) rather than read from the lab's cache. Frames that differ must be 0.
2. The deployment models (possession/models/v12_all.joblib) end to end on a bench match, twice.
3. v12.inputs on a bench match's damaged tracking: it runs, and its v7 costs decoded agree with the
   lab's v7 on most frames (not all: v12's inputs come from models trained on every bench match,
   the lab's out of fold).
"""

import numpy as np

from bench import chain, degrade, lab, lab_mix, lab_robust, sets, train_v12
from bench.train_restart import prepare_level
from bench.tune_decoder import decode
from possession import v12

# Params
MATCHES = ("J03WR9", "metrica1")
WHICH = "all"


def main():
    lab_mix.cache_short_evidence()
    matches = list(lab.each(WHICH))
    settings = lab_mix.fold_settings(WHICH)               # each stage's pick for each match, as the lab
    steps = train_v12.out_of_fold(matches, WHICH, settings)
    last = v12.CHAIN[WHICH][-1]
    for m in [x for x in matches if x.id in MATCHES]:
        others = [o for o in matches if o.id != m.id]
        own = {stage: settings[stage][m.id] for stage in ("robust",) + v12.CHAIN[WHICH]}
        models = train_v12.fit_chain(others, WHICH, steps, own)
        wrapped = v12.as_match(m.track, m.pred, m.halves, m.pitch, m.id)
        p = v12.robust_proba(wrapped, models["robust"], WHICH)
        p_lab = lab_robust.held_out_proba(m, WHICH)
        port = v12.run_chain(wrapped, models, WHICH, settings=own)
        ref = steps[last][m.id]
        print("{}: robust's P vs the lab's: max difference {:.1e}; letters differing from the lab's chain: {} of {} "
              "(spellF1 {:.4f}, lab {:.4f})".format(m.id, np.max(np.abs(p - p_lab)), int(np.sum(port != ref)), len(ref),
                                                     lab.score(m, port)["spell_f1"], lab.score(m, ref)["spell_f1"]), flush=True)

    m = next(x for x in matches if x.id == MATCHES[0])
    models = v12.load_models(WHICH)
    runs = [v12.run_chain(v12.as_match(m.track, m.pred, m.halves, m.pitch, m.id), models, WHICH) for _ in range(2)]
    print("deployment models on {}: same letters twice: {} (in sample, so its score means nothing)".format(
        m.id, np.array_equal(*runs)), flush=True)

    stats = degrade.load_stats()                          # the damaged tracking with players' speed (the
    data = {x: sets.load(x) for x in lab.matches(WHICH)}  # lab's cache keeps only what its stages read)
    prepared, scale = prepare_level(lab.LEVEL, list(data), data, stats)
    track = degrade.tracking(m.id, data[m.id], lab.LEVEL, prepared[m.id], scale[sets.source(m.id)], stats)
    del data, prepared
    fixed, pred = v12.inputs(track, m.halves, m.pitch, WHICH)
    v7_letters = decode(pred["v7_cost"].astype(np.float64), m.halves, chain.decoder(WHICH))
    graded = np.isin(m.truth, (0, 1, 2))
    print("v12.inputs on {}: v7 decoded from them agrees with the lab's v7 on {:.1%} of graded frames".format(
        m.id, np.mean(v7_letters[graded] == m.v7[graded])), flush=True)


if __name__ == "__main__":
    main()
