"""v14's port against the lab: on bench matches, v14's own code gives the lab's letters.

    OMP_NUM_THREADS=2 python -m bench.check_v14     # ~15 min; after `python -m bench.train_v14`

1. The lab's chain from v14's start (bench/search_metrica.py `confirm`: the start on `average`'s fold pick, no snap
   or move, each post stage out of fold on its setting for that match), recomputed here (it scored 0.7241).
2. For J03WR9 and metrica1, v14's own code (v14.mix of v13.gru_proba and v12.robust_proba, v14.run_chain_from, the
   match wrapped as ours is, the features computed directly) with fold models: the lab's fold GRU in float32 as the
   lab ran it, robust and the post stages trained without the match. Frames that differ must be 0. Then the same in
   float64 (v14's arithmetic).
3. The deployment models (possession/models/v14_all.joblib) end to end on J03WR9's damaged tracking, twice.
"""

import numpy as np

from bench import check_v13, degrade, lab, lab_mix, search_metrica, sets, train_v13, train_v14
from bench.train_restart import prepare_level
from possession import v12, v13, v14

# Params
WHICH = "all"
MATCHES = ("J03WR9", "metrica1")


def main():
    lab_mix.cache_short_evidence()
    matches = list(lab.each(WHICH))
    stages = v13.CHAIN[WHICH]
    settings = lab_mix.fold_settings(WHICH)
    settings["start"] = train_v13.start_folds(WHICH)
    steps = train_v14.out_of_fold(matches, settings)
    print("the lab's chain from v14's start, recomputed: spellF1 {:.4f} (search_metrica confirm: 0.7241)".format(
        np.mean([lab.score(m, steps[stages[-1]][m.id])["spell_f1"] for m in matches])), flush=True)

    print("\n1-2. The port with fold models, float32 (as the lab) then float64 (v14)", flush=True)
    for m in [x for x in matches if x.id in MATCHES]:
        own = {stage: settings[stage][m.id] for stage in ("start",) + stages}
        models = {**check_v13.fold_models(m, matches, WHICH, steps, own), "w_gru": v14.W_GRU, "snap": v14.SNAP}
        wrapped = v12.as_match(m.track, m.pred, m.halves, m.pitch, m.id)
        features = v13.blanked_features(wrapped, WHICH)
        p_robust = v12.robust_proba(wrapped, models["robust"], WHICH)
        p32 = v14.mix(v13.gru_proba(wrapped, models["gru"], WHICH, np.float32, features), p_robust)
        p64 = v14.mix(v13.gru_proba(wrapped, models["gru"], WHICH, np.float64, features), p_robust)
        p_lab = search_metrica.held_out_start(m, WHICH, v14.W_GRU)
        port32 = v14.run_chain_from(wrapped, p32, models, WHICH, settings=own)
        port64 = v14.run_chain_from(wrapped, p64, models, WHICH, settings=own)
        ref = steps[stages[-1]][m.id]
        print("{}: the start's P vs the lab's: largest difference {:.1e}; letters differing from the lab's chain: "
              "float32 {} of {}, float64 {} (spellF1 {:.4f}, lab {:.4f}); start P entries float32 vs float64: {}".format(
                  m.id, np.max(np.abs(p32 - p_lab)), int(np.sum(port32 != ref)), len(ref), int(np.sum(port64 != ref)),
                  lab.score(m, port64)["spell_f1"], lab.score(m, ref)["spell_f1"], int(np.sum(p32 != p64))), flush=True)
        del models

    m = next(x for x in matches if x.id == MATCHES[0])
    del matches, steps
    stats = degrade.load_stats()
    data = {x: sets.load(x) for x in lab.matches(WHICH)}
    prepared, scale = prepare_level(lab.LEVEL, list(data), data, stats)
    track = degrade.tracking(m.id, data[m.id], lab.LEVEL, prepared[m.id], scale[sets.source(m.id)], stats)
    del data, prepared
    runs = [v14.predict(track, None, m.halves, m.pitch, WHICH) for _ in range(2)]
    print("\n3. v14.predict end to end on {}'s damaged tracking: the same letters twice: {}; branch {}".format(
        m.id, np.array_equal(runs[0][0], runs[1][0]), runs[0][1]["branch"]), flush=True)


if __name__ == "__main__":
    main()
