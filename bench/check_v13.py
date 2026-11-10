"""v13's port against the lab: on bench matches, v13's own code gives the lab's letters.

    OMP_NUM_THREADS=2 python -m bench.check_v13 [--set all|idsse]   # ~15 min; after `python -m bench.train_v13`

1. The port, in float32 as the lab ran the GRU. The lab's `average`-start chain (lab_temporal.chains:
   each match's start on its fold's pick, each post stage out of fold on its setting for that match),
   recomputed here and checked against the scores the lab recorded. Then for J03WR9 and metrica1
   (`idsse`: J03WR9 and J03WMX),
   v13's own code (v13.start_proba's parts and v13.run_chain_from, the match wrapped as ours is, the
   features computed directly) with fold models: the lab's fold GRU (net_<match>.pt, with its fold's
   normaliser), robust and the post stages trained without the match. Frames that differ must be 0.
   For the record: the same without the float32 round trip in v13.average.
2. The same in float64, v13's arithmetic: the GRU's P against float32's, and the letters that change.
3. The deployment models (possession/models/v13_<set>.joblib): float32 against float64 on both matches
   (in sample, so no score here means anything); then v13.predict end to end on J03WR9's damaged
   tracking, twice: the same letters.
"""

import argparse
import json
import os

import numpy as np

from bench import chain, degrade, lab, lab_mix, lab_robust, lab_temporal, sets, train_v13
from bench.train_restart import prepare_level
from possession import v12, v13

# Params
MATCHES = {"all": ("J03WR9", "metrica1"), "idsse": ("J03WR9", "J03WMX")}   # one PathCRF-unseen, one not


def lab_reference(matches, which):
    """(the lab's average-start chain out of fold, {stage: {match: setting}}), as lab_temporal.chains made it."""
    settings = lab_mix.fold_settings(which)
    settings["start"] = train_v13.start_folds(which)
    return train_v13.out_of_fold(matches, which, settings), settings


def fold_gru(m, which):
    """The lab's GRU for this match's fold, as a v13 bundle: its saved weights and its fold's normaliser."""
    used = tuple(o for o in lab.matches(which) if o != m and sets.source(o) in lab_temporal.L.TRAIN_SOURCES)
    torch = lab_temporal.torch_ready()
    state = torch.load(os.path.join(lab_temporal.folder(which), "net_{}.pt".format(m)))
    names = lab_robust.cached(used[0], which)["names"]
    return v13.gru_bundle(state, lab_temporal.normaliser(used, which), names, trained_on=list(used))


def fold_models(m, matches, which, steps, own):
    """v13's models for one match's fold: none has seen it."""
    others = [o for o in matches if o.id != m.id]
    robust = lab_robust.fit(others, variant="blank", which=which)
    robust.pop("_p", None)
    return {"robust": robust, "gru": fold_gru(m.id, which), **train_v13.fit_post(others, which, steps, own)}


def inside(m):
    out = np.zeros(m.frames, bool)
    for _, first, last in m.halves:
        out[first:last + 1] = True
    return out


def float_gap(wrapped, m, models, which, settings, letters32):
    """The GRU in float32 and float64 on one match: its P (raw and rounded), the start's P and the letters."""
    features = v13.blanked_features(wrapped, which)
    raw32 = v13.gru_raw(wrapped, models["gru"], which, np.float32, features)
    raw64 = v13.gru_raw(wrapped, models["gru"], which, np.float64, features)
    p_robust = v12.robust_proba(wrapped, models["robust"], which)
    p32, p64 = (v13.average(v13.clipped(r), p_robust) for r in (raw32, raw64))
    letters32 = v13.run_chain_from(wrapped, p32, models, which, settings) if letters32 is None else letters32
    letters64 = v13.run_chain_from(wrapped, p64, models, which, settings)
    gap = np.abs(raw32 - raw64)[inside(m)]
    print("  float32 vs float64: GRU's P largest difference {:.1e} (median {:.0e}); rounded GRU P entries changed "
          "{} of {}; start's P entries changed {}; letters changed {} of {}".format(
              gap.max(), np.median(gap), int(np.sum(v13.clipped(raw32) != v13.clipped(raw64))), raw32.size,
              int(np.sum(p32 != p64)), int(np.sum(letters32 != letters64)), len(letters64)), flush=True)
    return letters64


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    which = parser.parse_args().which
    picked = MATCHES[which]
    lab_mix.cache_short_evidence()
    matches = list(lab.each(which))
    stages = v13.CHAIN[which]
    steps, settings = lab_reference(matches, which)
    with open(os.path.join(lab.RESULTS, "lab_temporal_chains{}.json".format(chain.suffix(which)))) as handle:
        recorded = json.load(handle)["scores"]["average"]
    mine = {m.id: lab.score(m, steps[stages[-1]][m.id])["spell_f1"] for m in matches}
    print("the lab's average-start chain recomputed: spellF1 {:.4f} (recorded {:.4f}); largest per-match difference "
          "{:.1e}".format(np.mean(list(mine.values())), np.mean(list(recorded.values())),
                          max(abs(mine[k] - recorded[k]) for k in recorded)), flush=True)

    print("\n1-2. The port with fold models, float32 (as the lab) then float64 (v13)", flush=True)
    for m in [x for x in matches if x.id in picked]:
        own = {stage: settings[stage][m.id] for stage in ("start",) + stages}
        models = fold_models(m, matches, which, steps, own)
        wrapped = v12.as_match(m.track, m.pred, m.halves, m.pitch, m.id)
        p_gru = v13.gru_proba(wrapped, models["gru"], which, np.float32)
        p_robust = v12.robust_proba(wrapped, models["robust"], which)
        p = v13.average(p_gru, p_robust)
        port = v13.run_chain_from(wrapped, p, models, which, settings=own)
        ref = steps[stages[-1]][m.id]
        p_lab = lab_temporal.held_out(m, which, "average")
        print("{}: GRU's P vs the lab's: largest difference {:.1e}; robust's {:.1e}; the start's {:.1e}; start letters "
              "differing {}; letters differing from the lab's chain: {} of {} (spellF1 {:.4f}, lab {:.4f})".format(
                  m.id, np.max(np.abs(p_gru - lab_temporal.held_out(m, which, "temporal"))),
                  np.max(np.abs(p_robust - lab_robust.held_out_proba(m, which, "blank"))), np.max(np.abs(p - p_lab)),
                  int(np.sum(np.asarray(lab_robust.finish(wrapped, lab_robust.L.decode_cost(wrapped, -np.log(p), **own["start"]),
                                                          which)) != steps["start"][m.id])),
                  int(np.sum(port != ref)), len(ref), lab.score(m, port)["spell_f1"], lab.score(m, ref)["spell_f1"]), flush=True)
        plain = v13.clipped((p_gru + p_robust) / 2)       # the mean without the float32 round trip
        plain_letters = v13.run_chain_from(wrapped, plain, models, which, settings=own)
        print("  for the record, the mean without the float32 round trip: start's P entries off the lab's {} of {}; "
              "letters differing {}".format(int(np.sum(np.abs(plain - p_lab) > 1e-6)), plain.size,
                                            int(np.sum(plain_letters != ref))), flush=True)
        letters64 = float_gap(wrapped, m, models, which, own, port)
        print("  float64 letters against the lab's chain: {} differ (spellF1 {:.4f})".format(
            int(np.sum(letters64 != ref)), lab.score(m, letters64)["spell_f1"]), flush=True)
        del models

    print("\n3. The deployment models ({})".format(os.path.relpath(v13.MODELS.format(which), lab.ROOT)), flush=True)
    models = v13.load_models(which)
    deployed = {}
    for m in [x for x in matches if x.id in picked]:
        wrapped = v12.as_match(m.track, m.pred, m.halves, m.pitch, m.id)
        print("{} (in sample):".format(m.id), flush=True)
        deployed[m.id] = float_gap(wrapped, m, models, which, None, None)
    m = next(x for x in matches if x.id == picked[0])
    del matches, steps
    stats = degrade.load_stats()                          # the damaged tracking with players' speed (the
    data = {x: sets.load(x) for x in lab.matches(which)}  # lab's cache keeps only what its stages read)
    prepared, scale = prepare_level(lab.LEVEL, list(data), data, stats)
    track = degrade.tracking(m.id, data[m.id], lab.LEVEL, prepared[m.id], scale[sets.source(m.id)], stats)
    del data, prepared
    runs = [v13.predict(track, None, m.halves, m.pitch, which)[0] for _ in range(2)]
    graded = np.isin(m.truth, (0, 1, 2))
    print("v13.predict end to end on {}'s damaged tracking: the same letters twice: {}; agrees with the deployment "
          "chain on the lab's inputs on {:.1%} of graded frames (v13's inputs come from models trained on every bench "
          "match, the lab's out of fold)".format(m.id, np.array_equal(*runs), np.mean(runs[0][graded] == deployed[m.id][graded])),
          flush=True)


if __name__ == "__main__":
    main()
