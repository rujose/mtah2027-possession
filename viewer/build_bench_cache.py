"""Viewer caches for the bench matches (other matches: true labels, no video).

    python -m viewer.build_bench_cache      # after `python -m bench.eval_pathcrf`; ~2 min

One folder per match under viewer/cache/bench/<match>/, the same files as the BAR-ATH cache minus
the homography (there is no video), plus the clean tracking next to the damaged one:

    meta.json                  frames, halves, pitch, slots and teams (damaged and clean), lanes
    players.bin, ball.bin      the damaged tracking, as our models see it (41 %, as on the bench)
    players_clean.bin, ...     the tracking as published
    labels.bin                 the match's own truth, the floor, v1, v3, v4 (= v5 here), v6, v7
    clues.bin                  v1's clues on the damaged tracking, and v4/v6/v7's learned signals

v4, v6 and v7 come from bench/eval_pathcrf.py's out-of-sample run (each match predicted by models
trained without it); v1, v3 and the floor need no training and are computed here. Derived from the
bench data, so gitignored like the rest of viewer/cache.
"""

import json
import os

import numpy as np

from bench import degrade, labels, sets
from bench.run import baseline, v1_without_tags
from bench.train_restart import prepare_level
from bench.tune_decoder import decode
from possession import v1, v3
from possession.data import AWAY, HOME, ROOT

# Params
OUT = os.path.join(ROOT, "viewer", "cache", "bench")
PREDICTIONS = os.path.join(ROOT, "data", "external", "bench_predictions")
LEVEL = 41
SCALE = 10000
MISSING = -32768
LETTERS = np.array(["H", "A", "D", "X"])     # scorer codes -> letters
CODES = {"X": 0, "H": 1, "A": 2, "D": 3}     # letters -> the viewer's label codes


def encode(metres, pitch):
    out = np.full(metres.shape, MISSING, dtype=np.int16)
    normalised = metres / np.array(pitch, dtype=np.float32)
    seen = ~np.isnan(normalised)
    out[seen] = np.round(normalised[seen] * SCALE).astype(np.int16)
    return out


def percent_lane(p):
    return np.where(np.isnan(p), 0, 1 + np.round(np.nan_to_num(p) * 100)).astype(np.uint8)


def build(match, data, track, predicted):
    folder = os.path.join(OUT, match)
    os.makedirs(folder, exist_ok=True)
    halves, pitch = data["halves"], data["pitch"]
    truth = LETTERS[sets.truth(match, data)]

    clues = v1.compute_clues(track, [], halves, pitch)
    v3_letters = decode(v1.build_costs(clues), halves, (v3.SWITCH_TEAM, v3.SWITCH_DEAD, v3.MIN_POSSESSION, v3.MIN_DEAD))
    label_lanes = [("truth (bench)", truth),
                   ("baseline_smooth50", baseline(50)(track, halves, pitch)),
                   ("v1", v1_without_tags(track, halves, pitch)), ("v3", v3_letters),
                   ("v4", predicted["v4"]), ("v6", predicted["v6"]), ("v7", predicted["v7"])]
    labels_bin = np.stack([np.vectorize(CODES.get)(np.asarray(l).astype(str)) for _, l in label_lanes]).astype(np.uint8)

    on_ball = np.zeros(len(clues["out"]), np.uint8)
    on_ball[clues["on_ball"][:, HOME] > 0] = 1
    on_ball[clues["on_ball"][:, AWAY] > 0] = 2
    clue_lanes = [("ball visible", (~np.isnan(track["ball"][:, 0])).astype(np.uint8)),
                  ("A: on the ball", on_ball), ("B: ball out", clues["out"].astype(np.uint8)),
                  ("B: everyone stopped", clues["stopped"].astype(np.uint8)),
                  ("B: players active", clues["active"].astype(np.uint8)),
                  ("v4: P(dead)", np.round(predicted["p_dead"] * 100).astype(np.uint8)),
                  ("v6: restart clue", percent_lane(predicted["restart_clue"])),
                  ("v7: PathCRF", percent_lane(predicted["pathcrf_clue"]))]
    clues_bin = np.stack([lane for _, lane in clue_lanes])

    encode(track["players"], pitch).tofile(os.path.join(folder, "players.bin"))
    encode(track["ball"], pitch).tofile(os.path.join(folder, "ball.bin"))
    encode(data["players"], pitch).tofile(os.path.join(folder, "players_clean.bin"))
    encode(data["ball"], pitch).tofile(os.path.join(folder, "ball_clean.bin"))
    labels_bin.tofile(os.path.join(folder, "labels.bin"))
    clues_bin.tofile(os.path.join(folder, "clues.bin"))
    meta = {"frames": len(truth), "fps": 25, "pitch": list(pitch), "scale": SCALE, "missing": MISSING,
            "slots": int(track["players"].shape[1]), "team": track["team"].tolist(), "directions": {},
            "halves": [list(h) for h in halves],
            "kickoff": {"1": halves[0][1], "2": halves[1][1]},
            "labelLanes": [n for n, _ in label_lanes], "clueLanes": [n for n, _ in clue_lanes],
            "tags": [], "video": False, "title": data["title"],
            "clean": {"slots": int(data["players"].shape[1]), "team": data["team"].tolist()}}
    with open(os.path.join(folder, "meta.json"), "w") as handle:
        json.dump(meta, handle)


def main():
    stats = degrade.load_stats()
    matches = sets.matches("all")
    data = {m: sets.load(m) for m in matches}
    prepared, scale = prepare_level(LEVEL, matches, data, stats)
    index = []
    for m in matches:
        path = os.path.join(PREDICTIONS, "{}.npz".format(m))
        if not os.path.exists(path):
            raise FileNotFoundError("{}: run `python -m bench.eval_pathcrf` first".format(path))
        with np.load(path) as predicted:
            predicted = {k: predicted[k] for k in predicted.files}
        track = degrade.tracking(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        build(m, data[m], track, predicted)
        index.append({"id": m, "title": data[m]["title"]})
        print("{}: cache written".format(m), flush=True)
    with open(os.path.join(OUT, "index.json"), "w") as handle:
        json.dump(index, handle)
    print("wrote {} bench match caches to {}".format(len(index), os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
