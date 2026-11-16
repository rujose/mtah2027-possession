"""Viewer caches for the bench matches (other matches: true labels, no video).

    python -m viewer.build_bench_cache      # after `python -m bench.eval_pathcrf`; ~2 min

One folder per match under viewer/cache/bench/<match>/, the same files as the BAR-ATH cache minus
the homography (there is no video), plus the clean tracking next to the damaged one:

    meta.json                  frames, halves, pitch, slots and teams (damaged and clean), lanes
    players.bin, ball.bin      the damaged tracking, as our models see it (41 %, as on the bench)
    players_clean.bin, ...     the tracking as published
    labels.bin                 the match's own truth, the floor, v1, v3, v4 (= v5 here), v6, v7 (= v8),
                               v9 (= v10), v11, v12 after each of its stages, and v12's errors: the
                               true possessions it misses, the predicted ones with no true match
    clues.bin                  v1's clues on the damaged tracking, v4/v6/v7's learned signals, v9's
                               seen passes, v11's touches, and v12's robust model (P(dead), P(home))

The latest versions (added 2026-11-15), when `python -m bench.viewer_letters` has saved them: v15 as it runs (the
stoppage switch picks branch A or B per match), its branch A (= v14's), its branch B, the night of 2026-11-14's
candidate `netmet5` (branch B with Metrica's games in its network's training; not adopted) and v15's errors. Each
match also gets a second entry, `<match>-stressed`: the same tracking with the fully stressed ball (real SkillCorner
stoppage balls pasted into the dead time, bench/lab_carried.py), the bench variant where v15 runs branch B, as on our
match. That entry has only the lanes computed on its own ball.

v4, v6 and v7 come from bench/eval_pathcrf.py's out-of-sample run (each match predicted by models
trained without it); v9 and v11 are that v7 with the restart snap and then the touch move, as
bench/touch_timing.py scores them (the "all" set's settings). v1, v3 and the floor need no training
and are computed here. On the bench the keeper repair never fires (v8 = v7) and there is no audio or
tags (v10 = v9). v12 is its chain's out-of-fold letters (`python -m bench.train_v12 --oof-only`: each
match's models trained without it), when saved. Derived from the bench data, so gitignored like the
rest of viewer/cache.
"""

import json
import os

import numpy as np

from bench import degrade, inside_v12, lab, lab_carried, lab_robust, labels, sets
from bench.run import baseline, v1_without_tags
from bench.train_restart import prepare_level
from bench.tune_decoder import decode
from possession import passes, touches, v1, v3, v9, v11
from possession.data import AWAY, HOME, ROOT

# Params
OUT = os.path.join(ROOT, "viewer", "cache", "bench")
PREDICTIONS = os.path.join(ROOT, "data", "external", "bench_predictions")
V12 = os.path.join(ROOT, "data", "external", "v12_oof")   # bench/train_v12.py --oof-only
LATEST = os.path.join(ROOT, "data", "external", "viewer_letters")   # bench/viewer_letters.py
LATEST_LANES = (("v15 · A (= v14)", "a"), ("v15 · B", "b"), ("netmet5 · B (candidate)", "netmet5"), ("v15", "v15"))
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


def marks(found, before, after):
    """1 / 2 from `before` frames before to `after` frames after each frame found for home / away:
    the frames the detector needed to see it (v9's passes, v11's touches)."""
    lane = np.zeros(len(found[HOME]), np.uint8)
    for code, side in ((1, HOME), (2, AWAY)):
        for k in np.flatnonzero(found[side]):
            lane[max(k - before, 0):k + after + 1] = code
    return lane


def v12_lanes(match, truth_codes, length):
    """v12's label lanes on a bench match, when `python -m bench.train_v12 --oof-only` has saved them:
    each stage's out-of-fold letters, then its errors at the metric's lowest bar (IoU 0.3)."""
    if not os.path.exists(os.path.join(V12, "{}.npy".format(match))):
        return []
    stages = (("v12: robust", "robust"), ("v12: + short", "short"), ("v12: + segments", "segments"), ("v12", "timing"))
    lanes = [(name, np.load(os.path.join(V12, stage, "{}.npy".format(match)))) for name, stage in stages]
    return lanes + error_lanes(truth_codes, lanes[-1][1], length, "errors:")


def error_lanes(truth_codes, letters, length, prefix):
    """A version's errors at the metric's lowest bar (IoU 0.3): the true possessions it misses, the predicted ones with
    no true match, each in its team's colour."""
    t, p, overlaps, _ = inside_v12.spell_table(truth_codes, letters)
    found, right = inside_v12.matched(overlaps, len(t), len(p), inside_v12.LOW)
    lanes = []
    for name, spells, ok in (("missed", t, found), ("spurious", p, right)):
        lane = np.full(length, "X", dtype="<U1")
        for (a, b, k), hit in zip(spells, ok):
            if not hit:
                lane[a:b] = "H" if k == 0 else "A"           # scorer codes: 0 H, 1 A
        lanes.append(("{} {}".format(prefix, name), lane))
    return lanes


def latest_lanes(match, variant, truth_codes, length):
    """The latest versions' letters (bench/viewer_letters.py), oldest first (the page shows the newest on top), then
    v15's errors. Empty when they haven't been saved."""
    lanes = []
    for name, key in LATEST_LANES:
        path = os.path.join(LATEST, variant, key, "{}.npy".format(match))
        if os.path.exists(path):
            lanes.append((name, np.load(path)))
    if lanes and lanes[-1][0] == "v15":
        lanes += error_lanes(truth_codes, lanes[-1][1], length, "errors: v15")
    return lanes


def v12_clues(match):
    """The robust model's held-out P(dead) (%) and P(home | in play) (as percent_lane), when saved."""
    try:
        p = lab_robust.held_out_proba(match)
    except FileNotFoundError:
        return []
    return [("v12: P(dead)", np.round(p[:, 2] * 100).astype(np.uint8)),
            ("v12: P(home)", percent_lane(p[:, 0] / (p[:, 0] + p[:, 1])))]


def build(match, data, track, predicted, variant="normal"):
    """One viewer entry. `variant` "stressed": the same match with the fully stressed ball, lanes computed on it only."""
    entry = match if variant == "normal" else "{}-{}".format(match, variant)
    folder = os.path.join(OUT, entry)
    os.makedirs(folder, exist_ok=True)
    halves, pitch = data["halves"], data["pitch"]
    truth = LETTERS[sets.truth(match, data)]

    clues = v1.compute_clues(track, [], halves, pitch)
    v3_letters = decode(v1.build_costs(clues), halves, (v3.SWITCH_TEAM, v3.SWITCH_DEAD, v3.MIN_POSSESSION, v3.MIN_DEAD))
    starts = v9.seen_passes(track)
    v9_letters, _ = v9.snap(predicted["v7"].astype(str), track, halves, v9.BACK, v9.AHEAD, starts)
    setting = v11.SETTINGS["all"]
    found = v11.seen_touches(track, setting["dv"])
    v11_letters, _ = v11.move(v9_letters, track, halves, setting["dv"], setting["back"], setting["ahead"], found)
    label_lanes = [("truth (bench)", truth),
                   ("baseline_smooth50", baseline(50)(track, halves, pitch)),
                   ("v1", v1_without_tags(track, halves, pitch)), ("v3", v3_letters)]
    if variant == "normal":            # v4-v12 were computed on the normal bench's ball only
        label_lanes += [("v4", predicted["v4"]), ("v6", predicted["v6"]), ("v7 = v8", predicted["v7"]),
                        ("v9 = v10", v9_letters), ("v11", v11_letters)]
        label_lanes += v12_lanes(match, sets.truth(match, data), len(truth))
    label_lanes += latest_lanes(match, variant, sets.truth(match, data), len(truth))
    labels_bin = np.stack([np.vectorize(CODES.get)(np.asarray(l).astype(str)) for _, l in label_lanes]).astype(np.uint8)

    on_ball = np.zeros(len(clues["out"]), np.uint8)
    on_ball[clues["on_ball"][:, HOME] > 0] = 1
    on_ball[clues["on_ball"][:, AWAY] > 0] = 2
    clue_lanes = [("ball visible", (~np.isnan(track["ball"][:, 0])).astype(np.uint8)),
                  ("A: on the ball", on_ball), ("B: ball out", clues["out"].astype(np.uint8)),
                  ("B: everyone stopped", clues["stopped"].astype(np.uint8)),
                  ("B: players active", clues["active"].astype(np.uint8))]
    if variant == "normal":
        clue_lanes += [("v4: P(dead)", np.round(predicted["p_dead"] * 100).astype(np.uint8)),
                       ("v6: restart clue", percent_lane(predicted["restart_clue"])),
                       ("v7: PathCRF", percent_lane(predicted["pathcrf_clue"]))]
    clue_lanes += [("v9: seen pass", marks(starts, 0, passes.LOOK - 1)),
                   ("v11: touch", marks(found, touches.LOOK, touches.LOOK))]
    if variant == "normal":
        clue_lanes += v12_clues(match)
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
            "tags": [], "video": False, "title": title(data, variant), "variant": variant,
            "clean": {"slots": int(data["players"].shape[1]), "team": data["team"].tolist()}}
    with open(os.path.join(folder, "meta.json"), "w") as handle:
        json.dump(meta, handle)


def title(data, variant):
    return data["title"] if variant == "normal" else "{} · stressed ball (v15 runs branch B)".format(data["title"])


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
        index.append({"id": m, "title": title(data[m], "normal")})
        stressed = lab_carried.stressed(m, "all", match=lab.load(m, "all")).track["ball"]
        build(m, data[m], {**track, "ball": stressed}, predicted, "stressed")
        index.append({"id": "{}-stressed".format(m), "title": title(data[m], "stressed")})
        print("{}: cache written (normal and stressed ball)".format(m), flush=True)
    with open(os.path.join(OUT, "index.json"), "w") as handle:
        json.dump(index, handle)
    print("wrote {} bench match caches to {}".format(len(index), os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
