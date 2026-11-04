"""Turn our match's data into compact binary files the viewer page can load in one go.

    python -m viewer.build_cache            # ~10 s; viewer/server.py runs it on first start
    python -m viewer.build_cache --models   # also a lane per CSV in evaluation/results/ (Jose's call)

Since the organisers' clarification of 2026-11-03 the viewer shows, for our match, only what we may
inspect to understand what counts as possession: the video, the tracking as delivered, Smart Tagging
and the public ground truth. None of our output is drawn over the video by default: checking our
output against it is not allowed (docs/limitations.md, section 0). With --models (decided by Jose on
2026-11-04, docs/decisions.md) the label lanes of every CSV in evaluation/results/ come back, to see
the versions run; nothing seen there may feed a model choice. The bench matches keep everything
(viewer/build_bench_cache.py): they are external data, with no video.

Everything is indexed by video frame. Files written to viewer/cache/ (gitignored: it is derived
from the gated data and must never be published):

    meta.json        frame count, fps, halves, pitch size, slot teams, lane names, tags
    players.bin      int16 [frames, slots, 2]  normalised x, y * 10000; -32768 = not seen
    ball.bin         int16 [frames, 2]         same encoding
    homography.bin   float32 [frames, 9]       pitch (0-1) -> pixel matrix, row by row
    labels.bin       uint8 [lanes, frames]     0 = X / none, 1 = H, 2 = A, 3 = D: the public truth
                                               (and with --models, each CSV in evaluation/results/)
    clues.bin        uint8 [clue lanes, frames] "ball visible": 1 where the tracking has the ball
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from possession.data import FPS, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking
from games import asset_paths

# Params
CACHE = os.path.join(ROOT, "viewer", "cache")
RESULTS = os.path.join(ROOT, "evaluation", "results")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SCALE = 10000          # normalised coordinate * SCALE fits int16 (the data spans about -0.12..1.49)
MISSING = -32768
CODES = {"X": 0, "H": 1, "A": 2, "D": 3}


def encode_positions(metres):
    """Metres -> normalised int16, MISSING where NaN."""
    normalised = metres / np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)
    out = np.full(normalised.shape, MISSING, dtype=np.int16)
    seen = ~np.isnan(normalised)
    out[seen] = np.round(normalised[seen] * SCALE).astype(np.int16)
    return out


def result_lanes(n_frames):
    """[(name, labels)] for each CSV in evaluation/results/: the baselines, then v1, v2, ... by number."""
    def order(path):
        name = os.path.splitext(os.path.basename(path))[0]
        return (1, int(name[1:]), name) if name[:1] == "v" and name[1:].isdigit() else (0, 0, name)
    lanes = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.csv")), key=order):
        labels = pd.read_csv(path).label.values.astype(str)
        assert len(labels) == n_frames, "{}: {} rows, the video has {} frames".format(path, len(labels), n_frames)
        lanes.append((os.path.splitext(os.path.basename(path))[0], labels))
    return lanes


def build(models=False):
    os.makedirs(CACHE, exist_ok=True)
    for stale in ("labels.bin", "clues.bin"):          # never leave an older build's model lanes behind
        if os.path.exists(os.path.join(CACHE, stale)):
            os.remove(os.path.join(CACHE, stale))
    tracking = load_tracking()
    tags = load_tags()
    n_frames = len(tracking["ball"])

    encode_positions(tracking["players"]).tofile(os.path.join(CACHE, "players.bin"))
    encode_positions(tracking["ball"]).tofile(os.path.join(CACHE, "ball.bin"))

    table = pd.read_csv(asset_paths()["homography"])
    matrices = table[["h{}{}".format(i, j) for i in range(3) for j in range(3)]].values.astype(np.float32)
    assert len(matrices) == n_frames
    matrices.tofile(os.path.join(CACHE, "homography.bin"))

    lanes = [("truth (first 5 min)", pd.read_csv(PUBLIC_GT).label.values.astype(str))]
    if models:
        lanes += result_lanes(n_frames)
    np.stack([np.vectorize(CODES.get)(labels) for _, labels in lanes]).astype(np.uint8).tofile(os.path.join(CACHE, "labels.bin"))
    (~np.isnan(tracking["ball"][:, 0])).astype(np.uint8)[None].tofile(os.path.join(CACHE, "clues.bin"))

    meta = {
        "frames": n_frames, "fps": FPS, "pitch": [PITCH_LENGTH, PITCH_WIDTH], "scale": SCALE,
        "missing": MISSING, "slots": len(tracking["team"]), "team": tracking["team"].tolist(),
        "directions": {str(team): direction for team, direction in tracking["directions"].items()},
        "halves": [[half, first, last] for half, first, last in graded_halves()],
        "labelLanes": [name for name, _ in lanes], "clueLanes": ["ball visible"],
        "tags": [{k: v for k, v in tag.items()} for tag in tags],
        "notes": False,                                   # no notes on our match (manually created data)
    }
    with open(os.path.join(CACHE, "meta.json"), "w") as handle:
        json.dump(meta, handle)
    if models:
        print("viewer cache: {} frames; the public truth, ball visibility and {} result lanes ({})".format(
            n_frames, len(lanes) - 1, ", ".join(name for name, _ in lanes[1:])))
    else:
        print("viewer cache: {} frames; the public truth and ball visibility only, no model output".format(n_frames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", action="store_true", help="add a label lane per CSV in evaluation/results/")
    build(parser.parse_args().models)
