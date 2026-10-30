"""Turn the match data into compact binary files the viewer page can load in one go.

    python -m viewer.build_cache        # ~10 s; viewer/server.py runs it on first start

Everything is indexed by video frame. Files written to viewer/cache/ (gitignored: it is derived
from the gated data and must never be published):

    meta.json        frame count, fps, halves, pitch size, slot teams, lane names, tags
    players.bin      int16 [frames, slots, 2]  normalised x, y * 10000; -32768 = not seen
    ball.bin         int16 [frames, 2]         same encoding
    homography.bin   float32 [frames, 9]       pitch (0-1) -> pixel matrix, row by row
    labels.bin       uint8 [lanes, frames]     0 = X / none, 1 = H, 2 = A, 3 = D
    clues.bin        uint8 [clue lanes, frames] 0 = off, 1 = on / Barcelona, 2 = Athletic

Label lanes are the public ground truth plus every CSV in evaluation/results/, so a new model
shows up in the viewer as soon as it writes its file. Clue lanes are model v1's.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from possession import v1
from possession.data import (FPS, HOME, AWAY, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves,
                             load_tags, load_tracking)
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


def label_lanes():
    """[(name, codes)] for the public truth and every submission in evaluation/results/."""
    lanes = [("truth (first 5 min)", pd.read_csv(PUBLIC_GT).label.values)]
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.csv"))):
        lanes.append((os.path.splitext(os.path.basename(path))[0], pd.read_csv(path).label.values))
    names = [name for name, _ in lanes]
    codes = np.stack([np.vectorize(CODES.get)(labels.astype(str)) for _, labels in lanes]).astype(np.uint8)
    return names, codes


def clue_lanes(tracking, tags):
    """v1's clues as small integer lanes, plus ball visibility."""
    clues = v1.compute_clues(tracking, tags, graded_halves())
    on_ball = np.zeros(len(clues["out"]), np.uint8)
    on_ball[clues["on_ball"][:, HOME] > 0] = 1
    on_ball[clues["on_ball"][:, AWAY] > 0] = 2
    tag_team = np.zeros_like(on_ball)
    tag_team[clues["tag_supports"][:, HOME]] = 1
    tag_team[clues["tag_supports"][:, AWAY]] = 2
    lanes = [
        ("ball visible", (~np.isnan(tracking["ball"][:, 0])).astype(np.uint8)),
        ("A: on the ball", on_ball),
        ("B: ball out", clues["out"].astype(np.uint8)),
        ("B: everyone stopped", clues["stopped"].astype(np.uint8)),
        ("B: players active", clues["active"].astype(np.uint8)),
        ("B: set-piece window", clues["set_piece"].astype(np.uint8)),
        ("C: tag says team", tag_team),
    ]
    return [name for name, _ in lanes], np.stack([lane for _, lane in lanes])


def build():
    os.makedirs(CACHE, exist_ok=True)
    tracking = load_tracking()
    tags = load_tags()
    n_frames = len(tracking["ball"])

    encode_positions(tracking["players"]).tofile(os.path.join(CACHE, "players.bin"))
    encode_positions(tracking["ball"]).tofile(os.path.join(CACHE, "ball.bin"))

    table = pd.read_csv(asset_paths()["homography"])
    matrices = table[["h{}{}".format(i, j) for i in range(3) for j in range(3)]].values.astype(np.float32)
    assert len(matrices) == n_frames
    matrices.tofile(os.path.join(CACHE, "homography.bin"))

    label_names, labels = label_lanes()
    labels.tofile(os.path.join(CACHE, "labels.bin"))
    clue_names, clues = clue_lanes(tracking, tags)
    clues.tofile(os.path.join(CACHE, "clues.bin"))

    meta = {
        "frames": n_frames, "fps": FPS, "pitch": [PITCH_LENGTH, PITCH_WIDTH], "scale": SCALE,
        "missing": MISSING, "slots": len(tracking["team"]), "team": tracking["team"].tolist(),
        "directions": {str(team): direction for team, direction in tracking["directions"].items()},
        "halves": [[half, first, last] for half, first, last in graded_halves()],
        "labelLanes": label_names, "clueLanes": clue_names,
        "tags": [{k: v for k, v in tag.items()} for tag in tags],
    }
    with open(os.path.join(CACHE, "meta.json"), "w") as handle:
        json.dump(meta, handle)
    print("viewer cache: {} frames, {} label lanes {}, {} clue lanes".format(
        n_frames, len(label_names), label_names, len(clue_names)))


if __name__ == "__main__":
    build()
