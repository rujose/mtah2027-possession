"""PathCRF (third_party/pathcrf, MPL-2.0) on our kind of tracking: P(home has the ball) from the
players' movement only (docs/models/v7-pathcrf.md, pieces 1 and 2).

The model expects what DFL's clean tracking gives: 11 continuous tracks per team, a known
goalkeeper, no holes, on a 105 x 68 m pitch, at 5 fps in 10-second windows. This adapter turns our
anonymous slot pieces into that, window by window:
  - per team, the (up to 11) slots present longest in the window; holes inside a slot's window are
    filled by straight lines, and its ends held;
  - velocity, speed and acceleration as PathCRF's own pipeline computes them (Savitzky-Golay
    filters at 25 fps), then every 5th frame;
  - the left team (by mean position over the half) first; each team's goalkeeper is its slot
    furthest from halfway in the window, as PathCRF's detect_keepers does over a period;
  - fewer than 11 players: padded with zeros, as PathCRF's own inference does for red cards.
The output used is the model's "who had the ball last" (its macro "prev" logits): a softmax over
the players, summed per team. Windows overlap by half and are blended linearly, as its inference
does.
"""

import hashlib
import json
import os
import sys
import urllib.request

import numpy as np
from scipy.signal import savgol_filter

from possession.data import FPS, HOME, ROOT

# Params
VENDOR = os.path.join(ROOT, "third_party", "pathcrf")
COMMIT = "1b7c6f85a6b78c92e4543fde72ca2b86a933f997"
WEIGHTS_URL = ("https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/{}/saved/140/model/"
               "state_dict_best.pt".format(COMMIT))
WEIGHTS = os.path.join(ROOT, "data", "external", "pathcrf", "state_dict_best.pt")
WEIGHTS_SHA256 = "c82235a17416354b5544f85721ca850304765737a2d9d5435dcdefd52a0d3700"
PITCH = (105.0, 68.0)          # PathCRF's pitch (its datatools/config.py)
SAMPLE = 5                     # 25 fps -> 5 fps, as it was trained
WINDOW = 50                    # steps: 10 s
STRIDE = 25                    # steps: windows overlap by half, as its inference.py does
TEAM_SIZE = 11
MIN_PLAYERS = 8                # per team in a window; with fewer, the window gives no answer
OUTSIDE = ((0.0, 34.0), (105.0, 34.0), (52.5, 0.0), (52.5, 68.0))   # out_left/right/bottom/top


def ensure_weights():
    """Download the weights once from the pinned commit; refuse them if the SHA-256 differs."""
    if os.path.exists(WEIGHTS) and sha256(WEIGHTS) == WEIGHTS_SHA256:
        return WEIGHTS
    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS + ".part")
    if sha256(WEIGHTS + ".part") != WEIGHTS_SHA256:
        raise RuntimeError("PathCRF weights: SHA-256 mismatch")
    os.replace(WEIGHTS + ".part", WEIGHTS)
    return WEIGHTS


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model():
    """PathCRF's model 140, on the CPU, one thread, weights only (no pickled code)."""
    import torch
    ensure_weights()
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from models.utils import build_model   # PathCRF's own builder
    with open(os.path.join(VENDOR, "saved", "140", "args.json")) as handle:
        args = json.load(handle)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = build_model(args, torch.device("cpu"))
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    return model


def kinematics(x, y):
    """vx, vy, speed, accel at 25 fps, as PathCRF's datatools/utils.py computes them."""
    vx = savgol_filter(np.diff(x) * FPS, window_length=15, polyorder=2)
    vy = savgol_filter(np.diff(y) * FPS, window_length=15, polyorder=2)
    ax = savgol_filter(np.diff(vx) * FPS, window_length=9, polyorder=2)
    ay = savgol_filter(np.diff(vy) * FPS, window_length=9, polyorder=2)
    vx, vy = np.r_[vx[0], vx], np.r_[vy[0], vy]
    accel = np.r_[0.0, np.hypot(ax, ay), 0.0]
    return vx, vy, np.hypot(vx, vy), accel


def filled(track):
    """A slot's positions over a window, holes filled by straight lines and the ends held."""
    seen = ~np.isnan(track[:, 0])
    index = np.arange(len(track))
    return np.stack([np.interp(index, index[seen], track[seen, k]) for k in (0, 1)], axis=1)


def window_input(players, team, frames, left_side, pitch):
    """(input [WINDOW, 26, 8] float32, side of each node [26] (+1 home, -1 away, 0 other)) or None."""
    length, width = pitch
    scale = np.array([PITCH[0] / length, PITCH[1] / width])
    rows = players[frames]                                     # [250, slots, 2]
    present = (~np.isnan(rows[:, :, 0])).sum(axis=0)
    nodes, sides, keepers = [], [], []
    for side in (left_side, 1 - left_side):
        slots = [s for s in np.argsort(-present, kind="stable") if team[s] == side and present[s] > 0][:TEAM_SIZE]
        if len(slots) < MIN_PLAYERS:
            return None
        tracks = [filled(rows[:, s]) * scale for s in slots]
        depth = [abs(t[:, 0].mean() - PITCH[0] / 2) for t in tracks]
        keeper = int(np.argmax(depth))
        team_nodes = []
        for i, t in enumerate(tracks):
            vx, vy, speed, accel = kinematics(t[:, 0], t[:, 1])
            features = np.stack([t[:, 0], t[:, 1], vx, vy, speed, accel], axis=1)[::SAMPLE]
            indicator = np.zeros((len(features), 2))
            indicator[:, 0] = -1.0 if side == left_side else 1.0
            indicator[:, 1] = 1.0 if i == keeper else 0.0
            team_nodes.append(np.concatenate([indicator, features], axis=1))
        team_nodes += [np.zeros_like(team_nodes[0])] * (TEAM_SIZE - len(team_nodes))   # padding
        nodes += team_nodes
        sides += [1 if side == HOME else -1] * len(slots) + [0] * (TEAM_SIZE - len(slots))
    steps = len(nodes[0])
    for ox, oy in OUTSIDE:
        nodes.append(np.concatenate([np.zeros((steps, 2)), np.tile([ox, oy, 0, 0, 0, 0], (steps, 1))], axis=1))
        sides.append(0)
    return np.stack(nodes, axis=1).astype(np.float32), np.array(sides)


def p_home_in_window(model, inputs, sides):
    """P(home had the ball last) per step: the model's "prev" logits, softmax over real players."""
    import torch
    with torch.no_grad():
        macro, _, _ = model(torch.tensor(inputs).unsqueeze(0))
    logits = macro[0, :, :, 0].numpy().astype(np.float64)            # [steps, 26]
    logits[:, sides == 0] = -np.inf                                  # pads and outside nodes
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights[:, sides == 1].sum(axis=1) / weights.sum(axis=1)


def p_home(model, tracking, halves, pitch, live):
    """P(home has the ball) per frame on `live` stretches (bool [frames]); NaN elsewhere.

    Each live stretch is cut into WINDOW-step windows overlapping by half (a stretch shorter than a
    window is one window), the model's answer per 5-frame step is spread over those 5 frames, and
    overlapping windows are blended linearly.
    """
    players, team = tracking["players"], tracking["team"]
    total = np.zeros(len(live))
    weight = np.zeros(len(live))
    for _, first, last in halves:
        half = np.arange(first, last + 1)
        left_side = HOME if np.nanmean(players[half][:, team == HOME, 0]) < np.nanmean(players[half][:, team != HOME, 0]) else 1 - HOME
        edges = np.flatnonzero(np.diff(np.r_[0, live[first:last + 1].astype(np.int8), 0]))
        for start, stop in zip(edges[::2] + first, edges[1::2] + first):
            span = WINDOW * SAMPLE
            if stop - start < 2 * SAMPLE + 15:      # too short for the filters
                continue
            starts = [start] if stop - start <= span else \
                list(range(start, stop - span + 1, STRIDE * SAMPLE)) + [stop - span]
            for w0 in sorted(set(starts)):
                frames = np.arange(w0, min(w0 + span, stop))
                built = window_input(players, team, frames, left_side, pitch)
                if built is None:
                    continue
                p = np.repeat(p_home_in_window(model, *built), SAMPLE)[:len(frames)]
                ramp = np.minimum(1.0, np.minimum(np.arange(1, len(frames) + 1), np.arange(len(frames), 0, -1))
                                  / (STRIDE * SAMPLE))
                total[frames] += ramp * p
                weight[frames] += ramp
    out = np.full(len(live), np.nan)
    has = weight > 0
    out[has] = total[has] / weight[has]
    return out
