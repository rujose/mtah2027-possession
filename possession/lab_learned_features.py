"""Per-frame features for the lab candidate `learned` (docs/lab/learned.md).

One row per frame, from what a match offers at prediction time and nothing else: its damaged
tracking (players, ball, team) and the out-of-sample outputs of the earlier models on it (P(dead),
the restart probabilities, PathCRF's P(home), v6's and v7's decoder costs). Never its truth.

Everything is computed within each half. Distances, counts and speeds come in home/away pairs, so
`frame_features(..., swap=True)` is the same match with the teams' names exchanged: the learned
model is trained on both and answers as the mean of both, so it can't learn "home has the ball
more" from a bench whose home teams happen to.
"""

import warnings

import numpy as np

from possession import features, touches
from possession.data import AWAY, FPS, HOME

# Params (by reasoning, not searched; docs/lab/learned.md gives the why of each)
NEAR, CLOSE = 2.0, 5.0       # m: "on the ball" (v1's distance) and "close enough to challenge"
FAR = 60.0                   # m: a team with nobody seen counts as this far
DIFF_CLIP = 20.0             # m: beyond this, the gap between the teams' nearest says nothing more
CAP = 10.0                   # s: "since / until" a touch or pass, clipped here
GAP_CAP = 30.0               # s: "since / until" the ball was seen, clipped here
LOOK = touches.LOOK          # frames: velocity before / after, as the touch detector
TURN_MIN_SPEED = 1.0         # m/s: below this the ball's course is jitter, not a direction
WINDOWS = (1, 3)             # s: means over the past and the next this-many seconds
# Inputs that carry PathCRF (dropped in the "no_pathcrf" variant).
PATHCRF = ("p_pathcrf", "c7_", "v7_cost")


def since(mask, halves, cap):
    """s since the last True frame within the half (0 on True frames); `cap` where there's none."""
    out = np.full(len(mask), np.nan, dtype=np.float32)
    for _, first, last in halves:
        m = mask[first:last + 1]
        index = np.arange(len(m))
        seen = np.maximum.accumulate(np.where(m, index, -10 ** 9))
        out[first:last + 1] = np.minimum((index - seen) / FPS, cap)
    return out


def until(mask, halves, cap):
    """s until the next True frame within the half; `cap` where there's none."""
    out = np.full(len(mask), np.nan, dtype=np.float32)
    for _, first, last in halves:
        out[first:last + 1] = since(mask[first:last + 1][::-1], [(0, 0, last - first)], cap)[::-1]
    return out


def filled(ball, halves, backward=False):
    """The ball's last (or next) seen position within the half; NaN before the first (after the last)."""
    out = np.full(ball.shape, np.nan, dtype=np.float32)
    for _, first, last in halves:
        b = ball[first:last + 1]
        if backward:
            b = b[::-1]
        index = np.maximum.accumulate(np.where(~np.isnan(b[:, 0]), np.arange(len(b)), -1))
        f = np.where(index[:, None] >= 0, b[np.maximum(index, 0)], np.nan)
        out[first:last + 1] = f[::-1] if backward else f
    return out


def home_left(players, team, halves, n):
    """bool [frames]: home defends the x = 0 end in this half (its players' mean x is lower)."""
    out = np.zeros(n, bool)
    for _, first, last in halves:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hx = np.nanmean(players[first:last + 1][:, team == HOME, 0])
            ax = np.nanmean(players[first:last + 1][:, team == AWAY, 0])
        out[first:last + 1] = hx < ax
    return out


def distances(players, team, ball, side):
    """float32 [frames, slots of `side`]: each seen player's distance to the ball (NaN unknown)."""
    p = players[:, team == side]
    return np.hypot(p[:, :, 0] - ball[:, None, 0], p[:, :, 1] - ball[:, None, 1]).astype(np.float32)


def nearest(d, ball_seen):
    """The nearest of each row; FAR if nobody is seen, NaN where the ball isn't."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmin(d, axis=1)
    out = np.where(np.isnan(out), FAR, np.minimum(out, FAR)).astype(np.float32)
    out[~ball_seen] = np.nan
    return out


def window_pair(f, name, values, halves):
    for seconds in WINDOWS:
        f["{}_past_{}s".format(name, seconds)] = features.window_mean(values, seconds * FPS, halves, ahead=False)
        f["{}_next_{}s".format(name, seconds)] = features.window_mean(values, seconds * FPS, halves, ahead=True)


def frame_features(track, halves, pitch, pred, found_touches, found_passes, swap=False):
    """(names, X float32 [frames, features]); NaN outside the halves.

    track: {"players", "ball", "team"}; pred: match.pred (p_dead, p_start, restart_clue, p_pathcrf,
    v6_cost, v7_cost); found_touches / found_passes: {HOME: bool [frames], AWAY: ...} (v11's and
    v9's detectors). swap=True: the same with the teams exchanged (home's columns hold away's)."""
    players, ball, team = track["players"], track["ball"], np.asarray(track["team"])
    n = len(ball)
    length, width = pitch
    if swap:
        team = np.where(team == HOME, AWAY, np.where(team == AWAY, HOME, team))
        found_touches = {HOME: found_touches[AWAY], AWAY: found_touches[HOME]}
        found_passes = {HOME: found_passes[AWAY], AWAY: found_passes[HOME]}
    f = {}
    seen = ~np.isnan(ball[:, 0])
    left = home_left(players, team, halves, n)

    # The ball: seen, its speed, change of speed and course, where it is.
    f["ball_seen"] = seen.astype(np.float32)
    f["since_ball_seen"] = since(seen, halves, GAP_CAP)
    f["until_ball_seen"] = until(seen, halves, GAP_CAP)
    before, after = np.full((n, 2), np.nan, np.float32), np.full((n, 2), np.nan, np.float32)
    before[LOOK:] = (ball[LOOK:] - ball[:-LOOK]) * (FPS / LOOK)
    after[:-LOOK] = (ball[LOOK:] - ball[:-LOOK]) * (FPS / LOOK)
    s_before, s_after = np.hypot(*before.T), np.hypot(*after.T)
    f["ball_speed_before"], f["ball_speed_after"] = s_before, s_after
    f["ball_accel"] = s_after - s_before
    f["ball_dv"] = np.hypot(*(after - before).T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (before * after).sum(axis=1) / (s_before * s_after)
        turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        turn[(s_before < TURN_MIN_SPEED) | (s_after < TURN_MIN_SPEED)] = np.nan
    f["ball_turn"] = turn.astype(np.float32)
    x, y = ball[:, 0], ball[:, 1]
    f["ball_x_from_home_goal"] = np.where(left, x, length - x)
    f["ball_y_from_touchline"] = np.minimum(y, width - y)
    f["ball_outside"] = np.maximum.reduce([-x, x - length, -y, y - width])   # m beyond a line; < 0 inside

    # The players around the ball, per team; and around where it was last / will next be seen.
    for where, b in (("", ball), ("_last_seen", filled(ball, halves)), ("_next_seen", filled(ball, halves, True))):
        b_seen = ~np.isnan(b[:, 0])
        d = {side: distances(players, team, b, side) for side in (HOME, AWAY)}
        near = {side: nearest(d[side], b_seen) for side in (HOME, AWAY)}
        f["d_home" + where], f["d_away" + where] = near[HOME], near[AWAY]
        f["d_diff" + where] = np.clip(near[HOME] - near[AWAY], -DIFF_CLIP, DIFF_CLIP)
        if where == "":
            for r, word in ((NEAR, "2m"), (CLOSE, "5m")):
                for side, name in ((HOME, "home"), (AWAY, "away")):
                    with np.errstate(invalid="ignore"):
                        count = (d[side] < r).sum(axis=1).astype(np.float32)
                    count[~b_seen] = np.nan
                    f["n_{}_{}".format(name, word)] = count
            with np.errstate(invalid="ignore"):
                on_h = ((near[HOME] < NEAR) & (near[HOME] < near[AWAY])).astype(np.float32)
                on_a = ((near[AWAY] < NEAR) & (near[AWAY] < near[HOME])).astype(np.float32)
            on_h[~b_seen], on_a[~b_seen] = np.nan, np.nan
            window_pair(f, "d_diff", f["d_diff"], halves)
            window_pair(f, "on_home", on_h, halves)
            window_pair(f, "on_away", on_a, halves)
    for name in ("ball_speed_after", "ball_dv"):
        window_pair(f, name, f[name], halves)

    # The teams: speed, spread, how far up the pitch (each from its own goal).
    speed = features.speeds(players, halves)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for side, name in ((HOME, "home"), (AWAY, "away")):
            px, py = players[:, team == side, 0], players[:, team == side, 1]
            f["speed_" + name] = np.nanmean(speed[:, team == side], axis=1)
            f["spread_" + name] = np.hypot(np.nanstd(px, axis=1), np.nanstd(py, axis=1))
            own_left = left if side == HOME else ~left
            f["up_" + name] = np.where(own_left, np.nanmean(px, axis=1), length - np.nanmean(px, axis=1))
        f["speed_all"] = np.nanmean(speed, axis=1)

    # Touches (v11's detector) and passes (v9's), per team.
    for kind, found in (("touch", found_touches), ("pass", found_passes)):
        for side, name in ((HOME, "home"), (AWAY, "away")):
            hit = np.asarray(found[side], bool)
            f["since_{}_{}".format(kind, name)] = since(hit, halves, CAP)
            f["until_{}_{}".format(kind, name)] = until(hit, halves, CAP)
            for seconds in (3,):
                w = seconds * FPS
                f["{}s_{}_past_{}s".format(kind, name, seconds)] = features.window_mean(hit.astype(np.float32), w, halves, False) * w
                f["{}s_{}_next_{}s".format(kind, name, seconds)] = features.window_mean(hit.astype(np.float32), w, halves, True) * w

    # The earlier models' out-of-sample outputs on this match.
    def flip(p):
        return (1.0 - p) if swap else p
    f["p_dead"] = pred["p_dead"]
    f["p_start"] = flip(pred["p_start"])
    f["restart_clue"] = flip(pred["restart_clue"])
    f["p_pathcrf"] = flip(pred["p_pathcrf"])
    window_pair(f, "p_pathcrf", f["p_pathcrf"].astype(np.float32), halves)
    for version in ("v6", "v7"):
        c = np.asarray(pred[version + "_cost"], dtype=np.float32)
        h, a, dead = (c[:, 1], c[:, 0], c[:, 2]) if swap else (c[:, 0], c[:, 1], c[:, 2])
        f[version + "_cost_home"], f[version + "_cost_away"], f[version + "_cost_dead"] = h, a, dead
        tag = "c" + version[1]
        window_pair(f, tag + "_team", a - h, halves)                      # > 0: home is cheaper
        window_pair(f, tag + "_dead", dead - np.minimum(h, a), halves)    # > 0: a team is cheaper than dead

    names = list(f)
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], axis=1)
    outside = np.ones(n, bool)
    for _, first, last in halves:
        outside[first:last + 1] = False
    X[outside] = np.nan
    return names, X


def columns(names, variant):
    """Indices of the features a variant uses: "all", or "no_pathcrf" (nothing PathCRF touched)."""
    if variant == "all":
        return np.arange(len(names))
    if variant == "no_pathcrf":
        return np.array([i for i, k in enumerate(names) if not k.startswith(PATHCRF)])
    raise ValueError(variant)
