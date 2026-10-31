"""Per-frame features from the players only, for v4's dead-ball model (docs/models/v4-dead-ball.md,
piece 1).

Order-free (no feature asks which slot is which player) and symmetric (swapping home and away
changes nothing), so anonymous slots and identity swaps don't matter. No ball: Metrica's sample
games blank the ball in dead time, so a model that saw it would learn "no ball = dead". Speed is
computed from positions on every match, ours included, so it means the same thing everywhere.
"""

import warnings

import numpy as np

from possession.data import AWAY, FPS, HOME

# Params
SPEED_SPAN = 2                  # frames each side: speed = distance over 0.16 s / 0.16 s
SLOW, FAST = 1.0, 4.0           # m/s: "standing" and "running"
LINE = 2.0                      # m from a touchline or goal line
CORNER = 3.0                    # m from a corner
CENTRE = 9.15                   # m: the centre circle's radius
SIX_DEPTH, SIX_HALF = 5.5, 9.16       # 6-yard box: depth, half of its width (18.32 m)
AREA_DEPTH, AREA_HALF = 16.5, 20.16   # penalty area: depth, half of its width (40.32 m)
CONTEXT = (1, 4, 20)            # seconds: means over the past and the next this-many seconds
WITH_CONTEXT = ("share_slow", "share_fast", "near_touchline", "near_goal_line", "in_six_yard_box",
                "in_penalty_area", "near_corner", "near_centre", "outside_pitch", "own_half_share")


def speeds(players, halves):
    """m/s per slot from positions, within each half; NaN where either end is missing."""
    out = np.full(players.shape[:2], np.nan, dtype=np.float32)
    for _, first, last in halves:
        p = players[first:last + 1]
        step = np.hypot(*(p[2 * SPEED_SPAN:] - p[:-2 * SPEED_SPAN]).transpose(2, 0, 1))
        out[first + SPEED_SPAN:last + 1 - SPEED_SPAN] = step / (2 * SPEED_SPAN / FPS)
    return out


def share(mask, seen):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen.sum(axis=1) > 0, mask.sum(axis=1) / seen.sum(axis=1), np.nan)


def base_features(players, team, halves, pitch):
    """{name: [frames]} for one match, before time context."""
    length, width = pitch
    x, y = players[:, :, 0], players[:, :, 1]
    seen = ~np.isnan(x)
    speed = speeds(players, halves)
    known = ~np.isnan(speed)
    home, away = team == HOME, team == AWAY
    f = {}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN frames give NaN, as intended
        f["speed_mean"] = np.nanmean(speed, axis=1)
        f["speed_median"] = np.nanmedian(speed, axis=1)
        f["speed_p90"] = np.nanpercentile(speed, 90, axis=1)
        f["share_slow"] = share(known & (speed < SLOW), known)
        f["share_fast"] = share(known & (speed > FAST), known)
        f["speed_gap_teams"] = np.abs(np.nanmean(speed[:, home], axis=1) - np.nanmean(speed[:, away], axis=1))

        f["spread_along"] = np.nanstd(x, axis=1)
        f["spread_across"] = np.nanstd(y, axis=1)
        centre = {side: np.stack([np.nanmean(x[:, team == side], axis=1),
                                  np.nanmean(y[:, team == side], axis=1)], axis=1)
                  for side in (HOME, AWAY)}
        f["team_centre_gap"] = np.hypot(*(centre[HOME] - centre[AWAY]).T)
        compact = np.stack([np.nanmean(np.hypot(x[:, team == side] - centre[side][:, :1],
                                                y[:, team == side] - centre[side][:, 1:]), axis=1)
                            for side in (HOME, AWAY)], axis=1)
        f["compact_max"] = np.nanmax(compact, axis=1)
        f["compact_min"] = np.nanmin(compact, axis=1)

        mid = width / 2
        f["near_touchline"] = np.maximum((seen & (y < LINE)).sum(axis=1), (seen & (y > width - LINE)).sum(axis=1))
        f["near_goal_line"] = np.maximum((seen & (x < LINE)).sum(axis=1), (seen & (x > length - LINE)).sum(axis=1))
        wide = np.abs(y - mid)
        f["in_six_yard_box"] = np.maximum((seen & (x < SIX_DEPTH) & (wide < SIX_HALF)).sum(axis=1),
                                          (seen & (x > length - SIX_DEPTH) & (wide < SIX_HALF)).sum(axis=1))
        f["in_penalty_area"] = np.maximum((seen & (x < AREA_DEPTH) & (wide < AREA_HALF)).sum(axis=1),
                                          (seen & (x > length - AREA_DEPTH) & (wide < AREA_HALF)).sum(axis=1))
        corners = [(0, 0), (0, width), (length, 0), (length, width)]
        f["near_corner"] = sum((seen & (np.hypot(x - cx, y - cy) < CORNER)).sum(axis=1) for cx, cy in corners)
        f["near_centre"] = (seen & (np.hypot(x - length / 2, y - mid) < CENTRE)).sum(axis=1)
        f["outside_pitch"] = (seen & ((x < 0) | (x > length) | (y < 0) | (y > width))).sum(axis=1)
        left = x < length / 2
        own_if_home_left = (seen & ((home & left) | (away & ~left))).sum(axis=1)
        own_if_home_right = (seen & ((home & ~left) | (away & left))).sum(axis=1)
        visible = seen.sum(axis=1)
        f["own_half_share"] = np.where(visible > 0, np.maximum(own_if_home_left, own_if_home_right)
                                       / np.maximum(visible, 1), np.nan)
    for name in f:
        f[name] = f[name].astype(np.float32)
    return f


def window_mean(values, window, halves, ahead):
    """Mean over the `window` frames ending at t (or starting at t if `ahead`), within each half,
    ignoring NaN; NaN where the whole window is NaN."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    for _, first, last in halves:
        v = values[first:last + 1].astype(np.float64)
        valid = ~np.isnan(v)
        total = np.r_[0.0, np.cumsum(np.where(valid, v, 0.0))]
        count = np.r_[0, np.cumsum(valid)]
        index = np.arange(len(v))
        if ahead:
            low, high = index, np.minimum(index + window, len(v))
        else:
            low, high = np.maximum(index - window + 1, 0), index + 1
        n = count[high] - count[low]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[first:last + 1] = np.where(n > 0, (total[high] - total[low]) / n, np.nan)
    return out


def frame_features(tracking, halves, pitch):
    """(names, X float32 [frames, features]) for every frame; NaN outside the halves."""
    f = base_features(tracking["players"], tracking["team"], halves, pitch)
    for name in WITH_CONTEXT:
        for seconds in CONTEXT:
            for ahead, word in ((False, "past"), (True, "next")):
                f["{}_{}_{}s".format(name, word, seconds)] = window_mean(f[name], seconds * FPS, halves, ahead)
    names = list(f)
    X = np.stack([f[n] for n in names], axis=1)
    outside = np.ones(len(X), bool)
    for _, first, last in halves:
        outside[first:last + 1] = False
    X[outside] = np.nan
    return names, X
