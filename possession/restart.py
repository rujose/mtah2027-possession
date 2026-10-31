"""Who restarts after a stoppage: per-team layout features, P(home restarts), and the clue built from
it (docs/models/v6-restart-team.md, pieces 1 and 3).

Features describe "this team" against "the other team" and the two ends of the pitch, never home or
away and never attack directions. Players only: the ball is missing at most restarts, and Metrica's
samples blank it in dead time. Every frame is seen in four orientations (teams swapped, pitch
flipped end to end, both), so the model can only learn shapes, not "home usually restarts".
"""

import warnings

import numpy as np

from possession import dead_ball, features
from possession.data import AWAY, FPS, HOME

# Params
LINE = 1.0                      # m inside a touchline that still counts as "on the line"
CORNER = 3.0                    # m from a corner
CENTRE = 9.15                   # m: the centre circle's radius
SIX_DEPTH, SIX_HALF = 5.5, 9.16
AREA_DEPTH, AREA_HALF = 16.5, 20.16
CONTEXT = 2                     # seconds: past and next means of the layout counts
WITH_CONTEXT = ("on_touchline", "corner_left", "corner_right", "six_left", "six_right",
                "area_left", "area_right", "centre_circle")
SMOOTH = 2 * FPS                # frames of dead-looking play R is averaged over
DEAD_LOOKING = 0.5              # P(dead) above this: the frame looks like a stoppage
MIN_STOPPAGE = 2 * FPS          # dead-looking frames in a row before the clue switches on: only
                                # 1 % of the bench's real stoppages are shorter (revision B)


def team_counts(x, y, mine, speed, pitch):
    """{name: [frames]} for the players in `mine` (a slot mask)."""
    length, width = pitch
    seen = ~np.isnan(x) & mine
    mid = width / 2
    wide = np.abs(y - mid)
    f = {}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["beyond_touchline"] = (seen & ((y < 0) | (y > width))).sum(axis=1)
        f["on_touchline"] = (seen & (((y >= 0) & (y < LINE)) | ((y <= width) & (y > width - LINE)))).sum(axis=1)
        f["beyond_goal_left"] = (seen & (x < 0)).sum(axis=1)
        f["beyond_goal_right"] = (seen & (x > length)).sum(axis=1)
        f["corner_left"] = (seen & ((np.hypot(x, y) < CORNER) | (np.hypot(x, y - width) < CORNER))).sum(axis=1)
        f["corner_right"] = (seen & ((np.hypot(x - length, y) < CORNER)
                                     | (np.hypot(x - length, y - width) < CORNER))).sum(axis=1)
        f["six_left"] = (seen & (x < SIX_DEPTH) & (wide < SIX_HALF)).sum(axis=1)
        f["six_right"] = (seen & (x > length - SIX_DEPTH) & (wide < SIX_HALF)).sum(axis=1)
        f["area_left"] = (seen & (x < AREA_DEPTH) & (wide < AREA_HALF)).sum(axis=1)
        f["area_right"] = (seen & (x > length - AREA_DEPTH) & (wide < AREA_HALF)).sum(axis=1)
        f["centre_circle"] = (seen & (np.hypot(x - length / 2, y - mid) < CENTRE)).sum(axis=1)
        count = seen.sum(axis=1)
        f["share_left_half"] = np.where(count > 0, (seen & (x < length / 2)).sum(axis=1) / np.maximum(count, 1), np.nan)
        f["mean_speed"] = np.nanmean(np.where(seen, speed, np.nan), axis=1)
        xs, ys = np.where(seen, x, np.nan), np.where(seen, y, np.nan)
        f["spread"] = np.sqrt(np.nanvar(xs, axis=1) + np.nanvar(ys, axis=1))
        f["centre_along"] = np.nanmean(xs, axis=1)
    return {k: v.astype(np.float32) for k, v in f.items()}


def orientation_features(players, team, halves, pitch, side, flipped):
    """(names, X [frames, features]) with `side` as "this team", the pitch flipped end to end or not."""
    x, y = players[:, :, 0].copy(), players[:, :, 1]
    if flipped:
        x = pitch[0] - x
    speed = features.speeds(players, halves)
    columns = {}
    for prefix, mine in (("this_", team == side), ("other_", team != side)):
        counts = team_counts(x, y, mine, speed, pitch)
        for name, values in counts.items():
            columns[prefix + name] = values
        for name in WITH_CONTEXT:
            for ahead, word in ((False, "past"), (True, "next")):
                columns["{}{}_{}_{}s".format(prefix, name, word, CONTEXT)] = features.window_mean(
                    counts[name], CONTEXT * FPS, halves, ahead)
    names = list(columns)
    X = np.stack([columns[n] for n in names], axis=1)
    outside = np.ones(len(X), bool)
    for _, first, last in halves:
        outside[first:last + 1] = False
    X[outside] = np.nan
    return names, X


ORIENTATIONS = ((HOME, False), (AWAY, False), (HOME, True), (AWAY, True))


def all_orientations(tracking, halves, pitch):
    """[(side, flipped, names, X)] for the four orientations."""
    return [(side, flipped) + orientation_features(tracking["players"], tracking["team"], halves,
                                                   pitch, side, flipped)
            for side, flipped in ORIENTATIONS]


def p_home(model, oriented):
    """P(home restarts) per frame, symmetric: the mean of the four orientations, each turned into
    "home" (an AWAY orientation's P(this team) is 1 - P(home))."""
    ps = []
    for side, _, names, X in oriented:
        if model["features"] != names:
            raise ValueError("the restart model's features differ from possession/restart.py")
        p = dead_ball.predict(model, X)
        ps.append(p if side == HOME else 1.0 - p)
    return np.round(np.mean(ps, axis=0), dead_ball.DECIMALS)


def restart_belief(p_home_frames, p_dead, halves, carry_frames):
    """R per frame, or NaN where the clue is silent.

    The clue switches on only once play has looked dead (P(dead) > DEAD_LOOKING) for MIN_STOPPAGE
    frames in a row (revision B), so short blips in open play never trigger it. From then on, R is
    the mean P(home restarts) over the last SMOOTH dead-looking frames, and it is carried for
    `carry_frames` after that stretch ends.
    """
    looks_dead = p_dead > DEAD_LOOKING
    smooth = features.window_mean(np.where(looks_dead, p_home_frames, np.nan).astype(np.float32),
                                  SMOOTH, halves, ahead=False)
    out = np.full(len(p_dead), np.nan)
    for _, first, last in halves:
        current, run, since = np.nan, 0, carry_frames + 1
        for t in range(first, last + 1):
            run = run + 1 if looks_dead[t] else 0
            if run >= MIN_STOPPAGE:
                current, since = smooth[t], 0
            elif not looks_dead[t] and since <= carry_frames:
                since += 1
            if since <= carry_frames and (run >= MIN_STOPPAGE or not looks_dead[t]):
                out[t] = current
    return out


def costs(belief, weight):
    """(cost of saying H, cost of saying A) per frame, with R = P(home restarts).

    Only the unlikely team pays, by the odds: weight x log(R / (1 - R)) on A when home is likelier,
    weight x log((1 - R) / R) on H when away is. The preference between the two teams is the same as
    "how surprised we'd be" (-log R against -log(1 - R)), but the likely team pays nothing, so being
    near a stoppage no longer makes staying dead relatively cheaper (revision A). Zero where silent.
    """
    r = np.clip(np.nan_to_num(belief, nan=0.5), *dead_ball.CLIP)
    odds = np.log(r / (1.0 - r))
    silent = np.isnan(belief)
    return (np.where(silent, 0.0, weight * np.maximum(-odds, 0.0)),
            np.where(silent, 0.0, weight * np.maximum(odds, 0.0)))
