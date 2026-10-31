"""Seen passes: frames where a player plays the ball away (docs/models/v9-restart-pass.md).

Frame k starts a pass by a team when the ball is seen at k and LOOK frames later, a player of that
team is within TAKER of it at k, the ball has moved at least AWAY further from that player by then,
and it covered those frames faster than PLAYED. A pass shows as a short run of such frames; its
start is the run's first frame. Used by v9 to put each restart on its pass, and meant for step 5
(count a gain of possession only on evidence of a touch).
"""

import numpy as np

from possession.data import FPS

# Params (by reasoning, not searched; the page gives the why of each)
TAKER = 2.0            # m: the ball is at the feet (or in the hands) of whoever plays it
LOOK = int(0.4 * FPS)  # frames: long enough to see it leave, short enough that it hasn't arrived
AWAY = 1.0             # m further from the taker after LOOK: leaving, not arriving
PLAYED = 4.0           # m/s over LOOK: played, not placed or dribbled


def pass_frames(players, team, ball, side):
    """bool [frames]: the pass condition holds at this frame for team `side`."""
    n = len(ball)
    out = np.zeros(n, bool)
    k = np.arange(n - LOOK)
    b0, b1 = ball[k], ball[k + LOOK]
    p0, p1 = players[k][:, team == side], players[k + LOOK][:, team == side]
    with np.errstate(invalid="ignore"):
        d0 = np.hypot(p0[:, :, 0] - b0[:, None, 0], p0[:, :, 1] - b0[:, None, 1])
        d1 = np.hypot(p1[:, :, 0] - b1[:, None, 0], p1[:, :, 1] - b1[:, None, 1])
        taker = np.argmin(np.where(np.isnan(d0), np.inf, d0), axis=1)
        near, later = d0[k, taker], d1[k, taker]
        speed = np.hypot(*(b1 - b0).T) / (LOOK / FPS)
        out[k] = (near < TAKER) & (later > near + AWAY) & (speed > PLAYED)
    return out


def pass_starts(players, team, ball, side):
    """bool [frames]: a pass by team `side` starts here (the first frame of each run)."""
    holds = pass_frames(players, team, ball, side)
    return holds & ~np.r_[False, holds[:-1]]
