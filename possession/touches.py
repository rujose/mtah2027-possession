"""Seen touches: frames where a team's player stops, deflects or plays the ball (docs/models/v11-touch-line.md).

Frame k is a touch by a team when the ball is seen at k and LOOK frames either side, a player of that
team is within REACH of it at k, and the ball's velocity over the LOOK frames after k differs from its
velocity over the LOOK frames before by at least the change asked for (ΔV, chosen on the bench). A
touch shows as a short run of such frames; the touch is the frame in the run where the change is
largest. Used by v11 to put each change of team on the new team's touch.
"""

import numpy as np

from possession.data import FPS

# Params (by reasoning, not searched; the page gives the why of each)
REACH = 2.0            # m: v1's "on the ball" distance, a stride plus tracking error
LOOK = int(0.2 * FPS)  # frames either side: short enough to separate two touches, long enough to see through jitter


def nearest(players, team, ball, side):
    """float [frames]: distance from the ball to team `side`'s nearest seen player (inf if none)."""
    with np.errstate(invalid="ignore"):
        d = np.hypot(players[:, team == side, 0] - ball[:, None, 0], players[:, team == side, 1] - ball[:, None, 1])
        return np.min(np.where(np.isnan(d), np.inf, d), axis=1)


def velocity_change(ball):
    """float [frames]: |velocity over the LOOK frames after - over the LOOK frames before| (m/s);
    NaN where the ball isn't seen at k - LOOK, k and k + LOOK."""
    n = len(ball)
    before, after = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    before[LOOK:] = (ball[LOOK:] - ball[:-LOOK]) / (LOOK / FPS)
    after[:-LOOK] = (ball[LOOK:] - ball[:-LOOK]) / (LOOK / FPS)
    return np.hypot(*(after - before).T)


def touches(players, team, ball, side, dv):
    """bool [frames]: a touch by team `side` (the peak of each run of frames that qualify)."""
    change = velocity_change(ball)
    with np.errstate(invalid="ignore"):
        holds = (nearest(players, team, ball, side) < REACH) & (change >= dv)
    out = np.zeros(len(ball), bool)
    edges = np.flatnonzero(np.diff(np.r_[0, holds.astype(np.int8), 0]))
    for first, end in zip(edges[::2], edges[1::2]):
        out[first + np.argmax(change[first:end])] = True
    return out
