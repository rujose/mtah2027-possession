"""Keeper repair: pieces of tracking that are clearly one goal's keeper but sit in the other team's
slots move to a new slot of the defending team (docs/models/v8-keeper-repair.md).

    python -m possession.repair        # what the rule finds on our match; writes nothing

Per half and per goal, the defending team is the one whose players stand nearer to that goal on
average. A suspect stretch: a slot of the other team inside the goal's 6-yard box, no defending-team
slot in it, and the ball seen more than BALL_FAR from the goal, for MIN_STAY frames in a row. Its
piece (the slot's unbroken run of presence within the half) is repaired if it is inside that
penalty area on at least MIN_INSIDE_AREA of its frames. Positions and speeds are never changed: the
piece moves, whole, to a new slot appended after the existing ones, so slot numbers keep meaning
what they meant.
"""

import numpy as np

from possession import v1
from possession.data import AWAY, FPS, HOME, PITCH_LENGTH, PITCH_WIDTH, graded_halves, load_tracking

# Params (the boxes are the Laws of the Game; the rest is reasoning, not fitted)
SIX_DEPTH, SIX_HALF_WIDTH = 5.5, 9.16        # 6-yard box: 5.5 m deep, 18.32 m wide
AREA_DEPTH, AREA_HALF_WIDTH = 16.5, 20.16    # penalty area: 16.5 m deep, 40.32 m wide
BALL_FAR = 35.0                              # m from the goal: about a third of the pitch
MIN_STAY = 5 * FPS                           # frames in a row
MIN_INSIDE_AREA = 0.9                        # share of the piece's frames inside the penalty area


def inside(x, y, goal_x, depth, half_width, width):
    """Inside the box of that depth and width in front of the goal at goal_x (False where NaN)."""
    return (np.abs(x - goal_x) < depth) & (np.abs(y - width / 2) < half_width)


def keeper_pieces(tracking, halves, pitch):
    """Every piece the rule looks at, repaired or not, as dicts:

        half, slot, start, end (exclusive video frames), defender (the team it would join),
        fired (frames of suspect stretches inside it), inside_area (share), repaired (bool)
    """
    length, width = pitch
    players, team, ball = tracking["players"], tracking["team"], tracking["ball"]
    pieces = []
    for half, first, last in halves:
        frames = np.arange(first, last + 1)
        x, y = players[frames, :, 0], players[frames, :, 1]
        home_left = np.nanmean(x[:, team == HOME]) < np.nanmean(x[:, team == AWAY])
        goals = ((0.0, HOME if home_left else AWAY), (length, AWAY if home_left else HOME))
        for goal_x, defender in goals:
            in_box = inside(x, y, goal_x, SIX_DEPTH, SIX_HALF_WIDTH, width)
            in_area = inside(x, y, goal_x, AREA_DEPTH, AREA_HALF_WIDTH, width)
            defenders_in = in_box[:, team == defender].any(axis=1)
            with np.errstate(invalid="ignore"):          # a missing ball is not "far"
                far = np.hypot(ball[frames, 0] - goal_x, ball[frames, 1] - width / 2) > BALL_FAR
            for slot in np.flatnonzero(team != defender):
                piece_starts, piece_ends = v1.runs(~np.isnan(x[:, slot]))
                fired = {}
                for start, end in zip(*v1.runs(in_box[:, slot] & ~defenders_in & far)):
                    if end - start >= MIN_STAY:
                        k = np.searchsorted(piece_starts, start, side="right") - 1
                        fired[k] = fired.get(k, 0) + int(end - start)
                for k, count in sorted(fired.items()):
                    start, end = piece_starts[k], piece_ends[k]
                    share = float(in_area[start:end, slot].mean())
                    pieces.append({"half": half, "slot": int(slot), "start": int(first + start),
                                   "end": int(first + end), "defender": int(defender), "fired": count,
                                   "inside_area": share, "repaired": share >= MIN_INSIDE_AREA})
    return pieces


def repair(tracking, halves, pitch):
    """(tracking with each repaired piece moved to a new slot of its defending team, the pieces).

    The input is not modified; with nothing to repair it is returned as is.
    """
    pieces = keeper_pieces(tracking, halves, pitch)
    moved = [p for p in pieces if p["repaired"]]
    if not moved:
        return tracking, pieces
    fixed = dict(tracking)
    n_slots = len(tracking["team"])
    fixed["team"] = np.concatenate([tracking["team"], np.array([p["defender"] for p in moved], tracking["team"].dtype)])
    for key in ("players", "speed"):
        if key in tracking:
            values = tracking[key]
            extra = np.full((len(values), len(moved)) + values.shape[2:], np.nan, values.dtype)
            fixed[key] = np.concatenate([values, extra], axis=1)
            for i, p in enumerate(moved):
                fixed[key][p["start"]:p["end"], n_slots + i] = values[p["start"]:p["end"], p["slot"]]
                fixed[key][p["start"]:p["end"], p["slot"]] = np.nan
    for i, p in enumerate(moved):
        p["new_slot"] = n_slots + i
    return fixed, pieces


def describe(pieces, team_name=("home", "away")):
    for p in pieces:
        print("  half {} slot {} -> {}: frames {}-{} ({:.0f} s), rule fired {:.0f} s, inside the area {:.0%}: {}".format(
            p["half"], p["slot"], team_name[p["defender"]], p["start"], p["end"], (p["end"] - p["start"]) / FPS,
            p["fired"] / FPS, p["inside_area"], "repaired as slot {}".format(p["new_slot"]) if p["repaired"]
            else "left alone (under {:.0%})".format(MIN_INSIDE_AREA)))


def main():
    pieces = repair(load_tracking(), graded_halves(), (PITCH_LENGTH, PITCH_WIDTH))[1]
    print("keeper repair on our match: {} piece(s) found, {} repaired".format(
        len(pieces), sum(p["repaired"] for p in pieces)))
    describe(pieces, ("Barcelona", "Athletic"))


if __name__ == "__main__":
    main()
