"""Model v8: v7 on tracking where keepers tracked as the other team are given back to their own.

    python -m possession.v8            # writes evaluation/results/v8.csv and prints diagnostics

Full explanation: docs/models/v8-keeper-repair.md. possession/repair.py moves each piece of
tracking that is clearly one goal's keeper but sits in the other team's slots to a new slot of the
defending team; v7 then runs unchanged on the repaired tracking. On the bench the rule never fires
(`python -m bench.keeper_repair`), so there v8 is v7.
"""

import os

import numpy as np

from possession import repair, v1, v4, v7
from possession.audio import whistle_events
from possession.data import AWAY, FPS, HOME, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v8.csv")


def predict(tracking, tags, events, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH)):
    """v7's five outputs on the repaired tracking, then the repaired tracking and the pieces."""
    halves = halves or graded_halves()
    fixed, pieces = repair.repair(tracking, halves, pitch)
    return v7.predict(fixed, tags, events, halves, pitch) + (fixed, pieces)


def on_ball_team(clues):
    """0 = nobody, 1 = Barcelona, 2 = Athletic, from v1's "who is on the ball"."""
    team = np.zeros(len(clues["on_ball"]), np.int8)
    team[clues["on_ball"][:, HOME] > 0] = 1
    team[clues["on_ball"][:, AWAY] > 0] = 2
    return team


def main():
    tracking = load_tracking()
    tags = load_tags()
    events = whistle_events(tracking)
    labels, clues, p_dead, p_start, p_pathcrf, fixed, pieces = predict(tracking, tags, events)
    v1.write_submission(OUTPUT, labels)
    print("wrote " + os.path.relpath(OUTPUT, ROOT))
    repair.describe(pieces, ("Barcelona", "Athletic"))

    before, clues_before, _, p_start_before, p_pathcrf_before = v7.predict(tracking, tags, events)
    v4.compare(labels, before, events, names=("v7", "v8"))
    for p in (q for q in pieces if q["repaired"]):
        span = slice(p["start"], p["end"])
        print("inside the repaired piece (frames {}-{}, {:.0f} s):".format(p["start"], p["end"], (p["end"] - p["start"]) / FPS))
        team_before, team_after = on_ball_team(clues_before)[span], on_ball_team(clues)[span]
        print("  'who is on the ball' changes team on {} frames ({:.1f} s)".format(
            np.sum(team_before != team_after), np.sum(team_before != team_after) / FPS))
        for name, values in (("restart model P(Barcelona restarts)", (p_start_before, p_start)),
                             ("PathCRF P(Barcelona has the ball)", (p_pathcrf_before, p_pathcrf))):
            a, b = values[0][span], values[1][span]
            both = ~np.isnan(a) & ~np.isnan(b)
            print("  {}: mean {:.3f} -> {:.3f} on {} frames where both answer; moved by > 0.1 on {}".format(
                name, a[both].mean() if both.any() else np.nan, b[both].mean() if both.any() else np.nan,
                both.sum(), np.sum(np.abs(a[both] - b[both]) > 0.1)))
        counts = {label: (np.sum(before[span] == label), np.sum(labels[span] == label)) for label in "HAD"}
        print("  labels in the piece, v7 -> v8: " + "  ".join("{} {} -> {}".format(k, *v) for k, v in counts.items()))


if __name__ == "__main__":
    main()
