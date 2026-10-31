"""Model v5: v4 with the referee's whistle given back its weight.

    python -m possession.v5            # writes evaluation/results/v5.csv and prints diagnostics

Full explanation: docs/models/v5-whistle-weight.md. In v4 the dead-ball model, when confident play
goes on, could outweigh a counted whistle (2.0 per frame against its 2.30 at P = 0.01). Here the
whistle keeps v2's weight on top of the strongest "live" evidence the model can give, so within its
3 s a counted whistle beats the model and both switch prices, as in v2. Everything else is v4's.
"""

import os

import numpy as np

from possession import dead_ball, v1, v2, v4
from possession.audio import whistle_events
from possession.data import ROOT, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v5.csv")
# v2's whistle weight + the most the model can say for "live" on one frame (P at its lower clip).
W_WHISTLE = v2.W_WHISTLE + v4.W_DEAD_BALL * -np.log(dead_ball.CLIP[0])


def build_costs(clues, p):
    """v4's costs, with the whistle at W_WHISTLE instead of v2's weight."""
    cost = v4.build_costs(clues, p)
    cost[:, v1.H] += (W_WHISTLE - v2.W_WHISTLE) * clues["whistle"]
    cost[:, v1.A] += (W_WHISTLE - v2.W_WHISTLE) * clues["whistle"]
    return cost


def predict(tracking, tags, events, halves=None, pitch=None):
    kwargs = {} if pitch is None else {"pitch": pitch}
    return v4.predict(tracking, tags, events, halves, build=build_costs, **kwargs)


def main():
    tracking = load_tracking()
    tags = load_tags()
    events = whistle_events(tracking)
    labels, clues, p = predict(tracking, tags, events)
    v1.write_submission(OUTPUT, labels)
    print("wrote " + os.path.relpath(OUTPUT, ROOT))
    print("whistle weight {:.3f} per frame for {} frames (v2-v4: {})".format(
        W_WHISTLE, v2.WHISTLE_WINDOW, v2.W_WHISTLE))
    before, _, _ = v4.predict(tracking, tags, events)
    v4.compare(labels, before, events, names=("v4", "v5"))
    counted = [e for e in events if e["counted"]]
    for e in counted:
        start = e["start"]
        print("  counted whistle at {:6d}: dead frames in the next 30 s, v4 {:3d}  v5 {:3d}".format(
            start, int((before[start:start + 30 * v1.FPS] == "D").sum()),
            int((labels[start:start + 30 * v1.FPS] == "D").sum())))


if __name__ == "__main__":
    main()
