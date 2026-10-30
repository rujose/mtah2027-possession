"""Model v2: model v1 + the referee's whistle, counted only when the players slow down after it.

    python -m possession.v2            # writes evaluation/results/v2.csv and prints diagnostics

Everything from v1 is reused unchanged (clues, weights, switch penalties, decider), so any difference
between v1.csv and v2.csv comes from the whistle. Full explanation: docs/models/v2-whistle.md
"""

import os

import numpy as np

from possession import v1
from possession.audio import whistle_events
from possession.data import FPS, ROOT, graded_halves, load_tags, load_tracking
from possession.decoder import viterbi

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v2.csv")
W_WHISTLE = 2.0            # against H and A; same trust as "ball out"
WHISTLE_WINDOW = 3 * FPS   # frames after the whistle starts: beats both switch penalties (150 vs 50)


def whistle_mask(events, n_frames):
    """True for the WHISTLE_WINDOW frames after each counted whistle."""
    mask = np.zeros(n_frames, bool)
    for event in events:
        if event["counted"]:
            mask[event["start"]:event["start"] + WHISTLE_WINDOW] = True
    return mask


def predict(tracking, tags, events):
    halves = graded_halves()
    clues = v1.compute_clues(tracking, tags, halves)
    clues["whistle"] = whistle_mask(events, len(clues["out"]))
    cost = v1.build_costs(clues)
    cost[:, v1.H] += W_WHISTLE * clues["whistle"]
    cost[:, v1.A] += W_WHISTLE * clues["whistle"]
    switch = v1.switch_matrix()
    labels = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in halves:
        labels[first:last + 1] = v1.LABELS[viterbi(cost[first:last + 1], switch)]
    return labels, clues


def compare(labels, before, events):
    """What changed against v1, frame by frame and per counted whistle."""
    graded = labels != "X"
    changed = (labels != before) & graded
    print("frames that differ from v1: {} ({:.2f}% of graded)".format(changed.sum(), 100 * changed[graded].mean()))
    starts, ends = v1.runs(changed)
    print("changed stretches: {}".format(len(starts)))
    for event in events:
        if not event["counted"]:
            continue
        s = event["start"]
        window = slice(s, s + 30 * FPS)
        print("  whistle at {:6d}  speed {:.2f} -> {:.2f}  v1 at whistle {}  v2 at whistle {}  frames changed in next 30 s: {}".format(
            s, event["speed_before"], event["speed_after"], before[s], labels[s], int(changed[window].sum())))


def main():
    tracking = load_tracking()
    tags = load_tags()
    events = whistle_events(tracking)
    labels, clues = predict(tracking, tags, events)
    v1.write_submission(OUTPUT, labels)
    print("wrote " + os.path.relpath(OUTPUT, ROOT))
    print("whistle events {}, counted {}".format(len(events), sum(e["counted"] for e in events)))
    v1.describe(labels, clues)
    before, _ = v1.predict(tracking, tags)
    compare(labels, before, events)


if __name__ == "__main__":
    main()
