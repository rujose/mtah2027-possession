"""Model v3: v2's clues, decoded with a minimum length per label and switch prices from the bench.

    python -m possession.v3            # writes evaluation/results/v3.csv and prints diagnostics

Full explanation: docs/models/v3-decoder.md. The four decoder settings were chosen on the bench by
`python -m bench.tune_decoder` (bench/results/decoder_choice.json) and are typed in below, so the
submission never needs the bench. Everything else is v2's, except the two Smart Tagging weights,
rescaled so they keep their meaning at the new prices.
"""

import os

import numpy as np

from possession import v1, v2
from possession.audio import whistle_events
from possession.data import (AWAY, HOME, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags,
                             load_tracking)
from possession.decoder import viterbi_min_length

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v3.csv")

# Chosen on the bench: leave-one-match-out, best mean spellF1, then the fewest possessions within
# 0.01 of it (docs/models/v3-decoder.md, piece 2 and decision 1).
# All 7 leave-one-match-out folds picked this same setting (held-out spellF1 0.369 +- 0.026, v1 0.224).
SWITCH_TEAM = 0.5          # H <-> A (v1: 15)
SWITCH_DEAD = 50.0         # H/A <-> D, both directions (v1: 25)
MIN_POSSESSION = 20        # frames (0.8 s) an H or A stretch must last (v1: none)
MIN_DEAD = 1               # frames a D stretch must last: no minimum (the bench found none helps)

# The bench can't see Smart Tagging, so these keep v1's meaning at the new prices instead of being
# chosen: the tag clue still needs ~3 s of tag-only evidence to flip possession, and the set-piece
# window weighs the same against the dead-ball price as in v1. Reasoning, not fitting.
W_TAG_TEAM = v1.W_TAG_TEAM * SWITCH_TEAM / v1.SWITCH_TEAM
W_SET_PIECE = v1.W_SET_PIECE * SWITCH_DEAD / v1.SWITCH_TO_DEAD


def build_costs(clues):
    """v2's costs, with the two rescaled weights in place of v1's.

    Built as v2's costs plus the difference, so that with v1's prices (difference zero) the costs,
    and the output, are exactly v2's.
    """
    cost = v1.build_costs(clues)
    cost[:, v1.H] += v2.W_WHISTLE * clues["whistle"]
    cost[:, v1.A] += v2.W_WHISTLE * clues["whistle"]
    cost[:, v1.A] += (W_TAG_TEAM - v1.W_TAG_TEAM) * clues["tag_supports"][:, HOME]
    cost[:, v1.H] += (W_TAG_TEAM - v1.W_TAG_TEAM) * clues["tag_supports"][:, AWAY]
    set_piece = (W_SET_PIECE - v1.W_SET_PIECE) * clues["set_piece"]
    cost[:, v1.H] += set_piece
    cost[:, v1.A] += set_piece
    return cost


def switch_matrix():
    switch = np.zeros((3, 3))
    switch[v1.H, v1.A] = switch[v1.A, v1.H] = SWITCH_TEAM
    switch[v1.H, v1.D] = switch[v1.A, v1.D] = SWITCH_DEAD
    switch[v1.D, v1.H] = switch[v1.D, v1.A] = SWITCH_DEAD
    return switch


def predict(tracking, tags, events, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH)):
    """Labels for every video frame: H/A/D on graded frames, X elsewhere."""
    halves = halves or graded_halves()
    clues = v1.compute_clues(tracking, tags, halves, pitch)
    clues["whistle"] = v2.whistle_mask(events, len(clues["out"]))
    cost = build_costs(clues)
    switch = switch_matrix()
    labels = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in halves:
        path = viterbi_min_length(cost[first:last + 1], switch,
                                  [MIN_POSSESSION, MIN_POSSESSION, MIN_DEAD])
        labels[first:last + 1] = v1.LABELS[path]
    return labels, clues


def compare(labels, before):
    """What changed against v2: frames, and possessions (no ground truth involved)."""
    graded = labels != "X"
    changed = (labels != before) & graded
    print("frames that differ from v2: {} ({:.2f}% of graded)".format(
        changed.sum(), 100 * changed[graded].mean()))
    for name, values in (("v2", before), ("v3", labels)):
        lengths = []
        for _, first, last in graded_halves():
            for label in "HA":
                starts, ends = v1.runs(values[first:last + 1] == label)
                lengths += list(ends - starts)
        lengths = np.array(lengths) / v1.FPS
        print("  {}: possessions {}  median {:.1f} s  mean {:.1f} s  under 1 s {}  dead {:.1f}%".format(
            name, len(lengths), np.median(lengths), lengths.mean(), int((lengths < 1).sum()),
            100 * np.mean(values[graded] == "D")))


def main():
    tracking = load_tracking()
    tags = load_tags()
    events = whistle_events(tracking)
    labels, clues = predict(tracking, tags, events)
    v1.write_submission(OUTPUT, labels)
    print("wrote " + os.path.relpath(OUTPUT, ROOT))
    print("settings: team {} dead {} min possession {} min dead {}; tag weight {:.4f}, set-piece {:.4f}".format(
        SWITCH_TEAM, SWITCH_DEAD, MIN_POSSESSION, MIN_DEAD, W_TAG_TEAM, W_SET_PIECE))
    v1.describe(labels, clues)
    before, _ = v2.predict(tracking, tags, events)
    compare(labels, before)


if __name__ == "__main__":
    main()
