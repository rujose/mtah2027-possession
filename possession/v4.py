"""Model v4: v3 + a dead-ball probability learned from the players on the bench.

    python -m possession.v4            # writes evaluation/results/v4.csv and prints diagnostics

Full explanation: docs/models/v4-dead-ball.md. The model (possession/models/) was trained on the
10 bench matches only, by `python -m bench.train_dead_ball`, which also chose the four settings
below (bench/results/dead_ball_choice.json). v1's "stopped" and "active" rules are replaced by the
model; everything else is v3's.
"""

import os

import numpy as np

from possession import dead_ball, features, v1, v2, v3
from possession.audio import whistle_events
from possession.data import (AWAY, FPS, HOME, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves,
                             load_tags, load_tracking)
from possession.decoder import viterbi_min_length

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v4.csv")

# Chosen on the bench: leave-one-match-out over all 10 matches, v3's tie rule, logistic
# regression unless the trees beat it by more than 0.01 (docs/models/v4-dead-ball.md, piece 4).
# All 10 folds picked this same setting; the trees beat logistic regression by 0.013 held out.
FAMILY = "trees"
W_DEAD_BALL = 0.5          # weight of the model's costs
SWITCH_DEAD = 50.0         # H/A <-> D, both directions (as v3)
MIN_DEAD = 1               # frames a D stretch must last: no minimum (as v3)

# Unchanged from v3: one change at a time.
SWITCH_TEAM = v3.SWITCH_TEAM
MIN_POSSESSION = v3.MIN_POSSESSION
W_TAG_TEAM = v3.W_TAG_TEAM
# Keeps v1's meaning at this dead-ball price, as in v3 (reasoning, not fitting).
W_SET_PIECE = v1.W_SET_PIECE * SWITCH_DEAD / v1.SWITCH_TO_DEAD
WHISTLE_CHECK = 3 * FPS    # frames after a whistle within which a stoppage counts as seen


def build_costs(clues, p):
    """v1's costs without "stopped" and "active", plus v2's whistle, v3's tag weight, the set-piece
    weight at this dead-ball price, and the model's costs."""
    without = dict(clues, stopped=np.zeros_like(clues["stopped"]), active=np.zeros_like(clues["active"]))
    cost = v1.build_costs(without)
    cost[:, v1.H] += v2.W_WHISTLE * clues["whistle"]
    cost[:, v1.A] += v2.W_WHISTLE * clues["whistle"]
    cost[:, v1.A] += (W_TAG_TEAM - v1.W_TAG_TEAM) * clues["tag_supports"][:, HOME]
    cost[:, v1.H] += (W_TAG_TEAM - v1.W_TAG_TEAM) * clues["tag_supports"][:, AWAY]
    set_piece = (W_SET_PIECE - v1.W_SET_PIECE) * clues["set_piece"]
    cost[:, v1.H] += set_piece
    cost[:, v1.A] += set_piece
    alive, dead = dead_ball.costs(p)
    cost[:, v1.H] += W_DEAD_BALL * alive
    cost[:, v1.A] += W_DEAD_BALL * alive
    cost[:, v1.D] += W_DEAD_BALL * dead
    return cost


def switch_matrix():
    switch = np.zeros((3, 3))
    switch[v1.H, v1.A] = switch[v1.A, v1.H] = SWITCH_TEAM
    switch[v1.H, v1.D] = switch[v1.A, v1.D] = SWITCH_DEAD
    switch[v1.D, v1.H] = switch[v1.D, v1.A] = SWITCH_DEAD
    return switch


def predict(tracking, tags, events, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH), build=None):
    """Labels for every video frame (H/A/D on graded frames, X elsewhere), the clues, and P(dead).

    `build` replaces build_costs (v5 passes its own); None means v4's.
    """
    halves = halves or graded_halves()
    clues = v1.compute_clues(tracking, tags, halves, pitch)
    clues["whistle"] = v2.whistle_mask(events, len(clues["out"]))
    names, X = features.frame_features(tracking, halves, pitch)
    model = dead_ball.load(FAMILY)
    if model["features"] != names:
        raise ValueError("the saved model's features differ from possession/features.py")
    p = dead_ball.predict(model, X)
    cost = (build or build_costs)(clues, p)
    switch = switch_matrix()
    labels = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in halves:
        path = viterbi_min_length(cost[first:last + 1], switch, [MIN_POSSESSION, MIN_POSSESSION, MIN_DEAD])
        labels[first:last + 1] = v1.LABELS[path]
    return labels, clues, p


def compare(labels, before, events, names=("v3", "v4")):
    """Against the earlier version, with no ground truth: possessions, dead ball, and the whistle check."""
    graded = labels != "X"
    changed = (labels != before) & graded
    print("frames that differ from the earlier version: {} ({:.2f}% of graded)".format(changed.sum(), 100 * changed[graded].mean()))
    for name, values in zip(names, (before, labels)):
        lengths, stoppages = [], []
        for _, first, last in graded_halves():
            half = values[first:last + 1]
            for label in "HA":
                starts, ends = v1.runs(half == label)
                lengths += list(ends - starts)
            starts, ends = v1.runs(half == "D")
            stoppages += [e - s for s, e in zip(starts, ends) if s > 0]
        lengths, stoppages = np.array(lengths) / FPS, np.array(stoppages) / FPS
        seen = {kind: sum(1 for e in events if (kind == "all" or e["counted"])
                          and (values[e["start"]:e["start"] + WHISTLE_CHECK] == "D").any())
                for kind in ("all", "counted")}
        print("  {}: possessions {}  median {:.1f} s  mean {:.1f} s  dead {:.1f}%  stoppages {} (median {:.1f} s)  "
              "whistles followed by D within 3 s: {}/{} (counted {}/{})".format(
                  name, len(lengths), np.median(lengths), lengths.mean(), 100 * np.mean(values[graded] == "D"),
                  len(stoppages), np.median(stoppages), seen["all"], len(events), seen["counted"],
                  sum(e["counted"] for e in events)))


def main():
    tracking = load_tracking()
    tags = load_tags()
    events = whistle_events(tracking)
    labels, clues, p = predict(tracking, tags, events)
    v1.write_submission(OUTPUT, labels)
    print("wrote " + os.path.relpath(OUTPUT, ROOT))
    print("settings: model {} weight {} dead price {} min dead {}; team {} min possession {}".format(
        FAMILY, W_DEAD_BALL, SWITCH_DEAD, MIN_DEAD, SWITCH_TEAM, MIN_POSSESSION))
    graded = labels != "X"
    print("P(dead) over graded frames: mean {:.3f}; above 0.5 on {:.1f}%".format(
        p[graded].mean(), 100 * np.mean(p[graded] > 0.5)))
    before, _ = v3.predict(tracking, tags, events)
    compare(labels, before, events)


if __name__ == "__main__":
    main()
