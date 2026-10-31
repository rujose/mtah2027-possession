"""Model v6: v5 + who restarts, learned from the players' layout at stoppages.

    python -m possession.v6            # writes evaluation/results/v6.csv and prints diagnostics

Full explanation: docs/models/v6-restart-team.md. The restart model (possession/models/restart_*)
was trained on the 10 bench matches only, by `python -m bench.train_restart`, which also chose the
three settings below (bench/results/restart_choice.json). Everything else is v5's.
"""

import os

import numpy as np

from possession import dead_ball, restart, v1, v4, v5
from possession.audio import whistle_events
from possession.data import (FPS, HOME, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags,
                             load_tracking)

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v6.csv")

# Chosen on the bench: leave-one-match-out over all 10 matches, v3's tie rule, logistic
# regression unless the trees beat it by more than 0.01 (docs/models/v6-restart-team.md, piece 4).
# The trees were ahead by only 0.004 held out, so logistic regression stays.
FAMILY = "logistic"
W_RESTART = 1.0            # weight of the restart clue
CARRY = 3.0                # seconds the clue lasts after the last dead-looking frame
TAG_WINDOW = 15 * FPS      # frames after a set-piece tag starts in which the cross-check looks


def predict(tracking, tags, events, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH)):
    """Labels for every video frame, the clues, P(dead) and P(home restarts)."""
    halves = halves or graded_halves()
    model = dead_ball.load(FAMILY, name="restart")
    p_home = restart.p_home(model, restart.all_orientations(tracking, halves, pitch))

    def build(clues, p_dead):
        cost = v5.build_costs(clues, p_dead)
        on_h, on_a = restart.costs(restart.restart_belief(p_home, p_dead, halves, int(CARRY * FPS)), W_RESTART)
        cost[:, v1.H] += on_h
        cost[:, v1.A] += on_a
        return cost

    labels, clues, p_dead = v4.predict(tracking, tags, events, halves, pitch=pitch, build=build)
    return labels, clues, p_dead, p_home


def first_team_after_stoppages(values):
    """{frame where a stoppage ends: the label that follows it} for one label array."""
    out = {}
    for _, first, last in graded_halves():
        starts, ends = v1.runs(values[first:last + 1] == "D")
        for s, e in zip(starts, ends):
            if s > 0 and first + e <= last:
                out[first + e] = values[first + e]
    return out


def compare(labels, before, p_home, p_dead, tags):
    """Against v5, with no ground truth: the team after each stoppage, and Smart Tagging's
    set-piece team as a cross-check (automatic tags, not ground truth)."""
    mine, theirs = first_team_after_stoppages(labels), first_team_after_stoppages(before)
    common = sorted(set(mine) & set(theirs))
    print("stoppage ends at the same frame in v5 and v6: {}; the team after it differs on {}".format(
        len(common), sum(mine[f] != theirs[f] for f in common)))
    agree = {"model": 0, "v5": 0, "v6": 0}
    checked = 0
    for tag in tags:
        side = v1.TEAM_NAMES.get(tag.get("Team"))
        if tag["code"] != "SET PIECES" or side is None:
            continue
        window = slice(tag["start"], tag["start"] + TAG_WINDOW)
        dead_looking = p_dead[window] > restart.DEAD_LOOKING
        if not dead_looking.any():
            continue
        checked += 1
        agree["model"] += int((p_home[window][dead_looking].mean() > 0.5) == (side == HOME))
        for name, values in (("v5", before), ("v6", labels)):
            part = values[window]
            dead = np.flatnonzero(part == "D")
            after = part[dead[-1] + 1:] if len(dead) else part
            live = after[np.isin(after, ["H", "A"])]
            agree[name] += int(len(live) > 0 and live[0] == ("H" if side == HOME else "A"))
    print("set-piece tags with a team and a dead-looking stretch: {}; agreeing with the tag's team: "
          "restart model {}, v5's first team after it {}, v6's {}".format(checked, agree["model"], agree["v5"], agree["v6"]))


def main():
    tracking = load_tracking()
    tags = load_tags()
    events = whistle_events(tracking)
    labels, clues, p_dead, p_home = predict(tracking, tags, events)
    v1.write_submission(OUTPUT, labels)
    print("wrote " + os.path.relpath(OUTPUT, ROOT))
    print("settings: restart model {} weight {} carry {} s".format(FAMILY, W_RESTART, CARRY))
    before, _, _ = v5.predict(tracking, tags, events)
    v4.compare(labels, before, events, names=("v5", "v6"))
    compare(labels, before, p_home, p_dead, tags)


if __name__ == "__main__":
    main()
