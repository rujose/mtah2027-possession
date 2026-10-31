"""Model v7: v6 + PathCRF's estimate of who has the ball, from the players' movement alone.

    python -m possession.v7            # writes evaluation/results/v7.csv and prints diagnostics

Full explanation: docs/models/v7-pathcrf.md. PathCRF (third_party/pathcrf, MPL-2.0, trained by its
authors on other matches) runs over the stretches v6 calls live; its P(home has the ball) becomes a
clue in which only the unlikely team pays, by the odds. The two settings below were chosen by
`python -m bench.eval_pathcrf` on the 4 bench matches PathCRF never saw
(bench/results/pathcrf_choice.json). Everything else is v6's.
"""

import os

import numpy as np

from possession import pathcrf_adapter, restart, v1, v4, v6
from possession.audio import whistle_events
from possession.data import PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v7.csv")
# All 4 folds over the unseen matches picked this (held-out spellF1 0.565 against v6's 0.501, 4/4).
W_PATHCRF = 0.5            # weight of the clue
WHERE = "live"             # "gaps": only while the ball is missing; "live": wherever it answers


def pathcrf_belief(p_pathcrf, ball):
    """The clue's R: PathCRF's P(home), rounded (reproducibility), silent where WHERE excludes."""
    belief = np.round(p_pathcrf, 3)
    if WHERE == "gaps":
        belief[~np.isnan(ball[:, 0])] = np.nan
    return belief


def predict(tracking, tags, events, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH)):
    """Labels for every video frame, the clues, P(dead), P(home restarts) and PathCRF's P(home)."""
    halves = halves or graded_halves()
    first, _, _, p_start = v6.predict(tracking, tags, events, halves, pitch)
    live = np.isin(first, ["H", "A"])
    p_pathcrf = pathcrf_adapter.p_home(pathcrf_adapter.load_model(), tracking, halves, pitch, live)
    belief = pathcrf_belief(p_pathcrf, tracking["ball"])

    def build(clues, p_dead):
        cost = v6.build_costs(clues, p_dead, p_start, halves)
        on_h, on_a = restart.costs(belief, W_PATHCRF)
        cost[:, v1.H] += on_h
        cost[:, v1.A] += on_a
        return cost

    labels, clues, p_dead = v4.predict(tracking, tags, events, halves, pitch=pitch, build=build)
    return labels, clues, p_dead, p_start, p_pathcrf


def main():
    tracking = load_tracking()
    tags = load_tags()
    events = whistle_events(tracking)
    labels, clues, p_dead, p_start, p_pathcrf = predict(tracking, tags, events)
    v1.write_submission(OUTPUT, labels)
    print("wrote " + os.path.relpath(OUTPUT, ROOT))
    print("settings: PathCRF weight {} where {}".format(W_PATHCRF, WHERE))
    graded = labels != "X"
    answered = ~np.isnan(p_pathcrf) & graded
    print("PathCRF answered on {:.1%} of graded frames ({:.1%} of those with the ball missing)".format(
        answered.mean() / graded.mean(), np.mean(np.isnan(tracking["ball"][answered, 0]))))
    before, _, _, _ = v6.predict(tracking, tags, events)
    v4.compare(labels, before, events, names=("v6", "v7"))


if __name__ == "__main__":
    main()
