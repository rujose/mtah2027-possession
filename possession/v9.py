"""Model v9: v8, with each possession after a stoppage starting at the restart pass.

    python -m possession.v9            # writes evaluation/results/v9.csv and prints diagnostics

Full explanation: docs/models/v9-restart-pass.md. After v8 decodes, each change from dead to a team
moves to the earliest pass by that team seen from BACK seconds before the change to AHEAD seconds
after (possession/passes.py). A dead frame becomes the team's, or a team frame becomes dead, only
between the change and the pass: the team never changes, no possession is created or removed, and
with no pass seen nothing moves. BACK and AHEAD were chosen by `python -m bench.restart_timing`
(bench/results/restart_timing_choice.json).
"""

import os

import numpy as np

from possession import passes, v4, v8
from possession.audio import whistle_events
from possession.data import AWAY, FPS, HOME, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking
from possession.v1 import write_submission

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v9.csv")
# All 10 folds picked this window (held-out spellF1 0.561 against v7's 0.556, better on 10/10).
BACK = 5               # s before the change to look for the pass
AHEAD = 1              # s after


def seen_passes(tracking):
    """{HOME: bool [frames], AWAY: ...}: where a seen pass by each team starts."""
    return {side: passes.pass_starts(tracking["players"], tracking["team"], tracking["ball"], side)
            for side in (HOME, AWAY)}


def snap(labels, tracking, halves, back=None, ahead=None, starts=None):
    """(labels with each D -> team change moved to its restart pass, [(change, pass) frames])."""
    back = BACK if back is None else back
    ahead = AHEAD if ahead is None else ahead
    starts = seen_passes(tracking) if starts is None else starts
    out = labels.copy()
    moves = []
    for _, first, last in halves:
        changes = np.flatnonzero((labels[first:last] == "D") & np.isin(labels[first + 1:last + 1], ["H", "A"])) + first + 1
        for g in changes:
            team_label = labels[g]
            lo, hi = max(g - int(back * FPS), first), min(g + int(ahead * FPS), last)
            found = np.flatnonzero(starts[HOME if team_label == "H" else AWAY][lo:hi + 1])
            if len(found) == 0:
                continue
            k = lo + found[0]                     # the earliest: the restart is the first pass
            if k < g and np.all(labels[k:g] == "D"):
                out[k:g] = team_label
            elif k > g:
                other = np.flatnonzero(labels[g:last + 1] != team_label)
                if k >= g + (other[0] if len(other) else last + 1 - g):
                    continue                      # the pass is after the possession ended
                out[g:k] = "D"
            else:
                continue
            moves.append((int(g), int(k)))
    return out, moves


def predict(tracking, tags, events, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH)):
    """v8's outputs with the labels snapped, then the moves: labels, clues, P(dead), P(home restarts),
    PathCRF's P(home), the repaired tracking, the keeper pieces, [(change, pass)]."""
    halves = halves or graded_halves()
    labels, clues, p_dead, p_start, p_pathcrf, fixed, pieces = v8.predict(tracking, tags, events, halves, pitch)
    snapped, moves = snap(labels, fixed, halves)
    return snapped, clues, p_dead, p_start, p_pathcrf, fixed, pieces, moves


def ball_at_starts(labels, ball):
    """At D -> team changes with the ball seen around them: shares where it's leaving, arriving, and
    already above 8 m/s in the 2 s before (no labels needed)."""
    speed = np.r_[np.nan, np.hypot(*np.diff(ball, axis=0).T) * FPS]
    counts = np.zeros(4)
    for g in np.flatnonzero((labels[:-1] == "D") & np.isin(labels[1:], ["H", "A"])) + 1:
        before, after = speed[max(g - FPS // 2, 0):g], speed[g:g + FPS // 2]
        if g < 2 * FPS or np.isnan(before).all() or np.isnan(after).all():
            continue
        b, a = np.nanmean(before), np.nanmean(after)
        counts += (1, a > b + 2, b > a + 2, np.nanmax(speed[g - 2 * FPS:g]) > 8)
    return {"seen": int(counts[0]), "leaving": counts[1] / counts[0], "arriving": counts[2] / counts[0],
            "fast_before": counts[3] / counts[0]}


def main():
    tracking = load_tracking()
    tags = load_tags()
    events = whistle_events(tracking)
    halves = graded_halves()
    before, _, _, _, _, fixed, _ = v8.predict(tracking, tags, events, halves)
    labels, moves = snap(before, fixed, halves)
    write_submission(OUTPUT, labels)
    print("wrote " + os.path.relpath(OUTPUT, ROOT))
    print("settings: look for the pass from {} s before the change to {} s after".format(BACK, AHEAD))

    n_changes = int(np.sum((before[:-1] == "D") & np.isin(before[1:], ["H", "A"])))
    shift = np.array([(k - g) / FPS for g, k in moves])
    print("dead -> team changes: {}; moved to a seen pass: {} ({:.0%}), earlier {}, later {}; "
          "shift median {:+.2f} s, quartiles {:+.2f} / {:+.2f} s".format(
              n_changes, len(moves), len(moves) / n_changes, np.sum(shift < 0), np.sum(shift > 0),
              np.median(shift), *np.percentile(shift, [25, 75])))
    for name, values in (("v8", before), ("v9", labels)):
        b = ball_at_starts(values, fixed["ball"])
        print("  {}: ball at the starts (seen around {}): leaving {:.0%}, arriving {:.0%}, "
              "already above 8 m/s in the 2 s before {:.0%}".format(name, b["seen"], b["leaving"], b["arriving"], b["fast_before"]))
    v4.compare(labels, before, events, names=("v8", "v9"))
    print("largest moves (frame of the change -> frame of the pass), to watch in the viewer:")
    for g, k in sorted(moves, key=lambda m: -abs(m[1] - m[0]))[:10]:
        print("  {} -> {} ({:+.1f} s, {})".format(g, k, (k - g) / FPS, before[g]))


if __name__ == "__main__":
    main()
