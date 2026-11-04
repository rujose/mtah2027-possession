"""Model v11: v10, with each change of team moved onto the new team's touch.

    python -m possession.v11           # Jose runs it on our match: writes evaluation/results/v11.csv

Full explanation: docs/models/v11-touch-line.md. After v10 (and v9's restart snap), each change from
one team to the other moves to the nearest touch by the new team seen from BACK seconds before the
change to AHEAD seconds after (possession/touches.py). Frames change hands only between the change
and the touch, and only inside the two possessions it separates: the teams and the number of
possessions never change, and with no touch seen nothing moves. ΔV and the window were chosen by
`python -m bench.touch_timing` (bench/results/touch_timing_choice[_idsse].json).
"""

import hashlib
import os

import numpy as np

from possession import touches, v1, v10
from possession.data import AWAY, FPS, HOME, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v11.csv")
# Per bench set, as v10's BENCH_SET switch: ΔV (m/s) and the window (s before, s after the change).
# all: 8 of 10 folds picked this (held-out spellF1 0.571 against v9's 0.561, better on 10/10).
# idsse: 4 of 7 folds picked this, 6 of 7 the 2 s back (0.561 against 0.550, better on 7/7).
SETTINGS = {
    "all": {"dv": 3.0, "back": 1.0, "ahead": 1.0},
    "idsse": {"dv": 3.0, "back": 2.0, "ahead": 1.0},
}
SIDE = {"H": HOME, "A": AWAY}


def seen_touches(tracking, dv):
    """{HOME: bool [frames], AWAY: ...}: each team's seen touches."""
    return {side: touches.touches(tracking["players"], tracking["team"], tracking["ball"], side, dv)
            for side in (HOME, AWAY)}


def team_changes(labels, first, last):
    """Frames in [first, last] where the labels go from one team straight to the other."""
    a, b = labels[first:last], labels[first + 1:last + 1]
    return np.flatnonzero(np.isin(a, ["H", "A"]) & np.isin(b, ["H", "A"]) & (a != b)) + first + 1


def move(labels, tracking, halves, dv, back, ahead, found=None):
    """(labels with each H <-> A change moved to the new team's nearest touch, [(change, touch)])."""
    found = seen_touches(tracking, dv) if found is None else found
    out = labels.copy()
    moves = []
    for _, first, last in halves:
        for g in team_changes(labels, first, last):
            old, new = labels[g - 1], labels[g]
            lo, hi = max(g - int(back * FPS), first), min(g + int(ahead * FPS), last)
            seen = np.flatnonzero(found[SIDE[new]][lo:hi + 1]) + lo
            if len(seen) == 0:
                continue
            k = seen[np.argmin(np.abs(seen - g))]      # the nearest; a tie goes to the earlier
            if k < g and k > first and np.all(out[k - 1:g] == old):
                out[k:g] = new                         # the old team keeps at least the frame before
            elif k > g and np.all(out[g:k + 1] == new):
                out[g:k] = old                         # the touch is inside the new team's possession
            else:
                continue
            moves.append((int(g), int(k)))
    return out, moves


def predict(tracking, tags, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH), bench_set=None):
    """(labels for every video frame, v10's info plus "touch_moves": [(change, touch)])."""
    halves = halves or graded_halves()
    setting = SETTINGS[bench_set or v10.BENCH_SET]
    labels, info = v10.predict(tracking, tags, halves, pitch, bench_set)
    labels, info["touch_moves"] = move(labels, info["fixed"], halves, setting["dv"], setting["back"], setting["ahead"])
    return labels, info


def main():
    """Only what Jose needs to check the file, as v10: no statistics about the match."""
    labels, _ = predict(load_tracking(), load_tags())
    v1.write_submission(OUTPUT, labels)
    with open(OUTPUT, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    path = os.path.relpath(OUTPUT, ROOT)
    print("wrote {} (bench set {}, restart snap {})".format(path, v10.BENCH_SET, "on" if v10.RESTART_SNAP else "off"))
    print("SHA-256 {}".format(digest))
    print("format check: .venv/bin/python evaluation/validate_submission.py {}".format(path))


if __name__ == "__main__":
    main()
