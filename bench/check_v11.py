"""v11 end to end on a bench match: it runs, and the move keeps the teams and the possessions.

    python -m bench.check_v11          # ~3 min; prints one line per bench set

v11 is v10 plus the move (possession/v11.py), and Claude never runs it on our match, so this runs it
on J03WR9 (the IDSSE match PathCRF never saw) with each bench set's settings and checks what the page
promises: only frames next to a change of team differ from v10, the sequence of possessions (their
teams, in order) is the same, and a second run of v10 plus the move gives the same labels.
"""

import numpy as np

from bench import degrade, sets
from bench.train_restart import prepare_level
from possession import v10, v11

# Params
MATCH = "J03WR9"
LEVEL = 41


def possessions(labels):
    """The team of each possession, in order."""
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    firsts = labels[np.r_[0, edges]]
    return firsts[np.isin(firsts, ["H", "A"])]


def main():
    stats = degrade.load_stats()
    data = {m: sets.load(m) for m in sets.matches("all")}
    prepared, scale = prepare_level(LEVEL, list(data), data, stats)
    track = degrade.tracking(MATCH, data[MATCH], LEVEL, prepared[MATCH], scale[sets.source(MATCH)], stats)
    halves, pitch = data[MATCH]["halves"], data[MATCH]["pitch"]
    for which in ("all", "idsse"):
        setting = v11.SETTINGS[which]
        before, base = v10.predict(track, [], halves, pitch, bench_set=which)
        after, info = v11.predict(track, [], halves, pitch, bench_set=which)
        again, _ = v11.move(before, base["fixed"], halves, setting["dv"], setting["back"], setting["ahead"])
        differ = np.flatnonzero(before != after)
        window = int(max(setting["back"], setting["ahead"]) * v11.FPS)
        changes = np.concatenate([v11.team_changes(before, first, last) for _, first, last in halves])
        near = np.all(np.min(np.abs(differ[:, None] - changes[None]), axis=1) <= window) if len(differ) else True
        same = np.array_equal(possessions(before), possessions(after))
        print("{} ({}): {} changes moved, {} frames differ from v10, all within the window of a change: {}; "
              "same possessions: {}; same labels twice: {}".format(
                  MATCH, which, len(info["touch_moves"]), len(differ), near, same, np.array_equal(after, again)), flush=True)


if __name__ == "__main__":
    main()
