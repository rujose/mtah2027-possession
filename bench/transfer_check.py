"""Label-free checks that v12 behaves on our match as it does on the bench (docs/reproduce.md).

    python -m bench.transfer_check           # the bench's ranges (v12 out of fold, damaged tracking)
    python -m bench.transfer_check --ours    # Jose runs it: our match against them, PASS / NEAR / WARN only

The bench can't tell us how v12 scores on our match, and other teams found bench gains that didn't
carry over. These three numbers need no truth, so they can be read on our match too:

- **dead share:** the share of graded frames v12 calls D;
- **possessions per minute:** v12's possessions per graded minute;
- **clear control:** where the ball is seen and slow at one player's feet, with nobody from the other team
  close, how often v12 names that player's team (frames v12 calls H or A).

Each bench number comes from v12's out-of-fold letters on the damaged tracking, the models' view. --ours
prints only whether each number sits inside the bench's range (PASS), just outside it (NEAR: within a
quarter of the range's width), or further (WARN), so no statistic of our match is shown.
"""

import argparse
import json
import os

import numpy as np

from bench import chain, lab, train_v12
from possession import touches
from possession.data import AWAY, FPS, HOME, ROOT, graded_halves, load_tracking

# Params (fixed before looking at our match)
REACH = 1.0          # m: the player at the ball (bench/labels.py's reading check)
SLOW = 2.0           # m/s: a ball this slow is at someone's feet
LOOK = 5             # frames: the ball's speed over 0.2 s, so the tracking's jitter doesn't read as movement
CLEAR = 3.0          # m: nobody from the other team this close
NEAR = 0.25          # NEAR: outside the range by at most this share of its width
OURS = os.path.join(ROOT, "evaluation", "results", "v12.csv")
RANGES = os.path.join(lab.RESULTS, "transfer_ranges.json")


def stats(letters, track, halves):
    """{dead share, possessions per minute, clear control} for one match."""
    letters = np.asarray(letters).astype("<U1")
    graded = np.zeros(len(letters), bool)
    for _, first, last in halves:
        graded[first:last + 1] = True
    possessions = 0
    for _, first, last in halves:
        w = letters[first:last + 1]
        possessions += int(np.sum(np.r_[True, w[1:] != w[:-1]] & np.isin(w, ["H", "A"])))
    ball = track["ball"]
    speed = np.full(len(ball), np.nan)
    speed[:-LOOK] = np.hypot(*(ball[LOOK:] - ball[:-LOOK]).T) * (FPS / LOOK)
    near = {s: touches.nearest(track["players"], track["team"], ball, s) for s in (HOME, AWAY)}
    with np.errstate(invalid="ignore"):
        home = (near[HOME] < REACH) & (near[AWAY] > CLEAR) & (speed < SLOW)
        away = (near[AWAY] < REACH) & (near[HOME] > CLEAR) & (speed < SLOW)
    called = graded & np.isin(letters, ["H", "A"])
    agree = np.sum(called & home & (letters == "H")) + np.sum(called & away & (letters == "A"))
    return {"dead share": float(np.mean(letters[graded] == "D")),
            "possessions per minute": possessions / (graded.sum() / FPS / 60),
            "clear control": float(agree / max(np.sum(called & (home | away)), 1))}


def bench_ranges():
    rows = {}
    for m in lab.each("all"):
        letters = np.load(os.path.join(train_v12.OOF + chain.suffix("all"), "{}.npy".format(m.id)))
        rows[m.id] = stats(letters, m.track, m.halves)
        truth = m.truth_letters.astype("<U1")
        rows[m.id].update({"truth: " + k: v for k, v in stats(truth, m.track, m.halves).items() if k != "clear control"})
        print("{}: {}".format(m.id, {k: round(v, 3) for k, v in rows[m.id].items()}), flush=True)
    keys = ("dead share", "possessions per minute", "clear control")
    ranges = {k: [min(r[k] for r in rows.values()), max(r[k] for r in rows.values())] for k in keys}
    with open(RANGES, "w") as handle:
        json.dump({"ranges": ranges, "per_match": rows}, handle, indent=1)
    print("bench ranges (v12 out of fold): {}".format({k: [round(a, 3), round(b, 3)] for k, (a, b) in ranges.items()}))


def ours():
    with open(RANGES) as handle:
        ranges = json.load(handle)["ranges"]
    letters = np.loadtxt(OURS, delimiter=",", skiprows=1, usecols=1, dtype="<U1")
    found = stats(letters, load_tracking(), graded_halves())
    for k, (lo, hi) in ranges.items():
        width = hi - lo
        x = found[k]
        verdict = "PASS" if lo <= x <= hi else "NEAR" if lo - NEAR * width <= x <= hi + NEAR * width else "WARN"
        print("{}: {}".format(k, verdict))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ours", action="store_true", help="our match against the bench's ranges: PASS / NEAR / WARN")
    if parser.parse_args().ours:
        return ours()
    bench_ranges()


if __name__ == "__main__":
    main()
