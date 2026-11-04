"""Checks on the lab candidate `learned` (docs/lab/learned.md, "Where the gain is"): bench only.

    python -m bench.lab_learned_check [--set all|idsse]   # ~3 min; after `python -m bench.lab_learned`

Uses the held-out P(H / A / D) that `bench.lab_learned` saved per match (each from a model trained
without that match), so nothing is refitted.

1. The ball in dead time, on each bench source: how often it is missing, and when seen, how often it
   stands still and lies outside the pitch. These are properties of each provider's feed.
2. Per-frame accuracy with the ball seen and missing, against v7.
3. The gain split in two (diagnostics, chosen after round 1, reported as such): the learned model's
   evidence for dead vs live with v7's team evidence ("dead_only"), and v7's dead vs live with the
   learned team evidence ("team_only"). Each decoded over a small grid, then v9 + v11, and judged
   leave one match out like any candidate.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from bench import chain, lab
from bench import lab_learned as L
from bench.tune_decoder import decode as decode_costs
from possession import lab_learned_features as llf
from possession.data import FPS

# Params
STILL = 0.3                       # m/s over 0.2 s: the jitter alone gives ~0.06; a ball being carried > 0.5
PARTS = ("team_only", "dead_only")
SWITCH_TEAM = (0.5, 1.0, 2.0, 4.0, 8.0)
SWITCH_DEAD = (50.0, 25.0, 10.0)
COLUMNS = ["part", "switch_team", "switch_dead"]


def held_out_p(match, which, variant="all"):
    return np.load(os.path.join(L.folder(which), "p_{}_{}.npy".format(variant, match.id))).astype(np.float64)


def split_costs(match, p, part):
    """v7's costs with one half of the evidence replaced by the learned model's.

    team_only: dead vs live from v7 (D keeps v7's cost; both teams get v7's cheaper team cost), plus
    the learned -log P(team | live), shifted so the likelier team pays nothing.
    dead_only: dead vs live from the learned model (-log P(D), -log (1 - P(D))), plus v7's team
    preference (each team's v7 cost above the cheaper of the two)."""
    c = match.pred["v7_cost"].astype(np.float64)
    out = np.empty_like(c)
    if part == "team_only":
        live = np.minimum(c[:, 0], c[:, 1])
        q = p[:, 0] / (p[:, 0] + p[:, 1])
        t = np.stack([-np.log(q), -np.log(1 - q)], axis=1)
        t -= t.min(axis=1, keepdims=True)
        out[:, 0], out[:, 1], out[:, 2] = live + t[:, 0], live + t[:, 1], c[:, 2]
    else:
        pref = c[:, :2] - np.minimum(c[:, 0], c[:, 1])[:, None]
        live_cost = -np.log(np.clip(1 - p[:, 2], *L.CLIP))
        out[:, 0], out[:, 1] = live_cost + pref[:, 0], live_cost + pref[:, 1]
        out[:, 2] = -np.log(p[:, 2])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=chain.SETS, default="all", dest="which")
    which = parser.parse_args().which
    rows, lines = [], []
    print("1-2. match | dead frames: ball missing | seen: still / outside | live frames: missing | seen: still | "
          "accuracy learned (argmax) ball seen / missing | v7 seen / missing")
    for match in lab.each(which):
        p = held_out_p(match, which)
        ball = match.track["ball"]
        missing = np.isnan(ball[:, 0])
        speed = np.full(len(ball), np.nan)
        speed[:-llf.LOOK] = np.hypot(*(ball[llf.LOOK:] - ball[:-llf.LOOK]).T) * (FPS / llf.LOOK)
        length, width = match.pitch
        with np.errstate(invalid="ignore"):
            outside = (ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width)
            still = speed < STILL
        dead, live = match.truth == 2, np.isin(match.truth, (0, 1))
        graded = dead | live
        argmax = np.array(["H", "A", "D"])[p.argmax(axis=1)]

        def acc(letters, mask):
            m = mask & graded
            return float((letters[m] == match.truth_letters[m]).mean())

        known = ~np.isnan(speed)
        print("{} ({}) | {:.1%} | {:.1%} / {:.1%} | {:.1%} | {:.1%} | {:.3f} / {:.3f} | {:.3f} / {:.3f}".format(
            match.id, match.source, missing[dead].mean(), still[dead & known].mean(), outside[dead & ~missing].mean(),
            missing[live].mean(), still[live & known].mean(), acc(argmax, ~missing), acc(argmax, missing),
            acc(match.v7, ~missing), acc(match.v7, missing)), flush=True)
        for part, team, dead_price in itertools.product(PARTS, SWITCH_TEAM, SWITCH_DEAD):
            at_v7 = decode_costs(split_costs(match, p, part), match.halves, (team, dead_price, 20, L.MIN_DEAD))
            s = lab.score(match, L.finish(match, at_v7, which))
            rows.append({"match": match.id, "part": part, "switch_team": team, "switch_dead": dead_price,
                         "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "d_f1": s["d_f1"],
                         "team_acc_alive": s["team_acc_alive"], "possessions": s["possessions"]})
    grid = pd.DataFrame(rows)
    tag = chain.suffix(which)
    grid.to_csv(os.path.join(lab.RESULTS, "lab_learned_split_grid{}.csv".format(tag)), index=False, float_format="%.5f")
    print("3. the gain split in two (diagnostic, after round 1):")
    for part in PARTS:
        sub = grid[grid["part"] == part].reset_index(drop=True)
        print("[{}] ".format(part) + lab.describe(lab.judge(sub, COLUMNS, which)).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
