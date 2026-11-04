"""What each bench stage hands to the next, per set of matches (docs/models/v10-clean-base.md).

    python -m bench.train_dead_ball --set idsse    # then train_restart, eval_pathcrf, restart_timing

"all" is the 10 matches v4-v9 were built on: its settings are those versions' constants, so the
scripts behave exactly as before. "idsse" is the 7 IDSSE matches only (CC BY 4.0), in case Metrica's
samples can't be used: each stage reads the previous stage's choice from its `*_idsse.json` and
writes its own files with the same suffix, never touching the "all" ones.
"""

import argparse
import json
import os

from bench import sets
from possession import v3, v4, v6, v7
from possession.data import ROOT

# Params
SETS = ("all", "idsse")
RESULTS = os.path.join(ROOT, "bench", "results")


def which_set(description):
    """The --set argument of a bench stage (default "all")."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--set", choices=SETS, default="all", dest="which")
    return parser.parse_args().which


def suffix(which):
    """Added to every file a stage writes: nothing for "all", "_idsse" for "idsse"."""
    return "" if which == "all" else "_" + which


def matches(which):
    return sets.matches(which)


def unseen(which, pathcrf_unseen):
    """The bench matches PathCRF never saw, within this set."""
    return tuple(m for m in pathcrf_unseen if m in matches(which))


def _choice(stage, which):
    with open(os.path.join(RESULTS, "{}_choice{}.json".format(stage, suffix(which)))) as handle:
        return json.load(handle)


def dead_ball(which):
    """v4's family, weight, dead price and minimum dead stretch."""
    if which == "all":
        return {"family": v4.FAMILY, "weight": v4.W_DEAD_BALL, "dead_price": v4.SWITCH_DEAD, "min_dead": v4.MIN_DEAD}
    c = _choice("dead_ball", which)
    return {"family": c["family"], "weight": c["weight"], "dead_price": c["dead_price"], "min_dead": c["min_dead"]}


def decoder(which):
    """(switch team, switch dead, min possession, min dead): v3's team settings (chosen on IDSSE
    alone), v4's dead-ball ones."""
    d = dead_ball(which)
    return (v3.SWITCH_TEAM, d["dead_price"], v3.MIN_POSSESSION, d["min_dead"])


def restart(which):
    """v6's family, weight and carry (s)."""
    if which == "all":
        return {"family": v6.FAMILY, "weight": v6.W_RESTART, "carry": v6.CARRY}
    c = _choice("restart", which)
    return {"family": c["family"], "weight": c["weight"], "carry": c["carry_s"]}


def pathcrf(which):
    """v7's weight and where the clue applies."""
    if which == "all":
        return {"weight": v7.W_PATHCRF, "where": v7.WHERE}
    c = _choice("pathcrf", which)["setting"]
    return {"weight": c["weight"], "where": c["where"]}


def predictions(which):
    """Where eval_pathcrf leaves its out-of-sample predictions (gitignored)."""
    return os.path.join(ROOT, "data", "external", "bench_predictions" + suffix(which))
