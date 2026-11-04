"""Lab candidate `decoder`: the decoder's team settings re-chosen, with touches as evidence (docs/lab/decoder.md).

    python -m bench.lab_decoder [--set all|idsse]   # ~5 min; after `python -m bench.lab --build`;
                                                     # writes bench/results/lab_decoder[_round1|_part1]*

v3 chose the minimum possession (20 frames) and the team switch price (0.5) on v1's clues, before
the dead-ball model, the restart model, PathCRF and v11's touches existed. This decodes v7's
out-of-sample costs again with each setting in GRID, then applies v9's restart snap and v11's touch
move exactly as the harness does (bench/lab.py `build`), and scores the result.

Part 1: the team settings alone (minimum possession x team switch price).
Part 2: the same settings with touches as evidence. Before decoding, each seen touch by team T
(v11's detector, v11's ΔV) makes T cheaper by TOUCH_WEIGHT on the touch frame. One frame only: the
minimum length makes any T stretch covering the touch at least `min_possession` long, and v11 then
puts its start on the touch.

Both parts were fixed before any result was seen and judged as one grid (round 1). Its best minimum
sat on the grid's edge, so round 2 extends that axis; the whole grid is judged again (lab.judge,
leave one match out against v11). Round 1 alone and Part 1 alone are reported beside it.
"""

import argparse
import itertools

import numpy as np
import pandas as pd

from bench import chain, lab
from bench.tune_decoder import decode as decode_costs
from possession import v1, v9, v11
from possession.data import AWAY, HOME

# Params (the page gives the why of each)
STAGE = "decode"                         # replaces v7's decoding; v9's snap and v11's move follow
# Round 1, fixed before any result was seen.
SWITCH_TEAM = (0.5, 0.25, 1.0)          # H <-> A price; v3's 0.5 first
MIN_POSSESSION = (20, 10, 15, 30)        # frames (0.4-1.2 s); v3's 20 first
TOUCH_WEIGHT = (0.0, 1.0, 2.0, 4.0)      # per seen touch, on the touch frame; 0 = Part 1
# Round 2, added after round 1: its best minimum (30) sat on the grid's edge with the score still
# rising, so the one axis with a slope at the edge goes three steps further. The whole grid is judged.
MIN_POSSESSION_2 = (40, 50, 60)          # frames (1.6-2.4 s)
COLUMNS = ["switch_team", "min_possession", "touch_weight"]


def _cross(teams, ks, weights):
    return [dict(zip(COLUMNS, (team, k, w))) for w in weights for team, k in itertools.product(teams, ks)]


ROUND_1 = _cross(SWITCH_TEAM, MIN_POSSESSION, TOUCH_WEIGHT)
GRID = ROUND_1 + _cross(SWITCH_TEAM, MIN_POSSESSION_2, TOUCH_WEIGHT)
assert GRID[0] == {"switch_team": 0.5, "min_possession": 20, "touch_weight": 0.0}   # current first
# Round 1's choice on every match of each set (bench/results/lab_decoder_round1[_idsse].json), for
# mixing if round 1 is accepted. The whole grid (rounds 1 + 2) chose (0.5, 40, 4) on "all" and FAILS
# the bar there (6/10, IDSSE down); (0.25, 40, 4) on "idsse", where it passes. See the page.
CHOSEN = {"all": {"switch_team": 1.0, "min_possession": 30, "touch_weight": 4.0},
          "idsse": {"switch_team": 0.5, "min_possession": 30, "touch_weight": 2.0}}


def evidence(match, which="all"):
    """The match's seen passes (v9) and touches (v11), computed once and kept on the match."""
    if getattr(match, "_decoder_evidence", None) is None:
        match._decoder_evidence = {"passes": v9.seen_passes(match.track),
                                   "touches": v11.seen_touches(match.track, v11.SETTINGS[which]["dv"])}
    return match._decoder_evidence


def touch_costs(cost, found, weight):
    """v7's costs with each seen touch by a team `weight` cheaper for that team, on its frame."""
    out = np.asarray(cost, dtype=np.float64).copy()
    out[found[HOME], v1.H] -= weight
    out[found[AWAY], v1.A] -= weight
    return out


def decode(match, switch_team, min_possession, touch_weight=0.0, which="all"):
    """Letters at v7's level (before v9's snap and v11's move). Dead-ball settings stay chain's.

    With the chain's own team settings and no touch weight this returns match.v7 itself: the saved
    costs are float32, and decoding them again differs from v7 on 0-7 frames per match (ties), which
    would move the current setting's score off v11's (J03WOH: +0.0005)."""
    team_now, dead_price, k_now, min_dead = chain.decoder(which)
    if not touch_weight and switch_team == team_now and int(min_possession) == k_now:
        return match.v7.copy()
    cost = match.pred["v7_cost"]
    if touch_weight:
        cost = touch_costs(cost, evidence(match, which)["touches"], touch_weight)
    return decode_costs(cost, match.halves, (switch_team, dead_price, int(min_possession), min_dead))


def finish(match, letters, which="all"):
    """v9's restart snap, then v11's touch move, with the set's settings (as bench/lab.py `build`)."""
    found = evidence(match, which)
    back, ahead = chain.restart_snap(which)
    snapped, _ = v9.snap(letters, match.track, match.halves, back, ahead, starts=found["passes"])
    s = v11.SETTINGS[which]
    moved, _ = v11.move(snapped, match.track, match.halves, s["dv"], s["back"], s["ahead"], found=found["touches"])
    return moved


def predict(match, which="all", **setting):
    """The full chain for one match: decode, snap, move."""
    return finish(match, decode(match, which=which, **setting), which)


def run_grid(which):
    rows = []
    for match in lab.each(which):
        for setting in GRID:
            decoded = decode(match, which=which, **setting)
            final = finish(match, decoded, which)
            if setting == GRID[0] and not (np.array_equal(decoded, match.v7) and np.array_equal(final, match.v11)):
                raise RuntimeError("{}: the current settings don't reproduce v7 / v11".format(match.id))
            s = lab.score(match, final)
            rows.append({"match": match.id, "source": match.source, "unseen": match.unseen, **setting,
                         "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                         "possessions": s["possessions"], "true_possessions": s["true_possessions"],
                         "spell_f1_decoded": lab.score(match, decoded)["spell_f1"]})
        print("{}: {} settings scored".format(match.id, len(GRID)), flush=True)
    return pd.DataFrame(rows)


def possessions(grid, report):
    """Mean possessions per match: held out (each fold's pick), v11 (the current settings), true."""
    held, current, true = [], [], []
    for fold in report["folds"]:
        rows = grid[grid.match == fold["match"]]
        pick = rows[np.all([rows[c] == fold["picked"][c] for c in COLUMNS], axis=0)].iloc[0]
        base = rows[np.all([rows[c] == GRID[0][c] for c in COLUMNS], axis=0)].iloc[0]
        held.append(pick.possessions)
        current.append(base.possessions)
        true.append(base.true_possessions)
    return {"held_out": float(np.mean(held)), "v11": float(np.mean(current)), "true": float(np.mean(true))}


def judge(grid, name, which):
    report = lab.judge(grid, COLUMNS, which)
    report["possessions"] = possessions(grid, report)
    print("{} ({} settings):".format(name, grid[COLUMNS].drop_duplicates().shape[0]))
    print(lab.describe(report))
    print("  picked per fold: {}".format({f["match"]: tuple(f["picked"].values()) for f in report["folds"]}))
    print("  possessions per match: held out {held_out:.0f}, v11 {v11:.0f}, true {true:.0f}".format(**report["possessions"]))
    lab.save(name, grid, report, which)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=chain.SETS, default="all", dest="which")
    which = parser.parse_args().which
    grid = run_grid(which)
    round_1 = np.zeros(len(grid), bool)
    for setting in ROUND_1:
        round_1 |= np.all([grid[c] == setting[c] for c in COLUMNS], axis=0)
    judge(grid[round_1].reset_index(drop=True), "decoder_round1", which)
    judge(grid[round_1 & (grid.touch_weight == 0)].reset_index(drop=True), "decoder_round1_part1", which)
    judge(grid[grid.touch_weight == 0].reset_index(drop=True), "decoder_part1", which)
    report = judge(grid, "decoder", which)
    if which == "all" and report["passes"]:
        print("passes on all: now run `python -m bench.lab_decoder --set idsse`")


if __name__ == "__main__":
    main()
