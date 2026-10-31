"""Choose v3's four decoder settings on the bench (docs/models/v3-decoder.md, piece 2).

    python -m bench.tune_decoder        # ~30 min on 4 processes; writes bench/results/decoder_*.csv/json

1. v1's clues (without tags: the bench has none) once per bench match at 41 % damage.
2. Every combination in the grid decoded and scored on every match, raw and merged labels.
3. Leave-one-match-out: for each match, the combination picked on the other 6 is scored on it.
4. The submission's setting: picked the same way on all 7.
Picking = best mean spellF1, then among those within TIE of it, the one finding the fewest
possessions (decision 1: the cautious choice), then grid order.
"""

import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import argparse

from bench import degrade, labels, sets
from bench.run import score
from possession import v1
from possession.data import ROOT
from possession.decoder import viterbi_min_length

# Params
TEAM_PRICES = (0.25, 0.5, 1, 2, 4, 8, 15)
DEAD_PRICES = (5, 10, 25, 50)
MIN_POSSESSION = (1, 5, 10, 20)        # frames
MIN_DEAD = (1, 25, 50)                 # frames
LEVEL = 41                             # decision 2: choose on our damage level only
CHECK_LEVELS = (30, 55)
TIE = 0.01                             # spellF1: differences this small are noise
WORKERS = 4
V1_SETTING = (15, 25, 1, 1)
V3_SETTING = (0.5, 50, 20, 1)          # what this search chose on IDSSE (possession/v3.py)
RESULTS = os.path.join(ROOT, "bench", "results")


def result_paths(which):
    """IDSSE keeps the original names (v3 was chosen there); other sets get a suffix."""
    suffix = "" if which == "idsse" else "_" + which
    return (os.path.join(RESULTS, "decoder_grid{}.csv".format(suffix)),
            os.path.join(RESULTS, "decoder_choice{}.json".format(suffix)))

_work = {}


def switch_matrix(team, dead):
    switch = np.zeros((3, 3))
    switch[v1.H, v1.A] = switch[v1.A, v1.H] = team
    switch[v1.H, v1.D] = switch[v1.A, v1.D] = switch[v1.D, v1.H] = switch[v1.D, v1.A] = dead
    return switch


def decode(cost, halves, setting):
    team, dead, k_poss, k_dead = setting
    out = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in halves:
        path = viterbi_min_length(cost[first:last + 1], switch_matrix(team, dead),
                                  [k_poss, k_poss, k_dead])
        out[first:last + 1] = v1.LABELS[path]
    return out


def bench_costs(level, which):
    """{match: (cost, halves)} with v1's clues on the damaged tracking, plus the two label sets.
    Each source is calibrated on its own, as in bench/run.py."""
    stats = degrade.load_stats()
    matches = sets.matches(which)
    data = {m: sets.load(m) for m in matches}
    prepared = {m: degrade.prepare(m, data[m], level, stats) for m in matches}
    scale = {name: degrade.calibrate([prepared[m] for m in matches if sets.source(m) == name], level)
             for name in {sets.source(m) for m in matches}}
    costs, truth = {}, {}
    for m in matches:
        track = degrade.tracking(m, data[m], level, prepared[m], scale[sets.source(m)], stats)
        clues = v1.compute_clues(track, [], data[m]["halves"], data[m]["pitch"])
        costs[m] = (v1.build_costs(clues), data[m]["halves"])
        raw = sets.truth(m, data[m])
        truth[m] = {"raw": raw, "merged": labels.merge_short(raw)}
    return costs, truth


def _init(costs, truth):
    _work["costs"], _work["truth"] = costs, truth


def _score_setting(args):
    level, setting = args
    rows = []
    for m, (cost, halves) in _work["costs"].items():
        letters = decode(cost, halves, setting)
        for variant, gt in _work["truth"][m].items():
            rows.append({"level": level, "team_price": setting[0], "dead_price": setting[1],
                         "min_possession": setting[2], "min_dead": setting[3],
                         "labels": variant, "match": m, **score(gt, letters)})
    return rows


def run_grid(level, settings, which):
    costs, truth = bench_costs(level, which)
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(costs, truth)) as pool:
        rows = [row for result in pool.map(_score_setting, [(level, s) for s in settings])
                for row in result]
    return pd.DataFrame(rows)


SETTING = ["team_price", "dead_price", "min_possession", "min_dead"]


def is_setting(frame, setting):
    """Rows of `frame` with exactly this setting."""
    return (frame[SETTING].to_numpy(dtype=float) == np.asarray(setting, dtype=float)).all(axis=1)


def pick(table):
    """table: one row per setting with mean spell_f1 and possessions over the matches used.
    Best mean, then fewest possessions within TIE of it, then grid order (table order)."""
    best = table["spell_f1"].max()
    close = table[table["spell_f1"] >= best - TIE]
    return close.sort_values("possessions", kind="stable").iloc[0]


def summarise(frame, matches):
    """Mean spellF1 and possessions per setting over `matches` (raw labels), in grid order."""
    used = frame[(frame["labels"] == "raw") & frame["match"].isin(matches)]
    return used.groupby(SETTING, sort=False)[["spell_f1", "possessions"]].mean().reset_index()


def mean_for(frame, setting, variant):
    """Mean spellF1 over the matches for one setting and label set."""
    rows = frame[(frame["labels"] == variant) & is_setting(frame, setting)]
    return float(rows["spell_f1"].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=sets.SETS, default="idsse", dest="which")
    which = parser.parse_args().which
    GRID_CSV, CHOICE_JSON = result_paths(which)
    settings = list(itertools.product(TEAM_PRICES, DEAD_PRICES, MIN_POSSESSION, MIN_DEAD))
    print("grid: {} settings x {} matches at {} % damage".format(
        len(settings), len(sets.matches(which)), LEVEL), flush=True)
    grid = run_grid(LEVEL, settings, which)
    os.makedirs(os.path.dirname(GRID_CSV), exist_ok=True)
    grid.to_csv(GRID_CSV, index=False, float_format="%.4f")
    matches = sorted(grid["match"].unique())

    # Leave one match out.
    folds = []
    for held_out in matches:
        chosen = pick(summarise(grid, [m for m in matches if m != held_out]))
        setting = tuple(chosen[SETTING])
        row = grid[(grid["labels"] == "raw") & (grid["match"] == held_out)
                   & is_setting(grid, setting)].iloc[0]
        v1_row = grid[(grid["labels"] == "raw") & (grid["match"] == held_out)
                      & is_setting(grid, V1_SETTING)].iloc[0]
        folds.append({"held_out": held_out, "setting": [float(v) for v in setting],
                      "held_out_spell_f1": float(row["spell_f1"]),
                      "v1_spell_f1": float(v1_row["spell_f1"]),
                      "possessions": int(row["possessions"]),
                      "true_possessions": int(row["true_possessions"])})
        print("fold {}: picked {} -> held-out spellF1 {:.3f} (v1 {:.3f}), possessions {} / {}".format(
            held_out, setting, row["spell_f1"], v1_row["spell_f1"], int(row["possessions"]),
            int(row["true_possessions"])), flush=True)
    held = np.array([f["held_out_spell_f1"] for f in folds])
    base = np.array([f["v1_spell_f1"] for f in folds])
    print("held-out spellF1 {:.3f} +- {:.3f}; v1 {:.3f}; better than v1 on {}/{}".format(
        held.mean(), held.std(ddof=1), base.mean(), int((held > base).sum()), len(held)))
    fixed = {}
    for name, fixed_setting in (("v1", V1_SETTING), ("v3", V3_SETTING)):
        rows = grid[(grid["labels"] == "raw") & is_setting(grid, fixed_setting)].set_index("match")
        fixed[name] = rows["spell_f1"].round(4).to_dict()
        print("fixed {} setting {}: mean spellF1 {:.3f}, possessions {:.0f} / {:.0f}".format(
            name, fixed_setting, rows["spell_f1"].mean(), rows["possessions"].mean(),
            rows["true_possessions"].mean()))

    # The submission's setting, and the plain best for transparency.
    everything = summarise(grid, matches)
    chosen = pick(everything)
    plain = everything.sort_values("spell_f1", ascending=False, kind="stable").iloc[0]
    setting = tuple(float(v) for v in chosen[SETTING])
    print("chosen on all {}: {} mean spellF1 {:.3f}, possessions {:.0f}; plain best {} {:.3f}".format(
        len(matches), setting, chosen["spell_f1"], chosen["possessions"], tuple(plain[SETTING]),
        plain["spell_f1"]))

    checks = {"merged_labels": {"chosen": mean_for(grid, setting, "merged"),
                                "v1": mean_for(grid, V1_SETTING, "merged")}}
    for level in CHECK_LEVELS:
        other = run_grid(level, [setting, V1_SETTING], which)
        checks["level_{}".format(level)] = {"chosen": mean_for(other, setting, "raw"),
                                            "v1": mean_for(other, V1_SETTING, "raw")}
    print("checks: {}".format(checks))

    choice = {"setting": dict(zip(SETTING, setting)),
              "bench_mean_spell_f1": float(chosen["spell_f1"]),
              "bench_mean_possessions": float(chosen["possessions"]),
              "plain_best": {"setting": dict(zip(SETTING, (float(v) for v in plain[SETTING]))),
                             "spell_f1": float(plain["spell_f1"])},
              "fixed_settings_spell_f1": fixed,
              "folds": folds, "held_out_mean": float(held.mean()),
              "held_out_sd": float(held.std(ddof=1)), "checks": checks}
    with open(CHOICE_JSON, "w") as handle:
        json.dump(choice, handle, indent=1)
    print("wrote {} and {}".format(os.path.relpath(GRID_CSV, ROOT), os.path.relpath(CHOICE_JSON, ROOT)))


if __name__ == "__main__":
    sys.exit(main())
