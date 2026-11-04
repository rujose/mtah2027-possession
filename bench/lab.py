"""The lab: one harness for trying candidate versions on the bench (docs/lab.md).

    python -m bench.lab --build [--set all|idsse]   # ~1 min; after `python -m bench.eval_pathcrf`
    python -m bench.lab [--set all|idsse]           # v11 per match, by source: the number to beat

Every candidate is a function of one match's inputs that returns a letter for every frame. Its
settings are chosen leave-one-match-out (`judge`), and it's kept only if it clears BAR, fixed here
before any candidate ran. Each match is cached once (data/external/lab[_idsse]/, gitignored) and
loaded one at a time, so several candidates can run side by side in 8 GB.

What a candidate may use to predict a match: that match's damaged tracking and the out-of-sample
outputs of v4-v11 on it (models trained without it). Never its truth or its clean tracking: those
aren't in the cache, except `truth`, which is for scoring and for training on the *other* matches.
"""

import argparse
import json
import os
import types

import numpy as np
import pandas as pd

from bench import chain, degrade, sets
from bench.eval_pathcrf import UNSEEN
from bench.run import score as bench_score
from bench.train_restart import prepare_level
from possession import v9, v11
from possession.data import ROOT

# Params
LEVEL = 41
CACHE = os.path.join(ROOT, "data", "external", "lab")
RESULTS = os.path.join(ROOT, "bench", "results")
LETTERS = np.array(["H", "A", "D", "X"])     # scorer codes -> letters
# The bar, fixed before any candidate ran: against v11 on the same matches, each match scored with
# the setting chosen on the others.
BAR = {
    "better_share": 0.7,        # at least this share of the matches improve (7 of 10, 5 of 7)
    "each_source": True,        # the held-out mean doesn't drop on IDSSE's matches, nor on Metrica's
    "unseen": True,             # nor on the matches PathCRF never saw (J03WR9 and Metrica's 3)
}


def folder(which):
    return CACHE + chain.suffix(which)


def matches(which="all"):
    return chain.matches(which)


def build(which):
    """Cache each match's damaged tracking, truth and every out-of-sample prediction (v4-v11)."""
    stats = degrade.load_stats()
    ms = matches(which)
    data = {m: sets.load(m) for m in ms}
    prepared, scale = prepare_level(LEVEL, ms, data, stats)
    back, ahead = chain.restart_snap(which)
    setting = v11.SETTINGS[which]
    os.makedirs(folder(which), exist_ok=True)
    for m in ms:
        track = degrade.tracking(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        halves = data[m]["halves"]
        with np.load(os.path.join(chain.predictions(which), "{}.npz".format(m))) as z:
            pred = {k: z[k] for k in z.files}
        if "v7_cost" not in pred:
            raise RuntimeError("{}: re-run `python -m bench.eval_pathcrf --set {}` (it now saves the costs)".format(m, which))
        v7 = pred["v7"].astype(str)
        v9_letters, _ = v9.snap(v7, track, halves, back, ahead)
        v11_letters, _ = v11.move(v9_letters, track, halves, setting["dv"], setting["back"], setting["ahead"])
        np.savez(os.path.join(folder(which), "{}.npz".format(m)),
                 players=track["players"], ball=track["ball"], team=track["team"],
                 truth=sets.truth(m, data[m]).astype(np.int8), halves=np.array(halves), pitch=np.array(data[m]["pitch"]),
                 v9=v9_letters, v11=v11_letters, **{"pred_" + k: v for k, v in pred.items()})
        print("{}: cached".format(m), flush=True)


def load(m, which="all"):
    """One match: .id, .source, .unseen, .halves, .pitch, .frames, .track (players, ball, team, as the
    models see them), .truth (scorer codes; for scoring and training on other matches only),
    .truth_letters, .v7 / .v9 / .v11 (letters), .pred (eval_pathcrf's out-of-sample outputs:
    v4, v6, v7, p_dead, restart_clue, pathcrf_clue, v6_cost, v7_cost, p_start, p_pathcrf)."""
    with np.load(os.path.join(folder(which), "{}.npz".format(m))) as z:
        f = {k: z[k] for k in z.files}
    halves = [tuple(int(x) for x in h) for h in f["halves"]]
    pred = {k[len("pred_"):]: v for k, v in f.items() if k.startswith("pred_")}
    pred = {k: (v.astype(str) if k in ("v4", "v6", "v7") else v) for k, v in pred.items()}
    return types.SimpleNamespace(
        id=m, source=sets.source(m), unseen=m in UNSEEN, halves=halves, pitch=tuple(float(x) for x in f["pitch"]),
        frames=len(f["truth"]), track={"players": f["players"], "ball": f["ball"], "team": f["team"]},
        truth=f["truth"], truth_letters=LETTERS[f["truth"]], v7=pred["v7"], v9=f["v9"].astype(str),
        v11=f["v11"].astype(str), pred=pred)


def each(which="all"):
    """The matches one at a time (memory: ~100 MB each)."""
    for m in matches(which):
        yield load(m, which)


def score(match, letters):
    """bench.run.score: spell_f1, macro_f1, possessions, ..."""
    return bench_score(match.truth, np.asarray(letters).astype(str))


def baseline(which="all"):
    """v11 per match: {match: {"spell_f1", "macro_f1", "source", "unseen"}} (cached in bench/results)."""
    path = os.path.join(RESULTS, "lab_baseline{}.json".format(chain.suffix(which)))
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    out = {}
    for match in each(which):
        s = score(match, match.v11)
        out[match.id] = {"spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "source": match.source, "unseen": match.unseen}
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    return out


def judge(grid, columns, which="all", base=None):
    """Leave one match out over a candidate's grid, against `base` (default v11).

    grid: one row per (match, setting) with the setting's `columns`, "spell_f1" and "macro_f1".
    Each match is scored with the setting best on the others (ties: the first in grid order, so
    list the simplest settings first). Returns the folds, the means overall, by source and on the
    unseen matches, the setting chosen on all of them, and whether it clears BAR."""
    base = base or baseline(which)
    grid = grid.copy()
    grid["_order"] = np.arange(len(grid))
    settings = grid.drop_duplicates(columns)[columns + ["_order"]]
    folds = []
    for m in sorted(grid["match"].unique()):
        others = grid[grid["match"] != m].groupby(columns, sort=False).spell_f1.mean().reset_index()
        others = others.merge(settings, on=columns).sort_values("_order", kind="stable")
        best = others.loc[others["spell_f1"].round(6).idxmax()]
        pick = {c: (best[c].item() if hasattr(best[c], "item") else best[c]) for c in columns}
        row = grid[(grid["match"] == m) & np.all([grid[c] == pick[c] for c in columns], axis=0)].iloc[0]
        folds.append({"match": m, "source": base[m]["source"], "unseen": base[m]["unseen"], "picked": pick,
                      "held_out": float(row["spell_f1"]), "base": base[m]["spell_f1"],
                      "macro_held_out": float(row["macro_f1"]), "macro_base": base[m]["macro_f1"]})
    f = pd.DataFrame(folds)
    overall = grid.groupby(columns, sort=False).spell_f1.mean().reset_index().merge(settings, on=columns)
    overall = overall.sort_values("_order", kind="stable")
    chosen = overall.loc[overall["spell_f1"].round(6).idxmax()]

    def means(rows):
        return {"held_out": float(rows.held_out.mean()), "base": float(rows.base.mean()),
                "better": int((rows.held_out > rows.base).sum()), "matches": len(rows)} if len(rows) else None

    by_source = {s: means(f[f.source == s]) for s in sorted(f.source.unique())}
    unseen = means(f[f.unseen])
    report = {"folds": folds, "all": means(f),
              "macro": {"held_out": float(f.macro_held_out.mean()), "base": float(f.macro_base.mean())},
              "by_source": by_source, "unseen": unseen,
              "chosen": {c: (chosen[c].item() if hasattr(chosen[c], "item") else chosen[c]) for c in columns}}
    checks = {"mean": report["all"]["held_out"] > report["all"]["base"],
              "share": report["all"]["better"] >= int(np.ceil(BAR["better_share"] * len(f) - 1e-9))}
    if BAR["each_source"]:
        checks["each_source"] = all(v["held_out"] >= v["base"] for v in by_source.values())
    if BAR["unseen"] and unseen:
        checks["unseen"] = unseen["held_out"] >= unseen["base"]
    report["checks"], report["passes"] = checks, all(checks.values())
    return report


def describe(report):
    """The report in a few lines."""
    a, lines = report["all"], []
    lines.append("held out {:.4f} vs v11 {:.4f} ({:+.4f}), better on {}/{}; macroF1 {:.4f} vs {:.4f}".format(
        a["held_out"], a["base"], a["held_out"] - a["base"], a["better"], a["matches"],
        report["macro"]["held_out"], report["macro"]["base"]))
    for name, v in list(report["by_source"].items()) + [("PathCRF-unseen", report["unseen"])]:
        if v:
            lines.append("  {}: {:.4f} vs {:.4f} ({:+.4f}), better on {}/{}".format(
                name, v["held_out"], v["base"], v["held_out"] - v["base"], v["better"], v["matches"]))
    lines.append("  chosen on all: {}; checks {}; passes: {}".format(report["chosen"], report["checks"], report["passes"]))
    return "\n".join(lines)


def save(name, grid, report, which="all"):
    """bench/results/lab_<name>_grid[_idsse].csv and lab_<name>[_idsse].json."""
    tag = chain.suffix(which)
    grid.to_csv(os.path.join(RESULTS, "lab_{}_grid{}.csv".format(name, tag)), index=False, float_format="%.5f")
    with open(os.path.join(RESULTS, "lab_{}{}.json".format(name, tag)), "w") as handle:
        json.dump(report, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--set", choices=chain.SETS, default="all", dest="which")
    args = parser.parse_args()
    if args.build:
        build(args.which)
        path = os.path.join(RESULTS, "lab_baseline{}.json".format(chain.suffix(args.which)))
        if os.path.exists(path):
            os.remove(path)
    base = baseline(args.which)
    for m, b in base.items():
        print("{} ({}{}): v11 spellF1 {:.4f}, macroF1 {:.4f}".format(m, b["source"], ", PathCRF-unseen" if b["unseen"] else "",
                                                                    b["spell_f1"], b["macro_f1"]))
    print("mean {:.4f}".format(np.mean([b["spell_f1"] for b in base.values()])))


if __name__ == "__main__":
    main()
