"""The convention audit: v12 scored against bench labels that follow the organisers' rule of control
(docs/convention-audit.md).

    python -m bench.convention_audit        # ~10 min; after `python -m bench.train_v12 --oof-only`

The organisers' truth changes possession only when the other team gains *controlled* possession: a
touch without control, a deflection or an aerial duel won without control doesn't count, and a loose
ball stays with the last team (CHALLENGE.md). IDSSE's frame labels, which robust learned from, change
hands more often. Here those labels are made to follow the rule mechanically, on the clean tracking
(the truth's side of the bench, never a model's input): a possession whose team never has the ball at
a player's feet (within CONTROL_RADIUS, the ball slower than CONTROL_SPEED, as bench/labels.py's
reading check) for at least K frames goes to whoever had it before. Two readings of "control", fixed
before the run, bracket the rule. Then v12 after each stage, and v12 without its `short` stage, are
scored against the original labels and against both readings. IDSSE's 7 matches only: Metrica's
labels already follow the rule (bench/metrica.py), and their ball is blanked in dead time.

**Result: the control readings over-merge** (125-168 possessions per 90 minutes against ~330 in
Metrica's rules-following labels): one-touch play rarely slows the ball under 2 m/s near a player, so
they can't stand for the rule. They're kept as measured. The fair stand-in is the third truth: DFL's
own events through the recipe that follows the organisers' rules (bench/convention.py, column (b)),
whose edges carry the events' timing error (about +-1.8 s) for every version alike.
"""

import json
import os

import numpy as np
import pandas as pd

from bench import chain, convention, lab, lab_mix, labels, sets, train_v12
from bench.run import score
from possession.data import AWAY, FPS, HOME

# Params (fixed before the run)
CONTROL_RADIUS = labels.CONTROL_RADIUS     # 1 m
CONTROL_SPEED = labels.CONTROL_SPEED       # 2 m/s
READINGS = {"control 0.2 s": 5, "control 1 s": 25}   # frames at a player's feet, at least
STAGES = (("robust", "robust"), ("+ short", "short"), ("+ segments", "segments"), ("v12", "timing"))
OUT = os.path.join(lab.RESULTS, "convention_audit.json")


def at_feet(data):
    """{0: bool [frames], 1: ...} (scorer codes: 0 H, 1 A): a player of that team within
    CONTROL_RADIUS of a ball slower than CONTROL_SPEED, on the clean tracking."""
    ball, players, team = data["ball"], data["players"], np.asarray(data["team"])
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.hypot(*np.diff(ball, axis=0).T) * FPS
    with np.errstate(invalid="ignore"):
        slow = speed < CONTROL_SPEED
        out = {}
        for code, side in ((0, HOME), (1, AWAY)):
            d = np.hypot(*(players[:, team == side] - ball[:, None, :]).transpose(2, 0, 1))
            out[code] = slow & (np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1) < CONTROL_RADIUS)
    return out


def follow_rule(truth, feet, k):
    """The labels with each possession whose team never controls for k frames given to whatever came
    before it (a chain of such possessions collapses into the label before the chain). After a
    stoppage the restart's team keeps it: the rules say there is always a clear possessor."""
    out = truth.copy()
    previous = None
    for a, b, v in labels.runs_of(truth):
        if v in (0, 1) and previous in (0, 1) and int(feet[v][a:b].sum()) < k:
            out[a:b] = previous
        else:
            previous = out[a] if v != 3 else None
    return out


def possessions(codes):
    return sum(1 for _, _, v in labels.runs_of(codes) if v in (0, 1))


def main():
    lab_mix.cache_short_evidence()
    matches = list(lab.each("all"))
    picks = lab_mix.fold_settings("all")
    # v12 without `short`: robust, then segments and timing retrained on that input, out of fold.
    letters = {m.id: np.load(os.path.join(train_v12.OOF, "robust", "{}.npy".format(m.id))).astype("<U1") for m in matches}
    for stage in ("segments", "timing"):
        letters = lab_mix.add_stage(stage, letters, matches, picks)
    no_short = letters
    rows, counts = [], []
    for m in [x for x in matches if x.source == "idsse"]:
        data = sets.load(m.id)
        feet = at_feet(data)
        truths = {"IDSSE's labels": m.truth}
        truths.update({name: follow_rule(m.truth, feet, k) for name, k in READINGS.items()})
        truths["DFL events, the rules"] = convention.dfl_event_labels(m.id, data)
        minutes = np.isin(m.truth, (0, 1, 2)).sum() / FPS / 60
        counts.append({"match": m.id, **{name: possessions(t) / minutes * 90 for name, t in truths.items()},
                       **{"frames changed, " + name: int(np.sum(t != m.truth)) for name, t in truths.items() if name != "IDSSE's labels"}})
        versions = {name: np.load(os.path.join(train_v12.OOF, stage, "{}.npy".format(m.id))).astype("<U1") for name, stage in STAGES}
        versions["v12 without short"] = no_short[m.id]
        for vname, pred in versions.items():
            for tname, t in truths.items():
                s = score(t, pred)
                rows.append({"match": m.id, "version": vname, "truth": tname, "spell_f1": s["spell_f1"],
                             "possessions": s["possessions"], "true": s["true_possessions"]})
        print("{}: done".format(m.id), flush=True)
        del data
    c = pd.DataFrame(counts).set_index("match")
    print("\npossessions per 90 minutes (graded), by truth:\n" + c.round(1).to_string())
    g = pd.DataFrame(rows)
    table = g.pivot_table(index="version", columns="truth", values="spell_f1", aggfunc="mean", sort=False)
    print("\nspellF1, mean of IDSSE's 7 matches:\n" + table.round(4).to_string())
    better = {}
    for tname in table.columns:
        p = g[g.truth == tname].pivot(index="match", columns="version", values="spell_f1")
        better[tname] = {"short helps (v12 vs without)": int((p["v12"] > p["v12 without short"]).sum()),
                         "mean gain of short": float((p["v12"] - p["v12 without short"]).mean())}
    print("\n" + json.dumps(better, indent=1))
    with open(OUT, "w") as handle:
        json.dump({"possessions_per_90": c.round(2).reset_index().to_dict("records"), "spell_f1": table.round(4).reset_index().to_dict("records"),
                   "short": better}, handle, indent=1)
    print("wrote {}".format(os.path.relpath(OUT, lab.ROOT)))


if __name__ == "__main__":
    main()
