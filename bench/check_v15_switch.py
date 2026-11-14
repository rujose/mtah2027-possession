"""The stoppage switch with v15's branch B: does it still pass the bar it passed in round 7 (docs/lab/gate.md)?

    python -m bench.search_b confirm --trials N                     # B's 3 variants (the B bar)
    python -m bench.search_b confirm --trials N --variants normal p10   # the other two (~15 min)
    python -m bench.search_b skillcorner --trials N
    python -m bench.check_v15_switch --trial N

The switch picks, per match, branch A or branch B from a signal with a fixed threshold (bench/lab_gate.py,
THRESHOLD 0.1445, set with v13's two branches). v15 keeps v14's A and the threshold and changes B to search trial N
(docs/lab/branch-b.md). This takes:
- branch A: v14's, on every bench variant, as bench/check_v14_switch.py scored it (every post stage refitted out of
  fold on the same variant: lab_gate's way, which flatters A a little on the stressed variants);
- branch B: trial N as it deploys (fixed settings, its stage models trained without the match on the trial's own
  data), from search_b confirm on all 5 variants. Trial 0 there is v13's B run the same way; it sits up to 0.004
  below lab_gate's B, which decoded each match with its fold's own pick and refitted every stage on the same
  variant's letters (the lab's protocol, the one A's scores come from: this check is a little harder on B than A);
- SkillCorner: v14's A from check_v14_switch, the new B from search_b skillcorner (deployment models).
Then the bar's four parts (`all` only: v15 is v14 on `idsse`), and the round-7 threshold rule recomputed with the new B
(information: the threshold stays). Nothing reads our match.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from bench import lab, lab_gate

# Params
WHICH = "all"
RESULTS = os.path.join(lab.RESULTS, "check_v15_switch")
SEARCH = os.path.join(lab.RESULTS, "search_b")
V14 = os.path.join(lab.RESULTS, "check_v14_switch")


def b_scores(trial):
    """{variant: {match: spellF1}} for trial `trial` and for trial 0, from search_b confirm's files."""
    frames = [pd.read_csv(os.path.join(SEARCH, f)) for f in sorted(os.listdir(SEARCH))
              if f.startswith("confirm_") and f.endswith(".csv")]
    df = pd.concat(frames).drop_duplicates(["trial", "match", "variant"])
    out = {}
    for t in (trial, 0):
        d = df[df.trial == t]
        out[t] = {v: dict(zip(d[d.variant == v].match, d[d.variant == v].spell_f1)) for v in lab_gate.VARIANTS}
        missing = [v for v in lab_gate.VARIANTS if len(out[t][v]) != len(lab.matches(WHICH))]
        if missing:
            raise SystemExit("trial {}: no confirm scores on {} (run search_b confirm --trials {} --variants {})".format(
                t, missing, trial, " ".join(missing)))
    return out[trial], out[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    ms = lab.matches(WHICH)
    sig = lab_gate.signal_table(WHICH)
    b_new, b_zero = b_scores(args.trial)
    mean = lambda d: float(np.mean([d[m] for m in ms]))
    rows, scores = [], {}
    for v in lab_gate.VARIANTS:
        with open(os.path.join(V14, "A_{}.json".format(v))) as handle:
            a = json.load(handle)
        old_b = lab_gate.final_scores(v, WHICH)["B"]
        scores[v] = {"A": a, "B": b_new[v]}
        picks = {m: lab_gate.picks_b(sig[v][m]) for m in ms}                 # the deployed threshold
        gate = {m: b_new[v][m] if picks[m] else a[m] for m in ms}
        rows.append({"variant": v, "A v14": mean(a), "B v13 (lab_gate)": mean(old_b), "B v13 (as deployed)": mean(b_zero[v]),
                     "B v15": mean(b_new[v]), "switch": mean(gate), "picks B": sum(picks.values()),
                     "switch - better branch": mean(gate) - max(mean(a), mean(b_new[v])), "switch - A": mean(gate) - mean(a),
                     "B v15 better than A on": int(sum(b_new[v][m] > a[m] for m in ms))})
    table = pd.DataFrame(rows).set_index("variant")
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string())
    sc_a = pd.read_csv(os.path.join(V14, "skillcorner.csv"), dtype={"match": str}).set_index("match")
    sc_b = pd.read_csv(os.path.join(SEARCH, next(f for f in sorted(os.listdir(SEARCH)) if f.startswith("skillcorner_")
                                                 and str(args.trial) in f[len("skillcorner_"):-4].split("_"))),
                       dtype={"match": str})
    sc_b = sc_b[sc_b.trial == args.trial].set_index("match").spell_f1
    sc_gain = sc_b - sc_a["A v14"].reindex(sc_b.index)
    picks_sc = int((sc_a["branch"].reindex(sc_b.index) == "B").sum())
    print("\nSkillCorner: v14's A {:.4f}, v15's B {:.4f}: {:+.4f}, better on {}/20; the switch picks B on {} of 20".format(
        sc_a["A v14"].mean(), sc_b.mean(), sc_gain.mean(), int((sc_gain > 0).sum()), picks_sc))
    on_all, gains, sigs = lab_gate.threshold_rule(ms, sig, scores)
    left_out = [lab_gate.threshold_rule([o for o in ms if o != m], sig, scores)[0] for m in ms]
    print("\nthe round-7 threshold rule with v15's B: {:.4f} on all 10 (deployed {}); left out: {:.4f} to {:.4f}".format(
        on_all, lab_gate.THRESHOLD, min(left_out), max(left_out)))
    print("mean B - A by variant: " + ", ".join("{} {:+.4f}".format(v, g) for v, g in zip(lab_gate.VARIANTS, gains)))
    checks = {
        "part 1 (normal: A everywhere or no loss beyond 0.002)": bool(table.loc["normal", "picks B"] == 0
                                                                       or table.loc["normal", "switch - A"] >= -0.002),
        "part 2 (stressed: B on 9+, gain +0.02)": bool(table.loc["stressed", "picks B"] >= 9
                                                        and table.loc["stressed", "switch - A"] >= 0.02),
        "part 3 (partial: within 0.005 of the better branch)": {v: bool(table.loc[v, "switch - better branch"] >= -0.005)
                                                                for v in ("p10", "p25", "p50")},
        "part 4 (SkillCorner: +0.02, 15/20 better)": bool(sc_gain.mean() >= 0.02 and (sc_gain > 0).sum() >= 15)}
    print(json.dumps(checks, indent=1))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "report_{}.json".format(args.trial)), "w") as handle:
        json.dump({"trial": args.trial, "table": table.reset_index().to_dict(orient="records"),
                   "skillcorner": {"A v14": float(sc_a["A v14"].mean()), "B v15": float(sc_b.mean()),
                                   "gain": float(sc_gain.mean()), "better": int((sc_gain > 0).sum()), "picks B": picks_sc},
                   "threshold_rule_all": on_all, "threshold_left_out": left_out,
                   "mean_b_minus_a": dict(zip(lab_gate.VARIANTS, gains)), "checks": checks}, handle, indent=1, default=float)


if __name__ == "__main__":
    main()
