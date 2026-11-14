"""The stoppage switch with v14's branch A: does it still pass the bar it passed with v13's (docs/lab/gate.md)?

    python -m bench.check_v14_switch chain --variant normal|p10|p25|p50|stressed   # one process per variant, ~5 min
    python -m bench.check_v14_switch report

The switch picks, per match, branch A or branch B (`augment`'s chain, unchanged) from a signal with a fixed
threshold (bench/lab_gate.py, THRESHOLD 0.1445, set with v13's A). v14 changes A (the GRU weighted 0.8, no snap or
move). This recomputes A's chain on the gate's bench variants exactly as lab_gate did for v13's A (held-out P on
each variant, `average`'s fold pick, every post stage refitted out of fold), keeps lab_gate's B scores and signals,
and applies the bar's parts 1-3 (`all` only: v14 is v13 on `idsse`). It also recomputes the threshold rule with
the new A, to see whether the fixed threshold still sits where the rule would put it. Nothing reads our match.
"""

import argparse
import json
import os

import numpy as np

from bench import lab, lab_carried, lab_gate, lab_mix
from possession import v13, v14

# Params
WHICH = "all"
RESULTS = os.path.join(lab.RESULTS, "check_v14_switch")


def chain(variant):
    """v14's branch A on one bench variant: {match: spellF1} (one variant per process: lab_short's cache is keyed by
    match id)."""
    lab_mix.cache_short_evidence()
    lab_gate.own_dead_cache()
    matches = [lab_gate.variant_match(m, WHICH, variant) for m in lab.matches(WHICH)]
    picks = lab_mix.fold_settings(WHICH)
    start = lab_carried.average_picks(WHICH)
    letters = {}
    for m in matches:
        p = v14.mix(lab_gate.held_out_p("gru", variant, m.id, WHICH), lab_gate.held_out_p("robust", variant, m.id, WHICH))
        decoded = v13.L.decode_cost(m, -np.log(p), **start[m.id])
        if v14.SNAP:
            decoded = v13.lab_robust.finish(m, decoded, WHICH)
        letters[m.id] = np.asarray(decoded).astype("<U1")
    for stage in v13.CHAIN[WHICH]:
        letters = lab_mix.add_stage(stage, letters, matches, picks)
    scores = {m.id: lab.score(m, letters[m.id])["spell_f1"] for m in matches}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "A_{}.json".format(variant)), "w") as handle:
        json.dump(scores, handle, indent=1)
    print("{}: v14's A {:.4f} (v13's A {:.4f})".format(
        variant, np.mean(list(scores.values())), np.mean(list(lab_gate.final_scores(variant, WHICH)["A"].values()))), flush=True)


def report():
    sig = lab_gate.signal_table(WHICH)
    ms = lab.matches(WHICH)
    scores, rows = {}, []
    for v in lab_gate.VARIANTS:
        with open(os.path.join(RESULTS, "A_{}.json".format(v))) as handle:
            a = json.load(handle)
        old = lab_gate.final_scores(v, WHICH)
        scores[v] = {"A": a, "B": old["B"]}
        picks = {m: lab_gate.picks_b(sig[v][m]) for m in ms}            # the deployed threshold
        gate = {m: old["B"][m] if picks[m] else a[m] for m in ms}
        mean = lambda d: float(np.mean([d[m] for m in ms]))
        rows.append({"variant": v, "A v13": mean(old["A"]), "A v14": mean(a), "B": mean(old["B"]), "switch": mean(gate),
                     "picks B": sum(picks.values()), "switch - better branch": mean(gate) - max(mean(a), mean(old["B"])),
                     "switch - A v14": mean(gate) - mean(a)})
    import pandas as pd
    table = pd.DataFrame(rows).set_index("variant")
    print(table.round(4).to_string())
    on_all, gains, sigs = lab_gate.threshold_rule(ms, sig, scores)
    left_out = [lab_gate.threshold_rule([o for o in ms if o != m], sig, scores)[0] for m in ms]
    print("\nthe threshold rule with v14's A: {:.4f} on all 10 (deployed 0.1445); left out: {:.4f} to {:.4f}".format(
        on_all, min(left_out), max(left_out)))
    print("mean B - A by variant: " + ", ".join("{} {:+.4f}".format(v, g) for v, g in zip(lab_gate.VARIANTS, gains)))
    checks = {"part 1 (normal: no loss)": bool(table.loc["normal", "switch - A v14"] >= -0.002),
              "part 2 (stressed: B on 9+, gain +0.02)": bool(table.loc["stressed", "picks B"] >= 9 and table.loc["stressed", "switch - A v14"] >= 0.02),
              "part 3 (partial: within 0.005)": {v: bool(table.loc[v, "switch - better branch"] >= -0.005) for v in ("p10", "p25", "p50")}}
    print(json.dumps(checks, indent=1))
    with open(os.path.join(RESULTS, "report.json"), "w") as handle:
        json.dump({"table": table.reset_index().to_dict(orient="records"), "threshold_rule_all": on_all,
                   "threshold_left_out": left_out, "checks": checks}, handle, indent=1, default=float)


def skillcorner():
    """Part 4: v14's branch A on SkillCorner's 20 matches (deployment models, from lab_carried's cached inputs), beside
    branch B's scores from bench/check_v13_switch.py (B is unchanged in v14)."""
    import pandas as pd
    from bench import lab_skillcorner, skillcorner as sc
    b = pd.read_csv(os.path.join(lab.RESULTS, "check_v13_switch.csv"), dtype={"match": str}).set_index("match")
    models = v14.load_models(WHICH)
    rows = []
    for m in sc.match_ids():
        gt = sc.load(m)["labels"]
        match, X, Xs = lab_carried.sc_prepared(m)
        p = v14.mix(v13.gru_proba(match, models["gru"], WHICH, None, (list(models["robust"]["names"]), X, Xs)),
                    v13.v12.robust_proba(match, models["robust"], WHICH))
        a = lab_skillcorner.score(gt, v14.run_chain_from(match, p, models, WHICH))["spell_f1"]
        rows.append({"match": m, "A v14": a, "A v13": b.loc[m, "A"], "B": b.loc[m, "B"], "branch": b.loc[m, "branch"]})
        print("{}: v14's A {:.4f} (v13's {:.4f}), B {:.4f}".format(m, a, b.loc[m, "A"], b.loc[m, "B"]), flush=True)
    f = pd.DataFrame(rows)
    gain = f["B"] - f["A v14"]
    print("SkillCorner: v14's A {:.4f} (v13's {:.4f}); the switch picks B on {} of 20: {:.4f}, +{:.4f} over v14's A, better on {} "
          "of 20 (part 4 needs +0.02 on 15)".format(f["A v14"].mean(), f["A v13"].mean(), int((f["branch"] == "B").sum()),
                                                   f["B"].mean(), gain.mean(), int((gain > 0).sum())))
    f.to_csv(os.path.join(RESULTS, "skillcorner.csv"), index=False, float_format="%.5f")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("chain", "report", "skillcorner"))
    parser.add_argument("--variant", choices=lab_gate.VARIANTS)
    args = parser.parse_args()
    if args.step == "skillcorner":
        return skillcorner()
    return chain(args.variant) if args.step == "chain" else report()


if __name__ == "__main__":
    main()
