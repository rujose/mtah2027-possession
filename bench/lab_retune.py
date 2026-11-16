"""Morning 2026-11-15, builder "retune" (docs/lab/night/retune.md): B's settings re-picked for netmet5's P, leave one
match out, among a list fixed before any netmet5 score of it.

    python -m bench.lab_retune candidates            # the list: trial 1015 + round 8's top 100 (round-8 data only)
    python -m bench.lab_retune fill                  # the mirror: netmet5's P cloned in, every cache empty
    python -m bench.lab_retune check                 # trials 0 and 1015 through this file's scoring path: must equal
                                                     #   bench/results/lab_netmet/ens4/confirm_1015.csv on all 60 units
    python -m bench.lab_retune score --worker I/N    # candidates I, I+N, ...: each with its own stage models (search_b's
                                                     #   confirm protocol), on B's 30 units
    python -m bench.lab_retune skillcorner --trials N [N ...]   # netmet5's four `all` nets averaged, as the builder did
    python -m bench.lab_retune judge                 # nested picks, the night's bar against v15's B; in-sample best

The mirror is lab_netmirror's (data/external/lab_retune/ens4/, bench/results/lab_retune/ens4/), filled with APFS clones of
the network builder's netmet5 P (data/external/lab_netmet/ens4/, read only). Every candidate's out-of-fold stage tables
are computed in this mirror (no tables_trial*.joblib is reused from anywhere; trial 0's are read from the shared
search_b cache by lab_netmirror, as the builder did, and trial 0 is used only in the check).

Nothing here reads our match.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from bench import lab, lab_gate, lab_netmirror, search_b, sets

# Params
N = lab_netmirror
SB = search_b
WHICH = "all"
NAME, SUB = "retune", "ens4"
SOURCE = N.root_of("netmet", "ens4")                 # netmet5's P (read only)
BUILDER = N.results_of("netmet", "ens4")             # the builder's confirm and SkillCorner for trial 1015
TOP = 100
FIRST = 1015
CHUNK = 6                                            # trials whose stage models are held at once, per worker
OUT = os.path.join(lab.RESULTS, "lab_retune")
REF = N.REFERENCE                                    # v15's B as confirmed (round 8)
REF_SC = N.REFERENCE_SC
GAIN, BETTER, VARIANT_LOSS, SC_LOSS = N.GAIN, N.BETTER, N.VARIANT_LOSS, N.SC_LOSS


def root():
    return N.root_of(NAME, SUB)


def results():
    return N.results_of(NAME, SUB)


def candidates_path():
    return os.path.join(OUT, "candidates.json")


# --------------------------------------------------------------------------------------------
# The list (round 8's stored per-trial results for gru_b's P; nothing of netmet5)
# --------------------------------------------------------------------------------------------

def candidates():
    frames = []
    for m, variant in SB.units():
        path = os.path.join(lab.RESULTS, "search_b", "unit_{}_{}.csv".format(m, variant))
        frames.append(pd.read_csv(path).drop_duplicates("trial").assign(unit="{}_{}".format(m, variant)))
    df = pd.concat(frames, ignore_index=True)
    per_unit = df.pivot_table(index="trial", columns="unit", values="spell_f1").dropna()
    assert per_unit.shape == (SB.TRIALS, 30), per_unit.shape
    mean = per_unit.mean(axis=1)
    # descending mean; equal means in trial order (stable)
    order = sorted(mean.index, key=lambda t: (-mean.loc[t], t))
    top = [int(t) for t in order[:TOP]]
    ids = [FIRST] + [t for t in top if t != FIRST]
    out = {"rule": "trial 1015, then round 8's top {} trials by their mean over all 30 units (search_b's unit CSVs, "
                   "gru_b's P); this order is the tie-break order".format(TOP),
           "trials": ids, "count": len(ids), "first_in_top": FIRST in top, "rank_of_1015": order.index(FIRST) + 1,
           "round8_mean": {str(t): round(float(mean.loc[t]), 6) for t in ids},
           "round8_trial0": round(float(mean.loc[0]), 6)}
    os.makedirs(OUT, exist_ok=True)
    with open(candidates_path(), "w") as handle:
        json.dump(out, handle, indent=1)
    print("{} candidates; 1015 is round 8's #{}; round-8 means {:.4f} .. {:.4f}".format(
        len(ids), out["rank_of_1015"], mean.loc[ids].max(), mean.loc[ids].min()))
    return out


def load_candidates():
    with open(candidates_path()) as handle:
        return json.load(handle)["trials"]


# --------------------------------------------------------------------------------------------
# The mirror
# --------------------------------------------------------------------------------------------

def fill():
    """netmet5's P (the builder's ens4 files) cloned into this mirror; nothing else."""
    r = root()
    for d in ("lab_branch_b", "search_b", "lab_dead"):
        os.makedirs(os.path.join(r, d), exist_ok=True)
    stale = [f for f in os.listdir(os.path.join(r, "search_b")) if f.startswith("tables_trial")]
    assert not stale, "the mirror's cache is not empty: {}".format(stale)
    n = 0
    for m in lab.matches(WHICH):
        for variant in N.BENCH_VARIANTS + ("train",):
            N.clone(N.p_file(SOURCE, variant, m), N.p_file(r, variant, m))
            n += 1
    with open(os.path.join(SOURCE, "members.json")) as handle:
        members = json.load(handle)
    with open(os.path.join(r, "members.json"), "w") as handle:
        json.dump({**members, "cloned_from": SOURCE}, handle)
    print("{} P files cloned from {}; missing: {}".format(n, SOURCE, N.missing_inputs(r)))


def use():
    N.use(root(), results())
    missing = N.missing_inputs(root())
    if missing:
        raise RuntimeError("mirror incomplete: {}".format(missing[:3]))


# --------------------------------------------------------------------------------------------
# Scoring: search_b.confirm's protocol, a few trials at a time
# --------------------------------------------------------------------------------------------

def score(trial_ids, out_path, chunk=CHUNK):
    """Each trial with its own stage models (its own out-of-fold chain on its own data, without the held-out match)
    on B's 30 units, exactly as search_b.confirm; rows appended per chunk of trials."""
    use()
    SB.cache_short_evidence()
    lab_gate.own_dead_cache()
    trials = SB.every_trial()
    done = set(pd.read_csv(out_path).trial) if os.path.exists(out_path) else set()
    todo = [i for i in trial_ids if i not in done]
    for k in range(0, len(todo), chunk):
        ids = todo[k:k + chunk]
        started = time.time()
        models = {i: {m: SB.trial_models(trials[i], without=m) for m in lab.matches(WHICH)} for i in ids}
        t_models = time.time() - started
        rows = []
        for m, variant in SB.units(SB.VARIANTS):
            match = lab_gate.variant_match(m, WHICH, variant)
            for i in ids:
                t = {**trials[i], "post": "own"}
                letters = SB.letters_of(match, variant, t, {"own": models[i][m]})
                rows.append({"trial": i, "match": m, "variant": variant,
                             "spell_f1": SB.bench_score(match.truth, letters)["spell_f1"]})
            del match
        df = pd.DataFrame(rows)
        df.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False, float_format="%.5f")
        bench = df.groupby(["trial", "match"]).spell_f1.mean().groupby("trial").mean()
        print("trials {}: stage models {:.0f} s, units {:.0f} s; B's bench {}".format(
            ids, t_models, time.time() - started - t_models,
            " ".join("{}={:.4f}".format(i, bench.loc[i]) for i in ids)), flush=True)
        del models


def check():
    """Trials 0 and 1015 through score(): every one of the 60 units must equal the builder's netmet5 confirm."""
    out_path = os.path.join(results(), "check_0_1015.csv")
    if os.path.exists(out_path):
        os.remove(out_path)
    os.makedirs(results(), exist_ok=True)
    score([0, FIRST], out_path, chunk=2)
    got = pd.read_csv(out_path)
    ref = pd.read_csv(os.path.join(BUILDER, "confirm_1015.csv"))
    res = N.compare(got, ref, ["trial", "match", "variant"])
    res["exact"] = bool(res["pass"] and res["max_abs_diff"] == 0.0)
    with open(os.path.join(results(), "check.json"), "w") as handle:
        json.dump(res, handle, indent=1)
    print("CHECK {}: {}".format("PASSES" if res["exact"] else "FAILS", res), flush=True)
    return res


def worker_path(i, n):
    return os.path.join(results(), "scores_w{}of{}.csv".format(i, n))


def run(worker, n_workers):
    with open(os.path.join(results(), "check.json")) as handle:
        assert json.load(handle)["exact"], "the check has not passed: nothing else is scored"
    ids = load_candidates()[worker::n_workers]
    score(ids, worker_path(worker, n_workers))


# --------------------------------------------------------------------------------------------
# SkillCorner and the judge
# --------------------------------------------------------------------------------------------

def skillcorner(trial_ids):
    with open(os.path.join(SOURCE, "members.json")) as handle:
        nets = json.load(handle)["deployment"]
    use()
    return N.skillcorner(NAME, SUB, nets, tuple(trial_ids))


def all_scores():
    frames = [pd.read_csv(os.path.join(results(), f)) for f in sorted(os.listdir(results()))
              if f.startswith("scores_w")]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["trial", "match", "variant"])
    return df


def pick(mean, order):
    """The best mean; ties broken by the candidates' order (1015 first, then round 8's)."""
    best = mean.max()
    return next(t for t in order if mean.loc[t] == best)


def judge(sc_trial=None):
    order = load_candidates()
    df = all_scores()
    df = df[df.trial.isin(order)]
    unit = df.pivot_table(index="trial", columns=["match", "variant"], values="spell_f1")
    complete = unit.dropna()
    scored = [t for t in order if t in complete.index]
    per_match = complete.T.groupby(level="match").mean().T          # trial x match (mean of B's 3 variants)
    ms = lab.matches(WHICH)
    ref = pd.read_csv(REF)
    ref = ref[(ref.trial == FIRST) & ref.variant.isin(SB.VARIANTS)].set_index(["match", "variant"]).spell_f1
    ref_match = ref.groupby(level="match").mean()
    ref_variant = ref.groupby(level="variant").mean()
    src = pd.Series({m: sets.source(m) for m in ms})

    picks = []
    for m in ms:
        others = [o for o in ms if o != m]
        t = pick(per_match.loc[scored, others].mean(axis=1), scored)
        picks.append({"held_out": m, "trial": int(t), "score": float(per_match.loc[t, m]), "v15": float(ref_match[m]),
                      "trial1015": float(per_match.loc[FIRST, m]) if FIRST in scored else None,
                      **{v: float(complete.loc[t, (m, v)]) for v in SB.VARIANTS}})
    p = pd.DataFrame(picks).set_index("held_out")
    g = p.score - p.v15
    per_variant = {v: float(p[v].mean() - ref_variant[v]) for v in SB.VARIANTS}
    per_source = {s: float(g[src == s].mean()) for s in ("idsse", "metrica")}
    all10 = pick(per_match.loc[scored, ms].mean(axis=1), scored)
    in_sample = per_match.loc[all10, ms] - ref_match.reindex(ms)
    out = {"candidates_scored": len(scored), "candidates_listed": len(order),
           "nested": {"bench": float(p.score.mean()), "v15": float(ref_match.mean()), "gain": float(g.mean()),
                      "better": int((g > 0).sum()), "per_match": g.round(5).to_dict(),
                      "picks": p.trial.astype(int).to_dict(), "per_variant": per_variant, "per_source": per_source},
           "trial1015": {"bench": float(per_match.loc[FIRST, ms].mean()) if FIRST in scored else None},
           "in_sample_best": {"trial": int(all10), "bench": float(per_match.loc[all10, ms].mean()),
                              "gain": float(in_sample.mean()), "better": int((in_sample > 0).sum()),
                              "per_match": in_sample.round(5).to_dict(),
                              "per_variant": {v: float(complete.loc[all10].xs(v, level="variant").mean() - ref_variant[v])
                                              for v in SB.VARIANTS},
                              "rank_of_1015": int(list(per_match.loc[scored, ms].mean(axis=1).rank(ascending=False,
                                                       method="min").loc[[FIRST]])[0]) if FIRST in scored else None}}
    means = per_match.loc[scored, ms].mean(axis=1).sort_values(ascending=False)
    out["top10"] = {str(int(t)): round(float(v), 5) for t, v in means.head(10).items()}
    out["spread"] = {"best": float(means.max()), "median": float(means.median()), "worst": float(means.min())}
    sc_files = [f for f in os.listdir(results()) if f.startswith("skillcorner_")]
    sc_t = sc_trial if sc_trial is not None else all10
    for f in sc_files:
        sc = pd.read_csv(os.path.join(results(), f), dtype={"match": str})
        if sc_t in set(sc.trial):
            rsc = pd.read_csv(REF_SC, dtype={"match": str})
            a = sc[sc.trial == sc_t].set_index("match").spell_f1
            b = rsc[rsc.trial == FIRST].set_index("match").spell_f1
            out["skillcorner"] = {"trial": int(sc_t), "file": f, "score": float(a.mean()), "v15": float(b.mean()),
                                  "gain": float((a - b).mean()), "better": int(((a - b) > 0).sum()), "n": int(len(a))}
            if FIRST in set(sc.trial):
                c = sc[sc.trial == FIRST].set_index("match").spell_f1
                bsc = pd.read_csv(os.path.join(BUILDER, "skillcorner_{}.csv".format(FIRST)), dtype={"match": str})
                bb = bsc[bsc.trial == FIRST].set_index("match").spell_f1
                out["skillcorner"]["check_1015_vs_builder_max_abs_diff"] = float((c - bb).abs().max())
            break
    n = out["nested"]
    parts = {"gain": n["gain"] >= GAIN, "better": n["better"] >= BETTER,
             "variants": min(n["per_variant"].values()) >= -VARIANT_LOSS, "sources": min(n["per_source"].values()) >= 0,
             "skillcorner": (out["skillcorner"]["gain"] >= -SC_LOSS) if "skillcorner" in out else None}
    out["parts"] = parts
    out["passes"] = bool(all(v for v in parts.values() if v is not None)) and parts["skillcorner"] is not None
    out["complete"] = len(scored) == len(order)
    with open(os.path.join(OUT, "judge.json"), "w") as handle:
        json.dump(out, handle, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    print(p.round(4).to_string())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("candidates", "fill", "check", "score", "skillcorner", "judge"))
    parser.add_argument("--worker", default="0/1")
    parser.add_argument("--trials", type=int, nargs="+", default=[])
    parser.add_argument("--sc-trial", type=int, default=None)
    args = parser.parse_args()
    if args.step == "candidates":
        return candidates()
    if args.step == "fill":
        return fill()
    if args.step == "check":
        return check()
    if args.step == "score":
        i, n = (int(x) for x in args.worker.split("/"))
        return run(i, n)
    if args.step == "skillcorner":
        return skillcorner(args.trials)
    return judge(args.sc_trial)


if __name__ == "__main__":
    main()
