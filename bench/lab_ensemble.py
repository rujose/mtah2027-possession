"""Lab candidate `ensemble` (round 7): the lab's per-frame models combined into one start (docs/lab/ensemble.md).

    python -m bench.lab_ensemble [--set all|idsse]            # ~15 min, < 0.5 GB: combine, decode over robust's grid,
                                                              # judge against robust; writes bench/results/lab_ensemble[_idsse]*
    python -m bench.lab_ensemble --chains [--set all|idsse]   # ~20 min: v12's chain from this start, against v12

Four per-frame models already give every bench match a held-out P(H / A / D), each match predicted by a
model trained without it: robust (v12's start), learned (robust before the blanking), weighted (alpha 0.5)
and motion. Averaging diverse models often beats each of them. Everything below was fixed before any run:

  mean    the arithmetic mean of the members' P
  geo     the geometric mean (the mean of log P)
  stack   a multinomial logistic regression on the four members' log P (12 numbers), trained on the other
          matches' held-out P and truth: IDSSE's only, as every member; every 10th graded frame; each frame
          also with the teams swapped, and predicted as the mean of both ways

mean and geo for each of COMBOS, stack on all four: 9 starts. Each P is renormalised, clipped and rounded
as the members are, decoded over robust's decoder grid (lab_learned's team switch x dead switch x minimum
possession), then v9's snap and v11's move. The start is chosen together with the decoder's prices, each
match on the other matches (lab.judge), against robust's own held-out score.

To use it: `start_proba(match, which, start)` -> P (saved by the run), `letters_of(match, which, setting)`
-> letters after snap and move; a fold's setting is in bench/results/lab_ensemble[_idsse].json.
"""

import argparse
import json
import os
import types

import numpy as np
import pandas as pd

from bench import chain, lab, lab_mix, lab_motion, lab_robust, lab_weighted, sets
from possession import v12

# Params (fixed before any run)
L = lab_robust.L
NAME = "ensemble"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_ensemble")    # gitignored (data/*)
MEMBERS = ("robust", "learned", "weighted", "motion")
WEIGHTED_ALPHA = 0.5             # weighted's member: halfway from frames to possessions (its folds' usual pick)
COMBOS = (("robust", "motion"), ("robust", "learned"), ("robust", "weighted"), MEMBERS)
RULES = ("mean", "geo")          # the arithmetic mean first (ties go to it)
STACK_C = 1.0                    # the logistic regression's regularisation (scikit-learn's default)
STARTS = ["{}_{}".format(rule, "+".join(c)) for rule in RULES for c in COMBOS] + ["stack_" + "+".join(MEMBERS)]
DECODER = lab_motion.COLUMNS     # switch_team, switch_dead, min_possession: robust's grid, 36 settings
COLUMNS = ["start"] + DECODER
GRID = [{"start": s, **d} for s in STARTS for d in lab_motion.GRID]
CHECK = "robust"                 # decoded over the same grid too: robust alone, a check on the harness (not judged)
MIN_GAIN = 0.005                 # round 7's bar: the chain must gain at least this over v12 (docs/lab.md)


def folder(which):
    return CACHE + chain.suffix(which)


def ref(m):
    return m if hasattr(m, "id") else types.SimpleNamespace(id=m)


def member(m, which, name):
    """One member's held-out P(H, A, D) on match m (id or lab match), as its own run saved it."""
    m = ref(m)
    if name == "robust":
        return lab_robust.held_out_proba(m, which, "blank")
    if name == "learned":
        return L.held_out_proba(m, which, "all")
    if name == "weighted":
        return lab_weighted.held_out(m, which, WEIGHTED_ALPHA)
    if name == "motion":
        return lab_motion.held_out(m, which)
    raise ValueError(name)


def truth_of(m, which):
    """The harness's truth codes for match id m (training the stacker on the other matches only)."""
    with np.load(os.path.join(lab.folder(which), "{}.npz".format(m))) as z:
        return z["truth"]


def tidy(p):
    """Renormalised, then clipped and rounded as the members are (v4's clip and rounding)."""
    p = p / p.sum(axis=1, keepdims=True)
    return np.round(np.clip(p, *L.CLIP), L.DECIMALS)


def combine(ps, rule):
    if rule == "mean":
        return tidy(np.mean(ps, axis=0))
    if rule == "geo":
        return tidy(np.exp(np.mean(np.log(ps), axis=0)))
    raise ValueError(rule)


def stack_features(ps, swap=False):
    """[frames, 3 x members]: each member's log P, with H and A exchanged if swap."""
    return np.hstack([np.log(p[:, L.SWAP] if swap else p) for p in ps])


def fit_stack(ids, which):
    """The stacker on these matches' IDSSE ones: every EVERY-th graded frame, as is and teams swapped."""
    from sklearn.linear_model import LogisticRegression
    Xs, ys = [], []
    for m in [m for m in ids if sets.source(m) in L.TRAIN_SOURCES]:
        truth = truth_of(m, which)
        rows = np.flatnonzero(np.isin(truth, (0, 1, 2)))[::L.EVERY]
        ps = [member(m, which, k)[rows] for k in MEMBERS]
        y = truth[rows].astype(np.int64)
        Xs += [stack_features(ps), stack_features(ps, swap=True)]
        ys += [y, L.SWAP[y]]
    return LogisticRegression(C=STACK_C, max_iter=2000).fit(np.concatenate(Xs), np.concatenate(ys))


def stack_proba(clf, ps):
    return tidy((clf.predict_proba(stack_features(ps)) + clf.predict_proba(stack_features(ps, swap=True))[:, L.SWAP]) / 2)


def start_proba(match, which, start):
    """A start's P on this match, saved by `train_all` (robust's own for CHECK)."""
    if start == CHECK:
        return member(match, which, "robust")
    return np.load(os.path.join(folder(which), "p_{}_{}.npy".format(start, ref(match).id))).astype(np.float64)


def frame_stats(p, truth):
    """Log loss and argmax accuracy on the graded frames (a diagnostic; never picks anything)."""
    graded = np.isin(truth, (0, 1, 2))
    y = truth[graded].astype(np.int64)
    q = p[graded]
    return {"log_loss": float(-np.log(q[np.arange(len(y)), y]).mean()), "acc": float((q.argmax(axis=1) == y).mean())}


def train_all(which):
    """Every start's P on each match (the stacker trained without it), and the frame diagnostics."""
    os.makedirs(folder(which), exist_ok=True)
    ms = lab.matches(which)
    models, rows, coefs = {}, [], {}
    for held in ms:
        ps = {k: member(held, which, k) for k in MEMBERS}
        out = {}
        for rule in RULES:
            for combo in COMBOS:
                out["{}_{}".format(rule, "+".join(combo))] = combine(np.stack([ps[k] for k in combo]), rule)
        used = tuple(m for m in ms if m != held and sets.source(m) in L.TRAIN_SOURCES)
        if used not in models:
            models = {used: fit_stack(used, which)}
        clf = models[used]
        coefs[held] = {"classes": [int(c) for c in clf.classes_], "members": list(MEMBERS),
                       "coef": np.round(clf.coef_, 4).tolist(), "intercept": np.round(clf.intercept_, 4).tolist()}
        out["stack_" + "+".join(MEMBERS)] = stack_proba(clf, [ps[k] for k in MEMBERS])
        for start, p in out.items():
            np.save(os.path.join(folder(which), "p_{}_{}.npy".format(start, held)), p.astype(np.float32))
        truth = truth_of(held, which)
        top = {k: p.argmax(axis=1) for k, p in ps.items()}
        graded = np.isin(truth, (0, 1, 2))
        for k, p in list(ps.items()) + list(out.items()):
            rows.append({"match": held, "p": k, **frame_stats(p, truth),
                         "disagree_with_robust": float((p.argmax(axis=1) != top["robust"])[graded].mean())})
        print("{}: {} starts saved".format(held, len(out)), flush=True)
    return pd.DataFrame(rows), coefs


def letters_of(match, which, setting, snap=True):
    p = start_proba(match, which, setting["start"])
    decoded = L.decode_cost(match, -np.log(p), **{k: setting[k] for k in DECODER})
    return np.asarray(L.finish(match, decoded, which) if snap else decoded).astype("<U1")


def grid(which):
    rows = []
    for match in lab.each(which):
        for setting in [{"start": CHECK, **d} for d in lab_motion.GRID] + GRID:
            s = lab.score(match, letters_of(match, which, setting))
            rows.append({"match": match.id, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                         "possessions": s["possessions"]})
        print("{}: grid scored".format(match.id), flush=True)
    return pd.DataFrame(rows)


def chains(which):
    """v12's chain from robust (v12) and from this start, each stage refitted on its setting for that match."""
    lab_mix.cache_short_evidence()
    matches = list(lab.each(which))
    picks = lab_mix.fold_settings(which)
    with open(os.path.join(lab.RESULTS, "lab_{}{}.json".format(NAME, chain.suffix(which)))) as handle:
        mine = {f["match"]: f["picked"] for f in json.load(handle)["folds"]}
    scores = {}
    for start in ("robust", NAME):
        if start == "robust":
            letters = {m.id: lab_weighted.start_letters(m, "robust", True, picks, which) for m in matches}
        else:
            letters = {m.id: letters_of(m, which, mine[m.id]) for m in matches}
        for stage in v12.CHAIN[which]:
            letters = lab_mix.add_stage(stage, letters, matches, picks)
        scores[start] = lab_mix.scores_of(letters, matches)
        print("{}: {:.4f}".format(start, np.mean([v["spell_f1"] for v in scores[start].values()])), flush=True)
    report = lab_mix.judge_scores(scores[NAME], scores["robust"], which)
    gain = report["all"]["held_out"] - report["all"]["base"]
    report["checks"]["gain"] = bool(gain >= MIN_GAIN)
    report["passes"] = all(report["checks"].values())
    print("\n{} chain against v12 (gain {:+.4f}, needs {:+.3f}):\n{}".format(NAME, gain, MIN_GAIN, lab.describe(report)))
    table = pd.DataFrame({k: {m: v["spell_f1"] for m, v in s.items()} for k, s in scores.items()})
    print(table.round(4).to_string())
    with open(os.path.join(lab.RESULTS, "lab_{}_chains{}.json".format(NAME, chain.suffix(which))), "w") as handle:
        json.dump({"scores": table.round(5).to_dict(), "picks": mine, "against_v12": report}, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--chains", action="store_true", help="v12's chain from this start, after the run")
    args = parser.parse_args()
    which, tag = args.which, chain.suffix(args.which)
    if args.chains:
        return chains(which)
    frames, coefs = train_all(which)
    print(frames.groupby("p", sort=False)[["log_loss", "acc", "disagree_with_robust"]].mean().round(4).to_string())
    g = grid(which)
    base = lab_weighted.robust_base(which)
    report = lab.judge(g[g.start != CHECK].reset_index(drop=True), COLUMNS, which, base=base)
    print("\nensemble (the start chosen with the prices, each match on the others), against robust:")
    print(lab.describe(report))
    print("picks: {}".format(pd.Series([str(f["picked"]) for f in report["folds"]]).value_counts().to_dict()))
    lab.save(NAME, g, report, which)
    alone = {}
    for start in [CHECK] + STARTS:      # each start judged on its own, for the page (the bar is judged above)
        r = lab.judge(g[g.start == start].reset_index(drop=True), COLUMNS, which, base=base)
        alone[start] = r
        print("[{} only] {}".format(start, lab.describe(r).replace("\n", "\n  ")))
    frames.to_csv(os.path.join(lab.RESULTS, "lab_{}_frames{}.csv".format(NAME, tag)), index=False, float_format="%.5f")
    with open(os.path.join(lab.RESULTS, "lab_{}_starts{}.json".format(NAME, tag)), "w") as handle:
        json.dump({"alone": alone, "stack_coefs": coefs}, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
