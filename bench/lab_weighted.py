"""Lab candidate `weighted`: robust trained possession by possession, not frame by frame (docs/lab/weighted.md).

    python -m bench.lab_weighted [--set all|idsse]            # ~40 min: train, decode over the grid, judge against robust
    python -m bench.lab_weighted --chains [--set all|idsse]   # ~30 min: the four chains below, each stage refitted

robust trains on frames, so a 1 s possession weighs 1/20 of a 20 s one, while the metric counts both
the same. Here the frames of a true possession of length n weigh n^-alpha: alpha 1 gives every
possession the same total weight. Dead frames keep their weight, and each match's live frames keep
their total, so the balance between live and dead doesn't move. Everything else is robust's: its
blanked features, IDSSE-only training, every 10th graded frame, the teams swapped, the same trees.
The decoder's prices are chosen again over the grid robust used, each match on the others.

--chains answers two questions together, every post stage retrained on each chain's own output:
does the start change (robust or weighted), and do v9's snap and v11's move still earn their place?
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from bench import chain, lab, lab_mix, lab_robust, labels, sets
from possession import v12

# Params (fixed before any run)
L = lab_robust.L
NAME = "weighted"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_weighted")    # gitignored (data/*)
ALPHAS = (0.5, 1.0)              # 1: every possession the same weight; 0.5: halfway from frames
COLUMNS = ["alpha", "switch_team", "switch_dead", "min_possession"]
GRID = [dict(zip(COLUMNS, (alpha, team, dead, k))) for alpha in ALPHAS for team in L.SWITCH_TEAM
        for dead in L.SWITCH_DEAD for k in L.MIN_POSSESSION]
# The chains: (start, whether v9's snap and v11's move run). The first is v12.
CHAINS = (("robust", True), ("robust", False), ("weighted", True), ("weighted", False))


def folder(which):
    return CACHE + chain.suffix(which)


def frame_weights(truth, alpha):
    """float [frames]: a frame of a true possession of length n weighs n^-alpha, scaled so the match's
    live frames keep their total weight; every other frame weighs 1."""
    w = np.ones(len(truth))
    for a, b, k in labels.runs_of(truth):
        if k in (0, 1):                  # scorer codes: 0 H, 1 A
            w[a:b] = float(b - a) ** -alpha
    live = np.isin(truth, (0, 1))
    w[live] *= live.sum() / w[live].sum()
    return w


def fit(matches, alpha, which):
    """The classifier on these matches' IDSSE ones, as lab_robust.fit, with the frames weighted."""
    Xs, ys, ws = [], [], []
    for m in [m for m in matches if sets.source(m) in L.TRAIN_SOURCES]:
        c = lab_robust.cached(m, which)
        rows = np.flatnonzero(np.isin(c["truth"], (0, 1, 2)))[::L.EVERY]
        y = c["truth"][rows].astype(np.int64)
        w = frame_weights(c["truth"], alpha)[rows]
        Xs += [c["X"][rows], c["X_swapped"][rows]]
        ys += [y, L.SWAP[y]]
        ws += [w, w]
        del c
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    del Xs
    return HistGradientBoostingClassifier(**L.TREES).fit(X, y, sample_weight=w)


def proba(m, clf, which):
    """P(H, A, D) per frame, as lab_robust.proba."""
    c = lab_robust.cached(m, which)
    p = (clf.predict_proba(c["X"]) + clf.predict_proba(c["X_swapped"])[:, L.SWAP]) / 2
    return np.round(np.clip(p, *L.CLIP), L.DECIMALS)


def held_out(match, which, alpha):
    return np.load(os.path.join(folder(which), "p_{:g}_{}.npy".format(alpha, match.id))).astype(np.float64)


def robust_base(which):
    """robust's held-out score per match, from its own judging: the number this must beat."""
    with open(os.path.join(lab.RESULTS, "lab_robust{}.json".format(chain.suffix(which)))) as handle:
        folds = json.load(handle)["folds"]
    return {f["match"]: {"spell_f1": f["held_out"], "macro_f1": f["macro_held_out"], "source": f["source"], "unseen": f["unseen"]}
            for f in folds}


def train_all(which):
    """Each match's P from models trained without it (Metrica's matches share one: same training set)."""
    os.makedirs(folder(which), exist_ok=True)
    ms = lab.matches(which)
    for alpha in ALPHAS:
        models = {}
        for held in ms:
            used = tuple(m for m in ms if m != held and sets.source(m) in L.TRAIN_SOURCES)
            if used not in models:
                models = {used: fit(used, alpha, which)}
            p = proba(held, models[used], which)
            np.save(os.path.join(folder(which), "p_{:g}_{}.npy".format(alpha, held)), p.astype(np.float32))
            print("alpha {:g}, {}: predicted".format(alpha, held), flush=True)


def grid(which):
    """One row per (match, setting): decoded, then v9's snap and v11's move, as robust was judged."""
    rows = []
    for match in lab.each(which):
        for alpha in ALPHAS:
            cost = -np.log(held_out(match, which, alpha))
            for setting in [s for s in GRID if s["alpha"] == alpha]:
                decoded = L.decode_cost(match, cost, **{k: v for k, v in setting.items() if k != "alpha"})
                s = lab.score(match, L.finish(match, decoded, which))
                rows.append({"match": match.id, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                             "d_f1": s["d_f1"], "possessions": s["possessions"]})
        print("{}: grid scored".format(match.id), flush=True)
    return pd.DataFrame(rows)


def start_letters(match, start, snap, picks, which):
    """The letters before the post stages: the start's held-out P decoded, with or without snap and move."""
    s = picks[start][match.id]
    p = lab_robust.held_out_proba(match, which, s["variant"]) if start == "robust" else held_out(match, which, s["alpha"])
    decoded = L.decode_cost(match, -np.log(p), **{k: v for k, v in s.items() if k != "alpha"})
    return np.asarray(L.finish(match, decoded, which) if snap else decoded).astype("<U1")


def chains(which):
    """The four chains, out of fold, each stage on its setting for that match, then every comparison."""
    lab_mix.cache_short_evidence()
    matches = list(lab.each(which))
    picks = lab_mix.fold_settings(which)
    with open(os.path.join(lab.RESULTS, "lab_{}{}.json".format(NAME, chain.suffix(which)))) as handle:
        picks["weighted"] = {f["match"]: f["picked"] for f in json.load(handle)["folds"]}
    scores = {}
    for start, snap in CHAINS:
        letters = {m.id: start_letters(m, start, snap, picks, which) for m in matches}
        for stage in v12.CHAIN[which]:
            letters = lab_mix.add_stage(stage, letters, matches, picks)
        scores[(start, snap)] = lab_mix.scores_of(letters, matches)
        print("{}, snap and move {}: {:.4f}".format(start, "on" if snap else "off", np.mean(
            [v["spell_f1"] for v in scores[(start, snap)].values()])), flush=True)
    a, b, c, d = CHAINS
    out = {}
    for name, x, y in (("no snap vs v12", b, a), ("weighted vs v12", c, a), ("weighted, no snap vs v12", d, a),
                       ("weighted, no snap vs no snap", d, b), ("weighted, no snap vs weighted", d, c)):
        report = lab_mix.judge_scores(scores[x], scores[y], which)
        out[name] = report
        print("\n{}:\n{}".format(name, lab.describe(report)), flush=True)
    table = pd.DataFrame({"{}{}".format(s, "" if on else ", no snap"): {m: v["spell_f1"] for m, v in scores[(s, on)].items()}
                          for s, on in CHAINS})
    print("\n" + table.round(4).to_string())
    with open(os.path.join(lab.RESULTS, "lab_{}_chains{}.json".format(NAME, chain.suffix(which))), "w") as handle:
        json.dump({"scores": table.round(5).to_dict(), "comparisons": out}, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--chains", action="store_true", help="the four chains, after the run")
    args = parser.parse_args()
    if args.chains:
        return chains(args.which)
    train_all(args.which)
    g = grid(args.which)
    report = lab.judge(g, COLUMNS, args.which, base=robust_base(args.which))
    print(lab.describe(report))
    print("possessions per match with each fold's pick: {:.0f}".format(np.mean([
        g[(g.match == f["match"]) & np.all([g[c] == f["picked"][c] for c in COLUMNS], axis=0)].possessions.iloc[0] for f in report["folds"]])))
    print("picks: {}".format(pd.Series([str(f["picked"]) for f in report["folds"]]).value_counts().to_dict()))
    lab.save(NAME, g, report, args.which)


if __name__ == "__main__":
    main()
