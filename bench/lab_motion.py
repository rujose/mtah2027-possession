"""Lab candidate `motion`: robust plus how each team moves, without the ball (docs/lab/motion.md).

    python -m bench.lab_motion [--set all|idsse]            # ~30 min: train, decode over robust's grid, judge against robust
    python -m bench.lab_motion --chains [--set all|idsse]   # ~20 min: v12's chain from this start, against v12

From another team's repo (ZesterJ, +0.07 for them): a team in possession pushes forward, the other
drops back, and that shows in the players alone, so it holds while the ball is missing. For each team,
with x measured from its own goal: the shift of its centroid, its front three and its back line (the
2nd-4th deepest, the keeper left out) over the next and last 2, 4 and 6 s, and the same for home minus
away: 27 numbers per frame, added to robust's 97. Everything else is robust's: blanked ball, IDSSE-only
training, every 10th graded frame, teams swapped, the same trees and decoder grid.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from bench import chain, lab, lab_mix, lab_robust, lab_weighted, sets
from possession import lab_learned_features as llf, v12
from possession.data import AWAY, FPS, HOME

# Params (fixed before any run)
L = lab_robust.L
NAME = "motion"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_motion")    # gitignored (data/*)
WINDOWS = (2, 4, 6)              # s: how far ahead and back the shift looks
LINE = 3                         # players in the front line and in the back line
MIN_PLAYERS = 5                  # fewer seen than this: the lines aren't measured
COLUMNS = ["switch_team", "switch_dead", "min_possession"]
GRID = [dict(zip(COLUMNS, (team, dead, k))) for team in L.SWITCH_TEAM for dead in L.SWITCH_DEAD for k in L.MIN_POSSESSION]


def folder(which):
    return CACHE + chain.suffix(which)


def lines(x):
    """(centroid, front line, back line) [frames] of one team's x from its own goal [frames, slots]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centroid = np.nanmean(x, axis=1)
    n = np.sum(~np.isnan(x), axis=1)
    s = np.sort(x, axis=1)                                   # NaN last
    rows = np.arange(len(x))
    front = np.mean([s[rows, np.clip(n - 1 - i, 0, x.shape[1] - 1)] for i in range(LINE)], axis=0)
    back = np.mean([s[rows, np.clip(1 + i, 0, x.shape[1] - 1)] for i in range(LINE)], axis=0)
    few = n < MIN_PLAYERS
    front[few], back[few] = np.nan, np.nan
    return centroid, front, back


def shift(v, frames, halves):
    """v[t + frames] - v[t - frames] inside each half (NaN where either side is outside it)."""
    out = np.full(len(v), np.nan, dtype=np.float32)
    for _, first, last in halves:
        idx = np.arange(first, last + 1)
        a, b = idx + frames, idx - frames
        ok = (a <= last) & (b >= first)
        out[idx[ok]] = v[a[ok]] - v[b[ok]]
    return out


def motion(track, halves, pitch, swap=False):
    """(names, float32 [frames, 27]): the teams' shifts; swap exchanges the teams, as robust's features."""
    players, team = track["players"], np.asarray(track["team"])
    if swap:
        team = np.where(team == HOME, AWAY, np.where(team == AWAY, HOME, team))
    left = llf.home_left(players, team, halves, len(players))[:, None]
    x = players[:, :, 0]
    home_x = np.where(left, x, pitch[0] - x)[:, team == HOME]
    away_x = np.where(left, pitch[0] - x, x)[:, team == AWAY]
    f = {}
    parts = {"home": lines(home_x), "away": lines(away_x)}
    for w in WINDOWS:
        for j, part in enumerate(("centroid", "front", "back")):
            h = shift(parts["home"][j], w * FPS, halves)
            a = shift(parts["away"][j], w * FPS, halves)
            f["{}_home_{}s".format(part, w)], f["{}_away_{}s".format(part, w)] = h, a
            f["{}_diff_{}s".format(part, w)] = h - a
    names = list(f)
    return names, np.stack([f[k] for k in names], axis=1).astype(np.float32)


def features(m, which):
    """robust's 97 cached features beside the 27 motion ones, as is and with the teams swapped."""
    path = os.path.join(folder(which), "{}.npz".format(m))
    if not os.path.exists(path):
        match = lab.load(m, which)
        _, X = motion(match.track, match.halves, match.pitch)
        _, Xs = motion(match.track, match.halves, match.pitch, swap=True)
        os.makedirs(folder(which), exist_ok=True)
        np.savez(path, X=X, X_swapped=Xs)
    c = lab_robust.cached(m, which)
    with np.load(path) as z:
        return np.hstack([c["X"], z["X"]]), np.hstack([c["X_swapped"], z["X_swapped"]]), c["truth"]


def fit(matches, which):
    Xs, ys = [], []
    for m in [m for m in matches if sets.source(m) in L.TRAIN_SOURCES]:
        X, Xw, truth = features(m, which)
        rows = np.flatnonzero(np.isin(truth, (0, 1, 2)))[::L.EVERY]
        y = truth[rows].astype(np.int64)
        Xs += [X[rows], Xw[rows]]
        ys += [y, L.SWAP[y]]
    return HistGradientBoostingClassifier(**L.TREES).fit(np.concatenate(Xs), np.concatenate(ys))


def held_out(match, which):
    return np.load(os.path.join(folder(which), "p_{}.npy".format(match.id))).astype(np.float64)


def train_all(which):
    ms = lab.matches(which)
    models = {}
    for held in ms:
        used = tuple(m for m in ms if m != held and sets.source(m) in L.TRAIN_SOURCES)
        if used not in models:
            models = {used: fit(used, which)}
        X, Xw, _ = features(held, which)
        p = (models[used].predict_proba(X) + models[used].predict_proba(Xw)[:, L.SWAP]) / 2
        p = np.round(np.clip(p, *L.CLIP), L.DECIMALS)
        np.save(os.path.join(folder(which), "p_{}.npy".format(held)), p.astype(np.float32))
        print("{}: predicted".format(held), flush=True)


def letters_of(match, which, setting, snap=True):
    decoded = L.decode_cost(match, -np.log(held_out(match, which)), **setting)
    return np.asarray(L.finish(match, decoded, which) if snap else decoded).astype("<U1")


def grid(which):
    rows = []
    for match in lab.each(which):
        for setting in GRID:
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
    print("\n{} chain against v12:\n{}".format(NAME, lab.describe(report)))
    table = pd.DataFrame({k: {m: v["spell_f1"] for m, v in s.items()} for k, s in scores.items()})
    print(table.round(4).to_string())
    with open(os.path.join(lab.RESULTS, "lab_{}_chains{}.json".format(NAME, chain.suffix(which))), "w") as handle:
        json.dump({"scores": table.round(5).to_dict(), "against_v12": report}, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--chains", action="store_true", help="v12's chain from this start, after the run")
    args = parser.parse_args()
    if args.chains:
        return chains(args.which)
    train_all(args.which)
    g = grid(args.which)
    report = lab.judge(g, COLUMNS, args.which, base=lab_weighted.robust_base(args.which))
    print(lab.describe(report))
    print("picks: {}".format(pd.Series([str(f["picked"]) for f in report["folds"]]).value_counts().to_dict()))
    lab.save(NAME, g, report, args.which)


if __name__ == "__main__":
    main()
