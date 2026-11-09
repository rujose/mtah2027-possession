"""Lab candidate `switch`: the decoder's team-switch price set frame by frame (docs/lab/switch.md).

    python -m bench.lab_switch [--set all|idsse]            # ~30 min: boundary model, decode over the grid, judge against robust
    python -m bench.lab_switch --chains [--set all|idsse]   # ~20 min: v12's chain from this start, against v12

robust's decoder charges the same price to change team on any frame (4), so a real interception and
a flicker of the probabilities cost alike. Here a boundary model gives each frame P(a true change of
team is within TOL of it), from robust's features there and robust's own probabilities around it,
trained leaving one match out on the set's matches. Changing team at frame k then costs
max(0, A - B * logit(p_k)): cheap where the model sees a change, dear where it doesn't. B = 0 is
robust's flat price. Changes into and out of D keep robust's price; nothing else changes.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from bench import chain, lab, lab_mix, lab_robust, lab_weighted
from possession import v1, v12

# Params (fixed before any run)
L = lab_robust.L
NAME = "switch"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_switch")    # gitignored (data/*)
TOL = 5                          # frames (0.2 s): "at a change", as timing's targets
OFFSETS = (-10, -5, 0, 5, 10)    # frames: robust's P around the frame, as the model's own view of a change
EVERY = 10                       # every 10th negative frame for training (neighbours are near-copies); all positives
CLIP = (1e-4, 1 - 1e-4)
A = (4.0, 8.0, 16.0)             # the flat part of the price; robust's own is 4
B = (0.0, 0.5, 1.0, 2.0)         # how much the boundary model moves it; 0 = robust's flat price
MIN_POSSESSION = (10, 20)        # frames, as robust's grid
SWITCH_DEAD = 25.0               # robust's
MIN_DEAD = 1
COLUMNS = ["a", "b", "min_possession"]
GRID = [{"a": a, "b": b, "min_possession": k} for b in B for a in A for k in MIN_POSSESSION]
H, AWAY, D = v1.H, v1.A, v1.D


def folder(which):
    return CACHE + chain.suffix(which)


def shifted(x, k):
    """x moved by k frames (x[t + k] at t), edges held."""
    idx = np.clip(np.arange(len(x)) + k, 0, len(x) - 1)
    return x[idx]


def features(m, which):
    """robust's features (blanked) at each frame, plus robust's held-out P around it."""
    c = lab_robust.cached(m, which)
    p = lab_robust.held_out_proba(m, which)
    margin = p[:, 0] - p[:, 1]
    extra = [shifted(p[:, j], k) for k in OFFSETS for j in range(3)]
    extra += [shifted(margin, k) - shifted(margin, -k) for k in (5, 10, 25)]
    X = np.hstack([c["X"], np.stack(extra, axis=1).astype(np.float32)])
    return X, c["truth"]


def targets(truth):
    """bool [frames]: a true direct change of team (H <-> A) within TOL frames."""
    a, b = truth[:-1], truth[1:]
    at = np.flatnonzero(np.isin(a, (0, 1)) & np.isin(b, (0, 1)) & (a != b)) + 1
    y = np.zeros(len(truth), bool)
    for f in at:
        y[max(f - TOL, 0):f + TOL + 1] = True
    return y


def fit(matches, which):
    Xs, ys = [], []
    for m in matches:
        X, truth = features(m, which)
        y = targets(truth)
        graded = np.isin(truth, (0, 1, 2))
        rows = np.flatnonzero(graded & y)
        rows = np.r_[rows, np.flatnonzero(graded & ~y)[::EVERY]]
        Xs.append(X[rows])
        ys.append(y[rows])
    return HistGradientBoostingClassifier(**L.TREES).fit(np.concatenate(Xs), np.concatenate(ys))


def viterbi_min_length(cost, switch, min_frames, team_price):
    """possession/decoder.py's viterbi_min_length, with the H <-> A price on frame t taken from
    team_price[t] instead of switch[H][A] (everything else, ties included, unchanged)."""
    INF = float("inf")
    cost = np.asarray(cost, dtype=np.float64)
    n_frames, n_labels = cost.shape
    k = [int(v) for v in min_frames]
    labels = range(n_labels)
    c = cost.tolist()
    total = np.vstack([np.zeros(n_labels), np.cumsum(cost, axis=0)]).T.tolist()
    sw = np.asarray(switch, dtype=np.float64).tolist()
    tp = np.asarray(team_price, dtype=np.float64).tolist()
    enter = [[INF] * n_frames for _ in labels]
    last = [[INF] * n_frames for _ in labels]
    enter_from = [[0] * n_frames for _ in labels]
    stayed = [[False] * n_frames for _ in labels]
    for j in labels:
        last[j][0] = c[0][j]
        enter[j][0] = c[0][j]
    for t in range(1, n_frames):
        for j in labels:
            best_i, best = 0, INF
            for i in labels:
                if i == j and k[j] > 1:
                    continue
                price = tp[t] if (i, j) in ((H, AWAY), (AWAY, H)) else sw[i][j]
                value = last[i][t - 1] + price
                if value < best:
                    best_i, best = i, value
            enter[j][t] = best + c[t][j]
            enter_from[j][t] = best_i
            if k[j] == 1:
                last[j][t] = enter[j][t]
                continue
            stay = last[j][t - 1] + sw[j][j] + c[t][j]
            start = t - k[j] + 1
            chain_ = enter[j][start] + total[j][t + 1] - total[j][start + 1] if start >= 1 else INF
            stayed[j][t] = stay < chain_
            last[j][t] = stay if stayed[j][t] else chain_
    end = n_frames - 1
    best_value, best_label, best_start = INF, 0, None
    for j in labels:
        for steps in range(1, k[j]):
            start = end - steps + 1
            if start < 1:
                break
            value = enter[j][start] + total[j][end + 1] - total[j][start + 1]
            if value < best_value:
                best_value, best_label, best_start = value, j, start
        if last[j][end] < best_value:
            best_value, best_label, best_start = last[j][end], j, None
    path = np.empty(n_frames, dtype=np.int8)
    t, j = end, best_label
    if best_start is not None:
        path[best_start:] = j
        t, j = best_start - 1, enter_from[j][best_start]
    while t >= 0:
        if t == 0 or k[j] == 1:
            path[t] = j
            if t > 0:
                j = enter_from[j][t]
            t -= 1
        elif stayed[j][t]:
            path[t] = j
            t -= 1
        else:
            start = t - k[j] + 1
            path[start:t + 1] = j
            t, j = start - 1, enter_from[j][start]
    return path


def decode(match, cost, p_change, a, b, min_possession, **_):
    """Letters at v7's level: robust's costs, its dead price, the team price frame by frame."""
    price = np.maximum(0.0, a - b * np.log(np.clip(p_change, *CLIP) / (1 - np.clip(p_change, *CLIP))))
    switch = np.zeros((3, 3))
    switch[H, D] = switch[AWAY, D] = switch[D, H] = switch[D, AWAY] = SWITCH_DEAD
    out = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in match.halves:
        path = viterbi_min_length(cost[first:last + 1], switch, [int(min_possession), int(min_possession), MIN_DEAD],
                                  price[first:last + 1])
        out[first:last + 1] = v1.LABELS[path]
    return out


def held_out(match, which):
    return np.load(os.path.join(folder(which), "p_{}.npy".format(match.id))).astype(np.float64)


def train_all(which):
    os.makedirs(folder(which), exist_ok=True)
    ms = lab.matches(which)
    for held in ms:
        model = fit([m for m in ms if m != held], which)
        X, truth = features(held, which)
        p = model.predict_proba(X)[:, 1]
        np.save(os.path.join(folder(which), "p_{}.npy".format(held)), p.astype(np.float32))
        y = targets(truth)
        graded = np.isin(truth, (0, 1, 2))
        top = graded & (p >= np.quantile(p[graded], 0.98))
        print("{}: boundary model: of the top 2 % of frames, {:.0%} are within 0.2 s of a true change (base {:.1%})".format(
            held, y[top].mean(), y[graded].mean()), flush=True)


def letters_of(match, which, setting, snap=True):
    p = lab_robust.held_out_proba(match, which)
    decoded = decode(match, -np.log(p), held_out(match, which), **setting)
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
    """v12's chain from robust (v12) and from this start, each stage refitted and on its setting for
    that match; this against v12."""
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
    flat = g[g.b == 0].groupby(["a", "min_possession"]).spell_f1.mean()
    print("flat price (b = 0), in sample: {}".format(flat.round(4).to_dict()))
    lab.save(NAME, g, report, args.which)


if __name__ == "__main__":
    main()
