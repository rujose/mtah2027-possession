"""Lab candidate `hsmm`: an explicit-duration decoder in place of the flat price and the hard minimum (docs/lab/hsmm.md).

    python -m bench.lab_hsmm --check [--set all|idsse]    # ~2 min: lambda = 0 against possession/decoder.py's viterbi_min_length
    python -m bench.lab_hsmm [--set all|idsse]            # ~15 min: robust's held-out P decoded over the grid, judged against robust
    python -m bench.lab_hsmm --chains [--set all|idsse]   # ~20 min: v12's chain from this start, against v12

robust's decoder (possession/decoder.py, viterbi_min_length) charges the same price for every change
of team and forbids possessions under 10 frames; past that, a 0.5 s possession and a 50 s one cost
the same. Here (a hidden semi-Markov decoder) a stretch of label s lasting d frames costs its
frames' -log P, plus a switch price into it, plus

    lam * (log g_max - log g_s(d))

where g_s is the density of a true stretch's length (possessions, H and A pooled; and dead balls),
from the OTHER matches' truth: a Gaussian kernel density of log length (Silverman's bandwidth),
divided by the length. It is 0 at the likeliest length and grows away from it, so the decoder
prefers realistic lengths; the level of the price stays the switch price's job. lam = 0 is the
min-length decoder (`--check`).

To be fast: the half is decoded in steps of STEP frames (5 Hz; a step's cost is its frames' sum),
lengths enumerated up to CAP (60 s possessions, 150 s dead), a stretch reaching the cap continuing
step by step at the cap's cost; then each boundary is moved within +-REFINE frames to the frame the
25 Hz costs like best. The first and last stretch of a half are cut by its edges: no minimum and no
length cost (as viterbi_min_length lets them be shorter). Then v9's snap and v11's move (L.finish).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from bench import chain, lab, lab_mix, lab_robust, lab_weighted, labels
from bench.tune_decoder import switch_matrix
from possession import v1, v12
from possession.data import FPS
from possession.decoder import viterbi_min_length

# Params (fixed before any run)
L = lab_robust.L
NAME = "hsmm"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_hsmm")    # gitignored (data/*)
STEP = 5                                       # frames per decoding step (5 Hz)
REFINE = 5                                     # frames: each boundary refined within +- this at 25 Hz
CAP = {"possession": 60 * FPS, "dead": 150 * FPS}    # frames: lengths enumerated up to here
SWITCH_DEAD = 25.0                             # robust's (every fold of both sets)
MIN_DEAD = 1                                   # robust's
LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)       # weight of the length cost; 0 = robust's decoder (4 and 8: see the page)
DEAD_TERM = (0, 1)                             # whether dead stretches pay a length cost too
MIN_POSSESSION = (10, 1)                       # frames: robust's hard minimum, or none
SWITCH_TEAM = (4.0, 2.0, 1.0, 8.0)             # robust's 4 first; lower ones because lam > 0 adds a cost per possession
COLUMNS = ["lam", "dead_term", "min_possession", "switch_team"]
GRID = [dict(zip(COLUMNS, (lam, dt, k, team))) for lam in LAMBDAS for dt in (DEAD_TERM if lam else (0,))
        for k in MIN_POSSESSION for team in SWITCH_TEAM]
INF = float("inf")
H, A, D = v1.H, v1.A, v1.D


def folder(which):
    return CACHE + chain.suffix(which)


# --- the length distributions, from the other matches' truth ---------------------------------------

_runs = {}


def runs(m, which):
    """{"possession": [...], "dead": [...]}: lengths (frames) of match m's true stretches, the first and
    last of each half left out (cut by the edges). Truth only: for the OTHER matches' distributions."""
    if (m, which) not in _runs:
        with np.load(os.path.join(lab.folder(which), "{}.npz".format(m))) as z:
            truth, halves = z["truth"], z["halves"]
        out = {"possession": [], "dead": []}
        for _, first, last in halves:
            for a, b, v in labels.runs_of(truth[first:last + 1])[1:-1]:
                if v in (0, 1):
                    out["possession"].append(b - a)
                elif v == 2:
                    out["dead"].append(b - a)
        _runs[(m, which)] = out
    return _runs[(m, which)]


def length_cost(lengths, cap_frames, step):
    """float [cap_steps + 1]: cost[d] for a stretch of d steps (index 0 unused) = log g_max - log g(d * step),
    g the density of a length in frames (a Gaussian kernel density of log length, Silverman's
    bandwidth, divided by the length)."""
    x = np.log(np.asarray(lengths, dtype=np.float64))
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 0.9 * min(x.std(), iqr / 1.34) * len(x) ** -0.2
    steps = np.arange(1, cap_frames // step + 1)
    at = np.log(steps * step)
    f = np.zeros(len(at))
    for chunk in np.array_split(np.arange(len(x)), max(1, len(x) // 2000)):
        f += np.exp(-0.5 * ((at[:, None] - x[None, chunk]) / h) ** 2).sum(axis=1)
    log_g = np.log(np.maximum(f / (len(x) * h * math.sqrt(2 * math.pi)) / (steps * step), 1e-300))
    return np.r_[0.0, log_g.max() - log_g]


def distributions(held, which):
    """The length costs for match `held`, from the other matches of the set (at STEP)."""
    others = [m for m in lab.matches(which) if m != held]
    out = {}
    for part in ("possession", "dead"):
        lengths = np.concatenate([runs(m, which)[part] for m in others])
        out[part] = length_cost(lengths, CAP[part], STEP)
    return out


# --- the decoder ------------------------------------------------------------------------------------

def segmental(C, sw, dmin, cap, dur):
    """Cheapest labelling of steps 0..T-1 where every stretch pays its length cost.

    C     float [T, labels]   per-step cost of each label
    sw    float [labels, labels]   price of a switch i -> j (the diagonal: of staying, normally 0)
    dmin  int [labels]   shortest stretch, in steps (not for the first and last of the half)
    cap   int [labels] >= dmin: stretches of dmin..cap-1 steps are enumerated, each paying dur[j][d];
          one reaching cap steps pays dur[j][cap] and then continues step by step at no extra cost
    dur   [labels] float arrays indexed by length in steps (0..cap); zeros = no length cost
    Returns int8 [T].

    With cap == dmin and dur all zero this is viterbi_min_length step for step (ties included): its
    "last step of the chain" is the long state here, and a label with cap 1 is its k == 1 case.
    The last stretch: any length, no minimum, no length cost under the cap (as viterbi_min_length's).
    """
    C = np.asarray(C, dtype=np.float64)
    T, n = C.shape
    lab_ = range(n)
    c = C.tolist()
    S = np.vstack([np.zeros(n), np.cumsum(C, axis=0)]).T.copy()     # S[j, t] = C[:t, j].sum()
    s_list = S.tolist()
    swl = np.asarray(sw, dtype=np.float64).tolist()
    durl = [np.asarray(d, dtype=np.float64) for d in dur]
    enter_np = np.full((n, T), INF)
    enter = [[INF] * T for _ in lab_]
    long_ = [[INF] * T for _ in lab_]
    done = [[INF] * T for _ in lab_]
    enter_from = [[0] * T for _ in lab_]
    stayed = [[False] * T for _ in lab_]
    kind = [[0] * T for _ in lab_]                   # 0: done via the long state; d > 0: an enumerated stretch of d steps
    for j in lab_:
        enter[j][0] = long_[j][0] = done[j][0] = c[0][j]
        enter_np[j, 0] = c[0][j]
    explicit = [cap[j] > dmin[j] for j in lab_]
    for t in range(1, T):
        for j in lab_:
            best_i, best = 0, INF
            for i in lab_:
                if i == j and cap[j] > 1:
                    continue
                value = done[i][t - 1] + swl[i][j]
                if value < best:
                    best_i, best = i, value
            e = best + c[t][j]
            enter[j][t] = e
            enter_np[j, t] = e
            enter_from[j][t] = best_i
            if cap[j] == 1:
                long_[j][t] = done[j][t] = e
                continue
            stay = long_[j][t - 1] + swl[j][j] + c[t][j]
            start = t - cap[j] + 1
            chain_ = enter[j][start] + s_list[j][t + 1] - s_list[j][start + 1] + float(durl[j][cap[j]]) if start >= 1 else INF
            stayed[j][t] = stay < chain_
            long_[j][t] = stay if stayed[j][t] else chain_
            done[j][t] = long_[j][t]
            if explicit[j]:
                a_hi = t - dmin[j] + 1                         # stretch of dmin steps
                a_lo = max(1, t - cap[j] + 2)                  # stretch of cap - 1 steps
                if a_hi >= a_lo:
                    v = (enter_np[j, a_lo:a_hi + 1] + S[j, t + 1]) - S[j, a_lo + 1:a_hi + 2] \
                        + durl[j][dmin[j]:t + 2 - a_lo][::-1]
                    k = int(np.argmin(v))
                    if v[k] < done[j][t]:
                        done[j][t] = float(v[k])
                        kind[j][t] = t - (a_lo + k) + 1
    end = T - 1
    best_value, best_label, best_start = INF, 0, None
    for j in lab_:
        for steps in range(1, cap[j]):
            start = end - steps + 1
            if start < 1:
                break
            value = enter[j][start] + s_list[j][end + 1] - s_list[j][start + 1]
            if value < best_value:
                best_value, best_label, best_start = value, j, start
        if long_[j][end] < best_value:
            best_value, best_label, best_start = long_[j][end], j, None
    path = np.empty(T, dtype=np.int8)
    t, j = end, best_label
    in_long = True                                     # long_[j][end] was the pick
    if best_start is not None:
        path[best_start:] = j
        t, j, in_long = best_start - 1, enter_from[j][best_start], False
    while t >= 0:
        if cap[j] == 1:
            path[t] = j
            if t > 0:
                j = enter_from[j][t]
            t, in_long = t - 1, False                   # entered from the label's completed stretch
            continue
        if not in_long:
            d = kind[j][t]
            if d > 0:
                path[t - d + 1:t + 1] = j
                t, j = t - d, enter_from[j][t - d + 1]
                continue
            in_long = True
        if t == 0:
            path[0] = j
            break
        if stayed[j][t]:
            path[t] = j
            t -= 1
        else:
            start = t - cap[j] + 1
            path[start:t + 1] = j
            t, j, in_long = start - 1, enter_from[j][start], False
    return path


def refine(path, cost, shortest, window=REFINE):
    """Each boundary of `path` (frames) moved within +-window to the frame the costs like best, left to
    right, every stretch kept at least shortest[label] frames (1 for the half's first and last)."""
    rs = labels.runs_of(path)
    if len(rs) < 2:
        return path
    CS = np.vstack([np.zeros(cost.shape[1]), np.cumsum(cost, axis=0)])
    out = np.empty_like(path)
    start = 0
    for i in range(len(rs) - 1):
        s1, s2 = rs[i][2], rs[i + 1][2]
        b, next_end = rs[i][1], rs[i + 1][1]
        m1 = 1 if i == 0 else shortest[s1]
        m2 = 1 if i + 1 == len(rs) - 1 else shortest[s2]
        lo, hi = max(b - window, start + m1), min(b + window, next_end - m2)
        cand = np.arange(lo, hi + 1)
        cand = cand[np.lexsort((cand, np.abs(cand - b)))]          # nearest the coarse boundary first on ties
        score = CS[cand, s1] - CS[cand, s2]
        new = int(cand[int(np.argmin(score))])
        out[start:new] = s1
        start = new
    out[start:] = rs[-1][2]
    return out


def plan(setting, dists, step):
    """(dmin, cap, dur) per label for segmental()."""
    lam, k = float(setting["lam"]), int(setting["min_possession"])
    d_poss = -(-k // step)
    d_dead = -(-MIN_DEAD // step)
    if lam > 0:
        cap_poss = CAP["possession"] // step
        dur_poss = lam * dists["possession"][:cap_poss + 1]
    else:
        cap_poss, dur_poss = d_poss, np.zeros(d_poss + 1)
    if lam > 0 and int(setting["dead_term"]):
        cap_dead = CAP["dead"] // step
        dur_dead = lam * dists["dead"][:cap_dead + 1]
    else:
        cap_dead, dur_dead = d_dead, np.zeros(d_dead + 1)
    if cap_poss <= d_poss:            # a minimum above the cap can't happen with these settings
        cap_poss = d_poss
    return [d_poss, d_poss, d_dead], [cap_poss, cap_poss, cap_dead], [dur_poss, dur_poss, dur_dead]


def decode_half(cost, setting, dists, step=STEP):
    """int8 [frames] for one half: coarse decoding at `step`, then each boundary refined at 25 Hz."""
    switch = switch_matrix(float(setting["switch_team"]), SWITCH_DEAD)
    dmin, cap, dur = plan(setting, dists, step)
    if step == 1:
        return segmental(cost, switch, dmin, cap, dur)
    C = np.add.reduceat(cost, np.arange(0, len(cost), step), axis=0)
    coarse = segmental(C, switch, dmin, cap, dur)
    path = np.repeat(coarse, step)[:len(cost)]
    return refine(path, cost, [int(setting["min_possession"]), int(setting["min_possession"]), MIN_DEAD])


def decode(match, cost, setting, dists, step=STEP):
    """Letters at v7's level (before v9's snap and v11's move)."""
    out = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in match.halves:
        out[first:last + 1] = v1.LABELS[decode_half(cost[first:last + 1], setting, dists, step)]
    return out


def letters_of(match, which, setting, dists=None, snap=True):
    """This start on one match: robust's held-out P, decoded with the setting, then snap and move."""
    dists = dists or distributions(match.id, which)
    decoded = decode(match, -np.log(lab_robust.held_out_proba(match, which)), setting, dists)
    return np.asarray(L.finish(match, decoded, which) if snap else decoded).astype("<U1")


# --- checks, grid, chains -----------------------------------------------------------------------------

def path_cost(cost, path, switch):
    return float(cost[np.arange(len(path)), path].sum() + switch[path[:-1], path[1:]].sum())


def check(which):
    """lam = 0 against viterbi_min_length: at 25 Hz, the same paths (cap = minimum: the same recursion;
    lengths enumerated to the caps with zero cost: the same cost); then the 5 Hz decoder's agreement."""
    rows = []
    for match in lab.each(which):
        cost = -np.log(lab_robust.held_out_proba(match, which))
        dists = distributions(match.id, which)
        for team, k in ((4.0, 10), (8.0, 20), (2.0, 1)):
            setting = {"lam": 0.0, "dead_term": 0, "min_possession": k, "switch_team": team}
            switch = switch_matrix(team, SWITCH_DEAD)
            for _, first, last in match.halves:
                cst = cost[first:last + 1]
                t0 = time.time()
                exact = viterbi_min_length(cst, switch, [k, k, MIN_DEAD])
                t_exact = time.time() - t0
                same = segmental(cst, switch, [k, k, 1], [k, k, 1], [np.zeros(k + 1)] * 2 + [np.zeros(2)])
                row = {"match": match.id, "switch_team": team, "min_possession": k, "frames": len(cst),
                       "identical_cap_min": bool(np.array_equal(exact, same)), "exact_s": t_exact}
                if (team, k) == (4.0, 10):
                    caps = [CAP["possession"], CAP["possession"], CAP["dead"]]
                    t0 = time.time()
                    enum = segmental(cst, switch, [k, k, 1], caps, [np.zeros(c + 1) for c in caps])
                    row["enum_s"] = time.time() - t0
                    row["identical_enumerated"] = bool(np.array_equal(exact, enum))
                    row["cost_gap_enumerated"] = path_cost(cst, enum, switch) - path_cost(cst, exact, switch)
                    t0 = time.time()
                    coarse = decode_half(cst, setting, dists)
                    row["coarse_s"] = time.time() - t0
                    row["agree_5hz"] = float((coarse == exact).mean())
                    row["cost_gap_5hz"] = path_cost(cst, coarse, switch) - path_cost(cst, exact, switch)
                    row["runs_exact"], row["runs_5hz"] = len(labels.runs_of(exact)), len(labels.runs_of(coarse))
                    for name, s in (("lam1", {"lam": 1.0, "dead_term": 1, "min_possession": 1, "switch_team": 2.0}),):
                        t0 = time.time()
                        decode_half(cst, s, dists)
                        row["{}_s".format(name)] = time.time() - t0
                rows.append(row)
        for row in [r for r in rows if r["match"] == match.id and r["switch_team"] == 4.0]:
            print("{}: {}".format(match.id, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "match"}), flush=True)
    df = pd.DataFrame(rows)
    print("\nidentical at 25 Hz, cap = minimum: {}/{} halves x settings".format(int(df.identical_cap_min.sum()), len(df)))
    main_ = df[df.switch_team == 4.0]
    print("identical at 25 Hz, lengths enumerated to the caps (zero cost): {}/{}; largest cost gap {:.2e}".format(
        int(main_.identical_enumerated.sum()), len(main_), main_.cost_gap_enumerated.abs().max()))
    print("5 Hz + refine against exact (team 4, min 10): frames agreeing {:.4f} (min {:.4f}); cost gap mean {:.2f}; "
          "stretches {:.0f} vs {:.0f} per half".format(main_.agree_5hz.mean(), main_.agree_5hz.min(),
                                                       main_.cost_gap_5hz.mean(), main_.runs_5hz.mean(), main_.runs_exact.mean()))
    print("seconds per half: exact {:.2f}, 5 Hz lam 0 {:.2f}, 5 Hz lam 1 with dead lengths {:.2f}, 25 Hz enumerated {:.1f}".format(
        main_.exact_s.mean(), main_.coarse_s.mean(), main_.lam1_s.mean(), main_.enum_s.mean()))
    os.makedirs(lab.RESULTS, exist_ok=True)
    df.to_csv(os.path.join(lab.RESULTS, "lab_{}_check{}.csv".format(NAME, chain.suffix(which))), index=False, float_format="%.6g")


def grid(which):
    rows = []
    for match in lab.each(which):
        dists = distributions(match.id, which)
        cost = -np.log(lab_robust.held_out_proba(match, which))
        for setting in GRID:
            t0 = time.time()
            decoded = decode(match, cost, setting, dists)
            seconds = time.time() - t0
            s = lab.score(match, L.finish(match, decoded, which))
            rows.append({"match": match.id, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                         "possessions": s["possessions"], "spell_f1_decoded": lab.score(match, decoded)["spell_f1"],
                         "decode_s": seconds})
        print("{}: grid scored ({:.0f} s)".format(match.id, sum(r["decode_s"] for r in rows if r["match"] == match.id)), flush=True)
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
        json.dump({"scores": table.round(5).to_dict(), "possessions": {k: {m: v["possessions"] for m, v in s.items()}
                                                                        for k, s in scores.items()},
                   "against_v12": report}, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--check", action="store_true", help="lam = 0 against the min-length decoder")
    parser.add_argument("--chains", action="store_true", help="v12's chain from this start, after the run")
    args = parser.parse_args()
    if args.check:
        return check(args.which)
    if args.chains:
        return chains(args.which)
    g = grid(args.which)
    report = lab.judge(g, COLUMNS, args.which, base=lab_weighted.robust_base(args.which))
    print(lab.describe(report))
    print("picks: {}".format(pd.Series([str(f["picked"]) for f in report["folds"]]).value_counts().to_dict()))
    print("possessions per match with each fold's pick: {:.0f}".format(np.mean([
        g[(g.match == f["match"]) & np.all([g[c] == f["picked"][c] for c in COLUMNS], axis=0)].possessions.iloc[0]
        for f in report["folds"]])))
    means = g.groupby(COLUMNS, sort=False).spell_f1.mean().reset_index()
    print("in sample, best setting's mean at each lam: {}".format(means.groupby("lam").spell_f1.max().round(4).to_dict()))
    print("decode seconds per match and setting: {:.2f}".format(g.groupby(["match"] + COLUMNS).decode_s.sum().mean()))
    lab.save(NAME, g, report, args.which)


if __name__ == "__main__":
    main()
