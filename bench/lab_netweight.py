"""Push 2026-11-15, track 2 `netweight` (docs/lab/push/netweight.md): branch B's GRU leaning harder on Metrica's games.

    python -m bench.lab_netweight draws                        # D3's two new training-stress draws of Metrica's 3, and
                                                               #   the check that this code with seed 1 gives netmet's copy
    python -m bench.lab_netweight repro [--fold metrica1]      # netmet seed 0 through this code path: net and P to the bit
    python -m bench.lab_netweight losscheck                    # the weighted loss line with every weight 1 (information)
    python -m bench.lab_netweight train --cand CAND --fold HELD [--seed S]   # one fold (or `all`), then its held-out P
    python -m bench.lab_netweight screen CAND                  # seeds 0+1 averaged, B rebuilt, judged against ens2 and v15
    python -m bench.lab_netweight final CAND                   # seeds 0-3 averaged, B, SkillCorner, against netmet5 and v15
    python -m bench.lab_netweight fresh CAND                   # seeds 2+3 against netmet's seeds 2+3 (information)
    python -m bench.lab_netweight frames                       # held-out log loss per seed and mean (information)

netmet's recipe (lab_netmirror.fit as lab_netmet.train calls it: leave one match out over all 10, the fold's DFL matches'
normaliser, gru_b's network and settings, the seed) with one change per candidate (CANDIDATES):
  W2, W4  every frame of a Metrica training match (as is and stressed, both orientations) carries loss weight 2 or 4;
          the batch loss is the weighted mean. Same chunks, order, initial weights and dropout as netmet's same seed.
  D3      each Metrica training match under 3 training-stress draws (netmet's TRAIN_SEED copy + seeds 11, 12, drawn
          from the training template pool only); weight 1.
With every weight 1 the loss is netmet's own line, so W1 (the reproduction) is netmet's loop to the bit.
Files: data/external/lab_netweight/<CAND>_s<S>/ (nets/, lab_branch_b/, search_b/), <CAND>_ens2|ens4|s23/ (the means),
draws/ (D3's stress draws); results in bench/results/lab_netweight/<sub>/.

The finalist W4 passed the track's bar (2026-11-16 03:30), so DEPLOY = "W4": its deployment nets are the four `all`
nets, data/external/lab_netweight/W4_s{0,1,2,3}/nets/net_all.pt (also cloned, clearly named, into deploy_W4/). For
bench/train_v16.py the interface is lab_netmet's: fit_recipe(used, norm_used, seed=S, cand="W4", log=...) returns
lab_netmirror.fit's dict; net_path(held, seed, "W4"), root("W4", seed), norm via lab_netmet.norm_for(held); the
4-seed mirror is ("netweight", "W4_ens4"). Nothing is adopted until Jose's go-ahead.

Nothing here reads our match.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from bench import lab, lab_branch_b, lab_carried, lab_netmet, lab_netmirror, lab_robust, sets
from possession import lab_learned_features as llf
from possession import v9, v11

# Params (fixed 10:28, 2026-11-15, before any score: docs/lab/push/netweight.md)
B = lab_branch_b
C = lab_carried
N = lab_netmirror
M = lab_netmet
NAME = "netweight"
WHICH = "all"
TRAIN = C.TRAIN_SEED                         # netmet's training-stress draw (1)
NEW_DRAWS = (11, 12)                         # D3's two new draws (training template pool)
CANDIDATES = {
    "W1": {"weight": 1, "draws": (TRAIN,)},                # netmet itself: the reproduction only
    "W2": {"weight": 2, "draws": (TRAIN,)},
    "W4": {"weight": 4, "draws": (TRAIN,)},
    "D3": {"weight": 1, "draws": (TRAIN,) + NEW_DRAWS},
    "W2D3": {"weight": 2, "draws": (TRAIN,) + NEW_DRAWS},  # the combination: only if its rule fires (the page)
    "W4D3": {"weight": 4, "draws": (TRAIN,) + NEW_DRAWS},
}
SCREEN_SEEDS = (0, 1)
FINAL_SEEDS = (0, 1, 2, 3)
REPRO_FOLD = "metrica1"
DEPLOY = "W4"                                # set 2026-11-16 03:30: the finalist passed the bar (the page's verdict)
DRAWS = os.path.join(N.root_of(NAME, "draws"))
# the bar (the page, fixed before any score)
MET_GAIN, DFL_LOSS_VS_NETMET5, MET_GAIN_VS_NETMET5 = 0.010, 0.003, 0.005


def sub_of(cand, seed):
    return "repro" if cand == "W1" else "{}_s{}".format(cand, seed)


def root(cand, seed=0):
    return N.root_of(NAME, sub_of(cand, seed))


def net_path(held, seed=0, cand=None):
    cand = cand or DEPLOY
    return os.path.join(root(cand, seed), "nets", "net_{}.pt".format(held))


# --------------------------------------------------------------------------------------------
# D3's training-stress draws
# --------------------------------------------------------------------------------------------

def draw_path(m, seed, kind="features"):
    return os.path.join(DRAWS, "{}_{}_{}.npz".format(kind, seed, m))


def make_draw(m, seed, pool):
    """lab_carried.stressed_ball + stressed + stressed_features for one match and stress seed, the training template
    pool given, in memory (nothing written)."""
    match = lab.load(m, WHICH)
    clean = sets.load(m)["ball"]
    ball, table = C.stress_ball(m, match.track["ball"], clean, match.truth, match.halves, match.pitch, seed, pool)
    del clean
    v6, v7 = C.ball_costs(match, ball, WHICH)
    match.track = {**match.track, "ball": ball}
    match.pred = {**match.pred, "v6_cost": v6, "v7_cost": v7}
    match.stressed = seed
    b = lab_robust.blanked(match)
    touches = v11.seen_touches(b.track, v11.SETTINGS[WHICH]["dv"])
    passes = v9.seen_passes(b.track)
    names, X = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes)
    _, Xs = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes, swap=True)
    return {"names": np.array(names), "X": X, "X_swapped": Xs, "truth": b.truth.astype(np.int8),
            "blanked": b.blanked}, ball, table


def draws():
    """The check (seed 1 must give the shared copy exactly), then the new draws of Metrica's 3 into DRAWS."""
    os.makedirs(DRAWS, exist_ok=True)
    started = time.time()
    pool = C.templates("train")
    print("training template pool: {} stoppages from SkillCorner's {} even-indexed matches ({:.0f} s)".format(
        len(pool), len(C.sc_pool("train")), time.time() - started), flush=True)
    out = {}
    m = "metrica1"
    shared_f, shared_b = C.feature_path(m, WHICH, TRAIN), C.stress_path(m, TRAIN)
    if not (os.path.exists(shared_f) and os.path.exists(shared_b)):
        raise RuntimeError("netmet's shared training-stress copy of {} is missing".format(m))
    f, ball, _ = make_draw(m, TRAIN, pool)
    with np.load(shared_f) as z:
        same = {k: bool(np.array_equal(f[k], z[k], equal_nan=f[k].dtype.kind == "f")) for k in ("X", "X_swapped", "truth", "blanked")}
        same["names"] = [str(k) for k in z["names"]] == [str(k) for k in f["names"]]
    with np.load(shared_b) as z:
        same["ball"] = bool(np.array_equal(ball, z["ball"], equal_nan=True))
    out["check_seed1_" + m] = same
    print("check, {} seed {} against the shared copy: {}".format(m, TRAIN, same), flush=True)
    if not all(same.values()):
        raise RuntimeError("the draw code does not reproduce netmet's training-stress copy")
    del f
    for m in lab.matches(WHICH):
        if sets.source(m) != "metrica":
            continue
        for seed in NEW_DRAWS:
            path = draw_path(m, seed)
            if os.path.exists(path):
                continue
            t0 = time.time()
            f, ball, table = make_draw(m, seed, pool)
            np.savez_compressed(path + ".part.npz", **f)          # compressed: the disk is tight
            os.replace(path + ".part.npz", path)
            np.savez_compressed(draw_path(m, seed, "ball"), ball=ball)
            table.to_csv(draw_path(m, seed, "ball").replace(".npz", ".csv"), index=False)
            with np.load(C.stress_path(m, TRAIN)) as z:
                b1 = z["ball"]
            dead = np.asarray(lab.load(m, WHICH).truth) == C.D
            differ = float(np.mean(~np.isclose(ball[dead], b1[dead], equal_nan=True).all(axis=1)))
            out["{}_{}".format(m, seed)] = {"stoppages": int(len(table)), "stressed": int(table.stressed.sum()),
                                           "dead_frames_differing_from_seed1": differ}
            print("{} draw {}: {} stoppages stressed; {:.1%} of dead frames' ball differ from netmet's draw ({:.0f} s)".format(
                m, seed, int(table.stressed.sum()), differ, time.time() - t0), flush=True)
            del f
    with open(os.path.join(DRAWS, "draws.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    return out


def draw_segments(m, norm, seed):
    """lab_branch_b.stressed_segments on one of D3's draws (TRAIN: netmet's own copy, from the shared cache)."""
    if seed == TRAIN:
        return B.stressed_segments(m, norm)
    path = draw_path(m, seed)
    if not os.path.exists(path):
        raise RuntimeError("{} missing: run `python -m bench.lab_netweight draws` first".format(path))
    with np.load(path) as z:
        c = {"X": z["X"], "X_swapped": z["X_swapped"], "truth": z["truth"]}
    truth = c["truth"].astype(np.int8)
    swapped = np.where(truth == 0, 1, np.where(truth == 1, 0, truth)).astype(np.int8)
    out = []
    for key, y in (("X", truth), ("X_swapped", swapped)):
        X = c[key]
        for _, first, last in B.T.halves_of(m, WHICH):
            out.append((B.T.standardised_half(X, first, last, norm, np.float16), y[first:last + 1].copy()))
    del c
    return out


# --------------------------------------------------------------------------------------------
# Training: lab_netmirror.fit (no head) with a per-match loss weight and extra draws for Metrica
# --------------------------------------------------------------------------------------------

def fit(used, norm_used, seed=0, weight=1, draws=(TRAIN,), force_weighted=False, log=print):
    """lab_netmirror.fit line for line without the owner head, with two hooks on Metrica's training matches: a loss
    weight on their frames (the batch loss the weighted mean) and extra training-stress draws after netmet's copy.
    Every weight 1: netmet's own loss line (force_weighted: the weighted line anyway, for the check)."""
    T = B.T
    torch = T.torch_ready()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = T.normaliser(tuple(norm_used), WHICH)
    if tuple(draws)[:1] != (TRAIN,):
        raise ValueError("the draws start with netmet's own copy")
    segs, seg_w = [], []
    for m in used:
        metrica = sets.source(m) == "metrica"
        one = T.segments(m, WHICH, norm) + B.stressed_segments(m, norm)
        if metrica:
            for s in draws[1:]:
                one += draw_segments(m, norm, s)
        segs += one
        seg_w += [float(weight) if metrica else 1.0] * len(one)
    weighted = force_weighted or any(w != 1.0 for w in seg_w)
    n_in = segs[0][0].shape[1] + len(norm["flagged"])
    net = T.network(n_in)
    params = list(net.parameters())
    opt = torch.optim.AdamW(params, lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    lossn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    started, history, history_plain = time.time(), [], []
    for epoch in range(T.EPOCHS):
        for group in opt.param_groups:
            group["lr"] = T.LR * 0.5 * (1 + np.cos(np.pi * epoch / T.EPOCHS))
        chunks = []
        for si, (Z, _) in enumerate(segs):
            o = int(rng.integers(T.STEP))
            k = len(range(o, len(Z), T.STEP))
            shift = int(rng.integers(T.CHUNK))
            for s in range(-shift, k, T.CHUNK):
                a, b = max(s, 0), min(s + T.CHUNK, k)
                if b > a:
                    chunks.append((si, o, a, b))
        order = rng.permutation(len(chunks))
        net.train()
        total, total_plain, count = 0.0, 0.0, 0
        for bi in range(0, len(order), T.BATCH):
            batch = [chunks[i] for i in order[bi:bi + T.BATCH]]
            x = np.zeros((len(batch), T.CHUNK, n_in), np.float32)
            y = np.full((len(batch), T.CHUNK), -100, np.int64)
            w = np.zeros((len(batch), T.CHUNK), np.float32)
            for j, (si, o, a, b) in enumerate(batch):
                Z, t = segs[si]
                idx = o + T.STEP * np.arange(a, b)
                x[j, :b - a] = T.inputs(Z[idx], norm)
                lab_ = t[idx].astype(np.int64)
                lab_[lab_ > 2] = -100
                y[j, :b - a] = lab_
                w[j, :b - a] = seg_w[si]
            logits = net(torch.from_numpy(x))
            if not weighted:
                loss = lossf(logits.reshape(-1, 3), torch.from_numpy(y).reshape(-1))
                main = plain = loss.item()
            else:
                yt = torch.from_numpy(y).reshape(-1)
                each = lossn(logits.reshape(-1, 3), yt)
                valid = (yt != -100).to(each.dtype)
                wt = torch.from_numpy(w).reshape(-1) * valid
                loss = (each * wt).sum() / wt.sum()
                main = loss.item()
                plain = float((each * valid).sum() / valid.sum())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, T.GRAD_CLIP)
            opt.step()
            total += main * len(batch)
            total_plain += plain * len(batch)
            count += len(batch)
        history.append(total / count)
        history_plain.append(total_plain / count)
        log("  epoch {}: train loss {:.4f}{} ({:.0f} s)".format(
            epoch + 1, history[-1], " (unweighted {:.4f})".format(history_plain[-1]) if weighted else "",
            time.time() - started))
    net.eval()
    return {"net": net, "norm": norm, "history": history, "history_plain": history_plain, "history_head": None,
            "seconds": time.time() - started, "trained_on": list(used), "norm_on": list(norm_used), "seed": seed,
            "head": None, "weight": weight, "draws": list(draws), "segments": len(segs)}


def recipe(cand=None):
    cand = cand or DEPLOY
    if cand not in CANDIDATES:
        raise ValueError("unknown candidate {}".format(cand))
    return CANDIDATES[cand]


def fit_recipe(used, norm_used, seed=0, cand=None, log=print):
    """The candidate's recipe with lab_netmirror.fit's calling convention (for bench/train_v16.py to swap in)."""
    r = recipe(cand)
    return fit(used, norm_used, seed=seed, weight=r["weight"], draws=r["draws"], log=log)


def train(held, seed=0, cand=None):
    """lab_netmet.train with the candidate's recipe: one fold (or `all`), then the held-out match's P into its mirror."""
    cand = cand or DEPLOY
    r = recipe(cand)
    path = net_path(held, seed, cand)
    used, norm_used = M.used_for(held), M.norm_for(held)
    if held != "all" and held not in lab.matches(WHICH):
        raise ValueError(held)
    if not os.path.exists(path):
        print("{} seed {} fold {}: training on {} (Metrica's weight {}, draws {}); normaliser on {}".format(
            cand, seed, held, " ".join(used), r["weight"], list(r["draws"]), " ".join(norm_used)), flush=True)
        model = fit(used, norm_used, seed=seed, weight=r["weight"], draws=r["draws"], log=lambda s: print(s, flush=True))
        N.save_net(model, path, extra={"candidate": cand, "weight": r["weight"], "draws": list(r["draws"]),
                                       "history_plain": model["history_plain"], "segments": model["segments"]})
        print("{} seed {} fold {}: saved ({:.0f} s)".format(cand, seed, held, model["seconds"]), flush=True)
    if held != "all":
        N.predict_into(root(cand, seed), held, path, norm_used)


# --------------------------------------------------------------------------------------------
# The checks before any candidate trains
# --------------------------------------------------------------------------------------------

def repro(fold=REPRO_FOLD):
    """netmet seed 0's fold through this code path (W1): the net's weights and held-out P equal netmet's to the bit."""
    torch = B.T.torch_ready()
    train(fold, 0, "W1")
    ours = torch.load(net_path(fold, 0, "W1"))
    theirs = torch.load(M.net_path(fold, 0))
    out = {"fold": fold, "weights_max_abs_diff": max(float((ours[k] - theirs[k]).abs().max()) for k in theirs),
           "weights_same_keys": sorted(ours) == sorted(theirs)}
    for variant in N.BENCH_VARIANTS + ("train",):
        a = np.load(N.p_file(root("W1"), variant, fold))
        b = np.load(N.p_file(M.root(0), variant, fold))
        out["p_" + variant] = {"max_abs_diff": float(np.abs(a.astype(np.float64) - b).max()), "identical": bool(np.array_equal(a, b))}
    out["pass"] = bool(out["weights_same_keys"] and out["weights_max_abs_diff"] == 0
                       and all(v["identical"] for k, v in out.items() if k.startswith("p_")))
    os.makedirs(N.results_of(NAME, "repro"), exist_ok=True)
    with open(os.path.join(N.results_of(NAME, "repro"), "repro.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("REPRODUCTION {}".format("PASSES" if out["pass"] else "FAILS"), flush=True)
    if not out["pass"]:
        raise SystemExit(1)
    return out


def losscheck(used=("J03WN1", "metrica1")):
    """1 epoch on 2 matches: the weighted loss line with every weight 1 against netmet's line (information)."""
    B.T.EPOCHS = 1
    norm_used = tuple(m for m in used if sets.source(m) in B.L.TRAIN_SOURCES)
    a = fit(used, norm_used, seed=0, log=print)
    b = fit(used, norm_used, seed=0, force_weighted=True, log=print)
    sa, sb = a["net"].state_dict(), b["net"].state_dict()
    out = {"used": list(used), "weights_max_abs_diff": max(float((sa[k] - sb[k]).abs().max()) for k in sa),
           "loss_plain": a["history"], "loss_weighted_line": b["history"]}
    with open(os.path.join(N.results_of(NAME, "repro"), "losscheck.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


# --------------------------------------------------------------------------------------------
# Means, the mirror, the comparisons
# --------------------------------------------------------------------------------------------

def combine_roots(roots, sub, meta):
    """The mean of several networks' held-out P (lab_netmirror.average, as ens2 / ens4) into lab_netweight/<sub>/."""
    out = N.root_of(NAME, sub)
    for m in lab.matches(WHICH):
        for variant in N.BENCH_VARIANTS + ("train",):
            p = N.average([np.load(N.p_file(r, variant, m)) for r in roots])
            path = N.p_file(out, variant, m)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, p.astype(np.float32))
    with open(os.path.join(out, "members.json"), "w") as handle:
        json.dump({**meta, "roots": list(roots)}, handle, indent=1)
    print("{}: the mean of {} written".format(sub, meta), flush=True)


def combine(cand, seeds, sub):
    missing = [s for s in seeds if N.missing_inputs(root(cand, s))]
    if missing:
        raise RuntimeError("{}: seeds {} incomplete".format(cand, missing))
    combine_roots([root(cand, s) for s in seeds], sub,
                  {"candidate": cand, "seeds": list(seeds), **{k: (list(v) if isinstance(v, tuple) else v)
                                                               for k, v in recipe(cand).items()},
                   "deployment": [net_path("all", s, cand) for s in seeds]})


def confirm_csv(name, sub):
    return os.path.join(N.results_of(name, sub), "confirm_{}.csv".format(N.TRIAL))


def per_match(path):
    df = pd.read_csv(path)
    d = df[(df.trial == N.TRIAL) & df.variant.isin(N.SB.VARIANTS)]
    return d.groupby("match").spell_f1.mean().reindex(lab.matches(WHICH)), d.groupby("variant").spell_f1.mean()


def summary(m_):
    src = np.array([sets.source(m) for m in m_.index])
    return {"bench": float(m_.mean()), "metrica": float(m_[src == "metrica"].mean()), "dfl": float(m_[src == "idsse"].mean())}


def compare(sub, ref_name, ref_sub, label):
    """The candidate's B against another network's B (both through trial 1015): per match, source, variant."""
    c_m, c_v = per_match(confirm_csv(NAME, sub))
    r_m, r_v = per_match(confirm_csv(ref_name, ref_sub))
    v_m, v_v = per_match(N.REFERENCE)
    g = c_m - r_m
    src = np.array([sets.source(m) for m in g.index])
    out = {"candidate": sub, "reference": label,
           "candidate_abs": summary(c_m), "reference_abs": summary(r_m), "v15_abs": summary(v_m),
           "candidate_vs_v15": {k: summary(c_m)[k] - summary(v_m)[k] for k in ("bench", "metrica", "dfl")},
           "reference_vs_v15": {k: summary(r_m)[k] - summary(v_m)[k] for k in ("bench", "metrica", "dfl")},
           "diff": {"bench": float(g.mean()), "metrica": float(g[src == "metrica"].mean()),
                    "dfl": float(g[src == "idsse"].mean()), "better": int((g > 0).sum()),
                    "metrica_better": int((g[src == "metrica"] > 0).sum()), "dfl_better": int((g[src == "idsse"] > 0).sum())},
           "per_match": g.round(5).to_dict(), "per_variant": (c_v - r_v).round(5).to_dict(),
           "per_match_vs_v15": (c_m - v_m).round(5).to_dict(), "reference_per_match_vs_v15": (r_m - v_m).round(5).to_dict()}
    with open(os.path.join(N.results_of(NAME, sub), "compare_{}.json".format(label)), "w") as handle:
        json.dump(out, handle, indent=1)
    print("{} against {}: B's bench {:+.4f} (better on {}/10), Metrica's 3 {:+.4f} ({}/3), DFL's 7 {:+.4f} ({}/7)".format(
        sub, label, out["diff"]["bench"], out["diff"]["better"], out["diff"]["metrica"], out["diff"]["metrica_better"],
        out["diff"]["dfl"], out["diff"]["dfl_better"]), flush=True)
    return out


def screen(cand):
    """Seeds 0+1 averaged (as ens2), B rebuilt in the mirror; against v15's B (the judge) and against ens2."""
    sub = "{}_ens2".format(cand)
    combine(cand, SCREEN_SEEDS, sub)
    N.confirm(NAME, sub)
    N.judge(NAME, sub)
    return compare(sub, "netmet", "ens2", "ens2")


def final(cand):
    """Seeds 0-3 averaged, B, SkillCorner with the four `all` nets; the bar's parts (a) and (b)."""
    sub = "{}_ens4".format(cand)
    nets = [net_path("all", s, cand) for s in FINAL_SEEDS]
    missing = [p for p in nets if not os.path.exists(p)]
    if missing:
        raise RuntimeError("deployment nets missing: {}".format(missing))
    combine(cand, FINAL_SEEDS, sub)
    N.confirm(NAME, sub)
    N.skillcorner(NAME, sub, nets)
    return bar(cand)


def bar(cand):
    sub = "{}_ens4".format(cand)
    j = N.judge(NAME, sub)
    c = compare(sub, "netmet", "ens4", "netmet5")
    sc = pd.read_csv(os.path.join(N.results_of(NAME, sub), "skillcorner_{}.csv".format(N.TRIAL)), dtype={"match": str})
    ref_sc = pd.read_csv(os.path.join(N.results_of("netmet", "ens4"), "skillcorner_{}.csv".format(N.TRIAL)), dtype={"match": str})
    v15_sc = pd.read_csv(N.REFERENCE_SC, dtype={"match": str})
    a_ = sc[sc.trial == N.TRIAL].set_index("match").spell_f1
    r_ = ref_sc[ref_sc.trial == N.TRIAL].set_index("match").spell_f1
    v_ = v15_sc[v15_sc.trial == N.TRIAL].set_index("match").spell_f1
    from bench import skillcorner as sk
    ids = [str(i) for i in sk.match_ids()]
    even, odd = ids[0::2], ids[1::2]
    sc_info = {"candidate": float(a_.mean()), "netmet5": float(r_.mean()), "v15": float(v_.mean()),
               "vs_v15": float((a_ - v_).mean()), "vs_v15_better": int(((a_ - v_) > 0).sum()),
               "vs_netmet5": float((a_ - r_).mean()), "vs_netmet5_better": int(((a_ - r_) > 0).sum()),
               "even10_vs_v15": float((a_ - v_)[even].mean()), "odd10_vs_v15": float((a_ - v_)[odd].mean()),
               "even10_vs_netmet5": float((a_ - r_)[even].mean()), "odd10_vs_netmet5": float((a_ - r_)[odd].mean())}
    mr = j["metrica_rule"]
    part_a = {"metrica_vs_v15": mr["metrica"], "metrica_ok": mr["metrica"] >= MET_GAIN,
              "metrica_better": mr["metrica_better"], "metrica_better_ok": mr["metrica_better"] == 3,
              "dfl_vs_v15": mr["dfl"], "dfl_ok": mr["dfl"] >= -N.DFL_LOSS,
              "worst_variant_vs_v15": min(j["per_variant"].values()), "variants_ok": bool(j["parts"]["variants"]),
              "skillcorner_vs_v15": j.get("skillcorner_gain"), "skillcorner_ok": bool(j["parts"]["skillcorner"])}
    part_a["passes"] = bool(part_a["metrica_ok"] and part_a["metrica_better_ok"] and part_a["dfl_ok"]
                            and part_a["variants_ok"] and part_a["skillcorner_ok"])
    d = c["diff"]
    part_b = {"metrica_vs_netmet5": d["metrica"], "metrica_ok": d["metrica"] >= MET_GAIN_VS_NETMET5,
              "dfl_vs_netmet5": d["dfl"], "dfl_ok": d["dfl"] >= -DFL_LOSS_VS_NETMET5,
              "bench_vs_netmet5": d["bench"], "bench_ok": d["bench"] >= 0}
    part_b["passes"] = bool(part_b["metrica_ok"] and part_b["dfl_ok"] and part_b["bench_ok"])
    out = {"candidate": cand, "a": part_a, "b": part_b, "passes": bool(part_a["passes"] and part_b["passes"]),
           "full_B_bar_information": {"gain": j["gain"], "better": j["better"], "per_variant": j["per_variant"],
                                      "per_source": j["per_source"], "parts": j["parts"], "passes": j["passes"]},
           "skillcorner": sc_info}
    with open(os.path.join(N.results_of(NAME, sub), "bar.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("THE BAR: (a) {}, (b) {} -> {}".format("passes" if part_a["passes"] else "fails",
                                                 "passes" if part_b["passes"] else "fails",
                                                 "PASSES" if out["passes"] else "FAILS"), flush=True)
    return out


def fresh(cand):
    """Information: the finalist's seeds 2+3 averaged against netmet's seeds 2+3 averaged (no screen selection on them)."""
    if not os.path.exists(confirm_csv(NAME, "netmet_s23")):
        combine_roots([M.root(s) for s in (2, 3)], "netmet_s23", {"candidate": "netmet", "seeds": [2, 3]})
        N.confirm(NAME, "netmet_s23")
    sub = "{}_s23".format(cand)
    combine(cand, (2, 3), sub)
    N.confirm(NAME, sub)
    return compare(sub, NAME, "netmet_s23", "netmet_s23")


def frames(cands):
    """Held-out log loss and accuracy per source: netmet's seeds and the candidates' (information)."""
    roots = {"netmet_s0": M.root(0), "netmet_s1": M.root(1), "netmet_ens2": N.root_of("netmet", "ens2")}
    for cand in cands:
        for s in FINAL_SEEDS:
            if os.path.isdir(root(cand, s)) and not N.missing_inputs(root(cand, s)):
                roots["{}_s{}".format(cand, s)] = root(cand, s)
        for k in ("ens2", "ens4"):
            r = N.root_of(NAME, "{}_{}".format(cand, k))
            if os.path.isdir(r):
                roots["{}_{}".format(cand, k)] = r
    df = N.frame_table(roots)
    df["provider"] = [sets.source(m) for m in df.match]
    os.makedirs(N.results_of(NAME, ""), exist_ok=True)
    df.to_csv(os.path.join(N.results_of(NAME, ""), "frames.csv"), index=False, float_format="%.5f")
    print(df[df.variant.isin(N.SB.VARIANTS)].groupby(["provider", "source"])[["log_loss", "acc"]].mean().round(4).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("draws", "repro", "losscheck", "train", "screen", "final", "bar", "fresh", "frames"))
    parser.add_argument("cand", nargs="?")
    parser.add_argument("--cand", dest="cand_opt")
    parser.add_argument("--fold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cands", nargs="+", default=["W2", "W4", "D3"])
    args = parser.parse_args()
    cand = args.cand_opt or args.cand
    if args.step == "draws":
        return draws()
    if args.step == "repro":
        return repro(args.fold or REPRO_FOLD)
    if args.step == "losscheck":
        return losscheck()
    if args.step == "train":
        return train(args.fold, args.seed, cand)
    if args.step == "screen":
        return screen(cand)
    if args.step == "final":
        return final(cand)
    if args.step == "bar":
        return bar(cand)
    if args.step == "fresh":
        return fresh(cand)
    return frames(args.cands)


if __name__ == "__main__":
    main()
