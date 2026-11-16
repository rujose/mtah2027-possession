"""Night 2026-11-15, builder "netsc2" (docs/lab/night/netsc2.md): netsc's recipe, SkillCorner's frames for dead or live only.

    python -m bench.lab_netsc2 check                         # the loop adds nothing when empty; = netsc's with 3-way SC labels;
                                                             # the 2-way loss against a hand computation
    python -m bench.lab_netsc2 train --fold HELD [--seed S]  # one fold, then its held-out P into the mirror
    python -m bench.lab_netsc2 tables [--seed S]             # trial 1015's stage tables, recomputed in the mirror
    python -m bench.lab_netsc2 confirm [--seed S] [--variants normal p10]
    python -m bench.lab_netsc2 skillcorner [--seed S]        # SkillCorner's 20 with this recipe's metrica1 net
    python -m bench.lab_netsc2 judge [--seed S]              # the robustness bar; the round-8 B bar for information
    python -m bench.lab_netsc2 scframes|frames [--seed S]    # the GRU alone: gru_b, netsc, netsc2 (information)

netsc's recipe exactly (lab_netsc: gru_b's loop, seed, folds and settings; SkillCorner's even 10 appended to every
fold's training, lab_carried.sc_prepared's features as they are, each half in both orientations, cut-away frames out of
the loss, the fold's DFL normaliser), except the loss on SkillCorner's frames: their labels are recoded D -> DEAD,
H or A -> LIVE, and on those frames the loss is the binary cross-entropy of P(D) against P(H) + P(A) (the network's
3-way softmax collapsed to {dead, live}). DFL frames keep gru_b's 3-way loss. Each graded frame counts once (the
batch's loss is the summed loss over the graded frames' count, as netsc's mean). A batch with no SkillCorner frame
takes gru_b's exact code path (CrossEntropyLoss, mean).

The training loop is lab_netmirror.fit copied line for line (owner head removed: netsc didn't use it), with the
SkillCorner segments appended after the last training match's stressed copy (where lab_netsc's wrapper put them) and
the loss split. Files: data/external/lab_netsc2/sS/, bench/results/lab_netsc2/sS/.

SkillCorner open data (github.com/SkillCorner/opendata), MIT; data by SkillCorner. Nothing here reads our match.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from bench import lab, lab_branch_b, lab_carried, lab_gate, lab_netmirror, lab_netsc, sets
from bench import skillcorner as sc

# Params
B = lab_branch_b
N = lab_netmirror
NS = lab_netsc
C = lab_carried
T = B.T
NAME = "netsc2"
WHICH = "all"
TRIAL = 1015
DEAD, LIVE = 10, 11                          # SkillCorner's recoded labels (> 2: out of the 3-way loss)
# the robustness bar (netsc's, docs/lab/night/ideas_outside.md idea 4; fixed before any score)
R_BENCH_LOSS, R_VARIANT_LOSS, R_SOURCE_LOSS = 0.002, 0.003, 0.005
R_SC_GAIN, R_SC_BETTER = 0.010, 7
# the round-8 B bar (docs/lab/night.md), for information
B_GAIN, B_BETTER, B_VARIANT_LOSS, B_SC_LOSS = 0.005, 7, 0.002, 0.005
NETSC_S0 = "s0"


def sub_of(seed):
    return "s{}".format(seed)


def root(seed):
    return N.root_of(NAME, seed if isinstance(seed, str) else sub_of(seed))


def results(seed):
    return N.results_of(NAME, seed if isinstance(seed, str) else sub_of(seed))


def net_path(seed, held):
    return os.path.join(root(seed), "nets", "net_{}.pt".format(held))


# --------------------------------------------------------------------------------------------
# SkillCorner's segments (recoded), the loss, the loop
# --------------------------------------------------------------------------------------------

def recode(y):
    """SkillCorner's 0 H / 1 A / 2 D / 3 X -> LIVE / LIVE / DEAD / 3."""
    return np.where(y == 2, DEAD, np.where((y == 0) | (y == 1), LIVE, y)).astype(np.int8)


def sc_segments(m, norm, binary=True):
    """lab_netsc.sc_segments (it refuses the odd 10), the labels recoded for the 2-way loss if `binary`."""
    out = NS.sc_segments(m, norm)
    if binary:
        out = [(Z, recode(y)) for Z, y in out]
    return out


def loss_of(torch, lossf, logits, codes):
    """The batch's loss. codes int64 [n, CHUNK]: 0-2 DFL's 3-way target, DEAD / LIVE SkillCorner's 2-way target,
    anything else ignored (the 3-way loop's -100 included). With no DEAD / LIVE frame this is gru_b's exact call."""
    y3 = codes.copy()
    y3[(y3 > 2) | (y3 < 0)] = -100
    flat = logits.reshape(-1, 3)
    yb = codes.reshape(-1)
    two = (yb == DEAD) | (yb == LIVE)
    if not two.any():
        return lossf(flat, torch.from_numpy(y3).reshape(-1)), None
    ls = torch.log_softmax(flat, dim=1)
    n3 = int((y3 != -100).sum())
    three_sum = torch.nn.functional.nll_loss(ls, torch.from_numpy(y3).reshape(-1), ignore_index=-100,
                                             reduction="sum")
    idx = torch.from_numpy(np.flatnonzero(two))
    lt = ls.index_select(0, idx)
    dead = torch.from_numpy((yb[two] == DEAD).astype(np.float32))
    log_dead = lt[:, 2]
    log_live = torch.logsumexp(lt[:, :2], dim=1)
    two_sum = -(dead * log_dead + (1 - dead) * log_live).sum()
    loss = (three_sum + two_sum) / (n3 + int(two.sum()))
    return loss, (float(three_sum.item()), n3, float(two_sum.item()), int(two.sum()))


def fit(used, seed=0, sc_ids=None, binary=True, log=print):
    """lab_netmirror.fit (seed, the fold's own normaliser, no head) line for line, with SkillCorner's segments appended
    after the last training match's stressed copy (as lab_netsc's wrapper) and the loss split (loss_of). sc_ids=()
    gives lab_netmirror.fit exactly; binary=False gives lab_netsc.fit exactly."""
    sc_ids = NS.train_ids() if sc_ids is None else list(sc_ids)
    torch = T.torch_ready()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = T.normaliser(used, WHICH)
    segs = []
    for m in used:
        segs += T.segments(m, WHICH, norm)
        segs += B.stressed_segments(m, norm)
    for s in sc_ids:
        started = time.time()
        segs += sc_segments(s, norm, binary)
        log("  SkillCorner {} added ({:.0f} s){}".format(s, time.time() - started, ", dead / live only" if binary else ""))
    n_in = segs[0][0].shape[1] + len(norm["flagged"])
    net = T.network(n_in)
    params = list(net.parameters())
    opt = torch.optim.AdamW(params, lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    started, history, parts = time.time(), [], []
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
        total, count = 0.0, 0
        s3, n3, s2, n2 = 0.0, 0, 0.0, 0
        for bi in range(0, len(order), T.BATCH):
            batch = [chunks[i] for i in order[bi:bi + T.BATCH]]
            x = np.zeros((len(batch), T.CHUNK, n_in), np.float32)
            y = np.full((len(batch), T.CHUNK), -100, np.int64)
            for j, (si, o, a, b) in enumerate(batch):
                Z, t = segs[si]
                idx = o + T.STEP * np.arange(a, b)
                x[j, :b - a] = T.inputs(Z[idx], norm)
                y[j, :b - a] = t[idx].astype(np.int64)
            logits = net(torch.from_numpy(x))
            loss, split = loss_of(torch, lossf, logits, y)
            main = loss.item()
            if split is not None:
                s3, n3, s2, n2 = s3 + split[0], n3 + split[1], s2 + split[2], n2 + split[3]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, T.GRAD_CLIP)
            opt.step()
            total += main * len(batch)
            count += len(batch)
        history.append(total / count)
        parts.append({"dfl_3way_in_mixed": s3 / max(n3, 1), "sc_2way": s2 / max(n2, 1), "sc_frames": n2})
        log("  epoch {}: train loss {:.4f}{} ({:.0f} s)".format(
            epoch + 1, history[-1],
            ", in mixed batches: DFL 3-way {:.4f}, SkillCorner 2-way {:.4f}".format(
                parts[-1]["dfl_3way_in_mixed"], parts[-1]["sc_2way"]) if n2 else "", time.time() - started))
    net.eval()
    return {"net": net, "norm": norm, "history": history, "history_head": None, "parts": parts,
            "seconds": time.time() - started, "trained_on": list(used), "norm_on": list(used), "seed": seed,
            "head": None, "skillcorner": sc_ids, "loss": "dead_live" if binary else "three_way"}


def train(seed, held):
    path = net_path(seed, held)
    used = B.used_for(held)
    if not os.path.exists(path):
        print("netsc2 seed {} fold {}: training on {} (as they are and stressed) + SkillCorner's even 10 {}, "
              "dead / live only".format(seed, held, " ".join(used), " ".join(NS.train_ids())), flush=True)
        model = fit(used, seed=seed, log=lambda s: print(s, flush=True))
        N.save_net(model, path, extra={"skillcorner": model["skillcorner"], "loss": model["loss"],
                                       "parts": model["parts"]})
        print("netsc2 seed {} fold {}: saved ({:.0f} s)".format(seed, held, model["seconds"]), flush=True)
    for m in NS.fold_matches(held):
        N.predict_into(root(seed), m, path, used)


# --------------------------------------------------------------------------------------------
# Checks (before any training): no score involved
# --------------------------------------------------------------------------------------------

def max_diff(a, b):
    sa, sb = a["net"].state_dict(), b["net"].state_dict()
    return max(float((sa[k] - sb[k]).abs().max()) for k in sa)


def check():
    out = {}
    torch = T.torch_ready()
    # (c) the 2-way loss against a hand computation (numpy, float64) on a mixed batch
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(3, 8, 3)).astype(np.float32)
    codes = np.array([[0, 1, 2, -100, DEAD, LIVE, 3, 2]] * 3, np.int64)
    codes[1] = [DEAD, DEAD, LIVE, LIVE, -100, 3, 3, 3]
    codes[2] = [0, 0, 1, 1, 2, 2, -100, -100]
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    got, split = loss_of(torch, lossf, torch.from_numpy(logits), codes)
    lp = logits.astype(np.float64).reshape(-1, 3)
    lp = lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))
    terms = []
    for row, c in zip(lp, codes.reshape(-1)):
        if c in (0, 1, 2):
            terms.append(-row[c])
        elif c == DEAD:
            terms.append(-row[2])
        elif c == LIVE:
            terms.append(-np.log(np.exp(row[0]) + np.exp(row[1])))
    want = float(np.mean(terms))
    only3 = codes.copy()
    only3[only3 > 2] = -100
    got3, split3 = loss_of(torch, lossf, torch.from_numpy(logits), only3)
    want3 = float(lossf(torch.from_numpy(logits).reshape(-1, 3), torch.from_numpy(only3).reshape(-1)).item())
    # on SkillCorner's dead frames the 2-way loss is the 3-way one (-log P(D))
    dead_only = np.full_like(codes, -100)
    dead_only[codes == DEAD] = DEAD
    as3 = dead_only.copy()
    as3[as3 == DEAD] = 2
    out["loss_check"] = {"mixed_got": float(got.item()), "mixed_hand": want,
                         "mixed_abs_diff": abs(float(got.item()) - want), "frames": len(terms),
                         "no_sc_frames_same_call": split3 is None and float(got3.item()) == want3,
                         "dead_frames_equal_3way": abs(float(loss_of(torch, lossf, torch.from_numpy(logits),
                                                                       dead_only)[0].item())
                                                       - float(lossf(torch.from_numpy(logits).reshape(-1, 3),
                                                                     torch.from_numpy(as3).reshape(-1)).item()))}
    print("(c) the loss: {}".format(out["loss_check"]), flush=True)
    epochs = T.EPOCHS
    T.EPOCHS = 1
    try:
        used = ("J03WN1", "J03WOH")
        # (a) no SkillCorner match: lab_netmirror.fit to the bit
        a = N.fit(used, seed=0, log=print)
        b = fit(used, seed=0, sc_ids=(), log=print)
        out["empty_max_abs_diff"] = max_diff(a, b)
        out["empty_history"] = [a["history"], b["history"]]
        print("(a) no SkillCorner match: max |diff| over weights {}".format(out["empty_max_abs_diff"]), flush=True)
        del a, b
        # (b) one SkillCorner match with its 3-way labels: lab_netsc.fit to the bit
        one = NS.train_ids()[:1]
        a = NS.fit(used, seed=0, sc_ids=one, log=print)
        b = fit(used, seed=0, sc_ids=one, binary=False, log=print)
        out["netsc_max_abs_diff"] = max_diff(a, b)
        out["netsc_history"] = [a["history"], b["history"]]
        print("(b) one SkillCorner match, 3-way: max |diff| against lab_netsc.fit {}".format(
            out["netsc_max_abs_diff"]), flush=True)
        del a
        # (b') the same with the 2-way loss: it runs, and it moves the weights
        c = fit(used, seed=0, sc_ids=one, binary=True, log=print)
        out["binary_vs_3way_max_abs_diff"] = max_diff(b, c)
        out["binary_history"] = c["history"]
        out["binary_parts"] = c["parts"]
        print("(b') the same, dead / live only: max |diff| against 3-way {} (must be > 0)".format(
            out["binary_vs_3way_max_abs_diff"]), flush=True)
        del b, c
    finally:
        T.EPOCHS = epochs
    # (d) the recoded labels, on metrica1's normaliser
    norm = T.normaliser(B.used_for(B.held_list()[-1]), WHICH)
    m = NS.train_ids()[0]
    segs3 = NS.sc_segments(m, norm)
    segs2 = sc_segments(m, norm)
    y3 = np.concatenate([t for _, t in segs3])
    y2 = np.concatenate([t for _, t in segs2])
    out["recoded_first_match"] = {
        "match": m, "segments": len(segs2), "frames": int(len(y2)),
        "three_way_counts": {int(k): int((y3 == k).sum()) for k in np.unique(y3)},
        "two_way_counts": {int(k): int((y2 == k).sum()) for k in np.unique(y2)},
        "dead_kept": bool(((y3 == 2) == (y2 == DEAD)).all()), "live_kept": bool((np.isin(y3, (0, 1)) == (y2 == LIVE)).all()),
        "ignored_kept": bool(((y3 == 3) == (y2 == 3)).all()),
        "inputs_same": bool(all(np.array_equal(a[0], b[0], equal_nan=True) for a, b in zip(segs3, segs2)))}
    print(json.dumps(out["recoded_first_match"], indent=1), flush=True)
    os.makedirs(results(0), exist_ok=True)
    out["pass"] = bool(out["empty_max_abs_diff"] == 0.0 and out["netsc_max_abs_diff"] == 0.0
                       and out["binary_vs_3way_max_abs_diff"] > 0 and out["loss_check"]["mixed_abs_diff"] < 1e-5
                       and out["loss_check"]["no_sc_frames_same_call"] and out["loss_check"]["dead_frames_equal_3way"] < 1e-6
                       and out["recoded_first_match"]["dead_kept"] and out["recoded_first_match"]["live_kept"]
                       and out["recoded_first_match"]["ignored_kept"] and out["recoded_first_match"]["inputs_same"])
    with open(os.path.join(results(0), "check.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print("CHECKS {}".format("PASS" if out["pass"] else "FAIL"), flush=True)
    return out


# --------------------------------------------------------------------------------------------
# B rebuilt in the mirror
# --------------------------------------------------------------------------------------------

def tables(seed):
    """Trial 1015's stage tables in the mirror (computed from this recipe's P on the training-stress copies)."""
    sub = seed if isinstance(seed, str) else sub_of(seed)
    N.use_named(NAME, sub)
    missing = N.missing_inputs(root(seed))
    if missing:
        raise RuntimeError("mirror incomplete: {} files missing, e.g. {}".format(len(missing), missing[:3]))
    if os.path.exists(os.path.join(root(seed), "search_b", "tables_trial{}.joblib".format(TRIAL))):
        raise RuntimeError("trial {} tables already in the mirror: never reused, remove them first".format(TRIAL))
    started = time.time()
    trials = N.SB.every_trial()
    t = N.SB.trial_tables(trials[TRIAL])
    print("trial {} stage tables in the mirror: {} stages ({:.0f} s)".format(TRIAL, len(t), time.time() - started),
          flush=True)


def confirm(seed, variants):
    return N.confirm(NAME, sub_of(seed), variants)


def skillcorner_run(seed, net=None):
    sub = seed if isinstance(seed, str) else sub_of(seed)
    return N.skillcorner(NAME, sub, [net or net_path(seed, B.held_list()[-1])])


# --------------------------------------------------------------------------------------------
# The judge
# --------------------------------------------------------------------------------------------

def judge(seed):
    """The robustness bar against v15's B; the B bar for information; netsc (seed 0) beside it, information only."""
    res = results(seed)
    cand = pd.read_csv(os.path.join(res, "confirm_{}.csv".format(TRIAL)))
    ref = pd.read_csv(N.REFERENCE)
    c_m, c_v = N.per_match(cand)
    r_m, r_v = N.per_match(ref)
    g = (c_m - r_m).reindex(lab.matches(WHICH))
    src = pd.Series({m: sets.source(m) for m in g.index})
    per_variant = (c_v - r_v)
    per_source = {s: float(g[src == s].mean()) for s in ("idsse", "metrica")}
    out = {"seed": seed, "bench": float(c_m.mean()), "v15": float(r_m.mean()), "gain": float(g.mean()),
           "better": int((g > 0).sum()), "per_match": g.round(5).to_dict(),
           "per_match_values": c_m.round(5).to_dict(), "per_variant": per_variant.round(5).to_dict(),
           "per_variant_values": c_v.round(5).to_dict(), "per_source": per_source,
           "per_match_variant": cand[(cand.trial == TRIAL) & cand.variant.isin(N.SB.VARIANTS)].pivot_table(
               index="match", columns="variant", values="spell_f1").sub(
               ref[(ref.trial == TRIAL) & ref.variant.isin(N.SB.VARIANTS)].pivot_table(
                   index="match", columns="variant", values="spell_f1")).round(5).to_dict(orient="index"),
           "trial0_check": float((N.per_match(cand, 0)[0] - N.per_match(ref, 0)[0]).abs().max())}
    sc_path = os.path.join(res, "skillcorner_{}.csv".format(TRIAL))
    have_sc = os.path.exists(sc_path)
    if have_sc:
        a = NS.sc_split(sc_path)
        b = NS.sc_split(N.REFERENCE_SC)
        a0 = NS.sc_split(sc_path, 0)
        b0 = NS.sc_split(N.REFERENCE_SC, 0)
        out["trial0_sc_check"] = float((a0 - b0).abs().max())
        for name, ids in (("odd", NS.test_ids()), ("even", NS.train_ids()), ("all", NS.test_ids() + NS.train_ids())):
            d = (a[ids] - b[ids])
            out["sc_" + name] = {"netsc2": float(a[ids].mean()), "v15": float(b[ids].mean()), "gain": float(d.mean()),
                                 "better": int((d > 0).sum()), "n": len(ids), "per_match": d.round(5).to_dict(),
                                 "values": a[ids].round(5).to_dict(), "vs_trial0": float((a[ids] - b0[ids]).mean())}
    r = {"bench": out["gain"] >= -R_BENCH_LOSS,
         "variants": float(per_variant.min()) >= -R_VARIANT_LOSS,
         "sources": min(per_source.values()) >= -R_SOURCE_LOSS,
         "sc_odd_gain": out["sc_odd"]["gain"] >= R_SC_GAIN if have_sc else None,
         "sc_odd_better": out["sc_odd"]["better"] >= R_SC_BETTER if have_sc else None}
    out["robustness_parts"] = r
    out["robustness_part1"] = bool(r["bench"] and r["variants"] and r["sources"])
    out["robustness_part2"] = bool(r["sc_odd_gain"] and r["sc_odd_better"]) if have_sc else None
    out["robustness_passes"] = bool(out["robustness_part1"] and out["robustness_part2"]) if have_sc else None
    bb = {"gain": out["gain"] >= B_GAIN, "better": out["better"] >= B_BETTER,
          "variants": float(per_variant.min()) >= -B_VARIANT_LOSS, "sources": min(per_source.values()) >= 0,
          "sc20": out["sc_all"]["gain"] >= -B_SC_LOSS if have_sc else None,
          "sc_odd10": out["sc_odd"]["gain"] >= -B_SC_LOSS if have_sc else None}
    out["b_bar_parts"] = bb
    out["b_bar_passes"] = bool(all(bb[k] for k in ("gain", "better", "variants", "sources", "sc20"))) if have_sc else None
    # netsc (seed 0) beside it: the same segments and seed, the 3-way loss on SkillCorner (information)
    other = N.results_of("netsc", NETSC_S0)
    p = os.path.join(other, "confirm_{}.csv".format(TRIAL))
    if os.path.exists(p):
        o_m, o_v = N.per_match(pd.read_csv(p))
        d = (c_m - o_m).reindex(lab.matches(WHICH))
        out["vs_netsc"] = {"netsc_bench": float(o_m.mean()), "gain": float(d.mean()), "better": int((d > 0).sum()),
                           "per_variant": (c_v - o_v).round(5).to_dict(),
                           "per_source": {s: float(d[src == s].mean()) for s in ("idsse", "metrica")},
                           "per_match": d.round(5).to_dict()}
        q = os.path.join(other, "skillcorner_{}.csv".format(TRIAL))
        if have_sc and os.path.exists(q):
            o = NS.sc_split(q)
            for name, ids in (("odd", NS.test_ids()), ("even", NS.train_ids())):
                dd = a[ids] - o[ids]
                out["vs_netsc"]["sc_" + name] = {"netsc": float(o[ids].mean()), "gain": float(dd.mean()),
                                                 "better": int((dd > 0).sum()), "per_match": dd.round(5).to_dict()}
    switch = os.path.join(res, "confirm_{}_normal_p10.csv".format(TRIAL))
    if os.path.exists(switch):
        s = pd.read_csv(switch)
        rr = pd.read_csv(N.REFERENCE_SWITCH)
        out["switch_table"] = {v: {"netsc2": float(s[(s.trial == TRIAL) & (s.variant == v)].spell_f1.mean()),
                                   "v15": float(rr[(rr.trial == TRIAL) & (rr.variant == v)].spell_f1.mean())}
                               for v in ("normal", "p10")}
    with open(os.path.join(res, "judge.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------------------------
# Information: the GRU alone (lab_netsc.frame_stats), gru_b against netsc and netsc2
# --------------------------------------------------------------------------------------------

def scframes(seed):
    """The GRU alone on SkillCorner's 20 (deployment nets: gru_b's, netsc's seed 0, this recipe's metrica1 net), the
    cut-aways out of the grading, odd and even apart."""
    held = B.held_list()[-1]
    used = B.used_for(held)
    nets = {"gru_b": os.path.join(N.SHARED_B, "net_{}.pt".format(held)),
            "netsc": os.path.join(N.root_of("netsc", NETSC_S0), "nets", "net_{}.pt".format(held)),
            "netsc2": net_path(seed, held)}
    models = {k: lab_gate.gru_model(used, WHICH, p) for k, p in nets.items()}
    rows = []
    for m in sc.match_ids():
        y, halves, data = NS.sc_labels(m)
        speed = C.ball_speed(data["ball"])
        match, X, Xs = C.sc_prepared(m)
        for k, model in models.items():
            p = lab_gate.gru_proba(model, {"X": X, "X_swapped": Xs}, match.halves)
            rows.append({"match": m, "pool": "odd" if m in NS.test_ids() else "even", "net": k,
                         **NS.frame_stats(p, y, speed)})
        print("{}: {}".format(m, " ".join("{} {:.4f}".format(r["net"], r["log_loss"]) for r in rows[-3:])), flush=True)
        del match, X, Xs, data
    df = pd.DataFrame(rows)
    os.makedirs(results(seed), exist_ok=True)
    df.to_csv(os.path.join(results(seed), "scframes.csv"), index=False, float_format="%.5f")
    print(df.groupby(["pool", "net"])[NS.COLUMNS + ["moving_dead_called_dead"]].mean().round(4).T.to_string())
    return df


def frames(seed):
    """The GRU alone on B's bench (held-out P: gru_b's, netsc's seed 0, this recipe's), B's 3 variants."""
    rows = []
    sources = {"gru_b": None, "netsc": N.root_of("netsc", NETSC_S0), "netsc2": root(seed)}
    for m in lab.matches(WHICH):
        for variant in N.SB.VARIANTS:
            y = np.asarray(lab_gate.variant_features(m, WHICH, variant)["truth"]).astype(np.int64)
            for name, r in sources.items():
                path = N.shared_p_file(variant, m) if r is None else N.p_file(r, variant, m)
                p = np.load(path).astype(np.float64)
                rows.append({"match": m, "source": sets.source(m), "variant": variant, "net": name,
                             **NS.frame_stats(p, y)})
        print("{}: done".format(m), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(results(seed), exist_ok=True)
    df.to_csv(os.path.join(results(seed), "frames.csv"), index=False, float_format="%.5f")
    print(df.groupby(["source", "net"])[NS.COLUMNS].mean().round(4).T.to_string())
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("check", "train", "tables", "confirm", "skillcorner", "judge", "scframes",
                                         "frames"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold")
    parser.add_argument("--variants", nargs="+", default=list(N.SB.VARIANTS), choices=N.BENCH_VARIANTS)
    args = parser.parse_args()
    if args.step == "check":
        return check()
    if args.step == "train":
        return train(args.seed, args.fold)
    if args.step == "tables":
        return tables(args.seed)
    if args.step == "confirm":
        return confirm(args.seed, args.variants)
    if args.step == "skillcorner":
        return skillcorner_run(args.seed)
    if args.step == "judge":
        return judge(args.seed)
    if args.step == "scframes":
        return scframes(args.seed)
    return frames(args.seed)


if __name__ == "__main__":
    main()
