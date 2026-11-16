"""Night 2026-11-14, builder "network" (docs/lab/night/network.md): branch B rebuilt for a new GRU, in a small mirror.

    python -m bench.lab_netmirror plumbing                     # gru_b's own P in the mirror: trial 1015 must reproduce
    python -m bench.lab_netmirror confirm NAME SUB [--variants normal p10]
    python -m bench.lab_netmirror skillcorner NAME SUB --nets NET [NET ...]
    python -m bench.lab_netmirror judge NAME SUB [--single]
    python -m bench.lab_netmirror switch NAME SUB              # the switch's check with the mirror's B (information)

Only the file behind B's GRU P changes (lab_branch_b.gru_b and search_b.p_of("gru_b", m, "train")), so B is rebuilt
by pointing those two caches at a mirror of the idea's own, data/external/lab_NAME/SUB/:

    lab_branch_b/p_gru_b_<variant>_<m>.npy    the GRU's held-out P on the bench variants
    search_b/p_gru_b_train_<m>.npy            the GRU's held-out P on each match's training-stress copy
    search_b/tables_trial1015.joblib          trial 1015's out-of-fold stage tables, recomputed here (never reused)
    lab_dead/                                 lab_gate's dead-ball feature cache: read from the shared ones, written here

`augment`'s P on the training-stress copies and trial 0's tables (B as built: no GRU in it) are read from the shared
search_b cache, never written. Results go to bench/results/lab_NAME/SUB/. Trial 1015's settings are fixed (no search).

Nothing here reads our match.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

import joblib
import numpy as np
import pandas as pd

from bench import lab, lab_branch_b, lab_carried, lab_gate, lab_mix, lab_robust, search_b, sets

# Params
B = lab_branch_b
SB = search_b
C = lab_carried
L = lab_robust.L
WHICH = "all"
TRIAL = 1015
SHARED_B = B.CACHE
SHARED_SB = SB.CACHE
REFERENCE = os.path.join(SB.RESULTS, "confirm_1015_554_586_1133_1269_-1_-2.csv")
REFERENCE_SC = os.path.join(SB.RESULTS, "skillcorner_1015_554_586_1133_1269.csv")
REFERENCE_SWITCH = os.path.join(SB.RESULTS, "confirm_1015_-1_normal_p10.csv")
TOLERANCE = 1e-5 + 1e-9
BENCH_VARIANTS = lab_gate.VARIANTS          # normal, p10, p25, p50, stressed
# the bar (docs/lab/night.md, fixed before any score)
GAIN, BETTER, VARIANT_LOSS, SC_LOSS = 0.005, 7, 0.002, 0.005
NOISE = 0.003
METRICA_GAIN, DFL_LOSS = 0.010, 0.005       # netmet's secondary rule (information)

_original_trial_tables = SB.trial_tables


def root_of(name, sub):
    return os.path.join(lab.ROOT, "data", "external", "lab_" + name, sub)


def results_of(name, sub):
    return os.path.join(lab.RESULTS, "lab_" + name, sub)


# --------------------------------------------------------------------------------------------
# The mirror
# --------------------------------------------------------------------------------------------

def train_p_path(kind, m):
    """augment's P on the training-stress copy from the shared cache (read only); the GRU's from the mirror."""
    return os.path.join(SHARED_SB if kind == "augment" else SB.CACHE, "p_{}_train_{}.npy".format(kind, m))


def trial_tables(trial):
    """Trial 0 (B as built, no GRU in it) from the shared cache, read only; every other trial computed in the mirror."""
    if trial["trial"] == 0:
        return joblib.load(os.path.join(SHARED_SB, "tables_trial0.joblib"))
    return _original_trial_tables(trial)


def use(root, results):
    """Point B's GRU caches, search_b's cache and results, and lab_gate's dead-ball cache at the mirror."""
    B.CACHE = os.path.join(root, "lab_branch_b")
    SB.CACHE = os.path.join(root, "search_b")
    SB.RESULTS = results
    for d in (B.CACHE, SB.CACHE, results, os.path.join(root, "lab_dead")):
        os.makedirs(d, exist_ok=True)
    SB.train_p_path = train_p_path
    SB.trial_tables = trial_tables

    def own_dead_cache():
        from bench import lab_dead

        def frame_features(match):
            key = hashlib.sha1(match.pred["p_dead"].tobytes() + match.track["ball"].tobytes()).hexdigest()[:12]
            name = "{}_{}.npy".format(match.id, key)
            for base in (lab_dead.CACHE, os.path.join(C.CACHE, "lab_dead"), os.path.join(lab_gate.CACHE, "lab_dead"),
                         os.path.join(root, "lab_dead")):
                if os.path.exists(os.path.join(base, name)):
                    return lab_dead.FEATURES["ball"], np.load(os.path.join(base, name))
            names, X_ = lab_dead.compute(match)
            np.save(os.path.join(root, "lab_dead", name), X_)
            print("  (lab_dead feature computed into the mirror: {})".format(name), flush=True)
            return names, X_
        lab_dead.frame_features = frame_features
    lab_gate.own_dead_cache = own_dead_cache


def use_named(name, sub):
    use(root_of(name, sub), results_of(name, sub))


def clone(src, dst):
    """An APFS clone (no disk cost; a later write to dst never reaches src); a plain copy elsewhere."""
    if os.path.exists(dst):
        os.remove(dst)
    if subprocess.call(["cp", "-c", src, dst], stderr=subprocess.DEVNULL) != 0:
        shutil.copyfile(src, dst)


def fill_from_gru_b(root):
    """The mirror filled with gru_b's own P (the plumbing check)."""
    os.makedirs(os.path.join(root, "lab_branch_b"), exist_ok=True)
    os.makedirs(os.path.join(root, "search_b"), exist_ok=True)
    for m in lab.matches(WHICH):
        for v in BENCH_VARIANTS:
            name = "p_gru_b_{}_{}.npy".format(v, m)
            clone(os.path.join(SHARED_B, name), os.path.join(root, "lab_branch_b", name))
        name = "p_gru_b_train_{}.npy".format(m)
        clone(os.path.join(SHARED_SB, name), os.path.join(root, "search_b", name))


def average(ps):
    """The ensemble's P: each member through float32 (as stored), the mean, clipped and rounded as every P."""
    mean = np.mean([np.asarray(p, np.float32).astype(np.float64) for p in ps], axis=0)
    return np.round(np.clip(mean, *L.CLIP), L.DECIMALS)


def missing_inputs(root):
    out = []
    for m in lab.matches(WHICH):
        for v in BENCH_VARIANTS:
            if not os.path.exists(os.path.join(root, "lab_branch_b", "p_gru_b_{}_{}.npy".format(v, m))):
                out.append("p_gru_b_{}_{}".format(v, m))
        if not os.path.exists(os.path.join(root, "search_b", "p_gru_b_train_{}.npy".format(m))):
            out.append("p_gru_b_train_{}".format(m))
    return out


# --------------------------------------------------------------------------------------------
# Confirm and SkillCorner in the mirror
# --------------------------------------------------------------------------------------------

def confirm(name, sub, variants=SB.VARIANTS):
    use_named(name, sub)
    missing = missing_inputs(root_of(name, sub))
    if missing:
        raise RuntimeError("mirror {}/{} incomplete: {} files missing, e.g. {}".format(name, sub, len(missing), missing[:3]))
    return SB.confirm([TRIAL], tuple(variants))


def skillcorner(name, sub, nets, trial_ids=(TRIAL,)):
    """search_b.skillcorner with the GRU P the mean of the given deployment networks (one: that network's P), each
    read through the 7 IDSSE matches' normaliser (as gru_b's deployment net), stage models from the mirror's tables."""
    from bench import lab_skillcorner
    from bench import skillcorner as sc
    use_named(name, sub)
    lab_mix.cache_short_evidence()
    trials = SB.every_trial()
    ids = [0] + [i for i in trial_ids if i != 0]
    aug = C.deployment_augment()
    used = B.used_for(B.held_list()[-1])
    grus = [lab_gate.gru_model(used, WHICH, path) for path in nets]
    models = {i: SB.trial_models(trials[i]) for i in ids}
    rows = []
    for m in sc.match_ids():
        gt = sc.load(m)["labels"]
        match, X, Xs = C.sc_prepared(m)
        p_aug = C.proba(aug, {"X": X, "X_swapped": Xs})
        ps = [lab_gate.gru_proba(g, {"X": X, "X_swapped": Xs}, match.halves) for g in grus]
        p_gru = ps[0] if len(ps) == 1 else average(ps)
        for i in ids:
            t = trials[i]
            p = p_aug if t["w_gru"] == 0 else B.mix(p_gru, p_aug, t["w_gru"])
            if t["f_dead"] != 1.0:
                p = p.copy()
                p[:, 2] *= t["f_dead"]
                p = np.round(np.clip(p / p.sum(axis=1, keepdims=True), *L.CLIP), L.DECIMALS)
            letters = L.decode_cost(match, -np.log(p), switch_team=t["switch_team"], switch_dead=t["switch_dead"],
                                    min_possession=int(t["min_possession"]))
            letters = np.asarray(lab_robust.finish(match, letters, WHICH) if t["snap"] else letters).astype("<U1")
            for stage in SB.CHAIN:
                if t[stage]:
                    letters = SB.apply_stage(stage, match, letters, models[i][stage], SB.trial_setting(t, stage))
            s = lab_skillcorner.score(gt, letters)
            rows.append({"match": m, "trial": i, "spell_f1": s["spell_f1"], "possessions": s.get("possessions")})
        print("{}: {}".format(m, " ".join("{:.4f}".format(r["spell_f1"]) for r in rows[-len(ids):])), flush=True)
        del match, X, Xs
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SB.RESULTS, "skillcorner_{}.csv".format("_".join(str(i) for i in ids[1:]))), index=False,
              float_format="%.5f")
    per = df.pivot_table(index="match", columns="trial", values="spell_f1")
    for i in ids[1:]:
        print("trial {}: SkillCorner {:.4f} (trial 0: {:.4f})".format(i, per[i].mean(), per[0].mean()), flush=True)
    return df


# --------------------------------------------------------------------------------------------
# The plumbing check and the judge
# --------------------------------------------------------------------------------------------

def compare(df, ref, keys, trials=(0, TRIAL)):
    ref = ref[ref.trial.isin(trials)].set_index(keys)["spell_f1"]
    got = df[df.trial.isin(trials)].set_index(keys)["spell_f1"].round(5)
    both = pd.concat([ref.rename("ref"), got.rename("got")], axis=1)
    diff = (both.ref - both.got).abs()
    return {"units": int(len(both)), "missing": int(both.isna().any(axis=1).sum()), "max_abs_diff": float(diff.max()),
            "pass": bool(both.notna().all().all() and diff.max() <= TOLERANCE)}


def plumbing():
    name, sub = "netmirror", "plumb"
    root = root_of(name, sub)
    fill_from_gru_b(root)
    df = confirm(name, sub)
    ref = pd.read_csv(REFERENCE)
    out = {"confirm": compare(df, ref, ["trial", "match", "variant"])}
    for t in (0, TRIAL):
        out["bench_trial{}".format(t)] = float(df[df.trial == t].groupby("match").spell_f1.mean().mean())
    print("plumbing, confirm: {}".format(out["confirm"]), flush=True)
    sc = skillcorner(name, sub, [os.path.join(SHARED_B, "net_{}.pt".format(B.held_list()[-1]))])
    out["skillcorner"] = compare(sc.assign(match=sc.match.astype(str)), pd.read_csv(REFERENCE_SC, dtype={"match": str}),
                                 ["trial", "match"])
    print("plumbing, SkillCorner: {}".format(out["skillcorner"]), flush=True)
    out["pass"] = out["confirm"]["pass"] and out["skillcorner"]["pass"]
    with open(os.path.join(results_of(name, sub), "plumbing.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print("PLUMBING {}".format("PASSES" if out["pass"] else "FAILS"), flush=True)
    return out


def per_match(df, trial=TRIAL):
    d = df[(df.trial == trial) & df.variant.isin(SB.VARIANTS)]
    return d.groupby("match").spell_f1.mean(), d.groupby("variant").spell_f1.mean()


def judge(name, sub, single=False, noise=None):
    """The night's bar for the mirror's trial 1015 against v15's B (trial 1015 in the round-8 confirm)."""
    res = results_of(name, sub)
    cand = pd.read_csv(os.path.join(res, "confirm_{}.csv".format(TRIAL)))
    ref = pd.read_csv(REFERENCE)
    c_m, c_v = per_match(cand)
    r_m, r_v = per_match(ref)
    g = (c_m - r_m).reindex(lab.matches(WHICH))
    src = pd.Series({m: sets.source(m) for m in g.index})
    out = {"bench": float(c_m.mean()), "v15": float(r_m.mean()), "gain": float(g.mean()), "better": int((g > 0).sum()),
           "per_match": g.round(5).to_dict(), "per_variant": (c_v - r_v).round(5).to_dict(),
           "per_source": {s: float(g[src == s].mean()) for s in ("idsse", "metrica")},
           "trial0_check": float((per_match(cand, 0)[0] - per_match(ref, 0)[0]).abs().max())}
    sc_path = os.path.join(res, "skillcorner_{}.csv".format(TRIAL))
    if os.path.exists(sc_path):
        sc = pd.read_csv(sc_path, dtype={"match": str})
        rsc = pd.read_csv(REFERENCE_SC, dtype={"match": str})
        a = sc[sc.trial == TRIAL].set_index("match").spell_f1
        b = rsc[rsc.trial == TRIAL].set_index("match").spell_f1
        out["skillcorner"] = float(a.mean())
        out["skillcorner_gain"] = float((a - b).mean())
        out["skillcorner_better"] = int(((a - b) > 0).sum())
    parts = {"gain": out["gain"] >= GAIN, "better": out["better"] >= BETTER,
             "variants": min(out["per_variant"].values()) >= -VARIANT_LOSS,
             "sources": min(out["per_source"].values()) >= 0,
             "skillcorner": out.get("skillcorner_gain", -1) >= -SC_LOSS if "skillcorner_gain" in out else None}
    out["parts"] = parts
    out["passes"] = bool(all(v for v in parts.values() if v is not None)) and parts["skillcorner"] is not None
    met = g[src == "metrica"]
    dfl = g[src == "idsse"]
    out["metrica_rule"] = {"metrica": float(met.mean()), "metrica_better": int((met > 0).sum()), "dfl": float(dfl.mean()),
                           "passes": bool(met.mean() >= METRICA_GAIN and (met > 0).all() and dfl.mean() >= -DFL_LOSS
                                          and parts["variants"] and parts["skillcorner"])}
    if single and noise is not None and noise >= NOISE:
        out["noise_rule"] = "pending: seed noise {:.4f} >= {}; needs the same recipe with seed 1".format(noise, NOISE)
    switch = os.path.join(res, "confirm_{}_normal_p10.csv".format(TRIAL))
    if os.path.exists(switch):
        s = pd.read_csv(switch)
        r = pd.read_csv(REFERENCE_SWITCH)
        out["switch_table"] = {v: {"candidate": float(s[(s.trial == TRIAL) & (s.variant == v)].spell_f1.mean()),
                                   "v15": float(r[(r.trial == TRIAL) & (r.variant == v)].spell_f1.mean())}
                               for v in ("normal", "p10")}
    with open(os.path.join(res, "judge.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1))
    return out


def switch(name, sub):
    """The stoppage switch's four-part check (bench/check_v15_switch.py, threshold unchanged) with the mirror's trial
    1015 as branch B: its confirm files (B's 3 variants, normal and p10) and SkillCorner file are read from the
    mirror's results; branch A, the signals and the threshold are the shared ones, read only. Output: the mirror's
    results/switch/report_1015.json. Information only (round 8 ran it for trial 1015)."""
    import sys
    from bench import check_v15_switch as CS
    CS.SEARCH = results_of(name, sub)
    CS.RESULTS = os.path.join(results_of(name, sub), "switch")
    argv, sys.argv = sys.argv, ["check_v15_switch", "--trial", str(TRIAL)]
    try:
        CS.main()
    finally:
        sys.argv = argv


# --------------------------------------------------------------------------------------------
# Training: gru_b's recipe (lab_branch_b.fit) with hooks, and held-out P into a mirror
# --------------------------------------------------------------------------------------------

def fit(used, norm_used=None, seed=0, owner=None, owner_weight=0.5, log=print):
    """lab_branch_b.fit line for line, with three hooks: the seed (lab_branch_b.fit reads lab_temporal.SEED), the
    matches the normaliser reads (default: the training matches themselves, as gru_b), and an optional training-only
    head (`owner`: {match: int8 target per frame, 0 home / 1 away / 2 nobody / -100 ignored}) on the GRU's output,
    built with torch's random state saved and restored, so the main net's initial weights, its dropout draws and the
    batch order are those of the same recipe without the head."""
    T = B.T
    torch = T.torch_ready()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = T.normaliser(tuple(norm_used) if norm_used is not None else used, WHICH)
    segs, extra = [], []
    for m in used:
        segs += T.segments(m, WHICH, norm)
        segs += B.stressed_segments(m, norm)
        if owner is not None:
            t = owner[m]
            swapped = np.where(t == 0, 1, np.where(t == 1, 0, t)).astype(np.int16)
            halves = T.halves_of(m, WHICH)
            one = [y[first:last + 1].copy() for y in (t, swapped) for _, first, last in halves]
            extra += one + one              # as is, then the training-stress copy (only the ball differs)
    n_in = segs[0][0].shape[1] + len(norm["flagged"])
    net = T.network(n_in)
    head = None
    if owner is not None:
        state = torch.get_rng_state()
        head = torch.nn.Linear(2 * T.HIDDEN, 3)
        torch.set_rng_state(state)
    params = list(net.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(params, lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    started, history, history_head = time.time(), [], []
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
        total, total_head, count = 0.0, 0.0, 0
        for bi in range(0, len(order), T.BATCH):
            batch = [chunks[i] for i in order[bi:bi + T.BATCH]]
            x = np.zeros((len(batch), T.CHUNK, n_in), np.float32)
            y = np.full((len(batch), T.CHUNK), -100, np.int64)
            y2 = np.full((len(batch), T.CHUNK), -100, np.int64)
            for j, (si, o, a, b) in enumerate(batch):
                Z, t = segs[si]
                idx = o + T.STEP * np.arange(a, b)
                x[j, :b - a] = T.inputs(Z[idx], norm)
                lab_ = t[idx].astype(np.int64)
                lab_[lab_ > 2] = -100
                y[j, :b - a] = lab_
                if head is not None:
                    y2[j, :b - a] = extra[si][idx]
            if head is None:
                logits = net(torch.from_numpy(x))
                loss = lossf(logits.reshape(-1, 3), torch.from_numpy(y).reshape(-1))
                main = loss.item()
            else:
                h, _ = net.gru(net.inp(torch.from_numpy(x)))
                logits = net.out(h)
                loss = lossf(logits.reshape(-1, 3), torch.from_numpy(y).reshape(-1))
                main = loss.item()
                if (y2 != -100).any():
                    aux = lossf(head(h).reshape(-1, 3), torch.from_numpy(y2).reshape(-1))
                    loss = loss + owner_weight * aux
                    total_head += aux.item() * len(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, T.GRAD_CLIP)
            opt.step()
            total += main * len(batch)
            count += len(batch)
        history.append(total / count)
        history_head.append(total_head / count)
        log("  epoch {}: train loss {:.4f}{} ({:.0f} s)".format(
            epoch + 1, history[-1], ", head {:.4f}".format(history_head[-1]) if head is not None else "",
            time.time() - started))
    net.eval()
    return {"net": net, "norm": norm, "history": history, "history_head": history_head if head is not None else None,
            "seconds": time.time() - started, "trained_on": list(used), "norm_on": list(norm_used or used),
            "seed": seed, "head": head}


def save_net(model, path, extra=None):
    torch = B.T.torch_ready()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model["net"].state_dict(), path + ".part")
    os.replace(path + ".part", path)
    if model.get("head") is not None:
        torch.save(model["head"].state_dict(), path.replace(".pt", "_head.pt"))
    with open(path.replace(".pt", ".json"), "w") as handle:
        json.dump({**{k: model[k] for k in ("history", "history_head", "seconds", "trained_on", "norm_on", "seed")},
                   **(extra or {})}, handle, indent=1)


def predict_into(root, m, net_path, norm_used):
    """A match's held-out P from one network: every bench variant into root/lab_branch_b, the training-stress copy
    into root/search_b (as lab_branch_b.predict and search_b.stress)."""
    started = time.time()
    model = lab_gate.gru_model(tuple(norm_used), WHICH, net_path)
    halves = B.T.halves_of(m, WHICH)
    os.makedirs(os.path.join(root, "lab_branch_b"), exist_ok=True)
    os.makedirs(os.path.join(root, "search_b"), exist_ok=True)
    for variant in BENCH_VARIANTS:
        p = lab_gate.gru_proba(model, lab_gate.variant_features(m, WHICH, variant), halves)
        np.save(os.path.join(root, "lab_branch_b", "p_gru_b_{}_{}.npy".format(variant, m)), p.astype(np.float32))
    c = C.stressed_features(m, WHICH, C.TRAIN_SEED)
    p = lab_gate.gru_proba(model, c, halves)
    np.save(os.path.join(root, "search_b", "p_gru_b_train_{}.npy".format(m)), p.astype(np.float32))
    print("{}: held-out P on 5 variants and the training-stress copy ({:.0f} s)".format(m, time.time() - started),
          flush=True)


def p_file(root, variant, m):
    if variant == "train":
        return os.path.join(root, "search_b", "p_gru_b_train_{}.npy".format(m))
    return os.path.join(root, "lab_branch_b", "p_gru_b_{}_{}.npy".format(variant, m))


def shared_p_file(variant, m):
    if variant == "train":
        return os.path.join(SHARED_SB, "p_gru_b_train_{}.npy".format(m))
    return os.path.join(SHARED_B, "p_gru_b_{}_{}.npy".format(variant, m))


def frame_table(roots):
    """Held-out log loss and argmax accuracy of each P source ({name: root, or None for gru_b}), per variant and
    match, on the graded frames (truth only to report)."""
    rows = []
    for m in lab.matches(WHICH):
        for variant in BENCH_VARIANTS + ("train",):
            c = C.stressed_features(m, WHICH, C.TRAIN_SEED) if variant == "train" else \
                lab_gate.variant_features(m, WHICH, variant)
            truth = np.asarray(c["truth"])
            del c
            for name, root in roots.items():
                path = shared_p_file(variant, m) if root is None else p_file(root, variant, m)
                if not os.path.exists(path):
                    continue
                f = B.T.frame_check(np.load(path).astype(np.float64), truth)
                rows.append({"source": name, "match": m, "variant": variant, **f})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("plumbing", "confirm", "skillcorner", "judge", "switch"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("sub", nargs="?")
    parser.add_argument("--variants", nargs="+", default=list(SB.VARIANTS), choices=BENCH_VARIANTS)
    parser.add_argument("--nets", nargs="+", default=[])
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--noise", type=float, default=None)
    args = parser.parse_args()
    if args.step == "plumbing":
        return plumbing()
    if args.step == "confirm":
        return confirm(args.name, args.sub, args.variants)
    if args.step == "skillcorner":
        return skillcorner(args.name, args.sub, args.nets)
    if args.step == "switch":
        return switch(args.name, args.sub)
    return judge(args.name, args.sub, args.single, args.noise)


if __name__ == "__main__":
    main()
