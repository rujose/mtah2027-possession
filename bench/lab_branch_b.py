"""Lab round 8, branch B (docs/lab/branch-b.md): the stoppage switch sends a match whose ball stays tracked and moving
through stoppages to branch B (`augment`'s chain, bench/lab_gate.py). B has no temporal network: its start is robust's
trees retrained with SkillCorner-like stoppage balls. This trains lab_temporal's GRU on the same recipe and tries it as
B's start.

    OMP_NUM_THREADS=2 python -m bench.lab_branch_b train --fold HELD    # ~10 min a fold, < 1.5 GB; folds side by side
    python -m bench.lab_branch_b predict                                 # each fold's P on every bench variant
    python -m bench.lab_branch_b chains VARIANT [--starts ...]           # ~2 min a start: B's chain from each start
    python -m bench.lab_branch_b report                                  # the B bar, every part

THE NETWORK (`gru_b`): lab_temporal.fit's network, settings and seed, trained on each training match's 97 blanked
features as they are AND under augment's training stress (lab_carried's TRAIN_SEED draw: SkillCorner's even-indexed
matches' stoppage balls), every half in both orientations: augment's training matches, read at 5 Hz in 60 s chunks
(so twice A's GRU's chunks an epoch, the same 20 epochs). The normaliser is the fold's normal one
(lab_temporal.normaliser), so its inputs read as A's GRU's do.
Folds as lab_temporal: leave one match out on `all`; Metrica's 3 share the network trained on all 7 IDSSE matches.

THE STARTS (fixed before anything was scored):
  augment   B as built (lab_carried / lab_gate): augment's held-out P
  gru_b     the network's held-out P
  mix_b     the mean of both, as v13.average (both through float32, clipped, rounded)
Each decoded with robust's fold pick for the match (as B), then v9's snap and v11's move (lab_carried.decoded), then
v12's post stages refitted out of fold on the same variant's letters (lab_mix.add_stage), as lab_gate.chains ran B.

THE BENCH: lab_gate's variants. B's own bench is p25, p50 and stressed (the switch sends 5, 10 and 10 of the 10
matches there); normal and p10 are reported for the switch's sake (it sends none there).

THE B BAR (fixed before anything was scored; docs/lab/branch-b.md):
  1. On the mean of p25, p50 and stressed per match: +0.005 or more, better on 7 or more of the 10.
  2. No one of those three variants loses more than 0.002 on the mean.
  3. SkillCorner (deployment models, all 20 matches): no loss beyond 0.005 on the mean.
  Then the switch's bar (docs/lab.md round 7) is rerun with the new B, all four parts.

Nothing here reads our match.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from bench import chain, lab, lab_carried, lab_gate, lab_mix, lab_robust, lab_temporal, sets

# Params
L = lab_robust.L
C = lab_carried
T = lab_temporal
WHICH = "all"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_branch_b")      # gitignored (data/*)
RESULTS = lab.RESULTS
STARTS = ("augment", "gru_b", "mix_b")
B_VARIANTS = ("p25", "p50", "stressed")
VARIANTS = lab_gate.VARIANTS
GAIN = 0.005             # part 1: mean gain over B's three variants
BETTER = 7               # part 1: matches better, of 10
VARIANT_LOSS = 0.002     # part 2
SC_LOSS = 0.005          # part 3


def held_list():
    """The folds: each IDSSE match, and one for Metrica's 3 (metrica1's name, trained on all 7 IDSSE)."""
    ms = lab.matches(WHICH)
    return [m for m in ms if sets.source(m) in L.TRAIN_SOURCES] + [m for m in ms if sets.source(m) not in L.TRAIN_SOURCES][:1]


def used_for(held):
    ms = lab.matches(WHICH)
    return tuple(m for m in ms if m != held and sets.source(m) in L.TRAIN_SOURCES)


def net_path(held):
    return os.path.join(CACHE, "net_{}.pt".format(held))


def net_for(m):
    """The saved network that never saw m."""
    return net_path(m if sets.source(m) in L.TRAIN_SOURCES else held_list()[-1])


# --------------------------------------------------------------------------------------------
# The network, trained on augment's recipe
# --------------------------------------------------------------------------------------------

def stressed_segments(m, norm):
    """lab_temporal.segments on the match under augment's training stress (lab_carried TRAIN_SEED)."""
    c = C.stressed_features(m, WHICH, C.TRAIN_SEED)
    truth = c["truth"].astype(np.int8)
    swapped = np.where(truth == 0, 1, np.where(truth == 1, 0, truth)).astype(np.int8)
    out = []
    for key, y in (("X", truth), ("X_swapped", swapped)):
        X = c[key]
        for _, first, last in T.halves_of(m, WHICH):
            out.append((T.standardised_half(X, first, last, norm, np.float16), y[first:last + 1].copy()))
    del c
    return out


def fit(used, log=print):
    """lab_temporal.fit with each match's stressed copy added after it (the same loop, seed and settings)."""
    torch = T.torch_ready()
    torch.manual_seed(T.SEED)
    rng = np.random.default_rng(T.SEED)
    norm = T.normaliser(used, WHICH)
    segs = []
    for m in used:
        segs += T.segments(m, WHICH, norm)
        segs += stressed_segments(m, norm)
    n_in = segs[0][0].shape[1] + len(norm["flagged"])
    net = T.network(n_in)
    opt = torch.optim.AdamW(net.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    started, history = time.time(), []
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
        for bi in range(0, len(order), T.BATCH):
            batch = [chunks[i] for i in order[bi:bi + T.BATCH]]
            x = np.zeros((len(batch), T.CHUNK, n_in), np.float32)
            y = np.full((len(batch), T.CHUNK), -100, np.int64)
            for j, (si, o, a, b) in enumerate(batch):
                Z, t = segs[si]
                idx = o + T.STEP * np.arange(a, b)
                x[j, :b - a] = T.inputs(Z[idx], norm)
                lab_ = t[idx].astype(np.int64)
                lab_[lab_ > 2] = -100
                y[j, :b - a] = lab_
            logits = net(torch.from_numpy(x))
            loss = lossf(logits.reshape(-1, 3), torch.from_numpy(y).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), T.GRAD_CLIP)
            opt.step()
            total += loss.item() * len(batch)
            count += len(batch)
        history.append(total / count)
        log("  epoch {}: train loss {:.4f} ({:.0f} s)".format(epoch + 1, history[-1], time.time() - started))
    net.eval()
    return {"net": net, "norm": norm, "history": history, "seconds": time.time() - started, "trained_on": list(used)}


def train(held):
    torch = T.torch_ready()
    os.makedirs(CACHE, exist_ok=True)
    used = used_for(held)
    print("{}: training on {} (as they are and stressed)".format(held, " ".join(used)), flush=True)
    model = fit(used, log=lambda s: print(s, flush=True))
    torch.save(model["net"].state_dict(), net_path(held) + ".part")
    os.replace(net_path(held) + ".part", net_path(held))
    with open(net_path(held).replace(".pt", ".json"), "w") as handle:
        json.dump({k: model[k] for k in ("history", "seconds", "trained_on")}, handle, indent=1)
    print("{}: saved ({:.0f} s)".format(held, model["seconds"]), flush=True)


# --------------------------------------------------------------------------------------------
# Held-out P on every variant
# --------------------------------------------------------------------------------------------

def p_path(variant, m):
    return os.path.join(CACHE, "p_gru_b_{}_{}.npy".format(variant, m))


def predict(only=None):
    for m in only or lab.matches(WHICH):
        started = time.time()
        model = lab_gate.gru_model(used_for(m) if sets.source(m) in L.TRAIN_SOURCES else used_for(held_list()[-1]),
                                   WHICH, net_for(m))
        halves = T.halves_of(m, WHICH)
        for variant in VARIANTS:
            if os.path.exists(p_path(variant, m)):
                continue
            p = lab_gate.gru_proba(model, lab_gate.variant_features(m, WHICH, variant), halves)
            np.save(p_path(variant, m), p.astype(np.float32))
        print("{}: gru_b on every variant ({:.0f} s)".format(m, time.time() - started), flush=True)


def gru_b(variant, m):
    return np.load(p_path(variant, m)).astype(np.float64)


def mix(p_gru, p_other, w=0.5):
    """v13.average for w = 0.5 (both through float32, clipped, rounded); v14.mix otherwise."""
    g = p_gru.astype(np.float32).astype(np.float64)
    r = p_other.astype(np.float32).astype(np.float64)
    return np.round(np.clip(w * g + (1 - w) * r, *L.CLIP), L.DECIMALS) if w != 0.5 else \
        np.round(np.clip((g + r) / 2, *L.CLIP), L.DECIMALS)


def start_p(start, variant, m):
    if start == "augment":
        return lab_gate.held_out_p("augment", variant, m, WHICH)
    if start == "gru_b":
        return gru_b(variant, m)
    if start == "mix_b":
        return mix(gru_b(variant, m), lab_gate.held_out_p("augment", variant, m, WHICH))
    raise ValueError(start)


# --------------------------------------------------------------------------------------------
# Chains
# --------------------------------------------------------------------------------------------

def chains_path(variant):
    return os.path.join(RESULTS, "lab_branch_b_chains_{}.json".format(variant))


def chains(variant, starts=STARTS):
    """Each start's chain on one bench variant (lab_gate.chains with B's start swapped); one variant per process."""
    from possession import v12
    lab_mix.cache_short_evidence()          # keyed by match id: one variant per process
    lab_gate.own_dead_cache()
    matches = [lab_gate.variant_match(m, WHICH, variant) for m in lab.matches(WHICH)]
    picks = lab_mix.fold_settings(WHICH)
    path = chains_path(variant)
    done = {}
    if os.path.exists(path):
        with open(path) as handle:
            done = json.load(handle)
    for start in starts:
        started = time.time()
        letters = {m.id: C.decoded(m, start_p(start, variant, m.id), picks["robust"][m.id], WHICH) for m in matches}
        steps = {"start": lab_mix.scores_of(letters, matches)}
        for stage in v12.CHAIN[WHICH]:
            letters = lab_mix.add_stage(stage, letters, matches, picks)
            steps[stage] = lab_mix.scores_of(letters, matches)
        done[start] = {k: {m: v["spell_f1"] for m, v in s.items()} for k, s in steps.items()}
        done[start + " macro"] = {m: v["macro_f1"] for m, v in steps[v12.CHAIN[WHICH][-1]].items()}
        done[start + " possessions"] = {m: v["possessions"] for m, v in steps[v12.CHAIN[WHICH][-1]].items()}
        os.makedirs(os.path.join(CACHE, "letters"), exist_ok=True)
        for m in matches:
            np.save(os.path.join(CACHE, "letters", "{}_{}_{}.npy".format(start, variant, m.id)), letters[m.id])
        print("{} {}: start {:.4f}, chain {:.4f} ({:.0f} s)".format(
            variant, start, np.mean(list(done[start]["start"].values())),
            np.mean(list(done[start][v12.CHAIN[WHICH][-1]].values())), time.time() - started), flush=True)
        with open(path, "w") as handle:
            json.dump(done, handle, indent=1)
    return done


def final(variant):
    from possession import v12
    with open(chains_path(variant)) as handle:
        done = json.load(handle)
    return {s: done[s][v12.CHAIN[WHICH][-1]] for s in STARTS if s in done}


def report():
    rows = []
    for variant in VARIANTS:
        if not os.path.exists(chains_path(variant)):
            continue
        for start, scores in final(variant).items():
            for m, v in scores.items():
                rows.append({"variant": variant, "start": start, "match": m, "spell_f1": v})
    df = pd.DataFrame(rows)
    table = df.pivot_table(index="variant", columns="start", values="spell_f1", aggfunc="mean").reindex(VARIANTS)
    print(table.round(4).to_string())
    out = {"means": table.round(5).to_dict()}
    b = df[df.variant.isin(B_VARIANTS)].pivot_table(index="match", columns="start", values="spell_f1", aggfunc="mean")
    if len(df[df.variant.isin(B_VARIANTS)].variant.unique()) == len(B_VARIANTS):
        for cand in [s for s in STARTS if s != "augment" and s in b]:
            g = b[cand] - b["augment"]
            per_variant = {v: float(table.loc[v, cand] - table.loc[v, "augment"]) for v in B_VARIANTS}
            out[cand] = {"gain": float(g.mean()), "better": int((g > 0).sum()), "per_match": g.round(5).to_dict(),
                         "per_variant": per_variant,
                         "part1": bool(g.mean() >= GAIN and (g > 0).sum() >= BETTER),
                         "part2": bool(min(per_variant.values()) >= -VARIANT_LOSS)}
            print("{}: B bench {:+.4f}, better on {}/10; per variant {}; parts 1-2: {} {}".format(
                cand, g.mean(), (g > 0).sum(), {k: round(v, 4) for k, v in per_variant.items()},
                out[cand]["part1"], out[cand]["part2"]))
    with open(os.path.join(RESULTS, "lab_branch_b.json"), "w") as handle:
        json.dump(out, handle, indent=1, default=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("train", "predict", "chains", "report"))
    parser.add_argument("variant", nargs="?", choices=VARIANTS)
    parser.add_argument("--fold", help="train: the held-out match of the fold (Metrica's: metrica1)")
    parser.add_argument("--starts", nargs="+", default=list(STARTS))
    parser.add_argument("--matches", nargs="+", default=None, help="predict: only these matches")
    args = parser.parse_args()
    if args.step == "train":
        return train(args.fold)
    if args.step == "predict":
        return predict(args.matches)
    if args.step == "chains":
        return chains(args.variant, args.starts)
    return report()


if __name__ == "__main__":
    main()
