"""Lab candidate `temporal` (round 7): a small bidirectional GRU over robust's features at 5 Hz (docs/lab/temporal.md).

    OMP_NUM_THREADS=2 python -m bench.lab_temporal --train [--set all|idsse]   # ~35 / ~30 min, < 1 GB: one network per fold
    python -m bench.lab_temporal [--set all|idsse]            # ~15 min: both variants decoded over robust's grid, judged against robust
    python -m bench.lab_temporal --chains [--set all|idsse]   # ~30 min: v12's chain from robust and from each variant, against v12

The leading public entries put a temporal network (a bidirectional GRU, or a temporal CNN seeing about
+-12 s) over per-frame features, then a Viterbi decoder. Robust is a per-frame tree model whose only
context is hand-made 1 s and 3 s windows. Here a 2-layer BiGRU reads robust's 97 blanked features
(standardised on the fold's training matches, clipped at +-5 sd, missing -> 0 plus a missing flag)
every 5th frame, so a 60 s chunk is 300 steps; each of the 5 phases (frames o, o+5, ...) is its own
sequence, so every frame gets its own P without upsampling. Trained on IDSSE only, each half with the
teams as they are and exchanged (labels H <-> A); P is the mean of both, as robust's. Leave one match
out; Metrica's 3 share the network trained on all 7 IDSSE matches. Two variants, fixed before any run:

  temporal   the network's P
  average    the mean of the network's P and robust's held-out P

Each is decoded over robust's decoder grid (team switch x dead switch x minimum), then v9's snap and
v11's move (`L.finish`), judged against robust; then as the start of v12's chain against v12.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from bench import chain, lab, lab_robust, lab_weighted, sets

# Params (fixed before any run; docs/lab/temporal.md gives the why of each)
L = lab_robust.L
NAME = "temporal"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_temporal")    # gitignored (data/*)
STEP = 5                 # frames between the network's steps: 25 Hz -> 5 Hz
CHUNK = 300              # steps (60 s): one training sequence, one prediction window
STRIDE = 150             # steps between prediction windows; overlaps averaged with triangular weights
Z_CLIP = 5.0             # standardised features clipped to +-5 sd
MISSING_SHARE = 0.001    # a feature missing on more of the training frames than this gets a missing flag
PROJ, HIDDEN, LAYERS, DROPOUT = 64, 64, 2, 0.2
EPOCHS = 20              # fixed from a one-fold timing test (9 s an epoch; time only, no held-out number seen)
BATCH = 16
LR = 1e-3                # AdamW, cosine decay to 0 over the epochs
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SEED = 0
THREADS = 2
VARIANTS = ("temporal", "average")
COLUMNS = ["switch_team", "switch_dead", "min_possession"]
GRID = [dict(zip(COLUMNS, (team, dead, k))) for team in L.SWITCH_TEAM for dead in L.SWITCH_DEAD for k in L.MIN_POSSESSION]


def folder(which):
    return CACHE + chain.suffix(which)


def halves_of(m, which):
    with np.load(os.path.join(lab.folder(which), "{}.npz".format(m))) as z:
        return [tuple(int(x) for x in h) for h in z["halves"]]


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------

def match_stats(m, which):
    """Per feature, over the match's in-half frames as is and swapped: count seen, sum, sum of squares, missing."""
    path = os.path.join(folder(which), "stats_{}.npz".format(m))
    if not os.path.exists(path):
        c = lab_robust.cached(m, which)
        inside = np.zeros(len(c["truth"]), bool)
        for _, first, last in halves_of(m, which):
            inside[first:last + 1] = True
        out = {"n": 0, "seen": 0, "sum": 0, "sq": 0}
        rows = np.flatnonzero(inside)
        for key in ("X", "X_swapped"):
            for a in range(0, len(rows), 20000):          # in blocks: float64 copies stay small
                X = c[key][rows[a:a + 20000]].astype(np.float64)
                ok = ~np.isnan(X)
                out["n"] += len(X)
                out["seen"] = out["seen"] + ok.sum(axis=0)
                out["sum"] = out["sum"] + np.where(ok, X, 0).sum(axis=0)
                out["sq"] = out["sq"] + np.where(ok, X * X, 0).sum(axis=0)
        os.makedirs(folder(which), exist_ok=True)
        np.savez(path, **out)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def normaliser(used, which):
    """Mean, sd and the flagged features, from the training matches only."""
    s = [match_stats(m, which) for m in used]
    n = sum(int(x["n"]) for x in s)
    seen = sum(x["seen"] for x in s)
    mean = sum(x["sum"] for x in s) / np.maximum(seen, 1)
    var = sum(x["sq"] for x in s) / np.maximum(seen, 1) - mean ** 2
    sd = np.sqrt(np.maximum(var, 0))
    sd[sd < 1e-6] = 1.0
    flagged = np.flatnonzero(1 - seen / n > MISSING_SHARE)
    return {"mean": mean.astype(np.float32), "sd": sd.astype(np.float32), "flagged": flagged}


def standardised(X, norm):
    """float32 [n, 97]: (X - mean) / sd clipped at +-Z_CLIP, NaN kept."""
    Z = (X - norm["mean"]) / norm["sd"]
    return np.clip(Z, -Z_CLIP, Z_CLIP, out=Z)


def inputs(Z, norm):
    """float32 [n, 97 + flags]: NaN -> 0, plus a missing flag per flagged feature."""
    Z = Z.astype(np.float32)
    flags = np.isnan(Z[:, norm["flagged"]]).astype(np.float32)
    return np.hstack([np.nan_to_num(Z, nan=0.0), flags])


def member(m, which, key):
    """One array of robust's cached features (lab_robust.cached's file), loaded alone to keep memory low."""
    if not os.path.exists(os.path.join(lab_robust.folder(which), "{}.npz".format(m))):
        raise RuntimeError("run `python -m bench.lab_robust --set {}` first".format(which))
    with np.load(os.path.join(lab_robust.folder(which), "{}.npz".format(m))) as z:
        return z[key]


def standardised_half(X, first, last, norm, dtype=np.float32):
    """standardised(X[first:last + 1]) in blocks, into one array of `dtype`."""
    Z = np.empty((last + 1 - first, X.shape[1]), dtype)
    for a in range(first, last + 1, 20000):
        b = min(a + 20000, last + 1)
        Z[a - first:b - first] = standardised(X[a:b], norm)
    return Z


def segments(m, which, norm):
    """Training sequences of one match: per orientation (as is, swapped) and half, (Z float16 [frames, 97], y int8)."""
    truth = member(m, which, "truth").astype(np.int8)
    swapped = np.where(truth == 0, 1, np.where(truth == 1, 0, truth)).astype(np.int8)
    out = []
    for key, y in (("X", truth), ("X_swapped", swapped)):
        X = member(m, which, key)
        for _, first, last in halves_of(m, which):
            out.append((standardised_half(X, first, last, norm, np.float16), y[first:last + 1].copy()))
        del X
    return out


# --------------------------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------------------------

def torch_ready():
    import torch
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    return torch


def network(n_in):
    torch = torch_ready()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, PROJ), nn.ReLU(), nn.Dropout(DROPOUT))
            self.gru = nn.GRU(PROJ, HIDDEN, num_layers=LAYERS, batch_first=True, bidirectional=True, dropout=DROPOUT)
            self.out = nn.Linear(2 * HIDDEN, 3)

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.out(h)

    return Net()


def fit(used, which, log=print):
    """One network on these (IDSSE) matches: EPOCHS passes, each over every half in both orientations,
    cut into CHUNK-step sequences at a random phase and offset, shuffled, BATCH at a time."""
    torch = torch_ready()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    norm = normaliser(used, which)
    segs = []
    for m in used:
        segs += segments(m, which, norm)
    n_in = segs[0][0].shape[1] + len(norm["flagged"])
    net = network(n_in)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    started, history = time.time(), []
    for epoch in range(EPOCHS):
        for group in opt.param_groups:
            group["lr"] = LR * 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS))
        chunks = []
        for si, (Z, _) in enumerate(segs):
            o = int(rng.integers(STEP))
            k = len(range(o, len(Z), STEP))
            shift = int(rng.integers(CHUNK))
            for s in range(-shift, k, CHUNK):
                a, b = max(s, 0), min(s + CHUNK, k)
                if b > a:
                    chunks.append((si, o, a, b))
        order = rng.permutation(len(chunks))
        net.train()
        total, count = 0.0, 0
        for bi in range(0, len(order), BATCH):
            batch = [chunks[i] for i in order[bi:bi + BATCH]]
            x = np.zeros((len(batch), CHUNK, n_in), np.float32)
            y = np.full((len(batch), CHUNK), -100, np.int64)
            for j, (si, o, a, b) in enumerate(batch):
                Z, t = segs[si]
                idx = o + STEP * np.arange(a, b)
                x[j, :b - a] = inputs(Z[idx], norm)
                lab_ = t[idx].astype(np.int64)
                lab_[lab_ > 2] = -100
                y[j, :b - a] = lab_
            logits = net(torch.from_numpy(x))
            loss = lossf(logits.reshape(-1, 3), torch.from_numpy(y).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            total += loss.item() * len(batch)
            count += len(batch)
        history.append(total / count)
        log("  epoch {}: train loss {:.4f} ({:.0f} s)".format(epoch + 1, history[-1], time.time() - started))
    net.eval()
    return {"net": net, "norm": norm, "history": history, "seconds": time.time() - started, "trained_on": list(used),
            "chunks": len(chunks)}


def run_half(net, Z, norm):
    """P [frames, 3] for one half (standardised, NaN kept): each phase its own 5 Hz sequence, read in
    CHUNK-step windows every STRIDE steps, overlaps averaged with triangular weights."""
    torch = torch_ready()
    out = np.zeros((len(Z), 3))
    for o in range(STEP):
        idx = np.arange(o, len(Z), STEP)
        x = inputs(Z[idx], norm)
        k = len(idx)
        w = min(CHUNK, k)
        starts = list(range(0, k - w + 1, STRIDE))
        if starts[-1] + w < k:
            starts.append(k - w)
        with torch.no_grad():
            prob = torch.softmax(net(torch.from_numpy(np.stack([x[s:s + w] for s in starts]))), dim=-1).numpy()
        tri = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1)).astype(np.float64)
        acc, weight = np.zeros((k, 3)), np.zeros(k)
        for s, p in zip(starts, prob):
            acc[s:s + w] += p * tri[:, None]
            weight[s:s + w] += tri
        out[idx] = acc / weight[:, None]
    return out


def proba(m, model, which):
    """P(H, A, D) per frame: the mean of the match as is and with the teams exchanged, clipped and rounded
    as robust's; 1/3 each outside the halves (the decoder never reads them)."""
    norm, halves = model["norm"], halves_of(m, which)
    p = np.full((len(member(m, which, "truth")), 3), 1 / 3)
    for _, first, last in halves:
        p[first:last + 1] = 0.0
    for key, cols in (("X", np.arange(3)), ("X_swapped", L.SWAP)):     # the mean of both orientations
        X = member(m, which, key)
        for _, first, last in halves:
            p[first:last + 1] += run_half(model["net"], standardised_half(X, first, last, norm), norm)[:, cols] / 2
        del X
    return np.round(np.clip(p, *L.CLIP), L.DECIMALS)


def held_out(m, which, variant="temporal"):
    """The saved held-out P of a variant (`average`: with robust's held-out P)."""
    p = np.load(os.path.join(folder(which), "p_{}.npy".format(getattr(m, "id", m)))).astype(np.float64)
    if variant == "average":
        p = np.round(np.clip((p + lab_robust.held_out_proba(m, which, "blank")) / 2, *L.CLIP), L.DECIMALS)
    return p


def frame_check(p, truth):
    graded = np.isin(truth, (0, 1, 2))
    y = truth[graded].astype(np.int64)
    q = p[graded]
    return {"log_loss": float(-np.log(q[np.arange(len(y)), y]).mean()), "acc": float((q.argmax(axis=1) == y).mean())}


def train_all(which):
    """Each match's P from a network trained without it; the training log and frame checks (truth only to report)."""
    torch = torch_ready()
    os.makedirs(folder(which), exist_ok=True)
    ms = lab.matches(which)
    models, rows = {}, []
    for held in ms:
        used = tuple(m for m in ms if m != held and sets.source(m) in L.TRAIN_SOURCES)
        path = os.path.join(folder(which), "p_{}.npy".format(held))
        if used not in models:
            print("{}: training on {}".format(held, " ".join(used)), flush=True)
            models = {used: fit(used, which)}
            torch.save(models[used]["net"].state_dict(), os.path.join(folder(which), "net_{}.pt".format(held)))
        model = models[used]
        p = proba(held, model, which)
        np.save(path, p.astype(np.float32))
        truth = lab_robust.cached(held, which)["truth"]
        row = {"match": held, "source": sets.source(held), "seconds": model["seconds"], "train_loss": model["history"][-1],
               "trained_on": " ".join(used)}
        for variant in VARIANTS + ("robust",):
            q = lab_robust.held_out_proba(held, which, "blank") if variant == "robust" else held_out(held, which, variant)
            row.update({"{}_{}".format(variant, k): v for k, v in frame_check(q, truth).items()})
        rows.append(row)
        print("{}: predicted; log loss {:.4f} (robust {:.4f}, average {:.4f}); argmax accuracy {:.4f} (robust {:.4f}); "
              "trained {:.0f} s".format(held, row["temporal_log_loss"], row["robust_log_loss"], row["average_log_loss"],
                                        row["temporal_acc"], row["robust_acc"], model["seconds"]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(lab.RESULTS, "lab_{}_frames{}.csv".format(NAME, chain.suffix(which))),
                              index=False, float_format="%.5f")


# --------------------------------------------------------------------------------------------
# Decoding and judging
# --------------------------------------------------------------------------------------------

def letters_of(match, which, variant, setting, snap=True):
    decoded = L.decode_cost(match, -np.log(held_out(match, which, variant)), **setting)
    return np.asarray(L.finish(match, decoded, which) if snap else decoded).astype("<U1")


def grid(which):
    rows = []
    for match in lab.each(which):
        for variant in VARIANTS:
            for setting in GRID:
                s = lab.score(match, letters_of(match, which, variant, setting))
                rows.append({"match": match.id, "variant": variant, **setting, "spell_f1": s["spell_f1"],
                             "macro_f1": s["macro_f1"], "possessions": s["possessions"]})
        print("{}: grid scored".format(match.id), flush=True)
    return pd.DataFrame(rows)


def report_name(variant):
    return NAME if variant == "temporal" else "{}_{}".format(NAME, variant)


def judge_all(which):
    g = grid(which)
    for variant in VARIANTS:
        sub = g[g.variant == variant].drop(columns="variant").reset_index(drop=True)
        report = lab.judge(sub, COLUMNS, which, base=lab_weighted.robust_base(which))
        print("\n[{} against robust]\n{}".format(variant, lab.describe(report)))
        print("  picks: {}".format(pd.Series([str(f["picked"]) for f in report["folds"]]).value_counts().to_dict()))
        lab.save(report_name(variant), sub, report, which)


def chains(which):
    """v12's chain from robust (v12) and from each variant, every stage refitted on its setting for that match."""
    from bench import lab_mix
    from possession import v12
    lab_mix.cache_short_evidence()
    matches = list(lab.each(which))
    picks = lab_mix.fold_settings(which)
    mine = {}
    for variant in VARIANTS:
        with open(os.path.join(lab.RESULTS, "lab_{}{}.json".format(report_name(variant), chain.suffix(which)))) as handle:
            mine[variant] = {f["match"]: f["picked"] for f in json.load(handle)["folds"]}
    scores = {}
    for start in ("robust",) + VARIANTS:
        if start == "robust":
            letters = {m.id: lab_weighted.start_letters(m, "robust", True, picks, which) for m in matches}
        else:
            letters = {m.id: letters_of(m, which, start, mine[start][m.id]) for m in matches}
        for stage in v12.CHAIN[which]:
            letters = lab_mix.add_stage(stage, letters, matches, picks)
        scores[start] = lab_mix.scores_of(letters, matches)
        print("{}: {:.4f}".format(start, np.mean([v["spell_f1"] for v in scores[start].values()])), flush=True)
    out = {}
    for variant in VARIANTS:
        out[variant] = lab_mix.judge_scores(scores[variant], scores["robust"], which)
        print("\n{} chain against v12:\n{}".format(variant, lab.describe(out[variant])))
    table = pd.DataFrame({k: {m: v["spell_f1"] for m, v in s.items()} for k, s in scores.items()})
    print(table.round(4).to_string())
    with open(os.path.join(lab.RESULTS, "lab_{}_chains{}.json".format(NAME, chain.suffix(which))), "w") as handle:
        json.dump({"scores": table.round(5).to_dict(), "against_v12": out}, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--train", action="store_true", help="train one network per fold and save the held-out P")
    parser.add_argument("--chains", action="store_true", help="v12's chain from each variant, after the run")
    args = parser.parse_args()
    if args.train:
        return train_all(args.which)
    if args.chains:
        return chains(args.which)
    judge_all(args.which)


if __name__ == "__main__":
    main()
