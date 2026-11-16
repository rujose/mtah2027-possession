"""Night 2026-11-15, builder "netsc" (docs/lab/night/netsc.md): SkillCorner's even-indexed 10 matches in gru_b's training.

    python -m bench.lab_netsc check                         # the wrapper adds nothing when empty; SkillCorner's segments sane
    python -m bench.lab_netsc train --fold HELD [--seed S]  # one fold (~16 min, ~2 GB), then its held-out P into the mirror
    python -m bench.lab_netsc tables [--seed S]             # trial 1015's stage tables, recomputed in the mirror
    python -m bench.lab_netsc confirm [--seed S] [--variants normal p10]
    python -m bench.lab_netsc skillcorner [--seed S]        # SkillCorner's 20 with this recipe's metrica1 net
    python -m bench.lab_netsc judge [--seed S]              # the robustness bar; the round-8 B bar for information
    python -m bench.lab_netsc scframes [--seed S]           # the GRU's frame log loss on SkillCorner, odd and even apart
    python -m bench.lab_netsc fill --sub ens1               # netens seed 1's P cloned into a mirror of ours (information:
                                                            # SkillCorner's seed noise without SkillCorner in training)
    python -m bench.lab_netsc tables|skillcorner --sub ens1 --net PATH

gru_b's recipe (lab_netmirror.fit = lab_branch_b.fit to the bit: seed, 20 epochs, both orientations, each training
match as it is and under augment's training stress, the fold's DFL normaliser), with SkillCorner's 10 even-indexed
matches (lab_carried.sc_pool('train')) appended to every fold's training: lab_carried.sc_prepared's features as they
are (no stressed copy), each half in both orientations, SkillCorner's own labels (bench/skillcorner.py), the cut-away
frames (the loader's `no_data`) left out of the loss. They are read through the fold's DFL normaliser (they are not in
it). The folds are gru_b's (lab_branch_b.held_list: the 7 DFL matches and metrica1's, on all 7 DFL); metrica1's net
is also SkillCorner's deployment net, as in v15. The odd 10 are never read in training.

The new segments go in by wrapping lab_branch_b.stressed_segments for the length of the call (lab_netmirror.fit reads
it through the module), so the training loop is the network builder's verified one, untouched.
Files: data/external/lab_netsc/sS/ (nets/, lab_branch_b/, search_b/, lab_dead/), bench/results/lab_netsc/sS/.

SkillCorner open data (github.com/SkillCorner/opendata), MIT; data by SkillCorner. Nothing here reads our match.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from bench import lab, lab_branch_b, lab_carried, lab_gate, lab_netmirror, sets
from bench import skillcorner as sc

# Params
B = lab_branch_b
N = lab_netmirror
C = lab_carried
T = B.T
NAME = "netsc"
WHICH = "all"
TRIAL = 1015
IGNORED = 3                                  # the loss ignores codes > 2 (X); cut-away frames get it
# the robustness bar (docs/lab/night/ideas_outside.md, idea 4; fixed there before any score)
R_BENCH_LOSS, R_VARIANT_LOSS, R_SOURCE_LOSS = 0.002, 0.003, 0.005
R_SC_GAIN, R_SC_BETTER = 0.010, 7
# the round-8 B bar (docs/lab/night.md), for information
B_GAIN, B_BETTER, B_VARIANT_LOSS, B_SC_LOSS = 0.005, 7, 0.002, 0.005


def sub_of(seed):
    return "s{}".format(seed)


def root(seed):
    return N.root_of(NAME, seed if isinstance(seed, str) else sub_of(seed))


def results(seed):
    return N.results_of(NAME, seed if isinstance(seed, str) else sub_of(seed))


NETENS_SEED1 = os.path.join(lab.ROOT, "data", "external", "lab_netens", "seed1")


def fill(sub, src=NETENS_SEED1):
    """Another mirror's P (bench variants and training-stress copies) cloned into ours, read only there; tables are
    never copied (recomputed here)."""
    out = root(sub)
    for m in lab.matches(WHICH):
        for variant in N.BENCH_VARIANTS + ("train",):
            a, b = N.p_file(src, variant, m), N.p_file(out, variant, m)
            os.makedirs(os.path.dirname(b), exist_ok=True)
            N.clone(a, b)
    print("{}: P cloned from {}; {} missing".format(sub, src, len(N.missing_inputs(out))), flush=True)


def net_path(seed, held):
    return os.path.join(root(seed), "nets", "net_{}.pt".format(held))


def train_ids():
    return list(C.sc_pool("train"))


def test_ids():
    return list(C.sc_pool("test"))


def fold_matches(held):
    """The bench matches whose held-out P comes from this fold's net (metrica1's: Metrica's 3)."""
    if sets.source(held) in B.L.TRAIN_SOURCES:
        return [held]
    return [m for m in lab.matches(WHICH) if sets.source(m) not in B.L.TRAIN_SOURCES]


# --------------------------------------------------------------------------------------------
# SkillCorner's segments, and the training wrapper
# --------------------------------------------------------------------------------------------

def sc_labels(m):
    """SkillCorner's labels (0 H, 1 A, 2 D, 3 X) with the cut-away frames made X (out of the loss), and the halves."""
    data = sc.load(m)
    y = data["labels"].astype(np.int8).copy()
    y[data["no_data"]] = IGNORED
    return y, data["halves"], data


def sc_segments(m, norm):
    """lab_temporal.segments for a SkillCorner match: per orientation (as is, swapped) and half, (Z float16, y int8)."""
    if m in test_ids():
        raise ValueError("{} is one of SkillCorner's odd 10: never trained on".format(m))
    y, halves, _ = sc_labels(m)
    swapped = np.where(y == 0, 1, np.where(y == 1, 0, y)).astype(np.int8)
    match, X, Xs = C.sc_prepared(m)
    del match
    out = []
    for Xk, yk in ((X, y), (Xs, swapped)):
        for _, first, last in halves:
            out.append((T.standardised_half(Xk, first, last, norm, np.float16), yk[first:last + 1].copy()))
    del X, Xs
    return out


def fit(used, seed=0, sc_ids=None, log=print):
    """lab_netmirror.fit (seed, the fold's own normaliser, no head) with SkillCorner's segments appended after the last
    training match's stressed copy. sc_ids=() gives lab_netmirror.fit exactly."""
    sc_ids = train_ids() if sc_ids is None else list(sc_ids)
    original = B.stressed_segments
    last = used[-1]

    def with_sc(m, norm):
        out = original(m, norm)
        if m == last:
            for s in sc_ids:
                started = time.time()
                out += sc_segments(s, norm)
                log("  SkillCorner {} added ({:.0f} s)".format(s, time.time() - started))
        return out

    B.stressed_segments = with_sc
    try:
        model = N.fit(used, seed=seed, log=log)
    finally:
        B.stressed_segments = original
    model["skillcorner"] = sc_ids
    return model


def train(seed, held):
    path = net_path(seed, held)
    used = B.used_for(held)
    if not os.path.exists(path):
        print("netsc seed {} fold {}: training on {} (as they are and stressed) + SkillCorner's even 10 {}".format(
            seed, held, " ".join(used), " ".join(train_ids())), flush=True)
        model = fit(used, seed=seed, log=lambda s: print(s, flush=True))
        N.save_net(model, path, extra={"skillcorner": model["skillcorner"]})
        print("netsc seed {} fold {}: saved ({:.0f} s)".format(seed, held, model["seconds"]), flush=True)
    for m in fold_matches(held):
        N.predict_into(root(seed), m, path, used)


# --------------------------------------------------------------------------------------------
# Checks (before any training): no score involved
# --------------------------------------------------------------------------------------------

def check():
    out = {}
    # (a) the wrapper with no SkillCorner match is lab_netmirror.fit to the bit (1 epoch, 2 matches)
    epochs = T.EPOCHS
    T.EPOCHS = 1
    try:
        used = ("J03WN1", "J03WOH")
        a = N.fit(used, seed=0, log=print)
        b = fit(used, seed=0, sc_ids=(), log=print)
        sa, sb = a["net"].state_dict(), b["net"].state_dict()
        out["empty_wrapper_max_abs_diff"] = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
        out["empty_wrapper_history"] = [a["history"], b["history"]]
        print("(a) the wrapper with no SkillCorner match: max |diff| over weights {}".format(
            out["empty_wrapper_max_abs_diff"]), flush=True)
    finally:
        T.EPOCHS = epochs
    # (b) SkillCorner's segments on metrica1's normaliser (the 7 DFL matches)
    norm = T.normaliser(B.used_for(B.held_list()[-1]), WHICH)
    rows = []
    for m in train_ids():
        y, halves, data = sc_labels(m)
        inside = np.zeros(len(y), bool)
        for _, first, last in halves:
            inside[first:last + 1] = True
        graded = inside & np.isin(y, (0, 1, 2))
        row = {"match": m, "frames_in_halves": int(inside.sum()),
               "left_out_cutaway": float((data["no_data"] & inside).sum() / inside.sum()),
               "graded": int(graded.sum())}
        for k, name in enumerate("HAD"):
            row["share_" + name] = float((y[graded] == k).mean())
        row["ball_missing_on_graded"] = float(np.isnan(data["ball"][graded, 0]).mean())
        rows.append(row)
        del data
    segs = sc_segments(train_ids()[0], norm)
    Z = np.concatenate([z for z, _ in segs]).astype(np.float32)
    ys = np.concatenate([t for _, t in segs])
    out["segments_first_match"] = {"segments": len(segs), "frames": int(len(Z)), "nan_share": float(np.isnan(Z).mean()),
                                   "abs_mean_of_feature_means": float(np.nanmean(np.abs(np.nanmean(Z, axis=0)))),
                                   "median_feature_sd": float(np.nanmedian(np.nanstd(Z, axis=0))),
                                   "clipped_share": float((np.abs(Z) >= T.Z_CLIP).mean()),
                                   "labels": {int(k): int((ys == k).sum()) for k in range(4)},
                                   "swapped_H_equals_A": bool((segs[0][1] == 0).sum() == (segs[len(segs) // 2][1] == 1).sum())}
    del Z, segs
    df = pd.DataFrame(rows)
    out["skillcorner_even10"] = df.round(4).to_dict(orient="records")
    print(df.round(3).to_string(), flush=True)
    print(json.dumps(out["segments_first_match"], indent=1), flush=True)
    os.makedirs(results(0), exist_ok=True)
    with open(os.path.join(results(0), "check.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    return out


# --------------------------------------------------------------------------------------------
# B rebuilt in the mirror
# --------------------------------------------------------------------------------------------

def tables(seed):
    """Trial 1015's stage tables in the mirror (computed from this recipe's P on the training-stress copies)."""
    N.use_named(NAME, seed if isinstance(seed, str) else sub_of(seed))
    missing = N.missing_inputs(root(seed))
    if missing:
        raise RuntimeError("mirror incomplete: {} files missing, e.g. {}".format(len(missing), missing[:3]))
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

def sc_split(path, trial=TRIAL):
    df = pd.read_csv(path, dtype={"match": str})
    return df[df.trial == trial].set_index("match").spell_f1


def judge(seed, against=None):
    """The robustness bar against v15's B; the B bar for information. `against`: another mirror's results folder
    (e.g. netens seed 1) for a paired reading, information only."""
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
           "per_variant": per_variant.round(5).to_dict(),
           "per_variant_values": c_v.round(5).to_dict(), "per_source": per_source,
           "trial0_check": float((N.per_match(cand, 0)[0] - N.per_match(ref, 0)[0]).abs().max())}
    sc_path = os.path.join(res, "skillcorner_{}.csv".format(TRIAL))
    have_sc = os.path.exists(sc_path)
    if have_sc:
        a = sc_split(sc_path)
        b = sc_split(N.REFERENCE_SC)
        a0 = sc_split(sc_path, 0)
        b0 = sc_split(N.REFERENCE_SC, 0)
        out["trial0_sc_check"] = float((a0 - b0).abs().max())
        for name, ids in (("odd", test_ids()), ("even", train_ids()), ("all", test_ids() + train_ids())):
            d = (a[ids] - b[ids])
            out["sc_" + name] = {"netsc": float(a[ids].mean()), "v15": float(b[ids].mean()), "gain": float(d.mean()),
                                 "better": int((d > 0).sum()), "n": len(ids), "per_match": d.round(5).to_dict(),
                                 "vs_trial0": float((a[ids] - b0[ids]).mean())}
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
    if against:
        p = os.path.join(against, "confirm_{}.csv".format(TRIAL))
        if os.path.exists(p):
            o_m, o_v = N.per_match(pd.read_csv(p))
            d = (c_m - o_m).reindex(lab.matches(WHICH))
            out["paired"] = {"against": against, "bench": float(o_m.mean()), "gain": float(d.mean()),
                             "better": int((d > 0).sum()), "per_variant": (c_v - o_v).round(5).to_dict(),
                             "per_source": {s: float(d[src == s].mean()) for s in ("idsse", "metrica")}}
            q = os.path.join(against, "skillcorner_{}.csv".format(TRIAL))
            if have_sc and os.path.exists(q):
                o = sc_split(q)
                for name, ids in (("odd", test_ids()), ("even", train_ids())):
                    dd = a[ids] - o[ids]
                    out["paired"]["sc_" + name] = {"other": float(o[ids].mean()), "gain": float(dd.mean()),
                                                   "better": int((dd > 0).sum())}
    switch = os.path.join(res, "confirm_{}_normal_p10.csv".format(TRIAL))
    if os.path.exists(switch):
        s = pd.read_csv(switch)
        rr = pd.read_csv(N.REFERENCE_SWITCH)
        out["switch_table"] = {v: {"netsc": float(s[(s.trial == TRIAL) & (s.variant == v)].spell_f1.mean()),
                                   "v15": float(rr[(rr.trial == TRIAL) & (rr.variant == v)].spell_f1.mean())}
                               for v in ("normal", "p10")}
    with open(os.path.join(res, "judge.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1))
    return out


# --------------------------------------------------------------------------------------------
# Information: the GRU's own frames on SkillCorner (deployment nets), odd and even apart
# --------------------------------------------------------------------------------------------

def frame_stats(p, y, speed=None):
    """The GRU alone, argmax, on graded frames (truth only to report): log loss, accuracy; on live frames the team's
    accuracy and the share called dead; on dead frames the share called dead (and, given the ball's speed, on dead
    frames whose ball is tracked and moving); around each direct change of team (H to A or A to B, live on both
    sides) the team's accuracy within 1 s, and the signed offset of the GRU's own switch to the new team (the
    nearest within 2 s; negative: the GRU switches before the truth)."""
    graded = np.isin(y, (0, 1, 2))
    f = T.frame_check(p, np.where(graded, y, 3))
    live = graded & (y < 2)
    dead = graded & (y == 2)
    team = p[:, :2].argmax(axis=1)
    out = {**f, "team_acc_live": float((team[live] == y[live]).mean()),
           "live_called_dead": float((p[live].argmax(axis=1) == 2).mean()),
           "dead_called_dead": float((p[dead].argmax(axis=1) == 2).mean())}
    if speed is not None:
        moving = dead & (speed > C.MOVING)
        out["moving_dead_called_dead"] = float((p[moving].argmax(axis=1) == 2).mean())
        out["moving_dead_share"] = float(moving.sum() / graded.sum())
    t = np.flatnonzero((y[1:] < 2) & (y[:-1] < 2) & (y[1:] != y[:-1]) & graded[1:] & graded[:-1]) + 1
    near = np.zeros(len(y), bool)
    offsets = []
    for c in t:
        near[max(c - 25, 0):c + 25] = True
        new, old = y[c], y[c - 1]
        a, b = max(c - 50, 1), min(c + 50, len(y) - 1)
        s = np.flatnonzero((team[a:b] == new) & (team[a - 1:b - 1] == old)) + a
        if len(s):
            offsets.append(int(s[np.argmin(np.abs(s - c))] - c))
    near &= live
    out.update({"changes": int(len(t)), "team_acc_near_change": float((team[near] == y[near]).mean()),
                "team_acc_away": float((team[live & ~near] == y[live & ~near]).mean()),
                "switch_found": float(len(offsets) / max(len(t), 1)),
                "switch_offset_median": float(np.median(offsets)) if offsets else float("nan"),
                "switch_offset_mean": float(np.mean(offsets)) if offsets else float("nan"),
                "switch_offset_abs": float(np.mean(np.abs(offsets))) if offsets else float("nan")})
    return out


COLUMNS = ["log_loss", "acc", "team_acc_live", "live_called_dead", "dead_called_dead", "team_acc_near_change",
           "team_acc_away", "switch_offset_median", "switch_offset_abs"]


def scframes(seed):
    """The GRU alone on SkillCorner (deployment nets: gru_b's and this recipe's metrica1 net), the cut-aways out of
    the grading, odd and even apart (frame_stats)."""
    used = B.used_for(B.held_list()[-1])
    nets = {"gru_b": os.path.join(N.SHARED_B, "net_{}.pt".format(B.held_list()[-1])),
            "netsc": net_path(seed, B.held_list()[-1])}
    models = {k: lab_gate.gru_model(used, WHICH, p) for k, p in nets.items()}
    rows = []
    for m in sc.match_ids():
        y, halves, data = sc_labels(m)
        speed = C.ball_speed(data["ball"])
        match, X, Xs = C.sc_prepared(m)
        for k, model in models.items():
            p = lab_gate.gru_proba(model, {"X": X, "X_swapped": Xs}, match.halves)
            rows.append({"match": m, "pool": "odd" if m in test_ids() else "even", "net": k,
                         **frame_stats(p, y, speed)})
        print("{}: {}".format(m, " ".join("{} {:.4f}".format(r["net"], r["log_loss"]) for r in rows[-2:])), flush=True)
        del match, X, Xs, data
    df = pd.DataFrame(rows)
    os.makedirs(results(seed), exist_ok=True)
    df.to_csv(os.path.join(results(seed), "scframes.csv"), index=False, float_format="%.5f")
    print(df.groupby(["pool", "net"])[COLUMNS + ["moving_dead_called_dead"]].mean().round(4).T.to_string())
    return df


def frames(seed):
    """The GRU alone on B's bench (held-out P: gru_b's against this recipe's), B's 3 variants (frame_stats)."""
    rows = []
    for m in lab.matches(WHICH):
        for variant in N.SB.VARIANTS:
            y = np.asarray(lab_gate.variant_features(m, WHICH, variant)["truth"]).astype(np.int64)
            for name, path in (("gru_b", N.shared_p_file(variant, m)), ("netsc", N.p_file(root(seed), variant, m))):
                p = np.load(path).astype(np.float64)
                rows.append({"match": m, "source": sets.source(m), "variant": variant, "net": name,
                             **frame_stats(p, y)})
        print("{}: done".format(m), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results(seed), "frames.csv"), index=False, float_format="%.5f")
    print(df.groupby(["source", "net"])[COLUMNS].mean().round(4).T.to_string())
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("check", "train", "tables", "confirm", "skillcorner", "judge", "scframes",
                                         "frames", "fill"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sub", default=None, help="a mirror other than sS (fill, tables, skillcorner)")
    parser.add_argument("--net", default=None, help="skillcorner: the deployment net (default: this recipe's metrica1)")
    parser.add_argument("--fold")
    parser.add_argument("--variants", nargs="+", default=list(N.SB.VARIANTS), choices=N.BENCH_VARIANTS)
    parser.add_argument("--against", default=None)
    args = parser.parse_args()
    if args.step == "check":
        return check()
    if args.step == "train":
        return train(args.seed, args.fold)
    if args.step == "fill":
        return fill(args.sub)
    if args.step == "tables":
        return tables(args.sub or args.seed)
    if args.step == "confirm":
        return confirm(args.seed, args.variants)
    if args.step == "skillcorner":
        return skillcorner_run(args.sub or args.seed, args.net)
    if args.step == "judge":
        return judge(args.seed, args.against)
    if args.step == "scframes":
        return scframes(args.seed)
    return frames(args.seed)


if __name__ == "__main__":
    main()
