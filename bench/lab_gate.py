"""Lab candidate `gate` (round 7, a switch): per match, branch A (v13: the temporal `average` start, then v12's post
stages) or branch B (`augment`'s chain), picked from a number the program computes on the tracking alone
(docs/lab/gate.md).

    python -m bench.lab_gate --prepare [--set all|idsse]       # partly stressed balls, robust's features on them (cached)
    OMP_NUM_THREADS=4 python -m bench.lab_gate --fit [--set ...] # robust's and augment's fold models; P on every variant
    python -m bench.lab_gate --gru [--set ...]                  # the lab's fold GRUs on every variant (normal checked)
    python -m bench.lab_gate --signals                          # the signal on every bench variant, both sets (no scores)
    python -m bench.lab_gate --chains VARIANT [--set ...]       # branch A and B, held out, on one bench variant
    python -m bench.lab_gate --sc-signals                       # the signal on SkillCorner's 20 (after the rule was fixed)
    python -m bench.lab_gate --skillcorner                      # branch A on SkillCorner (B: lab_carried's rows)
    python -m bench.lab_gate --report                           # the switch bar, every part: bench/results/lab_gate.json

Bench variants: `normal` (the lab's bench), `p10`, `p25`, `p50` (that share of each match's true stoppages given
lab_carried's test stress, a fixed nested draw), `stressed` (lab_carried's: every true stoppage).

For possession/v13.py: `signal(track, pred, halves, pitch) -> float` and `THRESHOLD` (pick B when signal > THRESHOLD).
Nothing here reads our match.

Results (2026-11-09, docs/lab/gate.md): the switch clears all four parts of round 7's switch bar on both sets. Normal
bench: A on 17/17 (±0). Stressed: B on 17/17 (+0.1075 `all`, +0.0573 `idsse`). 10 / 25 / 50 %: within 0.005 of the
better branch at every level, narrowly (worst -0.0046). SkillCorner: B on 20/20, 0.412 against A's 0.319 (+0.093,
20/20). The signal reads 0.02-0.10 on the normal bench, 0.36-0.63 stressed, 0.25-0.49 on SkillCorner.
"""

import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd

from bench import chain, lab, lab_carried, lab_mix, lab_robust, lab_temporal, sets
from bench.eval_pathcrf import UNSEEN
from possession import lab_learned_features as llf
from possession import v9, v11

# Params (fixed before anything was scored; docs/lab/gate.md gives the why of each)
NAME = "gate"
L = lab_robust.L
C = lab_carried
RESULTS = lab.RESULTS
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_gate")      # gitignored (data/*)
H, A, D, X = 0, 1, 2, 3
FPS = 25
# The partly stressed benches: that share of each match's stressed stoppages (lab_carried's test draw) kept stressed,
# the others given back their normal ball. One permutation per match, so 10 % sits inside 25 % inside 50 %.
LEVELS = {"p10": 0.10, "p25": 0.25, "p50": 0.50}
PARTIAL_SEED = 2027
VARIANTS = ("normal", "p10", "p25", "p50", "stressed")
STRESS = {"normal": 0.0, "p10": 0.10, "p25": 0.25, "p50": 0.50, "stressed": 1.0}
BRANCHES = ("A", "B")
# The signal (fixed 2026-11-09 ~19:25, from its distributions on the bench variants only: before any branch was scored
# and before it was computed on SkillCorner). "Where the players say the ball is dead, how often is the ball seen and
# moving, against how often it is where they say it's live":
DEAD_AT = 0.5            # v4's players-only P(dead) at or above this: the players say dead (more likely than not)
TRIM_END = 2 * FPS       # frames: the last 2 s of each such run left out (the restart kick, while v4 still reads a set piece)
LIVE_BELOW = 0.2         # v4's P(dead) under this: the players say live (the match's own baseline for a moving ball)
# A frame's ball counts as "seen and moving" if it is tracked, not blanked by robust's still rule (lab_robust.still_frames:
# still >= 1 s), and moves at >= lab_carried.MOVING (0.3 m/s) over the next LOOK frames (0.2 s), as lab_carried measured
# SkillCorner's dead-time ball. Inside the halves only, and only frames with at least one player tracked (amended
# ~19:30, still before any branch score and before SkillCorner's signal: a frame with no player gives v4 nothing to
# read, as in a broadcast's cut-aways, skillcorner.md; every in-half bench frame has players, so the bench is unchanged).
# The threshold rule (fixed with the signal, before any branch was scored): per set, leave one match out, the others'
# mean signal at the stress level where their mean B - A crosses 0 (linear between the two levels around the first
# crossing); B where the match's signal is above it. THRESHOLD is the rule on all 10 matches of `all` (v13's set).
THRESHOLD = 0.1445       # the rule on all 10 of `all` (0.14448, rounded; set 2026-11-09 ~19:53, after the `all` set's
                         # chains and before SkillCorner's signal was computed): B above it


def folder(which):
    return CACHE + chain.suffix(which)


# --------------------------------------------------------------------------------------------
# The signal
# --------------------------------------------------------------------------------------------

def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return zip(edges[::2], edges[1::2])


def signal_parts(track, pred, halves):
    """(the frames the players call dead, trimmed; those they call live; the ball seen and moving) as bool [frames], all
    inside the halves on frames with at least one player tracked."""
    ball = np.asarray(track["ball"])
    n = len(ball)
    p = np.nan_to_num(np.asarray(pred["p_dead"], dtype=np.float64)[:n])
    inside = np.zeros(n, bool)
    dead = np.zeros(n, bool)
    for _, first, last in halves:
        inside[first:last + 1] = True
    players = np.asarray(track["players"])[:n]
    inside &= ~np.all(np.isnan(players[..., 0]), axis=1)          # v4 has someone to read
    for _, first, last in halves:
        for a, b in _runs(p[first:last + 1] >= DEAD_AT):
            if b - TRIM_END > a:
                dead[first + a:first + b - TRIM_END] = True
    live = inside & (p < LIVE_BELOW)
    seen = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        moving = seen & ~lab_robust.still_frames(ball, halves) & (C.ball_speed(ball.astype(np.float64)) >= C.MOVING)
    return dead & inside, live, moving


def signal(track, pred, halves, pitch=None):
    """One number per match from the tracking and v12's inputs alone (no truth, no labels, no cache): the share of the
    frames v4's players-only model calls dead (P >= 0.5, the last 2 s of each run left out) where the ball is seen and
    moving, divided by the same share where it calls them live (P < 0.2); frames inside the halves with a player
    tracked. Near 0 when the dead ball is missing or frozen (the bench); near 0.5 when it is tracked and moving through
    stoppages (lab_carried's stressed bench). 0.0 when either part is empty. `pitch` is unused (kept for v13's call)."""
    dead, live, moving = signal_parts(track, pred, halves)
    if not dead.any() or not live.any() or not moving[live].any():
        return 0.0
    return float(moving[dead].mean() / moving[live].mean())


def picks_b(value, threshold=None):
    """The switch: branch B (augment's chain) when the signal is above the threshold, else branch A (v13)."""
    t = THRESHOLD if threshold is None else threshold
    return bool(t is not None and value > t)


# --------------------------------------------------------------------------------------------
# The partly stressed benches
# --------------------------------------------------------------------------------------------

def stoppage_order(m):
    """lab_carried's stressed stretches of the match (its test draw's table), in this page's fixed random order."""
    table = pd.read_csv(C.stress_path(m, C.TEST_SEED).replace(".npz", ".csv"))
    table = table[table.stressed.astype(bool)].reset_index(drop=True)
    rng = np.random.default_rng([PARTIAL_SEED, zlib.crc32(m.encode())])
    return table.iloc[rng.permutation(len(table))].reset_index(drop=True)


def chosen(m, share):
    """[(a, b)]: the stretches stressed at this share (the first round(share x n) of the fixed order)."""
    order = stoppage_order(m)
    k = int(round(share * len(order)))
    return [(int(r.a), int(r.a + r.length)) for r in order.iloc[:k].itertuples()]


def partial_ball(m, share, match=None):
    """The damaged ball with only the chosen stretches' frames taken from lab_carried's stressed ball."""
    normal = (match or lab.load(m, "all")).track["ball"]
    full = C.stressed_ball(m, C.TEST_SEED)
    ball = normal.copy()
    for a, b in chosen(m, share):
        ball[a:b] = full[a:b]
    return ball


def variant_match(m, which, variant):
    """The lab match on one bench variant: the ball replaced and v6's / v7's costs recomputed on it (as lab_carried)."""
    match = lab.load(m, which)
    if variant == "normal":
        match.variant = variant
        return match
    if variant == "stressed":
        match = C.stressed(m, which, match=match)
    else:
        ball = partial_ball(m, LEVELS[variant], match)
        v6, v7 = C.ball_costs(match, ball, which)
        match.track = {**match.track, "ball": ball}
        match.pred = {**match.pred, "v6_cost": v6, "v7_cost": v7}
        match.stressed = variant
    match.variant = variant
    return match


def feature_path(m, which, variant):
    return os.path.join(folder(which), "features_{}_{}.npz".format(variant, m))


def variant_features(m, which, variant):
    """robust's 97 features on a bench variant (normal: lab_robust's cache; stressed: lab_carried's; partly stressed:
    computed here as lab_carried.stressed_features does, cached compressed)."""
    if variant == "normal":
        return lab_robust.cached(m, which)
    if variant == "stressed":
        return C.stressed_features(m, which, C.TEST_SEED)
    path = feature_path(m, which, variant)
    if not os.path.exists(path):
        b = lab_robust.blanked(variant_match(m, which, variant))
        touches = v11.seen_touches(b.track, v11.SETTINGS[which]["dv"])
        passes = v9.seen_passes(b.track)
        names, X_ = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes)
        _, Xs = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes, swap=True)
        os.makedirs(folder(which), exist_ok=True)
        np.savez_compressed(path + ".part.npz", names=np.array(names), X=X_, X_swapped=Xs, truth=b.truth.astype(np.int8),
                            blanked=b.blanked)
        os.replace(path + ".part.npz", path)
    with np.load(path) as z:
        return {"names": [str(k) for k in z["names"]], "X": z["X"], "X_swapped": z["X_swapped"], "truth": z["truth"],
                "blanked": z["blanked"]}


def prepare(which):
    for m in lab.matches(which):
        for variant in LEVELS:
            started = time.time()
            variant_features(m, which, variant)
            print("{} {}: features ({:.0f} s)".format(m, variant, time.time() - started), flush=True)


# --------------------------------------------------------------------------------------------
# robust's and augment's fold models (lab_carried.fit_robust, refitted: lab_carried saved only their P)
# --------------------------------------------------------------------------------------------

def p_path(kind, variant, m, which):
    return os.path.join(folder(which), "p_{}_{}_{}.npy".format(kind, variant, m))


def held_out_p(kind, variant, m, which):
    """Held-out P of robust, augment or the GRU ("gru") on a bench variant. Normal and stressed robust / augment are
    lab_robust's and lab_carried's saved P; the GRU's normal P is lab_temporal's; the rest is computed here."""
    if kind in ("robust", "augment") and variant in ("normal", "stressed"):
        return C.held_out_p(kind, variant, m, which)
    if kind == "gru" and variant == "normal":
        return lab_temporal.held_out(m, which, "temporal")
    return np.load(p_path(kind, variant, m, which)).astype(np.float64)


def average_p(variant, m, which):
    """`average`: the mean of the GRU's P and robust's, clipped and rounded (lab_temporal.held_out)."""
    if variant == "normal":
        return lab_temporal.held_out(m, which, "average")
    p = held_out_p("gru", variant, m, which) + held_out_p("robust", variant, m, which)
    return np.round(np.clip(p / 2, *L.CLIP), L.DECIMALS)


def fit_folds(which):
    """Each fold's robust and augment (lab_carried.fit_robust, the same rows, trees and seed), then their P on the held
    match's partly stressed variants; the refit checked against the saved normal and stressed P."""
    import joblib
    from threadpoolctl import threadpool_limits
    ms = lab.matches(which)
    models, checks = {}, []
    for held in ms:
        used = tuple(m for m in ms if m != held and sets.source(m) in L.TRAIN_SOURCES)
        if used not in models:
            started = time.time()
            path = os.path.join(folder(which), "folds_{}.joblib".format(held))
            if os.path.exists(path):
                models = {used: joblib.load(path)}
            else:
                with threadpool_limits(4):
                    models = {used: {"robust": C.fit_robust(used, which), "augment": C.fit_robust(used, which, augment=True)}}
                os.makedirs(folder(which), exist_ok=True)
                joblib.dump(models[used], path, compress=3)
            print("{}: fold models on {} ({:.0f} s)".format(held, " ".join(used), time.time() - started), flush=True)
        f = models[used]
        with threadpool_limits(4):
            for variant in ("normal", "stressed"):
                c = variant_features(held, which, variant)
                for kind in ("robust", "augment"):
                    diff = float(np.abs(C.proba(f[kind], c) - held_out_p(kind, variant, held, which)).max())
                    checks.append({"match": held, "variant": variant, "kind": kind, "max_abs_diff": diff})
                    print("  {} {} {}: refit against saved P, max |diff| {:.4f}".format(held, variant, kind, diff), flush=True)
                del c
            for variant in LEVELS:
                c = variant_features(held, which, variant)
                for kind in ("robust", "augment"):
                    np.save(p_path(kind, variant, held, which), C.proba(f[kind], c).astype(np.float32))
                del c
    pd.DataFrame(checks).to_csv(os.path.join(folder(which), "refit_check.csv"), index=False)


# --------------------------------------------------------------------------------------------
# The lab's fold GRUs (lab_temporal) on any variant's features
# --------------------------------------------------------------------------------------------

def gru_file(held, which):
    """The saved network that never saw `held`: its own fold's, or (Metrica's 3 on `all`) the one trained on all 7."""
    if sets.source(held) in L.TRAIN_SOURCES:
        name = held
    else:
        name = [m for m in lab.matches(which) if sets.source(m) not in L.TRAIN_SOURCES][0]     # metrica1: the first saved
    return os.path.join(lab_temporal.folder(which), "net_{}.pt".format(name))


def gru_model(used, which, path):
    """{net, norm}: lab_temporal's network with its fold's weights and normaliser."""
    torch = lab_temporal.torch_ready()
    norm = lab_temporal.normaliser(used, which)
    net = lab_temporal.network(97 + len(norm["flagged"]))
    net.load_state_dict(torch.load(path))
    net.eval()
    return {"net": net, "norm": norm}


def gru_proba(model, c, halves):
    """lab_temporal.proba on given features (the match as is and with the teams exchanged, clipped and rounded)."""
    norm = model["norm"]
    p = np.full((len(c["X"]), 3), 1 / 3)
    for _, first, last in halves:
        p[first:last + 1] = 0.0
    for key, cols in (("X", np.arange(3)), ("X_swapped", L.SWAP)):
        for _, first, last in halves:
            Z = lab_temporal.standardised_half(c[key], first, last, norm)
            p[first:last + 1] += lab_temporal.run_half(model["net"], Z, norm)[:, cols] / 2
    return np.round(np.clip(p, *L.CLIP), L.DECIMALS)


def gru_all(which):
    """The GRU's held-out P on every variant; on the normal bench it must give lab_temporal's saved P."""
    ms = lab.matches(which)
    checks = []
    for held in ms:
        started = time.time()
        used = tuple(m for m in ms if m != held and sets.source(m) in L.TRAIN_SOURCES)
        model = gru_model(used, which, gru_file(held, which))
        halves = lab_temporal.halves_of(held, which)
        for variant in VARIANTS:
            if variant != "normal" and os.path.exists(p_path("gru", variant, held, which)):
                continue
            p = gru_proba(model, variant_features(held, which, variant), halves)
            if variant == "normal":
                diff = float(np.abs(p - held_out_p("gru", "normal", held, which)).max())
                checks.append({"match": held, "net": os.path.basename(gru_file(held, which)), "max_abs_diff": diff})
                print("  {}: GRU recomputed on the normal bench against lab_temporal's P, max |diff| {:.4f}".format(held, diff),
                      flush=True)
            else:
                np.save(p_path("gru", variant, held, which), p.astype(np.float32))
        print("{}: GRU on every variant ({:.0f} s)".format(held, time.time() - started), flush=True)
    pd.DataFrame(checks).to_csv(os.path.join(folder(which), "gru_check.csv"), index=False)



# --------------------------------------------------------------------------------------------
# The branches, held out, on one bench variant
# --------------------------------------------------------------------------------------------

def own_dead_cache():
    """lab_dead's feature cache (the IDSSE set's `dead` stage): read lab_dead's and lab_carried's, write only here."""
    import hashlib
    from bench import lab_dead

    def frame_features(match):
        key = hashlib.sha1(match.pred["p_dead"].tobytes() + match.track["ball"].tobytes()).hexdigest()[:12]
        name = "{}_{}.npy".format(match.id, key)
        for base in (lab_dead.CACHE, os.path.join(C.CACHE, "lab_dead"), os.path.join(CACHE, "lab_dead")):
            if os.path.exists(os.path.join(base, name)):
                return lab_dead.FEATURES["ball"], np.load(os.path.join(base, name))
        names, X_ = lab_dead.compute(match)
        os.makedirs(os.path.join(CACHE, "lab_dead"), exist_ok=True)
        np.save(os.path.join(CACHE, "lab_dead", name), X_)
        return names, X_
    lab_dead.frame_features = frame_features


def start_letters(match, branch, which, picks):
    """A: `average`'s P decoded with its own fold pick; B: augment's P decoded with robust's fold pick (lab_carried)."""
    if branch == "A":
        return C.decoded(match, average_p(match.variant, match.id, which), picks["average"][match.id], which)
    return C.decoded(match, held_out_p("augment", match.variant, match.id, which), picks["robust"][match.id], which)


def chains_path(variant, which):
    return os.path.join(RESULTS, "lab_gate_chains_{}{}.json".format(variant, chain.suffix(which)))


def chains(variant, which, branches=BRANCHES):
    """Each branch's chain on one bench variant: start, then v12's post stages, each refitted out of fold on the
    branch's own letters with v12's fold setting for that match (lab_mix.add_stage, as lab_temporal and lab_carried)."""
    from possession import v12
    lab_mix.cache_short_evidence()          # keyed by match id: one variant per process
    own_dead_cache()
    matches = [variant_match(m, which, variant) for m in lab.matches(which)]
    picks = lab_mix.fold_settings(which)
    picks["average"] = C.average_picks(which)
    path = chains_path(variant, which)
    done = {}
    if os.path.exists(path):
        with open(path) as handle:
            done = json.load(handle)
    for branch in branches:
        started = time.time()
        letters = {m.id: start_letters(m, branch, which, picks) for m in matches}
        steps = {"start": lab_mix.scores_of(letters, matches)}
        for stage in v12.CHAIN[which]:
            letters = lab_mix.add_stage(stage, letters, matches, picks)
            steps[stage] = lab_mix.scores_of(letters, matches)
        done[branch] = {k: {m: v["spell_f1"] for m, v in s.items()} for k, s in steps.items()}
        done[branch + " macro"] = {m: v["macro_f1"] for m, v in steps[v12.CHAIN[which][-1]].items()}
        done[branch + " possessions"] = {m: v["possessions"] for m, v in steps[v12.CHAIN[which][-1]].items()}
        os.makedirs(os.path.join(folder(which), "letters"), exist_ok=True)
        for m in matches:
            np.save(os.path.join(folder(which), "letters", "{}_{}_{}.npy".format(branch, variant, m.id)), letters[m.id])
        print("{} {} {}: start {:.4f}, chain {:.4f} ({:.0f} s)".format(
            which, variant, branch, np.mean(list(done[branch]["start"].values())),
            np.mean(list(done[branch][v12.CHAIN[which][-1]].values())), time.time() - started), flush=True)
        with open(path, "w") as handle:
            json.dump(done, handle, indent=1)
    return done


def final_scores(variant, which):
    """{branch: {match: the chain's spellF1}} on one variant."""
    from possession import v12
    with open(chains_path(variant, which)) as handle:
        done = json.load(handle)
    return {b: done[b][v12.CHAIN[which][-1]] for b in BRANCHES if b in done}


def reproduction(which):
    """A on the normal bench against lab_temporal_chains' `average`; B on normal and stressed against lab_carried's."""
    from possession import v12
    last = v12.CHAIN[which][-1]
    out = {}
    with open(os.path.join(RESULTS, "lab_temporal_chains{}.json".format(chain.suffix(which)))) as handle:
        temporal = json.load(handle)["scores"]["average"]
    mine = final_scores("normal", which)
    out["A normal vs lab_temporal_chains average"] = max(abs(mine["A"][m] - temporal[m]) for m in temporal)
    for variant in ("normal", "stressed"):
        with open(os.path.join(RESULTS, "lab_carried_chains_{}{}.json".format(variant, chain.suffix(which)))) as handle:
            carried = json.load(handle)["augment"][last]
        if os.path.exists(chains_path(variant, which)):
            mine = final_scores(variant, which)
            out["B {} vs lab_carried_chains augment".format(variant)] = max(abs(mine["B"][m] - carried[m]) for m in carried)
    return out


# --------------------------------------------------------------------------------------------
# Signals on the bench variants (no scores), the threshold rule, the switch's scores
# --------------------------------------------------------------------------------------------

def signals_path():
    return os.path.join(RESULTS, "lab_gate_signals.csv")


def bench_signals():
    """The signal on every bench match, variant and set (no truth: the variants' balls are the only thing that differ)."""
    rows = []
    for which in ("all", "idsse"):
        for m in lab.matches(which):
            for variant in VARIANTS:
                match = variant_match(m, which, variant)
                rows.append({"set": which, "match": m, "source": match.source, "variant": variant, "stress": STRESS[variant],
                             "signal": signal(match.track, match.pred, match.halves, match.pitch)})
                del match
            print("{} {}: {}".format(which, m, " ".join("{:.3f}".format(r["signal"]) for r in rows[-len(VARIANTS):])),
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(signals_path(), index=False, float_format="%.5f")
    return df


def signal_table(which):
    """{variant: {match: signal}}."""
    df = pd.read_csv(signals_path())
    df = df[df.set == which]
    return {v: dict(zip(df[df.variant == v].match, df[df.variant == v].signal)) for v in VARIANTS}


def threshold_rule(others, sig, scores):
    """The others' mean signal at the stress level where their mean B - A first turns positive (linear between the two
    levels around it). +inf if B never wins on them, -inf if it already wins on the normal bench."""
    g = [np.mean([scores[v]["B"][m] - scores[v]["A"][m] for m in others]) for v in VARIANTS]
    s = [np.mean([sig[v][m] for m in others]) for v in VARIANTS]
    for k, gain in enumerate(g):
        if gain > 0:
            if k == 0:
                return float("-inf"), g, s
            frac = -g[k - 1] / (gain - g[k - 1])
            return float(s[k - 1] + frac * (s[k] - s[k - 1])), g, s
    return float("inf"), g, s


def switch(which):
    """Per variant, each match's threshold (left out), pick and score; the means against both branches."""
    sig = signal_table(which)
    scores = {v: final_scores(v, which) for v in VARIANTS}
    ms = lab.matches(which)
    thresholds = {m: threshold_rule([o for o in ms if o != m], sig, scores)[0] for m in ms}
    on_all, g, s = threshold_rule(ms, sig, scores)
    rows = []
    for v in VARIANTS:
        for m in ms:
            b = picks_b(sig[v][m], thresholds[m])
            rows.append({"set": which, "variant": v, "match": m, "source": sets.source(m), "unseen": m in UNSEEN,
                         "signal": sig[v][m], "threshold": thresholds[m], "pick": "B" if b else "A",
                         "A": scores[v]["A"][m], "B": scores[v]["B"][m], "gate": scores[v]["B" if b else "A"][m]})
    return pd.DataFrame(rows), {"threshold on all matches": on_all, "mean B - A by variant": dict(zip(VARIANTS, g)),
                                "mean signal by variant": dict(zip(VARIANTS, s))}



# --------------------------------------------------------------------------------------------
# SkillCorner: branch A as v13 would run (deployment GRU and robust), B from lab_carried's run
# --------------------------------------------------------------------------------------------

def average_setting(which="all"):
    """`average`'s decoder setting chosen on all the set's matches (lab_temporal_average.json)."""
    with open(os.path.join(RESULTS, "lab_{}{}.json".format(lab_temporal.report_name("average"), chain.suffix(which)))) as h:
        return {k: lab_mix.literal(v) for k, v in json.load(h)["chosen"].items()}


def deployment_gru(which="all"):
    """The GRU trained on all 7 IDSSE matches (lab_temporal's net_metrica1.pt) and its normaliser."""
    used = tuple(m for m in lab.matches(which) if sets.source(m) in L.TRAIN_SOURCES)
    return gru_model(used, which, gru_file("metrica1", which))


def deploy_post_a(which="all"):
    """Post-stage deployment models for branch A, as lab_carried.deploy_post does for augment: `average`'s out-of-fold
    letters on the normal bench (decoded with its setting chosen on all), each stage with v12's chosen setting out of
    fold, then each stage fitted on every match's letters before it (bench/train_v12.py's fit_stage)."""
    import joblib
    from bench import train_v12
    from possession import v12
    path = os.path.join(folder(which), "post_A_{}.joblib".format(which))
    if os.path.exists(path):
        return joblib.load(path)
    lab_mix.cache_short_evidence()
    own_dead_cache()
    matches = list(lab.each(which))
    setting = train_v12.chosen(which)
    per_match = {stage: {m.id: s for m in matches} for stage, s in setting.items()}
    letters = {m.id: C.decoded(m, average_p("normal", m.id, which), average_setting(which), which) for m in matches}
    steps = {"robust": letters}
    for stage in v12.CHAIN[which]:
        letters = lab_mix.add_stage(stage, letters, matches, per_match)
        steps[stage] = letters
    models, before = {}, "robust"
    for stage in v12.CHAIN[which]:
        models[stage] = train_v12.fit_stage(stage, matches, steps[before], setting[stage])
        before = stage
    os.makedirs(folder(which), exist_ok=True)
    joblib.dump(models, path, compress=3)
    print("A: post-stage deployment models written; out of fold with the chosen settings {:.4f}".format(
        np.mean([lab.score(m, steps[v12.CHAIN[which][-1]][m.id])["spell_f1"] for m in matches])), flush=True)
    return models


def sc_path():
    return os.path.join(RESULTS, "lab_gate_skillcorner.csv")


def skillcorner_run(ids=None):
    """Branch A on each SkillCorner match: the deployment GRU's P averaged with v12's deployment robust's, decoded with
    `average`'s setting chosen on all 10, v9's snap, v11's move, then short -> segments -> timing with v12's deployment
    models (and, "A retrained", with post stages trained on `average`'s out-of-fold letters)."""
    from bench import lab_skillcorner, skillcorner
    from possession import v12
    which = "all"
    models = v12.load_models(which)
    models_a = {**models, "settings": {**models["settings"], "robust": average_setting(which)}}
    gru = deployment_gru(which)
    post_a = deploy_post_a(which)
    path = sc_path()
    rows = pd.read_csv(path, dtype={"match": str}).to_dict(orient="records") if os.path.exists(path) else []
    have = {str(r["match"]) for r in rows}
    for m in ids or skillcorner.match_ids():
        if m in have:
            continue
        started = time.time()
        data = skillcorner.load(m)
        gt = data["labels"]
        match, feats, feats_s = C.sc_prepared(m)
        c = {"X": feats[:, models["robust"]["cols"]], "X_swapped": feats_s[:, models["robust"]["cols"]]}
        p_rob = C.proba(models["robust"]["clf"], c)
        with np.load(lab_skillcorner.pred_path(m)) as z:
            cached_p = z["p_robust"]
        p_gru = gru_proba(gru, {"X": feats, "X_swapped": feats_s}, match.halves)
        p_avg = np.round(np.clip((p_gru + p_rob) / 2, *L.CLIP), L.DECIMALS)
        seen = ~np.isnan(data["ball"][:, 0])
        dead = gt == D
        for name, post in (("A", None), ("A retrained", post_a)):
            steps = C.deploy_chain(match, p_avg, models_a, which, post)
            for stage, letters in steps.items():
                sc = lab_skillcorner.score(gt, letters)
                called = letters == "D"
                rows.append({"match": m, "start": name, "stage": stage,
                             **{k: sc[k] for k in ("spell_f1", "macro_f1", "possessions", "true_possessions")},
                             "dead_called": float(called[gt != X].mean()),
                             "dead_recall_ball_seen": float(called[dead & seen].mean()),
                             "live_called_dead": float(called[np.isin(gt, (H, A))].mean()),
                             "p_dead_ball_seen": float(p_avg[dead & seen, 2].mean()),
                             "robust_p_max_diff_vs_cached": float(np.abs(p_rob - cached_p).max())})
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.5f")
        final = {r["start"]: r["spell_f1"] for r in rows if r["match"] == m and r["stage"] == "timing"}
        print("{}: {} (robust P against lab_skillcorner's cached {:.1e}; {:.0f} s)".format(
            m, ", ".join("{} {:.4f}".format(k, v) for k, v in final.items()), float(np.abs(p_rob - cached_p).max()),
            time.time() - started), flush=True)
        del match, feats, feats_s, c, data


def sc_signals():
    """The signal on SkillCorner's 20 matches (v12's inputs on its tracking, lab_carried's cache): run only after the
    signal and the threshold rule were fixed."""
    from bench import skillcorner
    rows = []
    for m in skillcorner.match_ids():
        match, _, _ = C.sc_prepared(m)
        rows.append({"match": m, "signal": signal(match.track, match.pred, match.halves, match.pitch)})
        print("{}: {:.4f}".format(m, rows[-1]["signal"]), flush=True)
        del match
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "lab_gate_skillcorner_signals.csv"), index=False, float_format="%.5f")
    return df



# --------------------------------------------------------------------------------------------
# The switch bar (docs/lab.md, round 7, "A switch"), every part
# --------------------------------------------------------------------------------------------

def quantiles(v):
    v = np.asarray(list(v), dtype=np.float64)
    return {"min": float(v.min()), "q25": float(np.percentile(v, 25)), "median": float(np.median(v)),
            "q75": float(np.percentile(v, 75)), "max": float(v.max()), "n": int(len(v))}


def report():
    out = {"bar": "docs/lab.md round 7, a switch, against branch A", "signal": {
        "dead_at": DEAD_AT, "trim_end_s": TRIM_END / FPS, "live_below": LIVE_BELOW, "moving": C.MOVING, "look": C.LOOK,
        "threshold": THRESHOLD}}
    tables = []
    for which in ("all", "idsse"):
        part = {"reproduction": reproduction(which)}
        for name in ("gru_check", "refit_check"):
            p = os.path.join(folder(which), name + ".csv")
            if os.path.exists(p):
                part[name + " max |diff|"] = float(pd.read_csv(p).max_abs_diff.max())
        df, rule = switch(which)
        tables.append(df)
        part["threshold rule"] = rule
        part["left-out thresholds"] = quantiles(df[df.variant == "normal"].threshold)
        sig = signal_table(which)
        part["signal by variant"] = {v: {"all": quantiles(sig[v].values()),
                                         **{s: quantiles([x for m, x in sig[v].items() if sets.source(m) == s])
                                            for s in sorted({sets.source(m) for m in sig[v]})}} for v in VARIANTS}
        per = {}
        for v in VARIANTS:
            d = df[df.variant == v]
            a, b, g = d.A.mean(), d.B.mean(), d.gate.mean()
            per[v] = {"A": float(a), "B": float(b), "gate": float(g), "gate - A": float(g - a), "B - A": float(b - a),
                      "better branch": "B" if b > a else "A", "gate - better": float(g - max(a, b)),
                      "picked B": int((d.pick == "B").sum()), "matches": len(d),
                      "gate better than A": int((d.gate > d.A).sum()), "gate worse than A": int((d.gate < d.A).sum()),
                      "by source gate - A": {s: float((d[d.source == s].gate - d[d.source == s].A).mean())
                                             for s in sorted(d.source.unique())},
                      "unseen gate - A": float((d[d.unseen].gate - d[d.unseen].A).mean())}
        part["by variant"] = per
        n = per["normal"]
        part["part 1 (normal: A everywhere, or loses <= 0.002)"] = bool(n["picked B"] == 0 or n["gate - A"] >= -0.002)
        s = per["stressed"]
        need = int(np.ceil(0.9 * s["matches"] - 1e-9))
        part["part 2 (stressed: B on >= 90 %, gain >= +0.02)"] = bool(s["picked B"] >= need and s["gate - A"] >= 0.02)
        part["part 2 needs B on"] = need
        part["part 3 (each partial level within 0.005 of the better branch)"] = {
            v: bool(per[v]["gate - better"] >= -0.005) for v in LEVELS}
        out[which] = part
    pd.concat(tables).to_csv(os.path.join(RESULTS, "lab_gate_switch.csv"), index=False, float_format="%.5f")
    sc_a = sc_path()
    sc_sig = os.path.join(RESULTS, "lab_gate_skillcorner_signals.csv")
    if os.path.exists(sc_a) and os.path.exists(sc_sig) and THRESHOLD is not None:
        a = pd.read_csv(sc_a, dtype={"match": str})
        a = a[a.stage == "timing"].pivot(index="match", columns="start", values="spell_f1")
        b = pd.read_csv(os.path.join(RESULTS, "lab_carried_skillcorner.csv"), dtype={"match": str})
        b = b[b.stage == "timing"].pivot(index="match", columns="start", values="spell_f1")
        sg = pd.read_csv(sc_sig, dtype={"match": str}).set_index("match").signal
        t = pd.DataFrame({"signal": sg, "A": a["A"], "A retrained": a["A retrained"], "B": b["augment"],
                          "B retrained": b["augment retrained"], "v12": b["v12"]})
        t["pick"] = np.where(t.signal > THRESHOLD, "B", "A")
        t["gate"] = np.where(t.pick == "B", t.B, t.A)
        t["gate retrained"] = np.where(t.pick == "B", t["B retrained"], t["A retrained"])
        t.to_csv(os.path.join(RESULTS, "lab_gate_skillcorner_switch.csv"), float_format="%.5f")
        sc = {"signal": quantiles(t.signal), "picked B": int((t.pick == "B").sum()), "matches": len(t)}
        for tag, ga, gb in (("", "A", "gate"), (" retrained", "A retrained", "gate retrained")):
            gain = t[gb] - t[ga]
            sc["gate{} - A{}".format(tag, tag)] = {"A": float(t[ga].mean()), "gate": float(t[gb].mean()),
                                                   "B": float(t["B" + tag].mean()), "gain": float(gain.mean()),
                                                   "better": int((gain > 0).sum())}
        g = sc["gate - A"]
        sc["part 4 (gain >= +0.02, 15 of 20 better)"] = bool(g["gain"] >= 0.02 and g["better"] >= 15)
        sc["v12"] = float(t.v12.mean())
        out["skillcorner"] = sc
    with open(os.path.join(RESULTS, "lab_gate.json"), "w") as handle:
        json.dump(out, handle, indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--prepare", action="store_true", help="partly stressed features")
    parser.add_argument("--fit", action="store_true", help="robust's and augment's fold models, P on every variant")
    parser.add_argument("--gru", action="store_true", help="the lab's fold GRUs on every variant")
    parser.add_argument("--signals", action="store_true", help="the signal on every bench variant (no scores)")
    parser.add_argument("--chains", choices=VARIANTS, help="branch A and B on one bench variant")
    parser.add_argument("--branches", nargs="+", default=list(BRANCHES))
    parser.add_argument("--sc-signals", action="store_true")
    parser.add_argument("--skillcorner", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        return prepare(args.which)
    if args.fit:
        return fit_folds(args.which)
    if args.gru:
        return gru_all(args.which)
    if args.signals:
        return bench_signals()
    if args.chains:
        return chains(args.chains, args.which, args.branches)
    if args.sc_signals:
        return sc_signals()
    if args.skillcorner:
        return skillcorner_run()
    if args.report:
        return report()


if __name__ == "__main__":
    main()
