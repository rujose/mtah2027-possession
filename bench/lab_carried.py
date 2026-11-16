"""Lab candidate `carried` (round 7, a robustness candidate): dead-ball calls that survive a ball tracked and
moving through a stoppage (docs/lab/carried.md).

    python -m bench.lab_carried --measure-sc                 # ~10 s: SkillCorner's dead-time ball, measured (the recipe's inputs)
    python -m bench.lab_carried --stress [--set all|idsse]   # ~1 min: the stressed bench (balls, costs, robust's features), checked
    OMP_NUM_THREADS=4 python -m bench.lab_carried --fit [--set ...]   # ~20 min: robust's fold models on the stressed bench, augment's
    python -m bench.lab_carried --grid [--set all|idsse]     # ~10 min: `players`' floor, chosen leave one match out
    python -m bench.lab_carried --chains BENCH [--set ...] [--starts ...]   # ~1.5 min a start: v12's chain, BENCH normal|stressed
    python -m bench.lab_carried --sc-prepare [--ids ...]     # ~2.5 min a match: v12's inputs and robust's features on SkillCorner
    python -m bench.lab_carried --deploy-post players|augment   # ~5 min: post-stage deployment models trained on a start's letters
    python -m bench.lab_carried --skillcorner                # ~15 s a match: v12 (reproduced), players, augment on SkillCorner
    python -m bench.lab_carried --report                     # the robustness bar, every part: bench/results/lab_carried.json

Results (2026-11-09, docs/lab/carried.md): neither candidate clears the bar. Both fail part 1 (normal bench: players
-0.016, augment -0.0095 on both sets) and pass parts 2 (stressed bench) and 3 (SkillCorner: players 0.391, augment
0.412 against v12's 0.308, 20/20). The `players_w0.5` start is post hoc, information only.

The problem (docs/lab/skillcorner.md, "What explains the gap" 1): in SkillCorner's stoppages the ball is tracked on
47 % of dead frames, moving 88 % of that time, beyond a line 31 % of it; robust learned "dead <=> no ball" from a bench
whose dead ball is frozen (IDSSE, blanked by robust) or blank (Metrica), so v12 calls 19 % of those frames D.

Everything below was fixed before anything was scored on it (docs/lab/carried.md gives the why of each):

THE STRESSED BENCH (a test). Each bench match's damaged tracking (lab cache, 41 %), with the ball inside every true
dead stretch replaced by a SkillCorner stoppage's ball, transplanted: a SkillCorner stoppage with the same kind of
restart (corner, throw-in, goal kick, kick-off, free kick; from where the ball is when play restarts) and one of the
K nearest lengths is drawn (seeded); its ball, relative to its own restart spot, is laid on the bench's restart spot
(mirrored so the two restarts are on the same side), time-stretched to the bench stoppage's length; its frames with
no ball (the broadcast's cut-aways) stay missing. The first 2 s are blended from where the bench's ball was when the
stoppage began (so it doesn't jump). Live frames are untouched. Then, as on any feed: v1's ball clues and v6's / v7's
costs recomputed, robust's still-ball blanking and its features recomputed. v4's P(dead), v6's restart P and PathCRF
read the players only and are kept (PathCRF's P is kept as the normal bench's; it answers only on v6's live frames,
which the stressed costs could move: an approximation). Templates for the test come from SkillCorner's odd-indexed
10 matches (sc_pool('test'), seed TEST_SEED); `augment` trains on the even-indexed 10 (sc_pool('train'), TRAIN_SEED).

THE CANDIDATES (each the start of v12's chain, decoded with robust's own decoder pick for that match):
  players   robust's P with P(dead) floored at w x v4's players-only P(dead) (H and A scaled down, their ratio kept);
            w chosen leave one match out on the start's score over the normal and stressed bench together.
  augment   robust's fold models retrained on each training match as it is plus the same match under the training
            stress (twice the rows); everything else robust's.
"""

import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from bench import chain, lab, lab_dead, lab_mix, lab_robust, sets, skillcorner
from bench.eval_pathcrf import UNSEEN, v7_cost
from bench.train_restart import cost_with_restart
from possession import lab_learned_features as llf
from possession import v1, v9, v11

# Params (fixed before anything was scored)
NAME = "carried"
L = lab_robust.L
RESULTS = lab.RESULTS
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_carried")      # gitignored (data/*)
H, A, D, X = 0, 1, 2, 3                    # scorer codes
FPS = 25
LOOK = 5                                   # frames (0.2 s): the ball's speed window, as lab_skillcorner's dead diagnosis
MOVING = 0.3                               # m/s over LOOK: "moving" (lab_skillcorner.dead_diagnosis; robust's STILL)
OUT_MARGIN = 0.5                           # m beyond a line: "beyond a line" (lab_skillcorner.OUT_MARGIN)
REL_BINS = 10                              # where in a stoppage, in tenths of its length
LENGTH_BINS = (0, 2, 5, 10, 20, 40, np.inf)      # s
BEFORE_RESTART = (0.2, 0.5, 1, 2, 4, 8, 16)      # s before the restart: how far the ball is from the restart spot
# The stress
TEST_SEED, TRAIN_SEED = 0, 1               # the test bench's draw, augment's training draw
NEAREST = 5                                # templates: one of the 5 of the same kind nearest in length (log ratio)
RAMP = 2 * FPS                             # frames: blended from where the bench's ball was, at most half the stoppage
SPOT_LOOK = FPS                            # frames: the restart spot is the first seen (clean) ball within 1 s of the restart
# `players`
FLOORS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)      # w; 0 is v12's start (ties go to the first)
BENCHES = ("normal", "stressed")


def sc_pool(kind):
    """SkillCorner's matches whose stoppages a stress draws from: test (odd-indexed) or train (even-indexed)."""
    ids = skillcorner.match_ids()
    return ids[1::2] if kind == "test" else ids[0::2]


def folder(which):
    return CACHE + chain.suffix(which)


# --------------------------------------------------------------------------------------------
# SkillCorner's dead-time ball (the stress recipe's only source)
# --------------------------------------------------------------------------------------------

def dead_runs(labels, halves):
    """[(a, b)] maximal runs of D inside each half, b exclusive."""
    out = []
    for _, first, last in halves:
        d = labels[first:last + 1] == D
        edges = np.flatnonzero(np.diff(np.r_[0, d.astype(np.int8), 0]))
        out += [(first + int(a), first + int(b)) for a, b in zip(edges[::2], edges[1::2])]
    return out


def ball_speed(ball):
    """m/s over the next LOOK frames (NaN where unknown)."""
    v = np.full(len(ball), np.nan)
    v[:-LOOK] = np.hypot(*(ball[LOOK:] - ball[:-LOOK]).T) * (25.0 / LOOK)
    return v


def beyond(ball, pitch, margin=OUT_MARGIN):
    """m beyond the nearest line (<= 0 inside the pitch); NaN where the ball is missing."""
    x, y = ball[:, 0], ball[:, 1]
    return np.max(np.stack([-x, x - pitch[0], -y, y - pitch[1]]), axis=0)


def restart_kind(p, pitch):
    """Where a restart is taken, from the ball at the first live frame: 'corner', 'touchline' (throw-in),
    'goal_area' (goal kick), 'centre' (kick-off) or 'field' (free kick)."""
    if np.isnan(p[0]):
        return "unknown"
    x, y = p
    corner = min(np.hypot(x - cx, y - cy) for cx in (0, pitch[0]) for cy in (0, pitch[1]))
    if corner <= 3.0:
        return "corner"
    if min(y, pitch[1] - y) <= 1.5:
        return "touchline"
    if np.hypot(x - pitch[0] / 2, y - pitch[1] / 2) <= 2.0:
        return "centre"
    if min(x, pitch[0] - x) <= 6.5 and abs(y - pitch[1] / 2) <= 10.0:
        return "goal_area"
    return "field"


def measure_sc_match(m):
    """One SkillCorner match: per stoppage and per dead frame, what its ball does."""
    data = skillcorner.load(m)
    labels, ball, players, pitch, halves = data["labels"], data["ball"].astype(np.float64), data["players"], data["pitch"], data["halves"]
    seen = ~np.isnan(ball[:, 0])
    speed = ball_speed(ball)
    out = beyond(ball, pitch)
    with np.errstate(invalid="ignore"):
        d_near = np.nanmin(np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1)), axis=1)
    blank = lab_robust.still_frames(data["ball"], halves)           # what robust's blanking would remove
    stops, frames = [], []
    for a, b in dead_runs(labels, halves):
        n = b - a
        ends_half = any(b - 1 == last for _, _, last in halves)
        starts_half = any(a == first for _, first, _ in halves)
        restart = ball[b] if b < len(ball) and labels[b] in (H, A) else np.array([np.nan, np.nan])
        if np.isnan(restart[0]) and b + 1 < len(ball) and labels[b + 1] in (H, A):
            restart = ball[b + 1]
        before = ball[a - 1] if a > 0 and labels[a - 1] in (H, A) else np.array([np.nan, np.nan])
        s = seen[a:b]
        # the still run that ends the stoppage (ball seen, under MOVING), in frames
        tail = 0
        for k in range(b - 1, a - 1, -1):
            if seen[k] and speed[k] < MOVING:
                tail += 1
            else:
                break
        runs_seen = np.flatnonzero(np.diff(np.r_[0, s.astype(np.int8), 0]))
        stops.append({"match": m, "a": a, "length_s": n / 25.0, "seen": float(s.mean()), "seen_first": bool(s[0]),
                      "seen_last": bool(s[-1]), "seen_runs": len(runs_seen) // 2, "starts_half": starts_half,
                      "ends_half": ends_half, "kind": restart_kind(restart, pitch),
                      "out_before": float(beyond(before[None], pitch)[0]),
                      "still_tail_s": tail / 25.0, "blanked": float(blank[a:b].mean()),
                      **{"to_restart_{}s".format(t): float(np.hypot(*(ball[b - int(t * 25)] - restart)))
                         if n > int(t * 25) and not np.isnan(restart[0]) and seen[b - int(t * 25)] else np.nan
                         for t in BEFORE_RESTART}})
        rel = np.minimum((np.arange(n) * REL_BINS) // n, REL_BINS - 1)
        k = np.arange(a, b)
        frames.append(pd.DataFrame({"match": m, "stop": len(stops) - 1, "rel": rel, "since_s": (k - a) / 25.0,
                                    "until_s": (b - k) / 25.0, "seen": seen[k], "speed": speed[k], "beyond": out[k],
                                    "near": d_near[k], "blanked": blank[k], "kind": stops[-1]["kind"],
                                    "length_s": n / 25.0}))
    live = np.isin(labels, (H, A))
    live_stats = {"match": m, "seen": float(seen[live].mean()), "moving": float(np.nanmean(speed[live & seen] >= MOVING)),
                  "beyond": float(np.nanmean(out[live & seen] > OUT_MARGIN)), "near_median": float(np.nanmedian(d_near[live & seen])),
                  "speed_median": float(np.nanmedian(speed[live & seen])), "blanked": float(blank[live].mean())}
    return pd.DataFrame(stops), pd.concat(frames, ignore_index=True), live_stats


def measure_sc():
    """SkillCorner's dead-time ball, pooled over its 20 matches: the numbers the stress recipe is built from."""
    stops, frames, live = [], [], []
    for m in skillcorner.match_ids():
        s, f, lv = measure_sc_match(m)
        stops.append(s)
        frames.append(f)
        live.append(lv)
        print("{}: {} stoppages".format(m, len(s)), flush=True)
    S, F = pd.concat(stops, ignore_index=True), pd.concat(frames, ignore_index=True)
    os.makedirs(CACHE, exist_ok=True)
    S.to_csv(os.path.join(CACHE, "sc_stoppages.csv"), index=False)
    sf = F[F.seen]
    q = lambda v, ps=(10, 25, 50, 75, 90): np.round(np.nanpercentile(v, ps), 2).tolist()   # noqa: E731
    out = {
        "dead frames": len(F), "stoppages": len(S), "stoppages a match": len(S) / 20,
        "seen share of dead frames": float(F.seen.mean()),
        "moving share of seen dead frames": float((sf.speed >= MOVING).mean()),
        "beyond a line share of seen dead frames": float((sf.beyond > OUT_MARGIN).mean()),
        "blanked by robust's rule, share of seen dead frames": float(sf.blanked.mean()),
        "speed of the seen dead ball, quantiles 10/25/50/75/90 (m/s)": q(sf.speed),
        "speed of the moving seen dead ball, quantiles": q(sf.speed[sf.speed >= MOVING]),
        "nearest player to the seen dead ball, quantiles (m)": q(sf.near),
        "beyond the line when beyond, quantiles (m)": q(sf.beyond[sf.beyond > OUT_MARGIN]),
        "stoppage length quantiles (s)": q(S.length_s),
        "live": {k: float(np.mean([v[k] for v in live])) for k in live[0] if k != "match"},
    }
    by_rel = F.groupby("rel").agg(seen=("seen", "mean"))
    by_rel["moving"] = sf.groupby("rel").speed.apply(lambda v: float((v >= MOVING).mean()))
    by_rel["beyond"] = sf.groupby("rel").beyond.apply(lambda v: float((v > OUT_MARGIN).mean()))
    by_rel["near_median"] = sf.groupby("rel").near.median()
    by_rel["speed_median"] = sf.groupby("rel").speed.median()
    out["by tenth of the stoppage"] = by_rel.round(3).to_dict(orient="index")
    S["length_bin"] = pd.cut(S.length_s, LENGTH_BINS, right=False).astype(str)
    F["length_bin"] = pd.cut(F.length_s, LENGTH_BINS, right=False).astype(str)
    sf = F[F.seen]
    by_len = S.groupby("length_bin", sort=False).agg(stoppages=("seen", "size"), seen_mean=("seen", "mean"),
                                                      seen_first=("seen_first", "mean"), seen_last=("seen_last", "mean"),
                                                      fully_seen=("seen", lambda v: float((v == 1).mean())),
                                                      never_seen=("seen", lambda v: float((v == 0).mean())),
                                                      seen_runs=("seen_runs", "mean"), still_tail_median=("still_tail_s", "median"))
    by_len["frames"] = F.groupby("length_bin", sort=False).size()
    by_len["seen_frames"] = F.groupby("length_bin", sort=False).seen.mean()
    by_len["moving"] = sf.groupby("length_bin", sort=False).speed.apply(lambda v: float((v >= MOVING).mean()))
    by_len["beyond"] = sf.groupby("length_bin", sort=False).beyond.apply(lambda v: float((v > OUT_MARGIN).mean()))
    out["by stoppage length (s)"] = by_len.round(3).to_dict(orient="index")
    kinds = S.groupby("kind").agg(stoppages=("seen", "size"), seen=("seen", "mean"), length_median=("length_s", "median"),
                                  out_before=("out_before", lambda v: float((v > OUT_MARGIN).mean())),
                                  still_tail_median=("still_tail_s", "median"))
    kinds["beyond"] = sf.groupby("kind").beyond.apply(lambda v: float((v > OUT_MARGIN).mean()))
    kinds["moving"] = sf.groupby("kind").speed.apply(lambda v: float((v >= MOVING).mean()))
    out["by restart kind"] = kinds.round(3).to_dict(orient="index")
    out["ball to the restart spot, median m, s before the restart"] = {
        str(t): float(np.nanmedian(S["to_restart_{}s".format(t)])) for t in BEFORE_RESTART}
    out["still tail before the restart (s), quantiles"] = q(S.still_tail_s)
    for side, col in (("since the stoppage started", "since_s"), ("until the restart", "until_s")):
        bins = pd.cut(sf[col], [0, 1, 2, 4, 8, 16, np.inf], right=False)
        g = sf.groupby(bins, observed=True)
        out["seen dead ball by s " + side] = pd.DataFrame({
            "seen": F.groupby(pd.cut(F[col], [0, 1, 2, 4, 8, 16, np.inf], right=False), observed=True).seen.mean(),
            "moving": g.speed.apply(lambda v: float((v >= MOVING).mean())),
            "beyond": g.beyond.apply(lambda v: float((v > OUT_MARGIN).mean())),
            "near_median": g.near.median(), "speed_median": g.speed.median()}).round(3).rename(index=str).to_dict(orient="index")
    with open(os.path.join(RESULTS, "lab_carried_skillcorner_ball.json"), "w") as handle:
        json.dump(out, handle, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    return out


# --------------------------------------------------------------------------------------------
# The stress: SkillCorner's dead-time ball transplanted into the bench's true stoppages
# --------------------------------------------------------------------------------------------

def first_seen(ball, frames):
    for k in frames:
        if 0 <= k < len(ball) and not np.isnan(ball[k, 0]):
            return ball[k].astype(np.float64)
    return None


def templates(pool):
    """SkillCorner stoppages with a restart spot: kind, length, the ball relative to the restart spot, seen."""
    out = []
    for m in sc_pool(pool):
        d = skillcorner.load(m)
        labels, ball, halves, pitch = d["labels"], d["ball"].astype(np.float64), d["halves"], d["pitch"]
        for a, b in dead_runs(labels, halves):
            if b >= len(labels) or labels[b] not in (H, A):
                continue
            spot = first_seen(ball, [k for k in range(b, b + SPOT_LOOK) if k < len(labels) and labels[k] in (H, A)])
            if spot is None:
                continue
            seg = ball[a:b]
            out.append({"match": m, "kind": restart_kind(spot, pitch), "length": b - a, "spot": spot,
                        "pitch": pitch, "rel": (seg - spot).astype(np.float32), "seen": ~np.isnan(seg[:, 0])})
        del d
    return out


def stress_ball(m, ball, clean, truth, halves, pitch, seed, pool_templates):
    """(the damaged ball with every true stoppage's ball replaced, a table of what was done)."""
    rng = np.random.default_rng([seed, zlib.crc32(m.encode())])
    by_kind = {}
    for t in pool_templates:
        by_kind.setdefault(t["kind"], []).append(t)
    new = ball.astype(np.float64).copy()
    rows = []
    for a, b in dead_runs(truth, halves):
        n = b - a
        if b >= len(truth) or truth[b] not in (H, A):
            rows.append({"a": a, "length": n, "stressed": False, "why": "no restart in the half"})
            continue
        live_after = [k for k in range(b, b + SPOT_LOOK) if k < len(truth) and truth[k] in (H, A)]
        spot = first_seen(clean, live_after)
        if spot is None:
            spot = first_seen(ball, live_after)
        if spot is None:
            rows.append({"a": a, "length": n, "stressed": False, "why": "no restart spot"})
            continue
        kind = restart_kind(spot, pitch)
        pool = by_kind.get(kind) or pool_templates
        lengths = np.array([t["length"] for t in pool], dtype=np.float64)
        near = np.argsort(np.abs(np.log(lengths / n)), kind="stable")[:NEAREST]
        t = pool[int(rng.choice(near))]
        idx = np.zeros(1, int) + t["length"] - 1 if n == 1 else np.round(np.arange(n) * (t["length"] - 1) / (n - 1)).astype(int)
        sign = np.array([1.0 if (t["spot"][i] - t["pitch"][i] / 2) * (spot[i] - pitch[i] / 2) >= 0 else -1.0 for i in (0, 1)])
        pos = spot + t["rel"][idx].astype(np.float64) * sign
        seen = t["seen"][idx]
        before = first_seen(clean, [k for k in range(a - 1, a - 1 - SPOT_LOOK, -1) if k >= 0 and truth[k] in (H, A)])
        if before is not None and seen.any():
            j0 = int(np.argmax(seen))
            r = max(1, min(RAMP, n // 2))
            w = np.clip(1.0 - np.arange(n) / r, 0.0, 1.0)[:, None]
            pos = pos + (before - pos[j0]) * w
        new[a:b] = np.where(seen[:, None], pos, np.nan)
        rows.append({"a": a, "length": n, "stressed": True, "kind": kind, "template": "{}:{}".format(t["match"], t["length"]),
                     "template_length": t["length"], "seen": float(seen.mean()), "ramp": before is not None})
    return new.astype(np.float32), pd.DataFrame(rows)


def stress_path(m, seed):
    return os.path.join(CACHE, "ball_{}_{}.npz".format(seed, m))


def stressed_ball(m, seed):
    """The match's stressed ball (cached; the same for both sets: the tracking is)."""
    path = stress_path(m, seed)
    if not os.path.exists(path):
        match = lab.load(m, "all" if m in lab.matches("all") else "idsse")
        clean = sets.load(m)["ball"]
        pool = templates("test" if seed == TEST_SEED else "train")
        ball, table = stress_ball(m, match.track["ball"], clean, match.truth, match.halves, match.pitch, seed, pool)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, ball=ball)
        table.to_csv(path.replace(".npz", ".csv"), index=False)
    with np.load(path) as z:
        return z["ball"]


def ball_costs(match, ball, which):
    """v6's and v7's costs on a new ball, as bench/eval_pathcrf.py made them (v1's clues without tags, stopped and
    active zeroed; v4's P(dead), v6's restart P and PathCRF's P as they are)."""
    # v1's clues read the players as delivered, also where the lab cache holds cleaned ones (bench/lab_clean.py)
    players = match.track.get("players_raw", match.track["players"])
    team = match.track.get("team_raw", match.track["team"])
    n, slots = players.shape[:2]
    track = {**match.track, "players": players, "team": team, "ball": ball, "speed": np.zeros((n, slots), np.float32)}
    clues = v1.compute_clues(track, [], match.halves, match.pitch)
    base = v1.build_costs(dict(clues, stopped=np.zeros_like(clues["stopped"]), active=np.zeros_like(clues["active"])))
    d, r, pc = chain.dead_ball(which), chain.restart(which), chain.pathcrf(which)
    v6 = cost_with_restart(base, match.pred["p_dead"], match.pred["p_start"], match.halves, r["weight"], r["carry"], d["weight"])
    v7 = v7_cost(v6, match.pred["p_pathcrf"], np.isnan(ball[:, 0]), pc["weight"], pc["where"])
    return v6.astype(np.float32), v7.astype(np.float32)


def stressed(m, which, seed=TEST_SEED, match=None):
    """The lab match with the stressed ball and the costs recomputed on it."""
    match = match or lab.load(m, which)
    ball = stressed_ball(m, seed)
    v6, v7 = ball_costs(match, ball, which)
    match.track = {**match.track, "ball": ball}
    match.pred = {**match.pred, "v6_cost": v6, "v7_cost": v7}
    match.stressed = seed
    return match


def match_of(m, which, bench):
    return lab.load(m, which) if bench == "normal" else stressed(m, which)


def ball_check(which):
    """What the stressed ball looks like, against SkillCorner's (before any score): truth only to find the stoppages."""
    rows = []
    for m in lab.matches(which):
        match = lab.load(m, which)
        for name, ball in (("normal", match.track["ball"]), ("stressed", stressed_ball(m, TEST_SEED)),
                           ("train stress", stressed_ball(m, TRAIN_SEED) if sets.source(m) == "idsse" else None)):
            if ball is None:
                continue
            seen = ~np.isnan(ball[:, 0])
            speed = ball_speed(ball.astype(np.float64))
            out = beyond(ball, match.pitch) > OUT_MARGIN
            blank = lab_robust.still_frames(ball, match.halves)
            for part, mask in (("dead", match.truth == D), ("live", np.isin(match.truth, (H, A)))):
                rows.append({"match": m, "source": match.source, "ball": name, "part": part, "frames": int(mask.sum()),
                             "seen": float(seen[mask].mean()), "moving": float(np.mean(speed[mask & seen] >= MOVING)),
                             "beyond": float(out[mask & seen].mean()), "blanked_of_seen": float(blank[mask & seen].mean()),
                             "missing_after_blanking": float((~seen | blank)[mask].mean())})
        print("{}: checked".format(m), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "lab_carried_ball{}.csv".format(chain.suffix(which))), index=False, float_format="%.4f")
    print(df.groupby(["ball", "part", "source"])[["seen", "moving", "beyond", "blanked_of_seen", "missing_after_blanking"]].mean().round(3).to_string())
    return df


# --------------------------------------------------------------------------------------------
# robust's features on the stressed bench; robust's and augment's fold models
# --------------------------------------------------------------------------------------------

def feature_path(m, which, seed):
    return os.path.join(folder(which), "features_{}_{}.npz".format(seed, m))


def stressed_features(m, which, seed):
    """robust's 97 features on the stressed match (blanked, touches and passes on the blanked feed), as
    lab_robust.cached; cached here."""
    path = feature_path(m, which, seed)
    if not os.path.exists(path):
        b = lab_robust.blanked(stressed(m, which, seed))
        touches = v11.seen_touches(b.track, v11.SETTINGS[which]["dv"])
        passes = v9.seen_passes(b.track)
        names, X = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes)
        _, Xs = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes, swap=True)
        os.makedirs(folder(which), exist_ok=True)
        np.savez(path, names=np.array(names), X=X, X_swapped=Xs, truth=b.truth.astype(np.int8), blanked=b.blanked)
    with np.load(path) as z:
        return {"names": [str(k) for k in z["names"]], "X": z["X"], "X_swapped": z["X_swapped"], "truth": z["truth"],
                "blanked": z["blanked"]}


def features_of(m, which, bench):
    return lab_robust.cached(m, which) if bench == "normal" else stressed_features(m, which, TEST_SEED)


def fit_robust(used, which, augment=False):
    """lab_robust.fit's classifier (blank), on `used` in order; augment adds each match under the training stress."""
    Xs, ys = [], []
    for m in used:
        parts = [lab_robust.cached(m, which)] + ([stressed_features(m, which, TRAIN_SEED)] if augment else [])
        for c in parts:
            rows = np.flatnonzero(np.isin(c["truth"], (0, 1, 2)))[::L.EVERY]
            y = c["truth"][rows].astype(np.int64)
            Xs += [c["X"][rows], c["X_swapped"][rows]]
            ys += [y, L.SWAP[y]]
            del c
    X, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs
    return HistGradientBoostingClassifier(**L.TREES).fit(X, y)


def proba(clf, c):
    p = (clf.predict_proba(c["X"]) + clf.predict_proba(c["X_swapped"])[:, L.SWAP]) / 2
    return np.round(np.clip(p, *L.CLIP), L.DECIMALS)


def p_path(kind, bench, m, which):
    return os.path.join(folder(which), "p_{}_{}_{}.npy".format(kind, bench, m))


def held_out_p(kind, bench, m, which):
    """robust's or augment's held-out P on the normal or stressed bench (robust on the normal bench: lab_robust's)."""
    if kind == "robust" and bench == "normal":
        return lab_robust.held_out_proba(m, which, "blank")
    return np.load(p_path(kind, bench, m, which)).astype(np.float64)


def stress(which):
    """The stressed balls and robust's features on them (test draw; training draw on the IDSSE matches); the check."""
    for m in lab.matches(which):
        started = time.time()
        stressed_features(m, which, TEST_SEED)
        if sets.source(m) in L.TRAIN_SOURCES:
            stressed_features(m, which, TRAIN_SEED)
        print("{}: stressed features ({:.0f} s)".format(m, time.time() - started), flush=True)
    ball_check(which)


def prepare(which):
    """Each fold's robust (on the stressed bench) and augment (both benches): held-out P saved."""
    ms = lab.matches(which)
    models = {}
    for held in ms:
        used = tuple(m for m in ms if m != held and sets.source(m) in L.TRAIN_SOURCES)
        if used not in models:
            started = time.time()
            models = {used: {"robust": fit_robust(used, which), "augment": fit_robust(used, which, augment=True)}}
            print("{}: fitted on {} ({:.0f} s)".format(held, " ".join(used), time.time() - started), flush=True)
        f = models[used]
        normal = lab_robust.cached(held, which)
        check = proba(f["robust"], normal)
        saved = lab_robust.held_out_proba(held, which, "blank")
        print("  {}: refitted robust reproduces its saved held-out P: max |diff| {:.4f}".format(held, np.abs(check - saved).max()),
              flush=True)
        np.save(p_path("augment", "normal", held, which), proba(f["augment"], normal).astype(np.float32))
        del normal
        st = stressed_features(held, which, TEST_SEED)
        np.save(p_path("robust", "stressed", held, which), proba(f["robust"], st).astype(np.float32))
        np.save(p_path("augment", "stressed", held, which), proba(f["augment"], st).astype(np.float32))
        del st


# --------------------------------------------------------------------------------------------
# `players`: robust's P(dead) floored by v4's players-only P(dead)
# --------------------------------------------------------------------------------------------

def floor(p, p_dead, w):
    """P with P(dead) at least w x v4's P(dead); H and A scaled so the row sums to 1 (their ratio kept); clipped
    and rounded as robust's. w = 0 returns P unchanged."""
    if not w:
        return p
    d = np.maximum(p[:, 2], np.clip(w * np.nan_to_num(np.asarray(p_dead, dtype=np.float64)), 0.0, L.CLIP[1]))
    scale = (1.0 - d) / np.maximum(1.0 - p[:, 2], 1e-9)
    q = np.stack([p[:, 0] * scale, p[:, 1] * scale, d], axis=1)
    return np.round(np.clip(q, *L.CLIP), L.DECIMALS)


def decoded(match, p, setting, which):
    """P decoded with a robust-style setting, then v9's snap and v11's move."""
    s = {k: v for k, v in setting.items() if k in ("switch_team", "switch_dead", "min_possession")}
    return np.asarray(L.finish(match, L.decode_cost(match, -np.log(p), **s), which)).astype("<U1")


def grid(which):
    """The start's spellF1 per (bench, match, w), robust's decoder pick for the match."""
    picks = lab_mix.fold_settings(which)
    rows = []
    for bench in BENCHES:
        for m in lab.matches(which):
            match = match_of(m, which, bench)
            p = held_out_p("robust", bench, m, which)
            for w in FLOORS:
                s = lab.score(match, decoded(match, floor(p, match.pred["p_dead"], w), picks["robust"][m], which))
                rows.append({"bench": bench, "match": m, "w": w, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                             "d_f1": s["d_f1"], "possessions": s["possessions"]})
            print("{} {}: {}".format(bench, m, " ".join("{:.4f}".format(r["spell_f1"]) for r in rows[-len(FLOORS):])), flush=True)
            del match
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(RESULTS, "lab_carried_players_grid{}.csv".format(chain.suffix(which))), index=False, float_format="%.5f")
    return g


def pick_floor(g, which):
    """w per match: best on the other matches' mean over both benches (ties: the first, the smallest w); and on all."""
    both = g.groupby(["match", "w"], sort=False).spell_f1.mean().reset_index()      # each match: mean of its two benches
    picks = {}
    for m in both.match.unique():
        others = both[both.match != m].groupby("w", sort=False).spell_f1.mean()
        picks[m] = float(others.index[int(np.argmax(others.round(6).values))])
    overall = both.groupby("w", sort=False).spell_f1.mean()
    chosen = float(overall.index[int(np.argmax(overall.round(6).values))])
    out = {"folds": picks, "chosen": chosen, "mean_by_w": {
        bench: g[g.bench == bench].groupby("w").spell_f1.mean().round(5).to_dict() for bench in BENCHES}}
    with open(os.path.join(RESULTS, "lab_carried_players{}.json".format(chain.suffix(which))), "w") as handle:
        json.dump(out, handle, indent=1)
    return out


def floor_picks(which):
    with open(os.path.join(RESULTS, "lab_carried_players{}.json".format(chain.suffix(which)))) as handle:
        return json.load(handle)


# --------------------------------------------------------------------------------------------
# Chains: each start, then v12's post stages refitted on its output
# --------------------------------------------------------------------------------------------

def average_picks(which):
    from bench import lab_temporal
    with open(os.path.join(RESULTS, "lab_{}{}.json".format(lab_temporal.report_name("average"), chain.suffix(which)))) as handle:
        return {f["match"]: f["picked"] for f in json.load(handle)["folds"]}


def start_letters(match, start, which, picks, w):
    """The letters before the post stages for a start: robust (v12), players, augment, average, average + players."""
    if start in ("average", "average_players"):
        from bench import lab_temporal
        p = lab_temporal.held_out(match.id, which, "average")
        setting = picks["average"][match.id]
    else:
        bench = "stressed" if getattr(match, "stressed", None) is not None else "normal"
        p = held_out_p("augment" if start == "augment" else "robust", bench, match.id, which)
        setting = picks["robust"][match.id]
    if start in ("players", "average_players"):
        p = floor(p, match.pred["p_dead"], w[match.id])
    elif start.startswith("players_w"):                 # post hoc, information only: a fixed floor (not a candidate)
        p = floor(p, match.pred["p_dead"], float(start[len("players_w"):]))
    return decoded(match, p, setting, which)


def own_dead_cache():
    """lab_dead's feature cache: read the existing one, write new files only under data/external/lab_carried/."""
    original = lab_dead.frame_features

    def frame_features(match):
        import hashlib
        key = hashlib.sha1(match.pred["p_dead"].tobytes() + match.track["ball"].tobytes()).hexdigest()[:12]
        name = "{}_{}.npy".format(match.id, key)
        for base in (lab_dead.CACHE, os.path.join(CACHE, "lab_dead")):
            if os.path.exists(os.path.join(base, name)):
                return lab_dead.FEATURES["ball"], np.load(os.path.join(base, name))
        names, X = lab_dead.compute(match)
        os.makedirs(os.path.join(CACHE, "lab_dead"), exist_ok=True)
        np.save(os.path.join(CACHE, "lab_dead", name), X)
        return names, X
    lab_dead.frame_features = frame_features
    return original


def chains(which, bench, starts):
    from possession import v12
    lab_mix.cache_short_evidence()
    own_dead_cache()
    matches = [match_of(m, which, bench) for m in lab.matches(which)]
    picks = lab_mix.fold_settings(which)
    if any(s.startswith("average") for s in starts):
        picks["average"] = average_picks(which)
    w = floor_picks(which)["folds"] if any("players" in s for s in starts) else {}
    path = os.path.join(RESULTS, "lab_carried_chains_{}{}.json".format(bench, chain.suffix(which)))
    for start in starts:
        started = time.time()
        letters = {m.id: start_letters(m, start, which, picks, w) for m in matches}
        steps = {"start": lab_mix.scores_of(letters, matches)}
        for stage in v12.CHAIN[which]:
            letters = lab_mix.add_stage(stage, letters, matches, picks)
            steps[stage] = lab_mix.scores_of(letters, matches)
        done = {}
        if os.path.exists(path):
            with open(path) as handle:
                done = json.load(handle)
        done[start] = {k: {m: v["spell_f1"] for m, v in s.items()} for k, s in steps.items()}
        done[start + " macro"] = {m: v["macro_f1"] for m, v in steps[v12.CHAIN[which][-1]].items()}
        done[start + " possessions"] = {m: v["possessions"] for m, v in steps[v12.CHAIN[which][-1]].items()}
        os.makedirs(os.path.join(CACHE, "letters"), exist_ok=True)
        for m in matches:
            np.save(os.path.join(CACHE, "letters", "{}_{}_{}{}.npy".format(start, bench, m.id, chain.suffix(which))), letters[m.id])
        final = done[start][v12.CHAIN[which][-1]]
        print("{} {}: start {:.4f}, chain {:.4f} ({:.0f} s)".format(bench, start, np.mean(list(done[start]["start"].values())),
                                                                    np.mean(list(final.values())), time.time() - started), flush=True)
        with open(path, "w") as handle:
            json.dump(done, handle, indent=1)


# --------------------------------------------------------------------------------------------
# SkillCorner: each candidate as v12's start, deployment models (docs/lab/skillcorner.md's run)
# --------------------------------------------------------------------------------------------

def deployment_augment():
    """augment's deployment model: the 7 IDSSE matches of `all`, each as it is and under the training stress."""
    import joblib
    path = os.path.join(CACHE, "augment_all.joblib")
    if not os.path.exists(path):
        used = tuple(m for m in lab.matches("all") if sets.source(m) in L.TRAIN_SOURCES)
        joblib.dump(fit_robust(used, "all", augment=True), path, compress=3)
    return joblib.load(path)


def deploy_chain(match, p, models, which="all", post=None):
    """v12.run_chain from a given P: decoded with the deployment setting, snapped and moved, then the post stages
    with `post` models (default: v12's deployment models)."""
    from bench import lab_segments, lab_short, lab_timing
    from possession import v12
    post = post or models
    settings = models["settings"]
    letters = lab_robust.finish(match, L.decode_cost(match, -np.log(p), **settings["robust"]), which)
    out = {"start": np.asarray(letters).astype("<U1")}
    for stage in v12.CHAIN[which]:
        s = settings[stage]
        if stage == "short":
            letters = lab_short.apply(match, letters, model=post[stage], **s)
        elif stage == "segments":
            letters = lab_segments.apply(match, letters, post[stage], **s)
        elif stage == "timing":
            letters = lab_timing.apply(match, letters, post[stage], **s)
        out[stage] = np.asarray(letters).astype("<U1")
    return out


def deploy_post(start, which="all"):
    """Post-stage deployment models for a start, as bench/train_v12.py trains v12's: the start's letters on every
    normal bench match out of fold (each stage with its setting chosen on all the matches), then each stage fitted
    on all the matches' letters before it."""
    import joblib
    from bench import train_v12
    from possession import v12
    path = os.path.join(CACHE, "post_{}_{}.joblib".format(start, which))
    if os.path.exists(path):
        return joblib.load(path)
    lab_mix.cache_short_evidence()
    own_dead_cache()
    matches = list(lab.each(which))
    setting = train_v12.chosen(which)
    per_match = {stage: {m.id: s for m in matches} for stage, s in setting.items()}
    w = floor_picks(which)["chosen"]
    letters = {}
    for m in matches:
        p = held_out_p("augment" if start == "augment" else "robust", "normal", m.id, which)
        if start == "players":
            p = floor(p, m.pred["p_dead"], w)
        letters[m.id] = decoded(m, p, setting["robust"], which)
    steps = {"robust": letters}
    for stage in v12.CHAIN[which]:
        letters = lab_mix.add_stage(stage, letters, matches, per_match)
        steps[stage] = letters
    models, before = {}, "robust"
    for stage in v12.CHAIN[which]:
        models[stage] = train_v12.fit_stage(stage, matches, steps[before], setting[stage])
        before = stage
    joblib.dump(models, path, compress=3)
    print("{}: post-stage deployment models written; out of fold with the chosen settings {:.4f}".format(
        start, np.mean([lab.score(m, steps[v12.CHAIN[which][-1]][m.id])["spell_f1"] for m in matches])), flush=True)
    return models


def sc_path(m):
    return os.path.join(CACHE, "skillcorner", "{}.npz".format(m))


def sc_prepare(ids=None):
    """v12's inputs on each SkillCorner match (v12.inputs, as lab_skillcorner ran it) and robust's features, cached."""
    from possession import v12
    for m in ids or skillcorner.match_ids():
        if os.path.exists(sc_path(m)):
            continue
        started = time.time()
        data = skillcorner.load(m)
        halves, pitch = data["halves"], data["pitch"]
        if os.environ.get("MTAH_LAB_CLEAN"):     # lab round 9: PathCRF and the features read the cleaned tracking
            from bench import lab_clean
            _, _, (fixed, pred) = v12.inputs(skillcorner.tracking(data), halves, pitch, "all",
                                             clean=lab_clean.cleaner(os.environ["MTAH_LAB_CLEAN"], halves))
        else:
            fixed, pred = v12.inputs(skillcorner.tracking(data), halves, pitch, "all")
        match = v12.as_match(fixed, pred, halves, pitch, "skillcorner_{}".format(m))
        b = lab_robust.blanked(match)
        touches, passes = v11.seen_touches(b.track, v11.SETTINGS["all"]["dv"]), v9.seen_passes(b.track)
        _, X = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes)
        _, Xs = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes, swap=True)
        os.makedirs(os.path.dirname(sc_path(m)), exist_ok=True)
        np.savez(sc_path(m) + ".part.npz", X=X, X_swapped=Xs, **{"track_" + k: v for k, v in fixed.items()},
                 **{"pred_" + k: v for k, v in pred.items()})
        os.replace(sc_path(m) + ".part.npz", sc_path(m))
        print("{}: inputs and features cached ({:.0f} s)".format(m, time.time() - started), flush=True)
        del data, fixed, pred, match, b, X, Xs


def sc_prepared(m):
    """(the match as v12 wraps it, robust's features, swapped) from sc_prepare's cache."""
    from possession import v12
    data = skillcorner.load(m)
    with np.load(sc_path(m)) as z:
        track = {k[len("track_"):]: z[k] for k in z.files if k.startswith("track_")}
        pred = {k[len("pred_"):]: z[k] for k in z.files if k.startswith("pred_")}
        X, Xs = z["X"], z["X_swapped"]
    return v12.as_match(track, pred, data["halves"], data["pitch"], "skillcorner_{}".format(m)), X, Xs


def skillcorner_run(ids=None):
    """v12 (its cached letters, reproduced), players and augment on each SkillCorner match."""
    from bench import lab_skillcorner
    from possession import v12
    which = "all"
    models = v12.load_models(which)
    aug = deployment_augment()
    w = floor_picks(which)["chosen"]
    import joblib
    post = {name: joblib.load(os.path.join(CACHE, "post_{}_{}.joblib".format(name, which))) for name in ("players", "augment")
            if os.path.exists(os.path.join(CACHE, "post_{}_{}.joblib".format(name, which)))}
    path = os.path.join(RESULTS, "lab_carried_skillcorner.csv")
    rows = pd.read_csv(path).to_dict(orient="records") if os.path.exists(path) else []
    have = {str(r["match"]) for r in rows}
    for m in ids or skillcorner.match_ids():
        if m in have:
            continue
        started = time.time()
        data = skillcorner.load(m)
        gt, halves, pitch = data["labels"], data["halves"], data["pitch"]
        match, feats, feats_s = sc_prepared(m)
        c = {"X": feats[:, models["robust"]["cols"]], "X_swapped": feats_s[:, models["robust"]["cols"]]}
        p_rob = proba(models["robust"]["clf"], c)
        p_aug = proba(aug, {"X": feats, "X_swapped": feats_s})
        with np.load(lab_skillcorner.pred_path(m)) as z:
            cached_v12, cached_p = z["v12"], z["p_robust"]
        seen = ~np.isnan(data["ball"][:, 0])
        dead = gt == D
        runs = [("v12", p_rob, None), ("players", floor(p_rob, match.pred["p_dead"], w), None), ("augment", p_aug, None)]
        runs += [(name + " retrained", p, post[name]) for name, p in (("players", runs[1][1]), ("augment", p_aug)) if name in post]
        runs += [("players_w0.5 (post hoc)", floor(p_rob, match.pred["p_dead"], 0.5), None)]   # information only, not a candidate
        for start, p, post_models in runs:
            steps = deploy_chain(match, p, models, which, post_models)
            if start == "v12":
                differ = int(np.sum(steps["timing"] != cached_v12))
                print("  {}: v12 reproduced (frames differing {}; robust P max |diff| {:.4f})".format(
                    m, differ, float(np.abs(p_rob - cached_p).max())), flush=True)
            for stage, letters in steps.items():
                sc = lab_skillcorner.score(gt, letters)
                called = letters == "D"
                rows.append({"match": m, "pool": "test" if m in sc_pool("test") else "train", "start": start, "stage": stage,
                             **{k: sc[k] for k in ("spell_f1", "macro_f1", "possessions", "true_possessions", "team_acc_alive",
                                                   "alive_dead_f1", "d_f1")},
                             "dead_called": float(called[gt != X].mean()),
                             "dead_recall_ball_seen": float(called[dead & seen].mean()),
                             "dead_recall_ball_missing": float(called[dead & ~seen].mean()) if (dead & ~seen).any() else np.nan,
                             "live_called_dead": float(called[np.isin(gt, (H, A))].mean()),
                             "p_dead_ball_seen": float(p[dead & seen, 2].mean())})
        pd.DataFrame(rows).to_csv(path, index=False, float_format="%.5f")
        final = {r["start"]: r["spell_f1"] for r in rows if r["match"] == m and r["stage"] == "timing"}
        print("{}: {} ({:.0f} s)".format(m, ", ".join("{} {:.4f}".format(k, v) for k, v in final.items()), time.time() - started), flush=True)
        del match, feats, feats_s, c, data


# --------------------------------------------------------------------------------------------
# The robustness bar (docs/lab.md, round 7), every part
# --------------------------------------------------------------------------------------------

def compare(cand, base, which):
    """Means, by source, unseen, matches better: candidate against base, per-match spellF1 dicts."""
    ms = sorted(cand)
    unseen = set(UNSEEN)
    d = {m: cand[m] - base[m] for m in ms}

    def part(sel):
        sel = [m for m in sel if m in d]
        return {"cand": float(np.mean([cand[m] for m in sel])), "base": float(np.mean([base[m] for m in sel])),
                "gain": float(np.mean([d[m] for m in sel])), "better": int(sum(d[m] > 0 for m in sel)), "matches": len(sel)}
    out = {"all": part(ms), "idsse": part([m for m in ms if sets.source(m) == "idsse"]),
           "metrica": part([m for m in ms if sets.source(m) == "metrica"]) if any(sets.source(m) == "metrica" for m in ms) else None,
           "unseen": part([m for m in ms if m in unseen]), "per_match": d}
    return out


def p_diagnostics(which):
    """Mean P(dead) of each start on the true dead frames with the ball seen / missing and on live frames, both
    benches (truth only to split; a diagnostic, nothing chosen from it)."""
    w = floor_picks(which)["folds"]
    rows = []
    for m in lab.matches(which):
        match = lab.load(m, which)
        for bench in BENCHES:
            ball = match.track["ball"] if bench == "normal" else stressed_ball(m, TEST_SEED)
            seen = ~np.isnan(ball[:, 0])
            dead, live = match.truth == D, np.isin(match.truth, (H, A))
            rob = held_out_p("robust", bench, m, which)
            for start, p in (("robust", rob), ("players", floor(rob, match.pred["p_dead"], w[m])),
                             ("augment", held_out_p("augment", bench, m, which))):
                rows.append({"match": m, "bench": bench, "start": start, "dead_seen": float(p[dead & seen, 2].mean()),
                             "dead_missing": float(p[dead & ~seen, 2].mean()), "live": float(p[live, 2].mean()),
                             "dead_seen_argmax_d": float((p[dead & seen].argmax(axis=1) == 2).mean()),
                             "live_argmax_d": float((p[live].argmax(axis=1) == 2).mean())})
    df = pd.DataFrame(rows)
    return df.groupby(["bench", "start"]).mean(numeric_only=True).round(3).to_dict(orient="index")


def report():
    from possession import v12
    out = {"bar": "docs/lab.md round 7, the robustness candidate"}
    for which in ("all", "idsse"):
        last = v12.CHAIN[which][-1]
        res = {}
        for bench in BENCHES:
            path = os.path.join(RESULTS, "lab_carried_chains_{}{}.json".format(bench, chain.suffix(which)))
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                res[bench] = json.load(handle)
        part = {}
        if "normal" in res:
            n = res["normal"]
            with open(os.path.join(RESULTS, "lab_temporal_chains{}.json".format(chain.suffix(which)))) as handle:
                temporal = json.load(handle)["scores"]          # v12's and average's chains, as lab_temporal ran them
            for name in ("robust", "average"):
                if name not in n:
                    n[name] = {last: temporal[name]}
                    part["{} normal chain from".format(name)] = "lab_temporal_chains"
                else:
                    part["{} normal chain: max |diff| against lab_temporal's".format(name)] = float(max(
                        abs(n[name][last][m] - temporal[name][m]) for m in temporal[name]))
            part["v12 normal (recomputed)"] = float(np.mean(list(n["robust"][last].values())))
            for c in ("players", "augment"):
                if c in n:
                    r = compare(n[c][last], n["robust"][last], which)
                    srcs = [r["idsse"]] + ([r["metrica"]] if r["metrica"] else [])
                    r["passes_part1"] = bool(abs(r["all"]["gain"]) <= 0.002 or r["all"]["gain"] > 0) and all(
                        x["gain"] >= -0.005 for x in srcs + [r["unseen"]])
                    part["normal " + c] = r
            for c, base in (("average_players", "average"), ("average_players", "robust"), ("average", "robust")):
                if c in n and base in n:
                    part["normal {} vs {}".format(c, base)] = compare(n[c][last], n[base][last], which)
        if "stressed" in res:
            st = res["stressed"]
            part["v12 stressed"] = float(np.mean(list(st["robust"][last].values())))
            if "normal" in res:
                part["v12 drop normal -> stressed"] = compare(st["robust"][last], res["normal"]["robust"][last], which)
            for c in ("players", "augment"):
                if c in st:
                    r = compare(st[c][last], st["robust"][last], which)
                    need = int(np.ceil(0.7 * r["all"]["matches"] - 1e-9))
                    r["passes_part2"] = bool(r["all"]["gain"] >= 0.02 and r["all"]["better"] >= need)
                    part["stressed " + c] = r
                    part["stressed {} start".format(c)] = compare(st[c]["start"], st["robust"]["start"], which)
        if os.path.exists(os.path.join(RESULTS, "lab_carried_players{}.json".format(chain.suffix(which)))):
            part["floor"] = floor_picks(which)
            part["p_dead diagnostics"] = {"{} {}".format(*k): v for k, v in p_diagnostics(which).items()}
        out[which] = part
    path = os.path.join(RESULTS, "lab_carried_skillcorner.csv")
    if os.path.exists(path):
        sc = pd.read_csv(path, dtype={"match": str})
        fin = sc[sc.stage == "timing"].pivot(index="match", columns="start", values="spell_f1")
        scp = {"matches": len(fin), "v12": float(fin["v12"].mean())}
        for c in ("players", "augment", "players retrained", "augment retrained"):
            if c in fin:
                g = fin[c] - fin["v12"]
                scp[c] = {"mean": float(fin[c].mean()), "gain": float(g.mean()), "better": int((g > 0).sum()),
                          "passes_part3": bool(g.mean() >= 0.02 and (g > 0).sum() >= 15),
                          "by_pool": {pool: {"gain": float(g[sc.drop_duplicates("match").set_index("match").loc[g.index, "pool"] == pool].mean()),
                                             "better": int((g[sc.drop_duplicates("match").set_index("match").loc[g.index, "pool"] == pool] > 0).sum())}
                                      for pool in ("test", "train")}}
        means = sc.groupby(["start", "stage"]).mean(numeric_only=True)
        scp["means"] = {"{} {}".format(*k): v for k, v in means.round(4).to_dict(orient="index").items()}
        out["skillcorner"] = scp
    with open(os.path.join(RESULTS, "lab_carried.json"), "w") as handle:
        json.dump(out, handle, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items()}, indent=1, default=lambda x: round(float(x), 4))[:20000])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--measure-sc", action="store_true", help="SkillCorner's dead-time ball, measured")
    parser.add_argument("--stress", action="store_true", help="the stressed bench: balls, features, the check")
    parser.add_argument("--fit", action="store_true", help="the fold models and their held-out P")
    parser.add_argument("--grid", action="store_true", help="players: the floor's grid and its picks")
    parser.add_argument("--chains", choices=BENCHES, help="v12's chain from each start on this bench")
    parser.add_argument("--starts", nargs="+", default=None)
    parser.add_argument("--ids", nargs="+", default=None, help="SkillCorner match ids (default: all 20)")
    parser.add_argument("--skillcorner", action="store_true")
    parser.add_argument("--sc-prepare", action="store_true", help="v12's inputs and robust's features on SkillCorner, cached")
    parser.add_argument("--deploy-post", choices=("players", "augment"), help="post-stage deployment models for a start")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.measure_sc:
        return measure_sc()
    if args.stress:
        return stress(args.which)
    if args.fit:
        return prepare(args.which)
    if args.grid:
        g = grid(args.which)
        out = pick_floor(g, args.which)
        print(json.dumps(out, indent=1))
        return
    if args.chains:
        default = ("robust", "players", "augment") + (("average", "average_players") if args.chains == "normal" else ())
        return chains(args.which, args.chains, args.starts or default)
    if args.deploy_post:
        return deploy_post(args.deploy_post)
    if args.sc_prepare:
        return sc_prepare(args.ids)
    if args.skillcorner:
        return skillcorner_run()
    if args.report:
        return report()


if __name__ == "__main__":
    main()
