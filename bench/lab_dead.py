"""Lab candidate `dead`: missed stoppages found by a second dead-ball classifier (docs/lab/dead.md).

    python -m bench.lab_dead --explore              # ~1 min, exploration: what the missed stoppages look like (truth read)
    python -m bench.lab_dead [--set all|idsse]      # ~5 min: leave one match out over the grid, judged against the mix
    python -m bench.lab_dead --nested [--set ..]    # ~15 min, after the run: settings picked with no model trained on the held-out match

Idea 10f, round 2 (on the mix). The mix misses ~16 true stoppages a match (median 5 s), and inside
them v4's P(dead) is low (median 0.32): v4 doesn't see them. A per-frame classifier is trained on
the other bench matches, on the frames the input letters call a team's and that sit at least CLEAR
from any D in them (the inside of predicted possessions), to say which of those frames are dead.
It reads v4's P(dead) with its context plus what v4 lacks: deceleration, the players' accelerations
and how coherently each team moves, and (feature set "ball") the ball: still or moving, seen outside
the pitch and how long ago, how long unseen, players closing on it or not, how long since a touch,
where it lies (corner, 6-yard box, centre spot). Then every stretch of those frames whose smoothed
probability stays above `thr` for at least `min_len` s becomes D; a sliver of a possession that the
cut leaves under SLIVER frames goes with it.

The ball set is trained on IDSSE matches only: Metrica's samples blank the ball while it's dead, so
a model trained on them would learn "no ball = dead" too strongly for our match (docs/ideas.md #2).

A post stage: `fit(matches, letters_of, **setting)` and `apply(match, letters, model, **setting)`
work on any input letters and read only the match's damaged tracking and the out-of-sample outputs
on it (v4's P(dead)), never its truth (read only to train on other matches and to score).
"""

import argparse
import hashlib
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from bench import lab, lab_budget, lab_mix
from bench.lab_stoppage import outside
from possession import features as v4_features
from possession import v11
from possession.data import AWAY, FPS, HOME

# Params (by reasoning, fixed before any held-out run; docs/lab/dead.md gives the why of each)
STAGE = "post"
NAME = "dead"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_dead")
SPAN = v4_features.SPEED_SPAN    # 2 frames each side: velocities over 0.16 s, as v4's speeds
ACCEL = 5                        # frames each side: accelerations from velocities 0.4 s apart
MOVING = 1.0                     # m/s: a player or the ball "moving" (below: jitter or walking on the spot)
FAST = v4_features.FAST          # 4 m/s: "running", as v4
SLOW = v4_features.SLOW          # 1 m/s: "standing", as v4
MIN_MOVERS = 3                   # players moving needed for a team's coherence to mean anything
K_NEAREST = 3                    # players nearest the ball whose closing speed is averaged
CLOSE = 5.0                      # m: "close to the ball"
CORNER = 3.0                     # m: the ball at a corner (corner kick)
SPOT = 2.0                       # m: the ball at the centre spot (kick-off)
SIX_DEPTH, SIX_HALF = v4_features.SIX_DEPTH, v4_features.SIX_HALF
DV = 3.0                         # m/s: v11's touch ΔV, so a "touch" means what it means there
CAP = 10.0                       # s: "since / until the ball moved" clipped here
GAP_CAP = 30.0                   # s: "since / until seen, seen outside, touched" clipped here
CONTEXT = (2, 5)                 # s: means over the past and the next this-many seconds
CLEAR = FPS                      # frames: a frame is eligible only this far from any D in the input letters
STEP = 5                         # training frames: every 5th (0.2 s apart; neighbours are near-copies), as v4
SMOOTH = FPS                     # frames: centred mean of P before thresholding (1 s), so a run doesn't flicker
SLIVER = 20                      # frames: a possession our cut leaves shorter than this goes to D (v3's minimum possession)
OFF = 1.0                        # a threshold no probability exceeds: the stage is off
CLASSIFIER = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 100,
              "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}

PLAYER_BASE = ["p_dead", "speed_mean", "share_slow", "share_fast", "accel_median", "team_speed_max",
               "team_speed_min", "coherence_max", "coherence_min"]
PLAYER_CONTEXT = ["p_dead", "speed_mean", "share_slow", "share_fast", "accel_median", "team_speed_max", "coherence_max"]
BALL_BASE = ["ball_seen", "since_seen", "until_seen", "ball_speed", "since_moving", "until_moving",
             "ball_out", "since_out", "until_out", "line_dist", "corner_dist", "in_six", "spot_dist",
             "near_dist", "near_count", "closing", "follow", "since_touch", "until_touch"]
BALL_CONTEXT = ["ball_seen", "ball_speed", "ball_out", "near_dist", "closing", "follow"]


def _context(names):
    return ["{}_{}_{}s".format(n, w, s) for n in names for s in CONTEXT for w in ("past", "next")]


PLAYERS = PLAYER_BASE + _context(PLAYER_CONTEXT) + ["speed_drop"]
FEATURES = {"players": PLAYERS, "ball": PLAYERS + BALL_BASE + _context(BALL_CONTEXT)}
TRAIN_ON = {"players": ("idsse", "metrica"), "ball": ("idsse",)}   # sources a feature set is trained on

# The grid, simplest first: the no-op, then players-only before the ball; within each, the most
# cautious first (highest threshold, longest minimum stretch).
THRESHOLDS = (0.7, 0.6, 0.5, 0.4, 0.3)
MIN_LENS = (4.0, 2.0, 1.0)
COLUMNS = ["features", "thr", "min_len"]
GRID = [{"features": "players", "thr": OFF, "min_len": 4.0}] + [
    {"features": f, "thr": t, "min_len": m} for f in FEATURES for t in THRESHOLDS for m in MIN_LENS]


# --------------------------------------------------------------------------------------------
# Per-frame features (prediction-time inputs only)
# --------------------------------------------------------------------------------------------

def since(mask, halves, cap):
    """s since the last True frame within the half (0 on True frames); `cap` where there's none."""
    out = np.full(len(mask), np.nan, dtype=np.float32)
    for _, first, last in halves:
        m = mask[first:last + 1]
        index = np.arange(len(m))
        seen = np.maximum.accumulate(np.where(m, index, -10 ** 9))
        out[first:last + 1] = np.minimum((index - seen) / FPS, cap)
    return out


def until(mask, halves, cap):
    """s until the next True frame within the half; `cap` where there's none."""
    out = np.full(len(mask), np.nan, dtype=np.float32)
    for _, first, last in halves:
        out[first:last + 1] = since(mask[first:last + 1][::-1], [(0, 0, last - first)], cap)[::-1]
    return out


def forward_filled(ball, halves):
    """The ball's last seen position within the half; NaN before the first."""
    out = np.full(ball.shape, np.nan, dtype=np.float32)
    for _, first, last in halves:
        b = ball[first:last + 1]
        index = np.maximum.accumulate(np.where(~np.isnan(b[:, 0]), np.arange(len(b)), -1))
        out[first:last + 1] = np.where(index[:, None] >= 0, b[np.maximum(index, 0)], np.nan)
    return out


def velocities(points, halves, span=SPAN):
    """m/s vectors from positions `span` frames either side, within each half; NaN where either end is missing.
    points: [frames, ..., 2]."""
    out = np.full(points.shape, np.nan, dtype=np.float32)
    for _, first, last in halves:
        p = points[first:last + 1]
        out[first + span:last + 1 - span] = (p[2 * span:] - p[:-2 * span]) / (2 * span / FPS)
    return out


def centred_mean(values, window, halves):
    """Mean over the `window` frames centred on t, within each half, ignoring NaN."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    for _, first, last in halves:
        v = values[first:last + 1].astype(np.float64)
        valid = ~np.isnan(v)
        total = np.r_[0.0, np.cumsum(np.where(valid, v, 0.0))]
        count = np.r_[0, np.cumsum(valid)]
        index = np.arange(len(v))
        low, high = np.maximum(index - window // 2, 0), np.minimum(index + window - window // 2, len(v))
        n = count[high] - count[low]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[first:last + 1] = np.where(n > 0, (total[high] - total[low]) / n, np.nan)
    return out


def base_signals(match):
    """{name: float32 [frames]} before time context."""
    tr, halves = match.track, match.halves
    players, team, ball = tr["players"], tr["team"], tr["ball"]
    length, width = match.pitch
    f = {"p_dead": match.pred["p_dead"].astype(np.float32)}
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN frames give NaN, as intended
        # The players
        v = velocities(players, halves)                         # [frames, slots, 2]
        speed = np.hypot(v[..., 0], v[..., 1])
        known = ~np.isnan(speed)
        f["speed_mean"] = np.nanmean(speed, axis=1)
        f["share_slow"] = v4_features.share(known & (speed < SLOW), known)
        f["share_fast"] = v4_features.share(known & (speed > FAST), known)
        dv = velocities(v, halves, ACCEL)                       # m/s^2
        f["accel_median"] = np.nanmedian(np.hypot(dv[..., 0], dv[..., 1]), axis=1)
        del dv
        moving = known & (speed > MOVING)
        unit = np.where(moving[..., None], v / np.where(moving, speed, 1.0)[..., None], np.nan)
        team_speed, coherence = [], []
        for side in (HOME, AWAY):
            cols = team == side
            team_speed.append(np.hypot(*np.nanmean(v[:, cols], axis=1).T))
            c = np.hypot(*np.nanmean(unit[:, cols], axis=1).T)
            coherence.append(np.where(moving[:, cols].sum(axis=1) >= MIN_MOVERS, c, np.nan))
        f["team_speed_max"], f["team_speed_min"] = np.fmax(*team_speed), np.fmin(*team_speed)
        f["coherence_max"], f["coherence_min"] = np.fmax(*coherence), np.fmin(*coherence)
        del unit

        # The ball
        seen = ~np.isnan(ball[:, 0])
        f["ball_seen"] = seen.astype(np.float32)
        f["since_seen"], f["until_seen"] = since(seen, halves, GAP_CAP), until(seen, halves, GAP_CAP)
        vb = velocities(ball, halves)
        bspeed = np.hypot(vb[:, 0], vb[:, 1])
        f["ball_speed"] = bspeed
        ball_moving = bspeed > MOVING
        f["since_moving"], f["until_moving"] = since(ball_moving, halves, CAP), until(ball_moving, halves, CAP)
        out = outside(ball, match.pitch) & seen
        f["ball_out"] = np.where(seen, out, np.nan).astype(np.float32)
        f["since_out"], f["until_out"] = since(out, halves, GAP_CAP), until(out, halves, GAP_CAP)
        pos = forward_filled(ball, halves)
        x, y = pos[:, 0], pos[:, 1]
        f["line_dist"] = np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))
        f["corner_dist"] = np.min([np.hypot(x - cx, y - cy) for cx in (0, length) for cy in (0, width)], axis=0)
        f["in_six"] = (((x < SIX_DEPTH) | (x > length - SIX_DEPTH)) & (np.abs(y - width / 2) < SIX_HALF)).astype(np.float32)
        f["in_six"][np.isnan(x)] = np.nan
        f["spot_dist"] = np.hypot(x - length / 2, y - width / 2)
        rel = ball[:, None, :] - players                         # from each player to the ball
        d = np.hypot(rel[..., 0], rel[..., 1])                  # NaN where either is unseen
        f["near_dist"] = np.nanmin(d, axis=1)
        f["near_count"] = np.where(seen, (d < CLOSE).sum(axis=1), np.nan).astype(np.float32)
        toward = ((v - vb[:, None, :]) * rel).sum(axis=2) / d    # closing speed on the ball, m/s
        toward = np.where(np.isnan(d) | np.isnan(toward), np.nan, toward)
        order = np.argsort(np.where(np.isnan(toward), np.inf, d), axis=1)[:, :K_NEAREST]
        f["closing"] = np.nanmean(np.take_along_axis(toward, order, axis=1), axis=1)
        del rel, d, toward, order
        mean_v = np.nanmean(v, axis=1)                          # the players' mean velocity
        mspeed = np.hypot(mean_v[:, 0], mean_v[:, 1])
        both = (bspeed > MOVING) & (mspeed > 0)
        f["follow"] = np.where(both, (mean_v * vb).sum(axis=1) / np.where(both, bspeed * mspeed, 1.0), np.nan)
        found = v11.seen_touches(tr, DV)
        touched = found[HOME] | found[AWAY]
        f["since_touch"], f["until_touch"] = since(touched, halves, GAP_CAP), until(touched, halves, GAP_CAP)
    return {k: np.asarray(val, dtype=np.float32) for k, val in f.items()}


def compute(match):
    """(names, X float32 [frames, features]): every feature of FEATURES["ball"]; NaN outside the halves."""
    f = base_signals(match)
    for name in PLAYER_CONTEXT + BALL_CONTEXT:
        for seconds in CONTEXT:
            for ahead, word in ((False, "past"), (True, "next")):
                f["{}_{}_{}s".format(name, word, seconds)] = v4_features.window_mean(f[name], seconds * FPS, match.halves, ahead)
    f["speed_drop"] = f["speed_mean_past_5s"] - f["speed_mean_next_5s"]
    names = FEATURES["ball"]
    X = np.stack([f[n] for n in names], axis=1).astype(np.float32)
    inside = np.zeros(len(X), bool)
    for _, first, last in match.halves:
        inside[first:last + 1] = True
    X[~inside] = np.nan
    return names, X


def frame_features(match):
    """compute(match), cached under data/external/lab_dead/ keyed by the match and its inputs."""
    key = hashlib.sha1(match.pred["p_dead"].tobytes() + match.track["ball"].tobytes()).hexdigest()[:12]
    path = os.path.join(CACHE, "{}_{}.npy".format(match.id, key))
    if os.path.exists(path):
        return FEATURES["ball"], np.load(path)
    names, X = compute(match)
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, X)
    return names, X


def columns(features):
    return [FEATURES["ball"].index(n) for n in FEATURES[features]]


# --------------------------------------------------------------------------------------------
# The interface: fit on some matches, apply to another
# --------------------------------------------------------------------------------------------

def eligible(letters, halves):
    """bool [frames]: a team's frame in the letters, at least CLEAR frames from any D in them (same half)."""
    letters = np.asarray(letters)
    dead = letters == "D"
    far = (since(dead, halves, CAP) >= CLEAR / FPS) & (until(dead, halves, CAP) >= CLEAR / FPS)
    return np.isin(letters, ["H", "A"]) & far


def rows(match, letters):
    """(X, y) training rows of one match: its eligible frames, every STEP-th; y = the truth says D."""
    _, X = frame_features(match)
    keep = eligible(letters, match.halves)
    keep[np.arange(len(keep)) % STEP != 0] = False
    return X[keep], (match.truth_letters[keep] == "D").astype(np.int8)


def train(parts, features):
    """The classifier on [(source, X, y)] parts, those of the sources TRAIN_ON[features] only."""
    use = [(X, y) for source, X, y in parts if source in TRAIN_ON[features]]
    X = np.concatenate([X for X, _ in use])[:, columns(features)]
    y = np.concatenate([y for _, y in use])
    return {"features": features, "clf": HistGradientBoostingClassifier(**CLASSIFIER).fit(X, y)}


def fit(matches, letters_of=None, features="players", **_):
    """The model trained on `matches` (each read once) with their input letters `letters_of(match)`
    (default: the mix's kept letters). Only `features` matters; thresholds are apply's."""
    letters_of = letters_of or (lambda m: lab_mix.kept_letters(m.id))
    parts = []
    for match in matches:
        if match.source in TRAIN_ON[features]:
            parts.append((match.source,) + rows(match, letters_of(match)))
    return train(parts, features)


def probabilities(match, model):
    """P(dead) per frame from the model, on every frame inside the halves (NaN outside)."""
    _, X = frame_features(match)
    p = np.full(len(X), np.nan, dtype=np.float32)
    inside = ~np.isnan(X).all(axis=1)
    p[inside] = model["clf"].predict_proba(X[inside][:, columns(model["features"])])[:, 1]
    return p


def act(letters, p, halves, thr=OFF, min_len=4.0):
    """(letters with the confident stretches written D, [(start, stop)] written).

    A stretch: a maximal run of eligible frames whose P, smoothed over SMOOTH frames (eligible frames
    only), stays above `thr`, at least `min_len` s long. Then a possession our cut left shorter than
    SLIVER frames, next to a written stretch, goes to D too."""
    letters = np.asarray(letters).astype("<U1")
    out = letters.copy()
    if thr >= OFF:
        return out, []
    ok = eligible(letters, halves)
    smooth = centred_mean(np.where(ok, p, np.nan), SMOOTH, halves)
    hit = ok & (smooth > thr)
    written = []
    for _, first, last in halves:
        h = hit[first:last + 1].astype(np.int8)
        edges = np.diff(np.r_[0, h, 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= min_len * FPS:
                out[first + a:first + b] = "D"
                written.append((first + a, first + b))
    for a, b in written:
        for side in ("left", "right"):
            _sliver(out, letters, a, b, side, halves)
    return out, written


def _sliver(out, letters, a, b, side, halves):
    """If the team run next to [a, b) in `out` is under SLIVER frames and was longer in `letters`
    (our cut made it), it becomes D."""
    first, last = next((f, l) for _, f, l in halves if f <= a <= l)
    k = a - 1 if side == "left" else b
    if k < first or k > last or out[k] not in ("H", "A"):
        return
    team, step = out[k], -1 if side == "left" else 1
    j = k
    while first <= j + step <= last and out[j + step] == team:
        j += step
    lo, hi = min(j, k), max(j, k) + 1
    if hi - lo >= SLIVER:
        return
    orig_lo, orig_hi = lo, hi
    while orig_lo - 1 >= first and letters[orig_lo - 1] == team:
        orig_lo -= 1
    while orig_hi <= last and letters[orig_hi] == team:
        orig_hi += 1
    if orig_hi - orig_lo > hi - lo:
        out[lo:hi] = "D"


def apply(match, letters, model, thr=OFF, min_len=4.0, **_):
    """Letters for every frame of `match`: `letters` with the stretches the model is confident are
    dead written D."""
    if thr >= OFF:
        return np.asarray(letters).astype("<U1").copy()
    return act(letters, probabilities(match, model), match.halves, thr, min_len)[0]


# --------------------------------------------------------------------------------------------
# The bench
# --------------------------------------------------------------------------------------------

def missed_runs(letters, truth):
    """[(start, stop, same team either side in the truth)] true D runs with no D in `letters`."""
    r = lab_budget.runs(truth)
    out = []
    for i, (a, b, k) in enumerate(r):
        if k == "D" and not np.any(letters[a:b] == "D"):
            prev = r[i - 1][2] if i else "E"
            nxt = r[i + 1][2] if i + 1 < len(r) else "E"
            out.append((a, b, prev == nxt and prev in ("H", "A")))
    return out


def explore(which):
    """Exploration (truth read on every match; nothing here picks a setting): the missed stoppages
    on the kept letters, and how each source's damaged ball behaves while dead."""
    rows_, ball, ceiling = [], [], []
    for m in lab.each(which):
        pred = np.asarray(lab_mix.kept_letters(m.id, which)).astype("<U1")
        truth = m.truth_letters.astype("<U1")
        missed = missed_runs(pred, truth)
        base = lab.score(m, pred)["spell_f1"]
        gains = {"match": m.id}
        for name, keep, shrink in (("all", lambda a, b, s: True, 0), ("same team either side", lambda a, b, s: s, 0),
                                   ("different teams", lambda a, b, s: not s, 0), ("2 s or longer", lambda a, b, s: b - a >= 2 * FPS, 0),
                                   ("4 s or longer", lambda a, b, s: b - a >= 4 * FPS, 0), ("0.5 s short each end", lambda a, b, s: True, FPS // 2),
                                   ("1 s short each end", lambda a, b, s: True, FPS), ("2 s short each end", lambda a, b, s: True, 2 * FPS)):
            fixed = pred.copy()
            for a, z, same in missed:
                if keep(a, z, same) and z - shrink > a + shrink:
                    fixed[a + shrink:z - shrink] = "D"
            gains[name] = lab.score(m, fixed)["spell_f1"] - base
        ceiling.append(gains)
        b = m.track["ball"]
        seen = ~np.isnan(b[:, 0])
        out = outside(b, m.pitch) & seen
        for label in ("D", "H", "A"):
            mask = truth == label
            ball.append({"match": m.id, "source": m.source, "label": "live" if label != "D" else "dead",
                         "seen": seen[mask].mean(), "outside_if_seen": out[mask].sum() / max(seen[mask].sum(), 1)})
        n_true = sum(1 for _, _, k in lab_budget.runs(truth) if k == "D")
        for a, z, same in missed_runs(pred, truth):
            p = pred[a:z]
            rows_.append({"match": m.id, "source": m.source, "true_stoppages": n_true, "len_s": (z - a) / FPS,
                          "same_team": same, "inside_one_possession": len(set(p)) == 1,
                          "seen": seen[a:z].mean(), "seen_outside": out[a:z].any(),
                          "p_dead_median": float(np.median(m.pred["p_dead"][a:z])),
                          "eligible": eligible(pred, m.halves)[a:z].mean()})
    f, g = pd.DataFrame(rows_), pd.DataFrame(ball)
    print("the damaged ball, by source (mean over matches):")
    print(g.groupby(["source", "label"])[["seen", "outside_if_seen"]].mean().round(3).to_string())
    per = f.groupby("match").agg(missed=("len_s", "size"), true=("true_stoppages", "first"),
                                 median_s=("len_s", "median"), same_team=("same_team", "sum"))
    print("\nmissed stoppages on the kept letters, per match:\n" + per.round(2).to_string())
    print("\nmissed: {} ({:.1f} a match); length quantiles 10/25/50/75/90 %: {} s".format(
        len(f), len(f) / f.match.nunique(), "/".join("{:.1f}".format(q) for q in f.len_s.quantile([.1, .25, .5, .75, .9]))))
    print("same team either side (two possessions merged): {} ({:.0%}); inside one predicted possession: {:.0%}".format(
        int(f.same_team.sum()), f.same_team.mean(), f.inside_one_possession.mean()))
    print(f.groupby("source")[["len_s", "seen", "p_dead_median", "eligible"]].median().round(3).to_string())
    print("ball seen outside somewhere inside: " + ", ".join(
        "{} {:.0%}".format(s, v) for s, v in f.groupby("source").seen_outside.mean().items()))
    print("share of their frames eligible (a team's, >= {} frames from a D): {:.0%}".format(CLEAR, f.eligible.mean()))
    c = pd.DataFrame(ceiling).set_index("match")
    print("\nthe ceiling: spellF1 gain if the missed stoppages were written D (mean over matches; better on):")
    print(pd.DataFrame({"gain": c.mean(), "better": (c > 0).sum()}).to_string(float_format="{:+.4f}".format))


def held_out_letters(which):
    """{match: the kept letters} (the mix's out-of-fold letters)."""
    return {m: np.asarray(lab_mix.kept_letters(m, which)).astype("<U1") for m in lab.matches(which)}


def training_parts(which, letters):
    """{match: (source, X, y)}: each match's training rows on its kept letters."""
    parts = {}
    for match in lab.each(which):
        parts[match.id] = (match.source,) + rows(match, letters[match.id])
        print("{}: {} training rows, {:.1%} dead".format(match.id, len(parts[match.id][2]), parts[match.id][2].mean()), flush=True)
    return parts


def score_settings(match, letters, probs):
    """One row per GRID setting on `match`: its scores and what it wrote."""
    truth = match.truth_letters.astype("<U1")
    missed = missed_runs(letters, truth)
    out = []
    for setting in GRID:
        new, written = act(letters, probs[setting["features"]], match.halves, setting["thr"], setting["min_len"])
        s = lab.score(match, new)
        right = sum(np.mean(truth[a:b] == "D") >= 0.5 for a, b in written)
        found = sum(np.any(new[a:b] == "D") for a, b, _ in missed)
        found_same = sum(np.any(new[a:b] == "D") for a, b, same in missed if same)
        out.append({"match": match.id, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "d_f1": s["d_f1"],
                    "possessions": s["possessions"], "true_possessions": s["true_possessions"],
                    "written": len(written), "written_right": int(right), "written_s": sum(b - a for a, b in written) / FPS,
                    "missed": len(missed), "found": int(found), "found_same": int(found_same),
                    "frames_changed": int((new != letters).sum()), "frames_right": int(((new != letters) & (truth == "D")).sum())})
    return out


def auc(match, letters, p):
    """AUC of P on the match's eligible frames (D against a team's), or NaN."""
    ok = eligible(letters, match.halves) & ~np.isnan(p)
    y = match.truth_letters[ok] == "D"
    return roc_auc_score(y, p[ok]) if 0 < y.sum() < len(y) else np.nan


def run(which):
    letters = held_out_letters(which)
    parts = training_parts(which, letters)
    rows_, quality = [], []
    for match in lab.each(which):
        others = [parts[m] for m in parts if m != match.id]
        models = {f: train(others, f) for f in FEATURES}
        probs = {f: probabilities(match, models[f]) for f in FEATURES}
        quality.append({"match": match.id, "source": match.source, "v4_p_dead": auc(match, letters[match.id], match.pred["p_dead"]),
                        **{f: auc(match, letters[match.id], probs[f]) for f in FEATURES}})
        rows_ += score_settings(match, letters[match.id], probs)
        print("{}: AUC on eligible frames v4 {:.3f}, players {:.3f}, ball {:.3f}".format(
            match.id, quality[-1]["v4_p_dead"], quality[-1]["players"], quality[-1]["ball"]), flush=True)
    grid = pd.DataFrame(rows_)
    report = lab.judge(grid, COLUMNS, which, base=lab_mix.kept_base(which))
    means = grid.groupby(COLUMNS, sort=False)[["spell_f1", "macro_f1", "d_f1", "written", "written_right", "found",
                                               "found_same", "missed"]].mean()
    print("means over the matches (not held out):\n" + means.to_string(float_format="%.4f"))
    print(lab.describe(report).replace("vs v11", "vs the mix"))
    picked = []
    for fold in report["folds"]:
        p = fold["picked"]
        picked.append(grid[(grid.match == fold["match"]) & np.all([grid[c] == p[c] for c in COLUMNS], axis=0)].iloc[0])
    picked = pd.DataFrame(picked)
    noop = grid[grid.thr == OFF].set_index("match")
    report["d_f1"] = {"held_out": float(picked.d_f1.mean()), "base": float(noop.d_f1.mean())}
    print("  D's F1 held out {:.4f} vs the mix {:.4f}".format(report["d_f1"]["held_out"], report["d_f1"]["base"]))
    print(picked[["match"] + COLUMNS + ["spell_f1", "written", "written_right", "written_s", "missed", "found", "found_same",
                                        "frames_changed", "frames_right", "possessions", "true_possessions"]].to_string(index=False))
    q = pd.DataFrame(quality)
    print("held-out AUC on eligible frames:\n" + q.round(3).to_string(index=False))
    print(q[["v4_p_dead"] + list(FEATURES)].mean().round(3).to_string())
    report["held_out_auc"] = q.to_dict("records")
    report["held_out_actions"] = picked.to_dict("records")
    lab.save(NAME, grid, report, which)
    return report


def nested(which):
    """Robustness check of the same grid: lab.judge picks match m's setting from the other matches'
    scores, but their models were trained with m's truth in. Here every score that picks m's setting
    comes from models trained without m too (for each other match j: trained without m and j; the
    model without {m, j} is the same for both, so it's trained once per pair)."""
    grid = pd.read_csv(os.path.join(lab.RESULTS, "lab_{}_grid{}.csv".format(NAME, "" if which == "all" else "_" + which)))
    letters = held_out_letters(which)
    parts = training_parts(which, letters)
    ms = list(parts)
    rows_ = []
    for x, m in enumerate(ms):
        for j in ms[x + 1:]:
            models = {f: train([parts[k] for k in ms if k not in (m, j)], f) for f in FEATURES}
            for held, scored in ((m, j), (j, m)):
                match = lab.load(scored, which)
                probs = {f: probabilities(match, models[f]) for f in FEATURES}
                for r in score_settings(match, letters[scored], probs):
                    rows_.append({"without": held, **r})
        print("{}: scored without it and each other match".format(m), flush=True)
    inner = pd.DataFrame(rows_)
    base = lab_mix.kept_base(which)
    folds = []
    for m in ms:
        means = inner[inner.without == m].groupby(COLUMNS, sort=False).spell_f1.mean().reset_index()
        means["_order"] = [next(i for i, g in enumerate(GRID) if all(g[c] == r[c] for c in COLUMNS)) for _, r in means.iterrows()]
        best = means.sort_values("_order", kind="stable").reset_index(drop=True)
        pick = best.loc[best.spell_f1.round(6).idxmax()]
        setting = {c: pick[c] for c in COLUMNS}
        row = grid[(grid.match == m) & np.all([grid[c] == setting[c] for c in COLUMNS], axis=0)].iloc[0]
        folds.append({"match": m, "source": base[m]["source"], "unseen": base[m]["unseen"], "picked": setting,
                      "held_out": float(row.spell_f1), "base": base[m]["spell_f1"]})
    f = pd.DataFrame(folds)
    out = {}
    for name, part in [("all", f)] + [(s, f[f.source == s]) for s in sorted(f.source.unique())] + [("PathCRF-unseen", f[f.unseen])]:
        out[name] = {"held_out": float(part.held_out.mean()), "base": float(part.base.mean()),
                     "better": int((part.held_out > part.base).sum()), "matches": len(part)}
        print("nested {}: {:.4f} vs the mix {:.4f} ({:+.4f}), better on {}/{}".format(
            name, part.held_out.mean(), part.base.mean(), part.held_out.mean() - part.base.mean(),
            int((part.held_out > part.base).sum()), len(part)))
    print(f[["match", "held_out", "base", "picked"]].to_string(index=False))
    path = os.path.join(lab.RESULTS, "lab_{}_nested{}.json".format(NAME, "" if which == "all" else "_" + which))
    pd.Series({"folds": folds, "summary": out}).to_json(path, indent=1, default_handler=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--explore", action="store_true", help="exploration: the missed stoppages (truth read)")
    parser.add_argument("--nested", action="store_true", help="robustness: settings picked with no model trained on the held-out match (after the main run)")
    args = parser.parse_args()
    if args.explore:
        return explore(args.which)
    if args.nested:
        return nested(args.which)
    report = run(args.which)
    if args.which == "all" and report["passes"]:
        print("\npasses on all: the same on IDSSE alone")
        run("idsse")


if __name__ == "__main__":
    main()
