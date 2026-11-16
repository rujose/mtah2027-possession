"""Lab round 9: cleaning the tracking before PathCRF and robust's features read it (docs/lab/cleaning.md).

    OMP_NUM_THREADS=2 python -m bench.lab_clean screen VARIANT [VARIANT ...]   # ~20 min each: P1 and P2
    python -m bench.lab_clean check                    # c0's held-out P against robust's cached one (harness)
    python -m bench.lab_clean report                   # every screened variant against c0; the stage gates
    python -m bench.lab_clean build-lab VARIANT --to COPY/data/external/lab   # stage 3: a cleaned lab cache

A VARIANT is one or more steps joined by "+", applied in order to each bench match's damaged tracking:
  c0                  nothing (the harness on the tracking as the bench gives it)
  o_players           CEILING: the true players (continuous, all 22) with the damaged ball
  o_link              CEILING: the damaged pieces, same holes, joined into the true players
  o_ball1, o_ball3    CEILING: the true ball in every gap of at most 1 s (3 s), both ends seen
  link_g{G}_v{V}      join pieces of a team end-to-start: gap <= G s, start within 1 m + V m/s x gap of the
                      end moved on by its last velocity (at most 1 s of it); greedy by that distance
  ..._fill{F}         after joining, straight lines across holes of at most F s ("all": every joined hole)
  ball{G}             straight lines across ball gaps of at most G s, both ends seen
  smooth{W}           Savitzky-Golay (order 2) over W frames on every unbroken piece of a player

What changes: PathCRF's P (and v7's cost built from it) and robust's 97 features, nothing else (the
dead-ball and restart outputs, v6's cost, v9's snap and v11's move read the tracking as delivered).
P1: PathCRF's team accuracy on live frames (v6 says live, the truth says H or A), ball missing / seen.
P2: robust's trees retrained leave one match out on the variant's features (IDSSE only, as robust),
decoded with robust's chosen setting, snapped and moved on the original tracking; held-out spellF1.

Nothing here reads our match.
"""

import argparse
import json
import os
import re
import shutil
import time
import types
import warnings

import numpy as np
from scipy.signal import savgol_filter

from bench import chain, lab, lab_robust, sets
from bench.eval_pathcrf import UNSEEN, v7_cost
from possession import lab_learned_features as llf
from possession import pathcrf_adapter, v1, v9, v11
from possession.data import FPS, HOME

# Params
L = lab_robust.L
WHICH = "all"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_clean")     # gitignored (data/*)
RESULTS = lab.RESULTS
R0 = 1.0                  # m: the start may always be this far from the predicted point
VEL_FRAMES = 10           # frames: the end velocity, over the piece's last 0.4 s
VEL_CAP = 1.0             # s: the end is moved on by its velocity for at most this long
CHOSEN = lab_robust.CHOSEN


# --------------------------------------------------------------------------------------------
# The cleaners
# --------------------------------------------------------------------------------------------

def pieces(players, team, halves):
    """Every unbroken run of a slot within a half: (start, end exclusive, slot, team)."""
    out = []
    for _, first, last in halves:
        on = ~np.isnan(players[first:last + 1, :, 0])
        for s in range(players.shape[1]):
            for a, e in zip(*v1.runs(on[:, s])):
                out.append((first + int(a), first + int(e), s, int(team[s])))
    return out


def end_state(players, piece):
    """(last position, velocity m/s) of a piece."""
    a, e, s, _ = piece
    p = players[e - 1, s].astype(np.float64)
    k = min(VEL_FRAMES, e - 1 - a)
    v = (p - players[e - 1 - k, s]) * FPS / k if k > 0 else np.zeros(2)
    return p, v


def link(players, team, halves, gap_s, vmax):
    """Chains of pieces: [[piece, piece, ...], ...], every piece in exactly one chain."""
    ps = pieces(players, team, halves)
    gap = int(round(gap_s * FPS))
    by_start = sorted(range(len(ps)), key=lambda i: ps[i][0])
    starts = np.array([ps[i][0] for i in by_start])
    half_of = {}
    for i, (a, _, _, _) in enumerate(ps):
        half_of[i] = next(h for h, (_, f, l) in enumerate(halves) if f <= a <= l)
    candidates = []
    for i, pi in enumerate(ps):
        p_end, v_end = end_state(players, pi)
        lo = np.searchsorted(starts, pi[1], side="left")
        hi = np.searchsorted(starts, pi[1] - 1 + gap, side="right")
        for j in (by_start[k] for k in range(lo, hi)):
            pj = ps[j]
            if pj[3] != pi[3] or half_of[j] != half_of[i]:
                continue
            dt = (pj[0] - (pi[1] - 1)) / FPS
            target = p_end + v_end * min(dt, VEL_CAP)
            d = float(np.hypot(*(players[pj[0], pj[2]] - target)))
            if d <= R0 + vmax * dt:
                candidates.append((d, i, j))
    candidates.sort()
    succ, pred = {}, {}
    for d, i, j in candidates:
        if i not in succ and j not in pred:
            succ[i], pred[j] = j, i
    chains = []
    for i in range(len(ps)):
        if i in pred:
            continue
        c, k = [ps[i]], i
        while k in succ:
            k = succ[k]
            c.append(ps[k])
        chains.append(c)
    return chains


def chain_track(players, c, fill_frames):
    """(start, end exclusive, positions [end - start, 2]) of a chain, holes of at most `fill_frames`
    filled by straight lines (None: every hole)."""
    a, e = c[0][0], c[-1][1]
    out = np.full((e - a, 2), np.nan, dtype=np.float32)
    for (s0, s1, slot, _) in c:
        out[s0 - a:s1 - a] = players[s0:s1, slot]
    for (_, e0, s0, _), (a1, _, s1, _) in zip(c, c[1:]):
        hole = a1 - e0
        if hole <= 0 or (fill_frames is not None and hole > fill_frames):
            continue
        p0, p1 = players[e0 - 1, s0].astype(np.float64), players[a1, s1].astype(np.float64)
        w = (np.arange(1, hole + 1) / (hole + 1))[:, None]
        out[e0 - a:a1 - a] = ((1 - w) * p0 + w * p1).astype(np.float32)
    return a, e, out


def pack(tracks, team_of, n_frames, slots_per_team):
    """Tracks (start, end, positions) of each team into its delivered number of slots, each to the slot that has been
    free longest (bench/degrade.py's rule, as the delivered tracking): two people never sit back to back in a slot
    unless every slot is busy, so speeds taken across a slot's frames don't jump from one person to another. A slot
    is added only if a team has more tracks at once than slots. (2026-11-14: the first version reused the first free
    slot, which put people back to back; found on stage 2's first result, fixed, every setting rerun.)"""
    slots = []                      # (team, [tracks])
    for side in sorted(set(team_of)):
        free_at = [-10 ** 9] * slots_per_team.get(side, 0)
        held = [[] for _ in free_at]
        for (a, e, pos) in sorted((t for t, s in zip(tracks, team_of) if s == side), key=lambda t: t[0]):
            k = int(np.argmin(free_at)) if free_at else None
            if k is None or free_at[k] > a:
                free_at.append(-10 ** 9)
                held.append([])
                k = len(free_at) - 1
            free_at[k] = e
            held[k].append((a, e, pos))
        slots += [(side, h) for h in held]
    players = np.full((n_frames, len(slots), 2), np.nan, dtype=np.float32)
    for k, (_, ts) in enumerate(slots):
        for a, e, pos in ts:
            players[a:e, k] = pos
    return players, np.array([s for s, _ in slots], dtype=np.int8)


def do_link(track, halves, gap_s, vmax, fill):
    players, team = track["players"], track["team"]
    chains = link(players, team, halves, gap_s, vmax)
    fill_frames = None if fill == "all" else int(round(float(fill) * FPS))
    tracks = [chain_track(players, c, fill_frames) for c in chains]
    sides, counts = np.unique(np.asarray(team), return_counts=True)
    new_players, new_team = pack(tracks, [c[0][3] for c in chains], len(players),
                                 {int(k): int(n) for k, n in zip(sides, counts)})
    return {**track, "players": new_players, "team": new_team}


def ball_gaps(ball, halves, max_frames):
    """(first missing, end exclusive) of every ball gap within a half, seen at both ends, <= max_frames."""
    out = []
    for _, first, last in halves:
        miss = np.isnan(ball[first:last + 1, 0])
        for a, e in zip(*v1.runs(miss)):
            if a > 0 and first + e <= last and e - a <= max_frames:
                out.append((first + int(a), first + int(e)))
    return out


def do_ball(track, halves, gap_s):
    ball = track["ball"].copy()
    for a, e in ball_gaps(ball, halves, int(round(gap_s * FPS))):
        p0, p1 = ball[a - 1].astype(np.float64), ball[e].astype(np.float64)
        w = (np.arange(1, e - a + 1) / (e - a + 1))[:, None]
        ball[a:e] = ((1 - w) * p0 + w * p1).astype(np.float32)
    return {**track, "ball": ball}


def do_smooth(track, halves, window):
    players = track["players"].copy()
    for a, e, s, _ in pieces(players, track["team"], halves):
        if e - a >= window:
            for k in (0, 1):
                players[a:e, s, k] = savgol_filter(players[a:e, s, k].astype(np.float64), window, 2).astype(np.float32)
    return {**track, "players": players}


def person_of_slots(track, clean, halves):
    """For each damaged piece, the true person whose clean position it copies: [(piece, person)]."""
    out = []
    cp = clean["players"].astype(np.float32)
    for piece in pieces(track["players"], track["team"], halves):
        a, e, s, _ = piece
        hit = np.flatnonzero(np.all(cp[a] == track["players"][a, s], axis=1))
        if len(hit) > 1:            # two people at the same point on the piece's first frame: the whole piece decides
            piece_xy = track["players"][a:e, s]
            hit = [h for h in hit if np.array_equal(cp[a:e, h], piece_xy, equal_nan=True)][:1]
        if len(hit) != 1:
            raise RuntimeError("piece at {} slot {}: {} matching people".format(a, s, len(hit)))
        out.append((piece, int(hit[0])))
    return out


def do_o_link(track, clean, halves):
    players = np.full(clean["players"].shape, np.nan, dtype=np.float32)
    for (a, e, s, _), person in person_of_slots(track, clean, halves):
        players[a:e, person] = track["players"][a:e, s]
    return {**track, "players": players, "team": np.asarray(clean["team"]).astype(np.int8)}


def do_o_linkfill(track, clean, halves):
    """DIAGNOSTIC (added 2026-11-14 ~20:30, after stage 2's first result): perfect joins, then straight lines across
    every hole inside each true person's track within a half (no truth inside the holes)."""
    out = do_o_link(track, clean, halves)
    players = out["players"].copy()
    for _, first, last in halves:
        for k in range(players.shape[1]):
            col = players[first:last + 1, k]
            seen = ~np.isnan(col[:, 0])
            if seen.sum() < 2:
                continue
            idx = np.flatnonzero(seen)
            inside = np.zeros(len(col), bool)
            inside[idx[0]:idx[-1] + 1] = True
            hole = inside & ~seen
            if hole.any():
                t = np.arange(len(col))
                for d in (0, 1):
                    col[hole, d] = np.interp(t[hole], idx, col[seen, d].astype(np.float64)).astype(np.float32)
            players[first:last + 1, k] = col
    return {**out, "players": players}


TEAM_SHIFT_FROM = 3.0     # s: holes longer than this follow their team's block (o_linkteam; fixed before it ran)


def do_o_linkteam(track, clean, halves):
    """DIAGNOSTIC (added 2026-11-14 ~20:45, after o_linkfill): perfect joins; holes of at most TEAM_SHIFT_FROM s get a
    straight line, longer ones the straight line plus how far the visible teammates' centre moved off its own line
    over the hole (players move with their team's block)."""
    out = do_o_link(track, clean, halves)
    joined = out["players"].astype(np.float64)
    players = out["players"].copy()
    team = out["team"]
    for _, first, last in halves:
        P = joined[first:last + 1]
        for k in range(P.shape[1]):
            col = P[:, k]
            seen = ~np.isnan(col[:, 0])
            if seen.sum() < 2:
                continue
            idx = np.flatnonzero(seen)
            gaps = np.flatnonzero(np.diff(idx) > 1)
            if not len(gaps):
                continue
            mates = np.flatnonzero((team == team[k]) & (np.arange(len(team)) != k))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                centre = np.nanmean(P[:, mates], axis=1)
            for g in gaps:
                a, b = idx[g], idx[g + 1]
                n = b - a - 1
                t = (np.arange(1, n + 1) / (n + 1))[:, None]
                fill = (1 - t) * col[a] + t * col[b]
                if n > TEAM_SHIFT_FROM * FPS:
                    shift = centre[a + 1:b] - ((1 - t) * centre[a] + t * centre[b])
                    fill = fill + np.where(np.isnan(shift), 0.0, shift)
                players[first + a + 1:first + b, k] = fill.astype(np.float32)
    return {**out, "players": players}


NEAR_BALL = 15.0          # m: o_near's radius around the true ball (fixed before it ran)


def do_o_near(track, clean, halves):
    """DIAGNOSTIC (added 2026-11-14 ~21:05, for the next phase): perfect joins (o_link), and a missing player's true
    position only where he is within NEAR_BALL of the true ball (roughly what a camera following the ball sees)."""
    out = do_o_link(track, clean, halves)
    players = out["players"].copy()
    cp = clean["players"].astype(np.float32)
    ball = clean["ball"][:, :2].astype(np.float64)
    with np.errstate(invalid="ignore"):
        near = np.hypot(cp[:, :, 0] - ball[:, None, 0], cp[:, :, 1] - ball[:, None, 1]) <= NEAR_BALL
    fill = np.isnan(players[:, :, 0]) & ~np.isnan(cp[:, :, 0]) & near
    players[fill] = cp[fill]
    return {**out, "players": players}


def do_o_players(track, clean):
    return {**track, "players": clean["players"].astype(np.float32), "team": np.asarray(clean["team"]).astype(np.int8)}


def do_o_ball(track, clean, halves, gap_s):
    ball = track["ball"].copy()
    for a, e in ball_gaps(ball, halves, int(round(gap_s * FPS))):
        ball[a:e] = clean["ball"][a:e, :2]
    return {**track, "ball": ball.astype(np.float32)}


STEP = re.compile(r"^(c0|o_players|o_linkfill|o_linkteam|o_link|o_near|o_ball(?P<ob>[\d.]+)|link_g(?P<g>[\d.]+)_v(?P<v>[\d.]+)(_fill(?P<f>[\d.]+|all))?"
                  r"|ball(?P<b>[\d.]+)|smooth(?P<w>\d+))$")


def steps(variant):
    out = []
    for part in variant.split("+"):
        m = STEP.match(part)
        if not m:
            raise ValueError("unknown step {!r}".format(part))
        out.append((part, m))
    return out


def clean_track(variant, match):
    """The match's tracking after the variant's steps (the input dict is never modified)."""
    track = dict(match.track)
    clean = None
    for part, m in steps(variant):
        if part.startswith("o_"):
            clean = clean or sets.load(match.id)
        if part == "c0":
            continue
        if part == "o_players":
            track = do_o_players(track, clean)
        elif part == "o_near":
            track = do_o_near(track, clean, match.halves)
        elif part == "o_linkteam":
            track = do_o_linkteam(track, clean, match.halves)
        elif part == "o_linkfill":
            track = do_o_linkfill(track, clean, match.halves)
        elif part == "o_link":
            track = do_o_link(track, clean, match.halves)
        elif m.group("ob"):
            track = do_o_ball(track, clean, match.halves, float(m.group("ob")))
        elif m.group("g"):
            track = do_link(track, match.halves, float(m.group("g")), float(m.group("v")), m.group("f") or 0)
        elif m.group("b"):
            track = do_ball(track, match.halves, float(m.group("b")))
        elif m.group("w"):
            track = do_smooth(track, match.halves, int(m.group("w")))
    return track


def cleaner(variant, halves):
    """A variant as a function of a tracking dict (for possession/v12.inputs's `clean`): no ceilings, they need
    the truth."""
    if any(part.startswith("o_") for part, _ in steps(variant)):
        raise ValueError("{}: a ceiling can't be deployed".format(variant))

    def clean(tracking):
        return clean_track(variant, types.SimpleNamespace(id=None, track=tracking, halves=halves))
    return clean


# --------------------------------------------------------------------------------------------
# Stage 3: a lab cache whose tracking is cleaned, for an isolated copy of the repo
# --------------------------------------------------------------------------------------------

def build_lab(variant, dest):
    """bench/lab.py's cache for every match of `all`, with the variant's players (and ball) as the tracking, PathCRF's
    P and v7's cost recomputed on them, and the delivered players kept as players_raw / team_raw (v1's clues on the
    stressed copies read those: bench/lab_carried.py ball_costs). Written to `dest` (another copy's data/external/lab)."""
    os.makedirs(dest, exist_ok=True)
    for m in lab.matches(WHICH):
        with np.load(os.path.join(lab.folder(WHICH), "{}.npz".format(m))) as z:
            f = {k: z[k] for k in z.files}
        match = lab.load(m, WHICH)
        cm, _ = cleaned_match(variant, match)
        f.update(players=cm.track["players"], team=cm.track["team"], ball=cm.track["ball"],
                 players_raw=match.track["players"], team_raw=match.track["team"],
                 pred_p_pathcrf=cm.pred["p_pathcrf"], pred_v7_cost=cm.pred["v7_cost"])
        np.savez(os.path.join(dest, "{}.npz".format(m)), **f)
        print("{} {}: cleaned lab cache written".format(variant, m), flush=True)


# --------------------------------------------------------------------------------------------
# The screen
# --------------------------------------------------------------------------------------------

def folder(variant):
    return os.path.join(CACHE, variant)


_PATHCRF = None


def pathcrf_model():
    global _PATHCRF
    if _PATHCRF is None:
        _PATHCRF = pathcrf_adapter.load_model()          # float32, as bench/eval_pathcrf.py ran it
    return _PATHCRF


def cleaned_match(variant, match):
    """(the match as the cleaned inputs' consumers see it, P1's counts)."""
    track = clean_track(variant, match)
    pred = dict(match.pred)
    live = np.isin(match.pred["v6"], ["H", "A"])
    if track["players"] is not match.track["players"]:
        pc = chain.pathcrf(WHICH)
        p = pathcrf_adapter.p_home(pathcrf_model(), track, match.halves, match.pitch, live)
        pred["p_pathcrf"] = p
        missing = np.isnan(track["ball"][:, 0])
        pred["v7_cost"] = v7_cost(match.pred["v6_cost"].astype(np.float64), p, missing, pc["weight"], pc["where"]).astype(np.float32)
    p = pred["p_pathcrf"]
    truth = match.truth
    gone = np.isnan(match.track["ball"][:, 0])
    where = np.isin(truth, (0, 1)) & live & ~np.isnan(p)
    p1 = {}
    for zone, mask in (("missing", where & gone), ("seen", where & ~gone)):
        p1[zone] = [int(((p[mask] > 0.5) == (truth[mask] == 0)).sum()), int(mask.sum())]
    fields = {k: v for k, v in vars(match).items() if not k.startswith("_")}
    fields.update(track=track, pred=pred)
    return types.SimpleNamespace(**fields), p1


def features_of(variant, match):
    """Robust's features on the cleaned, blanked tracking: (names, X, X_swapped, P1)."""
    cm, p1 = cleaned_match(variant, match)
    b = lab_robust.blanked(cm)
    touches = v11.seen_touches(b.track, v11.SETTINGS[WHICH]["dv"])
    passes = v9.seen_passes(b.track)
    names, X = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes)
    _, Xs = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes, swap=True)
    return names, X, Xs, p1


def feature_path(variant, m):
    return os.path.join(folder(variant), "{}.npz".format(m))


def build_features(variant, log=print):
    os.makedirs(folder(variant), exist_ok=True)
    p1 = {}
    for m in lab.matches(WHICH):
        path = feature_path(variant, m)
        if os.path.exists(path):
            with np.load(path) as z:
                p1[m] = json.loads(str(z["p1"]))
            continue
        t0 = time.time()
        match = lab.load(m, WHICH)
        names, X, Xs, p1[m] = features_of(variant, match)
        np.savez(path, names=np.array(names), X=X, X_swapped=Xs, truth=match.truth.astype(np.int8),
                 p1=json.dumps(p1[m]))
        log("{} {}: features in {:.0f} s; PathCRF right, ball missing {}/{}".format(
            variant, m, time.time() - t0, *p1[m]["missing"]))
    return p1


def load_features(variant, m):
    with np.load(feature_path(variant, m)) as z:
        return [str(k) for k in z["names"]], z["X"], z["X_swapped"], z["truth"]


def fit(variant, used):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs, ys = [], []
    for m in used:
        _, X, Xw, truth = load_features(variant, m)
        rows = np.flatnonzero(np.isin(truth, (0, 1, 2)))[::L.EVERY]
        y = truth[rows].astype(np.int64)
        Xs += [X[rows], Xw[rows]]
        ys += [y, L.SWAP[y]]
    return HistGradientBoostingClassifier(**L.TREES).fit(np.concatenate(Xs), np.concatenate(ys))


def proba(clf, variant, m):
    from threadpoolctl import threadpool_limits
    _, X, Xw, _ = load_features(variant, m)
    with threadpool_limits(2):
        p = clf.predict_proba(X)
        p = (p + clf.predict_proba(Xw)[:, L.SWAP]) / 2
    return np.round(np.clip(p, *L.CLIP), L.DECIMALS)


def held_list():
    ms = lab.matches(WHICH)
    return [m for m in ms if sets.source(m) in L.TRAIN_SOURCES] + [m for m in ms if sets.source(m) not in L.TRAIN_SOURCES][:1]


def p_path(variant, m):
    return os.path.join(folder(variant), "p_{}.npy".format(m))


def screen(variant, keep=False, log=print):
    """P1 and P2 for one variant: bench/results/lab_clean_<variant>.json."""
    t0 = time.time()
    p1 = build_features(variant, log)
    ms = lab.matches(WHICH)
    trained = [m for m in ms if sets.source(m) in L.TRAIN_SOURCES]
    for held in held_list():
        targets = [held] if held in trained else [m for m in ms if m not in trained]
        if all(os.path.exists(p_path(variant, m)) for m in targets):
            continue
        clf = fit(variant, [m for m in trained if m != held])
        for m in targets:
            np.save(p_path(variant, m), proba(clf, variant, m).astype(np.float32))
        log("{}: fold {} fitted".format(variant, held))
    out = {}
    for m in ms:
        match = lab.load(m, WHICH)
        p = np.load(p_path(variant, m)).astype(np.float64)
        graded = np.isin(match.truth, (0, 1, 2))
        y = match.truth[graded].astype(np.int64)
        logloss = float(-np.log(p[graded][np.arange(len(y)), y]).mean())
        letters = L.decode_cost(match, lab_robust.cost_from(match, p, CHOSEN["blend"]), **CHOSEN)
        letters = lab_robust.finish(match, letters, WHICH)
        s = lab.score(match, letters)
        out[m] = {"source": match.source, "unseen": m in UNSEEN, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                  "logloss": logloss, "p1": p1[m]}
    with open(os.path.join(RESULTS, "lab_clean_{}.json".format(variant)), "w") as handle:
        json.dump({"variant": variant, "matches": out, "minutes": (time.time() - t0) / 60}, handle, indent=1)
    if not keep:
        for m in ms:
            os.remove(feature_path(variant, m))
    log("{}: spellF1 {:.4f}, log loss {:.4f} ({:.0f} min)".format(
        variant, np.mean([v["spell_f1"] for v in out.values()]), np.mean([v["logloss"] for v in out.values()]),
        (time.time() - t0) / 60))
    return out


def check():
    """c0's held-out P against robust's (data/external/lab_robust/p_blank_<m>.npy): the harness must
    give the same numbers on the uncleaned tracking."""
    worst = 0.0
    for m in lab.matches(WHICH):
        mine = np.load(p_path("c0", m)).astype(np.float64)
        theirs = lab_robust.held_out_proba(m, WHICH)
        d = float(np.abs(mine - theirs).max())
        worst = max(worst, d)
        print("{}: largest difference in P {:.2e}".format(m, d))
    print("harness {}".format("REPRODUCES robust" if worst < 1e-6 else "DIFFERS from robust ({:.2e})".format(worst)))


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------

def load_result(variant):
    path = os.path.join(RESULTS, "lab_clean_{}.json".format(variant))
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)["matches"]


def summary(variant, base):
    r = load_result(variant)
    if r is None:
        return None
    ms = sorted(r)
    d = np.array([r[m]["spell_f1"] - base[m]["spell_f1"] for m in ms])
    met = np.array([sets.source(m) == "metrica" for m in ms])

    def acc(group, zone):
        right = sum(r[m]["p1"][zone][0] for m in group)
        n = sum(r[m]["p1"][zone][1] for m in group)
        return right / n if n else float("nan")

    unseen = [m for m in ms if m in UNSEEN]
    return {"variant": variant, "spell_f1": float(np.mean([r[m]["spell_f1"] for m in ms])), "gain": float(d.mean()),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "idsse": float(d[~met].mean()),
            "metrica": float(d[met].mean()),
            "logloss": float(np.mean([r[m]["logloss"] - base[m]["logloss"] for m in ms])),
            "p1_missing_unseen": acc(unseen, "missing"), "p1_seen_unseen": acc(unseen, "seen"),
            "p1_missing_all": acc(ms, "missing"), "per_match": {m: float(x) for m, x in zip(ms, d)}}


def report(variants=None):
    base = load_result("c0")
    if base is None:
        raise RuntimeError("screen c0 first")
    names = variants or sorted(f[len("lab_clean_"):-len(".json")] for f in os.listdir(RESULTS)
                               if f.startswith("lab_clean_") and f.endswith(".json") and f != "lab_clean_report.json"
                               and not f.startswith("lab_clean_stage2"))
    rows = [s for s in (summary(v, base) for v in names) if s]
    b = summary("c0", base)
    print("variant | P2 spellF1 | gain | better/worse | IDSSE | Metrica | log loss | P1 missing (unseen) | P1 seen (unseen) | P1 missing (all 10)")
    for s in sorted(rows, key=lambda s: -s["gain"]):
        print("{:<28s} | {:.4f} | {:+.4f} | {}/{} | {:+.4f} | {:+.4f} | {:+.4f} | {:.1%} ({:+.1f}) | {:.1%} | {:.1%} ({:+.1f})".format(
            s["variant"], s["spell_f1"], s["gain"], s["better"], s["worse"], s["idsse"], s["metrica"], s["logloss"],
            s["p1_missing_unseen"], 100 * (s["p1_missing_unseen"] - b["p1_missing_unseen"]), s["p1_seen_unseen"],
            s["p1_missing_all"], 100 * (s["p1_missing_all"] - b["p1_missing_all"])))
    with open(os.path.join(RESULTS, "lab_clean_report.json"), "w") as handle:
        json.dump(rows, handle, indent=1)
    return rows


STAGE2 = {"gain": 0.003, "better": 6}     # docs/lab/cleaning.md: held-out gain, matches better of 10; Metrica not down


def link_order(v):
    """Simplest first (ties go to it): shortest gap, then the stricter speed, then the least filling."""
    m = STEP.match(v)
    fill = m.group("f")
    return (float(m.group("g")), float(m.group("v")), -1 if fill is None else (1e9 if fill == "all" else float(fill)))


def stage2(family="link"):
    """Leave one match out over a family's settings (each match scored with the setting best on the other 9, ties to the
    simplest), against c0; the stage-2 gate."""
    base = load_result("c0")
    names = [f[len("lab_clean_"):-len(".json")] for f in os.listdir(RESULTS)
             if f.startswith("lab_clean_{}_".format(family)) and f.endswith(".json")]
    names = sorted((v for v in names if "+" not in v), key=link_order)
    results = {v: load_result(v) for v in names}
    ms = sorted(base)
    folds = []
    for m in ms:
        others = [o for o in ms if o != m]
        means = [np.mean([results[v][o]["spell_f1"] for o in others]) for v in names]
        pick = names[int(np.argmax(np.round(means, 6)))]
        folds.append({"match": m, "picked": pick, "held_out": results[pick][m]["spell_f1"], "base": base[m]["spell_f1"]})
    d = np.array([f["held_out"] - f["base"] for f in folds])
    met = np.array([sets.source(f["match"]) == "metrica" for f in folds])
    overall = [np.mean([results[v][m]["spell_f1"] for m in ms]) for v in names]
    out = {"settings": len(names), "folds": folds, "gain": float(d.mean()), "better": int((d > 0).sum()),
           "metrica": float(d[met].mean()), "idsse": float(d[~met].mean()),
           "chosen_on_all": names[int(np.argmax(np.round(overall, 6)))]}
    out["passes"] = bool(out["gain"] >= STAGE2["gain"] and out["better"] >= STAGE2["better"] and out["metrica"] >= 0)
    for f in folds:
        print("{}: picked {} -> {:+.4f}".format(f["match"], f["picked"], f["held_out"] - f["base"]))
    print("{} settings; held out {:+.4f}, better on {}/10, IDSSE {:+.4f}, Metrica {:+.4f}; chosen on all 10: {}; "
          "stage-2 gate {}".format(len(names), out["gain"], out["better"], out["idsse"], out["metrica"],
                                   out["chosen_on_all"], "PASSED" if out["passes"] else "not passed"))
    with open(os.path.join(RESULTS, "lab_clean_stage2_{}.json".format(family)), "w") as handle:
        json.dump(out, handle, indent=1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=("screen", "check", "report", "drop", "build-lab", "stage2"))
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--keep", action="store_true", help="keep the variant's feature files")
    parser.add_argument("--to", help="build-lab: the other copy's data/external/lab")
    args = parser.parse_args()
    if args.what == "screen":
        for v in args.variants:
            steps(v)
        for v in args.variants:
            screen(v, keep=args.keep, log=lambda s: print(s, flush=True))
    elif args.what == "check":
        check()
    elif args.what == "report":
        report(args.variants or None)
    elif args.what == "build-lab":
        build_lab(args.variants[0], args.to)
    elif args.what == "stage2":
        stage2()
    elif args.what == "drop":
        for v in args.variants:
            shutil.rmtree(folder(v), ignore_errors=True)


if __name__ == "__main__":
    main()
