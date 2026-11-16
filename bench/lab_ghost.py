"""Night builder "ghost" (docs/lab/night/ghost.md): inputs that point to players missing near the ball.

    python -m bench.lab_ghost unit                                   # the unit check on J03WR9 (swap; rates vs the probe)
    OMP_NUM_THREADS=2 python -m bench.lab_ghost screen VARIANT ...   # P2, round 9's screen with its own paths
    python -m bench.lab_ghost check                                  # g0's held-out P against round 9's c0 (harness)
    python -m bench.lab_ghost report                                 # every screened variant against c0
    python -m bench.lab_ghost gate                                   # ghost_a vs ghost_b leave one match out; the gate

VARIANTS (fixed before any score):
  g0        robust's 97 columns alone (the harness check: must give round 9's c0, 0.6747)
  ghost_a   robust's 97 + 22 ghost columns (ghosts and ghost touches)
  ghost_b   ghost_a + miss_home, miss_away (24)

A pending ghost of team X at frame t: the last point of a piece of X that ended in the last G s (unless a piece of X
started since within reach of that point), or the first point of a piece of X that starts in the next G s (unless a
piece of X ends before it within reach); only while X has fewer players seen than E_X (its 99th-percentile count in
the half, at most 11). Reach: 1 m + 7 m/s x the gap (round 9's link_g10_v7). Everything from the damaged, blanked
tracking robust's features read; never the clean tracking (the unit check reads it only to count).

The screen writes only its own files: data/external/lab_ghost/ and bench/results/lab_ghost_*.json. It reads robust's
cached features (data/external/lab_robust/<m>.npz) and round 9's c0 results, never writes them.

Nothing here reads our match.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np

from bench import lab, lab_robust, sets
from bench.eval_pathcrf import UNSEEN
from possession import features as pfeatures
from possession import lab_learned_features as llf
from possession import touches as ptouches
from possession import v1, v11
from possession.data import AWAY, FPS, HOME

# Params (fixed before any score; docs/lab/night/ghost.md)
L = lab_robust.L
WHICH = "all"
CHOSEN = lab_robust.CHOSEN
G_S = 10.0                 # s: how far back / ahead a piece's end / start counts as a ghost
R0, VMAX = 1.0, 7.0        # m, m/s: "he reappeared nearby" (round 9's link_g10_v7 reach)
FAR = llf.FAR              # m: no ghost
GAP_CAP = 10.0             # s: a ghost's time gap, capped (and the value when there is none)
N_RADIUS, N_CAP = 10.0, 3  # m, count: ghosts near the ball
GHOST_TOUCH = 5.0          # m: an orphan ball event goes to a team whose nearest ghost is this close (and nearer)
DV = v11.SETTINGS[WHICH]["dv"]       # m/s: v11's velocity change on `all` (3.0)
REACH = ptouches.REACH               # m: v11's "a player is on the ball" (2.0)
TOUCH_CAP = llf.CAP                  # s: since / until, as robust's touch features
TOUCH_WINDOW = 3                     # s: counts over the past / next, as robust's touch features
MISS_CAP = 3
PCTL = 99
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_ghost")      # gitignored (data/*)
RESULTS = lab.RESULTS
BASE_RESULT = os.path.join(RESULTS, "lab_clean_c0.json")            # round 9's c0 (read only)
VARIANTS = {"g0": 0, "ghost_a": 22, "ghost_b": 24,                 # how many ghost columns, in NAMES' order
            "ghost_den": 22,                                         # idea 2: ghost_a + 6 den_* columns
            "o_near_seen": 0, "o_near_nopc": 0}                      # diagnostics (read the truth; never candidates)
DIAG = ("o_near_seen", "o_near_nopc")
# idea 2: the clean-tracking values (round 9's o_players features) the regressors predict; (column, minus column)
DEN = (("d_diff", None), ("d_diff_last_seen", None), ("d_diff_next_seen", None), ("n_home_2m", "n_away_2m"),
       ("d_home", None), ("d_away", None))
DEN_NAMES = ["den_d_diff", "den_d_diff_last_seen", "den_d_diff_next_seen", "den_n2_diff", "den_d_home", "den_d_away"]
REG = {k: v for k, v in L.TREES.items()}                                # robust's tree settings, squared error
GATE = {"gain": 0.003, "better": 6, "logloss": 0.001}              # held-out; Metrica's 3 not down on the mean
WHERES = ("", "_last_seen", "_next_seen")


def names():
    out = []
    for where in WHERES:
        out += ["gd_home" + where, "gd_away" + where, "gt_home" + where, "gt_away" + where]
    out += ["gn_home_10m", "gn_away_10m"]
    for side in ("home", "away"):
        out += ["since_gtouch_" + side, "until_gtouch_" + side, "gtouches_{}_past_{}s".format(side, TOUCH_WINDOW),
                "gtouches_{}_next_{}s".format(side, TOUCH_WINDOW)]
    out += ["miss_home", "miss_away"]
    return out


NAMES = names()
assert len(NAMES) == 24


# --------------------------------------------------------------------------------------------
# The ghost columns
# --------------------------------------------------------------------------------------------

def pieces(players, team, halves):
    """Every unbroken run of a slot within a half: (start, end exclusive, slot, team, half index) (as lab_clean.pieces)."""
    out = []
    for h, (_, first, last) in enumerate(halves):
        on = ~np.isnan(players[first:last + 1, :, 0])
        for s in range(players.shape[1]):
            for a, e in zip(*v1.runs(on[:, s])):
                out.append((first + int(a), first + int(e), s, int(team[s]), h))
    return out


def seen_counts(players, team, side):
    return (~np.isnan(players[:, team == side, 0])).sum(axis=1)


def expected(players, team, halves, side):
    """int [frames]: E_X, min(11, the 99th percentile of the side's seen count over each half); 0 outside the halves."""
    count = seen_counts(players, team, side)
    out = np.zeros(len(players), np.int64)
    for _, first, last in halves:
        out[first:last + 1] = min(11, int(np.percentile(count[first:last + 1], PCTL, method="nearest")))
    return out


def ghost_pairs(players, team, halves):
    """Every (frame, x, y, side, gap s) where a ghost of `side` is pending (the E_X gate applied)."""
    g = int(round(G_S * FPS))
    ps = pieces(players, team, halves)
    out = []
    count = {side: seen_counts(players, team, side) for side in (HOME, AWAY)}
    need = {side: expected(players, team, halves, side) for side in (HOME, AWAY)}
    for h, (_, first, last) in enumerate(halves):
        for side in (HOME, AWAY):
            mine = [p for p in ps if p[3] == side and p[4] == h]
            if not mine:
                continue
            a_ = np.array([p[0] for p in mine])
            e_ = np.array([p[1] - 1 for p in mine])                      # last frame
            pa = np.array([players[p[0], p[2]] for p in mine], np.float64)
            pe = np.array([players[p[1] - 1, p[2]] for p in mine], np.float64)
            frames, xs, ys, gaps = [], [], [], []
            for i in range(len(mine)):
                # end ghost: pending on (e, min(e + g, first reappearance within reach - 1)]
                e, p = e_[i], pe[i]
                cand = (a_ > e) & (a_ <= e + g)
                stop = min(e + g, last)
                if cand.any():
                    dt = (a_[cand] - e) / FPS
                    ok = np.hypot(*(pa[cand] - p).T) <= R0 + VMAX * dt
                    if ok.any():
                        stop = min(stop, int(a_[cand][ok].min()) - 1)
                if stop > e:
                    t = np.arange(e + 1, stop + 1)
                    frames.append(t)
                    xs.append(np.full(len(t), p[0]))
                    ys.append(np.full(len(t), p[1]))
                    gaps.append((t - e) / FPS)
                # start ghost: pending on [max(a - g, last end within reach + 1), a)
                a, p = a_[i], pa[i]
                cand = (e_ < a) & (e_ >= a - g)
                begin = max(a - g, first)
                if cand.any():
                    dt = (a - e_[cand]) / FPS
                    ok = np.hypot(*(pe[cand] - p).T) <= R0 + VMAX * dt
                    if ok.any():
                        begin = max(begin, int(e_[cand][ok].max()) + 1)
                if begin < a:
                    t = np.arange(begin, a)
                    frames.append(t)
                    xs.append(np.full(len(t), p[0]))
                    ys.append(np.full(len(t), p[1]))
                    gaps.append((a - t) / FPS)
            if frames:
                t = np.concatenate(frames)
                keep = count[side][t] < need[side][t]
                out.append((t[keep], np.concatenate(xs)[keep], np.concatenate(ys)[keep],
                            np.full(int(keep.sum()), side), np.concatenate(gaps)[keep]))
    if not out:
        z = np.zeros(0)
        return z.astype(np.int64), z, z, z.astype(np.int64), z
    return tuple(np.concatenate([o[k] for o in out]) for k in range(5))


def nearest_ghost(pairs, side, ball, n):
    """(distance to the side's nearest pending ghost (inf if none), its gap s (nan if none), ghosts within N_RADIUS)
    per frame, for a ball array (NaN rows give inf / nan / 0)."""
    t, x, y, s, gap = pairs
    k = s == side
    t, x, y, gap = t[k], x[k], y[k], gap[k]
    d = np.hypot(x - ball[t, 0], y - ball[t, 1])
    d = np.where(np.isnan(d), np.inf, d)
    best = np.full(n, np.inf)
    best_gap = np.full(n, np.nan)
    if len(t):
        order = np.lexsort((gap, d, t))          # by frame, then distance, then the shorter gap
        t_o = t[order]
        first = np.r_[True, t_o[1:] != t_o[:-1]]
        best[t_o[first]] = d[order][first]
        best_gap[t_o[first]] = gap[order][first]
    near = np.bincount(t[d < N_RADIUS], minlength=n)[:n].astype(np.float64)
    return best, best_gap, near


def orphan_events(players, team, ball, halves):
    """int [events]: the frames of ball events (a run of v11's velocity change >= DV) that no visible player of either
    team is within REACH of on any frame of the run; the run's largest change."""
    change = ptouches.velocity_change(ball)
    near = np.minimum(ptouches.nearest(players, team, ball, HOME), ptouches.nearest(players, team, ball, AWAY)) < REACH
    out = []
    for _, first, last in halves:
        with np.errstate(invalid="ignore"):
            ev = change[first:last + 1] >= DV
        for a, e in zip(*v1.runs(ev)):
            if not near[first + a:first + e].any():
                out.append(first + a + int(np.nanargmax(change[first + a:first + e])))
    return np.array(out, np.int64)


def ghost_columns(track, halves, swap=False, pairs=None):
    """(NAMES, X float32 [frames, 24]) from the blanked tracking; NaN outside the halves. swap=True: the same with the
    teams exchanged (home's columns hold away's), as llf.frame_features."""
    players, ball, team = track["players"], track["ball"], np.asarray(track["team"])
    n = len(ball)
    if swap:
        team = np.where(team == HOME, AWAY, np.where(team == AWAY, HOME, team))
    if pairs is None or swap:
        pairs = ghost_pairs(players, team, halves)
    f = {}
    seen_ball = None
    for where, b in (("", ball), ("_last_seen", llf.filled(ball, halves)), ("_next_seen", llf.filled(ball, halves, True))):
        b = b.astype(np.float64)
        b_seen = ~np.isnan(b[:, 0])
        for side, word in ((HOME, "home"), (AWAY, "away")):
            d, gap, near = nearest_ghost(pairs, side, b, n)
            gd = np.minimum(d, FAR)
            gt = np.where(np.isnan(gap), GAP_CAP, np.minimum(gap, GAP_CAP))
            gd[~b_seen], gt[~b_seen] = np.nan, np.nan
            f["gd_" + word + where], f["gt_" + word + where] = gd, gt
            if where == "":
                cnt = np.minimum(near, N_CAP)
                cnt[~b_seen] = np.nan
                f["gn_{}_10m".format(word)] = cnt
                if seen_ball is None:
                    seen_ball = {}
                seen_ball[side] = d
    # ghost touches: orphan events given to the side whose nearest ghost is within GHOST_TOUCH and nearer
    ev = orphan_events(players, team, ball, halves)
    w = TOUCH_WINDOW * FPS
    for side, word in ((HOME, "home"), (AWAY, "away")):
        other = AWAY if side == HOME else HOME
        hit = np.zeros(n, bool)
        if len(ev):
            mine, theirs = seen_ball[side][ev], seen_ball[other][ev]
            hit[ev[(mine <= GHOST_TOUCH) & (mine < theirs)]] = True
        f["since_gtouch_" + word] = llf.since(hit, halves, TOUCH_CAP)
        f["until_gtouch_" + word] = llf.until(hit, halves, TOUCH_CAP)
        f["gtouches_{}_past_{}s".format(word, TOUCH_WINDOW)] = pfeatures.window_mean(hit.astype(np.float32), w, halves, False) * w
        f["gtouches_{}_next_{}s".format(word, TOUCH_WINDOW)] = pfeatures.window_mean(hit.astype(np.float32), w, halves, True) * w
    for side, word in ((HOME, "home"), (AWAY, "away")):
        f["miss_" + word] = np.clip(expected(players, team, halves, side) - seen_counts(players, team, side),
                                    0, MISS_CAP).astype(np.float32)
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in NAMES], axis=1)
    outside = np.ones(n, bool)
    for _, first, last in halves:
        outside[first:last + 1] = False
    X[outside] = np.nan
    return list(NAMES), X


# --------------------------------------------------------------------------------------------
# Caches (all ours)
# --------------------------------------------------------------------------------------------

def cols_path(m):
    return os.path.join(CACHE, "cols", "{}.npz".format(m))


def build_cols(m, log=print):
    path = cols_path(m)
    if not os.path.exists(path):
        t0 = time.time()
        b = lab_robust.blanked(lab.load(m, WHICH))
        names_, X = ghost_columns(b.track, b.halves)
        _, Xs = ghost_columns(b.track, b.halves, swap=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, names=np.array(names_), X=X, X_swapped=Xs)
        log("{}: ghost columns in {:.0f} s".format(m, time.time() - t0))
    with np.load(path) as z:
        return z["X"], z["X_swapped"]


def load_base(m):
    """Robust's cached 97 (read only; raises if absent rather than writing robust's cache)."""
    path = os.path.join(lab_robust.folder(WHICH), "{}.npz".format(m))
    if not os.path.exists(path):
        raise RuntimeError("robust's cache is missing {}".format(path))
    with np.load(path) as z:
        return [str(k) for k in z["names"]], z["X"], z["X_swapped"], z["truth"]


def load_features(variant, m):
    if variant in DIAG:
        return diag_features(variant, m)
    names_, X, Xs, truth = load_base(m)
    k = VARIANTS[variant]
    if k:
        G, Gs = build_cols(m)
        X, Xs = np.hstack([X, G[:, :k]]), np.hstack([Xs, Gs[:, :k]])
        names_ = names_ + NAMES[:k]
    if variant == "ghost_den":
        with np.load(den_path(m)) as z:
            X, Xs = np.hstack([X, z["X"]]), np.hstack([Xs, z["X_swapped"]])
        names_ = names_ + DEN_NAMES
    return names_, X, Xs, truth


# --------------------------------------------------------------------------------------------
# The diagnostics (read the clean tracking; never candidates): o_near with one change each
# --------------------------------------------------------------------------------------------

def diag_track(variant, match, clean):
    """o_near (lab_clean.do_o_near: perfect joins, a missing player's true position within 15 m of the true ball);
    o_near_seen fills only where robust's blanked ball is seen. (track, frames filled, frames o_near would fill)."""
    from bench import lab_clean
    out = lab_clean.do_o_link(dict(match.track), clean, match.halves)
    players = out["players"].copy()
    cp = clean["players"].astype(np.float32)
    ball = clean["ball"][:, :2].astype(np.float64)
    with np.errstate(invalid="ignore"):
        near = np.hypot(cp[:, :, 0] - ball[:, None, 0], cp[:, :, 1] - ball[:, None, 1]) <= lab_clean.NEAR_BALL
    fill = np.isnan(players[:, :, 0]) & ~np.isnan(cp[:, :, 0]) & near
    full = int(fill.sum())
    if variant == "o_near_seen":
        seen = ~np.isnan(lab_robust.blanked(match).track["ball"][:, 0])
        fill &= seen[:, None]
    players[fill] = cp[fill]
    return {**out, "players": players}, int(fill.sum()), full


def diag_features(variant, m, log=print):
    path = os.path.join(folder(variant), "{}.npz".format(m))
    if not os.path.exists(path):
        import types
        from bench import chain, lab_clean
        from bench.eval_pathcrf import v7_cost
        from possession import pathcrf_adapter, v9
        t0 = time.time()
        match = lab.load(m, WHICH)
        track, filled_, full = diag_track(variant, match, sets.load(m))
        pred = dict(match.pred)
        if variant != "o_near_nopc":              # as lab_clean.cleaned_match: PathCRF and v7's cost on the filled players
            live = np.isin(match.pred["v6"], ["H", "A"])
            pc = chain.pathcrf(WHICH)
            p = pathcrf_adapter.p_home(lab_clean.pathcrf_model(), track, match.halves, match.pitch, live)
            pred["p_pathcrf"] = p
            pred["v7_cost"] = v7_cost(match.pred["v6_cost"].astype(np.float64), p, np.isnan(track["ball"][:, 0]),
                                      pc["weight"], pc["where"]).astype(np.float32)
        fields = {k: v for k, v in vars(match).items() if not k.startswith("_")}
        fields.update(track=track, pred=pred)
        b = lab_robust.blanked(types.SimpleNamespace(**fields))
        found = v11.seen_touches(b.track, v11.SETTINGS[WHICH]["dv"])
        passes = v9.seen_passes(b.track)
        names_, X = llf.frame_features(b.track, b.halves, b.pitch, b.pred, found, passes)
        _, Xs = llf.frame_features(b.track, b.halves, b.pitch, b.pred, found, passes, swap=True)
        os.makedirs(folder(variant), exist_ok=True)
        np.savez(path, names=np.array(names_), X=X, X_swapped=Xs, truth=match.truth.astype(np.int8))
        log("{} {}: features in {:.0f} s; player-frames filled {} of o_near's {}".format(
            variant, m, time.time() - t0, filled_, full))
    with np.load(path) as z:
        return [str(k) for k in z["names"]], z["X"], z["X_swapped"], z["truth"]


# --------------------------------------------------------------------------------------------
# Idea 2: the near-ball "denoiser" (ghost_den)
# --------------------------------------------------------------------------------------------

def den_path(m):
    return os.path.join(CACHE, "den", "{}.npz".format(m))


def den_targets(m):
    """[frames, 6] targets from round 9's o_players features (read only), both orientations."""
    with np.load(os.path.join(lab.ROOT, "data", "external", "lab_clean", "o_players", "{}.npz".format(m))) as z:
        names_ = [str(k) for k in z["names"]]
        out = []
        for key in ("X", "X_swapped"):
            X = z[key]
            cols = []
            for a, b in DEN:
                v = X[:, names_.index(a)].astype(np.float64)
                if b is not None:
                    v = v - X[:, names_.index(b)]
                cols.append(v)
            out.append(np.stack(cols, axis=1))
    return out


def den_mask(names_, X):
    """bool [frames, 6]: where the damaged column each den_* shadows is NaN (its prediction is NaN there)."""
    return np.stack([np.isnan(X[:, names_.index(a)]) for a, _ in DEN], axis=1)


def den_fit(used):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xs, Ts = [], []
    for m in used:
        _, X, Xw, truth = load_features("ghost_a", m)
        T, Tw = den_targets(m)
        rows = np.flatnonzero(np.isin(truth, (0, 1, 2)))[::L.EVERY]
        Xs += [X[rows], Xw[rows]]
        Ts += [T[rows], Tw[rows]]
        del X, Xw
    X, T = np.concatenate(Xs), np.concatenate(Ts)
    models = []
    for j in range(len(DEN)):
        ok = ~np.isnan(T[:, j])
        models.append(HistGradientBoostingRegressor(**REG).fit(X[ok], T[ok, j]))
    return models


def build_den(log=print):
    """Cross-fitted den_* columns: each IDSSE match from the other 6 IDSSE, each Metrica match from all 7."""
    from threadpoolctl import threadpool_limits
    ms = lab.matches(WHICH)
    trained = [m for m in ms if sets.source(m) in L.TRAIN_SOURCES]
    for held in held_list():
        targets = [held] if held in trained else [m for m in ms if m not in trained]
        if all(os.path.exists(den_path(m)) for m in targets):
            continue
        t0 = time.time()
        models = den_fit([m for m in trained if m != held])
        for m in targets:
            names_, X, Xw, _ = load_features("ghost_a", m)
            out = []
            with threadpool_limits(2):
                for Z in (X, Xw):
                    P = np.stack([r.predict(Z) for r in models], axis=1).astype(np.float32)
                    P[den_mask(names_, Z)] = np.nan
                    out.append(P)
            os.makedirs(os.path.dirname(den_path(m)), exist_ok=True)
            np.savez(den_path(m), names=np.array(DEN_NAMES), X=out[0], X_swapped=out[1])
            T, _ = den_targets(m)
            graded = np.isin(lab.load(m, WHICH).truth, (0, 1, 2))
            err = []
            for j, (a, _) in enumerate(DEN):
                ok = graded & ~np.isnan(T[:, j]) & ~np.isnan(out[0][:, j])
                dmg = X[:, names_.index(a)] - (X[:, names_.index(DEN[j][1])] if DEN[j][1] else 0)
                err.append("{} {:.2f} (damaged {:.2f})".format(DEN_NAMES[j], np.mean(np.abs(out[0][ok, j] - T[ok, j])),
                                                              np.nanmean(np.abs(dmg[ok] - T[ok, j]))))
            log("den {}: mean abs error against the clean value, {}".format(m, "; ".join(err)))
        log("den fold {} ({:.0f} s)".format(held, time.time() - t0))


def folder(variant):
    return os.path.join(CACHE, variant)


def p_path(variant, m):
    return os.path.join(folder(variant), "p_{}.npy".format(m))


# --------------------------------------------------------------------------------------------
# The screen (lab_clean.screen's procedure, its own paths)
# --------------------------------------------------------------------------------------------

def fit(variant, used):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs, ys = [], []
    for m in used:
        _, X, Xw, truth = load_features(variant, m)
        rows = np.flatnonzero(np.isin(truth, (0, 1, 2)))[::L.EVERY]
        y = truth[rows].astype(np.int64)
        Xs += [X[rows], Xw[rows]]
        ys += [y, L.SWAP[y]]
        del X, Xw
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


def score_p(match, p):
    graded = np.isin(match.truth, (0, 1, 2))
    y = match.truth[graded].astype(np.int64)
    logloss = float(-np.log(p[graded][np.arange(len(y)), y]).mean())
    letters = L.decode_cost(match, lab_robust.cost_from(match, p, CHOSEN["blend"]), **CHOSEN)
    letters = lab_robust.finish(match, letters, WHICH)
    s = lab.score(match, letters)
    return {"source": match.source, "unseen": match.id in UNSEEN, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
            "logloss": logloss}


def screen(variant, log=print):
    """P2 for one variant: bench/results/lab_ghost_<variant>.json; held-out P in data/external/lab_ghost/<variant>/."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    t0 = time.time()
    ms = lab.matches(WHICH)
    if VARIANTS[variant]:
        for m in ms:
            build_cols(m, log)
    if variant == "ghost_den":
        build_den(log)
    if variant in DIAG:
        for m in ms:
            diag_features(variant, m, log)
    os.makedirs(folder(variant), exist_ok=True)
    trained = [m for m in ms if sets.source(m) in L.TRAIN_SOURCES]
    for held in held_list():
        targets = [held] if held in trained else [m for m in ms if m not in trained]
        if all(os.path.exists(p_path(variant, m)) for m in targets):
            continue
        clf = fit(variant, [m for m in trained if m != held])
        for m in targets:
            np.save(p_path(variant, m), proba(clf, variant, m).astype(np.float32))
        log("{}: fold {} fitted ({:.0f} min)".format(variant, held, (time.time() - t0) / 60))
        del clf
    out = {}
    for m in ms:
        out[m] = score_p(lab.load(m, WHICH), np.load(p_path(variant, m)).astype(np.float64))
    with open(os.path.join(RESULTS, "lab_ghost_{}.json".format(variant)), "w") as handle:
        json.dump({"variant": variant, "columns": len(load_features(variant, ms[0])[0]), "matches": out,
                   "minutes": (time.time() - t0) / 60}, handle, indent=1)
    log("{}: spellF1 {:.4f}, log loss {:.4f} ({:.0f} min)".format(
        variant, np.mean([v["spell_f1"] for v in out.values()]), np.mean([v["logloss"] for v in out.values()]),
        (time.time() - t0) / 60))
    return out


def check():
    """g0's held-out P against round 9's c0 (data/external/lab_clean/c0/p_<m>.npy, read only) and robust's own."""
    worst = 0.0
    for m in lab.matches(WHICH):
        mine = np.load(p_path("g0", m)).astype(np.float64)
        c0 = np.load(os.path.join(lab.ROOT, "data", "external", "lab_clean", "c0", "p_{}.npy".format(m))).astype(np.float64)
        rob = lab_robust.held_out_proba(m, WHICH)
        d = max(float(np.abs(mine - c0).max()), float(np.abs(mine - rob).max()))
        worst = max(worst, d)
        print("{}: largest difference in P (vs c0, vs robust) {:.2e}".format(m, d))
    r, base = load_result("g0"), load_result(None)
    sf = np.mean([r[m]["spell_f1"] for m in r])
    print("g0 spellF1 {:.4f} (c0 {:.4f}); harness {}".format(
        sf, np.mean([base[m]["spell_f1"] for m in base]),
        "REPRODUCES c0" if worst < 1e-6 else "DIFFERS ({:.2e})".format(worst)))


# --------------------------------------------------------------------------------------------
# The report and the gate
# --------------------------------------------------------------------------------------------

def load_result(variant):
    path = BASE_RESULT if variant is None else os.path.join(RESULTS, "lab_ghost_{}.json".format(variant))
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)["matches"]


def summary(variant, base):
    r = load_result(variant)
    if r is None:
        return None
    ms = sorted(base)
    d = np.array([r[m]["spell_f1"] - base[m]["spell_f1"] for m in ms])
    met = np.array([sets.source(m) == "metrica" for m in ms])
    return {"variant": variant, "spell_f1": float(np.mean([r[m]["spell_f1"] for m in ms])), "gain": float(d.mean()),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "idsse": float(d[~met].mean()),
            "metrica": float(d[met].mean()),
            "logloss": float(np.mean([r[m]["logloss"] - base[m]["logloss"] for m in ms])),
            "per_match": {m: float(x) for m, x in zip(ms, d)}}


def report(variants=None):
    base = load_result(None)
    names_ = variants or sorted(f[len("lab_ghost_"):-len(".json")] for f in os.listdir(RESULTS)
                                if f.startswith("lab_ghost_") and f.endswith(".json") and f[len("lab_ghost_"):-len(".json")] in VARIANTS)
    rows = [s for s in (summary(v, base) for v in names_) if s]
    print("variant | P2 spellF1 | gain | better/worse | IDSSE | Metrica | log loss")
    for s in rows:
        print("{:<10s} | {:.4f} | {:+.4f} | {}/{} | {:+.4f} | {:+.4f} | {:+.4f}".format(
            s["variant"], s["spell_f1"], s["gain"], s["better"], s["worse"], s["idsse"], s["metrica"], s["logloss"]))
        print("    " + ", ".join("{} {:+.4f}".format(m, x) for m, x in s["per_match"].items()))
    with open(os.path.join(RESULTS, "lab_ghost_report.json"), "w") as handle:
        json.dump(rows, handle, indent=1)
    return rows


def gate(candidates=("ghost_a", "ghost_b"), tag="gate"):
    """Leave one match out between the candidates (each match scored with the one whose mean is better on the other 9,
    ties to the first, the simpler), against c0; the gate fixed on the page."""
    base = load_result(None)
    results = {v: load_result(v) for v in candidates}
    ms = sorted(base)
    folds = []
    for m in ms:
        others = [o for o in ms if o != m]
        means = [np.mean([results[v][o]["spell_f1"] for o in others]) for v in candidates]
        pick = candidates[int(np.argmax(np.round(means, 6)))]
        folds.append({"match": m, "picked": pick, "held_out": results[pick][m]["spell_f1"], "base": base[m]["spell_f1"],
                      "logloss": results[pick][m]["logloss"], "base_logloss": base[m]["logloss"]})
    d = np.array([f["held_out"] - f["base"] for f in folds])
    dl = np.array([f["logloss"] - f["base_logloss"] for f in folds])
    met = np.array([sets.source(f["match"]) == "metrica" for f in folds])
    overall = [np.mean([results[v][m]["spell_f1"] for m in ms]) for v in candidates]
    out = {"candidates": list(candidates), "folds": folds, "gain": float(d.mean()), "better": int((d > 0).sum()),
           "worse": int((d < 0).sum()), "metrica": float(d[met].mean()), "idsse": float(d[~met].mean()),
           "logloss": float(dl.mean()), "chosen_on_all": candidates[int(np.argmax(np.round(overall, 6)))]}
    out["passes"] = bool(out["gain"] >= GATE["gain"] and out["better"] >= GATE["better"] and out["metrica"] >= 0
                         and out["logloss"] <= GATE["logloss"])
    for f in folds:
        print("{}: picked {} -> {:+.4f} (log loss {:+.4f})".format(f["match"], f["picked"], f["held_out"] - f["base"],
                                                                 f["logloss"] - f["base_logloss"]))
    print("held out {:+.4f}, better on {}/10 (worse {}), IDSSE {:+.4f}, Metrica {:+.4f}, log loss {:+.4f}; chosen on "
          "all 10: {}; gate {}".format(out["gain"], out["better"], out["worse"], out["idsse"], out["metrica"],
                                       out["logloss"], out["chosen_on_all"], "PASSED" if out["passes"] else "not passed"))
    with open(os.path.join(RESULTS, "lab_ghost_{}.json".format(tag)), "w") as handle:
        json.dump(out, handle, indent=1)
    return out


# --------------------------------------------------------------------------------------------
# The unit check (J03WR9): the swap, and the ghost touches' rates against the probe's (clean tracking only to count)
# --------------------------------------------------------------------------------------------

def swapped_name(k):
    return k.replace("home", "\0").replace("away", "home").replace("\0", "away")


def unit(m="J03WR9"):
    from bench import lab_clean
    t0 = time.time()
    match = lab.load(m, WHICH)
    b = lab_robust.blanked(match)
    names_, X = ghost_columns(b.track, b.halves)
    _, Xs = ghost_columns(b.track, b.halves, swap=True)
    print("{}: columns in {:.0f} s".format(m, time.time() - t0))
    ok = True
    for i, k in enumerate(names_):
        j = names_.index(swapped_name(k))
        same = np.array_equal(Xs[:, i], X[:, j], equal_nan=True)
        ok &= same
        if not same:
            print("  swap MISMATCH {} vs {}".format(k, names_[j]))
    print("swap check: {}".format("X_swapped's columns = X's with home/away exchanged, bit for bit" if ok else "FAILED"))
    live = np.isin(match.truth, (0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, k in enumerate(names_):
            col = X[live, i]
            print("  {:<22s} live: NaN {:5.1%}, mean {:7.3f}, share != none {:5.1%}".format(
                k, np.isnan(col).mean(), np.nanmean(col),
                np.mean(col[~np.isnan(col)] < (FAR if k.startswith("gd_") else (GAP_CAP if k.startswith("gt_") else 1e9)))
                if k.startswith(("gd_", "gt_")) else np.mean(col[~np.isnan(col)] > 0)))
    # rates against the probe (clean tracking only to count)
    clean = sets.load(m)
    P, team, ball = b.track["players"], np.asarray(b.track["team"]), b.track["ball"][:, :2]
    cP, cteam = clean["players"].astype(np.float32), np.asarray(clean["team"])
    cball = clean["ball"][:, :2].astype(np.float64)
    vis = np.zeros(cP.shape[:2], bool)
    for (a, e, s, _), person in lab_clean.person_of_slots(match.track, clean, match.halves):
        vis[a:e, person] = True
    ev = orphan_events(P, team, b.track["ball"], b.halves)
    ev = ev[live[ev]]
    with np.errstate(invalid="ignore"):
        cd = np.hypot(cP[ev, :, 0] - cball[ev, None, 0], cP[ev, :, 1] - cball[ev, None, 1])
    cd = np.where(np.isnan(cd), np.inf, cd)
    ct = cd.argmin(1)
    creach = cd[np.arange(len(ev)), ct] <= REACH
    inv = creach & ~vis[ev, ct]
    th = NAMES.index("since_gtouch_home")
    ta = NAMES.index("since_gtouch_away")
    got_h, got_a = X[ev, th] == 0, X[ev, ta] == 0
    got = got_h | got_a
    gteam = np.where(got_h, HOME, AWAY)
    sel = inv & got
    print("orphan events (live): {}; with an invisible true toucher {} ({:.1%}); of those, given a ghost touch {:.1%} "
          "(probe ~48 %), team right {:.1%} (probe ~95-98 %)".format(
              len(ev), inv.sum(), inv.mean(), sel.sum() / max(inv.sum(), 1),
              (gteam[sel] == cteam[ct[sel]]).mean() if sel.any() else float("nan")))
    other = got & ~inv
    print("ghost touches on orphan events whose toucher is visible or nobody within 2 m (clean): {} of {} ghost "
          "touches".format(int(other.sum()), int(got.sum())))
    seen = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        cdf = np.hypot(cP[:, :, 0] - cball[:, None, 0], cP[:, :, 1] - cball[:, None, 1])
    cdf = np.where(np.isnan(cdf), np.inf, cdf)
    near = cdf.argmin(1)
    onball = cdf[np.arange(len(near)), near] <= REACH
    inv_f = live & seen & onball & ~vis[np.arange(len(near)), near]
    vis_f = live & seen & onball & vis[np.arange(len(near)), near]
    poss = np.where(match.truth == 0, HOME, AWAY)
    gdh, gda = X[:, NAMES.index("gd_home")], X[:, NAMES.index("gd_away")]
    with np.errstate(invalid="ignore"):
        dd = np.hypot(P[:, :, 0] - ball[:, None, 0], P[:, :, 1] - ball[:, None, 1])
    vd = np.nanmin(np.where(np.isnan(dd), np.inf, dd), axis=1)
    g = np.minimum(gdh, gda)
    gteam_f = np.where(gdh < gda, HOME, AWAY)
    for name, mask in (("on-ball visible", vis_f), ("on-ball INVISIBLE", inv_f)):
        has = mask & (g <= 5.0) & (g < vd)
        print("frames [{}]: {}; a ghost within 5 m and nearer than every visible player {:.1%}, its team right {:.1%} "
              "(probe J03WR9: 47 % / 92 % invisible; 0.4 % visible)".format(
                  name, mask.sum(), has.sum() / mask.sum(), (gteam_f[has] == poss[has]).mean() if has.any() else np.nan))


def where(variant):
    """Diagnostic (the clean tracking only to count): the variant's held-out P against c0's on live frames, split by
    whether the on-ball player (clean: nearest to the clean ball, within 2 m) is seen, and whether robust's ball is."""
    from bench import lab_clean
    acc = {}
    for m in lab.matches(WHICH):
        match = lab.load(m, WHICH)
        clean = sets.load(m)
        cP, cball = clean["players"].astype(np.float32), clean["ball"][:, :2].astype(np.float64)
        vis = np.zeros(cP.shape[:2], bool)
        for (a, e, s, _), person in lab_clean.person_of_slots(match.track, clean, match.halves):
            vis[a:e, person] = True
        with np.errstate(invalid="ignore"):
            cd = np.hypot(cP[:, :, 0] - cball[:, None, 0], cP[:, :, 1] - cball[:, None, 1])
        cd = np.where(np.isnan(cd), np.inf, cd)
        near = cd.argmin(1)
        onball = cd[np.arange(len(near)), near] <= REACH
        inv = ~vis[np.arange(len(near)), near]
        seen = ~np.isnan(lab_robust.blanked(match).track["ball"][:, 0])
        live = np.isin(match.truth, (0, 1))
        y = match.truth.astype(np.int64)
        p0 = np.load(os.path.join(lab.ROOT, "data", "external", "lab_clean", "c0", "p_{}.npy".format(m))).astype(np.float64)
        p1 = np.load(p_path(variant, m)).astype(np.float64)
        for name, mask in (("ball seen, on-ball seen", live & seen & onball & ~inv),
                           ("ball seen, on-ball MISSING", live & seen & onball & inv),
                           ("ball blanked, on-ball seen", live & ~seen & onball & ~inv),
                           ("ball blanked, on-ball MISSING", live & ~seen & onball & inv),
                           ("live, nobody within 2 m", live & ~onball)):
            k = np.flatnonzero(mask)
            r = acc.setdefault(name, np.zeros(5))
            right0 = np.where(y[k] == 0, p0[k, 0] > p0[k, 1], p0[k, 1] > p0[k, 0])
            right1 = np.where(y[k] == 0, p1[k, 0] > p1[k, 1], p1[k, 1] > p1[k, 0])
            r += [len(k), right0.sum(), right1.sum(), -np.log(p0[k, y[k]]).sum(), -np.log(p1[k, y[k]]).sum()]
    print("{} against c0, live frames, all 10 matches pooled:".format(variant))
    print("frames | share of live | team right c0 -> {} | log loss c0 -> {}".format(variant, variant))
    total = sum(r[0] for r in acc.values())
    for name, r in acc.items():
        print("{:<30s} | {:6.1%} | {:.1%} -> {:.1%} ({:+.1f}) | {:.4f} -> {:.4f} ({:+.4f})".format(
            name, r[0] / total, r[1] / r[0], r[2] / r[0], 100 * (r[2] - r[1]) / r[0], r[3] / r[0], r[4] / r[0],
            (r[4] - r[3]) / r[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=("unit", "screen", "check", "report", "gate", "cols", "where"))
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()
    if args.what == "unit":
        unit(*(args.variants[:1] or []))
    elif args.what == "cols":
        for m in args.variants or lab.matches(WHICH):
            build_cols(m, log=lambda s: print(s, flush=True))
    elif args.what == "screen":
        for v in args.variants:
            if v not in VARIANTS:
                raise ValueError(v)
        for v in args.variants:
            screen(v, log=lambda s: print(s, flush=True))
    elif args.what == "check":
        check()
    elif args.what == "where":
        for v in args.variants:
            where(v)
    elif args.what == "report":
        report(args.variants or None)
    elif args.what == "gate":
        if args.variants:
            gate(tuple(args.variants), tag="gate_" + "_".join(args.variants))
        else:
            gate()


if __name__ == "__main__":
    main()
