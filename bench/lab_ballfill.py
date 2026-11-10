"""Lab candidate `ballfill` (round 7): the missing ball reconstructed, and robust told who is near it (docs/lab/ballfill.md).

    python -m bench.lab_ballfill --impute [--set all|idsse]   # ~5 min: the imputer, leave one match out; its error against interpolation
    python -m bench.lab_ballfill [--set all|idsse]            # ~25 min: robust + 11 filled-ball features; train, decode over robust's grid, judge against robust
    python -m bench.lab_ballfill --chains [--set all|idsse]   # ~25 min: v12's chain from this start, against v12

robust's team call is right 93 % of the time with the ball seen and 86 % with it missing (inside-v12.md),
and the ball is missing on ~41 % of the bench's frames (more in robust's view, where a ball still for
1 s counts as missing). Here the missing ball is estimated from what the match offers at prediction
time, and robust gets a few numbers about who is near the estimate.

The imputer works on robust's view of the ball (lab_robust.blanked: the still ball is missing too, so
nothing here can read a still ball). For each missing frame inside a half with a seen ball before or
after it: plain interpolation between the last and next seen positions (or the one that exists), plus
an offset (x, y) from two gradient-boosted regressors fed IMPUTE_NAMES, all team-agnostic (so they
can't learn which side the bench's home teams are): the gap (length, time since and until, how far
through), the anchors (where, how far outside the pitch, the ball's velocity there, dead reckoning
from each), the players around the interpolated point (the two nearest, the centroid, how many within
5 m, their mean speed), and the player nearest the ball when last seen / next seen, where he is now
(and where the ball would be had it moved with him). Trained leave one match out, the target the clean
ball (bench.sets.load) on the other matches' missing frames: IDSSE's everywhere (tracked through dead
time), Metrica's where its clean ball exists (live play: it blanks dead time). The `idsse` set's
imputer trains on the other IDSSE matches only. Never the predicted match's clean ball.

The features, 11 per frame beside robust's 97, on the filled ball (the seen ball where seen, the
estimate where missing): imputed or not; the nearest home and away player and their difference (with
its past / next 1 s and 3 s means); where the ball is (x from home's goal, from the touchline, beyond a
line). No speed of the filled ball (a reconstructed track's speed is mostly the gap's length again,
and robust keeps the ball's speed out of anything blanked). Everything else is robust's: its cached
features, IDSSE-only training, every 10th graded frame, the teams swapped, the same trees, the same
decoder grid, then v9's snap and v11's move on the original tracking.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from bench import chain, lab, lab_mix, lab_robust, lab_weighted, sets
from possession import features as pf
from possession import lab_learned_features as llf, v12
from possession.data import AWAY, FPS, HOME

# Params (fixed before any held-out run; the page gives the why of each)
L = lab_robust.L
NAME = "ballfill"
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_ballfill")    # gitignored (data/*)
LOOK = llf.LOOK                  # frames (0.2 s): the ball's velocity at an anchor, as robust's features
RECKON = 1.0                     # s: dead reckoning from an anchor's velocity, at most this far
CROWD = 5.0                      # m: players counted around the interpolated point
MARGIN = 10.0                    # m: the estimate is kept within the pitch plus this
IMPUTE_EVERY = 4                 # train the imputer on every 4th missing frame (0.16 s apart)
REGRESSOR = {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 100,
             "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}
GAP_BINS = (0.0, 1.0, 3.0, 10.0, 30.0, np.inf)   # s, for the error table only
# Fallback, fixed beforehand: if the imputer's pooled held-out error (mean metres, every missing frame
# with a clean target) isn't below plain interpolation's, the features read interpolation instead.
COLUMNS = ["switch_team", "switch_dead", "min_possession"]
GRID = [dict(zip(COLUMNS, (team, dead, k))) for team in L.SWITCH_TEAM for dead in L.SWITCH_DEAD for k in L.MIN_POSSESSION]
IMPUTE_NAMES = ["gap_s", "since_s", "until_s", "frac", "has_last", "has_next",
                "interp_x", "interp_y", "interp_edge", "last_outside", "next_outside",
                "move_dx", "move_dy", "move_d", "v_last_x", "v_last_y", "v_next_x", "v_next_y",
                "reck_last_dx", "reck_last_dy", "reck_next_dx", "reck_next_dy",
                "p1_d", "p1_dx", "p1_dy", "p2_d", "p2_dx", "p2_dy",
                "c_last_d0", "c_last_dx", "c_last_dy", "carry_last_dx", "carry_last_dy",
                "c_next_d0", "c_next_dx", "c_next_dy", "carry_next_dx", "carry_next_dy", "c_same_team",
                "cent_dx", "cent_dy", "n_crowd", "speed_all"]


def folder(which):
    return CACHE + chain.suffix(which)


# --------------------------------------------------------------------------------------------
# The imputer
# --------------------------------------------------------------------------------------------

def in_halves(n, halves):
    out = np.zeros(n, bool)
    for _, first, last in halves:
        out[first:last + 1] = True
    return out


def anchors(seen, halves):
    """(last, next) int [frames]: the last seen frame at or before each frame, the next at or after,
    within its half; -1 where there's none (and outside the halves)."""
    n = len(seen)
    last, nxt = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    for _, first, end in halves:
        idx = np.arange(first, end + 1)
        s = seen[first:end + 1]
        last[first:end + 1] = np.maximum.accumulate(np.where(s, idx, -1))
        r = np.minimum.accumulate(np.where(s, idx, n)[::-1])[::-1]
        nxt[first:end + 1] = np.where(r == n, -1, r)
    return last, nxt


def velocity(ball, halves, ahead):
    """m/s over the LOOK frames before (or after) each frame, within the half; NaN where unknown."""
    out = np.full(ball.shape, np.nan, np.float32)
    for _, first, end in halves:
        b = ball[first:end + 1].astype(np.float64)
        step = (b[LOOK:] - b[:-LOOK]) * (FPS / LOOK)
        if ahead:
            out[first:end + 1 - LOOK] = step
        else:
            out[first + LOOK:end + 1] = step
    return out


def outside(p, pitch):
    """m beyond a line (< 0 inside), as robust's ball_outside."""
    length, width = pitch
    return np.maximum.reduce([-p[:, 0], p[:, 0] - length, -p[:, 1], p[:, 1] - width])


def nearest_slot(players, at):
    """(slot, distance) of the player nearest each point at[frame]; -1 / NaN where nobody is seen."""
    d = np.hypot(*(players - at[:, None, :]).transpose(2, 0, 1))
    d = np.where(np.isnan(d), np.inf, d)
    slot = d.argmin(axis=1)
    dist = d[np.arange(len(d)), slot]
    slot[np.isinf(dist)] = -1
    return slot, np.where(np.isinf(dist), np.nan, dist)


def impute_inputs(track, halves, pitch):
    """(rows, interp [rows, 2], X [rows, IMPUTE_NAMES]) for every frame inside a half whose ball is
    missing and that has a seen ball before or after it in the half. Reads the damaged tracking only."""
    ball, players, team = track["ball"], track["players"], np.asarray(track["team"])
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    last, nxt = anchors(seen, halves)
    rows = np.flatnonzero(in_halves(n, halves) & ~seen & ((last >= 0) | (nxt >= 0)))
    lo, hi = last[rows], nxt[rows]
    hl, hr = lo >= 0, hi >= 0
    both = hl & hr
    pl = np.where(hl[:, None], ball[np.maximum(lo, 0)], np.nan).astype(np.float64)
    pr = np.where(hr[:, None], ball[np.maximum(hi, 0)], np.nan).astype(np.float64)
    since = np.where(hl, (rows - lo) / FPS, np.nan)
    until = np.where(hr, (hi - rows) / FPS, np.nan)
    frac = np.where(both, (rows - lo) / np.maximum(hi - lo, 1), np.nan)
    interp = np.where(both[:, None], pl + (pr - pl) * frac[:, None], np.where(hl[:, None], pl, pr))
    f = {"gap_s": np.where(both, (hi - lo - 1) / FPS, np.nan), "since_s": since, "until_s": until, "frac": frac,
         "has_last": hl.astype(float), "has_next": hr.astype(float),
         "interp_x": interp[:, 0], "interp_y": interp[:, 1]}
    length, width = pitch
    f["interp_edge"] = np.minimum.reduce([interp[:, 0], length - interp[:, 0], interp[:, 1], width - interp[:, 1]])
    f["last_outside"], f["next_outside"] = outside(pl, pitch), outside(pr, pitch)
    move = pr - pl
    f["move_dx"], f["move_dy"], f["move_d"] = move[:, 0], move[:, 1], np.hypot(move[:, 0], move[:, 1])
    v_last = np.where(hl[:, None], velocity(ball, halves, False)[np.maximum(lo, 0)], np.nan)
    v_next = np.where(hr[:, None], velocity(ball, halves, True)[np.maximum(hi, 0)], np.nan)
    f["v_last_x"], f["v_last_y"], f["v_next_x"], f["v_next_y"] = v_last[:, 0], v_last[:, 1], v_next[:, 0], v_next[:, 1]
    reck_last = pl + v_last * np.minimum(since, RECKON)[:, None] - interp
    reck_next = pr - v_next * np.minimum(until, RECKON)[:, None] - interp
    f["reck_last_dx"], f["reck_last_dy"] = reck_last[:, 0], reck_last[:, 1]
    f["reck_next_dx"], f["reck_next_dy"] = reck_next[:, 0], reck_next[:, 1]

    # The players around the interpolated point.
    p = players[rows].astype(np.float64)                     # [rows, slots, 2]
    off = p - interp[:, None, :]
    d = np.hypot(off[..., 0], off[..., 1])
    d_inf = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(d_inf, axis=1)[:, :2]
    k = np.arange(len(rows))
    for j, name in enumerate(("p1", "p2")):
        dj = d_inf[k, order[:, j]]
        ok = np.isfinite(dj)
        f[name + "_d"] = np.where(ok, dj, np.nan)
        f[name + "_dx"] = np.where(ok, off[k, order[:, j], 0], np.nan)
        f[name + "_dy"] = np.where(ok, off[k, order[:, j], 1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cent = np.nanmean(off, axis=1)
        f["speed_all"] = np.nanmean(pf.speeds(players, halves)[rows], axis=1)
    f["cent_dx"], f["cent_dy"] = cent[:, 0], cent[:, 1]
    f["n_crowd"] = (d_inf < CROWD).sum(axis=1).astype(float)

    # The player nearest the ball when last seen (next seen): where he is now, and where the ball
    # would be had it moved with him.
    slots = {}
    for word, anchor, has, pos in (("last", lo, hl, pl), ("next", hi, hr, pr)):
        frames = np.unique(anchor[has])
        slot_at, dist_at = nearest_slot(players[frames].astype(np.float64), ball[frames].astype(np.float64))
        where = np.searchsorted(frames, np.maximum(anchor, 0))
        where = np.minimum(where, max(len(frames) - 1, 0))
        slot = np.where(has, slot_at[where] if len(frames) else -1, -1)
        d0 = np.where(has, dist_at[where] if len(frames) else np.nan, np.nan)
        ok = slot >= 0
        now = np.where(ok[:, None], p[k, np.maximum(slot, 0)], np.nan)
        then = np.where(ok[:, None], players[np.maximum(anchor, 0), np.maximum(slot, 0)].astype(np.float64), np.nan)
        carried = pos + (now - then) - interp
        f["c_{}_d0".format(word)] = d0
        f["c_{}_dx".format(word)], f["c_{}_dy".format(word)] = now[:, 0] - interp[:, 0], now[:, 1] - interp[:, 1]
        f["carry_{}_dx".format(word)], f["carry_{}_dy".format(word)] = carried[:, 0], carried[:, 1]
        slots[word] = slot
    same = (slots["last"] >= 0) & (slots["next"] >= 0)
    f["c_same_team"] = np.where(same, (team[np.maximum(slots["last"], 0)] == team[np.maximum(slots["next"], 0)]).astype(float),
                                np.nan)
    X = np.stack([np.asarray(f[name], dtype=np.float32) for name in IMPUTE_NAMES], axis=1)
    return rows, interp.astype(np.float32), X


def impute_cache(m, which):
    """{rows, interp, X, target, gap, truth}: the imputer's inputs on robust's view of the ball, the clean
    ball's offset from interpolation (NaN where the clean ball is missing; for training on the OTHER
    matches and for measuring), and the truth at those frames (for the error table only)."""
    path = os.path.join(folder(which), "imp_{}.npz".format(m))
    if not os.path.exists(path):
        match = lab.load(m, which)
        b = lab_robust.blanked(match)
        rows, interp, X = impute_inputs(b.track, b.halves, b.pitch)
        clean = sets.load(m)["ball"][rows].astype(np.float32)
        os.makedirs(folder(which), exist_ok=True)
        np.savez(path, rows=rows, interp=interp, X=X, target=clean - interp, truth=match.truth[rows],
                 frames=np.int64(match.frames))
        del match, b
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def fit_imputer(matches, which):
    """Two regressors (x, y offsets from interpolation) on these matches' missing frames with a clean target."""
    Xs, ys = [], []
    for m in matches:
        c = impute_cache(m, which)
        ok = np.flatnonzero(np.isfinite(c["target"]).all(axis=1))[::IMPUTE_EVERY]
        Xs.append(c["X"][ok])
        ys.append(c["target"][ok])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    with threadpool_limits(2):
        return [HistGradientBoostingRegressor(**REGRESSOR).fit(X, y[:, j]) for j in range(2)], len(y)


def impute_all(which):
    """Each match's filled ball from an imputer trained without it; then the error table."""
    ms = lab.matches(which)
    os.makedirs(folder(which), exist_ok=True)
    for m in ms:
        started = time.time()
        impute_cache(m, which)
        print("{}: imputer inputs cached ({:.0f} s)".format(m, time.time() - started), flush=True)
    for held in ms:
        started = time.time()
        models, n_rows = fit_imputer([m for m in ms if m != held], which)
        c = impute_cache(held, which)
        with threadpool_limits(2):
            offset = np.stack([model.predict(c["X"]) for model in models], axis=1).astype(np.float32)
        match = lab.load(held, which)
        pitch = np.array(match.pitch, np.float32)
        estimate = np.clip(c["interp"] + offset, -MARGIN, pitch + MARGIN)
        ball = lab_robust.blanked(match).track["ball"]
        filled = {"imputer": ball.copy(), "interpolation": ball.copy()}
        filled["imputer"][c["rows"]] = estimate
        filled["interpolation"][c["rows"]] = c["interp"]
        np.savez(os.path.join(folder(which), "fill_{}.npz".format(held)), estimate=estimate, **filled)
        print("{}: imputer fit on {} rows, filled {} frames ({:.0f} s)".format(held, n_rows, len(c["rows"]),
                                                                              time.time() - started), flush=True)
        del match, ball, filled
    return measure(which)


def team_of_nearest(players, team, at):
    slot, _ = nearest_slot(players.astype(np.float64), at.astype(np.float64))
    return np.where(slot >= 0, team[np.maximum(slot, 0)], -1)


def measure(which):
    """The imputer's held-out error against interpolation, in metres, by source, gap length, live / dead;
    and how often the player nearest each estimate is of the team in possession (live frames)."""
    rows = []
    for m in lab.matches(which):
        c = impute_cache(m, which)
        with np.load(os.path.join(folder(which), "fill_{}.npz".format(m))) as z:
            estimate = z["estimate"]
        ok = np.isfinite(c["target"]).all(axis=1)
        clean = c["interp"] + c["target"]
        match = lab.load(m, which)
        team = np.asarray(match.track["team"])
        players = match.track["players"][c["rows"]]
        last = lab_robust.blanked(match).track["ball"]
        lo, _ = anchors(~np.isnan(last[:, 0]), match.halves)
        last_seen = np.where((lo[c["rows"]] >= 0)[:, None], last[np.maximum(lo[c["rows"]], 0)], c["interp"])
        live = np.isin(c["truth"], (0, 1))
        nearest = {}
        for name, at in (("last_seen", last_seen), ("interpolation", c["interp"]), ("imputer", estimate),
                         ("clean", np.where(ok[:, None], clean, c["interp"]))):
            nearest[name] = team_of_nearest(players, team, at)
        gap = c["X"][:, IMPUTE_NAMES.index("gap_s")]
        d_int = np.hypot(*(c["interp"] - clean).T)
        d_imp = np.hypot(*(estimate - clean).T)
        frame = pd.DataFrame({"match": m, "source": sets.source(m), "gap_s": gap, "truth": c["truth"],
                              "err_interp": d_int, "err_imputer": d_imp, "has_target": ok,
                              **{"team_" + k: (v == c["truth"]) & live for k, v in nearest.items()}, "live": live})
        rows.append(frame)
        del match, players, last
    df = pd.concat(rows, ignore_index=True)
    df["gap"] = pd.cut(df.gap_s.fillna(-1), [-np.inf, -0.5] + list(GAP_BINS[1:]), right=False,
                       labels=["half edge"] + ["{:g}-{:g} s".format(a, b) for a, b in zip(GAP_BINS[:-1], GAP_BINS[1:])])
    df["state"] = np.where(df.truth == 2, "dead", np.where(df.live, "live", "other"))
    t = df[df.has_target]

    def err(g):
        return pd.Series({"frames": len(g), "interp_mean": g.err_interp.mean(), "imputer_mean": g.err_imputer.mean(),
                          "interp_median": g.err_interp.median(), "imputer_median": g.err_imputer.median(),
                          "interp_within_2m": (g.err_interp < 2).mean(), "imputer_within_2m": (g.err_imputer < 2).mean(),
                          "interp_within_5m": (g.err_interp < 5).mean(), "imputer_within_5m": (g.err_imputer < 5).mean()})

    tables = {"overall": err(t).to_frame("all").T, "by_source": t.groupby("source").apply(err),
              "by_state": t.groupby("state").apply(err), "by_gap": t.groupby("gap", observed=True).apply(err),
              "by_gap_live": t[t.state == "live"].groupby("gap", observed=True).apply(err),
              "by_match": t.groupby("match").apply(err)}
    live = t[t.live]           # live frames with a clean ball, so the four columns compare on the same frames
    team_cols = ["team_last_seen", "team_interpolation", "team_imputer", "team_clean"]
    tables["nearest_team_live"] = live.groupby("source")[team_cols].mean()
    tables["nearest_team_live"].loc["all"] = live[team_cols].mean()
    tables["nearest_team_live_by_gap"] = live.groupby("gap", observed=True)[team_cols].mean()
    overall = tables["overall"].iloc[0]
    use = "imputer" if overall.imputer_mean < overall.interp_mean else "interpolation"
    for name, table in tables.items():
        print("\n{}:\n{}".format(name, table.round(3).to_string()))
    print("\nthe features read: {} (imputer {:.2f} m against interpolation {:.2f} m, pooled)".format(
        use, overall.imputer_mean, overall.interp_mean))
    out = {"use": use, **{k: v.reset_index().to_dict(orient="records") for k, v in tables.items()}}
    with open(os.path.join(lab.RESULTS, "lab_{}_impute{}.json".format(NAME, chain.suffix(which))), "w") as handle:
        json.dump(out, handle, indent=1, default=str)
    return out


def chosen_fill(which):
    with open(os.path.join(lab.RESULTS, "lab_{}_impute{}.json".format(NAME, chain.suffix(which)))) as handle:
        return json.load(handle)["use"]


# --------------------------------------------------------------------------------------------
# The features, training, prediction (robust's procedure plus the filled-ball features)
# --------------------------------------------------------------------------------------------

def fill_features(track, halves, pitch, filled, imputed, swap=False):
    """(names, float32 [frames, 11]) on the filled ball; swap exchanges the teams, as robust's features."""
    players, team = track["players"], np.asarray(track["team"])
    if swap:
        team = np.where(team == HOME, AWAY, np.where(team == AWAY, HOME, team))
    n = len(filled)
    length, width = pitch
    known = ~np.isnan(filled[:, 0])
    f = {"fill_imputed": np.where(known, imputed.astype(np.float32), np.nan)}
    near = {side: llf.nearest(llf.distances(players, team, filled, side), known) for side in (HOME, AWAY)}
    f["fill_d_home"], f["fill_d_away"] = near[HOME], near[AWAY]
    f["fill_d_diff"] = np.clip(near[HOME] - near[AWAY], -llf.DIFF_CLIP, llf.DIFF_CLIP)
    llf.window_pair(f, "fill_d_diff", f["fill_d_diff"], halves)
    left = llf.home_left(players, team, halves, n)
    x, y = filled[:, 0], filled[:, 1]
    f["fill_x_from_home_goal"] = np.where(left, x, length - x)
    f["fill_y_from_touchline"] = np.minimum(y, width - y)
    f["fill_outside"] = np.maximum.reduce([-x, x - length, -y, y - width])
    names = list(f)
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], axis=1)
    X[~in_halves(n, halves)] = np.nan
    return names, X


def features(m, which):
    """robust's 97 cached features beside the 11 filled-ball ones, as is and with the teams swapped."""
    path = os.path.join(folder(which), "feat_{}.npz".format(m))
    if not os.path.exists(path):
        match = lab.load(m, which)
        b = lab_robust.blanked(match)
        with np.load(os.path.join(folder(which), "fill_{}.npz".format(m))) as z:
            filled = z[chosen_fill(which)]
        imputed = np.isnan(b.track["ball"][:, 0]) & ~np.isnan(filled[:, 0])
        names, X = fill_features(match.track, match.halves, match.pitch, filled, imputed)
        _, Xs = fill_features(match.track, match.halves, match.pitch, filled, imputed, swap=True)
        np.savez(path, names=np.array(names), X=X, X_swapped=Xs)
        del match, b
    c = lab_robust.cached(m, which)
    with np.load(path) as z:
        names = c["names"] + [str(k) for k in z["names"]]
        return names, np.hstack([c["X"], z["X"]]), np.hstack([c["X_swapped"], z["X_swapped"]]), c["truth"], c["blanked"]


def fit(matches, which):
    Xs, ys = [], []
    for m in [m for m in matches if sets.source(m) in L.TRAIN_SOURCES]:
        _, X, Xw, truth, _ = features(m, which)
        rows = np.flatnonzero(np.isin(truth, (0, 1, 2)))[::L.EVERY]
        y = truth[rows].astype(np.int64)
        Xs += [X[rows], Xw[rows]]
        ys += [y, L.SWAP[y]]
        del X, Xw
    with threadpool_limits(2):
        return HistGradientBoostingClassifier(**L.TREES).fit(np.concatenate(Xs), np.concatenate(ys))


def held_out(match, which):
    return np.load(os.path.join(folder(which), "p_{}.npy".format(getattr(match, "id", match)))).astype(np.float64)


def fill_importance(clf, names, X, truth):
    """Rise in held-out log loss when the 11 filled-ball features are shuffled (every 25th graded frame),
    and, for scale, robust's ball family."""
    rows = np.flatnonzero(np.isin(truth, (0, 1, 2)))[::25]
    Z0, y = X[rows], truth[rows].astype(np.int64)

    def loss(Z):
        with threadpool_limits(2):
            p = np.clip(clf.predict_proba(Z), 1e-6, 1)
        return float(-np.log(p[np.arange(len(y)), y]).mean())

    base, rng, out = loss(Z0), np.random.default_rng(0), {}
    for group, test in (("filled ball", lambda k: k.startswith("fill_")), ("ball", L.GROUPS["ball"])):
        idx = [i for i, k in enumerate(names) if test(k)]
        Z = Z0.copy()
        Z[:, idx] = Z[rng.permutation(len(Z))][:, idx]
        out[group] = loss(Z) - base
    return base, out


def train_all(which):
    ms = lab.matches(which)
    models, extra = {}, []
    for held in ms:
        started = time.time()
        used = tuple(m for m in ms if m != held and sets.source(m) in L.TRAIN_SOURCES)
        if used not in models:
            models = {used: fit(used, which)}
        names, X, Xw, truth, blanked = features(held, which)
        clf = models[used]
        with threadpool_limits(2):
            p = (clf.predict_proba(X) + clf.predict_proba(Xw)[:, L.SWAP]) / 2
        p = np.round(np.clip(p, *L.CLIP), L.DECIMALS)
        np.save(os.path.join(folder(which), "p_{}.npy".format(held)), p.astype(np.float32))
        base, imp = fill_importance(clf, names, X, truth)
        # Frame accuracy of the team call in play, ball seen or missing (robust's view), against robust's.
        p_robust = lab_robust.held_out_proba(held, which, "blank")
        seen = X[:, names.index("ball_seen")] == 1
        live = np.isin(truth, (0, 1))
        row = {"match": held, "log_loss": base, **{"imp " + k: v for k, v in imp.items()}}
        for name, q in (("ballfill", p), ("robust", p_robust)):
            team_right = q[:, :2].argmax(axis=1) == truth
            row["team_acc_seen_" + name] = float(team_right[live & seen].mean())
            row["team_acc_missing_" + name] = float(team_right[live & ~seen].mean())
            row["frame_acc_" + name] = float((q.argmax(axis=1) == truth)[np.isin(truth, (0, 1, 2))].mean())
        extra.append(row)
        print("{}: predicted ({:.0f} s); team right in play, ball missing {:.3f} (robust {:.3f}), seen {:.3f} ({:.3f}); "
              "filled-ball importance {:.4f}".format(held, time.time() - started, row["team_acc_missing_ballfill"],
                                                   row["team_acc_missing_robust"], row["team_acc_seen_ballfill"],
                                                   row["team_acc_seen_robust"], imp["filled ball"]), flush=True)
        del X, Xw
    return pd.DataFrame(extra)


def letters_of(match, which, setting, snap=True):
    decoded = L.decode_cost(match, -np.log(held_out(match, which)), **setting)
    return np.asarray(L.finish(match, decoded, which) if snap else decoded).astype("<U1")


def grid(which):
    rows = []
    for match in lab.each(which):
        for setting in GRID:
            s = lab.score(match, letters_of(match, which, setting))
            rows.append({"match": match.id, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                         "possessions": s["possessions"]})
        print("{}: grid scored".format(match.id), flush=True)
    return pd.DataFrame(rows)


def chains(which):
    """v12's chain from robust (v12) and from this start, each stage refitted on its setting for that match."""
    lab_mix.cache_short_evidence()
    matches = list(lab.each(which))
    picks = lab_mix.fold_settings(which)
    with open(os.path.join(lab.RESULTS, "lab_{}{}.json".format(NAME, chain.suffix(which)))) as handle:
        mine = {f["match"]: f["picked"] for f in json.load(handle)["folds"]}
    scores = {}
    for start in ("robust", NAME):
        if start == "robust":
            letters = {m.id: lab_weighted.start_letters(m, "robust", True, picks, which) for m in matches}
        else:
            letters = {m.id: letters_of(m, which, mine[m.id]) for m in matches}
        for stage in v12.CHAIN[which]:
            letters = lab_mix.add_stage(stage, letters, matches, picks)
        scores[start] = lab_mix.scores_of(letters, matches)
        print("{}: {:.4f}".format(start, np.mean([v["spell_f1"] for v in scores[start].values()])), flush=True)
    report = lab_mix.judge_scores(scores[NAME], scores["robust"], which)
    print("\n{} chain against v12:\n{}".format(NAME, lab.describe(report)))
    table = pd.DataFrame({k: {m: v["spell_f1"] for m, v in s.items()} for k, s in scores.items()})
    print(table.round(4).to_string())
    with open(os.path.join(lab.RESULTS, "lab_{}_chains{}.json".format(NAME, chain.suffix(which))), "w") as handle:
        json.dump({"scores": table.round(5).to_dict(), "against_v12": report}, handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--impute", action="store_true", help="only the imputer and its error")
    parser.add_argument("--measure", action="store_true", help="only the imputer's error table (after --impute)")
    parser.add_argument("--chains", action="store_true", help="v12's chain from this start, after the run")
    args = parser.parse_args()
    if args.impute:
        return impute_all(args.which)
    if args.measure:
        return measure(args.which)
    if args.chains:
        return chains(args.which)
    frames = train_all(args.which)
    g = grid(args.which)
    report = lab.judge(g, COLUMNS, args.which, base=lab_weighted.robust_base(args.which))
    print(lab.describe(report))
    print("picks: {}".format(pd.Series([str(f["picked"]) for f in report["folds"]]).value_counts().to_dict()))
    print(frames.round(4).to_string())
    report["frames"] = frames.to_dict(orient="records")
    report["fill"] = chosen_fill(args.which)
    lab.save(NAME, g, report, args.which)


if __name__ == "__main__":
    main()
