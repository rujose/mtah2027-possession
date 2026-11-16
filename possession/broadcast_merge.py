"""The broadcast merge (push track 3, docs/lab/push/broadcast.md): players the broadcast tracking sees and the tactical
tracking lacks, added to the tactical tracking before branch B's chain reads it.

    from possession import broadcast_merge
    merged, status = broadcast_merge.merge_broadcast(tracking, halves)            # the broadcast set from games.py
    merged, status = broadcast_merge.merge_broadcast(tracking, halves, paths={"atd_meta": ..., "atd_raw": ...,
                                                                               "homography": ...})

Reads the broadcast set by path in the formats docs/data.md describes (files 6-8: EPTS metadata XML, the raw
tracking text `frame:x,y,s;...:x,y,s`, the homography CSV with a `valid` column), lines it up with the tactical
tracking by the ball (tools/align_broadcast.py's method: one shift per graded half, the shift where the two balls sit
on top of each other), and merges with rule M (bench/lab_broadcast.py RULES["M"], fixed before any score). Prints
nothing. `status` is one word: "merged", or "fallback-<reason>" when anything fails, and then `merged` is `tracking`
itself, unchanged (the program then runs exactly as without the broadcast).

Call it on the tactical tracking as delivered (possession.data.load_tracking("tactical")) with the graded halves,
before v12.inputs (v8's repair and every input are then computed from the merged tracking, as on the bench).

Rule M, per valid broadcast frame: (1) the broadcast is registered on the tactical players (median vector to each
broadcast player's nearest tactical player within 5 m, if at least 3); (2) the frame is dropped if the median distance
left is above 1.5 m; (3) a broadcast player with no tactical player of his team within 2.5 m and none of the other
team within 1.5 m is added; (4) each run of added frames of a broadcast slot becomes a piece in a free slot of its
team (empty from a frame before to a frame after it, the one free longest), a new slot only if none is free.
"""

import warnings
import xml.etree.ElementTree as ET
from bisect import bisect_left

import numpy as np

from possession import v1
from possession.data import AWAY, HOME, PITCH_LENGTH, PITCH_WIDTH

# Params (docs/lab/push/broadcast.md; fixed before any score)
RULE = {"register": True, "gate": 1.5, "r_same": 2.5, "r_other": 1.5, "cap": None}       # rule M
RULE_M2 = dict(RULE, cap=11)  # M2 (post hoc, docs/lab/push/broadcast.md): a team never has more than 11
REG_REACH = 5.0               # m: a broadcast player registers on a tactical one this near
REG_MIN = 3                   # broadcast players needed to register a frame
# the alignment (tools/align_broadcast.py's method) and when to trust it
MIN_OVERLAP = 0.2             # a shift needs shared ball frames >= 20 % of the half's ball frames
RMS_OK = 12.0                 # m: rms ball distance at the chosen shift (docs/data.md: 5.63 / 7.39 m; next best 36.9 / 37.8)
RUNNER_RATIO = 3.0            # the next-best shift (250+ frames away) must be this many times worse
TEAM_CHECK_M = 1.5            # m: a broadcast player this near a tactical one counts for the team-order check


# --------------------------------------------------------------------------------------------
# Reading (the formats of files 6-8, as possession/data.py reads files 1-3)
# --------------------------------------------------------------------------------------------

def read_meta(path):
    """Team of each slot (HOME for the metadata's first team) and the teams' names in order."""
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    ids = [t.get("id") for t in teams]
    names = [(t.findtext("Name") or t.get("name") or "").strip().lower() for t in teams]
    slot_team = np.array([HOME if p.get("teamId") == ids[0] else AWAY for p in root.iter("Player")], dtype=np.int8)
    return slot_team, names


def read_raw(path, slots, pitch=(PITCH_LENGTH, PITCH_WIDTH)):
    """players [frames, slots, 2] and ball [frames, 2] in metres; line N is frame N - 1."""
    lines = {}
    with open(path) as handle:
        for line in handle:
            frame_id, players, ball = line.rstrip("\n").split(":")
            lines[int(frame_id) - 1] = (players.replace(";", ",") + "," + ball)
    n = max(lines) + 1
    rows = np.full((n, slots * 3 + 3), np.nan, dtype=np.float32)
    for i, text in lines.items():
        rows[i] = np.array(text.split(","), dtype=np.float32)
    scale = np.array(pitch, dtype=np.float32)
    return rows[:, :-3].reshape(n, slots, 3)[:, :, :2] * scale, rows[:, -3:-1] * scale


def read_valid(path, n):
    import pandas as pd
    h = pd.read_csv(path, usecols=["frame", "valid"])
    valid = np.zeros(n, bool)
    f = h["frame"].to_numpy()
    keep = (f >= 0) & (f < n)
    valid[f[keep]] = h["valid"].to_numpy()[keep].astype(bool)
    return valid


# --------------------------------------------------------------------------------------------
# The alignment (tools/align_broadcast.py's global_shift, unchanged)
# --------------------------------------------------------------------------------------------

def xcorr(a, b):
    size = 1 << (len(a) + len(b) - 2).bit_length()
    r = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    return np.concatenate([r[-(len(b) - 1):], r[:len(a)]])


def global_shift(tac_ball, bro_ball, first, last):
    """(shift, rms at it, rms of the next-best shift 250+ frames away): broadcast frame = tactical frame - shift."""
    tb = tac_ball[first:last + 1]
    mt, mb = ~np.isnan(tb[:, 0]), ~np.isnan(bro_ball[:, 0])
    xt, xb = np.nan_to_num(tb), np.nan_to_num(bro_ball)
    count = xcorr(mt.astype(float), mb.astype(float))
    squares, cross = np.zeros_like(count), np.zeros_like(count)
    for axis in range(2):
        squares += xcorr(mt * xt[:, axis] ** 2, mb.astype(float)) + xcorr(mt.astype(float), mb * xb[:, axis] ** 2)
        cross += xcorr(mt * xt[:, axis], mb * xb[:, axis])
    with np.errstate(invalid="ignore"):
        mean_sq = np.where(count > MIN_OVERLAP * mt.sum(), (squares - 2 * cross) / np.maximum(count, 1), np.inf)
    lags = np.arange(-(len(mb) - 1), len(mt))
    order = np.argsort(mean_sq)
    best = order[0]
    runner = next(i for i in order if abs(lags[i] - lags[best]) > 250)
    return int(first + lags[best]), float(np.sqrt(mean_sq[best])), float(np.sqrt(mean_sq[runner]))


def on_tactical_timeline(bro_players, bro_valid, shifts, halves, n):
    """The broadcast's players and validity on the tactical frames (NaN / invalid outside the halves)."""
    players = np.full((n, bro_players.shape[1], 2), np.nan, dtype=np.float32)
    valid = np.zeros(n, bool)
    for (_, first, last), shift in zip(halves, shifts):
        t = np.arange(first, last + 1)
        b = t - shift
        inside = (b >= 0) & (b < len(bro_players))
        players[t[inside]] = bro_players[b[inside]]
        valid[t[inside]] = bro_valid[b[inside]]
    return players, valid


# --------------------------------------------------------------------------------------------
# The merge: a copy of bench/lab_broadcast.py's merge(), the function every bench score used
# ( checks the two, and this module end to end, on a bench match)
# --------------------------------------------------------------------------------------------

def merge(players, team, bro_players, bro_team, bro_valid, halves, rule=None, ball=None, chunk=4000):
    """The tactical tracking with the broadcast's extra players added as pieces in free slots of their team.
    Returns (players, team, added [n, S_bro] bool, registered broadcast positions [n, S_bro, 2], info)."""
    r = {**RULE, "d": None, **(rule or {})}
    n = len(players)
    team = np.asarray(team).astype(np.int8)
    bro_team = np.asarray(bro_team).astype(np.int8)
    S = bro_players.shape[1]
    keep = np.zeros((n, S), bool)
    reg = np.full(bro_players.shape, np.nan, np.float32)
    dropped = np.zeros(n, bool)
    shift = np.zeros((n, 2), np.float32)
    same_team = bro_team[:, None] == team[None, :]                               # [S, slots]
    last_ball = None
    if r["d"] is not None and ball is not None:
        import pandas as pd
        last_ball = pd.DataFrame(ball).ffill().values
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        D = bro_players[c0:c1].astype(np.float32)
        T = players[c0:c1].astype(np.float32)
        present = ~np.isnan(D[:, :, 0]) & bro_valid[c0:c1, None]
        D = np.where(present[:, :, None], D, np.nan)

        def dist(Dx):
            d = np.sqrt(((Dx[:, :, None, :] - T[:, None, :, :]) ** 2).sum(-1))       # [c, S, slots]
            return np.where(np.isnan(d), np.inf, d)
        d = dist(D)
        if r["register"]:
            j = np.argmin(d, axis=2)
            dmin = np.take_along_axis(d, j[:, :, None], 2)[:, :, 0]
            nearest = np.take_along_axis(T, j[:, :, None].repeat(2, 2), 1)          # [c, S, 2]
            cand = dmin <= REG_REACH
            v = np.where(cand[:, :, None], nearest - D, np.nan)
            ok = cand.sum(1) >= REG_MIN
            with np.errstate(all="ignore"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    s = np.nanmedian(v, axis=1)
            s = np.where(ok[:, None], s, 0.0).astype(np.float32)
            D = D + s[:, None, :]
            d = dist(D)
            shift[c0:c1] = s
            if r["gate"] is not None:
                dmin2 = d.min(axis=2)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    res = np.nanmedian(np.where(cand, dmin2, np.nan), axis=1)
                bad = ok & (res > r["gate"])
                dropped[c0:c1] = bad
                present &= ~bad[:, None]
        near_same = ((d <= r["r_same"]) & same_team[None]).any(2)
        near_other = ((d <= r["r_other"]) & ~same_team[None]).any(2) if r["r_other"] > 0 else np.zeros_like(near_same)
        k = present & ~near_same & ~near_other
        if r.get("cap"):
            dsame = np.where(same_team[None], d, np.inf).min(2)                     # [c, S]
            for side in (HOME, AWAY):
                allowed = np.maximum(r["cap"] - (~np.isnan(T[:, team == side, 0])).sum(1), 0)
                cand = k & (bro_team == side)[None, :]
                order = np.argsort(np.where(cand, -dsame, np.inf), axis=1, kind="stable")
                rank = np.empty_like(order)
                np.put_along_axis(rank, order, np.broadcast_to(np.arange(S), order.shape).copy(), 1)
                k = k & ~(cand & (rank >= allowed[:, None]))
        if last_ball is not None:
            lb = last_ball[c0:c1]
            with np.errstate(invalid="ignore"):
                k &= np.hypot(D[:, :, 0] - lb[:, None, 0], D[:, :, 1] - lb[:, None, 1]) <= r["d"]
        keep[c0:c1] = k
        reg[c0:c1] = D
    # pieces: runs of kept frames of a broadcast slot within a half
    pieces = []
    for _, first, last in halves:
        span = keep[first:last + 1]
        for jj in range(S):
            for a, e in zip(*v1.runs(span[:, jj])):
                pieces.append((first + int(a), first + int(e), jj))
    pieces.sort()
    # each piece to the slot of its team free on [a - 1, e + 1) that has been free longest (degrade's rule)
    occupied = []                                                                # per slot: sorted (start, end)
    for k in range(players.shape[1]):
        a_, e_ = v1.runs(~np.isnan(players[:, k, 0]))
        occupied.append(list(zip(a_.tolist(), e_.tolist())))
    starts = [[iv[0] for iv in occ] for occ in occupied]
    new_team = list(team.tolist())
    placed = []
    for a, e, jj in pieces:
        side = int(bro_team[jj])
        best, best_end = None, None
        lo, hi = max(a - 1, 0), min(e + 1, n)
        for k in range(len(new_team)):
            if new_team[k] != side:
                continue
            i = bisect_left(starts[k], hi)                    # intervals starting before hi
            prev = occupied[k][i - 1] if i > 0 else None
            if prev is not None and prev[1] > lo:             # overlaps [lo, hi)
                continue
            end_before = prev[1] if prev is not None else -1
            if best is None or end_before < best_end:
                best, best_end = k, end_before
        if best is None:
            new_team.append(side)
            occupied.append([])
            starts.append([])
            best = len(new_team) - 1
        i = bisect_left(starts[best], a)
        occupied[best].insert(i, (a, e))
        starts[best].insert(i, a)
        placed.append((a, e, jj, best))
    out = np.full((n, len(new_team), 2), np.nan, np.float32)
    out[:, :players.shape[1]] = players
    for a, e, jj, k in placed:
        out[a:e, k] = reg[a:e, jj]
    info = {"pieces": len(placed), "appended_slots": len(new_team) - players.shape[1],
            "dropped_frames": int(dropped.sum()), "added_slot_frames": int(keep.sum()),
            "registered_frames": int((np.abs(shift).sum(1) > 0).sum())}
    return out, np.array(new_team, np.int8), keep, reg, info


def merge_broadcast(tracking, halves, paths=None, data_dir=None, rule=None):
    """(the tracking with the broadcast's extra players, status). Never prints; never raises. rule: RULE (M, the
    default) or RULE_M2."""
    try:
        if paths is None:
            from games import asset_paths
            p = asset_paths(data_dir=data_dir, source="broadcast")
            paths = {k: p[k] for k in ("atd_meta", "atd_raw", "homography")}
        import os
        if not all(os.path.exists(paths[k]) for k in ("atd_meta", "atd_raw", "homography")):
            return tracking, "fallback-missing"
        slot_team, _ = read_meta(paths["atd_meta"])
        bro_players, bro_ball = read_raw(paths["atd_raw"], len(slot_team))
        bro_valid = read_valid(paths["homography"], len(bro_players))
        ball = np.asarray(tracking["ball"], dtype=np.float64)
        shifts = []
        for _, first, last in halves:
            shift, rms, runner = global_shift(ball, bro_ball, first, last)        # as the tool: the whole ball track
            if not (np.isfinite(rms) and rms <= RMS_OK and runner >= RUNNER_RATIO * rms):
                return tracking, "fallback-alignment"
            shifts.append(shift)
        n = len(tracking["players"])
        players_b, valid_b = on_tactical_timeline(bro_players, bro_valid, shifts, halves, n)
        team_b = team_order(tracking, players_b, valid_b, slot_team)
        merged_players, merged_team, _, _, _ = merge(tracking["players"], tracking["team"], players_b, team_b, valid_b,
                                                     halves, rule)
        out = dict(tracking, players=merged_players, team=merged_team)
        if "speed" in tracking:
            speed = np.full(merged_players.shape[:2], np.nan, dtype=np.float32)
            speed[:, :tracking["speed"].shape[1]] = tracking["speed"]
            out["speed"] = speed
        return out, "merged"
    except Exception:            # any failure: the program runs as without the broadcast
        return tracking, "fallback-error"


def team_order(tracking, players_b, valid_b, slot_team, step=25):
    """The broadcast's labels, flipped if its first team is the tactical tracking's second: on every step-th valid
    frame, broadcast players within TEAM_CHECK_M of a tactical player vote same / other label."""
    T, team = tracking["players"], np.asarray(tracking["team"])
    same = other = 0
    for t in np.flatnonzero(valid_b)[::step]:
        d = np.hypot(players_b[t, :, None, 0] - T[t, None, :, 0], players_b[t, :, None, 1] - T[t, None, :, 1])
        d = np.where(np.isnan(d), np.inf, d)
        j = d.argmin(1)
        hit = d[np.arange(len(j)), j] <= TEAM_CHECK_M
        same += int((slot_team[hit] == team[j[hit]]).sum())
        other += int((slot_team[hit] != team[j[hit]]).sum())
    return slot_team if same >= other else (1 - slot_team).astype(np.int8)
