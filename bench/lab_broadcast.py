"""Push track 3, "broadcast": would players seen by the broadcast tracking near the ball, merged into the tactical
tracking, improve branch B? (docs/lab/push/broadcast.md)

On each bench match a broadcast view is simulated from the CLEAN tracking (external data): a camera window that
follows the true ball with a lag, blocks of invalid frames (replays, close-ups) mostly after stoppages, tracklets
broken into anonymous slots, position noise, team-label errors. The simulation's settings (central, optimistic,
pessimistic, and an `ideal` diagnostic) only use the statistics docs/data.md gives for our match's broadcast set; they
are fixed on the page before any score. The merge rule then adds broadcast players the tactical tracking lacks; after
it, v8's repair and every input computed from the tracking are recomputed and v15's branch B runs, with the
machinery of bench/lab_trackerr.py (b16_c0's models, read only), on B's bench.

    python -m bench.lab_broadcast calibrate                       # the window size and tracklet break rate per setting
    python -m bench.lab_broadcast stats SETTING [--matches ...]    # the simulated view and the merge, counted (no score)
    python -m bench.lab_broadcast setup SETTING                    # an isolated copy of the repo for one setting
    # in that copy (scratchpad/broadcast/wc_SETTING):
    python -m bench.lab_broadcast build SETTING --matches ...      # the merged lab cache (repair, every input recomputed)
    python -m bench.lab_broadcast score SETTING --matches ...      # features, augment's and gru_b's P, trial 1015, scores
    python -m bench.lab_broadcast report                           # every setting against `off`; the reproduction check
    python -m bench.lab_broadcast clean SETTING                    # delete the copy's bulky caches once scored
    python -m bench.lab_broadcast port                             # possession/broadcast_merge.py end to end (bench data)

SETTING (harness): off (no merge: must reproduce v15's B to 1e-5), central, optimistic, pessimistic, ideal, literal.
`--models DIR` (build and score) runs another copy's trained models instead of b16_c0's (the judge's retrained B).
Nothing here reads our match.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import time
import zlib
from bisect import bisect_left

import numpy as np
import pandas as pd

from bench import lab, sets
from possession import v1
from possession.data import AWAY, FPS, HOME

# --------------------------------------------------------------------------------------------
# Params (fixed on docs/lab/push/broadcast.md before any score)
# --------------------------------------------------------------------------------------------

SEED = 20261115
WHICH = "all"
TRIAL = 1015
VARIANTS = ("p25", "p50", "stressed")
NEAR_M = 15.0                 # report: an added piece this near the true ball is "near" (o_near's radius)
ASPECT = 1.0                  # the window's extent across the pitch (y) over its extent along it (x)
ZOOM_TAU_S = 10.0             # s: the zoom's AR(1) time constant
OFFSET_TAU_S = 2.0            # s: the homography's common offset, AR(1)
SCALE_TAU_S = 5.0             # s: its scale/shear error, AR(1)
GAP_MEAN_FRAMES = 8           # a tracklet break leaves the player undetected this long on average (geometric)
MIN_TRACKLET = 25             # frames: shorter pieces are not reported by the tracker (1 s)
BRIDGE_FRAMES = 12            # a person out of the window for at most this long (valid frames) stays tracked: edge flicker
WINDOW_FRAMES = 3000          # the time-alignment residual is drawn per 2-minute window (align_broadcast's windows)
OUTLIER_EPISODE_S = 2.0       # s: a homography outlier episode
OUTLIER_SD_M = 5.0            # m: its common offset, per axis
REFEREE_EPISODE_S = 20.0      # s: the false person (pessimistic) is reported in episodes this long
SLOTS = {HOME: 26, AWAY: 25}  # the broadcast's slot counts (docs/data.md)

# The simulated broadcast views. target_valid: mean players inside the window per VALID frame, calibrated by the
# window's size (changed 11:05, before any score: see the page); invalid: share of the halves' frames invalid;
# anchored: share of the invalid frames in blocks anchored at stoppages (the rest at random times); track_s: mean
# tracklet length; tau_s: camera lag; noise per axis (m): sd_h common homography offset, sd_p per tracklet, sd_j per
# frame, sd_e scale/shear error (m per m from the camera's centre); dt_frames: time-alignment residual, uniform in
# [-dt, dt] per window; off40: share of windows off by 40 frames; outlier: share of valid frames in homography outlier
# episodes; team_err: share of tracklets with the wrong team; referee: share of valid frames on which a false person
# (a referee) is reported near the ball, team label random per tracklet.
SIM = {
    "central": dict(target_valid=11.5, invalid=0.34, anchored=0.75, track_s=16.0, tau_s=0.5, zoom_sd=0.15,
                    sd_h=0.55, sd_p=0.30, sd_j=0.15, sd_e=0.010, dt_frames=2, off40=0.0, outlier=0.0,
                    team_err=0.01, referee=0.0),
    "optimistic": dict(target_valid=17.0, invalid=0.30, anchored=0.85, track_s=20.0, tau_s=0.3, zoom_sd=0.15,
                       sd_h=0.30, sd_p=0.20, sd_j=0.10, sd_e=0.005, dt_frames=0, off40=0.0, outlier=0.0,
                       team_err=0.005, referee=0.0),
    "pessimistic": dict(target_valid=9.0, invalid=0.40, anchored=0.50, track_s=10.0, tau_s=1.0, zoom_sd=0.25,
                        sd_h=1.00, sd_p=0.50, sd_j=0.30, sd_e=0.020, dt_frames=2, off40=0.04, outlier=0.03,
                        team_err=0.025, referee=0.20),
    # DIAGNOSTIC (not a candidate): central's camera, every frame valid, no noise, no team errors, no misalignment
    "ideal": dict(target_valid=None, invalid=0.0, anchored=0.0, track_s=16.0, tau_s=0.5, zoom_sd=0.15,
                  sd_h=0.0, sd_p=0.0, sd_j=0.0, sd_e=0.0, dt_frames=0, off40=0.0, outlier=0.0,
                  team_err=0.0, referee=0.0),
}
# DIAGNOSTICS (added 12:50, after `pessimistic` was scored; written on the page before they run): pessimistic's draws
# (seed_as) with one element reverted at a time, to say which part of its loss comes from what.
SIM["pess_noref"] = dict(SIM["pessimistic"], referee=0.0, seed_as="pessimistic")
SIM["pess_noref_cnoise"] = dict(SIM["pessimistic"], referee=0.0, outlier=0.0, seed_as="pessimistic",
                                **{k: SIM["central"][k] for k in ("sd_h", "sd_p", "sd_j", "sd_e")})
# A second draw of the pessimistic view (added 14:40, before it runs): the same settings and window, other random
# streams; for how much the robustness verdicts depend on the draw
SIM["pessimistic2"] = dict(SIM["pessimistic"], seed_as="pessimistic2", calib_as="pessimistic")
# Invalid blocks anchored at a stoppage (a run of true D frames of at least ANCHOR_MIN_S): from its start + U(delay)
# to its end + U(overrun); at least BLOCK_MIN_S long. Random blocks: lognormal length (median, sigma of the log).
ANCHOR_MIN_S = 4.0
ANCHOR_DELAY_S = (1.0, 3.0)
ANCHOR_OVERRUN_S = (-3.0, 1.0)
BLOCK_MIN_S = 2.0
RANDOM_BLOCK_S = (5.0, 0.5)

# The merge rules. M (the candidate): per valid frame, register the broadcast on the tactical players (the median
# vector from each detection to its nearest tactical player, any team, over detections with one within REG_REACH m,
# if at least REG_MIN of them), drop the frame if the median residual after it is above `gate` m, then add each
# detection with no tactical player of its own team within r_same m and none of the other team within r_other m.
# literal (diagnostic): the brief's rule as written: no registration, same team within 3 m only.
RULES = {"M": dict(register=True, gate=1.5, r_same=2.5, r_other=1.5, d=None),
         "literal": dict(register=False, gate=None, r_same=3.0, r_other=0.0, d=None),
         # M2 (post hoc, 13:05, after `pessimistic`): M, and a team never has more than `cap` (its tactical players on
         # the frame plus those added); the added ones farthest from any tactical teammate first
         "M2": dict(register=True, gate=1.5, r_same=2.5, r_other=1.5, d=None, cap=11)}
REG_REACH = 5.0
REG_MIN = 3

# The harness settings: (simulated view, merge rule); `off` is no merge at all.
SETTINGS = {"off": None, "central": ("central", "M"), "optimistic": ("optimistic", "M"),
            "pessimistic": ("pessimistic", "M"), "ideal": ("ideal", "M"), "literal": ("central", "literal"),
            "pess_noref": ("pess_noref", "M"), "pess_noref_cnoise": ("pess_noref_cnoise", "M"),
            "central_M2": ("central", "M2"), "pessimistic_M2": ("pessimistic", "M2"),
            "pessimistic2": ("pessimistic2", "M"), "pessimistic2_M2": ("pessimistic2", "M2"),
            "optimistic_M2": ("optimistic", "M2")}

MAIN = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRATCH = "/private/tmp/claude-501/-Users-joseruiz-Code-more-than-a-hack-2027/39e3effe-3bf8-4611-8584-0402cfeb00b7/scratchpad"
B16 = os.path.join(SCRATCH, "b16_c0")
WORK = os.path.join(SCRATCH, "broadcast")
SHARED = os.path.join(MAIN, "data", "external", "lab_broadcast")          # gitignored (data/*)
RESULTS = os.path.join(MAIN, "bench", "results", "lab_broadcast")
CONFIRM = os.path.join(MAIN, "bench", "results", "search_b", "confirm_1015_554_586_1133_1269_-1_-2.csv")
CALIB = os.path.join(SHARED, "calibration.json")
LETTER_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def rng_for(*parts):
    return np.random.default_rng([SEED] + [zlib.crc32(str(p).encode()) for p in parts])


def b16_lab(m):
    return os.path.join(B16, "data", "external", "lab", "{}.npz".format(m))


def lab_inputs(m):
    """The bench's damaged tracking as v15's B reads it (b16_c0's lab cache, read only)."""
    with np.load(b16_lab(m)) as z:
        return {k: z[k] for k in z.files if k not in ("players_raw", "team_raw")}


def in_halves(n, halves):
    out = np.zeros(n, bool)
    for _, first, last in halves:
        out[first:last + 1] = True
    return out


# --------------------------------------------------------------------------------------------
# The simulated broadcast view
# --------------------------------------------------------------------------------------------

def ar1(rng, n, halves, sd, tau_s, dims):
    """A stationary AR(1) process per half, [n, dims], standard deviation sd, time constant tau_s."""
    out = np.zeros((n, dims))
    if sd <= 0:
        return out
    a = np.exp(-1.0 / (tau_s * FPS))
    from scipy.signal import lfilter
    for _, first, last in halves:
        k = last + 1 - first
        e = rng.normal(0.0, sd * np.sqrt(1 - a * a), (k, dims))
        e[0] = rng.normal(0.0, sd, dims)
        out[first:last + 1] = lfilter([1.0], [1.0, -a], e, axis=0)
    return out


def follow(ball, halves, tau_s):
    """The camera's centre: the ball followed with an exponential lag; held where the ball is missing, the half's first
    seen ball before it, the pitch's centre if a half has none."""
    out = np.full(ball.shape, np.nan)
    a = 1 - np.exp(-1.0 / (tau_s * FPS))
    b = ball.astype(np.float64)
    for _, first, last in halves:
        seen = np.flatnonzero(~np.isnan(b[first:last + 1, 0]))
        c = b[first + seen[0]].copy() if len(seen) else np.array([52.5, 34.0])
        x0, x1 = c
        bx, by = b[first:last + 1, 0].tolist(), b[first:last + 1, 1].tolist()
        ox, oy = [0.0] * len(bx), [0.0] * len(bx)
        for i in range(len(bx)):
            if bx[i] == bx[i]:          # not NaN
                x0 += a * (bx[i] - x0)
                x1 += a * (by[i] - x1)
            ox[i], oy[i] = x0, x1
        out[first:last + 1, 0], out[first:last + 1, 1] = ox, oy
    return out


def time_source(n, halves, sim, rng):
    """The clean frame the broadcast shows at each frame: t + the alignment residual of t's 2-minute window."""
    src = np.arange(n)
    for _, first, last in halves:
        for start in range(first, last + 1, WINDOW_FRAMES):
            end = min(start + WINDOW_FRAMES, last + 1)
            d = int(rng.integers(-sim["dt_frames"], sim["dt_frames"] + 1)) if sim["dt_frames"] else 0
            if sim["off40"] and rng.random() < sim["off40"]:
                d = 40
            src[start:end] = np.clip(np.arange(start, end) + d, first, last)
    return src


def invalid_mask(truth, halves, sim, rng):
    """Frames the broadcast doesn't show the pitch on (replays, close-ups, crowd)."""
    n = len(truth)
    inh = in_halves(n, halves)
    frames = int(inh.sum())
    target = sim["invalid"] * frames
    bad = np.zeros(n, bool)
    if target <= 0:
        return bad
    blocks = []
    for _, first, last in halves:
        dead = truth[first:last + 1] == 2
        for a, e in zip(*v1.runs(dead)):
            if e - a < ANCHOR_MIN_S * FPS:
                continue
            s = first + a + int(rng.uniform(*ANCHOR_DELAY_S) * FPS)
            t = min(first + e + int(rng.uniform(*ANCHOR_OVERRUN_S) * FPS), last + 1)
            if t - s >= BLOCK_MIN_S * FPS:
                blocks.append((s, t))
    order = rng.permutation(len(blocks))
    for i in order:
        if bad.sum() >= sim["anchored"] * target:
            break
        s, t = blocks[i]
        bad[s:t] = True
    spans = [(first, last) for _, first, last in halves]
    lengths = np.array([last + 1 - first for first, last in spans], float)
    while bad.sum() < target:
        first, last = spans[int(rng.choice(len(spans), p=lengths / lengths.sum()))]
        length = max(int(np.exp(rng.normal(np.log(RANDOM_BLOCK_S[0]), RANDOM_BLOCK_S[1])) * FPS), int(BLOCK_MIN_S * FPS))
        s = int(rng.integers(first, last + 1))
        bad[s:min(s + length, last + 1)] = True
    return bad & inh


def view_geometry(m, name, clean, truth, halves):
    """Everything about the view that doesn't depend on the window's size: (normalised distance to the window's edge
    [n, P] (inside iff <= W0 / 2), valid [n], src [n], camera [n, 2], zoom factor [n])."""
    sim = SIM[name]
    n = len(truth)
    key = sim.get("seed_as", name)
    src = time_source(n, halves, sim, rng_for(m, key, "time"))
    bad = invalid_mask(truth, halves, sim, rng_for(m, key, "invalid"))
    valid = in_halves(n, halves) & ~bad
    cam = follow(clean["ball"][src, :2], halves, sim["tau_s"])
    zoom = np.exp(ar1(rng_for(m, key, "zoom"), n, halves, sim["zoom_sd"], ZOOM_TAU_S, 1)[:, 0])
    pos = clean["players"][src].astype(np.float64)
    with np.errstate(invalid="ignore"):
        a = np.maximum(np.abs(pos[:, :, 0] - cam[:, None, 0]), np.abs(pos[:, :, 1] - cam[:, None, 1]) / ASPECT)
    a = a / zoom[:, None]
    a[~valid] = np.inf
    a[np.isnan(a)] = np.inf
    return a.astype(np.float32), valid, src, cam, zoom


def bridged(inview, valid):
    """The in-view mask with out-of-window gaps of at most BRIDGE_FRAMES valid frames closed (a tracker keeps a player
    who flickers across the picture's edge)."""
    out = inview.copy()
    for p in range(inview.shape[1]):
        a, e = v1.runs(~inview[:, p])
        for s, t in zip(a, e):
            if 0 < s and t < len(valid) and t - s <= BRIDGE_FRAMES and valid[s:t].all():
                out[s:t, p] = True
    return out


def tracklets_of(inview, halves, lam, rng):
    """Cut every person's in-view runs into tracklets: breaks at rate lam per frame, each followed by a geometric gap.
    [(start, end exclusive, person)]."""
    out = []
    u = rng.random(4_000_000)
    k = 0
    for p in range(inview.shape[1]):
        for a, e in zip(*v1.runs(inview[:, p])):
            t = int(a)
            while t < e:
                if lam > 0:
                    length = int(np.ceil(np.log(1 - u[k % len(u)]) / np.log(1 - lam)))
                    k += 1
                else:
                    length = e - t
                end = min(t + max(length, 1), int(e))
                if end - t >= MIN_TRACKLET:
                    out.append((t, end, p))
                gap = int(np.ceil(np.log(1 - u[k % len(u)]) / np.log(1 - 1.0 / GAP_MEAN_FRAMES)))
                k += 1
                t = end + gap
    return out


def calibrate():
    """The window's base size W0 (m) per setting, such that the mean count inside over all frames of the halves is
    target_valid (per valid frame); then the break rate per setting, such that the mean tracklet length is track_s (pooled, 10 matches)."""
    ms = lab.matches(WHICH)
    out = {}
    for name, sim in SIM.items():
        if sim["target_valid"] is None:
            continue
        values, frames, vframes, geo = [], 0, 0, {}
        for m in ms:
            clean = sets.load(m)
            f = lab_inputs(m)
            halves = [tuple(int(x) for x in h) for h in f["halves"]]
            a, valid, src, cam, zoom = view_geometry(m, name, clean, f["truth"], halves)
            frames += int(in_halves(len(a), halves).sum())
            vframes += int(valid.sum())
            values.append(a[np.isfinite(a)])
            geo[m] = (a, valid, halves)
        allv = np.sort(np.concatenate(values))
        want = int(round(sim["target_valid"] * vframes))
        w0 = float(2 * allv[min(want, len(allv) - 1)])
        lo, hi = 0.0, 0.01
        for _ in range(14):
            lam = (lo + hi) / 2
            lengths = []
            for m, (a, valid, halves) in geo.items():
                inview = bridged(a <= w0 / 2, valid)
                lengths += [e - s for s, e, _ in tracklets_of(inview, halves, lam, rng_for(m, name, "tracklets"))]
            mean_s = float(np.mean(lengths)) / FPS
            if mean_s > sim["track_s"]:
                lo = lam
            else:
                hi = lam
        out[name] = {"w0": w0, "lam": float(lam), "mean_tracklet_s": mean_s, "frames": frames, "valid_frames": vframes,
                     "inside_valid": float(np.sum(allv <= w0 / 2) / vframes),
                     "inside_all": float(np.sum(allv <= w0 / 2) / frames)}
        print(name, out[name], flush=True)
        del geo, values, allv
    out["ideal"] = dict(out["central"])
    os.makedirs(SHARED, exist_ok=True)
    with open(CALIB, "w") as handle:
        json.dump(out, handle, indent=1)


def calibration(name):
    with open(CALIB) as handle:
        return json.load(handle)[SIM[name].get("calib_as", SIM[name].get("seed_as", name))]


def simulate(m, name, clean, truth, halves):
    """The broadcast tracking of one bench match, as the broadcast files would give it after alignment:
    players [n, S, 2] (NaN where not reported), team [S], valid [n] (homography valid), and for counting only
    (never read by the merge) the true person behind each slot-frame (person [n, S], -1 none, -2 the false person)."""
    sim, cal = SIM[name], calibration(name)
    key = sim.get("seed_as", name)
    n = len(truth)
    a, valid, src, cam, zoom = view_geometry(m, name, clean, truth, halves)
    w0 = cal["w0"]
    inview = bridged(a <= w0 / 2, valid)
    del a
    pos = clean["players"][src].astype(np.float64)
    ctruth_team = np.asarray(clean["team"])
    tr = tracklets_of(inview, halves, cal["lam"], rng_for(m, key, "tracklets"))
    people = [(s, e, p, int(ctruth_team[p])) for s, e, p in tr]
    fake = None
    if sim["referee"] > 0:                       # a false person near the ball (a referee), in episodes
        rng = rng_for(m, key, "referee")
        ball = follow(clean["ball"][src, :2], halves, 2.0)
        theta = np.cumsum(rng.normal(0, 0.02, n))
        fake = ball + 10.0 * np.c_[np.cos(theta), np.sin(theta)]
        on = np.zeros(n, bool)
        ep = int(REFEREE_EPISODE_S * FPS)
        target = sim["referee"] * valid.sum()
        while (on & valid).sum() < target:
            s = int(rng.integers(0, n))
            on[s:s + ep] = True
        with np.errstate(invalid="ignore"):
            inside = (np.maximum(np.abs(fake[:, 0] - cam[:, 0]), np.abs(fake[:, 1] - cam[:, 1]) / ASPECT) / zoom
                      <= w0 / 2) & valid & on
        for s, e, _ in tracklets_of(inside[:, None], halves, cal["lam"], rng_for(m, key, "referee_tracklets")):
            people.append((s, e, -2, int(rng.integers(0, 2))))
    # noise: the homography's common offset and scale/shear error, outlier episodes, per tracklet, per frame
    common = ar1(rng_for(m, key, "offset"), n, halves, sim["sd_h"], OFFSET_TAU_S, 2)
    shear = ar1(rng_for(m, key, "shear"), n, halves, sim["sd_e"], SCALE_TAU_S, 4).reshape(n, 2, 2)
    if sim["outlier"] > 0:
        rng = rng_for(m, key, "outlier")
        ep = int(OUTLIER_EPISODE_S * FPS)
        hit = np.zeros(n, bool)
        while (hit & valid).sum() < sim["outlier"] * valid.sum():
            s = int(rng.integers(0, n))
            hit[s:s + ep] = True
            common[s:s + ep] += rng.normal(0, OUTLIER_SD_M, 2)
    rng_p, rng_j, rng_t = rng_for(m, key, "bias"), rng_for(m, key, "jitter"), rng_for(m, key, "team")
    people.sort()
    # slots: each labelled team's tracklets dealt to the slot free longest (degrade's rule), appended if all busy
    free = {HOME: [-1] * SLOTS[HOME], AWAY: [-1] * SLOTS[AWAY]}
    assigned = []
    for s, e, p, t in people:
        label = t if p == -2 or rng_t.random() >= sim["team_err"] else 1 - t
        k = int(np.argmin(free[label]))
        if free[label][k] > s:
            free[label].append(-1)
            k = len(free[label]) - 1
        free[label][k] = e
        assigned.append((s, e, p, t, label, k))
    order = [(HOME, k) for k in range(len(free[HOME]))] + [(AWAY, k) for k in range(len(free[AWAY]))]
    column = {key: i for i, key in enumerate(order)}
    S = len(order)
    players = np.full((n, S, 2), np.nan, np.float32)
    person = np.full((n, S), -1, np.int16)
    for s, e, p, t, label, k in assigned:
        j = column[(label, k)]
        truth_xy = fake[s:e] if p == -2 else pos[s:e, p]
        rel = truth_xy - cam[s:e]
        xy = (truth_xy + common[s:e] + np.einsum("tij,tj->ti", shear[s:e], rel)
              + rng_p.normal(0, sim["sd_p"], 2) + rng_j.normal(0, sim["sd_j"], (e - s, 2)))
        players[s:e, j] = xy.astype(np.float32)
        person[s:e, j] = p
    team = np.array([side for side, _ in order], np.int8)
    return {"players": players, "team": team, "valid": valid, "person": person, "src": src}


# --------------------------------------------------------------------------------------------
# The merge (deployable: it reads only the two trackings and the broadcast's validity)
# --------------------------------------------------------------------------------------------

def merge(players, team, bro_players, bro_team, bro_valid, halves, rule, ball=None, chunk=4000):
    """The tactical tracking with the broadcast's extra players added as pieces in free slots of their team.
    Returns (players, team, added [n, S_bro] bool, registered broadcast positions [n, S_bro, 2], info)."""
    r = RULES[rule] if isinstance(rule, str) else rule
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
                import warnings
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


# --------------------------------------------------------------------------------------------
# Counting what the simulation and the merge do (needs the truth: diagnostics only, never an input)
# --------------------------------------------------------------------------------------------

def seen_persons(clean_players, src, players, chunk=4000):
    """[n, P]: True where the person's clean position at frame t is in the tactical tracking at t (exactly)."""
    n, P = clean_players.shape[:2]
    out = np.zeros((n, P), bool)
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        cp = clean_players[c0:c1].astype(np.float32)
        T = players[c0:c1]
        d = np.abs(cp[:, :, None, :] - T[:, None, :, :]).sum(-1)
        out[c0:c1] = (np.nan_to_num(d, nan=np.inf) < 1e-3).any(2)
    return out


def describe(m, name, rule, clean, f, bro, keep, reg):
    """The view's statistics (as docs/data.md gives them for our match) and what the merge added."""
    halves = [tuple(int(x) for x in h) for h in f["halves"]]
    n = len(f["truth"])
    inh = in_halves(n, halves)
    frames = int(inh.sum())
    det = ~np.isnan(bro["players"][:, :, 0])
    valid = bro["valid"]
    dead = (f["truth"] == 2) & inh
    lengths = []
    for jj in range(det.shape[1]):
        a, e = v1.runs(det[:, jj] & inh)
        # a slot's run may hold two tracklets back to back: split where the person changes
        for s, t in zip(a, e):
            p = bro["person"][s:t, jj]
            cuts = np.flatnonzero(np.diff(p) != 0) + 1
            lengths += np.diff(np.r_[0, cuts, t - s]).tolist()
    person = bro["person"]
    real = person >= 0
    cp = clean["players"].astype(np.float64)
    pidx = np.where(real, person, 0)
    tt = np.arange(n)[:, None].repeat(det.shape[1], 1)
    truth_xy = cp[bro["src"][tt], pidx]
    err = np.hypot(*(bro["players"].astype(np.float64) - truth_xy).transpose(2, 0, 1))
    err_reg = np.hypot(*(reg.astype(np.float64) - truth_xy).transpose(2, 0, 1))
    team_true = np.asarray(clean["team"])[pidx]
    wrong = real & det & (bro["team"][None, :] != team_true)
    # was the detection's person seen in the tactical tracking at the same (tactical) frame?
    seen_now = seen_persons(clean["players"], np.arange(n), f["players"])
    seen_det_now = np.take_along_axis(seen_now, pidx, 1) & real
    added = keep
    ball = clean["ball"][:, :2].astype(np.float64)
    with np.errstate(invalid="ignore"):
        near_true = np.hypot(cp[:, :, 0] - ball[:, None, 0], cp[:, :, 1] - ball[:, None, 1]) <= NEAR_M
    on = ~np.isnan(cp[:, :, 0]) & inh[:, None]
    missing_near = on & ~seen_now & near_true                                  # o_near's set
    covered = np.zeros_like(missing_near)
    good = added & real & ~wrong
    ti, sj = np.nonzero(good)
    pp = person[ti, sj]
    close = np.hypot(*(reg[ti, sj].astype(np.float64) - cp[ti, pp]).T) <= 2.0
    covered[ti[close], pp[close]] = True
    out = {
        "frames": frames, "valid_share": float(valid[inh].mean()), "invalid_in_dead": float(dead[~valid & inh].mean()),
        "players_per_frame_all": float(det[inh].sum() / frames),
        "players_per_valid_frame": float(det[valid].sum() / max(valid.sum(), 1)),
        "tracklets": len(lengths), "tracklet_mean_s": float(np.mean(lengths) / FPS) if lengths else 0.0,
        "error_median_m": float(np.median(err[det & real])) if (det & real).any() else 0.0,
        "error_p90_m": float(np.percentile(err[det & real], 90)) if (det & real).any() else 0.0,
        "error_registered_median_m": float(np.nanmedian(err_reg[det & real & valid[:, None]])) if (det & real).any() else 0.0,
        "wrong_team_share": float(wrong[det].mean()) if det.any() else 0.0,
        "false_person_share": float((person[det] == -2).mean()) if det.any() else 0.0,
        "tactical_missing_per_frame": float((on & ~seen_now).sum() / frames),
        "missing_near_per_frame": float(missing_near.sum() / frames),
        "added_per_frame": float(added[inh].sum() / frames),
        "added_per_valid_frame": float(added[valid].sum() / max(valid.sum(), 1)),
        "added_missing_right_team": float((added & real & ~seen_det_now & ~wrong).sum() / frames),
        "added_missing_wrong_team": float((added & real & ~seen_det_now & wrong).sum() / frames),
        "added_duplicate_right_team": float((added & real & seen_det_now & ~wrong).sum() / frames),
        "added_duplicate_wrong_team": float((added & real & seen_det_now & wrong).sum() / frames),
        "added_false_person": float((added & (person == -2)).sum() / frames),
        "near_missing_covered": float(covered[missing_near].mean()) if missing_near.any() else 0.0,
    }
    return out


# --------------------------------------------------------------------------------------------
# The merged tracking of one match, and the stats command
# --------------------------------------------------------------------------------------------

def merged_tracking(m, setting, clean=None, f=None, with_info=True):
    """(players, team, zone [n], info) for one bench match under a harness setting (None: the delivered tracking)."""
    f = f if f is not None else lab_inputs(m)
    halves = [tuple(int(x) for x in h) for h in f["halves"]]
    if SETTINGS[setting] is None:
        return f["players"], f["team"], np.zeros(len(f["truth"]), np.int8), {"setting": setting}
    name, rule = SETTINGS[setting]
    clean = clean if clean is not None else sets.load(m)
    bro = simulate(m, name, clean, f["truth"], halves)
    players, team, keep, reg, minfo = merge(f["players"], f["team"], bro["players"], bro["team"], bro["valid"],
                                            halves, rule)
    ball = clean["ball"][:, :2].astype(np.float64)
    zone = np.zeros(len(players), np.int8)
    t_idx, j_idx = np.nonzero(keep)
    with np.errstate(invalid="ignore"):
        near = np.hypot(*(reg[t_idx, j_idx].astype(np.float64) - ball[t_idx]).T) <= NEAR_M
    zone[t_idx] = np.maximum(zone[t_idx], 1)
    zone[t_idx[near]] = 2
    info = {"setting": setting, "sim": name, "rule": rule, "merge": minfo}
    if with_info:
        info["view"] = describe(m, name, rule, clean, f, bro, keep, reg)
    return players, team, zone, info


def stats(setting, matches):
    os.makedirs(os.path.join(SHARED, setting), exist_ok=True)
    rows = {}
    for m in matches:
        started = time.time()
        _, _, zone, info = merged_tracking(m, setting)
        info["zone_near_share"] = float((zone == 2).mean())
        rows[m] = info
        with open(os.path.join(SHARED, setting, "stats_{}.json".format(m)), "w") as handle:
            json.dump(info, handle, indent=1)
        v = info["view"]
        print("{} {}: {:.1f} players/frame ({:.1f} valid), valid {:.1%} ({:.0%} of invalid in dead), tracklets {:.1f} s, "
              "error {:.2f} m (registered {:.2f}), wrong team {:.1%}; added {:.2f}/frame: missing {:.3f} + wrong team "
              "{:.3f}, duplicates {:.3f} + wrong {:.3f}, false {:.3f}; near-ball missing covered {:.1%} ({:.0f} s)".format(
                  setting, m, v["players_per_frame_all"], v["players_per_valid_frame"], v["valid_share"],
                  v["invalid_in_dead"], v["tracklet_mean_s"], v["error_median_m"], v["error_registered_median_m"],
                  v["wrong_team_share"], v["added_per_frame"], v["added_missing_right_team"], v["added_missing_wrong_team"],
                  v["added_duplicate_right_team"], v["added_duplicate_wrong_team"], v["added_false_person"],
                  v["near_missing_covered"], time.time() - started), flush=True)
    keys = list(next(iter(rows.values()))["view"].keys())
    mean = {k: float(np.mean([r["view"][k] for r in rows.values()])) for k in keys}
    print("{} mean of {}: {}".format(setting, len(rows), {k: round(v, 4) for k, v in mean.items()}), flush=True)


# --------------------------------------------------------------------------------------------
# The harness (lab_trackerr's machinery, in an isolated copy of b16_c0)
# --------------------------------------------------------------------------------------------

def setting_dir(setting, tag=""):
    return os.path.join(SHARED, setting + tag)


def check_copy():
    if not os.path.realpath(lab.ROOT).startswith(os.path.realpath(WORK)):
        raise RuntimeError("runs only inside a setting's copy (scratchpad/broadcast/wc_*)")


def build(setting, matches, tag=""):
    """The setting's lab cache (in this copy's data/external/lab), its zones and what the merge did (shared)."""
    import joblib
    from bench import lab_trackerr as TE
    from possession import repair
    check_copy()
    os.makedirs(lab.folder(WHICH), exist_ok=True)
    out = setting_dir(setting, tag)
    os.makedirs(out, exist_ok=True)
    for m in matches:
        started = time.time()
        f = lab_inputs(m)
        halves, pitch = TE.halves_pitch(f)
        players, team, zone, info = merged_tracking(m, setting, f=f, with_info=setting != "off")
        track = TE.as_track(players, team, f["ball"])
        fixed, rpieces = repair.repair(track, halves, pitch)                                  # v8, as v12.inputs
        info["repair_fired"] = int(sum(p["repaired"] for p in rpieces))
        fold = joblib.load(TE.fold_path(m))
        pred, v9l, v11l = TE.inputs_of(fixed, halves, pitch, fold)
        if setting == "off":
            info["check"] = {k: TE.max_diff(pred[k], f["pred_" + k]) for k in pred}
            info["check"].update(v9=int(np.sum(v9l != f["v9"])), v11=int(np.sum(v11l != f["v11"])),
                                 players=bool(np.array_equal(fixed["players"], f["players"], equal_nan=True)),
                                 team=bool(np.array_equal(fixed["team"], f["team"])))
            print("{}: check against b16_c0's lab cache {}".format(m, info["check"]), flush=True)
        dtypes = {k[len("pred_"):]: f[k].dtype for k in f if k.startswith("pred_")}
        np.savez_compressed(os.path.join(lab.folder(WHICH), "{}.npz".format(m)),
                            players=fixed["players"].astype(np.float32), ball=f["ball"],
                            team=np.asarray(fixed["team"]).astype(np.int8), truth=f["truth"], halves=f["halves"],
                            pitch=f["pitch"], v9=np.asarray(v9l).astype(f["v9"].dtype),
                            v11=np.asarray(v11l).astype(f["v11"].dtype),
                            **{"pred_" + k: np.asarray(v).astype(dtypes[k]) for k, v in pred.items()})
        np.save(os.path.join(out, "zone_{}.npy".format(m)), zone)
        with open(os.path.join(out, "info_{}.json".format(m)), "w") as handle:
            json.dump(info, handle, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        print("{}{} {}: lab cache written; merge {}; repair fired on {} piece(s) ({:.0f} s)".format(
            setting, tag, m, info.get("merge"), info["repair_fired"], time.time() - started), flush=True)


V16_ROOT = os.path.join(MAIN, "data", "external", "lab_netmet")       # v16's B (docs/models/v16-netmet.md), read only
V16_SEEDS = (0, 1, 2, 3)
V16_CONFIRM = os.path.join(MAIN, "bench", "results", "lab_netmet", "ens4", "confirm_1015.csv")
# Track 2's finalist W4 (docs/lab/push/netweight.md), read only; tag "_w4" (added 2026-11-16 by the coordinator: the
# merge's no-retrain runs repeated on W4 before any swap). Same four seeds, same normalisers, its own trial-1015 tables.
W4_ROOT = os.path.join(MAIN, "data", "external", "lab_netweight")
W4_CONFIRM = os.path.join(MAIN, "bench", "results", "lab_netweight", "W4_ens4", "confirm_1015.csv")


def is_w4(tag):
    return "_w4" in tag


def is_v16(tag):
    """B as four averaged GRUs: v16's (netmet5) or, with "_w4", track 2's W4."""
    return "_v16" in tag or is_w4(tag)


def v16_nets(m, tag="_v16"):
    if is_w4(tag):
        return [os.path.join(W4_ROOT, "W4_s{}".format(s), "nets", "net_{}.pt".format(m)) for s in V16_SEEDS]
    return [os.path.join(V16_ROOT, "main" if s == 0 else "seed{}".format(s), "nets", "net_{}.pt".format(m)) for s in V16_SEEDS]


def v16_tables(tag):
    if is_w4(tag):
        return os.path.join(W4_ROOT, "W4_ens4", "search_b", "tables_trial{}.joblib".format(TRIAL))
    return os.path.join(V16_ROOT, "ens4", "search_b", "tables_trial{}.joblib".format(TRIAL))


def v16_average(ps):
    """lab_netmirror.average: each member through float32 (as stored), the mean, clipped and rounded as every P."""
    from bench import lab_robust
    L = lab_robust.L
    mean = np.mean([np.asarray(p, np.float32).astype(np.float64) for p in ps], axis=0)
    return np.round(np.clip(mean, *L.CLIP), L.DECIMALS)


def score(setting, matches, tag=""):
    """lab_trackerr's score: features, augment's and gru_b's P on B's 3 variants, trial 1015's chain, scores."""
    from bench import lab_branch_b as B
    from bench import lab_carried as C
    from bench import lab_gate, lab_temporal, search_b
    from bench.run import score as bench_score
    check_copy()
    for t in (0, TRIAL):
        if not os.path.exists(os.path.join(search_b.CACHE, "tables_trial{}.joblib".format(t))):
            raise RuntimeError("tables_trial{}.joblib must be in this copy (never rebuilt here)".format(t))
    search_b.cache_short_evidence()
    lab_gate.own_dead_cache()
    trial = search_b.every_trial()[TRIAL]
    out = setting_dir(setting, tag)
    os.makedirs(out, exist_ok=True)
    for m in matches:
        started = time.time()
        aug = search_b.fold_models(m)["augment"]
        if is_v16(tag):        # v16's B: the four netmet seeds' fold nets for m, each through its fold's DFL normaliser
            norm = tuple(x for x in lab.matches(WHICH) if x != m and sets.source(x) in B.L.TRAIN_SOURCES)
            grus = [lab_gate.gru_model(norm, WHICH, path) for path in v16_nets(m, tag)]
        else:
            held = m if sets.source(m) in B.L.TRAIN_SOURCES else B.held_list()[-1]
            gru = lab_gate.gru_model(B.used_for(held), WHICH, B.net_for(m))
        halves = lab_temporal.halves_of(m, WHICH)
        for variant in VARIANTS:
            if variant == "stressed":
                c = C.stressed_features(m, WHICH, C.TEST_SEED)
                feature_file, p_aug = C.feature_path(m, WHICH, C.TEST_SEED), C.p_path("augment", "stressed", m, WHICH)
            else:
                c = lab_gate.variant_features(m, WHICH, variant)
                feature_file, p_aug = lab_gate.feature_path(m, WHICH, variant), lab_gate.p_path("augment", variant, m, WHICH)
            np.save(p_aug, C.proba(aug, c).astype(np.float32))
            if is_v16(tag):
                p_gru = v16_average([lab_gate.gru_proba(g, c, halves) for g in grus])
            else:
                p_gru = lab_gate.gru_proba(gru, c, halves)
            np.save(B.p_path(variant, m), p_gru.astype(np.float32))
            del c
            os.remove(feature_file)
        models = search_b.trial_models(trial, without=m)
        rows = []
        for variant in VARIANTS:
            match = lab_gate.variant_match(m, WHICH, variant)
            letters = search_b.letters_of(match, variant, {**trial, "post": "own"}, {"own": models})
            s = bench_score(match.truth, letters)
            rows.append({"setting": setting + tag, "match": m, "source": sets.source(m), "variant": variant,
                         "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"]})
            codes = np.vectorize(LETTER_CODE.get, otypes=[np.int8])(np.asarray(letters).astype(str))
            np.save(os.path.join(out, "letters_{}_{}.npy".format(m, variant)), codes)
            p = search_b.start_p(m, variant, trial["w_gru"], trial["f_dead"])
            y = np.clip(match.truth.astype(np.int64), 0, 2)
            ll = -np.log(p[np.arange(len(y)), y])
            np.save(os.path.join(out, "logloss_{}_{}.npy".format(m, variant)), ll.astype(np.float16))
            if variant == VARIANTS[0]:
                np.save(os.path.join(out, "truth_{}.npy".format(m)), match.truth.astype(np.int8))
            del match
        pd.DataFrame(rows).to_csv(os.path.join(out, "scores_{}.csv".format(m)), index=False)
        print("{}{} {}: {} ({:.0f} s)".format(setting, tag, m, " ".join("{} {:.5f}".format(r["variant"], r["spell_f1"])
                                                                       for r in rows), time.time() - started), flush=True)


def setup(setting, models=None, seed_dir=None):
    """scratchpad/broadcast/wc_SETTING: b16_c0's code; data/external with the raw data linked, the trained models
    cloned (APFS clones: no disk) from b16_c0 (or from `models`, another copy) and every tracking-dependent cache empty.
    seed_dir (the noise rule): gru_b's fold nets and trial 1015's tables replaced by another seed's (lab_netens/seedS:
    nets/net_*.pt, search_b/tables_trial1015.joblib)."""
    src_root = models or B16
    wc = os.path.join(WORK, "wc_" + setting)
    os.makedirs(wc, exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "/data", "--exclude", "/viewer", "--exclude", "/docs", "--exclude", ".DS_Store",
                    "--exclude", "__pycache__", B16 + "/", wc + "/"], check=True)
    for name in ("lab_broadcast.py", "lab_trackerr.py"):
        link = os.path.join(wc, "bench", name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join(MAIN, "bench", name), link)
    ext, srcext = os.path.join(wc, "data", "external"), os.path.join(src_root, "data", "external")
    os.makedirs(ext, exist_ok=True)
    for d in ("idsse", "metrica", "skillcorner", "pathcrf"):
        target = os.path.join(MAIN, "data", "external", d)
        if not os.path.lexists(os.path.join(ext, d)):
            os.symlink(target, os.path.join(ext, d))
    clones = {"lab_temporal": ["stats_*.npz"], "lab_carried": ["ball_0_*.npz", "ball_0_*.csv"],
              "lab_gate": ["folds_*.joblib"], "lab_branch_b": ["net_*.pt"],
              "search_b": ["tables_trial0.joblib", "tables_trial{}.joblib".format(TRIAL)]}
    for d in ("lab", "lab_robust", "lab_temporal", "lab_carried", "lab_gate", "lab_branch_b", "search_b", "lab_dead"):
        os.makedirs(os.path.join(ext, d), exist_ok=True)
    for d, patterns in clones.items():
        for pattern in patterns:
            found = sorted(glob.glob(os.path.join(srcext, d, pattern)))
            if not found:
                raise RuntimeError("{}: nothing matches {}/{}".format(src_root, d, pattern))
            for src in found:
                dst = os.path.join(ext, d, os.path.basename(src))
                if not os.path.exists(dst):
                    subprocess.run(["cp", "-c", src, dst], check=True)
    if is_v16(setting):          # setting here is SETTING + TAG
        dst = os.path.join(ext, "search_b", "tables_trial{}.joblib".format(TRIAL))
        os.remove(dst)
        subprocess.run(["cp", "-c", v16_tables(setting), dst], check=True)
        missing = [p for m in lab.matches(WHICH) for p in v16_nets(m, setting) if not os.path.exists(p)]
        if missing:
            raise RuntimeError("v16 nets missing: {}".format(missing[:3]))
    if seed_dir:
        for src in sorted(glob.glob(os.path.join(seed_dir, "nets", "net_*.pt"))):
            dst = os.path.join(ext, "lab_branch_b", os.path.basename(src))
            if os.path.exists(dst):
                os.remove(dst)
            subprocess.run(["cp", "-c", src, dst], check=True)
        dst = os.path.join(ext, "search_b", "tables_trial{}.joblib".format(TRIAL))
        os.remove(dst)
        subprocess.run(["cp", "-c", os.path.join(seed_dir, "search_b", "tables_trial{}.joblib".format(TRIAL)), dst], check=True)
    print("{}: ready (models from {}{})".format(wc, src_root, "; nets and tables from " + seed_dir if seed_dir else ""))


def clean(setting):
    ext = os.path.join(WORK, "wc_" + setting, "data", "external")
    for d in ("lab", "lab_robust", "lab_carried", "lab_gate", "lab_branch_b", "search_b", "lab_dead"):
        path = os.path.join(ext, d)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
    print("{}: caches removed".format(setting))


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------

def load_scores(name):
    files = sorted(glob.glob(os.path.join(SHARED, name, "scores_*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def reproduction(name="off", reference=CONFIRM):
    mine = load_scores(name)
    ref = pd.read_csv(reference)
    ref = ref[ref.trial == TRIAL]
    j = ref.merge(mine, on=["match", "variant"], suffixes=("_v15", "_mine"))
    j["diff"] = (j.spell_f1_mine - j.spell_f1_v15).abs()
    return j, float(j["diff"].max()), len(j)


def zone_table(name, base):
    out = {z: {"frames": 0, "right_base": 0, "right": 0, "changed": 0, "ll_base": 0.0, "ll": 0.0}
           for z in ("near", "far", "none")}
    for path in sorted(glob.glob(os.path.join(SHARED, name, "letters_*.npy"))):
        stem = os.path.basename(path)[len("letters_"):-len(".npy")]
        m, variant = stem.rsplit("_", 1)
        base_path = os.path.join(SHARED, base, os.path.basename(path))
        if not os.path.exists(base_path):
            continue
        new, old = np.load(path), np.load(base_path)
        truth = np.load(os.path.join(SHARED, base, "truth_{}.npy".format(m)))
        zp = os.path.join(SHARED, name, "zone_{}.npy".format(m))
        zone = np.load(zp) if os.path.exists(zp) else np.zeros(len(truth), np.int8)
        graded = np.isin(truth, (0, 1, 2))
        lp = np.load(os.path.join(SHARED, name, "logloss_{}_{}.npy".format(m, variant))).astype(np.float64)
        lb = np.load(os.path.join(SHARED, base, "logloss_{}_{}.npy".format(m, variant))).astype(np.float64)
        for z, code in (("near", 2), ("far", 1), ("none", 0)):
            w = graded & (zone == code)
            o = out[z]
            o["frames"] += int(w.sum())
            o["right_base"] += int((old[w] == truth[w]).sum())
            o["right"] += int((new[w] == truth[w]).sum())
            o["changed"] += int((old[w] != new[w]).sum())
            o["ll_base"] += float(lb[w].sum())
            o["ll"] += float(lp[w].sum())
    return out


def summary(name, base_name):
    s, base = load_scores(name), load_scores(base_name)
    if s is None or base is None:
        return None
    base = base[["match", "variant", "source", "spell_f1"]]
    u = s.merge(base, on=["match", "variant", "source"], suffixes=("", "_base"))
    u["d"] = u.spell_f1 - u.spell_f1_base
    pm = u.groupby(["match", "source"])[["spell_f1", "spell_f1_base", "d"]].mean().reset_index()
    infos = {}
    for path in glob.glob(os.path.join(SHARED, name, "info_*.json")):
        with open(path) as handle:
            infos[os.path.basename(path)[5:-5]] = json.load(handle)
    view = None
    if infos and all("view" in i for i in infos.values()):
        keys = list(next(iter(infos.values()))["view"].keys())
        view = {k: float(np.mean([i["view"][k] for i in infos.values()])) for k in keys}
    return {"setting": name, "base": base_name, "units": len(u), "matches": len(pm),
            "b_bench": float(pm.spell_f1.mean()), "b_bench_base": float(pm.spell_f1_base.mean()),
            "gain": float(pm.d.mean()), "better": int((pm.d > 1e-9).sum()), "worse": int((pm.d < -1e-9).sum()),
            "per_variant": {v: float(g.d.mean()) for v, g in u.groupby("variant")},
            "per_variant_score": {v: float(g.spell_f1.mean()) for v, g in u.groupby("variant")},
            "per_source": {k: float(g.d.mean()) for k, g in pm.groupby("source")},
            "per_match": {r.match: {"score": float(r.spell_f1), "base": float(r.spell_f1_base), "d": float(r.d)}
                          for r in pm.itertuples()},
            "zones": zone_table(name, base_name), "view": view,
            "merge": {m: i.get("merge") for m, i in sorted(infos.items())}}


def report():
    os.makedirs(RESULTS, exist_ok=True)
    out = {}
    if load_scores("off") is not None:
        j, worst, n = reproduction("off")
        print("reproduction: {} units, max |off - v15's B| = {:.2e} ({}); B's bench {:.4f}".format(
            n, worst, "PASSES" if n == 30 and worst <= 1e-5 else "FAILS",
            load_scores("off").groupby("match").spell_f1.mean().mean()))
        j.to_csv(os.path.join(RESULTS, "reproduction.csv"), index=False, float_format="%.6f")
        out["reproduction"] = {"units": n, "max_abs_diff": worst}
    if load_scores("off_v16") is not None:
        j, worst, n = reproduction("off_v16", V16_CONFIRM)
        print("reproduction v16: {} units, max |off_v16 - netmet5's B| = {:.2e} ({}); B's bench {:.4f}".format(
            n, worst, "PASSES" if n == 30 and worst <= 1e-5 else "FAILS",
            load_scores("off_v16").groupby("match").spell_f1.mean().mean()))
        j.to_csv(os.path.join(RESULTS, "reproduction_v16.csv"), index=False, float_format="%.6f")
        out["reproduction_v16"] = {"units": n, "max_abs_diff": worst}
    if load_scores("off_w4") is not None:
        j, worst, n = reproduction("off_w4", W4_CONFIRM)
        print("reproduction W4: {} units, max |off_w4 - W4's B| = {:.2e} ({}); B's bench {:.4f}".format(
            n, worst, "PASSES" if n == 30 and worst <= 1e-5 else "FAILS",
            load_scores("off_w4").groupby("match").spell_f1.mean().mean()))
        j.to_csv(os.path.join(RESULTS, "reproduction_w4.csv"), index=False, float_format="%.6f")
        out["reproduction_w4"] = {"units": n, "max_abs_diff": worst}
    rows = []
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SHARED, "*")) if os.path.isdir(p))
    for name in names:
        if name == "off" or load_scores(name) is None:
            continue
        # seed-1 runs against v15's B on seed 1; everything else (the retrained B's too) against v15's B
        base = ("off_s1" if name.endswith("_s1") else "off_v16" if name.endswith("_v16")
                else "off_w4" if name.endswith("_w4") else "off")
        if name == base:
            continue
        s = summary(name, base)
        if s is None:
            continue
        rows.append(s)
        print("{:<22s} B's bench {:.4f} (off {:.4f}): {:+.4f}, better/worse {}/{}; p25/p50/stressed {}; IDSSE/Metrica {}".format(
            name, s["b_bench"], s["b_bench_base"], s["gain"], s["better"], s["worse"],
            " / ".join("{:+.4f}".format(s["per_variant"][v]) for v in VARIANTS if v in s["per_variant"]),
            " / ".join("{:+.4f}".format(s["per_source"][k]) for k in ("idsse", "metrica") if k in s["per_source"])))
        for z, o in s["zones"].items():
            if o["frames"]:
                print("      {:<4s}: {:>8d} graded frames, accuracy {:.4f} -> {:.4f} ({:+.4f}), changed {:.2%}, "
                      "log loss {:.4f} -> {:.4f}".format(z, o["frames"], o["right_base"] / o["frames"], o["right"] / o["frames"],
                                                         (o["right"] - o["right_base"]) / o["frames"], o["changed"] / o["frames"],
                                                         o["ll_base"] / o["frames"], o["ll"] / o["frames"]))
        s2 = load_scores(name)
        s2.to_csv(os.path.join(RESULTS, "scores_{}.csv".format(name)), index=False, float_format="%.6f")
    out["settings"] = rows
    with open(os.path.join(RESULTS, "report.json"), "w") as handle:
        json.dump(out, handle, indent=1)



# --------------------------------------------------------------------------------------------
# The port: possession/broadcast_merge.py against this file's merge, end to end through the files
# --------------------------------------------------------------------------------------------

PORT_MATCH = "J03WR9"
PORT_OFFSETS = (500, 2500)      # broadcast frame = tactical frame + these (1st, 2nd half); other footage between


def port(m=PORT_MATCH):
    """The central simulation of a bench match written in docs/data.md's formats (EPTS-like meta XML, raw text,
    homography CSV) on its own timeline, read back by possession/broadcast_merge.py, aligned and merged; against
    merge() on the simulated arrays. Also: the teams in the other order in the metadata; a missing file; an unrelated
    ball. Bench data only."""
    import shutil
    import tempfile
    from possession import broadcast_merge as BM
    from possession.data import PITCH_LENGTH, PITCH_WIDTH
    f = lab_inputs(m)
    clean = sets.load(m)
    halves = [tuple(int(x) for x in h) for h in f["halves"]]
    bro = simulate(m, "central", clean, f["truth"], halves)
    out = {"match": m}
    track = {"players": f["players"], "team": f["team"], "ball": f["ball"],
             "speed": np.zeros(f["players"].shape[:2], np.float32)}
    nb = halves[1][2] + PORT_OFFSETS[1] + 300
    rng = np.random.default_rng(0)
    BP = np.full((nb, bro["players"].shape[1], 2), np.nan, np.float32)
    BB = np.full((nb, 2), np.nan, np.float32)
    BV = np.zeros(nb, bool)
    for (_, a, b), o in zip(halves, PORT_OFFSETS):
        t = np.arange(a, b + 1)
        BP[t + o] = bro["players"][t]
        BV[t + o] = bro["valid"][t]
        ball = clean["ball"][bro["src"][t], :2] + rng.normal(0, 0.5, (len(t), 2))
        ball[~bro["valid"][t]] = np.nan
        BB[t + o] = ball
    gap = np.arange(halves[0][2] + PORT_OFFSETS[0] + 1, halves[1][1] + PORT_OFFSETS[1])
    BP[gap[::7], :3] = rng.uniform(0, 60, (len(gap[::7]), 3, 2))
    d = tempfile.mkdtemp(dir=WORK)
    paths = {k: os.path.join(d, v) for k, v in (("atd_meta", "meta.xml"), ("atd_raw", "raw.txt"),
                                                  ("homography", "homography.csv"))}

    def write_meta(first):
        ids = ["T_A", "T_B"] if first == "home" else ["T_B", "T_A"]
        with open(paths["atd_meta"], "w") as h:
            h.write('<?xml version="1.0"?>\n<Main><Metadata><Teams>')
            for tid in ids:
                h.write('<Team id="{}"><Name>{}</Name></Team>'.format(tid, tid))
            h.write("</Teams><Players>")
            for k, side in enumerate(bro["team"]):
                h.write('<Player id="P{}" teamId="{}"><ShirtNumber>{}</ShirtNumber></Player>'.format(
                    k + 1, "T_A" if side == HOME else "T_B", k + 1))
            h.write("</Players></Metadata></Main>\n")

    def fmt(x, y):
        return "{:.9f},{:.9f},NaN".format(x / PITCH_LENGTH, y / PITCH_WIDTH) if x == x else "NaN,NaN,NaN"
    write_meta("home")
    with open(paths["atd_raw"], "w") as h:
        for i in range(nb):
            h.write("{}:{}:{}\n".format(i + 1, ";".join(fmt(x, y) for x, y in BP[i]), fmt(*BB[i])))
    with open(paths["homography"], "w") as h:
        h.write("frame," + ",".join("h{}{}".format(a, b) for a in range(3) for b in range(3)) + ",reprojection_error,valid\n")
        for i in range(nb):
            h.write("{},{},0.1,{}\n".format(i, ",".join(["0"] * 9), int(BV[i])))
    _, rb = BM.read_raw(paths["atd_raw"], len(bro["team"]))
    out["shifts"] = [{"true": -o, "found": BM.global_shift(f["ball"].astype(np.float64), rb, a, b)}
                     for (_, a, b), o in zip(halves, PORT_OFFSETS)]
    for rule_name, rule in (("M", BM.RULE), ("M2", BM.RULE_M2)):
        ref_p, ref_t, _, _, _ = merge(f["players"], f["team"], bro["players"], bro["team"], bro["valid"], halves, rule_name)
        mod_p, mod_t, _, _, _ = BM.merge(f["players"], f["team"], bro["players"], bro["team"], bro["valid"], halves, rule)
        res = {"merge_identical": bool(np.array_equal(ref_p, mod_p, equal_nan=True) and np.array_equal(ref_t, mod_t))}
        for first in ("home", "away"):
            write_meta(first)
            got, status = BM.merge_broadcast(track, halves, paths=paths, rule=rule)
            P = got["players"]
            same = P.shape == ref_p.shape and np.array_equal(got["team"], ref_t)
            a_ = np.isnan(P[:, :, 0]) if same else None
            b_ = np.isnan(ref_p[:, :, 0]) if same else None
            res["files_{}_first".format(first)] = {
                "status": status, "same_slots": bool(same),
                "slot_frames_differ": int((a_ != b_).sum()) if same else None,
                "max_abs_diff_m": float(np.nanmax(np.abs(P - ref_p))) if same else None}
        out[rule_name] = res
    write_meta("home")
    got, status = BM.merge_broadcast(track, halves, paths=dict(paths, atd_raw=os.path.join(d, "none.txt")))
    out["missing_file"] = {"status": status, "unchanged": got is track}
    with open(os.path.join(d, "raw2.txt"), "w") as h:
        for i in range(nb):
            h.write("{}:{}:{}\n".format(i + 1, ";".join(["NaN,NaN,NaN"] * BP.shape[1]),
                                          "{:.6f},{:.6f},NaN".format(*rng.uniform(0, 1, 2))))
    got, status = BM.merge_broadcast(track, halves, paths=dict(paths, atd_raw=os.path.join(d, "raw2.txt")))
    out["unrelated_ball"] = {"status": status, "unchanged": got is track}
    shutil.rmtree(d)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "port.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("calibrate", "stats", "setup", "build", "score", "report", "clean", "port"))
    parser.add_argument("setting", nargs="?", choices=list(SETTINGS))
    parser.add_argument("--matches", nargs="+", default=None)
    parser.add_argument("--tag", default="", help="suffix of the results folder (e.g. _rt for the retrained models)")
    parser.add_argument("--models", default=None, help="setup: the copy whose trained models to clone (default b16_c0)")
    parser.add_argument("--seed-dir", default=None, help="setup: another seed's gru_b nets and tables (lab_netens/seedS)")
    args = parser.parse_args()
    ms = args.matches or lab.matches(WHICH)
    if args.step == "calibrate":
        return calibrate()
    if args.step == "stats":
        return stats(args.setting, ms)
    if args.step == "setup":
        return setup(args.setting + args.tag, args.models, args.seed_dir)
    if args.step == "build":
        return build(args.setting, ms, args.tag)
    if args.step == "score":
        return score(args.setting, ms, args.tag)
    if args.step == "clean":
        return clean(args.setting + args.tag)
    if args.step == "port":
        return port()
    return report()


if __name__ == "__main__":
    main()
