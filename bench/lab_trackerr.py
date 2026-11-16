"""Tracking errors: how much would wrong teams and misplaced players cost v15's branch B? (docs/lab/tracking-errors.md)

A sensitivity measurement, not a candidate: nothing here is adopted and no model changes. Errors are injected into
the bench's damaged tracking as the models see it (the lab cache's players and team), per match with a fixed seed;
then v8's keeper repair and every input computed from the tracking are recomputed on it, out of fold, and v15's
branch B (search trial 1015) runs with every model trained on CLEAN bench data (b16_c0's: the verified rebuild of
v15's B, used read-only), on B's bench (10 matches x p25, p50, stressed).

    python -m bench.lab_trackerr prep --matches M [M ...]     # once: clean training rows, pieces, keepers (shared)
    python -m bench.lab_trackerr folds                        # once: the fold dead-ball and restart models (clean rows)
    python -m bench.lab_trackerr setup SETTING                # an isolated copy of the repo for one setting
    # in that copy (scratchpad/trackerr/wc_SETTING):
    python -m bench.lab_trackerr build SETTING --matches ...  # the corrupted lab cache (repair, every input recomputed)
    python -m bench.lab_trackerr score SETTING --matches ...  # features, augment's and gru_b's P, trial 1015, scores
    python -m bench.lab_trackerr report                       # every setting against `none`; the reproduction check
    python -m bench.lab_trackerr clean SETTING                # delete the copy's bulky caches once scored

SETTING: none (no corruption: must reproduce v15's B to 1e-5), T1, T3, T10, K10, G3 (the error model is on the page,
fixed before any score). Nothing here reads our match.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import time
import zlib

import joblib
import numpy as np
import pandas as pd

from bench import chain, lab, lab_clean, sets
from bench.eval_pathcrf import v7_cost
from bench.run import score as bench_score
from bench.train_dead_ball import training_rows, v4_cost
from bench.train_restart import cost_with_restart, restart_labels, restart_rows
from bench.tune_decoder import decode
from possession import dead_ball, features, pathcrf_adapter, repair, restart, v1, v9, v11
from possession.data import AWAY, HOME

# Params (fixed before any score; docs/lab/tracking-errors.md)
WHICH = "all"
SETTINGS = {"none": None, "T1": ("team", 0.01), "T3": ("team", 0.03), "T10": ("team", 0.10),
            "K10": ("keeper", 0.10), "G3": ("ghost", 0.03),
            # DIAGNOSTIC (added 08:55, before any corrupted score): K10's very errors with v8's repair switched off
            "K10nr": ("keeper", 0.10),
            # DIAGNOSTIC (added 09:28, after T1/T3 were scored): T1's and T3's rates with another draw of pieces (the
            # setting's name seeds it), for how much the drop depends on which pieces are hit
            "T1b": ("team", 0.01), "T3b": ("team", 0.03)}
SAME_ERRORS = {"K10nr": "K10"}     # a diagnostic draws the errors of the setting it mirrors
NO_REPAIR = ("K10nr",)
SEED = 20261115
GHOST_M = (10.0, 20.0)       # G3: the offset's length, uniform
NEAR_M = 15.0                # analysis: a corrupted piece this near the true ball ("near"; o_near's radius)
KEEPER_SEEN = 0.5            # a keeper candidate is seen on at least this share of the half
TRIAL = 1015
VARIANTS = ("p25", "p50", "stressed")
LETTER_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}

MAIN = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))       # the main repo (this file's real home)
SCRATCH = "/private/tmp/claude-501/-Users-joseruiz-Code-more-than-a-hack-2027/39e3effe-3bf8-4611-8584-0402cfeb00b7/scratchpad"
B16 = os.path.join(SCRATCH, "b16_c0")                                     # the verified rebuild of v15's B (read-only)
WORK = os.path.join(SCRATCH, "trackerr")                                  # the per-setting copies
SHARED = os.path.join(MAIN, "data", "external", "lab_trackerr")           # gitignored (data/*)
RESULTS = os.path.join(MAIN, "bench", "results", "lab_trackerr")
CONFIRM = os.path.join(MAIN, "bench", "results", "search_b", "confirm_1015_554_586_1133_1269_-1_-2.csv")


def b16_lab(m):
    return os.path.join(B16, "data", "external", "lab", "{}.npz".format(m))


def setting_dir(setting):
    return os.path.join(SHARED, setting)


def halves_pitch(f):
    return [tuple(int(x) for x in h) for h in f["halves"]], tuple(float(x) for x in f["pitch"])


def as_track(players, team, ball):
    """A tracking dict as eval_pathcrf's consumers read it (the speed channel only feeds v1's `stopped` and `active`,
    which the bench's costs zero: lab_carried.ball_costs)."""
    return {"players": players, "team": team, "ball": ball, "speed": np.zeros(players.shape[:2], np.float32)}


# --------------------------------------------------------------------------------------------
# Once: the clean training rows (eval_pathcrf's), the pieces and their people, the keepers
# --------------------------------------------------------------------------------------------

def keepers_of(clean, halves):
    """{(half index, team): person}: per half and team, the person seen on >= KEEPER_SEEN of the half whose mean
    distance to his own goal is smallest (own goal: the end his team stands nearer to, as repair.py), and his share of
    seen frames inside his own penalty area (a check)."""
    length, width = clean["pitch"]
    P, team = clean["players"], np.asarray(clean["team"])
    out = {}
    for hi, (_, first, last) in enumerate(halves):
        x, y = P[first:last + 1, :, 0], P[first:last + 1, :, 1]
        home_left = np.nanmean(x[:, team == HOME]) < np.nanmean(x[:, team == AWAY])
        for side in (HOME, AWAY):
            goal_x = 0.0 if (side == HOME) == home_left else length
            best = None
            for p in np.flatnonzero(team == side):
                seen = ~np.isnan(x[:, p])
                if seen.mean() < KEEPER_SEEN:
                    continue
                d = float(np.mean(np.hypot(x[seen, p] - goal_x, y[seen, p] - width / 2)))
                if best is None or d < best[1]:
                    area = repair.inside(x[seen, p], y[seen, p], goal_x, repair.AREA_DEPTH, repair.AREA_HALF_WIDTH, width)
                    best = (int(p), d, float(area.mean()))
            out[(hi, side)] = {"person": best[0], "mean_goal_distance": best[1], "in_area": best[2]}
    return out


def prep(matches):
    os.makedirs(SHARED, exist_ok=True)
    for m in matches:
        started = time.time()
        with np.load(b16_lab(m)) as z:
            f = {k: z[k] for k in ("players", "team", "ball", "truth", "halves", "pitch")}
        halves, pitch = halves_pitch(f)
        track = as_track(f["players"], f["team"], f["ball"])
        truth = f["truth"]
        dnames, Xd = features.frame_features(track, halves, pitch)
        Xdr, ydr = training_rows(Xd, truth)
        oriented = restart.all_orientations(track, halves, pitch)
        Xsr, ysr = restart_rows(oriented, restart_labels(truth))
        np.savez(os.path.join(SHARED, "rows_{}.npz".format(m)), Xd=Xdr, yd=ydr, Xs=Xsr, ys=ysr,
                 dnames=np.array(dnames), rnames=np.array(oriented[0][2]))
        del Xd, oriented
        clean = sets.load(m)
        found = lab_clean.person_of_slots(track, clean, halves)
        rows = []
        for (a, e, s, t), person in found:
            hi = next(i for i, (_, first, last) in enumerate(halves) if first <= a <= last)
            rows.append((a, e, s, t, person, hi))
        arr = np.array(rows, dtype=np.int64)
        keep = keepers_of(clean, halves)
        np.savez(os.path.join(SHARED, "pieces_{}.npz".format(m)), pieces=arr,
                 clean_ball=clean["ball"][:, :2].astype(np.float32))
        with open(os.path.join(SHARED, "keepers_{}.json".format(m)), "w") as handle:
            json.dump({"{}_{}".format(*k): v for k, v in keep.items()}, handle, indent=1)
        print("{}: rows, {} pieces, keepers {} ({:.0f} s)".format(
            m, len(arr), {"{}_{}".format(*k): (v["person"], round(v["in_area"], 2)) for k, v in keep.items()},
            time.time() - started), flush=True)


def fold_path(held):
    return os.path.join(SHARED, "fold_{}.joblib".format(held))


def folds():
    """eval_pathcrf's fold models: v4's dead-ball and v6's restart model on the other 9 matches' clean rows."""
    ms = lab.matches(WHICH)
    rows = {}
    for m in ms:
        with np.load(os.path.join(SHARED, "rows_{}.npz".format(m))) as z:
            rows[m] = {k: z[k] for k in z.files}
    dsetting, rsetting = chain.dead_ball(WHICH), chain.restart(WHICH)
    dnames = [str(x) for x in rows[ms[0]]["dnames"]]
    rnames = [str(x) for x in rows[ms[0]]["rnames"]]
    for held in ms:
        others = [m for m in ms if m != held]
        dead_model = dead_ball.fit(dsetting["family"], np.concatenate([rows[m]["Xd"] for m in others]),
                                   np.concatenate([rows[m]["yd"] for m in others]), dnames)
        start_model = dead_ball.fit(rsetting["family"], np.concatenate([rows[m]["Xs"] for m in others]),
                                    np.concatenate([rows[m]["ys"] for m in others]), rnames)
        joblib.dump({"dead": dead_model, "start": start_model}, fold_path(held), compress=3)
        print("{}: fold models".format(held), flush=True)


# --------------------------------------------------------------------------------------------
# The errors
# --------------------------------------------------------------------------------------------

def pick(order, length, target):
    """Pieces of `order`, in order, until their frames reach `target`; a piece that would cross the target is taken
    only if it brings the total nearer to it, else the next ones are tried (2026-11-15 08:48, before any corrupted
    score: the plain "until reached" overshot by a whole piece, 17 % for one keeper in a dry run of the errors alone)."""
    out, acc = [], 0
    for i in order:
        if acc >= target:
            break
        n = int(length[i])
        if acc + n <= target or acc + n - target < target - acc:
            out.append(int(i))
            acc += n
    return out


def corrupt(setting, m, players, team):
    """(players, team, info): the setting's errors on the damaged tracking (inputs not modified)."""
    kind, share = SETTINGS[setting]
    with np.load(os.path.join(SHARED, "pieces_{}.npz".format(m))) as z:
        P, clean_ball = z["pieces"], z["clean_ball"]
    with open(os.path.join(SHARED, "keepers_{}.json".format(m))) as handle:
        keep = json.load(handle)
    seed_name = SAME_ERRORS.get(setting, setting)
    rng = np.random.default_rng([SEED, zlib.crc32(m.encode()), zlib.crc32(seed_name.encode())])
    a_, e_, s_, t_, person, half = (P[:, k] for k in range(6))
    length = e_ - a_
    keeper_of = {(int(k.split("_")[0]), int(k.split("_")[1])): v["person"] for k, v in keep.items()}
    is_keeper = np.array([keeper_of.get((int(h), int(t))) == int(p) for h, t, p in zip(half, t_, person)])
    groups = []
    if kind in ("team", "ghost"):
        groups.append(("outfield", np.flatnonzero(~is_keeper)))
    else:
        for k in sorted(set(keeper_of.values())):
            groups.append(("keeper {}".format(k), np.flatnonzero(is_keeper & (person == k))))
    chosen, shares = [], {}
    for name, idx in groups:
        total = int(length[idx].sum())
        got = pick(idx[rng.permutation(len(idx))], length, share * total)
        chosen += got
        shares[name] = {"eligible_frames": total, "affected_frames": int(length[got].sum()), "pieces": len(got),
                        "share": float(length[got].sum() / max(total, 1))}
    chosen.sort(key=lambda i: (int(a_[i]), int(s_[i])))
    players = players.copy()
    team = np.asarray(team).copy()
    n = len(players)
    zone = np.zeros(n, np.int8)                 # 0 no corrupted piece, 1 one (none near the true ball), 2 one near it
    moved, appended = [], 0
    occ = ~np.isnan(players[:, :, 0])
    for i in chosen:
        a, e, s = int(a_[i]), int(e_[i]), int(s_[i])
        true_pos = players[a:e, s].astype(np.float64)
        with np.errstate(invalid="ignore"):
            near = np.hypot(*(true_pos - clean_ball[a:e]).T) <= NEAR_M
        zone[a:e] = np.maximum(zone[a:e], np.where(near, 2, 1).astype(np.int8))
        if kind == "ghost":
            r = rng.uniform(*GHOST_M)
            th = rng.uniform(0.0, 2 * np.pi)
            pos = true_pos + np.array([r * np.cos(th), r * np.sin(th)])
            pos[:, 0] = np.clip(pos[:, 0], 0.0, PITCH[m][0])
            pos[:, 1] = np.clip(pos[:, 1], 0.0, PITCH[m][1])
            players[a:e, s] = pos.astype(np.float32)
            moved.append({"start": a, "end": e, "slot": s, "person": int(person[i]), "offset_m": float(r)})
            continue
        other = AWAY if team[s] == HOME else HOME
        lo, hi = max(a - 1, 0), min(e + 1, n)
        best, best_last = None, None
        for k in np.flatnonzero(team == other):
            if occ[lo:hi, k].any():
                continue
            before = np.flatnonzero(occ[:a, k])
            last = int(before[-1]) if len(before) else -1
            if best is None or last < best_last:
                best, best_last = int(k), last
        if best is None:                          # no free slot of the other team: a new one
            players = np.concatenate([players, np.full((n, 1, 2), np.nan, np.float32)], axis=1)
            occ = np.concatenate([occ, np.zeros((n, 1), bool)], axis=1)
            team = np.concatenate([team, np.array([other], team.dtype)])
            best = players.shape[1] - 1
            appended += 1
        players[a:e, best] = players[a:e, s]
        players[a:e, s] = np.nan
        occ[a:e, best], occ[a:e, s] = True, False
        moved.append({"start": a, "end": e, "slot": s, "new_slot": best, "person": int(person[i]),
                      "keeper": bool(is_keeper[i])})
    info = {"setting": setting, "groups": shares, "pieces": moved, "appended_slots": appended}
    return players, team, zone, info


PITCH = {}


# --------------------------------------------------------------------------------------------
# Every input computed from the tracking, as eval_pathcrf and lab.build made them (out of fold)
# --------------------------------------------------------------------------------------------

_PATHCRF = None


def pathcrf_model():
    global _PATHCRF
    if _PATHCRF is None:
        _PATHCRF = pathcrf_adapter.load_model()          # float32, as bench/eval_pathcrf.py ran it
    return _PATHCRF


def inputs_of(track, halves, pitch, fold):
    """eval_pathcrf's out-of-sample outputs on one match's tracking, plus lab.build's v9 / v11 letters."""
    dsetting, rsetting, decoder = chain.dead_ball(WHICH), chain.restart(WHICH), chain.decoder(WHICH)
    pc = chain.pathcrf(WHICH)
    _, Xd = features.frame_features(track, halves, pitch)
    clues = v1.compute_clues(track, [], halves, pitch)
    base = v1.build_costs(dict(clues, stopped=np.zeros_like(clues["stopped"]), active=np.zeros_like(clues["active"])))
    oriented = restart.all_orientations(track, halves, pitch)
    p_dead = dead_ball.predict(fold["dead"], Xd)
    p_start = restart.p_home(fold["start"], oriented)
    del Xd, oriented
    v6c = cost_with_restart(base, p_dead, p_start, halves, rsetting["weight"], rsetting["carry"], dsetting["weight"])
    v6l = decode(v6c, halves, decoder)
    missing = np.isnan(track["ball"][:, 0])
    live = np.isin(v6l, ["H", "A"])
    p_pc = pathcrf_adapter.p_home(pathcrf_model(), track, halves, pitch, live)
    v7c = v7_cost(v6c, p_pc, missing, pc["weight"], pc["where"])
    v7l = decode(v7c, halves, decoder)
    belief = p_pc.copy()
    if pc["where"] == "gaps":
        belief[~missing] = np.nan
    pred = {"v4": decode(v4_cost(base, p_dead, dsetting["weight"]), halves, decoder), "v6": v6l, "v7": v7l,
            "p_dead": p_dead,
            "restart_clue": restart.restart_belief(p_start, p_dead, halves, int(rsetting["carry"] * v1.FPS)),
            "pathcrf_clue": belief, "v6_cost": v6c.astype(np.float32), "v7_cost": v7c.astype(np.float32),
            "p_start": p_start, "p_pathcrf": p_pc}
    back, ahead = chain.restart_snap(WHICH)
    s = v11.SETTINGS[WHICH]
    v9l, _ = v9.snap(v7l.astype(str), track, halves, back, ahead)
    v11l, _ = v11.move(v9l, track, halves, s["dv"], s["back"], s["ahead"])
    return pred, v9l, v11l


def build(setting, matches):
    """The setting's lab cache (in this copy's data/external/lab), and what was corrupted (shared)."""
    if not os.path.realpath(lab.ROOT).startswith(os.path.realpath(WORK)):
        raise RuntimeError("build runs only inside a setting's copy (scratchpad/trackerr/wc_SETTING)")
    os.makedirs(lab.folder(WHICH), exist_ok=True)
    out = setting_dir(setting)
    os.makedirs(out, exist_ok=True)
    for m in matches:
        started = time.time()
        with np.load(b16_lab(m)) as z:
            f = {k: z[k] for k in z.files if k not in ("players_raw", "team_raw")}
        halves, pitch = halves_pitch(f)
        PITCH[m] = pitch
        players, team = f["players"], f["team"]
        info = {"setting": setting}
        zone = np.zeros(len(players), np.int8)
        if SETTINGS[setting] is not None:
            players, team, zone, info = corrupt(setting, m, players, team)
        track = as_track(players, team, f["ball"])
        if setting in NO_REPAIR:
            fixed, rpieces = track, []
        else:
            fixed, rpieces = repair.repair(track, halves, pitch)                          # v8, as v12.inputs
        info["repair"] = [{k: v for k, v in p.items()} for p in rpieces]
        info["repair_undid"] = undone(info, rpieces)
        if SETTINGS[setting] is not None and SETTINGS[setting][0] == "keeper":
            # diagnostic only: the same rule with the stressed variant's ball (more of it seen in stoppages)
            from bench import lab_carried as C
            stressed_track = dict(track, ball=C.stressed_ball(m, C.TEST_SEED))
            info["repair_undid_stressed_ball"] = undone(info, repair.keeper_pieces(stressed_track, halves, pitch))
        fold = joblib.load(fold_path(m))
        pred, v9l, v11l = inputs_of(fixed, halves, pitch, fold)
        if setting == "none":
            info["check"] = {k: max_diff(pred[k], f["pred_" + k]) for k in pred}
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
        g = info.get("groups", {})
        print("{} {}: lab cache written; {}; repair fired on {} piece(s), undid {} ({:.0f} s)".format(
            setting, m, {k: "{} pieces, {:.1%}".format(v["pieces"], v["share"]) for k, v in g.items()},
            sum(p["repaired"] for p in rpieces), info["repair_undid"], time.time() - started), flush=True)


def max_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "US" or b.dtype.kind in "US":
        return int(np.sum(a.astype(str) != b.astype(str)))
    a, b = a.astype(np.float64), b.astype(np.float64)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
    return float(d.max()) if d.size else 0.0


def undone(info, rpieces):
    """Flipped pieces the repair moved back to their true team: {pieces, frames} (of those flipped)."""
    flipped = [p for p in info.get("pieces", []) if "new_slot" in p]
    out = {"flipped_pieces": len(flipped), "flipped_frames": int(sum(p["end"] - p["start"] for p in flipped)),
           "undone_pieces": 0, "undone_frames": 0, "other_repairs": 0}
    for r in rpieces:
        if not r["repaired"]:
            continue
        hit = [p for p in flipped if p["new_slot"] == r["slot"] and r["start"] <= p["start"] < r["end"]]
        if hit:
            out["undone_pieces"] += len(hit)
            out["undone_frames"] += int(sum(min(p["end"], r["end"]) - p["start"] for p in hit))
        else:
            out["other_repairs"] += 1
    return out


# --------------------------------------------------------------------------------------------
# Features, P, trial 1015 on B's bench
# --------------------------------------------------------------------------------------------

def score(setting, matches):
    from bench import lab_branch_b as B
    from bench import lab_carried as C
    from bench import lab_gate, lab_temporal, search_b
    if not os.path.realpath(lab.ROOT).startswith(os.path.realpath(WORK)):
        raise RuntimeError("score runs only inside a setting's copy (scratchpad/trackerr/wc_SETTING)")
    for t in (0, TRIAL):
        if not os.path.exists(os.path.join(search_b.CACHE, "tables_trial{}.joblib".format(t))):
            raise RuntimeError("b16_c0's tables_trial{}.joblib must be in this copy (never rebuilt here)".format(t))
    search_b.cache_short_evidence()
    lab_gate.own_dead_cache()
    trial = search_b.every_trial()[TRIAL]
    out = setting_dir(setting)
    for m in matches:
        started = time.time()
        aug = search_b.fold_models(m)["augment"]
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
            np.save(B.p_path(variant, m), lab_gate.gru_proba(gru, c, halves).astype(np.float32))
            del c
            os.remove(feature_file)
        models = search_b.trial_models(trial, without=m)
        rows = []
        for variant in VARIANTS:
            match = lab_gate.variant_match(m, WHICH, variant)
            letters = search_b.letters_of(match, variant, {**trial, "post": "own"}, {"own": models})
            s = bench_score(match.truth, letters)
            rows.append({"setting": setting, "match": m, "source": sets.source(m), "variant": variant,
                         "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"]})
            codes = np.vectorize(LETTER_CODE.get, otypes=[np.int8])(np.asarray(letters).astype(str))
            np.save(os.path.join(out, "letters_{}_{}.npy".format(m, variant)), codes)
            # the start's frame log loss (the P the decoder reads), for where the points go
            p = search_b.start_p(m, variant, trial["w_gru"], trial["f_dead"])
            y = np.clip(match.truth.astype(np.int64), 0, 2)
            ll = -np.log(p[np.arange(len(y)), y])
            np.save(os.path.join(out, "logloss_{}_{}.npy".format(m, variant)), ll.astype(np.float16))
            if variant == VARIANTS[0]:
                np.save(os.path.join(out, "truth_{}.npy".format(m)), match.truth.astype(np.int8))
            del match
        pd.DataFrame(rows).to_csv(os.path.join(out, "scores_{}.csv".format(m)), index=False)
        print("{} {}: {} ({:.0f} s)".format(setting, m, " ".join("{} {:.5f}".format(r["variant"], r["spell_f1"]) for r in rows),
                                            time.time() - started), flush=True)


# --------------------------------------------------------------------------------------------
# A setting's copy of the repo
# --------------------------------------------------------------------------------------------

def setup(setting):
    """scratchpad/trackerr/wc_SETTING: b16_c0's code; data/external with the raw data linked, the trained models
    cloned (APFS clones: no disk) and every tracking-dependent cache empty."""
    wc = os.path.join(WORK, "wc_" + setting)
    os.makedirs(wc, exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "/data", "--exclude", "/viewer", "--exclude", "/docs", "--exclude", ".DS_Store",
                    "--exclude", "__pycache__", B16 + "/", wc + "/"], check=True)
    link = os.path.join(wc, "bench", "lab_trackerr.py")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.join(MAIN, "bench", "lab_trackerr.py"), link)
    ext, b16ext = os.path.join(wc, "data", "external"), os.path.join(B16, "data", "external")
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
            for src in sorted(glob.glob(os.path.join(b16ext, d, pattern))):
                dst = os.path.join(ext, d, os.path.basename(src))
                if not os.path.exists(dst):
                    subprocess.run(["cp", "-c", src, dst], check=True)
    print("{}: ready".format(wc))


def clean(setting):
    """Delete the copy's tracking-dependent caches (scores, letters and the corruption's record stay shared)."""
    ext = os.path.join(WORK, "wc_" + setting, "data", "external")
    for d in ("lab", "lab_robust", "lab_carried", "lab_gate", "lab_branch_b", "search_b", "lab_dead"):
        path = os.path.join(ext, d)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
    print("{}: caches removed".format(setting))


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------

def load_scores(setting):
    files = sorted(glob.glob(os.path.join(setting_dir(setting), "scores_*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def reproduction():
    mine = load_scores("none")
    ref = pd.read_csv(CONFIRM)
    ref = ref[ref.trial == TRIAL]
    j = ref.merge(mine, on=["match", "variant"], suffixes=("_v15", "_mine"))
    j["diff"] = (j.spell_f1_mine - j.spell_f1_v15).abs()
    return j, float(j["diff"].max()), len(j)


def zone_table(setting):
    """Frame accuracy of the letters (graded frames) by zone, against `none`, pooled over the units."""
    out = {z: {"frames": 0, "right_none": 0, "right": 0, "changed": 0, "right_to_wrong": 0, "wrong_to_right": 0,
               "ll_frames": 0, "logloss_none": 0.0, "logloss": 0.0} for z in ("near", "far", "none")}
    for path in sorted(glob.glob(os.path.join(setting_dir(setting), "letters_*.npy"))):
        name = os.path.basename(path)[len("letters_"):-len(".npy")]
        m, variant = name.rsplit("_", 1)
        base_path = os.path.join(setting_dir("none"), os.path.basename(path))
        if not os.path.exists(base_path):
            continue
        new, base = np.load(path), np.load(base_path)
        truth = np.load(os.path.join(setting_dir("none"), "truth_{}.npy".format(m)))
        zp = os.path.join(setting_dir(setting), "zone_{}.npy".format(m))
        zone = np.load(zp) if os.path.exists(zp) else np.zeros(len(truth), np.int8)
        graded = np.isin(truth, (0, 1, 2))
        for z, code in (("near", 2), ("far", 1), ("none", 0)):
            w = graded & (zone == code)
            r0, r1 = base[w] == truth[w], new[w] == truth[w]
            o = out[z]
            o["frames"] += int(w.sum())
            o["right_none"] += int(r0.sum())
            o["right"] += int(r1.sum())
            o["changed"] += int((base[w] != new[w]).sum())
            o["right_to_wrong"] += int((r0 & ~r1).sum())
            o["wrong_to_right"] += int((~r0 & r1).sum())
            lp, lb = (os.path.join(d, "logloss_{}_{}.npy".format(m, variant)) for d in (setting_dir(setting), setting_dir("none")))
            if os.path.exists(lp) and os.path.exists(lb):
                o["ll_frames"] += int(w.sum())
                o["logloss"] += float(np.load(lp)[w].astype(np.float64).sum())
                o["logloss_none"] += float(np.load(lb)[w].astype(np.float64).sum())
    return out


def summary(setting, base):
    s = load_scores(setting)
    if s is None:
        return None
    per_unit = s.merge(base, on=["match", "variant", "source"], suffixes=("", "_none"))
    per_unit["d"] = per_unit.spell_f1 - per_unit.spell_f1_none
    per_match = per_unit.groupby(["match", "source"])[["spell_f1", "spell_f1_none", "d"]].mean().reset_index()
    infos = {}
    for path in glob.glob(os.path.join(setting_dir(setting), "info_*.json")):
        with open(path) as handle:
            infos[os.path.basename(path)[5:-5]] = json.load(handle)
    out = {"setting": setting, "units": len(per_unit), "matches": len(per_match),
           "b_bench": float(per_match.spell_f1.mean()), "b_bench_none": float(per_match.spell_f1_none.mean()),
           "drop": float(per_match.d.mean()), "better": int((per_match.d > 1e-9).sum()),
           "worse": int((per_match.d < -1e-9).sum()),
           "per_variant": {v: float(g.d.mean()) for v, g in per_unit.groupby("variant")},
           "per_source": {k: float(g.d.mean()) for k, g in per_match.groupby("source")},
           "per_match": {r.match: {"score": float(r.spell_f1), "none": float(r.spell_f1_none), "d": float(r.d)}
                         for r in per_match.itertuples()},
           "zones": zone_table(setting) if setting != "none" else None}
    if infos:
        groups = [g for i in infos.values() for g in i.get("groups", {}).values()]
        out["affected_share"] = (float(sum(g["affected_frames"] for g in groups) / max(sum(g["eligible_frames"] for g in groups), 1))
                                 if groups else 0.0)
        out["appended_slots"] = int(sum(i.get("appended_slots", 0) for i in infos.values()))
        u = [i["repair_undid"] for i in infos.values() if "repair_undid" in i]
        out["repair"] = {k: int(sum(x[k] for x in u)) for k in u[0]} if u else None
        us = [i["repair_undid_stressed_ball"] for i in infos.values() if "repair_undid_stressed_ball" in i]
        out["repair_stressed_ball"] = {k: int(sum(x[k] for x in us)) for k in us[0]} if us else None
        out["per_match_corruption"] = {m: {"groups": i.get("groups"), "repair_undid": i.get("repair_undid")}
                                       for m, i in sorted(infos.items())}
    return out


def report():
    os.makedirs(RESULTS, exist_ok=True)
    j, worst, n = reproduction()
    print("reproduction: {} units, max |mine - v15's B| = {:.2e} ({})".format(n, worst, "PASSES" if n == 30 and worst <= 1e-5 else "FAILS"))
    j.to_csv(os.path.join(RESULTS, "reproduction.csv"), index=False, float_format="%.6f")
    base = load_scores("none")[["match", "variant", "source", "spell_f1"]]
    rows = []
    for setting in SETTINGS:
        s = summary(setting, base)
        if s is None:
            continue
        rows.append(s)
        print("{:<5s} B's bench {:.4f} (none {:.4f}): {:+.4f}, better/worse {}/{}; p25/p50/stressed {}; IDSSE/Metrica {}".format(
            setting, s["b_bench"], s["b_bench_none"], s["drop"], s["better"], s["worse"],
            " / ".join("{:+.4f}".format(s["per_variant"][v]) for v in VARIANTS if v in s["per_variant"]),
            " / ".join("{:+.4f}".format(s["per_source"][k]) for k in ("idsse", "metrica") if k in s["per_source"])))
        if s.get("zones"):
            for z, o in s["zones"].items():
                if o["frames"]:
                    print("      {:<4s}: {:>8d} graded frames, accuracy {:.4f} -> {:.4f} ({:+.4f}), changed {:.2%}".format(
                        z, o["frames"], o["right_none"] / o["frames"], o["right"] / o["frames"],
                        (o["right"] - o["right_none"]) / o["frames"], o["changed"] / o["frames"]))
        if s.get("repair"):
            print("      repair: {}".format(s["repair"]))
    with open(os.path.join(RESULTS, "report.json"), "w") as handle:
        json.dump({"reproduction": {"units": n, "max_abs_diff": worst}, "settings": rows}, handle, indent=1)
    for setting in SETTINGS:
        s = load_scores(setting)
        if s is not None:
            s.to_csv(os.path.join(RESULTS, "scores_{}.csv".format(setting)), index=False, float_format="%.6f")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("prep", "folds", "setup", "build", "score", "report", "clean"))
    parser.add_argument("setting", nargs="?", choices=list(SETTINGS))
    parser.add_argument("--matches", nargs="+", default=None)
    args = parser.parse_args()
    ms = args.matches or lab.matches(WHICH)
    if args.step == "prep":
        return prep(ms)
    if args.step == "folds":
        return folds()
    if args.step == "setup":
        return setup(args.setting)
    if args.step == "build":
        return build(args.setting, ms)
    if args.step == "score":
        return score(args.setting, ms)
    if args.step == "clean":
        return clean(args.setting)
    return report()


if __name__ == "__main__":
    main()
