"""v17's glue on a bench match (docs/models/v17-broadcast.md); never our match.

    python -m bench.check_v17 [--match J03WR9] [--variant stressed]       # ~20 min

On one bench match (b16_c0's lab cache: the bench's damaged tactical tracking, as track 3's harness read it; its ball
from a bench variant, by default the fully stressed one, where the switch takes branch B), the
central simulated broadcast (bench/lab_broadcast.simulate) is written as files 6-8 in docs/data.md's formats on its
own timeline (bench/lab_broadcast.port's writer and offsets), then run through possession/v17.py:

1. no broadcast (paths None) and a missing file: v17's letters equal v16's, frame for frame, and the branch is v16's;
   on branch B's own path, a missing file falls back ("fallback-missing") to the delivered match: the same letters;
2. with the files: the status is "merged" with either team order in the metadata; the alignment finds each half's
   shift within 2 frames (the central simulation's residual); and the tracking v17's branch B reads (after v8's
   repair) equals the repaired output of bench/lab_broadcast.merge(..., "M2") on the broadcast read back from the
   files and placed with the shifts found (the same merge every M2 score used), up to the files' text round trip;
   against the simulated arrays as scored (no alignment) the difference is reported as information;
3. predict() with the files takes v16's branch (and, when it is B, the status is "merged" and its letters are B's on
   the merged match); B's letters on the merged
   match are counted against v16's B letters
   (information only: the deployed models saw every bench match, so this is no score);
4. the delivered tracking is not modified by any of it.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np

from bench import lab, lab_broadcast as LB, lab_gate, sets
from bench import lab_trackerr as TE
from possession import broadcast_merge as BM, repair, v16, v17
from possession.data import HOME, PITCH_LENGTH, PITCH_WIDTH

# Params
WHICH = "all"
OUT = os.path.join(lab.RESULTS, "check_v17")
ROUND_TRIP_M = 1e-4           # the files keep 9 decimals of a 0-1 coordinate: 105 m x 5e-10 x rounding


def digest(track):
    h = hashlib.sha256()
    for k in ("players", "team", "ball", "speed"):
        h.update(np.ascontiguousarray(track[k]).tobytes())
    return h.hexdigest()


def write_files(d, m, f, halves, bro):
    """bench/lab_broadcast.port's files: the simulated broadcast on its own timeline (PORT_OFFSETS), its ball the
    clean ball seen by the camera plus noise, junk in the half-time gap."""
    clean = sets.load(m)
    nb = halves[1][2] + LB.PORT_OFFSETS[1] + 300
    rng = np.random.default_rng(0)
    BP = np.full((nb, bro["players"].shape[1], 2), np.nan, np.float32)
    BB = np.full((nb, 2), np.nan, np.float32)
    BV = np.zeros(nb, bool)
    for (_, a, b), o in zip(halves, LB.PORT_OFFSETS):
        t = np.arange(a, b + 1)
        BP[t + o] = bro["players"][t]
        BV[t + o] = bro["valid"][t]
        ball = clean["ball"][bro["src"][t], :2] + rng.normal(0, 0.5, (len(t), 2))
        ball[~bro["valid"][t]] = np.nan
        BB[t + o] = ball
    gap = np.arange(halves[0][2] + LB.PORT_OFFSETS[0] + 1, halves[1][1] + LB.PORT_OFFSETS[1])
    BP[gap[::7], :3] = rng.uniform(0, 60, (len(gap[::7]), 3, 2))
    paths = {k: os.path.join(d, v) for k, v in (("atd_meta", "meta.xml"), ("atd_raw", "raw.txt"),
                                                  ("homography", "homography.csv"))}

    def fmt(x, y):
        return "{:.9f},{:.9f},NaN".format(x / PITCH_LENGTH, y / PITCH_WIDTH) if x == x else "NaN,NaN,NaN"
    with open(paths["atd_raw"], "w") as h:
        for i in range(nb):
            h.write("{}:{}:{}\n".format(i + 1, ";".join(fmt(x, y) for x, y in BP[i]), fmt(*BB[i])))
    with open(paths["homography"], "w") as h:
        h.write("frame," + ",".join("h{}{}".format(a, b) for a in range(3) for b in range(3)) + ",reprojection_error,valid\n")
        for i in range(nb):
            h.write("{},{},0.1,{}\n".format(i, ",".join(["0"] * 9), int(BV[i])))
    return paths


def write_meta(path, bro_team, first):
    ids = ["T_A", "T_B"] if first == "home" else ["T_B", "T_A"]
    with open(path, "w") as h:
        h.write('<?xml version="1.0"?>\n<Main><Metadata><Teams>')
        for tid in ids:
            h.write('<Team id="{}"><Name>{}</Name></Team>'.format(tid, tid))
        h.write("</Teams><Players>")
        for k, side in enumerate(bro_team):
            h.write('<Player id="P{}" teamId="{}"><ShirtNumber>{}</ShirtNumber></Player>'.format(
                k + 1, "T_A" if side == HOME else "T_B", k + 1))
        h.write("</Players></Metadata></Main>\n")


def compare_tracks(a, b):
    same_shape = a["players"].shape == b["players"].shape and np.array_equal(a["team"], b["team"])
    if not same_shape:
        return {"same_slots": False, "shape": [list(a["players"].shape), list(b["players"].shape)]}
    na, nb_ = np.isnan(a["players"][:, :, 0]), np.isnan(b["players"][:, :, 0])
    return {"same_slots": True, "slot_frames_differ": int((na != nb_).sum()),
            "max_abs_diff_m": float(np.nanmax(np.abs(a["players"] - b["players"])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--match", default=LB.PORT_MATCH)
    parser.add_argument("--variant", default="stressed", choices=("normal", "p25", "p50", "stressed"))
    args = parser.parse_args()
    m = args.match
    f = LB.lab_inputs(m)
    halves, pitch = TE.halves_pitch(f)
    ball = f["ball"] if args.variant == "normal" else lab_gate.variant_match(m, WHICH, args.variant).track["ball"]
    track = TE.as_track(f["players"], f["team"], np.asarray(ball, np.float32))
    before = digest(track)
    out = {"match": m, "variant": args.variant, "rule": v17.RULE}

    # v16 as the reference
    l16, i16 = v16.predict(track, None, halves, pitch, WHICH)
    out["v16_branch"] = i16["branch"]
    print("v16 on {}: branch {}".format(m, i16["branch"]), flush=True)

    # 1. no broadcast, and a missing file
    d = tempfile.mkdtemp(dir=os.path.join(lab.ROOT, "data", "external"))
    try:
        l, i = v17.predict(track, None, halves, pitch, WHICH, paths=None)
        out["no_broadcast"] = {"status": i["broadcast"], "branch": i["branch"], "frames_differ_v16": int((l != l16).sum())}
        print("no broadcast:", out["no_broadcast"], flush=True)
        bro = LB.simulate(m, "central", sets.load(m), f["truth"], halves)
        paths = write_files(d, m, f, halves, bro)
        write_meta(paths["atd_meta"], bro["team"], "home")
        missing = dict(paths, atd_raw=os.path.join(d, "none.txt"))
        l, i = v17.predict(track, None, halves, pitch, WHICH, paths=missing)
        out["missing_file"] = {"status": i["broadcast"], "branch": i["branch"], "frames_differ_v16": int((l != l16).sum())}
        print("missing file:", out["missing_file"], flush=True)

        # 2. with the files: the tracking B reads against the harness's M2 merge, repaired. The reference takes the
        # broadcast as read back from the files and placed with the shifts the alignment finds on this ball (on a
        # stressed ball they can be a frame off the true ones: within the ±2-frame residual the central simulation
        # scored), so it checks the glue exactly; the merge on the simulated arrays as scored is information.
        bp, bb = BM.read_raw(paths["atd_raw"], len(bro["team"]))
        bv = BM.read_valid(paths["homography"], len(bp))
        found = [BM.global_shift(np.asarray(track["ball"], np.float64), bb, a, b) for (_, a, b) in halves]
        out["alignment"] = [{"true": -o, "found": s, "rms_m": r, "next_best_m": q}
                            for (s, r, q), o in zip(found, LB.PORT_OFFSETS)]
        print("alignment:", out["alignment"], flush=True)
        P, V = BM.on_tactical_timeline(bp, bv, [s for s, _, _ in found], halves, len(track["players"]))
        ref_p, ref_t, _, _, info = LB.merge(f["players"], f["team"], P, bro["team"], V, halves, "M2")
        ref, _ = repair.repair(TE.as_track(ref_p, ref_t, track["ball"]), halves, pitch)
        sim_p, sim_t, _, _, sim_info = LB.merge(f["players"], f["team"], bro["players"], bro["team"], bro["valid"], halves, "M2")
        sim, _ = repair.repair(TE.as_track(sim_p, sim_t, track["ball"]), halves, pitch)
        out["harness_merge"] = {"found_shifts": info, "as_scored": sim_info}
        for first in ("away", "home"):
            write_meta(paths["atd_meta"], bro["team"], first)
            mb, status = v17.b_match(track, halves, pitch, WHICH, paths)
            out["files_{}_first".format(first)] = dict(compare_tracks(mb.track, ref), status=status)
            out["files_{}_first".format(first)]["vs_as_scored"] = compare_tracks(mb.track, sim)
            print("files, {} first:".format(first), out["files_{}_first".format(first)], flush=True)

        # 3. predict with the files (metadata home first, as written last)
        l, i = v17.predict(track, None, halves, pitch, WHICH, paths=paths)
        out["predict_files"] = {"status": i["broadcast"], "branch": i["branch"], "branch_as_v16": i["branch"] == i16["branch"]}
        if i["branch"] == "A":
            out["predict_files"]["frames_differ_v16"] = int((l != l16).sum())
        delivered = v17.b_match(track, halves, pitch, WHICH, None)[0]
        lb16, _ = v17.b_letters(delivered, WHICH)
        fb, fb_status = v17.b_match(track, halves, pitch, WHICH, missing)
        lfb, _ = v17.b_letters(fb, WHICH)
        out["b_fallback"] = {"status": fb_status, "frames_differ_delivered_b": int((lfb != lb16).sum())}
        print("B's own path, missing file:", out["b_fallback"], flush=True)
        lb17, _ = v17.b_letters(mb, WHICH)
        if i["branch"] == "B":
            out["predict_files"]["frames_differ_b_merged"] = int((l != lb17).sum())
        out["b_letters_differ"] = {"frames": int((lb16 != lb17).sum()), "share": float((lb16 != lb17).mean())}
        print("predict with files:", out["predict_files"], "B letters differ:", out["b_letters_differ"], flush=True)
    finally:
        shutil.rmtree(d)

    # 4. the delivered tracking untouched
    out["delivered_unchanged"] = digest(track) == before
    expect = ("off", "fallback-missing") if i16["branch"] == "B" else ("not-used", "not-used")
    ok = (out["no_broadcast"]["frames_differ_v16"] == 0 and out["no_broadcast"]["branch"] == i16["branch"]
          and out["no_broadcast"]["status"] == expect[0]
          and out["missing_file"]["frames_differ_v16"] == 0 and out["missing_file"]["status"] == expect[1]
          and out["b_fallback"]["status"] == "fallback-missing" and out["b_fallback"]["frames_differ_delivered_b"] == 0
          and all(out["files_{}_first".format(s)]["status"] == "merged" and out["files_{}_first".format(s)]["same_slots"]
                  and out["files_{}_first".format(s)]["slot_frames_differ"] == 0
                  and out["files_{}_first".format(s)]["max_abs_diff_m"] <= ROUND_TRIP_M for s in ("away", "home"))
          and out["predict_files"]["branch_as_v16"] and out["predict_files"].get("frames_differ_v16", 0) == 0
          and out["predict_files"].get("frames_differ_b_merged", 0) == 0
          and all(abs(a["found"] - a["true"]) <= 2 for a in out["alignment"])
          and out["predict_files"]["status"] == ("merged" if i16["branch"] == "B" else "not-used")
          and out["delivered_unchanged"])
    out["passes"] = bool(ok)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "glue_{}_{}.json".format(m, args.variant)), "w") as handle:
        json.dump(out, handle, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print("check_v17 on {} ({}): {}".format(m, args.variant, "PASSES" if ok else "FAILS"), flush=True)


if __name__ == "__main__":
    main()
