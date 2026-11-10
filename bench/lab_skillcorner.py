"""Lab round 7, `skillcorner`: v12 on SkillCorner's 20 open matches, a test it has never seen.

    python -m bench.skillcorner download && python -m bench.skillcorner parse   # data and labels (bench/skillcorner.py)
    python -m bench.lab_skillcorner run [ids]      # v12 (and v10 / v11) on each match: ~3-5 min a match
    python -m bench.lab_skillcorner report         # scores, per stage, label-free numbers, the bench beside them

Nothing here is trained or tuned on SkillCorner: v12 runs exactly as on our match, `v12.predict(tracking,
tags=None, halves, pitch)` with the deployment models (possession/models/v12_all.joblib). The stages are
recomputed from the same inputs only to read the letters after each one; the last must equal predict's.
Letters are kept in data/external/skillcorner/predictions/ (gitignored), scores in
bench/results/lab_skillcorner*.

SkillCorner's labels (bench/skillcorner.py) were fixed before anything was scored. One secondary score,
also fixed before: the truth with SkillCorner's frames that have no tracking at all (broadcast cut-aways)
made X, so only frames SkillCorner actually tracked are graded.

Added after seeing scores, to explain them (never to choose anything): two oracles (the truth's stoppages
with v12's teams; the truth's team where both say live), a post hoc mix (v11's stoppages, v12's teams), the
organisers' baseline, where the ball is in the truth's stoppages, team-change timing against the truth, and
how close the new team is to the ball at a true change (SkillCorner's labels and the bench's, each on its
own tracking). docs/lab/skillcorner.md has the results.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from bench import lab, lab_robust, lab_segments, lab_short, lab_timing, skillcorner, transfer_check
from bench import run as bench_run
from bench.run import CODE
from possession import repair, v10, v11, v12
from possession.data import ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
WHICH = "all"                     # v12's deployment set (v10.BENCH_SET)
PRED = os.path.join(skillcorner.FOLDER, "predictions")
RESULTS = os.path.join(ROOT, "bench", "results")
STAGES = ("robust_frames", "robust_decoded", "robust", "short", "segments", "timing")
REFERENCES = ("v10", "v11", "baseline_raw", "baseline_smooth25", "baseline_smooth125")
BASELINES = ("baseline_raw", "baseline_smooth25", "baseline_smooth125")


def pred_path(match):
    return os.path.join(PRED, "{}.npz".format(match))


# --------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------

def stage_letters(match, models):
    """Letters after each stage of v12's chain (v12.run_chain, step by step)."""
    settings = models["settings"]
    p = v12.robust_proba(match, models["robust"], WHICH)
    out = {"robust_frames": np.array(["H", "A", "D"])[np.argmax(p, axis=1)]}
    out["robust_decoded"] = np.asarray(lab_robust.L.decode_cost(match, -np.log(p), **settings["robust"])).astype("<U1")
    out["robust"] = np.asarray(lab_robust.finish(match, out["robust_decoded"], WHICH)).astype("<U1")
    out["short"] = lab_short.apply(match, out["robust"], model=models["short"], **settings["short"]).astype("<U1")
    out["segments"] = lab_segments.apply(match, out["short"], models["segments"], **settings["segments"]).astype("<U1")
    out["timing"] = lab_timing.apply(match, out["segments"], models["timing"], **settings["timing"]).astype("<U1")
    return out, p


def run_match(match_id, references=True):
    data = skillcorner.load(match_id)
    track, halves, pitch = skillcorner.tracking(data), data["halves"], data["pitch"]
    started = time.time()
    labels, info = v12.predict(track, tags=None, halves=halves, pitch=pitch, bench_set=WHICH)
    t_v12 = time.time() - started
    fixed, _ = repair.repair(track, halves, pitch)
    match = v12.as_match(fixed, info["pred"], halves, pitch, "skillcorner_{}".format(match_id))
    stages, p = stage_letters(match, v12.load_models(WHICH))
    differ = int(np.sum(stages["timing"] != labels))
    if differ:
        raise RuntimeError("{}: the stage-by-stage chain differs from v12.predict on {} frames".format(match_id, differ))
    out = {"v12": labels.astype("<U1"), **{"stage_" + k: v for k, v in stages.items()},
           "p_robust": p.astype(np.float32), "p_pathcrf": info["pred"]["p_pathcrf"].astype(np.float32),
           "p_dead": info["pred"]["p_dead"].astype(np.float32)}
    t_ref = 0.0
    if references:
        started = time.time()
        letters10, info10 = v10.predict(track, [], halves, pitch, WHICH)
        s = v11.SETTINGS[WHICH]
        letters11, _ = v11.move(letters10, info10["fixed"], halves, s["dv"], s["back"], s["ahead"])
        out.update(v10=np.asarray(letters10).astype("<U1"), v11=np.asarray(letters11).astype("<U1"))
        t_ref = time.time() - started
    os.makedirs(PRED, exist_ok=True)
    np.savez_compressed(pred_path(match_id), **out)
    print("{}: v12 {:.0f} s (+ stages), v10/v11 {:.0f} s; stages reproduce predict".format(match_id, t_v12, t_ref), flush=True)


def run(ids):
    for m in ids:
        if os.path.exists(pred_path(m)):
            print("{}: have".format(m), flush=True)
            continue
        run_match(m)


# --------------------------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------------------------

def score(gt, letters):
    """spellF1, macroF1, mean spell precision / recall over the IoUs, possessions, team accuracy live."""
    pred = np.array([CODE[x] for x in np.asarray(letters).tolist()], dtype=np.int8)
    r = pm.evaluate(gt, pred)
    per = r["spell_f1"]
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    return {"spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
            "precision": float(np.mean([v["precision"] for v in per.values()])),
            "recall": float(np.mean([v["recall"] for v in per.values()])),
            "f1_iou_0.3": per[0.3]["f1"], "f1_iou_0.7": per[0.7]["f1"],
            "possessions": r["n_pred_spells"], "true_possessions": r["n_gt_spells"],
            "team_acc_alive": r["team_acc_both_alive"], "alive_dead_f1": r["alive_dead_f1"],
            "d_f1": pm.per_class_prf(cm3, pm.D)[2], "frames_right": r["accuracy"]}


def team_accuracy(gt, letters, mask):
    """H-vs-A accuracy where truth and letters both say live, inside `mask`; and those frames' count."""
    letters = np.asarray(letters)
    both = mask & np.isin(gt, (pm.H, pm.A)) & np.isin(letters, ["H", "A"])
    right = (gt == pm.H) & (letters == "H") | (gt == pm.A) & (letters == "A")
    return float(np.mean(right[both])) if both.any() else float("nan"), int(both.sum())


def oracle_dead(gt, letters, halves):
    """Diagnosis only: `letters` with the truth's stoppages (D where the truth says D; where the truth
    is live and the letters say D, the letters' last team, else their next one), teams as predicted."""
    out = np.asarray(letters).astype("<U1").copy()
    out[gt == pm.D] = "D"
    for _, a, b in halves:
        w = out[a:b + 1]
        live = np.isin(gt[a:b + 1], (pm.H, pm.A))
        team = np.isin(w, ["H", "A"])
        if not team.any():
            continue
        carry = np.maximum.accumulate(np.where(team, np.arange(len(w)), -1))
        carry[carry < 0] = np.flatnonzero(team)[0]
        fill = live & ~team
        w[fill] = w[carry][fill]
        out[a:b + 1] = w
    return out


def oracle_team(gt, letters):
    """Diagnosis only: `letters` with the truth's team wherever both say live (stoppages as predicted)."""
    out = np.asarray(letters).astype("<U1").copy()
    both = np.isin(gt, (pm.H, pm.A)) & np.isin(out, ["H", "A"])
    out[both] = np.array(["H", "A", "D", "X"])[gt[both]]
    return out


def match_rows(match_id):
    data = skillcorner.load(match_id)
    gt, halves = data["labels"], data["halves"]
    with np.load(pred_path(match_id)) as z:
        letters = {k: z[k] for k in z.files if k in ("v12", "v10", "v11") or k.startswith("stage_")}
    track = skillcorner.tracking(data)
    for name in BASELINES:                                  # the organisers' baseline, as bench/run.py runs it
        letters[name] = bench_run.MODELS[name](track, halves, data["pitch"])
    letters["oracle_dead_v12"] = oracle_dead(gt, letters["v12"], halves)
    letters["oracle_team_v12"] = oracle_team(gt, letters["v12"])
    if "v11" in letters:
        letters["oracle_dead_v11"] = oracle_dead(gt, letters["v11"], halves)
        # post hoc, no setting: v11's stoppages with v12's teams (a lead for the bench, not a candidate)
        v11_codes = np.array([CODE[x] for x in letters["v11"].tolist()], dtype=np.int8)
        letters["v11_dead_v12_teams"] = oracle_dead(v11_codes, letters["v12"], halves)
    seen = ~np.isnan(data["ball"][:, 0])
    gt_tracked = gt.copy()
    gt_tracked[data["no_data"]] = pm.X
    rows = []
    for name, l in letters.items():
        row = {"match": match_id, "letters": name, **score(gt, l)}
        tracked = score(gt_tracked, l)
        row.update({"tracked_" + k: tracked[k] for k in ("spell_f1", "macro_f1", "possessions", "true_possessions")})
        for part, mask in (("ball_seen", ~np.isnan(data["ball"][:, 0]) & ~data["no_data"]),
                           ("ball_detected", data["ball_detected"] & ~data["no_data"]),
                           ("ball_not_detected", ~data["ball_detected"] & ~data["no_data"]),
                           ("no_tracking", data["no_data"])):
            row["team_acc_" + part], row["frames_" + part] = team_accuracy(gt, l, mask)
        dead = gt == pm.D
        called = np.asarray(l) == "D"
        row.update({"dead_share_called": float(np.mean(called[gt != pm.X])),
                    "dead_recall_ball_seen": float(np.mean(called[dead & seen])),
                    "dead_recall_ball_missing": float(np.mean(called[dead & ~seen])) if (dead & ~seen).any() else float("nan"),
                    "live_called_dead": float(np.mean(called[np.isin(gt, (pm.H, pm.A))])),
                    "true_dead_ball_seen_share": float(np.mean(seen[dead]))})
        if name in ("v12", "v11", "v10", "stage_robust", "stage_short", "stage_segments"):
            row.update({"label_free: " + k: v for k, v in transfer_check.stats(l, track, halves).items()})
        rows.append(row)
    truth_letters = np.array(["H", "A", "D", "X"])[gt]
    rows.append({"match": match_id, "letters": "truth",
                 **{"label_free: " + k: v for k, v in transfer_check.stats(truth_letters, track, halves).items()}})
    return rows


LENGTHS = ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, np.inf))   # seconds, as docs/inside-v12.md


def errors(match_id, name="v12"):
    """True possessions by length (found at IoU 0.3 / 0.7), and the missed (0.3) and spurious ones: how
    many touch a stretch with no tracking at all, and how much of them the ball is detected."""
    data = skillcorner.load(match_id)
    gt = data["labels"]
    with np.load(pred_path(match_id)) as z:
        letters = z[name]
    pred = pm.fill_abstentions(np.array([CODE[x] for x in letters.tolist()], dtype=np.int8), pm.valid_runs(gt))
    runs = pm.valid_runs(gt)
    g, p = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(g, p)
    found = {t: pm.matched_gt_flags(overlaps, len(g), t) for t in (0.3, 0.7)}
    mirrored = sorted(((iou, j, i) for iou, i, j in overlaps), key=lambda x: (-x[0], x[1], x[2]))
    kept = pm.matched_gt_flags(mirrored, len(p), 0.3)
    no_data, detected = data["no_data"], data["ball_detected"]
    out = {"lengths": {}, "missed": {}, "spurious": {}}
    length = np.array([(b - a) / skillcorner.FPS for a, b, _ in g])
    for lo, hi in LENGTHS:
        k = (length >= lo) & (length < hi)
        out["lengths"]["{}-{}".format(lo, hi)] = {"true": int(k.sum()), "found_0.3": int((found[0.3] & k).sum()),
                                                   "found_0.7": int((found[0.7] & k).sum())}
    for kind, spells, flags in (("missed", g, ~found[0.3]), ("spurious", p, ~kept)):
        sel = [s for s, f in zip(spells, flags) if f]
        touch = [no_data[a:b].any() for a, b, _ in sel]
        out[kind] = {"count": len(sel), "touch_no_tracking": int(np.sum(touch)),
                     "median_s": float(np.median([(b - a) / skillcorner.FPS for a, b, _ in sel])) if sel else 0.0,
                     "ball_detected_share": float(np.mean([detected[a:b].mean() for a, b, _ in sel])) if sel else 0.0}
    out["true_touch_no_tracking"] = int(np.sum([no_data[a:b].any() for a, b, _ in g]))
    out["true"] = len(g)
    out["predicted"] = len(p)
    return out


OUT_MARGIN = 0.5      # m beyond a line: the ball is out of the pitch


def dead_diagnosis(match_id):
    """Where the truth says D: how often the ball is seen, moving, or beyond a line, and robust's P(D)
    there; the same for live frames. Diagnosis only."""
    data = skillcorner.load(match_id)
    gt, ball, (length, width) = data["labels"], data["ball"], data["pitch"]
    with np.load(pred_path(match_id)) as z:
        p = z["p_robust"]
    seen = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        out = (ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > length + OUT_MARGIN) | (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > width + OUT_MARGIN)
        speed = np.full(len(ball), np.nan)
        speed[:-5] = np.hypot(*(ball[5:] - ball[:-5]).T) * (skillcorner.FPS / 5)
        moving = speed >= 0.3
    res = {}
    for name, mask in (("dead", gt == pm.D), ("live", np.isin(gt, (pm.H, pm.A)))):
        res[name] = {"ball_seen": float(seen[mask].mean()), "ball_seen_moving": float(moving[mask & seen].mean()),
                     "ball_seen_out": float(out[mask & seen].mean()),
                     "p_dead_ball_seen": float(p[mask & seen, 2].mean()),
                     "p_dead_ball_missing": float(p[mask & ~seen, 2].mean()) if (mask & ~seen).any() else float("nan")}
    return res


CHANGE_WINDOW = 125   # frames (5 s): a true H <-> A change is "met" by a predicted one to the same team this close


def team_changes(letters):
    """[(frame, new team)] for every H <-> A change (not through a stoppage)."""
    letters = np.asarray(letters).astype("<U1")
    k = np.flatnonzero(letters[1:] != letters[:-1]) + 1
    return [(int(i), letters[i]) for i in k if letters[i - 1] in ("H", "A") and letters[i] in ("H", "A")]


def change_offsets(truth_letters, letters):
    """Signed seconds from each true H <-> A change to the nearest predicted change to the same team within
    CHANGE_WINDOW; and the number of true changes."""
    pred = team_changes(letters)
    frames = np.array([f for f, _ in pred])
    teams = np.array([t for _, t in pred])
    out = []
    for f, t in team_changes(truth_letters):
        near = (teams == t) & (np.abs(frames - f) <= CHANGE_WINDOW) if len(frames) else np.zeros(0, bool)
        if near.any():
            j = np.argmin(np.where(near, np.abs(frames - f), np.inf))
            out.append((frames[j] - f) / skillcorner.FPS)
    return np.array(out), len(team_changes(truth_letters))


def offsets_summary(pairs):
    off = np.concatenate([o for o, _ in pairs])
    n = sum(k for _, k in pairs)
    return {"true changes a match": n / len(pairs), "met within 5 s": len(off) / n, "median s": float(np.median(off)),
            "quartiles s": np.percentile(off, [25, 75]).tolist(), "within 0.5 s": float(np.mean(np.abs(off) <= 0.5))}


def timing_offsets(ids):
    """Team-change timing against the truth: SkillCorner (v12, v11) and the bench (v12 out of fold)."""
    from bench import chain, train_v12
    letters = np.array(["H", "A", "D", "X"])
    out = {}
    for name in ("v12", "v11"):
        pairs = []
        for m in ids:
            gt = letters[skillcorner.load(m)["labels"]]
            with np.load(pred_path(m)) as z:
                pairs.append(change_offsets(gt, z[name]))
        out["skillcorner " + name] = offsets_summary(pairs)
    pairs = []
    for m in lab.each(WHICH):
        oof = np.load(os.path.join(train_v12.OOF + chain.suffix(WHICH), "{}.npy".format(m.id))).astype("<U1")
        pairs.append(change_offsets(m.truth_letters.astype("<U1"), oof))
    out["bench v12 (out of fold)"] = offsets_summary(pairs)
    return out


SHIFTS = (-25, -12, 0, 12, 25)    # frames around a true change: -1, -0.5, 0, +0.5, +1 s


def change_geometry(gt_letters, ball, players, team):
    """At each true H <-> A change and around it: the new team's nearest distance to the ball (median, m),
    and the share of changes with that player within 1.5 m at the change. Labels against their own tracking."""
    rows = []
    for f, t in team_changes(gt_letters):
        side = 0 if t == "H" else 1
        row = []
        for dt in SHIFTS:
            k = f + dt
            if not 0 <= k < len(ball) or np.isnan(ball[k, 0]):
                row.append(np.nan)
                continue
            d = np.hypot(*(players[k, team == side] - ball[k]).T)
            row.append(np.nanmin(d) if np.any(~np.isnan(d)) else np.nan)
        rows.append(row)
    a = np.array(rows)
    return {"median m at -1, -0.5, 0, +0.5, +1 s": np.nanmedian(a, axis=0).tolist(),
            "within 1.5 m at the change": float(np.nanmean(a[:, 2] <= 1.5)), "changes": len(a)}


def change_geometry_all(ids):
    """change_geometry on SkillCorner's 20 (its labels, its tracking) and the bench's 10 (truth, clean tracking)."""
    from bench import sets
    letters = np.array(["H", "A", "D", "X"])
    out = {}
    for m in ids:
        d = skillcorner.load(m)
        out[m] = change_geometry(letters[d["labels"]], d["ball"], d["players"], d["team"])
    for m in sets.matches("all"):
        d = sets.load(m)
        out[m] = change_geometry(letters[sets.truth(m, d)], d["ball"], d["players"], d["team"])
        del d
    groups = {"skillcorner": list(ids), "idsse": [m for m in out if m.startswith("J")],
              "metrica": [m for m in out if m.startswith("metrica")]}
    return {"per_match": out, **{g: {"median m at -1, -0.5, 0, +0.5, +1 s": np.mean([out[m]["median m at -1, -0.5, 0, +0.5, +1 s"] for m in ms], axis=0).tolist(),
                                     "within 1.5 m at the change": float(np.mean([out[m]["within 1.5 m at the change"] for m in ms]))}
                                 for g, ms in groups.items()}}


def bench_reference():
    """The bench's own numbers to stand beside SkillCorner's: v12 and its stages out of fold
    (bench/results/inside_v12.json), and the organisers' baseline on all 10 matches, clean and at 41 %
    damage (bench/results/scores.csv, raw labels)."""
    with open(os.path.join(RESULTS, "inside_v12.json")) as handle:
        inside = json.load(handle)
    c = inside["confusion"]
    scores = pd.read_csv(os.path.join(RESULTS, "scores.csv"))
    scores = scores[(scores["labels"] == "raw") & scores["model"].isin(BASELINES)]
    base = scores.groupby(["model", "level"]).spell_f1.agg(["mean", "count"]).reset_index()
    return {"versions": inside["versions"], "team_acc_alive_v12": (c[0][0] + c[1][1]) / (c[0][0] + c[0][1] + c[1][0] + c[1][1]),
            "by_length": inside["by_length"],
            "baselines": {"{} at {} %".format(r.model, r.level): {"spell_f1": float(r["mean"]), "matches": int(r["count"])}
                          for _, r in base.iterrows() if r.level in (0, 41)}}


def bench_label_checks():
    """The same label checks on the bench's own truth, clean tracking (bench/idsse.py, bench/metrica.py caches):
    so SkillCorner's numbers have something to stand beside."""
    from bench import sets
    out = {}
    for m in sets.matches("all"):
        d = sets.load(m)
        gt = sets.truth(m, d)
        n = len(gt)
        seen = ~np.isnan(d["ball"][:, 0])
        data = {"labels": gt, "halves": d["halves"], "ball": d["ball"], "players": d["players"], "team": d["team"],
                "ball_detected": seen, "no_data": np.zeros(n, bool), "players_detected": np.ones(n, np.float32),
                "fill_counts": {}}
        out[m] = skillcorner.label_checks(data)
        print("bench {}: {}".format(m, {k: round(v, 3) for k, v in out[m].items() if isinstance(v, float)}), flush=True)
        del d, data
    return out


def report():
    ids = [m for m in skillcorner.match_ids() if os.path.exists(pred_path(m))]
    rows = []
    for m in ids:
        rows += match_rows(m)
        print("{}: scored".format(m), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(RESULTS, "lab_skillcorner.csv"), index=False, float_format="%.5f")
    with open(os.path.join(RESULTS, "lab_skillcorner_labels.json")) as handle:
        label_checks = json.load(handle)
    bench_labels_path = os.path.join(RESULTS, "lab_skillcorner_bench_labels.json")
    if not os.path.exists(bench_labels_path):
        with open(bench_labels_path, "w") as handle:
            json.dump(bench_label_checks(), handle, indent=1)
    with open(bench_labels_path) as handle:
        bench_labels = json.load(handle)
    with open(transfer_check.RANGES) as handle:
        ranges = json.load(handle)
    summary = summarise(frame, label_checks, bench_labels, ranges, ids)
    summary["bench"] = bench_reference()
    summary["timing_offsets"] = timing_offsets(ids)
    summary["change_geometry"] = change_geometry_all(ids)
    dd = {m: dead_diagnosis(m) for m in ids}
    summary["dead_diagnosis"] = {"per_match": dd, **{side: {k: float(np.nanmean([d[side][k] for d in dd.values()])) for k in dd[ids[0]][side]}
                                                     for side in ("dead", "live")}}
    errs = {m: errors(m) for m in ids}
    summary["errors_v12"] = {"per_match": errs,
                             "lengths": {k: {c: int(sum(e["lengths"][k][c] for e in errs.values())) for c in ("true", "found_0.3", "found_0.7")}
                                         for k in errs[ids[0]]["lengths"]},
                             **{kind: {c: float(np.mean([e[kind][c] for e in errs.values()])) for c in errs[ids[0]][kind]}
                                for kind in ("missed", "spurious")},
                             "true": float(np.mean([e["true"] for e in errs.values()])),
                             "true_touch_no_tracking": float(np.mean([e["true_touch_no_tracking"] for e in errs.values()]))}
    with open(os.path.join(RESULTS, "lab_skillcorner.json"), "w") as handle:
        json.dump(summary, handle, indent=1, default=float)
    show(summary)


def summarise(frame, label_checks, bench_labels, ranges, ids):
    numeric = frame.drop(columns=["match"]).groupby("letters").agg(["mean", "min", "max"])
    means = {name: {col: float(numeric.loc[name, (col, "mean")]) for col in numeric.columns.levels[0]
                    if not np.isnan(numeric.loc[name, (col, "mean")])} for name in numeric.index}
    v12_rows = frame[frame.letters == "v12"].set_index("match")
    per_match = {m: {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                     for k, v in v12_rows.loc[m].items()} for m in v12_rows.index}
    stages = {}
    previous = None
    for name in ["stage_" + s for s in STAGES]:
        rows = frame[frame.letters == name].set_index("match").spell_f1
        stages[name] = {"spell_f1": float(rows.mean()), "min": float(rows.min()), "max": float(rows.max())}
        if previous is not None:
            gain = rows - previous
            stages[name].update(gain=float(gain.mean()), better=int((gain > 0).sum()), matches=len(gain))
        previous = rows
    refs = {}
    v12 = frame[frame.letters == "v12"].set_index("match").spell_f1
    for name in REFERENCES:
        rows = frame[frame.letters == name].set_index("match").spell_f1
        if len(rows):
            refs[name] = {"spell_f1": float(rows.mean()), "v12_gain": float((v12 - rows).mean()),
                          "v12_better": int(((v12 - rows) > 0).sum()), "matches": len(rows)}
    # label-free numbers, v12 on SkillCorner against the bench's ranges
    lf = {}
    for k, (lo, hi) in ranges["ranges"].items():
        vals = frame[frame.letters == "v12"]["label_free: " + k]
        truth = frame[frame.letters == "truth"]["label_free: " + k] if k != "clear control" else None
        lf[k] = {"bench range (v12 out of fold)": [lo, hi], "skillcorner v12 mean": float(vals.mean()),
                 "skillcorner v12 range": [float(vals.min()), float(vals.max())],
                 "inside the bench range": int(((vals >= lo) & (vals <= hi)).sum()), "matches": len(vals)}
        if truth is not None:
            lf[k]["skillcorner truth mean"] = float(truth.mean())
            lf[k]["skillcorner truth range"] = [float(truth.min()), float(truth.max())]
            bench_truth = [r["truth: " + k] for r in ranges["per_match"].values()]
            lf[k]["bench truth range"] = [float(min(bench_truth)), float(max(bench_truth))]
    labels = {k: {"mean": float(np.mean([label_checks[m][k] for m in ids])),
                  "range": [float(min(label_checks[m][k] for m in ids)), float(max(label_checks[m][k] for m in ids))]}
              for k in label_checks[ids[0]] if isinstance(label_checks[ids[0]][k], (int, float))}
    bench = {}
    for src, names in (("idsse", [m for m in bench_labels if not m.startswith("metrica")]),
                       ("metrica", [m for m in bench_labels if m.startswith("metrica")])):
        bench[src] = {k: {"mean": float(np.mean([bench_labels[m][k] for m in names])),
                          "range": [float(min(bench_labels[m][k] for m in names)), float(max(bench_labels[m][k] for m in names))]}
                      for k in bench_labels[names[0]] if isinstance(bench_labels[names[0]][k], (int, float))}
    corr = float(np.corrcoef(v12_rows.spell_f1, v12_rows.true_possessions / [label_checks[m]["minutes"] for m in v12_rows.index])[0, 1])
    drivers = {"true possessions per minute": [label_checks[m]["possessions per minute"] for m in v12_rows.index],
               "no tracking share": [label_checks[m]["no tracking share"] for m in v12_rows.index],
               "reading check (labels)": [label_checks[m]["reading check"] for m in v12_rows.index],
               "dead recall, ball seen": v12_rows.dead_recall_ball_seen.tolist(),
               "team accuracy live": v12_rows.team_acc_alive.tolist(),
               "alive/dead F1": v12_rows.alive_dead_f1.tolist()}
    correlations = {k: float(np.corrcoef(v12_rows.spell_f1, v)[0, 1]) for k, v in drivers.items()}
    return {"matches": ids, "means": means, "per_match_v12": per_match, "stages": stages, "references": refs,
            "label_free": lf, "labels": labels, "bench_labels": bench,
            "corr_spell_f1_vs_true_possessions_per_minute": corr, "correlations_v12_spell_f1": correlations}


def show(s):
    m = s["means"]["v12"]
    print("\nv12 on {} SkillCorner matches: spellF1 {:.4f} (range {:.4f}-{:.4f}), macroF1 {:.4f}, precision {:.3f}, "
          "recall {:.3f}, possessions {:.0f} vs true {:.0f}, team accuracy live {:.3f}".format(
              len(s["matches"]), m["spell_f1"], min(r["spell_f1"] for r in s["per_match_v12"].values()),
              max(r["spell_f1"] for r in s["per_match_v12"].values()), m["macro_f1"], m["precision"], m["recall"],
              m["possessions"], m["true_possessions"], m["team_acc_alive"]))
    print("  tracked frames only: spellF1 {:.4f}, macroF1 {:.4f}".format(m["tracked_spell_f1"], m["tracked_macro_f1"]))
    for name, v in s["stages"].items():
        print("  {:<22s} {:.4f}{}".format(name, v["spell_f1"], "  ({:+.4f}, better on {}/{})".format(
            v["gain"], v["better"], v["matches"]) if "gain" in v else ""))
    for name, v in s["references"].items():
        print("  {} {:.4f}: v12 {:+.4f}, better on {}/{}".format(name, v["spell_f1"], v["v12_gain"], v["v12_better"], v["matches"]))
    for k, v in s["label_free"].items():
        print("  {}: {}".format(k, {a: (np.round(b, 3).tolist() if isinstance(b, list) else round(b, 3) if isinstance(b, float) else b)
                                    for a, b in v.items()}))
    print("  v12's spellF1 across matches, correlation with: {}".format(
        {k: round(v, 2) for k, v in s["correlations_v12_spell_f1"].items()}))
    for name in ("v12", "oracle_dead_v12", "oracle_team_v12", "v11", "oracle_dead_v11", "v11_dead_v12_teams"):
        if name in s["means"]:
            v = s["means"][name]
            print("  {:<16s} spellF1 {:.4f} macroF1 {:.4f} team live {:.3f} alive/dead F1 {:.3f} dead called {:.3f} "
                  "dead recall ball seen {:.3f} / missing {:.3f}".format(
                      name, v["spell_f1"], v["macro_f1"], v["team_acc_alive"], v["alive_dead_f1"], v["dead_share_called"],
                      v["dead_recall_ball_seen"], v["dead_recall_ball_missing"]))
    for k, v in s.get("timing_offsets", {}).items():
        print("  team changes, {}: {}".format(k, {a: (np.round(b, 2).tolist() if isinstance(b, list) else round(b, 3)) for a, b in v.items()}))
    for k in ("skillcorner", "idsse", "metrica"):
        v = s["change_geometry"][k]
        print("  at a true team change, {}: the new team's nearest (m) at -1..+1 s {}, within 1.5 m at it {:.2f}".format(
            k, np.round(v["median m at -1, -0.5, 0, +0.5, +1 s"], 2).tolist(), v["within 1.5 m at the change"]))
    d = s.get("dead_diagnosis")
    if d:
        print("  dead diagnosis: {}".format({k: {a: round(b, 3) for a, b in d[k].items()} for k in ("dead", "live")}))
    e = s.get("errors_v12")
    if e:
        n = len(s["matches"])
        for k, v in e["lengths"].items():
            print("  true {:>8s} s: {:5.1f} a match, found {:.0%} at IoU 0.3, {:.0%} at 0.7".format(
                k, v["true"] / n, v["found_0.3"] / max(v["true"], 1), v["found_0.7"] / max(v["true"], 1)))
        for kind in ("missed", "spurious"):
            print("  {}: {:.1f} a match, median {:.1f} s, touching a stretch with no tracking {:.1f}, ball detected {:.0%}".format(
                kind, e[kind]["count"], e[kind]["median_s"], e[kind]["touch_no_tracking"], e[kind]["ball_detected_share"]))
        print("  true possessions touching a stretch with no tracking: {:.1f} of {:.1f}".format(e["true_touch_no_tracking"], e["true"]))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    if what == "run":
        run(sys.argv[2:] or skillcorner.match_ids())
    elif what == "report":
        report()


if __name__ == "__main__":
    main()
