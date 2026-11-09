"""An X-ray of v12 on the bench: what each stage does, what it gets right, what's left (docs/inside-v12.md).

    python -m bench.inside_v12          # ~3 min; after `python -m bench.train_v12 --oof-only`

Exploration: the truth is read on every match, so nothing here may pick a setting. Every letter comes
from models trained without the match (out of fold). Prints the tables and writes:
    bench/results/inside_v12.json          the same tables
    bench/results/inside_v12_errors.csv    every missed and spurious possession, with where to find it
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from bench import chain, lab, lab_robust, train_v12
from possession import touches, v11
from possession.data import AWAY, FPS, HOME, ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
WHICH = "all"
STAGES = ("robust", "short", "segments", "timing")
CODE = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}
LENGTHS = ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 1e9))     # s
LOW, HIGH = 0.3, 0.7                                                 # the metric's lowest and highest IoU
CLOSE = 1.0                  # m: a team's player "on the ball" (docs/lab/short.md: 1 m separates real touches)
CLOSE_FRAMES = 3             # frames on the ball for a missed possession to count as visible in the tracking
EXAMPLE = ("J03WR9", 60)     # the match and the seconds shown stage by stage
OUT = os.path.join(lab.RESULTS, "inside_v12")


def letters_of(m, stage):
    return np.load(os.path.join(train_v12.OOF + chain.suffix(WHICH), stage, "{}.npy".format(m))).astype("<U1")


def codes(letters):
    return np.array([CODE[x] for x in np.asarray(letters).astype("<U1").tolist()], dtype=np.int8)


def spell_table(truth, letters):
    """(true spells, predicted spells, overlaps, runs): as the metric sees them."""
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(codes(letters), runs)
    t, p = pm.possession_spells(truth, runs), pm.possession_spells(filled, runs)
    return t, p, pm.spell_overlaps(t, p), runs


def matched(overlaps, n_t, n_p, threshold):
    """(bool per true spell, bool per predicted spell): the metric's one-to-one greedy match."""
    ft, fp = np.zeros(n_t, bool), np.zeros(n_p, bool)
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if not ft[i] and not fp[j]:
            ft[i], fp[j] = True, True
    return ft, fp


def summary(match, letters):
    """The numbers of one version on one match: the score, its parts, and frames."""
    s = lab.score(match, letters)
    t, p, ov, _ = spell_table(match.truth, letters)
    out = {"spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "d_f1": s["d_f1"], "possessions": len(p), "true": len(t)}
    precision, recall = [], []
    for iou in pm.SPELL_IOUS:
        tp = int(matched(ov, len(t), len(p), iou)[0].sum())
        precision.append(tp / max(len(p), 1))
        recall.append(tp / max(len(t), 1))
        out["p{:.1f}".format(iou)], out["r{:.1f}".format(iou)] = precision[-1], recall[-1]
    out["precision"], out["recall"] = float(np.mean(precision)), float(np.mean(recall))
    graded = np.isin(match.truth, (pm.H, pm.A, pm.D))
    out["frames_right"] = float(np.mean(np.asarray(letters)[graded] == match.truth_letters[graded]))
    return out


def stage_effect(match, before, after):
    """What one stage changed: frames, and of them how many it made right and wrong."""
    graded = np.isin(match.truth, (pm.H, pm.A, pm.D))
    changed = (before != after) & graded
    truth = match.truth_letters.astype("<U1")
    return {"frames_changed": int(changed.sum()), "made_right": int(np.sum(changed & (after == truth))),
            "made_wrong": int(np.sum(changed & (before == truth))),
            "still_wrong": int(np.sum(changed & (after != truth) & (before != truth)))}


def robust_inside(match, p):
    """The classifier alone, frame by frame, before any decoding."""
    graded = np.isin(match.truth, (pm.H, pm.A, pm.D))
    top = np.argmax(p, axis=1)                                       # 0 H, 1 A, 2 D: the scorer's codes
    truth = match.truth
    seen = ~np.isnan(match.track["ball"][:, 0])
    confusion = np.zeros((3, 3), int)
    for i in range(3):
        for j in range(3):
            confusion[i, j] = int(np.sum(graded & (truth == i) & (top == j)))
    live = graded & np.isin(truth, (0, 1))
    conf = p.max(axis=1)
    bins = []
    for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)):
        where = graded & (conf >= lo) & (conf < hi)
        bins.append({"bin": "{:.2f}-{:.2f}".format(lo, min(hi, 1.0)), "frames": int(where.sum()), "right": int(np.sum(where & (top == truth)))})
    return {"confusion": confusion, "seen_frames": int(np.sum(graded & seen)), "seen_right": int(np.sum(graded & seen & (top == truth))),
            "gap_frames": int(np.sum(graded & ~seen)), "gap_right": int(np.sum(graded & ~seen & (top == truth))),
            "live_frames": int(live.sum()), "team_right": int(np.sum(live & (np.argmax(p[:, :2], axis=1) == truth))),
            "live_seen": int(np.sum(live & seen)), "team_right_seen": int(np.sum(live & seen & (np.argmax(p[:, :2], axis=1) == truth))),
            "live_gap": int(np.sum(live & ~seen)), "team_right_gap": int(np.sum(live & ~seen & (np.argmax(p[:, :2], axis=1) == truth))),
            "bins": bins}


def missed_rows(match, letters):
    """One row per true possession with no predicted one at IoU >= LOW, and per predicted one with no
    true one: what it is, where to find it, and what the tracking shows of it."""
    t, p, ov, _ = spell_table(match.truth, letters)
    ft, fp = matched(ov, len(t), len(p), LOW)
    track = match.track
    near = {k: touches.nearest(track["players"], track["team"], track["ball"], side) for k, side in (("H", HOME), ("A", AWAY))}
    found = v11.seen_touches(track, v11.SETTINGS[WHICH]["dv"])
    seen = ~np.isnan(track["ball"][:, 0])
    truth = match.truth_letters.astype("<U1")
    pred = np.asarray(letters).astype("<U1")
    index = {(a, b): i for i, (a, b, _) in enumerate(pm.segments(match.truth, pm.valid_runs(match.truth)))}
    seg = pm.segments(match.truth, pm.valid_runs(match.truth))
    rows = []
    for (a, b, k), ok in zip(t, ft):
        if ok:
            continue
        team = "H" if k == pm.H else "A"
        other = "A" if team == "H" else "H"
        i = index[(a, b)]
        before = seg[i - 1][2] if i else None
        after = seg[i + 1][2] if i + 1 < len(seg) else None
        on = near[team][a:b] < CLOSE
        rows.append({"match": match.id, "kind": "missed", "frame": a, "seconds": (b - a) / FPS, "team": team,
                     "ball_seen": float(seen[a:b].mean()), "frames_on_ball": int(np.sum(on)),
                     "visible": bool(np.sum(on) >= CLOSE_FRAMES), "touch_seen": bool(found[HOME if team == "H" else AWAY][a:b].any()),
                     "between_other": before == CODE[other] and after == CODE[other],
                     "next_to_dead": before == pm.D or after == pm.D,
                     "said_team": float(np.mean(pred[a:b] == team)), "said_other": float(np.mean(pred[a:b] == other)),
                     "said_dead": float(np.mean(pred[a:b] == "D")),
                     "overlapping": bool(np.any(pred[a:b] == team))})
    for (a, b, k), ok in zip(p, fp):
        if ok:
            continue
        team = "H" if k == pm.H else "A"
        other = "A" if team == "H" else "H"
        rows.append({"match": match.id, "kind": "spurious", "frame": a, "seconds": (b - a) / FPS, "team": team,
                     "ball_seen": float(seen[a:b].mean()), "true_team": float(np.mean(truth[a:b] == team)),
                     "true_other": float(np.mean(truth[a:b] == other)), "true_dead": float(np.mean(truth[a:b] == "D"))})
    return rows


def by_length(match, letters):
    """Per length band: true possessions and how many are found at each IoU; predicted ones and how many are right."""
    t, p, ov, _ = spell_table(match.truth, letters)
    rows = []
    masks = {iou: matched(ov, len(t), len(p), iou) for iou in (LOW, 0.5, HIGH)}
    for lo, hi in LENGTHS:
        in_t = [lo <= (b - a) / FPS < hi for a, b, _ in t]
        in_p = [lo <= (b - a) / FPS < hi for a, b, _ in p]
        row = {"match": match.id, "lo": lo, "true": int(np.sum(in_t)), "pred": int(np.sum(in_p))}
        for iou, (ft, fp) in masks.items():
            row["found{:.1f}".format(iou)] = int(np.sum(ft[in_t])) if len(t) else 0
            row["right{:.1f}".format(iou)] = int(np.sum(fp[in_p])) if len(p) else 0
        rows.append(row)
    return rows


def missed_oracles(match, letters):
    """spellF1 gain if only the missed possessions the tracking shows were written in, or only the rest."""
    t, p, ov, _ = spell_table(match.truth, letters)
    ft, _ = matched(ov, len(t), len(p), LOW)
    near = {k: touches.nearest(match.track["players"], match.track["team"], match.track["ball"], side) for k, side in ((pm.H, HOME), (pm.A, AWAY))}
    base = lab.score(match, letters)["spell_f1"]
    out = {}
    for name, want in (("visible", True), ("not visible", False)):
        fixed = np.asarray(letters).astype("<U1").copy()
        for (a, b, k), ok in zip(t, ft):
            if not ok and (np.sum(near[k][a:b] < CLOSE) >= CLOSE_FRAMES) == want:
                fixed[a:b] = "H" if k == pm.H else "A"
        out[name] = lab.score(match, fixed)["spell_f1"] - base
    return out


def runs_text(letters, a, b):
    """Runs of the letters in [a, b) as text: "H 3.2 s, A 1.0 s, ..."."""
    w = np.asarray(letters)[a:b]
    edges = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1, len(w)]
    return " ".join("{}{:.1f}".format(w[edges[i]], (edges[i + 1] - edges[i]) / FPS) for i in range(len(edges) - 1))


def example(steps, match, seconds):
    """The stretch of `seconds` where the most stages changed something, one line per stage."""
    span = seconds * FPS
    change = np.zeros(match.frames)
    names = list(steps)
    for x, y in zip(names, names[1:]):
        change += (steps[x] != steps[y]).astype(float) * (1 + names.index(y))
    score = np.convolve(change > 0, np.ones(span), "same")
    centre = int(np.argmax(score))
    a = max(centre - span // 2, 0)
    b = a + span
    print("\n{}: frames {} to {} (each run: label, seconds)".format(match.id, a, b))
    print("  {:<18} {}".format("truth", runs_text(match.truth_letters.astype("<U1"), a, b)))
    for name in names:
        print("  {:<18} {}".format(name, runs_text(steps[name], a, b)))
    return {"match": match.id, "from": a, "to": b, "truth": runs_text(match.truth_letters.astype("<U1"), a, b),
            **{name: runs_text(steps[name], a, b) for name in names}}


def main():
    settings = train_v12.chosen(WHICH)["robust"]
    versions, effects, inside, errors, lengths, oracles, shown = [], [], [], [], [], [], None
    for match in lab.each(WHICH):
        p = lab_robust.held_out_proba(match, WHICH)
        argmax = np.array(["H", "A", "D"])[np.argmax(p, axis=1)]
        graded_or_half = np.zeros(match.frames, bool)
        for _, first, last in match.halves:
            graded_or_half[first:last + 1] = True
        argmax = np.where(graded_or_half, argmax, "X")
        decoded = lab_robust.L.decode_cost(match, -np.log(p), **settings)
        steps = {"v11": match.v11.astype("<U1"), "robust, frame by frame": argmax.astype("<U1"), "+ decoder": np.asarray(decoded).astype("<U1"),
                 "+ snap and move": letters_of(match.id, "robust"), "+ short": letters_of(match.id, "short"),
                 "+ segments": letters_of(match.id, "segments"), "+ timing (v12)": letters_of(match.id, "timing")}
        for name, letters in steps.items():
            versions.append({"match": match.id, "source": match.source, "unseen": match.unseen, "version": name, **summary(match, letters)})
        chain_names = list(steps)[2:]
        for x, y in zip(chain_names, chain_names[1:]):
            effects.append({"match": match.id, "stage": y, **stage_effect(match, steps[x], steps[y])})
        inside.append(robust_inside(match, p))
        final = steps["+ timing (v12)"]
        errors += missed_rows(match, final)
        lengths += by_length(match, final)
        oracles.append(missed_oracles(match, final))
        if match.id == EXAMPLE[0]:
            shown = example({k: v for k, v in steps.items() if k not in ("v11", "robust, frame by frame")}, match, EXAMPLE[1])
        print("{}: done".format(match.id), flush=True)

    v = pd.DataFrame(versions)
    order = list(dict.fromkeys(v.version))
    cols = ["spell_f1", "precision", "recall", "possessions", "true", "frames_right", "macro_f1", "d_f1"]
    table = v.groupby("version", sort=False)[cols].mean().loc[order]
    print("\nversion by version (mean of the 10 matches):\n" + table.round(3).to_string())
    last = v[v.version == "+ timing (v12)"]
    by_iou = {}
    for iou in pm.SPELL_IOUS:
        pr, re = last["p{:.1f}".format(iou)], last["r{:.1f}".format(iou)]
        by_iou["{:.1f}".format(iou)] = {"precision": float(pr.mean()), "recall": float(re.mean()), "f1": float((2 * pr * re / (pr + re)).mean())}
    print("\nv12 by IoU:\n" + pd.DataFrame(by_iou).T.round(3).to_string())
    print("\nv12 per match:\n" + last.set_index("match")[["source", "spell_f1", "precision", "recall", "possessions", "true", "frames_right"]].round(3).to_string())
    corr = float(np.corrcoef(last.true, last.spell_f1)[0, 1])
    idsse = last[last.source == "idsse"]
    print("correlation of v12's score with the true possessions: {:.2f} (IDSSE alone {:.2f})".format(corr, float(np.corrcoef(idsse.true, idsse.spell_f1)[0, 1])))

    e = pd.DataFrame(effects).groupby("stage", sort=False).sum(numeric_only=True) / v.match.nunique()
    print("\nwhat each stage changes (frames per match):\n" + e.round(0).to_string())

    confusion = sum(i["confusion"] for i in inside)
    print("\nrobust alone, frames (row: truth H A D; column: its top answer):\n{}".format(confusion))
    totals = {k: sum(i[k] for i in inside) for k in ("seen_frames", "seen_right", "gap_frames", "gap_right", "live_frames", "team_right", "live_seen", "team_right_seen", "live_gap", "team_right_gap")}
    frames = {"right, ball seen": totals["seen_right"] / totals["seen_frames"], "right, ball missing": totals["gap_right"] / totals["gap_frames"],
              "team right in play": totals["team_right"] / totals["live_frames"], "team right in play, ball seen": totals["team_right_seen"] / totals["live_seen"],
              "team right in play, ball missing": totals["team_right_gap"] / totals["live_gap"]}
    print({k: round(x, 3) for k, x in frames.items()})
    bins = pd.DataFrame([b for i in inside for b in i["bins"]]).groupby("bin").sum()
    bins["right_share"] = bins.right / bins.frames
    bins["frames_share"] = bins.frames / bins.frames.sum()
    print("\nrobust's confidence against how often it's right:\n" + bins.round(3).to_string())

    l = pd.DataFrame(lengths).groupby("lo").sum(numeric_only=True) / v.match.nunique()
    for iou in (LOW, 0.5, HIGH):
        l["recall{:.1f}".format(iou)] = l["found{:.1f}".format(iou)] / l.true
        l["precision{:.1f}".format(iou)] = l["right{:.1f}".format(iou)] / l.pred
    print("\nv12 by possession length (per match):\n" + l[["true", "pred", "recall0.3", "recall0.5", "recall0.7", "precision0.3", "precision0.5", "precision0.7"]].round(3).to_string())

    err = pd.DataFrame(errors)
    err.to_csv(OUT + "_errors.csv", index=False, float_format="%.3f")
    missed = err[err.kind == "missed"]
    spurious = err[err.kind == "spurious"]
    n = v.match.nunique()
    anatomy = {"missed per match": len(missed) / n, "median s": float(missed.seconds.median()), "visible share": float(missed.visible.mean()),
               "touch seen share": float(missed.touch_seen.mean()), "ball seen (mean share)": float(missed.ball_seen.mean()),
               "between other team": float(missed.between_other.mean()), "next to dead": float(missed.next_to_dead.mean()),
               "v12 says other team": float(missed.said_other.mean()), "v12 says dead": float(missed.said_dead.mean()),
               "v12 says right team": float(missed.said_team.mean()), "overlapped by right team": float(missed.overlapping.mean())}
    print("\nmissed possessions:\n" + json.dumps({k: round(x, 3) for k, x in anatomy.items()}, indent=1))
    o = pd.DataFrame(oracles).mean()
    print("spellF1 gain if written in: the visible ones {:+.4f}, the rest {:+.4f}".format(o["visible"], o["not visible"]))
    spurious_anatomy = {"spurious per match": len(spurious) / n, "median s": float(spurious.seconds.median()), "true team share of its frames": float(spurious.true_team.mean()),
                        "other team": float(spurious.true_other.mean()), "dead": float(spurious.true_dead.mean()),
                        "mostly own team": float((spurious.true_team > 0.5).mean()), "mostly other team": float((spurious.true_other > 0.5).mean()), "mostly dead": float((spurious.true_dead > 0.5).mean())}
    print("\nspurious possessions:\n" + json.dumps({k: round(x, 3) for k, x in spurious_anatomy.items()}, indent=1))
    with open(OUT + ".json", "w") as handle:
        json.dump({"versions": table.round(4).reset_index().to_dict("records"), "per_match": last.round(4).to_dict("records"),
                   "effects": e.round(1).reset_index().to_dict("records"), "confusion": confusion.tolist(), "frames": frames,
                   "confidence": bins.round(4).reset_index().to_dict("records"), "by_length": l.round(4).reset_index().to_dict("records"),
                   "missed": anatomy, "missed_oracles": o.round(4).to_dict(), "by_iou": by_iou, "spurious": spurious_anatomy,
                   "example": shown}, handle, indent=1, default=str)
    print("\nwrote {} and {}".format(os.path.relpath(OUT + ".json", ROOT), os.path.relpath(OUT + "_errors.csv", ROOT)))


if __name__ == "__main__":
    main()
