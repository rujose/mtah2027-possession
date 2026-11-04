"""Lab candidate `robust` (round 4): `learned` made unable to read the bench's frozen dead-time ball (docs/lab/robust.md).

    python -m bench.lab_robust --measure [--set all|idsse]   # seconds: what the blanking removes, per source
    python -m bench.lab_robust [--set all|idsse]             # ~20 min / ~8 min, < 0.6 GB; after `python -m bench.lab --build`
                                                             # and `python -m bench.lab_learned`; writes bench/results/lab_robust[_idsse]*
    python -m bench.lab_robust --ball-parts                  # ~6 min: a diagnostic, the ball family's importance split three ways

`learned` (bench/lab_learned.py) is a per-frame P(H / A / D) whose main input is the ball, and the
IDSSE feed it learns from freezes the ball in dead time (whenever seen while dead it stands still
99 % of the time). Whether our ball does is unknown and not looked at. Two variants, one grid, both
fixed before any held-out run:

  blank     every frame of a run where the ball is seen and stands still (under STILL m/s over the
            0.2 s before or after) for at least STILL_FOR s becomes "ball missing", on every match, in
            training and in prediction; then learned's 97 features recomputed on that tracking
            (touches and passes included: the model sees a feed in which a still ball doesn't exist).
  no_speed  learned's features on the tracking as it is, without the ball's speed family (speed
            before / after, change of speed, ΔV, change of course, and their 1 s / 3 s means): the
            ball's being seen, its gaps and where it is stay.

Everything else is learned's: IDSSE-only training, every 10th graded frame with the teams exchanged,
the same trees, the same clip and rounding, the same decoder grid; then v9's snap and v11's move on
the ORIGINAL tracking (`finish`), as every post stage reads it. The earlier models' outputs (P(dead),
restart, PathCRF, v6 / v7 costs) are the harness's, computed on the original tracking.

For mixing (as lab_learned): `held_out_proba(match, which, variant)`, `decode_held_out(match, which,
**setting)` (CHOSEN by default) -> letters at v7's level, then `finish(match, letters, which)`.
"""

import argparse
import itertools
import json
import os
import time
import types

import numpy as np
import pandas as pd

from bench import chain, lab, sets
from bench import lab_learned as L
from bench.lab_learned_check import STILL
from possession import lab_learned_features as llf
from possession import v9, v11
from possession.data import FPS, ROOT

# Params (fixed before any held-out run; docs/lab/robust.md gives the why of each)
STAGE = "decode"
CACHE = os.path.join(ROOT, "data", "external", "lab_robust")     # gitignored (data/*)
LOOK = llf.LOOK          # frames (0.2 s): the speed window, as learned's features and the touch detector
# STILL (0.3 m/s, from bench/lab_learned_check.py): our jitter alone gives ~0.06 m/s, a carried ball > 0.5.
STILL_FOR = 1.0          # s: a seen ball still this long is blanked (by reasoning; see the page)
STILL_FOR_SHOWN = (0.5, 1.0, 2.0, 3.0)   # --measure only: what other choices would have blanked
# The pre-registered fallback: had STILL_FOR blanked more than this share of either source's live
# frames (truth H / A), it would have become 2 s. It didn't (the page has the measurement).
LIVE_LIMIT = 0.02
SPEED_FEATURES = ("ball_speed", "ball_accel", "ball_dv", "ball_turn")   # name prefixes dropped by no_speed
VARIANTS = ("blank", "no_speed")         # the main idea first (ties go to it)
COLUMNS = L.COLUMNS                      # variant, blend, switch_team, switch_dead, min_possession
GRID = [dict(zip(COLUMNS, s)) for s in itertools.product(VARIANTS, L.BLENDS, L.SWITCH_TEAM, L.SWITCH_DEAD,
                                                          L.MIN_POSSESSION)]
# Chosen on all 10 by `python -m bench.lab_robust` (bench/results/lab_robust.json), and the same on
# the IDSSE set (lab_robust_idsse.json). Every fold of both sets picked `blank`, no blend, dead 25;
# team 4 in all 10 folds on all (4 or 8 on the IDSSE set); minimum 10 in 7 of 10 (20 in 3).
CHOSEN = {"variant": "blank", "blend": 0.0, "switch_team": 4.0, "switch_dead": 25.0, "min_possession": 10}
# no_speed's own choice (both sets), for a mix that wants it: decode_held_out(match, which, **CHOSEN_NO_SPEED)
CHOSEN_NO_SPEED = {"variant": "no_speed", "blend": 0.0, "switch_team": 4.0, "switch_dead": 25.0, "min_possession": 20}


def folder(which):
    return CACHE + chain.suffix(which)


# --------------------------------------------------------------------------------------------
# The blanking
# --------------------------------------------------------------------------------------------

def still_frames(ball, halves, still_for=STILL_FOR):
    """bool [frames]: frames of a run, within a half, where the ball is seen and moves under STILL
    m/s over the 0.2 s before or after it (either window, where known), lasting >= still_for s."""
    out = np.zeros(len(ball), bool)
    need = int(round(still_for * FPS))
    for _, first, last in halves:
        b = ball[first:last + 1].astype(np.float64)
        fwd = np.full(len(b), np.nan)
        fwd[:-LOOK] = np.hypot(*(b[LOOK:] - b[:-LOOK]).T) * (FPS / LOOK)
        bwd = np.full(len(b), np.nan)
        bwd[LOOK:] = fwd[:-LOOK]
        with np.errstate(invalid="ignore"):
            slow = (np.fmin(fwd, bwd) < STILL) & ~np.isnan(b[:, 0])      # unknown on both sides: not still
        edges = np.flatnonzero(np.diff(np.r_[0, slow.astype(np.int8), 0]))
        for a, e in zip(edges[::2], edges[1::2]):
            if e - a >= need:
                out[first + a:first + e] = True
    return out


def blanked(match, still_for=STILL_FOR):
    """The same match with the still ball blanked: a new namespace (the original is untouched), with
    .blanked (bool [frames]) the frames that lost their ball."""
    mask = still_frames(match.track["ball"], match.halves, still_for)
    ball = match.track["ball"].copy()
    ball[mask] = np.nan
    fields = {k: v for k, v in vars(match).items() if not k.startswith("_")}
    fields["track"] = {**match.track, "ball": ball}
    return types.SimpleNamespace(**fields, blanked=mask)


def forward_still(ball):
    """lab_learned_check's measure: still (< STILL m/s over the next 0.2 s), NaN where unknown."""
    speed = np.full(len(ball), np.nan)
    speed[:-LOOK] = np.hypot(*(ball[LOOK:] - ball[:-LOOK]).T) * (FPS / LOOK)
    with np.errstate(invalid="ignore"):
        return np.where(np.isnan(speed), np.nan, speed < STILL)


def measure(which):
    """Per match and per source: the share of live and of dead frames the blanking removes, the ball
    missing before / after, and the still share of the ball still seen. Truth only to count."""
    rows = []
    for match in lab.each(which):
        ball = match.track["ball"]
        dead, live = match.truth == 2, np.isin(match.truth, (0, 1))
        missing = np.isnan(ball[:, 0])
        still0 = forward_still(ball)
        for t in STILL_FOR_SHOWN:
            mask = still_frames(ball, match.halves, t)
            after = ball.copy()
            after[mask] = np.nan
            still1 = forward_still(after)
            row = {"match": match.id, "source": match.source, "still_for": t}
            for name, part in (("dead", dead), ("live", live)):
                k0, k1 = part & ~np.isnan(still0), part & ~np.isnan(still1)
                row.update({
                    "{}_frames".format(name): int(part.sum()),
                    "{}_blanked".format(name): float(mask[part].mean()),
                    "{}_seen_blanked".format(name): float(mask[part & ~missing].mean()),
                    "{}_missing_before".format(name): float(missing[part].mean()),
                    "{}_missing_after".format(name): float((missing | mask)[part].mean()),
                    "{}_seen_still_before".format(name): float(still0[k0].mean()),
                    "{}_seen_still_after".format(name): float(still1[k1].mean()) if k1.any() else np.nan})
            row["all_blanked"] = float(mask.mean())
            rows.append(row)
        print("{}: measured".format(match.id), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(lab.RESULTS, "lab_robust_blanking{}.csv".format(chain.suffix(which))), index=False,
              float_format="%.5f")
    for t in STILL_FOR_SHOWN:
        print("\nstill for >= {} s ({}):".format(t, "THE RULE" if t == STILL_FOR else "for information"))
        print("source | live blanked (range) | dead blanked (range) | seen dead blanked | missing dead before -> after | "
              "missing live before -> after | seen still, dead before -> after | live before -> after")
        for source, g in df[df.still_for == t].groupby("source"):
            w = {k: g[k + "_frames"] for k in ("dead", "live")}

            def pooled(col, part):
                return float((g[col] * w[part]).sum() / w[part].sum())

            print("{} | {:.2%} ({:.2%}-{:.2%}) | {:.1%} ({:.1%}-{:.1%}) | {:.1%} | {:.1%} -> {:.1%} | {:.1%} -> {:.1%} | "
                  "{:.1%} -> {:.1%} | {:.1%} -> {:.1%}".format(
                      source, pooled("live_blanked", "live"), g.live_blanked.min(), g.live_blanked.max(),
                      pooled("dead_blanked", "dead"), g.dead_blanked.min(), g.dead_blanked.max(),
                      g.dead_seen_blanked.mean(), pooled("dead_missing_before", "dead"), pooled("dead_missing_after", "dead"),
                      pooled("live_missing_before", "live"), pooled("live_missing_after", "live"),
                      g.dead_seen_still_before.mean(), g.dead_seen_still_after.mean(),
                      g.live_seen_still_before.mean(), g.live_seen_still_after.mean()))
    rule = df[df.still_for == STILL_FOR]
    worst = rule.groupby("source").apply(lambda g: (g.live_blanked * g.live_frames).sum() / g.live_frames.sum())
    print("\nlive frames blanked by the rule, per source: {}; limit {:.0%}: {}".format(
        worst.round(4).to_dict(), LIVE_LIMIT, "within" if (worst <= LIVE_LIMIT).all() else "EXCEEDED, use 2 s"))
    return df


# --------------------------------------------------------------------------------------------
# Features, training, prediction (learned's procedure, on the robust inputs)
# --------------------------------------------------------------------------------------------

def cached(m, which="all", match=None):
    """{names, X, X_swapped, truth, blanked}: learned's features on the blanked tracking, saved once."""
    path = os.path.join(folder(which), "{}.npz".format(m))
    if not os.path.exists(path):
        b = blanked(match or lab.load(m, which))
        touches = v11.seen_touches(b.track, v11.SETTINGS[which]["dv"])
        passes = v9.seen_passes(b.track)
        names, X = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes)
        _, Xs = llf.frame_features(b.track, b.halves, b.pitch, b.pred, touches, passes, swap=True)
        os.makedirs(folder(which), exist_ok=True)
        np.savez(path, names=np.array(names), X=X, X_swapped=Xs, truth=b.truth.astype(np.int8), blanked=b.blanked)
    with np.load(path) as z:
        return {"names": [str(k) for k in z["names"]], "X": z["X"], "X_swapped": z["X_swapped"],
                "truth": z["truth"], "blanked": z["blanked"]}


def features(m, which, variant):
    """(names, cols) and the cache a variant reads: blank -> ours, all 97; no_speed -> learned's
    (never written here), without SPEED_FEATURES."""
    if variant == "blank":
        c = cached(m, which)
        return c, np.arange(len(c["names"]))
    if variant == "no_speed":
        if not os.path.exists(os.path.join(L.folder(which), "{}.npz".format(m))):
            raise RuntimeError("run `python -m bench.lab_learned --set {}` first".format(which))
        c = L.cached(m, which)
        return c, np.array([i for i, k in enumerate(c["names"]) if not k.startswith(SPEED_FEATURES)])
    raise ValueError(variant)


def fit(matches, variant=CHOSEN["variant"], which="all", sources=L.TRAIN_SOURCES, **_):
    """learned's `fit` on a robust variant's features."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs, ys, names, cols = [], [], None, None
    used = [getattr(m, "id", m) for m in matches if sets.source(getattr(m, "id", m)) in sources]
    for m in used:
        c, cols = features(m, which, variant)
        names = c["names"]
        rows = np.flatnonzero(np.isin(c["truth"], (0, 1, 2)))[::L.EVERY]
        y = c["truth"][rows].astype(np.int64)
        Xs += [c["X"][rows][:, cols], c["X_swapped"][rows][:, cols]]
        ys += [y, L.SWAP[y]]
        del c
    X, y = np.concatenate(Xs), np.concatenate(ys)
    del Xs
    clf = HistGradientBoostingClassifier(**L.TREES).fit(X, y)
    return {"variant": variant, "which": which, "names": [names[i] for i in cols], "cols": cols, "clf": clf,
            "rows": len(y), "trained_on": used, "_p": {}}


def proba(match, model):
    """P(H, A, D) per frame: the mean of the match as it is and with the teams exchanged, clipped and
    rounded (learned's `proba`)."""
    key = getattr(match, "id", match)
    if key not in model["_p"]:
        from threadpoolctl import threadpool_limits
        c, _ = features(key, model["which"], model["variant"])
        with threadpool_limits(2):
            p = model["clf"].predict_proba(c["X"][:, model["cols"]])
            p = (p + model["clf"].predict_proba(c["X_swapped"][:, model["cols"]])[:, L.SWAP]) / 2
        model["_p"][key] = np.round(np.clip(p, *L.CLIP), L.DECIMALS)
    return model["_p"][key]


def held_out_proba(match, which="all", variant=CHOSEN["variant"]):
    """The P(H, A, D) `main` saved for this match, from the model trained without it:
    data/external/lab_robust[_idsse]/p_<variant>_<match>.npy."""
    return np.load(os.path.join(folder(which), "p_{}_{}.npy".format(variant, getattr(match, "id", match)))).astype(np.float64)


def cost_from(match, p, blend=0.0):
    """-log P, blended with v7's costs (both variants keep PathCRF)."""
    return L.cost_from(match, p, "all", blend)


def decode(match, model, blend=0.0, **setting):
    """Letters at v7's level (before v9's snap and v11's move); settings default to CHOSEN."""
    return L.decode_cost(match, cost_from(match, proba(match, model), blend), **{**CHOSEN, **setting})


def decode_held_out(match, which="all", **setting):
    """`decode` from the saved held-out P (no refit); settings default to CHOSEN."""
    setting = {**CHOSEN, **setting}
    p = held_out_proba(match, which, setting["variant"])
    return L.decode_cost(match, cost_from(match, p, setting["blend"]), **setting)


def finish(match, letters, which="all"):
    """v9's restart snap, then v11's touch move, on the ORIGINAL tracking (learned's `finish`)."""
    return L.finish(match, letters, which)


def importances(match, model, which):
    """learned's: rise in held-out log loss when one family is shuffled (every 25th graded frame)."""
    c, _ = features(match.id, which, model["variant"])
    rows = np.flatnonzero(np.isin(c["truth"], (0, 1, 2)))[::25]
    X, y = c["X"][rows][:, model["cols"]], c["truth"][rows].astype(np.int64)
    del c

    def loss(Z):
        p = np.clip(model["clf"].predict_proba(Z), 1e-6, 1)
        return float(-np.log(p[np.arange(len(y)), y]).mean())

    base, rng, out = loss(X), np.random.default_rng(0), {}
    for group, test in L.GROUPS.items():
        idx = [i for i, k in enumerate(model["names"]) if test(k)]
        if not idx:
            continue
        Z = X.copy()
        Z[:, idx] = Z[rng.permutation(len(Z))][:, idx]
        out[group] = loss(Z) - base
    return base, out


# --------------------------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------------------------

def learned_base(which):
    """learned's held-out spellF1 / macroF1 per match, as a `lab.judge` base."""
    with open(os.path.join(lab.RESULTS, "lab_learned{}.json".format(chain.suffix(which)))) as handle:
        r = json.load(handle)
    return {f["match"]: {"spell_f1": f["held_out"], "macro_f1": f["macro_held_out"], "source": f["source"],
                         "unseen": f["unseen"]} for f in r["folds"]}


def picked(grid, report, column):
    """Per match, `column` of the grid row at the setting its fold picked."""
    out = {}
    for f in report["folds"]:
        g = grid[grid["match"] == f["match"]]
        for c, v in f["picked"].items():
            g = g[g[c] == v]
        out[f["match"]] = float(g.iloc[0][column])
    return out


def learned_picked(which, column):
    """learned's held-out `column` per match (its grid at its folds' picks)."""
    tag = chain.suffix(which)
    grid = pd.read_csv(os.path.join(lab.RESULTS, "lab_learned_grid{}.csv".format(tag)))
    with open(os.path.join(lab.RESULTS, "lab_learned{}.json".format(tag))) as handle:
        return picked(grid, json.load(handle), column)


BALL_PARTS = {   # the ball family split three ways, for `--ball-parts` (a diagnostic, after the run)
    "ball: seen / gaps": lambda k: k in ("ball_seen", "since_ball_seen", "until_ball_seen"),
    "ball: speed": lambda k: k.startswith(SPEED_FEATURES),
    "ball: where": lambda k: k in ("ball_x_from_home_goal", "ball_y_from_touchline", "ball_outside"),
}
PARTS_FOLDS = ("J03WMX", "J03WR9", "metrica1")   # two IDSSE (one PathCRF-unseen), one Metrica


def ball_parts(which):
    """Diagnostic (chosen after the run, reported as such): where inside the ball family the
    importance sits, for learned and both variants, on PARTS_FOLDS (refitted; same seed)."""
    rows = []
    for held in PARTS_FOLDS:
        match = lab.load(held, which)
        others = [m for m in lab.matches(which) if m != held]
        for name in ("learned",) + VARIANTS:
            model = L.fit(others, variant="all", which=which) if name == "learned" else fit(others, variant=name, which=which)
            c = L.cached(held, which) if name == "learned" else features(held, which, name)[0]
            idx_rows = np.flatnonzero(np.isin(c["truth"], (0, 1, 2)))[::25]
            X, y = c["X"][idx_rows][:, model["cols"]], c["truth"][idx_rows].astype(np.int64)
            del c

            def loss(Z):
                p = np.clip(model["clf"].predict_proba(Z), 1e-6, 1)
                return float(-np.log(p[np.arange(len(y)), y]).mean())

            base, rng = loss(X), np.random.default_rng(0)
            row = {"match": held, "model": name, "log_loss": base}
            for part, test in list(BALL_PARTS.items()) + [("ball (whole)", L.GROUPS["ball"])]:
                idx = [i for i, k in enumerate(model["names"]) if test(k)]
                if idx:
                    Z = X.copy()
                    Z[:, idx] = Z[rng.permutation(len(Z))][:, idx]
                    row[part] = loss(Z) - base
            rows.append(row)
            print(row, flush=True)
            del model
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(lab.RESULTS, "lab_robust_ball_parts{}.csv".format(chain.suffix(which))), index=False,
              float_format="%.5f")
    print(df.groupby("model", sort=False).mean(numeric_only=True).round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=chain.SETS, default="all", dest="which")
    parser.add_argument("--measure", action="store_true", help="only what the blanking removes, per source")
    parser.add_argument("--ball-parts", action="store_true", help="diagnostic: the ball family's importance split three ways")
    args = parser.parse_args()
    which, tag = args.which, chain.suffix(args.which)
    if args.measure:
        measure(which)
        return
    if args.ball_parts:
        ball_parts(which)
        return
    ms = lab.matches(which)
    for m in ms:       # the blanked features, once per match
        if not os.path.exists(os.path.join(folder(which), "{}.npz".format(m))):
            started = time.time()
            cached(m, which)
            print("{}: blanked features cached ({:.0f} s)".format(m, time.time() - started), flush=True)
    rows, extra = [], []
    for held in ms:
        match = lab.load(held, which)
        others = [m for m in ms if m != held]
        v11_score = lab.score(match, match.v11)
        graded = np.isin(match.truth, (0, 1, 2))
        mask = cached(held, which)["blanked"]
        p_learned = L.held_out_proba(match, which, "all")
        for variant in VARIANTS:
            started = time.time()
            model = fit(others, variant=variant, which=which)
            p = proba(match, model)
            np.save(os.path.join(folder(which), "p_{}_{}.npy".format(variant, held)), p.astype(np.float32))
            base, imp = importances(match, model, which)
            argmax = np.array(["H", "A", "D"])[p.argmax(axis=1)]
            argmax_learned = np.array(["H", "A", "D"])[p_learned.argmax(axis=1)]

            def acc(letters, where):
                w = where & graded
                return float((letters[w] == match.truth_letters[w]).mean()) if w.any() else np.nan

            extra.append({"match": held, "variant": variant, "train_rows": model["rows"], "features": len(model["cols"]),
                          "trained_on": " ".join(model["trained_on"]), "frames_blanked": float(mask[graded].mean()),
                          "frame_acc_argmax": acc(argmax, graded), "frame_acc_argmax_learned": acc(argmax_learned, graded),
                          "acc_blanked_frames": acc(argmax, mask), "acc_blanked_frames_learned": acc(argmax_learned, mask),
                          "acc_other_frames": acc(argmax, ~mask), "acc_other_frames_learned": acc(argmax_learned, ~mask),
                          "v11_d_f1": v11_score["d_f1"], "log_loss": base, **{"imp " + k: v for k, v in imp.items()}})
            for setting in (s for s in GRID if s["variant"] == variant):
                at_v7 = L.decode_cost(match, cost_from(match, p, setting["blend"]), **setting)
                final = finish(match, at_v7, which)
                s = lab.score(match, final)
                rows.append({"match": held, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                             "possessions": s["possessions"], "true_possessions": s["true_possessions"],
                             "d_f1": s["d_f1"], "team_acc_alive": s["team_acc_alive"],
                             "spell_f1_decoded": lab.score(match, at_v7)["spell_f1"],
                             "frame_acc": L.frame_accuracy(match, final)})
            print("{} {}: fit on {} rows x {} features, {:.0f} s; frame accuracy argmax {:.3f} (learned {:.3f}); "
                  "best spellF1 here {:.4f} (v11 {:.4f}); ball importance {:.3f}".format(
                      held, variant, model["rows"], len(model["cols"]), time.time() - started,
                      extra[-1]["frame_acc_argmax"], extra[-1]["frame_acc_argmax_learned"],
                      max(r["spell_f1"] for r in rows if r["match"] == held and r["variant"] == variant),
                      v11_score["spell_f1"], imp.get("ball", np.nan)), flush=True)
            del model
        del match
    grid = pd.DataFrame(rows)
    frames = pd.DataFrame(extra)
    report = lab.judge(grid, COLUMNS, which)                                  # the bar: against v11
    print("[robust, whole grid, vs v11] " + lab.describe(report).replace("\n", "\n  "))
    vs_learned = lab.judge(grid, COLUMNS, which, base=learned_base(which))     # for the record
    print("[robust, whole grid, vs learned] " + lab.describe(vs_learned).replace("\n", "\n  "))
    d_f1 = picked(grid, report, "d_f1")
    d_learned = learned_picked(which, "d_f1")
    d_v11 = frames.groupby("match").v11_d_f1.first().to_dict()
    report["d_f1"] = {"held_out": float(np.mean(list(d_f1.values()))), "learned": float(np.mean(list(d_learned.values()))),
                      "v11": float(np.mean(list(d_v11.values()))), "per_match": d_f1}
    report["possessions"] = float(np.mean(list(picked(grid, report, "possessions").values())))
    report["vs_learned"] = {k: vs_learned[k] for k in ("all", "macro", "by_source", "unseen", "checks", "passes")}
    report["vs_learned"]["folds"] = [{k: f[k] for k in ("match", "held_out", "base")} for f in vs_learned["folds"]]
    print("D F1 held out {:.4f} (learned {:.4f}, v11 {:.4f}); possessions a match {:.0f}".format(
        report["d_f1"]["held_out"], report["d_f1"]["learned"], report["d_f1"]["v11"], report["possessions"]))
    report["variants"] = {}
    for variant in VARIANTS:     # each variant judged on its own, for the page
        sub = grid[grid["variant"] == variant].reset_index(drop=True)
        r1, r2 = lab.judge(sub, COLUMNS, which), lab.judge(sub, COLUMNS, which, base=learned_base(which))
        d1 = picked(sub, r1, "d_f1")
        report["variants"][variant] = {"vs_v11": {k: r1[k] for k in ("all", "macro", "by_source", "unseen", "chosen", "checks", "passes")},
                                       "vs_learned": {k: r2[k] for k in ("all", "macro", "by_source", "unseen")},
                                       "d_f1": float(np.mean(list(d1.values()))),
                                       "folds": [{k: f[k] for k in ("match", "picked", "held_out", "base")} for f in r1["folds"]]}
        print("[{} only, vs v11] ".format(variant) + lab.describe(r1).replace("\n", "\n  "))
        print("[{} only, vs learned] ".format(variant) + lab.describe(r2).replace("\n", "\n  "))
        print("  D F1 {:.4f}".format(report["variants"][variant]["d_f1"]))
    imp_cols = [c for c in frames.columns if c.startswith("imp ")]
    report["importances"] = frames.groupby("variant")[imp_cols].mean().to_dict(orient="index")
    for variant, imp in report["importances"].items():
        order = sorted(((k[4:], v) for k, v in imp.items() if v == v), key=lambda kv: -kv[1])
        print("importances, {}: {}".format(variant, ", ".join("{} {:.3f}".format(k, v) for k, v in order)))
    lab.save("robust", grid, report, which)
    frames.to_csv(os.path.join(lab.RESULTS, "lab_robust_frames{}.csv".format(tag)), index=False, float_format="%.5f")
    print("wrote bench/results/lab_robust_grid{0}.csv, lab_robust{0}.json, lab_robust_frames{0}.csv".format(tag))


if __name__ == "__main__":
    main()
