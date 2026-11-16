"""Model v12: the lab's chain (docs/lab.md), ported to our match.

    python -m possession.v12           # writes evaluation/results/v12.csv; prints only its path and SHA-256

Full explanation: docs/models/v12-lab-chain.md. The inputs are built exactly as the bench built them
(bench/eval_pathcrf.py: v1's clues without Smart Tagging, the dead-ball and restart models, PathCRF),
then the chain of the bench set in v10.BENCH_SET:
  all:   robust -> short -> segments -> timing
  idsse: robust -> short -> segments -> recall -> dead
robust is a per-frame classifier, P(H / A / D) from the tracking and those inputs, with any ball
standing still for 1 s blanked; its -log P is decoded, then v9's snap and v11's move. Each later stage
is the lab's own code (bench/lab_*.py), with the models `python -m bench.train_v12` trained on the
bench matches (possession/models/v12_<set>.joblib). `python -m bench.check_v12` shows the port gives
the lab's letters on bench matches. Last, the second half's graded frames before its kick-off are D
(FAQ: "those 2,934 frames are graded and all of them are D"), the kick-off found from Smart Tagging's
tag as v1-v10 did; the bench has no tags, so there it changes nothing.
"""

import hashlib
import os
import types

import joblib
import numpy as np

from bench import chain, lab_dead, lab_recall, lab_robust, lab_segments, lab_short, lab_timing
from bench.eval_pathcrf import v7_cost
from bench.train_restart import cost_with_restart
from bench.tune_decoder import decode
from possession import dead_ball, features, pathcrf_adapter, repair, restart, v1, v9, v10, v11
from possession import lab_learned_features as llf
from possession.data import FPS, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v12.csv")
MODELS = os.path.join(ROOT, "possession", "models", "v12_{}.joblib")
# The chain kept on each bench set (docs/lab.md, rounds 3-4): each stage beat the chain without it.
CHAIN = {"all": ("short", "segments", "timing"), "idsse": ("short", "segments", "recall", "dead")}
SWAP = np.array([1, 0, 2])          # H <-> A on P's columns, as the robust model was trained
# PathCRF's arithmetic: float64 makes it the same on any CPU to ~1e-16 (float32 differs by ~1e-7
# between processors); on the bench the two give the same letters (docs/reproduce.md).
PATHCRF_FLOAT64 = True


def inputs(tracking, halves, pitch, which, clean=None):
    """(the repaired tracking, the inputs the lab's stages read): as bench/eval_pathcrf.py builds them,
    with the models trained on every bench match of the set.

    clean (lab round 9, docs/lab/cleaning.md): a function of the repaired tracking. Given, a third value is
    returned, (the cleaned tracking, the inputs with PathCRF's P and v7's cost from it); everything else
    reads the tracking as delivered either way."""
    fixed, _ = repair.repair(tracking, halves, pitch)                                   # v8
    clues = v1.compute_clues(fixed, [], halves, pitch)                                 # no Smart Tagging
    base = v1.build_costs(dict(clues, stopped=np.zeros_like(clues["stopped"]), active=np.zeros_like(clues["active"])))
    setting = v10.SETTINGS[which]
    names, X = features.frame_features(fixed, halves, pitch)                           # v4
    model = dead_ball.load(*setting["dead_ball"])
    if model["features"] != names:
        raise ValueError("the saved dead-ball model's features differ from possession/features.py")
    p_dead = dead_ball.predict(model, X)
    start_model = dead_ball.load(*setting["restart"])                                  # v6
    p_start = restart.p_home(start_model, restart.all_orientations(fixed, halves, pitch))
    d, r, pc = chain.dead_ball(which), chain.restart(which), chain.pathcrf(which)
    v6_cost = cost_with_restart(base, p_dead, p_start, halves, r["weight"], r["carry"], d["weight"])
    live = np.isin(decode(v6_cost, halves, chain.decoder(which)), ["H", "A"])          # v7
    pathcrf = pathcrf_adapter.load_model()
    p_pathcrf = pathcrf_adapter.p_home(pathcrf.double() if PATHCRF_FLOAT64 else pathcrf, fixed, halves, pitch, live)
    missing = np.isnan(fixed["ball"][:, 0])
    pred = {"p_dead": p_dead, "p_start": p_start, "p_pathcrf": p_pathcrf, "v6_cost": v6_cost.astype(np.float32),
            "v7_cost": v7_cost(v6_cost, p_pathcrf, missing, pc["weight"], pc["where"]).astype(np.float32),
            "restart_clue": restart.restart_belief(p_start, p_dead, halves, int(r["carry"] * FPS))}
    if clean is None:
        return fixed, pred
    cleaned = clean(fixed)
    p_clean = pathcrf_adapter.p_home(pathcrf, cleaned, halves, pitch, live)    # already float64 if PATHCRF_FLOAT64
    pred_clean = dict(pred, p_pathcrf=p_clean, v7_cost=v7_cost(v6_cost, p_clean, np.isnan(cleaned["ball"][:, 0]), pc["weight"],
                                                               pc["where"]).astype(np.float32))
    return fixed, pred, (cleaned, pred_clean)


def as_match(track, pred, halves, pitch, name="ours"):
    """A match the lab's stages can read: they use only .track, .pred, .halves and .pitch to predict."""
    return types.SimpleNamespace(id=name, source=name, unseen=True, halves=halves, pitch=pitch,
                                 frames=len(track["ball"]), track=track, pred=pred)


def robust_proba(match, model, which):
    """P(H, A, D) per frame: lab_robust's features on the blanked ball, the mean of the match as it is
    and with the teams exchanged, clipped and rounded (lab_robust.cached + proba, without the cache)."""
    b = lab_robust.blanked(match)
    found = v11.seen_touches(b.track, v11.SETTINGS[which]["dv"])
    passes = v9.seen_passes(b.track)
    names, X = llf.frame_features(b.track, b.halves, b.pitch, b.pred, found, passes)
    _, Xs = llf.frame_features(b.track, b.halves, b.pitch, b.pred, found, passes, swap=True)
    if [names[i] for i in model["cols"]] != model["names"]:
        raise ValueError("the robust model's features differ from possession/lab_learned_features.py")
    p = model["clf"].predict_proba(X[:, model["cols"]])
    p = (p + model["clf"].predict_proba(Xs[:, model["cols"]])[:, SWAP]) / 2
    return np.round(np.clip(p, *lab_robust.L.CLIP), lab_robust.L.DECIMALS)


def dead_apply(match, letters, model, thr, min_len, **_):
    """lab_dead.apply without its feature cache (it keys files by match id)."""
    _, X = lab_dead.compute(match)
    p = np.full(len(X), np.nan, dtype=np.float32)
    inside = ~np.isnan(X).all(axis=1)
    p[inside] = model["clf"].predict_proba(X[inside][:, lab_dead.columns(model["features"])])[:, 1]
    return lab_dead.act(letters, p, match.halves, thr, min_len)[0]


def run_chain(match, models, which, settings=None):
    """Letters for every frame: robust, decoded, snapped and moved, then each post stage of the set."""
    settings = settings or models["settings"]
    s = settings["robust"]
    cost = -np.log(robust_proba(match, models["robust"], which))
    letters = lab_robust.finish(match, lab_robust.L.decode_cost(match, cost, **s), which)
    for stage in CHAIN[which]:
        model, s = models[stage], settings[stage]
        if stage == "short":
            letters = lab_short.apply(match, letters, model=model, **s)
        elif stage == "segments":
            letters = lab_segments.apply(match, letters, model, **s)
        elif stage == "timing":
            letters = lab_timing.apply(match, letters, model, **s)
        elif stage == "recall":
            letters = lab_recall.apply(match, letters, model, **{k: v for k, v in s.items() if k != "model"})
        elif stage == "dead":
            letters = dead_apply(match, letters, model, **s)
    return np.asarray(letters).astype("<U1")


def before_kickoff(tags, n_frames, halves):
    """bool [frames]: the second half's graded frames before its kick-off tag (v1's clue)."""
    return v1.clue_tags(tags or [], n_frames, halves)[1]


def load_models(which):
    return joblib.load(MODELS.format(which))


def predict(tracking, tags=None, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH), bench_set=None):
    """(labels for every video frame, {"pred": the inputs}). Smart Tagging feeds no model (no bench match
    has it); its second-half kick-off tag only marks the frames before kick-off as D."""
    halves = halves or graded_halves()
    which = bench_set or v10.BENCH_SET
    fixed, pred = inputs(tracking, halves, pitch, which)
    match = as_match(fixed, pred, halves, pitch)
    labels = run_chain(match, load_models(which), which)
    labels[before_kickoff(tags, len(labels), halves)] = "D"
    return labels, {"pred": pred}


def main():
    """Only what Jose needs to check the file, as v10 and v11: no statistics about the match."""
    labels, _ = predict(load_tracking(), load_tags())
    v1.write_submission(OUTPUT, labels)
    with open(OUTPUT, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    path = os.path.relpath(OUTPUT, ROOT)
    print("wrote {} (bench set {})".format(path, v10.BENCH_SET))
    print("SHA-256 {}".format(digest))
    print("format check: .venv/bin/python evaluation/validate_submission.py {}".format(path))


if __name__ == "__main__":
    main()
