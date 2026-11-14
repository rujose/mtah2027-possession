"""A nested random search of branch B's settings on the stressed benches (docs/lab/branch-b.md, lab round 8).

    python -m bench.search_b stress            # ~5 min: augment's training stress on Metrica's 3; fold P on every match's copy
    python -m bench.search_b prepare           # ~25 min: out-of-fold chains, then each match's stage models for 3 kinds of data
    python -m bench.search_b run --worker I/N  # units I, I+N, ... (a unit: one match on one of B's variants), every trial
    python -m bench.search_b report            # nested picks (leave one match out), the marginals, the pick's settings
    python -m bench.search_b confirm --trials N [N ...] [--variants normal p10]   # each trial with its own stage models
    python -m bench.search_b skillcorner --trials N [N ...]                       # deployment models on SkillCorner's 20

Why: the stoppage switch sends a match whose ball stays tracked and moving through stoppages to branch B. B's settings
were never searched: its decoder and post-stage settings are v12's (chosen for branch A's bench), and its post stages
were trained on `augment`'s letters on the NORMAL bench (lab_carried.deploy_post), though B only runs where the ball
looks like the stressed bench's. lab_gate judged B with post stages refitted on the same variant's letters, so the
deployed B was never scored the way it runs.

As bench/search_metrica.py, honestly:
- **The trials are drawn at random, fixed by a seed, before any is scored.** Trial 0 is B as deployed.
- **A unit is one match on one of B's variants** (lab_gate's p25, p50, stressed: the test stress, SkillCorner's odd
  matches' stoppage balls); a match's score is the mean of its 3 units.
- **Stage models for each match are trained without it** (on the other 9), on one of 3 kinds of data, a trial
  setting: `normal` (the other 9 as they are: what B deploys), `stressed` (the other 9 under augment's TRAINING stress:
  SkillCorner's even matches, never the test's templates), `both`. Each kind's letters are the reference start's
  out-of-fold chain on that data (stacked, as lab_mix.add_stage). The reference start is fixed by
  bench/lab_branch_b.py's chains before this runs: `mix_b` if it beats `augment` on B's 3 variants, else `augment`.
- **Nested:** for each match, the trial with the best mean on the other 9 matches is picked; only its score on the
  held-out match counts. (The other matches' stage models saw the held-out match's labels: a second-order leak,
  accepted as in search_metrica.)

What a trial sets: the GRU's weight in the start (gru_b against augment; 0 is augment, B as built), a factor on
P(dead), the decoder's switch prices and minimum possession, v9's snap and v11's move on/off, each post stage on/off
with its thresholds, and the stage models' training data. Nothing reads our match.
"""

import argparse
import glob
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from bench import lab, lab_branch_b, lab_carried, lab_gate, lab_mix, lab_robust, lab_segments, lab_short, lab_timing
from bench import sets, train_v12
from bench.run import score as bench_score

# Params
WHICH = "all"
C = lab_carried
B = lab_branch_b
L = lab_robust.L
CACHE = os.path.join(lab.ROOT, "data", "external", "search_b")      # gitignored (data/*)
RESULTS = os.path.join(lab.RESULTS, "search_b")
VARIANTS = ("p25", "p50", "stressed")
POST = ("normal", "stressed", "both")
CHAIN = ("short", "segments", "timing")
TRIALS = 1500
SEED = 20261113
OFF_SHARE = 0.2
SPACE = {
    "w_gru": (0.0, 1.0, False),
    "f_dead": (0.5, 2.0, True),
    "switch_team": (0.5, 16.0, True),
    "switch_dead": (8.0, 100.0, True),
    "short_p_min": (0.3, 0.9, False),
    "segments_flip": (0.5, 0.95, False),
    "segments_merge": (0.5, 0.95, False),
    "timing_p_min": (0.2, 0.8, False),
}
MIN_POSSESSION = (4, 30)
PARAMS = ["trial", "w_gru", "f_dead", "switch_team", "switch_dead", "min_possession", "snap", "short", "short_p_min",
          "segments", "segments_flip", "segments_merge", "timing", "timing_p_min", "post"]
STRESSED_ID = "{}~train"         # the id of a match's copy under augment's training stress


def settings():
    return train_v12.chosen(WHICH)


def b_trial():
    """Trial 0: B as deployed (v13/v14's branch B)."""
    s = settings()
    return {"trial": 0, "w_gru": 0.0, "f_dead": 1.0, "switch_team": s["robust"]["switch_team"],
            "switch_dead": s["robust"]["switch_dead"], "min_possession": int(s["robust"]["min_possession"]), "snap": True,
            "short": True, "short_p_min": s["short"]["p_min"], "segments": True, "segments_flip": s["segments"]["flip"],
            "segments_merge": s["segments"]["merge"], "timing": True, "timing_p_min": s["timing"]["p_min"], "post": "normal"}


def named_trials():
    """B with only its start changed (lab_branch_b's starts, outside the search): -1 mix_b, -2 gru_b."""
    return [{**b_trial(), "trial": -1, "w_gru": 0.5}, {**b_trial(), "trial": -2, "w_gru": 1.0}]


def every_trial():
    return {t["trial"]: t for t in draw_trials() + named_trials()}


def draw_trials(n=TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    trials = [b_trial()]
    for i in range(1, n):
        t = {"trial": i}
        for k, (lo, hi, log) in SPACE.items():
            t[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
        t["min_possession"] = int(rng.integers(MIN_POSSESSION[0], MIN_POSSESSION[1] + 1))
        t["snap"] = bool(rng.random() < 0.5)
        for stage in CHAIN:
            t[stage] = bool(rng.random() >= OFF_SHARE)
        t["post"] = POST[int(rng.integers(len(POST)))]
        trials.append(t)
    return trials


# --------------------------------------------------------------------------------------------
# augment's training stress on every match, and the fold P on it
# --------------------------------------------------------------------------------------------

def train_p_path(kind, m):
    return os.path.join(CACHE, "p_{}_train_{}.npy".format(kind, m))


def fold_models(m):
    """lab_gate's fold models (robust, augment) that never saw m."""
    held = m if sets.source(m) in L.TRAIN_SOURCES else B.held_list()[-1]
    return joblib.load(os.path.join(lab_gate.folder(WHICH), "folds_{}.joblib".format(held)))


def stress():
    """Each match under augment's training stress (lab_carried TRAIN_SEED): features, then augment's and gru_b's
    held-out P on it."""
    os.makedirs(CACHE, exist_ok=True)
    for m in lab.matches(WHICH):
        if all(os.path.exists(train_p_path(k, m)) for k in ("augment", "gru_b")):
            continue
        started = time.time()
        c = C.stressed_features(m, WHICH, C.TRAIN_SEED)
        np.save(train_p_path("augment", m), C.proba(fold_models(m)["augment"], c).astype(np.float32))
        used = B.used_for(m) if sets.source(m) in L.TRAIN_SOURCES else B.used_for(B.held_list()[-1])
        model = lab_gate.gru_model(used, WHICH, B.net_for(m))
        np.save(train_p_path("gru_b", m), lab_gate.gru_proba(model, c, B.T.halves_of(m, WHICH)).astype(np.float32))
        del c
        print("{}: training-stress copy, augment's and gru_b's P ({:.0f} s)".format(m, time.time() - started), flush=True)


def p_of(kind, m, data):
    """Held-out P of augment or gru_b on a match: data `normal`, `train` (augment's training stress) or a variant."""
    if data == "train":
        return np.load(train_p_path(kind, m)).astype(np.float64)
    if kind == "augment":
        return lab_gate.held_out_p("augment", data, m, WHICH)
    return B.gru_b(data, m)


def start_p(m, data, w_gru, f_dead=1.0):
    aug = p_of("augment", m, data)
    p = aug if w_gru == 0 else B.mix(p_of("gru_b", m, data), aug, w_gru)
    if f_dead != 1.0:
        p = p.copy()
        p[:, 2] *= f_dead
        p = np.round(np.clip(p / p.sum(axis=1, keepdims=True), *L.CLIP), L.DECIMALS)
    return p


# --------------------------------------------------------------------------------------------
# Stage models for each match, trained without it, on 3 kinds of data
# --------------------------------------------------------------------------------------------

def reference():
    """The reference start's GRU weight: 0.5 (mix_b) if it beat augment on B's 3 variants in lab_branch_b, else 0."""
    rows = []
    for variant in VARIANTS:
        f = B.final(variant)
        rows += [f["mix_b"][m] - f["augment"][m] for m in f["augment"]]
    return 0.5 if np.mean(rows) > 0 else 0.0


def models_path(m, post):
    return os.path.join(CACHE, "post_{}_without_{}.joblib".format(post, m))


def train_copy(m):
    match = C.stressed(m, WHICH, seed=C.TRAIN_SEED)
    match.id = STRESSED_ID.format(m)
    return match


def stage_table(stage, x, letters, setting):
    """What train_v12.fit_stage builds from one match before it trains (built once here, trained on many times)."""
    if stage == "short":
        return lab_short.training_rows(x, letters, setting["length"], setting["dead"])
    if stage == "segments":
        tab = lab_segments.table(x, letters)
        return pd.concat([tab, lab_segments.targets(x, tab)], axis=1)
    tab = lab_timing.table(x, letters)
    return tab.assign(right=lab_timing.targets(x, tab))


def stage_model(stage, tables, setting):
    """train_v12.fit_stage's model from the tables, in the matches' order."""
    if stage == "short":
        return lab_short.fit_rows(tables)
    if stage == "segments":
        return lab_segments.train(tables, setting["features"])
    return lab_timing.train(tables, setting["features"])


def prepare():
    """The reference start's out-of-fold chain on the normal bench and on augment's training stress, then for each
    match the stage models on the other 9 (normal, stressed, both)."""
    os.makedirs(CACHE, exist_ok=True)
    lab_mix.cache_short_evidence()          # keyed by id: the stressed copies have their own
    lab_gate.own_dead_cache()
    w = reference()
    s = settings()
    print("reference start: GRU weight {}".format(w), flush=True)
    ms = lab.matches(WHICH)
    steps = {}
    for data in ("normal", "train"):
        matches = [lab.load(m, WHICH) if data == "normal" else train_copy(m) for m in ms]
        per_match = {stage: {x.id: s[stage] for x in matches} for stage in CHAIN}
        letters = {x.id: C.decoded(x, start_p(m, data, w), s["robust"], WHICH) for m, x in zip(ms, matches)}
        steps[data] = {"start": letters}
        for stage in CHAIN:
            letters = lab_mix.add_stage(stage, letters, matches, per_match)
            steps[data][stage] = letters
        print("{}: out-of-fold chain {:.4f}".format(data, np.mean([lab.score(x, letters[x.id])["spell_f1"] for x in matches])),
              flush=True)
        # every match's table for each stage, on the letters of the stage before it (out of fold)
        tables, before = {}, "start"
        for stage in CHAIN:
            tables[stage] = {x.id: stage_table(stage, x, steps[data][before][x.id], s[stage]) for x in matches}
            before = stage
        if data == "normal":                 # the tables give fit_stage's model: checked on the first fold
            held, others = matches[0], matches[1:]
            before = "start"
            for stage in CHAIN:
                a = train_v12.fit_stage(stage, others, steps[data][before], s[stage])
                b = stage_model(stage, [tables[stage][x.id] for x in others], s[stage])
                la = apply_stage(stage, held, steps[data][before][held.id], a, s[stage])
                lb = apply_stage(stage, held, steps[data][before][held.id], b, s[stage])
                print("  check {} on {}: frames differing between fit_stage's model and the tables' {}".format(
                    stage, held.id, int(np.sum(la != lb))), flush=True)
                before = stage
        steps[data] = {"tables": tables, "ids": [x.id for x in matches]}
        del matches, letters
    for m in ms:
        if all(os.path.exists(models_path(m, post)) for post in POST):
            continue
        started = time.time()
        for post in POST:
            data = {"normal": ("normal",), "stressed": ("train",), "both": ("normal", "train")}[post]
            models = {}
            for stage in CHAIN:
                tabs = [steps[d]["tables"][stage][i] for d in data for i in steps[d]["ids"] if i not in (m, STRESSED_ID.format(m))]
                models[stage] = stage_model(stage, tabs, s[stage])
            joblib.dump(models, models_path(m, post), compress=3)
        print("{}: stage models without it, 3 kinds ({:.0f} s)".format(m, time.time() - started), flush=True)
    with open(os.path.join(CACHE, "reference.json"), "w") as handle:
        json.dump({"w_gru": w}, handle)


# --------------------------------------------------------------------------------------------
# One trial on one unit
# --------------------------------------------------------------------------------------------

def apply_stage(stage, match, letters, model, setting):
    if stage == "short":
        return lab_short.apply(match, letters, model=model, **setting)
    if stage == "segments":
        return lab_segments.apply(match, letters, model, **setting)
    return lab_timing.apply(match, letters, model, **setting)


def letters_of(match, variant, trial, models):
    s = settings()
    p = start_p(match.id, variant, trial["w_gru"], trial["f_dead"])
    letters = L.decode_cost(match, -np.log(p), switch_team=trial["switch_team"], switch_dead=trial["switch_dead"],
                            min_possession=int(trial["min_possession"]))
    if trial["snap"]:
        letters = lab_robust.finish(match, letters, WHICH)
    letters = np.asarray(letters).astype("<U1")
    m = models[trial["post"]]
    if trial["short"]:
        letters = lab_short.apply(match, letters, model=m["short"], **{**s["short"], "p_min": trial["short_p_min"]})
    if trial["segments"]:
        letters = lab_segments.apply(match, letters, m["segments"],
                                     **{**s["segments"], "flip": trial["segments_flip"], "merge": trial["segments_merge"]})
    if trial["timing"]:
        letters = lab_timing.apply(match, letters, m["timing"], **{**s["timing"], "p_min": trial["timing_p_min"]})
    return np.asarray(letters).astype("<U1")


def units(variants=VARIANTS):
    return [(m, v) for m in lab.matches(WHICH) for v in variants]


def unit_path(m, variant):
    return os.path.join(RESULTS, "unit_{}_{}.csv".format(m, variant))


def run(worker, n_workers, limit=None):
    os.makedirs(RESULTS, exist_ok=True)
    lab_mix.cache_short_evidence()
    trials = draw_trials()[:limit]
    for m, variant in units()[worker::n_workers]:
        path = unit_path(m, variant)
        done = set(pd.read_csv(path)["trial"]) if os.path.exists(path) else set()
        todo = [t for t in trials if t["trial"] not in done]
        if not todo:
            continue
        started = time.time()
        match = lab_gate.variant_match(m, WHICH, variant)
        models = {post: joblib.load(models_path(m, post)) for post in POST}
        rows = []
        for t in todo:
            rows.append({"trial": t["trial"], "spell_f1": bench_score(match.truth, letters_of(match, variant, t, models))["spell_f1"]})
            if len(rows) == 50:
                pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                rows = []
        if rows:
            pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print("{} {}: {} trials ({:.0f} s)".format(m, variant, len(todo), time.time() - started), flush=True)
        del match, models


# --------------------------------------------------------------------------------------------
# Confirming a trial: its stage models trained on its own chain's letters (not the reference start's)
# --------------------------------------------------------------------------------------------

def trial_setting(trial, stage):
    s = settings()[stage]
    return {"short": {**s, "p_min": trial["short_p_min"]}, "segments": {**s, "flip": trial["segments_flip"],
            "merge": trial["segments_merge"]}, "timing": {**s, "p_min": trial["timing_p_min"]}}[stage]


def base_id(x):
    return x.id.split("~")[0]


def trial_start(x, data, trial):
    p = start_p(base_id(x), data, trial["w_gru"], trial["f_dead"])
    letters = L.decode_cost(x, -np.log(p), switch_team=trial["switch_team"], switch_dead=trial["switch_dead"],
                            min_possession=int(trial["min_possession"]))
    return np.asarray(lab_robust.finish(x, letters, WHICH) if trial["snap"] else letters).astype("<U1")


def oof_tables(matches, data, trial):
    """The trial's chain out of fold on `matches` (one kind of data), each stage's model trained on the other matches'
    tables (lab_mix.add_stage's result, built once per match); returns every match's table per stage."""
    s = settings()
    letters = {x.id: trial_start(x, data, trial) for x in matches}
    tables = {}
    for stage in CHAIN:
        if not trial[stage]:
            continue
        tabs = {x.id: stage_table(stage, x, letters[x.id], s[stage]) for x in matches}
        new = {}
        for x in matches:
            model = stage_model(stage, [tabs[o.id] for o in matches if o.id != x.id], s[stage])
            new[x.id] = apply_stage(stage, x, letters[x.id], model, trial_setting(trial, stage))
        tables[stage], letters = tabs, new
    return tables


def trial_tables(trial):
    """{stage: {id: table}}: the trial's out-of-fold chain on its stage models' data (cached)."""
    path = os.path.join(CACHE, "tables_trial{}.joblib".format(trial["trial"]))
    if os.path.exists(path):
        return joblib.load(path)
    kinds = {"normal": ("normal",), "stressed": ("train",), "both": ("normal", "train")}[trial["post"]]
    tables = {}
    for data in kinds:
        matches = [lab.load(m, WHICH) if data == "normal" else train_copy(m) for m in lab.matches(WHICH)]
        for stage, tabs in oof_tables(matches, data, trial).items():
            tables.setdefault(stage, {}).update(tabs)
        del matches
    joblib.dump(tables, path, compress=3)
    return tables


def trial_models(trial, without=None):
    """{stage: model}: the trial's stage models trained on its data, without one match (or on all of them)."""
    s = settings()
    return {stage: stage_model(stage, [t for i, t in tabs.items() if i.split("~")[0] != without], s[stage])
            for stage, tabs in trial_tables(trial).items()}


def cache_short_evidence():
    """lab_short.evidence once per match AND variant: it reads the ball, and a match keeps its id on every variant
    (lab_mix's cache is keyed by id alone, which is right only for one variant per process, as run's workers are)."""
    original, seen = lab_short.evidence, {}

    def cached(match):
        key = (match.id, getattr(match, "variant", None))
        if key not in seen:
            seen[key] = original(match)
        return seen[key]
    lab_short.evidence = cached


def confirm(trial_ids, variants=VARIANTS):
    """B as deployed (trial 0) and each trial, with their own stage models, on every unit; the B bar's parts 1-2.
    Other variants (normal, p10: for the switch's bar) go to a file of their own and skip the B bar."""
    cache_short_evidence()
    lab_gate.own_dead_cache()
    trials = every_trial()
    ids = [0] + [i for i in trial_ids if i != 0]
    models = {}
    for i in ids:
        started = time.time()
        models[i] = {m: trial_models(trials[i], without=m) for m in lab.matches(WHICH)}
        print("trial {}: stage models ({:.0f} s)".format(i, time.time() - started), flush=True)
    rows = []
    for m, variant in units(variants):
        match = lab_gate.variant_match(m, WHICH, variant)
        for i in ids:
            t = {**trials[i], "post": "own"}
            letters = letters_of(match, variant, t, {"own": models[i][m]})
            rows.append({"trial": i, "match": m, "variant": variant, "spell_f1": bench_score(match.truth, letters)["spell_f1"]})
        print("{} {}: {}".format(m, variant, " ".join("{:.4f}".format(r["spell_f1"]) for r in rows[-len(ids):])), flush=True)
        del match
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    tail = "" if tuple(variants) == VARIANTS else "_" + "_".join(variants)
    df.to_csv(os.path.join(RESULTS, "confirm_{}{}.csv".format("_".join(str(i) for i in ids[1:]), tail)), index=False,
              float_format="%.5f")
    per = df.pivot_table(index="match", columns="trial", values="spell_f1", aggfunc="mean")
    by_variant = df.pivot_table(index="variant", columns="trial", values="spell_f1", aggfunc="mean")
    print(by_variant.round(4).to_string())
    if tail:
        return df
    for i in ids[1:]:
        g = per[i] - per[0]
        worst = float((by_variant[i] - by_variant[0]).min())
        print("trial {}: B bench {:+.4f}, better on {}/10; worst variant {:+.4f}; part 1 {}, part 2 {}".format(
            i, g.mean(), int((g > 0).sum()), worst, bool(g.mean() >= B.GAIN and (g > 0).sum() >= B.BETTER),
            bool(worst >= -B.VARIANT_LOSS)))
    return df


def skillcorner(trial_ids):
    """Each trial with deployment models (augment and gru_b on all 7 DFL matches, stage models on all 10) on
    SkillCorner's 20 matches; trial 0 must give v13's branch B there."""
    from bench import lab_skillcorner
    from bench import skillcorner as sc
    lab_mix.cache_short_evidence()
    trials = every_trial()
    ids = [0] + [i for i in trial_ids if i != 0]
    aug = C.deployment_augment()
    gru = lab_gate.gru_model(B.used_for(B.held_list()[-1]), WHICH, B.net_path(B.held_list()[-1]))
    models = {i: trial_models(trials[i]) for i in ids}
    rows = []
    for m in sc.match_ids():
        started = time.time()
        gt = sc.load(m)["labels"]
        match, X, Xs = C.sc_prepared(m)
        p_aug = C.proba(aug, {"X": X, "X_swapped": Xs})
        p_gru = lab_gate.gru_proba(gru, {"X": X, "X_swapped": Xs}, match.halves)
        for i in ids:
            t = trials[i]
            p = p_aug if t["w_gru"] == 0 else B.mix(p_gru, p_aug, t["w_gru"])
            if t["f_dead"] != 1.0:
                p = p.copy()
                p[:, 2] *= t["f_dead"]
                p = np.round(np.clip(p / p.sum(axis=1, keepdims=True), *L.CLIP), L.DECIMALS)
            letters = L.decode_cost(match, -np.log(p), switch_team=t["switch_team"], switch_dead=t["switch_dead"],
                                    min_possession=int(t["min_possession"]))
            letters = np.asarray(lab_robust.finish(match, letters, WHICH) if t["snap"] else letters).astype("<U1")
            for stage in CHAIN:
                if t[stage]:
                    letters = apply_stage(stage, match, letters, models[i][stage], trial_setting(t, stage))
            rows.append({"match": m, "trial": i, "spell_f1": lab_skillcorner.score(gt, letters)["spell_f1"]})
        print("{}: {} ({:.0f} s)".format(m, " ".join("{:.4f}".format(r["spell_f1"]) for r in rows[-len(ids):]),
                                         time.time() - started), flush=True)
        del match, X, Xs
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "skillcorner_{}.csv".format("_".join(str(i) for i in ids[1:]))), index=False,
              float_format="%.5f")
    per = df.pivot_table(index="match", columns="trial", values="spell_f1")
    for i in ids[1:]:
        g = per[i] - per[0]
        print("trial {}: SkillCorner {:.4f} against B's {:.4f}: {:+.4f}, better on {}/20; part 3 {}".format(
            i, per[i].mean(), per[0].mean(), g.mean(), int((g > 0).sum()), bool(g.mean() >= -B.SC_LOSS)))
    return df


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------

def load_scores():
    """[trial x match] mean over B's 3 variants, and [trial x unit]."""
    frames = []
    for m, variant in units():
        path = unit_path(m, variant)
        if os.path.exists(path):
            frames.append(pd.read_csv(path).drop_duplicates("trial").assign(match=m, variant=variant))
    df = pd.concat(frames, ignore_index=True)
    per_unit = df.pivot_table(index="trial", columns=["match", "variant"], values="spell_f1")
    complete = per_unit.dropna()
    per_match = complete.T.groupby(level="match").mean().T
    return per_match, complete


def nested(per_match):
    rows = []
    for m in per_match.columns:
        others = [o for o in per_match.columns if o != m]
        pick = int(per_match[others].mean(axis=1).idxmax())
        rows.append({"held_out": m, "trial": pick, "score": per_match.loc[pick, m], "b": per_match.loc[0, m]})
    out = pd.DataFrame(rows)
    out["gain"] = out["score"] - out["b"]
    return out


def report():
    per_match, per_unit = load_scores()
    trials = pd.DataFrame(draw_trials())[PARAMS].set_index("trial").loc[per_match.index]
    print("{} trials scored on all {} units".format(len(per_match), per_unit.shape[1]))
    picks = nested(per_match)
    print(picks.round(4).to_string(index=False))
    print("nested: {:.4f} against B's {:.4f}: {:+.4f}, better on {}/{}".format(
        picks.score.mean(), picks.b.mean(), picks.gain.mean(), int((picks.gain > 0).sum()), len(picks)))
    mean = per_match.mean(axis=1)
    best = int(mean.idxmax())
    print("in sample (flattered): trial {} at {:.4f}; B {:.4f}".format(best, mean.max(), mean.loc[0]))
    print("\nthe best trial's settings: {}".format(trials.loc[best].to_dict()))
    print("\nthe nested picks' settings:\n" + trials.loc[sorted(set(picks.trial))].round(3).to_string())
    top = trials.assign(score=mean).sort_values("score", ascending=False)
    print("\nmarginals, top 100 trials against all:")
    for k in ("snap", "short", "segments", "timing", "post"):
        print("  {}: top {}  all {}".format(k, top.head(100)[k].value_counts().to_dict(), top[k].value_counts().to_dict()))
    for k in ("w_gru", "f_dead", "switch_team", "switch_dead", "min_possession"):
        print("  {}: rank correlation with the score {:+.2f}; top-100 median {:.3f}".format(
            k, top[[k, "score"]].corr(method="spearman").iloc[0, 1], top.head(100)[k].median()))
    by_variant = per_unit.T.groupby(level="variant").mean().T
    print("\nby variant, B and the in-sample best:\n" + by_variant.loc[[0, best]].round(4).to_string())
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "report.json"), "w") as handle:
        json.dump({"trials": len(per_match), "picks": picks.to_dict(orient="records"),
                   "nested": {"score": float(picks.score.mean()), "b": float(picks.b.mean()), "gain": float(picks.gain.mean()),
                              "better": int((picks.gain > 0).sum())},
                   "in_sample_best": {"trial": best, "score": float(mean.max()), "settings": trials.loc[best].to_dict()}},
                  handle, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("stress", "prepare", "run", "report", "confirm", "skillcorner"))
    parser.add_argument("--worker", default="0/1")
    parser.add_argument("--limit", type=int, default=None, help="run: only the first N trials (timing)")
    parser.add_argument("--trials", type=int, nargs="+", default=[], help="confirm: the trials (0, B, is always run)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=lab_gate.VARIANTS,
                        help="confirm: the bench variants (B's 3 by default)")
    args = parser.parse_args()
    if args.step == "stress":
        return stress()
    if args.step == "prepare":
        return prepare()
    if args.step == "confirm":
        return confirm(args.trials, tuple(args.variants))
    if args.step == "skillcorner":
        return skillcorner(args.trials)
    if args.step == "run":
        i, n = (int(x) for x in args.worker.split("/"))
        return run(i, n, args.limit)
    return report()


if __name__ == "__main__":
    main()
