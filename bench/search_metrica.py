"""A nested random search of v13's settings on Metrica's convention (docs/search-metrica.md).

    python -m bench.search_metrica prepare            # ~10 min: out-of-fold letters, one set of stage models per game
    python -m bench.search_metrica run --worker I/N   # trials I, I+N, I+2N, ... (run N workers side by side)
    python -m bench.search_metrica report             # nested picks, match vs half protocol, guard on DFL's 7

Why: the answer key follows Metrica's owner convention (docs/owner-rule.md), and every lab verdict so far was
decided mostly by DFL's 7 matches. This searches v13's branch A settings for Metrica's 3 sample games, honestly:

- **The trials are drawn at random, fixed by a seed, before any is scored**, so no game's score steers which
  settings get tried (an adaptive search guided by all 3 games would leak the held-out one into the search).
- **Each trial is scored on each Metrica game with stage models trained without that game** (the lab's fold
  models, refitted on v13's own out-of-fold letters; trained once, then applied with each trial's thresholds).
- **Match protocol (honest):** for each game, the trial with the best mean on the other two games' halves is
  picked; only its score on the held-out game counts.
- **Half protocol (Jose's question, 2026-11-12):** for each half, the trial with the best mean on the other five
  halves (its sibling half included) is picked; its score on the held-out half. Both protocols use the same trials,
  so the gap between them is how much a held-out half flatters.
- **Guard:** each pick also runs on DFL's 7 matches; the report says what it costs there.

What a trial sets (v13 = trial 0): the GRU's weight in the start (v13 0.5), a factor on P(dead), the decoder's
switch prices and minimum possession, v9's snap and v11's move on/off, and each post stage on/off with its
thresholds. Nothing reads our match.
"""

import argparse
import glob
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from bench import lab, lab_mix, lab_robust, lab_segments, lab_short, lab_temporal, lab_timing, sets, train_v13
from bench.run import score as bench_score
from possession import v13

# Params
WHICH = "all"
GAMES = ("metrica1", "metrica2", "metrica3")
CACHE = os.path.join(lab.ROOT, "data", "external", "search_metrica")      # gitignored (data/*)
RESULTS = os.path.join(lab.RESULTS, "search_metrica")
TRIALS = 2000              # ~3 s a trial on the 3 games (timed 2026-11-12), so ~25 min on 4 workers
SEED = 20261112
OFF_SHARE = 0.2            # each post stage is off in this share of trials
SPACE = {                  # (low, high, log scale?)
    "w_gru": (0.0, 1.0, False),
    "f_dead": (0.5, 2.0, True),
    "switch_team": (0.5, 16.0, True),
    "switch_dead": (8.0, 100.0, True),
    "short_p_min": (0.3, 0.9, False),
    "segments_flip": (0.5, 0.95, False),
    "segments_merge": (0.5, 0.95, False),
    "timing_p_min": (0.2, 0.8, False),
}
MIN_POSSESSION = (4, 30)   # frames, integers
PARAMS = ["trial", "w_gru", "f_dead", "switch_team", "switch_dead", "min_possession", "snap", "short", "short_p_min",
          "segments", "segments_flip", "segments_merge", "timing", "timing_p_min"]     # one column order for every row


def v13_trial():
    """Trial 0: v13's branch A as built."""
    s = train_v13.chosen(WHICH)
    return {"trial": 0, "w_gru": 0.5, "f_dead": 1.0, "switch_team": s["start"]["switch_team"],
            "switch_dead": s["start"]["switch_dead"], "min_possession": int(s["start"]["min_possession"]), "snap": True,
            "short": True, "short_p_min": s["short"]["p_min"], "segments": True, "segments_flip": s["segments"]["flip"],
            "segments_merge": s["segments"]["merge"], "timing": True, "timing_p_min": s["timing"]["p_min"]}


def draw_trials(n=TRIALS, seed=SEED):
    """v13 plus n - 1 random settings, fixed by the seed before anything is scored."""
    rng = np.random.default_rng(seed)
    trials = [v13_trial()]
    for i in range(1, n):
        t = {"trial": i}
        for k, (lo, hi, log) in SPACE.items():
            t[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
        t["min_possession"] = int(rng.integers(MIN_POSSESSION[0], MIN_POSSESSION[1] + 1))
        t["snap"] = bool(rng.random() < 0.5)
        for stage in ("short", "segments", "timing"):
            t[stage] = bool(rng.random() >= OFF_SHARE)
        trials.append(t)
    return trials


# --------------------------------------------------------------------------------------------
# Preparing: the stage models for each game, trained without it
# --------------------------------------------------------------------------------------------

def models_path(game):
    return os.path.join(CACHE, "post_without_{}.joblib".format(game))


def prepare():
    os.makedirs(CACHE, exist_ok=True)
    lab_mix.cache_short_evidence()
    matches = list(lab.each(WHICH))
    setting = train_v13.chosen(WHICH)
    per_match = {stage: {m.id: s for m in matches} for stage, s in setting.items()}
    steps = train_v13.out_of_fold(matches, WHICH, per_match)
    for game in GAMES:
        others = [m for m in matches if m.id != game]
        models = train_v13.fit_post(others, WHICH, steps, setting)
        joblib.dump(models, models_path(game), compress=3)
        print("{}: stage models trained on the other 9 matches".format(game), flush=True)


# --------------------------------------------------------------------------------------------
# One trial on one match
# --------------------------------------------------------------------------------------------

def start_p(match, w_gru, f_dead):
    """The start's P: the GRU's and robust's held-out P mixed (w_gru = 0.5 is v13's `average` exactly), P(dead)
    scaled by f_dead and renormalised, clipped and rounded as v13's."""
    gru = lab_temporal.held_out(match, WHICH, "temporal")
    rob = lab_robust.held_out_proba(match, WHICH, "blank")
    if w_gru == 0.5:
        p = v13.average(gru, rob)
    else:
        p = v13.clipped(w_gru * gru.astype(np.float32).astype(np.float64) + (1 - w_gru) * rob.astype(np.float32).astype(np.float64))
    if f_dead != 1.0:
        p = p.copy()
        p[:, 2] *= f_dead
        p = v13.clipped(p / p.sum(axis=1, keepdims=True))
    return p


def letters_of(match, trial, models):
    """Every frame's letter under a trial's settings, with fixed stage models."""
    s = train_v13.chosen(WHICH)
    p = start_p(match, trial["w_gru"], trial["f_dead"])
    letters = lab_robust.L.decode_cost(match, -np.log(p), switch_team=trial["switch_team"], switch_dead=trial["switch_dead"],
                                       min_possession=int(trial["min_possession"]))
    if trial["snap"]:
        letters = lab_robust.finish(match, letters, WHICH)
    letters = np.asarray(letters).astype("<U1")
    if trial["short"]:
        letters = lab_short.apply(match, letters, model=models["short"], **{**s["short"], "p_min": trial["short_p_min"]})
    if trial["segments"]:
        letters = lab_segments.apply(match, letters, models["segments"],
                                     **{**s["segments"], "flip": trial["segments_flip"], "merge": trial["segments_merge"]})
    if trial["timing"]:
        letters = lab_timing.apply(match, letters, models["timing"], **{**s["timing"], "p_min": trial["timing_p_min"]})
    return np.asarray(letters).astype("<U1")


def scores(match, letters):
    """spellF1 on the whole match and on each half alone (the other half made ungraded, X)."""
    out = {"match": bench_score(match.truth, letters)["spell_f1"]}
    for i, (_, first, last) in enumerate(match.halves):
        truth = np.full_like(match.truth, 3)
        truth[first:last + 1] = match.truth[first:last + 1]
        out["half{}".format(i + 1)] = bench_score(truth, letters)["spell_f1"]
    return out


def run(worker, n_workers):
    os.makedirs(RESULTS, exist_ok=True)
    lab_mix.cache_short_evidence()
    trials = draw_trials()
    games = {g: lab.load(g, WHICH) for g in GAMES}
    models = {g: joblib.load(models_path(g)) for g in GAMES}
    path = os.path.join(RESULTS, "trials_{}.csv".format(worker))
    done = set(pd.read_csv(path)["trial"]) if os.path.exists(path) else set()
    for t in trials[worker::n_workers]:
        if t["trial"] in done:
            continue
        started = time.time()
        row = {k: t[k] for k in PARAMS}
        for g, match in games.items():
            for k, v in scores(match, letters_of(match, t, models[g])).items():
                row["{}_{}".format(g, k)] = v
        pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print("trial {}: {:.4f} mean over the 3 games ({:.0f} s)".format(
            t["trial"], np.mean([row["{}_match".format(g)] for g in GAMES]), time.time() - started), flush=True)


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------

def load_trials():
    """Every scored trial: its scores from the CSVs, its settings from the seeded draw. (Before 2026-11-12 ~15:00 a
    worker's file could hold its rows' settings in two column orders; the scores, written after all 14 settings,
    and the trial ids were always in place, so only the settings are re-read.)"""
    frame = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RESULTS, "trials_*.csv"))], ignore_index=True)
    score_cols = [c for c in frame.columns if c.startswith(GAMES)]
    frame = frame[["trial"] + score_cols].drop_duplicates("trial")
    params = pd.DataFrame(draw_trials())[PARAMS]
    return params.merge(frame, on="trial").sort_values("trial").reset_index(drop=True)


def halves_of(game):
    return ["{}_half1".format(game), "{}_half2".format(game)]


def nested(frame):
    """Both protocols: for each held-out unit, the trial picked on the other units and its held-out score."""
    rows = []
    for g in GAMES:                                   # match protocol: pick on the other games' halves
        others = [c for o in GAMES if o != g for c in halves_of(o)]
        pick = frame.loc[frame[others].mean(axis=1).idxmax()]
        for h in halves_of(g):
            rows.append({"protocol": "match", "held_out": h, "trial": int(pick["trial"]), "score": pick[h],
                         "v13": frame.loc[frame["trial"] == 0, h].item()})
    units = [h for g in GAMES for h in halves_of(g)]
    for h in units:                                   # half protocol: pick on the other five halves
        pick = frame.loc[frame[[u for u in units if u != h]].mean(axis=1).idxmax()]
        rows.append({"protocol": "half", "held_out": h, "trial": int(pick["trial"]), "score": pick[h],
                     "v13": frame.loc[frame["trial"] == 0, h].item()})
    return pd.DataFrame(rows)


def guard(trial_ids):
    """Each pick on DFL's 7 matches (their own fold stage models would be fairer; these are the
    Metrica-held-out ones, so this is a rough check of what a pick costs there)."""
    models = joblib.load(models_path(GAMES[0]))
    trials = {t["trial"]: t for t in draw_trials()}
    out = {}
    for m in [m for m in lab.matches(WHICH) if sets.source(m) == "idsse"]:
        match = lab.load(m, WHICH)
        out[m] = {i: bench_score(match.truth, letters_of(match, trials[i], models))["spell_f1"] for i in [0] + list(trial_ids)}
        print("{}: {}".format(m, {k: round(v, 4) for k, v in out[m].items()}), flush=True)
    return pd.DataFrame(out).T


def report():
    lab_mix.cache_short_evidence()
    frame = load_trials()
    print("{} trials scored".format(len(frame)))
    picks = nested(frame)
    picks["gain"] = picks["score"] - picks["v13"]
    print(picks.round(4).to_string(index=False))
    summary = picks.groupby("protocol").agg(score=("score", "mean"), v13=("v13", "mean"), gain=("gain", "mean"),
                                            better=("gain", lambda x: int((x > 0).sum())), units=("gain", "size"))
    print("\n" + summary.round(4).to_string())
    in_sample = frame[[c for g in GAMES for c in halves_of(g)]].mean(axis=1)
    print("\nthe best trial on all 6 halves (in sample, flattered): trial {} at {:.4f}; v13 {:.4f}".format(
        int(frame.loc[in_sample.idxmax(), "trial"]), in_sample.max(), in_sample[frame["trial"] == 0].item()))
    ids = sorted(set(picks.loc[picks["protocol"] == "match", "trial"]))
    dfl = guard(ids)
    print("\nDFL's 7 (rough guard):\n" + dfl.round(4).to_string())
    with open(os.path.join(RESULTS, "report.json"), "w") as handle:
        json.dump({"trials": len(frame), "picks": picks.to_dict(orient="records"), "summary": summary.reset_index().to_dict(orient="records"),
                   "in_sample_best": {"trial": int(frame.loc[in_sample.idxmax(), "trial"]), "score": float(in_sample.max())},
                   "dfl_guard": dfl.to_dict()}, handle, indent=1, default=float)


# --------------------------------------------------------------------------------------------
# Confirming a candidate the lab's way (every stage refitted out of fold, each match on its fold settings)
# --------------------------------------------------------------------------------------------

def held_out_start(match, which, w_gru):
    gru = lab_temporal.held_out(match, which, "temporal")
    rob = lab_robust.held_out_proba(match, which, "blank")
    if w_gru == 0.5:
        return v13.average(gru, rob)
    return v13.clipped(w_gru * gru.astype(np.float32).astype(np.float64) + (1 - w_gru) * rob.astype(np.float32).astype(np.float64))


def confirm(which, w_gru, snap):
    """v13's chain (the lab's `average` start, snapped) against the candidate start, each followed by v12's post stages
    refitted on its letters (lab_mix.add_stage), judged with lab_mix.judge_scores; the start decoded with `average`'s
    pick for each match (train_v13.start_folds), as lab_temporal.chains did."""
    lab_mix.cache_short_evidence()
    matches = list(lab.each(which))
    picks, start = lab_mix.fold_settings(which), train_v13.start_folds(which)
    scores_by = {}
    for name, (w, sn) in (("v13", (0.5, True)), ("candidate", (w_gru, snap))):
        letters = {}
        for m in matches:
            decoded = lab_robust.L.decode_cost(m, -np.log(held_out_start(m, which, w)), **start[m.id])
            letters[m.id] = np.asarray(lab_robust.finish(m, decoded, which) if sn else decoded).astype("<U1")
        for stage in v13.CHAIN[which]:
            letters = lab_mix.add_stage(stage, letters, matches, picks)
        scores_by[name] = lab_mix.scores_of(letters, matches)
        print("{}: {:.4f}".format(name, np.mean([v["spell_f1"] for v in scores_by[name].values()])), flush=True)
    report = lab_mix.judge_scores(scores_by["candidate"], scores_by["v13"], which)
    print(lab.describe(report))
    table = pd.DataFrame({k: {m: v["spell_f1"] for m, v in s.items()} for k, s in scores_by.items()})
    table["gain"] = table["candidate"] - table["v13"]
    print(table.round(4).to_string())
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "confirm_w{}_snap{}{}.json".format(w_gru, int(snap), "" if which == "all" else "_" + which)), "w") as handle:
        json.dump({"w_gru": w_gru, "snap": snap, "which": which, "scores": table.round(5).to_dict(), "against_v13": report},
                  handle, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("prepare", "run", "report", "confirm"))
    parser.add_argument("--worker", default="0/1", help="I/N: this worker's share of the trials")
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which", help="confirm: the bench set")
    parser.add_argument("--w-gru", type=float, default=0.8, help="confirm: the GRU's weight in the start")
    parser.add_argument("--no-snap", action="store_true", help="confirm: v9's snap and v11's move off")
    args = parser.parse_args()
    if args.step == "prepare":
        return prepare()
    if args.step == "run":
        i, n = (int(x) for x in args.worker.split("/"))
        return run(i, n)
    if args.step == "confirm":
        return confirm(args.which, args.w_gru, not args.no_snap)
    return report()


if __name__ == "__main__":
    main()
