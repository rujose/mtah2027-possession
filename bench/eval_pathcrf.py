"""PathCRF on the bench, and v7's settings (docs/models/v7-pathcrf.md, pieces 3 and 5).

    python -m bench.eval_pathcrf        # ~15 min; writes bench/results/pathcrf_*

1. Out-of-sample v6 on all 10 bench matches at 41 % damage (its dead-ball and restart models
   trained without each match, as in bench/train_restart.py).
2. PathCRF on each match's damaged tracking, over the stretches v6 calls live.
3. Its quality: team accuracy inside ball gaps and with the ball seen, against v6 on the same
   frames; changes of possession caught inside gaps.
4. v7 = v6 + PathCRF's clue; weight and where it applies chosen by leave-one-match-out over the
   4 matches PathCRF never saw (J03WR9 and Metrica's 3); the 6 it trained or validated on are
   reported separately, as flattering.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd

from bench import chain, degrade, labels, sets
from bench.run import score
from bench.train_dead_ball import pick, training_rows, v4_cost
from bench.train_restart import (accounting, cost_with_restart, match_pass, prepare_level,
                                 restart_labels, restart_rows)
from bench.tune_decoder import decode
from possession import dead_ball, pathcrf_adapter, restart, v1
from possession.data import ROOT

# Params
UNSEEN = ("J03WR9", "metrica1", "metrica2", "metrica3")   # third_party/pathcrf/VENDORED.md
WEIGHTS = (0.25, 0.5, 1, 2)
WHERE = ("gaps", "live")       # only while the ball is missing, or on every live frame it answers
LEVEL = 41
RESULTS = os.path.join(ROOT, "bench", "results")
pm = labels.pm
LETTER = {"H": 0, "A": 1, "D": 2, "X": 3}


def v7_cost(v6_cost, p_pathcrf, ball_missing, weight, where):
    """v6's costs plus PathCRF's clue: only the unlikely team pays, by the odds (restart.costs)."""
    belief = p_pathcrf.copy()
    if where == "gaps":
        belief[~ball_missing] = np.nan
    on_h, on_a = restart.costs(belief, weight)
    cost = v6_cost.copy()
    cost[:, v1.H] += on_h
    cost[:, v1.A] += on_a
    return cost


def changes_caught(gt, letters, ball_missing):
    """True direct changes of possession (no stoppage between) while the ball is missing, and how
    many the prediction makes to the right team within 2 s."""
    pred = np.array([LETTER[x] for x in letters.tolist()])
    runs = labels.runs_of(gt)
    total = caught = 0
    for (a1, b1, k1), (a2, b2, k2) in zip(runs, runs[1:]):
        if k1 in (pm.H, pm.A) and k2 in (pm.H, pm.A) and k1 != k2 and ball_missing[a2]:
            total += 1
            w = pred[max(0, a2 - 2 * v1.FPS):a2 + 2 * v1.FPS + 1]
            caught += int(any(w[i] != w[i + 1] and w[i] in (pm.H, pm.A) and w[i + 1] == k2 for i in range(len(w) - 1)))
    return total, caught


def main():
    which = chain.which_set("PathCRF on the bench, and v7's settings")
    tag = chain.suffix(which)
    dsetting, decoder, rsetting = chain.dead_ball(which), chain.decoder(which), chain.restart(which)
    unseen = chain.unseen(which, UNSEEN)
    stats = degrade.load_stats()
    matches = chain.matches(which)
    data = {m: sets.load(m) for m in matches}
    truth = {m: sets.truth(m, data[m]) for m in matches}
    who = {m: restart_labels(truth[m]) for m in matches}

    # 1. Out-of-sample v6 (as bench/train_restart.py does it).
    prepared, scale = prepare_level(LEVEL, matches, data, stats)
    dead_rows, start_rows, base = {}, {}, {}
    for m in matches:
        dnames, Xd, base[m], oriented = match_pass(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        dead_rows[m], start_rows[m], rnames = training_rows(Xd, truth[m]), restart_rows(oriented, who[m]), oriented[0][2]
    model = pathcrf_adapter.load_model()
    v6_cost, v6_letters, p_pc, missing, p_dead_of, p_start_of = {}, {}, {}, {}, {}, {}
    for held in matches:
        others = [m for m in matches if m != held]
        dead_model = dead_ball.fit(dsetting["family"], np.concatenate([dead_rows[m][0] for m in others]),
                                   np.concatenate([dead_rows[m][1] for m in others]), dnames)
        start_model = dead_ball.fit(rsetting["family"], np.concatenate([start_rows[m][0] for m in others]),
                                    np.concatenate([start_rows[m][1] for m in others]), rnames)
        _, Xd, _, oriented = match_pass(held, data[held], LEVEL, prepared[held], scale[sets.source(held)], stats)
        p_dead = dead_ball.predict(dead_model, Xd)
        p_start = restart.p_home(start_model, oriented)
        p_dead_of[held], p_start_of[held] = p_dead, p_start
        halves = data[held]["halves"]
        v6_cost[held] = cost_with_restart(base[held], p_dead, p_start, halves, rsetting["weight"], rsetting["carry"],
                                          dsetting["weight"])
        v6_letters[held] = decode(v6_cost[held], halves, decoder)
        # 2. PathCRF over v6's live stretches, on the damaged tracking.
        track = degrade.tracking(held, data[held], LEVEL, prepared[held], scale[sets.source(held)], stats)
        missing[held] = np.isnan(track["ball"][:, 0])
        live = np.isin(v6_letters[held], ["H", "A"])
        p_pc[held] = pathcrf_adapter.p_home(model, track, halves, data[held]["pitch"], live)
        print("{}: v6 decoded, PathCRF answered on {:.1%} of v6's live frames".format(
            held, np.mean(~np.isnan(p_pc[held][live]))), flush=True)

    # 3. Quality, against v6 on the same frames.
    quality = {}
    for name, group in (("unseen", list(unseen)), ("seen", [m for m in matches if m not in unseen])):
        q = {}
        for zone in ("gaps", "seen_ball"):
            right_pc = right_v6 = n = 0
            for m in group:
                gt = truth[m]
                pred = np.array([LETTER[x] for x in v6_letters[m].tolist()])
                where = np.isin(gt, (pm.H, pm.A)) & ~np.isnan(p_pc[m]) & np.isin(pred, (pm.H, pm.A))
                where &= missing[m] if zone == "gaps" else ~missing[m]
                n += int(where.sum())
                right_pc += int(((p_pc[m][where] > 0.5) == (gt[where] == pm.H)).sum())
                right_v6 += int((pred[where] == gt[where]).sum())
            q[zone] = {"frames": n, "pathcrf_right": right_pc / n, "v6_right": right_v6 / n}
        quality[name] = q
    print("quality: {}".format(json.dumps(quality, indent=1)), flush=True)

    # 4. The grid, and leave-one-match-out over the unseen matches.
    rows = []
    for weight, where in itertools.product(WEIGHTS, WHERE):
        for m in matches:
            letters = decode(v7_cost(v6_cost[m], p_pc[m], missing[m], weight, where), data[m]["halves"], decoder)
            for variant, gt in (("raw", truth[m]), ("merged", labels.merge_short(truth[m]))):
                rows.append({"weight": weight, "where": where, "labels": variant, "match": m, **score(gt, letters)})
    grid = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    grid.to_csv(os.path.join(RESULTS, "pathcrf_grid{}.csv".format(tag)), index=False, float_format="%.4f")
    v6_score = {m: score(truth[m], v6_letters[m])["spell_f1"] for m in matches}

    def table(ms):
        used = grid[(grid["labels"] == "raw") & grid["match"].isin(ms)]
        return used.groupby(["weight", "where"], sort=False)[["spell_f1", "possessions"]].mean().reset_index()

    folds = []
    for held in unseen:
        # With one unseen match (IDSSE only) the weight is chosen on that match: not held out.
        chosen = pick(table([m for m in unseen if m != held] or [held]))
        row = grid[(grid["labels"] == "raw") & (grid["match"] == held) & (grid["weight"] == chosen["weight"])
                   & (grid["where"] == chosen["where"])].iloc[0]
        folds.append({"held_out": held, "setting": [float(chosen["weight"]), chosen["where"]],
                      "held_out_spell_f1": float(row["spell_f1"]), "v6_spell_f1": v6_score[held]})
        print("fold {}: picked {} -> {:.3f} (v6 {:.3f})".format(held, (chosen["weight"], chosen["where"]),
                                                                row["spell_f1"], v6_score[held]), flush=True)
    chosen = pick(table(list(unseen)))
    setting = (float(chosen["weight"]), chosen["where"])
    seen = [m for m in matches if m not in unseen]
    seen_rows = grid[(grid["labels"] == "raw") & grid["match"].isin(seen) & (grid["weight"] == setting[0])
                     & (grid["where"] == setting[1])]
    held = np.array([f["held_out_spell_f1"] for f in folds])
    print("held out on the unseen {}: {:.3f} (v6 {:.3f}), better on {}/{}; chosen {}; on the {} seen: {:.3f} (v6 {:.3f})".format(
        len(unseen), held.mean(), np.mean([v6_score[m] for m in unseen]),
        int((held > np.array([v6_score[m] for m in unseen])).sum()), len(unseen),
        setting, len(seen), seen_rows["spell_f1"].mean(), np.mean([v6_score[m] for m in seen])), flush=True)

    # Checks for the chosen setting.
    v7_letters = {m: decode(v7_cost(v6_cost[m], p_pc[m], missing[m], *setting), data[m]["halves"], decoder) for m in matches}
    checks = {}
    for name, group in (("unseen", list(unseen)), ("seen", seen)):
        sums = {}
        for m in group:
            for key, value in accounting(truth[m], v6_letters[m], v7_letters[m]).items():
                sums[key] = sums.get(key, 0) + value
        t6 = [changes_caught(truth[m], v6_letters[m], missing[m]) for m in group]
        t7 = [changes_caught(truth[m], v7_letters[m], missing[m]) for m in group]
        checks[name] = {"accounting": sums,
                        "changes_in_gaps": sum(t[0] for t in t6),
                        "caught_v6": sum(t[1] for t in t6), "caught_v7": sum(t[1] for t in t7)}
    print("checks: {}".format(json.dumps(checks, indent=1)), flush=True)
    with open(os.path.join(RESULTS, "pathcrf_choice{}.json".format(tag)), "w") as handle:
        json.dump({"setting": {"weight": setting[0], "where": setting[1]}, "folds": folds,
                   "held_out_mean": float(held.mean()), "held_out": len(unseen) > 1, "quality": quality, "checks": checks,
                   "v6_spell_f1": v6_score}, handle, indent=1)
    # Out-of-sample predictions per match, for the bench viewer (viewer/build_bench_cache.py).
    predictions = chain.predictions(which)
    os.makedirs(predictions, exist_ok=True)
    for m in matches:
        halves = data[m]["halves"]
        belief = p_pc[m].copy()
        if setting[1] == "gaps":
            belief[~missing[m]] = np.nan
        np.savez_compressed(
            os.path.join(predictions, "{}.npz".format(m)),
            v4=decode(v4_cost(base[m], p_dead_of[m], dsetting["weight"]), halves, decoder),
            v6=v6_letters[m], v7=v7_letters[m], p_dead=p_dead_of[m],
            restart_clue=restart.restart_belief(p_start_of[m], p_dead_of[m], halves, int(rsetting["carry"] * v1.FPS)),
            pathcrf_clue=belief,
            # For bench/lab.py: the decoder's inputs, so later versions can decode again.
            v6_cost=v6_cost[m].astype(np.float32),
            v7_cost=v7_cost(v6_cost[m], p_pc[m], missing[m], *setting).astype(np.float32),
            p_start=p_start_of[m], p_pathcrf=p_pc[m])
    print("wrote bench/results/pathcrf_grid{0}.csv and pathcrf_choice{0}.json, and {1}".format(tag, os.path.relpath(predictions, ROOT)))


if __name__ == "__main__":
    main()
