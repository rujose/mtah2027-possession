"""Train v6's restart model on the bench and choose its settings (docs/models/v6-restart-team.md,
pieces 2 and 4).

    python -m bench.train_restart       # ~15-20 min; writes bench/results/restart_* and possession/models/restart_*

1. On all 10 bench matches at 41 % damage: v4's dead-ball features and costs, and the restart
   layout features in four orientations.
2. Out-of-sample P(dead) (v4's family, trees) and P(home restarts) (per candidate family): each match
   is predicted by models trained on the other 9.
3. Every combination of family, weight and carry decoded and scored; v4's decoder settings stay.
4. Leave one match out per family; logistic regression unless the trees beat it by more than
   FAMILY_MARGIN; the submission's setting picked on all 10 with v3's tie rule.
5. The chosen family trained on all 10 matches and saved for possession/v6.py.
"""

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bench import chain, degrade, labels, sets
from bench.run import score
from bench.train_dead_ball import pick, training_rows, v4_cost
from bench.tune_decoder import decode
from possession import dead_ball, features, restart, v1, v3, v4
from possession.data import ROOT

# Params
FAMILIES = ("logistic", "trees")
WEIGHTS = (0.25, 0.5, 1, 2)
CARRIES = (1, 3, 5)             # seconds the clue lasts after the last dead-looking frame
LEVEL = 41
CHECK_LEVELS = (30, 55)
EVERY = 5                       # every 5th dead frame for training
FAMILY_MARGIN = 0.01
WORKERS = 4
SETTING = ["family", "weight", "carry"]
DECODER = (v3.SWITCH_TEAM, v4.SWITCH_DEAD, v3.MIN_POSSESSION, v4.MIN_DEAD)   # v4's, unchanged
RESULTS = os.path.join(ROOT, "bench", "results")
pm = labels.pm

_work = {}


def restart_labels(gt):
    """Per frame: the team (pm.H / pm.A) that takes the restart after the stoppage this frame is in;
    -1 outside stoppages, and in a stoppage that ends a half (no restart follows)."""
    out = np.full(len(gt), -1, dtype=np.int8)
    for a, b, v in labels.runs_of(gt):
        if v == pm.D and b < len(gt) and gt[b] in (pm.H, pm.A):
            out[a:b] = gt[b]
    return out


def restart_rows(oriented, who):
    """Every EVERY-th stoppage frame in all four orientations: (X, 1 = "this team" restarts)."""
    frames = np.flatnonzero(who >= 0)[::EVERY]
    X = np.concatenate([o[3][frames] for o in oriented])
    y = np.concatenate([(who[frames] == side).astype(np.int8) for side, _, _, _ in oriented])
    return X, y


def prepare_level(level, matches, data, stats):
    prepared = {m: degrade.prepare(m, data[m], level, stats) for m in matches}
    scale = {name: degrade.calibrate([prepared[m] for m in matches if sets.source(m) == name], level)
             for name in {sets.source(m) for m in matches}}
    return prepared, scale


def match_pass(match, data, level, prepared, scale, stats):
    """Everything one match needs at one damage level, recomputed on demand (the damage is seeded)."""
    track = degrade.tracking(match, data, level, prepared, scale, stats)
    dnames, Xd = features.frame_features(track, data["halves"], data["pitch"])
    clues = v1.compute_clues(track, [], data["halves"], data["pitch"])
    base = v1.build_costs(dict(clues, stopped=np.zeros_like(clues["stopped"]),
                               active=np.zeros_like(clues["active"])))
    return dnames, Xd, base, restart.all_orientations(track, data["halves"], data["pitch"])


def cost_with_restart(base, p_dead, p_home, halves, weight, carry, dead_weight=None):
    """v4's bench costs plus the restart clue (docs/models/v6-restart-team.md, piece 3).
    `dead_weight` is v4's weight unless a set's own chain gives another (bench/chain.py)."""
    cost = v4_cost(base, p_dead, v4.W_DEAD_BALL if dead_weight is None else dead_weight)
    on_h, on_a = restart.costs(restart.restart_belief(p_home, p_dead, halves, int(carry * v1.FPS)), weight)
    cost[:, v1.H] += on_h
    cost[:, v1.A] += on_a
    return cost


def _init(payload):
    _work.update(payload)


def _score_setting(setting):
    family, weight, carry = setting
    rows = []
    for m in _work["matches"]:
        cost = cost_with_restart(_work["base"][m], _work["p_dead"][m], _work["p_home"][family][m],
                                 _work["halves"][m], weight, carry, _work["dead_weight"])
        letters = decode(cost, _work["halves"][m], _work["decoder"])
        for variant, gt in _work["truth"][m].items():
            rows.append({"family": family, "weight": weight, "carry": carry, "labels": variant,
                         "match": m, **score(gt, letters)})
    return rows


def summarise(grid, matches, family=None):
    used = grid[(grid["labels"] == "raw") & grid["match"].isin(matches)]
    if family is not None:
        used = used[used["family"] == family]
    return used.groupby(SETTING, sort=False)[["spell_f1", "possessions"]].mean().reset_index()


def lookup(grid, setting, match, variant="raw"):
    hit = grid[(grid["labels"] == variant) & (grid["match"] == match)]
    for key, value in zip(SETTING, setting):
        hit = hit[hit[key] == value]
    return hit.iloc[0]


def restart_report(gt, letters):
    """For every stoppage followed by play: found (v4's sense), the predicted end minus the true one
    (s), and whether the first possession after the predicted end has the restarting team."""
    pred = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[x] for x in letters.tolist()], dtype=np.int8)
    out = []
    for a, b, v in labels.runs_of(gt):
        if v != pm.D or a == 0 or gt[a - 1] == pm.X or b >= len(gt) or gt[b] == pm.X:
            continue
        dead = np.flatnonzero(pred[a:b] == pm.D)
        if (pred[a:b] == pm.D).mean() < 0.5 or not len(dead):
            out.append((False, np.nan, False))
            continue
        j = a + dead[-1]
        while j < len(pred) and pred[j] == pm.D:
            j += 1
        out.append((True, (j - b) / v1.FPS, j < len(pred) and pred[j] == gt[b]))
    return out


def spell_scores(gt, pred):
    """(true possessions, score 0-5 per true possession: at how many of spellF1's IoU thresholds it
    is matched, one-to-one as the scorer does it, number of predicted possessions)."""
    runs = pm.valid_runs(gt)
    true, predicted = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(true, predicted)
    flags = np.stack([pm.matched_gt_flags(overlaps, len(true), t) for t in pm.SPELL_IOUS], axis=1)
    return true, flags.sum(axis=1), len(predicted)


def accounting(gt, before_letters, after_letters):
    """The two-sided accounting (docs/models/v6-restart-team.md, diagnosis): scores summed over the
    first possession after each restart and over all other possessions, before and after; predicted
    possessions; matched at IoU 0.5."""
    to_codes = lambda letters: np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[x] for x in letters.tolist()], dtype=np.int8)
    true, before, n_before = spell_scores(gt, to_codes(before_letters))
    _, after, n_after = spell_scores(gt, to_codes(after_letters))
    starts = {a: i for i, (a, _, _) in enumerate(true)}
    first = np.zeros(len(true), bool)
    for a, b, v in labels.runs_of(gt):
        if v == pm.D and a > 0 and gt[a - 1] != pm.X and b in starts:
            first[starts[b]] = True
    return {"after_restart_before": int(before[first].sum()), "after_restart_after": int(after[first].sum()),
            "other_before": int(before[~first].sum()), "other_after": int(after[~first].sum()),
            "predicted_before": n_before, "predicted_after": n_after,
            "matched_05_before": int((before >= 3).sum()), "matched_05_after": int((after >= 3).sum()),
            "true": len(true)}


def follow_through(gt, letters, belief):
    """Of the found restarts where the clue points to the right team at the predicted restart, the
    share where the decoded first team is right too (and how many were overridden)."""
    pred = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[x] for x in letters.tolist()], dtype=np.int8)
    right = overridden = 0
    for a, b, v in labels.runs_of(gt):
        if v != pm.D or a == 0 or gt[a - 1] == pm.X or b >= len(gt) or gt[b] == pm.X:
            continue
        dead = np.flatnonzero(pred[a:b] == pm.D)
        if (pred[a:b] == pm.D).mean() < 0.5 or not len(dead):
            continue
        j = a + dead[-1]
        while j < len(pred) and pred[j] == pm.D:
            j += 1
        known = belief[a:j + 1][~np.isnan(belief[a:j + 1])]
        if not len(known) or (known[-1] > 0.5) != (gt[b] == pm.H) or j >= len(pred):
            continue
        right += int(pred[j] == gt[b])
        overridden += int(pred[j] != gt[b])
    return right, overridden


def summed_restarts(reports):
    rows = [r for rep in reports for r in rep]
    found = [r for r in rows if r[0]]
    err = np.array([r[1] for r in found])
    return {"restarts": len(rows), "found": len(found),
            "right_team_after_found": float(np.mean([r[2] for r in found])),
            "end_within_1s": float(np.mean(np.abs(err) <= 1)), "end_within_3s": float(np.mean(np.abs(err) <= 3))}


def main():
    which = chain.which_set("v6's restart model and settings on the bench")
    tag = chain.suffix(which)
    dsetting, decoder = chain.dead_ball(which), chain.decoder(which)
    stats = degrade.load_stats()
    matches = chain.matches(which)
    data = {m: sets.load(m) for m in matches}
    truth, who = {}, {}
    for m in matches:
        raw = sets.truth(m, data[m])
        truth[m] = {"raw": raw, "merged": labels.merge_short(raw)}
        who[m] = restart_labels(raw)

    # Pass 1: training rows and costs.
    prepared, scale = prepare_level(LEVEL, matches, data, stats)
    dead_rows, start_rows, base = {}, {}, {}
    for m in matches:
        dnames, Xd, base[m], oriented = match_pass(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        dead_rows[m] = training_rows(Xd, truth[m]["raw"])
        start_rows[m] = restart_rows(oriented, who[m])
        rnames = oriented[0][2]
        print("{}: restart rows {}".format(m, len(start_rows[m][1])), flush=True)

    # Fold models: v4's dead-ball family (trees), and both restart candidates.
    dead_models, start_models = {}, {f: {} for f in FAMILIES}
    for held in matches:
        others = [m for m in matches if m != held]
        dead_models[held] = dead_ball.fit(dsetting["family"], np.concatenate([dead_rows[m][0] for m in others]),
                                          np.concatenate([dead_rows[m][1] for m in others]), dnames)
        for family in FAMILIES:
            start_models[family][held] = dead_ball.fit(
                family, np.concatenate([start_rows[m][0] for m in others]),
                np.concatenate([start_rows[m][1] for m in others]), rnames)
        print("fold models for {} trained".format(held), flush=True)

    # Pass 2: out-of-sample probabilities.
    p_dead, p_home = {}, {f: {} for f in FAMILIES}
    for m in matches:
        _, Xd, _, oriented = match_pass(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        p_dead[m] = dead_ball.predict(dead_models[m], Xd)
        for family in FAMILIES:
            p_home[family][m] = restart.p_home(start_models[family][m], oriented)
    del prepared

    # Frame-level restart accuracy on stoppage frames, per family.
    quality = {}
    for family in FAMILIES:
        right = total = 0
        for m in matches:
            frames = who[m] >= 0
            said_home = p_home[family][m][frames] > 0.5
            right += int((said_home == (who[m][frames] == pm.H)).sum())
            total += int(frames.sum())
        quality[family] = right / total
        print("{}: restart team right on {:.1%} of stoppage frames (out of sample)".format(family, quality[family]), flush=True)

    # The grid.
    settings = list(itertools.product(FAMILIES, WEIGHTS, CARRIES))
    payload = {"matches": matches, "base": base, "p_dead": p_dead, "p_home": p_home,
               "halves": {m: data[m]["halves"] for m in matches}, "truth": truth,
               "decoder": decoder, "dead_weight": dsetting["weight"]}
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(payload,)) as pool:
        grid = pd.DataFrame([r for result in pool.map(_score_setting, settings) for r in result])
    os.makedirs(RESULTS, exist_ok=True)
    grid.to_csv(os.path.join(RESULTS, "restart_grid{}.csv".format(tag)), index=False, float_format="%.4f")

    # v4 / v5 on the same matches and probabilities, as the reference.
    v4_letters = {m: decode(v4_cost(base[m], p_dead[m], dsetting["weight"]), data[m]["halves"], decoder) for m in matches}
    v4_score = {m: score(truth[m]["raw"], v4_letters[m])["spell_f1"] for m in matches}

    folds = {f: [] for f in FAMILIES}
    for family in FAMILIES:
        for held in matches:
            chosen = pick(summarise(grid, [m for m in matches if m != held], family))
            setting = (chosen["family"], float(chosen["weight"]), float(chosen["carry"]))
            row = lookup(grid, setting, held)
            folds[family].append({"held_out": held, "setting": list(setting),
                                  "held_out_spell_f1": float(row["spell_f1"]), "v4_spell_f1": v4_score[held],
                                  "possessions": int(row["possessions"]), "true_possessions": int(row["true_possessions"])})
        held_scores = np.array([f["held_out_spell_f1"] for f in folds[family]])
        print("{}: held-out spellF1 {:.3f} +- {:.3f}; v4/v5 {:.3f}; better on {}/{}".format(
            family, held_scores.mean(), held_scores.std(ddof=1), np.mean(list(v4_score.values())),
            int((held_scores > np.array([v4_score[m] for m in matches])).sum()), len(matches)), flush=True)
    held_mean = {f: float(np.mean([x["held_out_spell_f1"] for x in folds[f]])) for f in FAMILIES}
    family = "trees" if held_mean["trees"] > held_mean["logistic"] + FAMILY_MARGIN else "logistic"
    for fold in folds[family]:
        print("  fold {held_out}: {setting} -> {held_out_spell_f1:.3f} (v4/v5 {v4_spell_f1:.3f}), "
              "possessions {possessions} / {true_possessions}".format(**fold))

    chosen = pick(summarise(grid, matches, family))
    setting = (chosen["family"], float(chosen["weight"]), float(chosen["carry"]))
    plain = summarise(grid, matches, family).sort_values("spell_f1", ascending=False, kind="stable").iloc[0]
    print("family {}; chosen on all {}: {} mean spellF1 {:.3f}, possessions {:.0f}; plain best {} {:.3f}".format(
        family, len(matches), setting, chosen["spell_f1"], chosen["possessions"],
        (plain["family"], float(plain["weight"]), float(plain["carry"])), plain["spell_f1"]), flush=True)

    # Checks.
    v6_letters = {m: decode(cost_with_restart(base[m], p_dead[m], p_home[family][m], data[m]["halves"],
                                              setting[1], setting[2], dsetting["weight"]), data[m]["halves"], decoder)
                  for m in matches}
    checks = {"restarts": {}, "per_set": {}, "levels": {}}
    for name in sorted({sets.source(m) for m in matches}, key=("idsse", "metrica").index):
        ms = [m for m in matches if sets.source(m) == name]
        checks["restarts"][name] = {
            "v4_v5": summed_restarts([restart_report(truth[m]["raw"], v4_letters[m]) for m in ms]),
            "v6": summed_restarts([restart_report(truth[m]["raw"], v6_letters[m]) for m in ms])}
        checks["per_set"][name] = {"v6": float(np.mean([lookup(grid, setting, m)["spell_f1"] for m in ms])),
                                   "v4_v5": float(np.mean([v4_score[m] for m in ms]))}
    sums = {}
    agree = overridden = 0
    for m in matches:
        for key, value in accounting(truth[m]["raw"], v4_letters[m], v6_letters[m]).items():
            sums[key] = sums.get(key, 0) + value
        belief = restart.restart_belief(p_home[family][m], p_dead[m], data[m]["halves"], int(setting[2] * v1.FPS))
        r, o = follow_through(truth[m]["raw"], v6_letters[m], belief)
        agree, overridden = agree + r, overridden + o
    checks["accounting"] = sums
    checks["follow_through"] = {"clue_right_and_v6_right": agree, "overridden": overridden}
    checks["merged_labels"] = {
        "v6": float(np.mean([lookup(grid, setting, m, "merged")["spell_f1"] for m in matches])),
        "v4_v5": float(np.mean([score(truth[m]["merged"], v4_letters[m])["spell_f1"] for m in matches]))}
    for level in CHECK_LEVELS:
        prepared, scale = prepare_level(level, matches, data, stats)
        v6s, v4s = [], []
        for m in matches:
            _, Xd, b, oriented = match_pass(m, data[m], level, prepared[m], scale[sets.source(m)], stats)
            pd_ = dead_ball.predict(dead_models[m], Xd)
            ph = restart.p_home(start_models[family][m], oriented)
            v6s.append(score(truth[m]["raw"], decode(cost_with_restart(b, pd_, ph, data[m]["halves"], setting[1], setting[2], dsetting["weight"]),
                                                     data[m]["halves"], decoder))["spell_f1"])
            v4s.append(score(truth[m]["raw"], decode(v4_cost(b, pd_, dsetting["weight"]), data[m]["halves"], decoder))["spell_f1"])
        checks["levels"][str(level)] = {"v6": float(np.mean(v6s)), "v4_v5": float(np.mean(v4s))}
        del prepared
    print("checks: {}".format(json.dumps(checks, indent=1)), flush=True)

    final = dead_ball.fit(family, np.concatenate([start_rows[m][0] for m in matches]),
                          np.concatenate([start_rows[m][1] for m in matches]), rnames)
    path = dead_ball.save(final, name="restart" + tag)
    choice = {"family": family, "weight": setting[1], "carry_s": setting[2],
              "bench_mean_spell_f1": float(chosen["spell_f1"]), "bench_mean_possessions": float(chosen["possessions"]),
              "plain_best": {"setting": [plain["family"], float(plain["weight"]), float(plain["carry"])],
                             "spell_f1": float(plain["spell_f1"])},
              "held_out_mean": held_mean, "restart_frame_accuracy": quality, "folds": folds,
              "v4_v5_spell_f1": v4_score, "checks": checks, "model_file": os.path.relpath(path, ROOT),
              "features": rnames}
    with open(os.path.join(RESULTS, "restart_choice{}.json".format(tag)), "w") as handle:
        json.dump(choice, handle, indent=1)
    print("wrote bench/results/restart_grid{0}.csv, bench/results/restart_choice{0}.json and {1}".format(
        tag, os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
