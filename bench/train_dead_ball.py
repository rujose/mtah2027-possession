"""Train v4's dead-ball model on the bench and choose its settings (docs/models/v4-dead-ball.md,
pieces 2 and 4).

    python -m bench.train_dead_ball     # ~10 min; writes bench/results/dead_ball_* and possession/models/

1. Players-only features on all 10 bench matches at 41 % damage (possession/features.py).
2. For each model family and each match, a model trained on the other 9 predicts that match, so
   every probability used below is out of sample.
3. Every combination of family, weight, dead-ball price and minimum stoppage decoded and scored
   (team price and minimum possession stay v3's).
4. Leave one match out per family; logistic regression unless the trees beat it by more than
   FAMILY_MARGIN; the submission's setting picked on all 10 with v3's tie rule.
5. The chosen family trained on all 10 matches and saved for possession/v4.py.
"""

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bench import degrade, labels, sets
from bench.run import score
from bench.tune_decoder import TIE, decode
from possession import dead_ball, features, v1, v3
from possession.data import ROOT

# Params
FAMILIES = ("logistic", "trees")
WEIGHTS = (0.25, 0.5, 1, 2)
DEAD_PRICES = (5, 10, 25, 50)
MIN_DEAD = (1, 25, 50)
LEVEL = 41
CHECK_LEVELS = (30, 55)
EVERY = 5                       # train on every 5th frame: neighbours are near-copies
FAMILY_MARGIN = 0.01            # trees must beat logistic regression by more than this
WORKERS = 4
SETTING = ["family", "weight", "dead_price", "min_dead"]
RESULTS = os.path.join(ROOT, "bench", "results")

_work = {}


def match_inputs(match, data, level, prepared, scale, stats):
    """Features, the costs without v1's stopped/active rules, v3's own costs, and the labels."""
    track = degrade.tracking(match, data, level, prepared, scale, stats)
    names, X = features.frame_features(track, data["halves"], data["pitch"])
    clues = v1.compute_clues(track, [], data["halves"], data["pitch"])
    without = dict(clues, stopped=np.zeros_like(clues["stopped"]), active=np.zeros_like(clues["active"]))
    return names, X, v1.build_costs(without), v1.build_costs(clues)


def level_inputs(level, matches, data, stats):
    prepared = {m: degrade.prepare(m, data[m], level, stats) for m in matches}
    scale = {name: degrade.calibrate([prepared[m] for m in matches if sets.source(m) == name], level)
             for name in {sets.source(m) for m in matches}}
    out = {}
    for m in matches:
        out[m] = match_inputs(m, data[m], level, prepared[m], scale[sets.source(m)], stats)
    return out


def training_rows(X, gt):
    """Every EVERY-th frame with a label: (X rows, 1 = dead)."""
    rows = np.flatnonzero(np.isin(gt, (labels.pm.H, labels.pm.A, labels.pm.D)))[::EVERY]
    return X[rows], (gt[rows] == labels.pm.D).astype(np.int8)


def v4_cost(base, p, weight):
    """docs/models/v4-dead-ball.md, piece 3."""
    cost = base.copy()
    alive, dead = dead_ball.costs(p)
    cost[:, v1.H] += weight * alive
    cost[:, v1.A] += weight * alive
    cost[:, v1.D] += weight * dead
    return cost


def _init(payload):
    _work.update(payload)


def _score_setting(setting):
    family, weight, dead, min_dead = setting
    rows = []
    for m in _work["matches"]:
        cost = v4_cost(_work["base"][m], _work["p"][family][m], weight)
        letters = decode(cost, _work["halves"][m], (v3.SWITCH_TEAM, dead, v3.MIN_POSSESSION, min_dead))
        for variant, gt in _work["truth"][m].items():
            rows.append({"family": family, "weight": weight, "dead_price": dead, "min_dead": min_dead,
                         "labels": variant, "match": m, **score(gt, letters)})
    return rows


def pick(table):
    """v3's rule: best mean spellF1, then the fewest possessions within TIE of it, then grid order."""
    best = table["spell_f1"].max()
    return table[table["spell_f1"] >= best - TIE].sort_values("possessions", kind="stable").iloc[0]


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


def dead_ball_report(gt, letters):
    """Dead-ball precision/recall and stoppages found, missed, merged; false stoppages."""
    pred = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[x] for x in letters.tolist()], dtype=np.int8)
    D, X = labels.pm.D, labels.pm.X
    out = {"tp": int(((gt == D) & (pred == D)).sum()), "fp": int(((gt != D) & (gt != X) & (pred == D)).sum()),
           "fn": int(((gt == D) & (pred != D)).sum()), "stoppages": 0, "found": 0, "merged": 0, "false": 0}
    runs = labels.runs_of(gt)
    for i, (a, b, v) in enumerate(runs):
        if v != D or i == 0 or runs[i - 1][2] == X:
            continue
        out["stoppages"] += 1
        if (pred[a:b] == D).sum() >= 0.5 * (b - a):
            out["found"] += 1
        elif i + 1 < len(runs) and runs[i - 1][2] == runs[i + 1][2] and (pred[a:b] != D).all():
            out["merged"] += 1
    pruns = labels.runs_of(pred)
    for i, (a, b, v) in enumerate(pruns):
        if v == D and i > 0 and pruns[i - 1][2] != X and (gt[a:b] == D).mean() < 0.5:
            out["false"] += 1
    return out


def summed_report(reports):
    total = {k: sum(r[k] for r in reports) for k in reports[0]}
    total["precision"] = total["tp"] / max(total["tp"] + total["fp"], 1)
    total["recall"] = total["tp"] / max(total["tp"] + total["fn"], 1)
    return total


def main():
    stats = degrade.load_stats()
    matches = sets.matches("all")
    data = {m: sets.load(m) for m in matches}
    truth = {}
    for m in matches:
        raw = sets.truth(m, data[m])
        truth[m] = {"raw": raw, "merged": labels.merge_short(raw)}

    inputs = level_inputs(LEVEL, matches, data, stats)
    names = inputs[matches[0]][0]
    rows = {m: training_rows(inputs[m][1], truth[m]["raw"]) for m in matches}
    print("features: {}; training rows per match ~{}".format(len(names), len(rows[matches[0]][1])), flush=True)

    # Out-of-sample probabilities and frame-level quality, per family.
    p, fold_models, quality = {f: {} for f in FAMILIES}, {f: {} for f in FAMILIES}, {}
    for family in FAMILIES:
        tp = fp = fn = 0
        for held_out in matches:
            X = np.concatenate([rows[m][0] for m in matches if m != held_out])
            y = np.concatenate([rows[m][1] for m in matches if m != held_out])
            model = dead_ball.fit(family, X, y, names)
            fold_models[family][held_out] = model
            p[family][held_out] = dead_ball.predict(model, inputs[held_out][1])
            gt = truth[held_out]["raw"]
            said, real = p[family][held_out] > 0.5, gt == labels.pm.D
            scored = gt != labels.pm.X
            tp += int((said & real & scored).sum())
            fp += int((said & ~real & scored).sum())
            fn += int((~said & real & scored).sum())
        quality[family] = {"frame_precision": tp / (tp + fp), "frame_recall": tp / (tp + fn),
                           "frame_f1": 2 * tp / (2 * tp + fp + fn)}
        print("{}: out-of-sample dead-ball frames at P > 0.5: precision {:.3f} recall {:.3f} F1 {:.3f}".format(
            family, *quality[family].values()), flush=True)

    # The grid.
    settings = list(itertools.product(FAMILIES, WEIGHTS, DEAD_PRICES, MIN_DEAD))
    payload = {"matches": matches, "p": p, "base": {m: inputs[m][2] for m in matches},
               "halves": {m: data[m]["halves"] for m in matches}, "truth": truth}
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(payload,)) as pool:
        grid = pd.DataFrame([r for result in pool.map(_score_setting, settings) for r in result])
    os.makedirs(RESULTS, exist_ok=True)
    grid.to_csv(os.path.join(RESULTS, "dead_ball_grid.csv"), index=False, float_format="%.4f")

    # v3 on the same matches, as the reference.
    v3_setting = (v3.SWITCH_TEAM, v3.SWITCH_DEAD, v3.MIN_POSSESSION, v3.MIN_DEAD)
    v3_letters = {m: decode(inputs[m][3], data[m]["halves"], v3_setting) for m in matches}
    v3_score = {m: score(truth[m]["raw"], v3_letters[m])["spell_f1"] for m in matches}

    # Leave one match out, per family.
    folds = {f: [] for f in FAMILIES}
    for family in FAMILIES:
        for held_out in matches:
            chosen = pick(summarise(grid, [m for m in matches if m != held_out], family))
            setting = tuple(chosen[SETTING])
            row = lookup(grid, setting, held_out)
            folds[family].append({"held_out": held_out, "setting": [setting[0]] + [float(v) for v in setting[1:]],
                                  "held_out_spell_f1": float(row["spell_f1"]), "v3_spell_f1": v3_score[held_out],
                                  "possessions": int(row["possessions"]), "true_possessions": int(row["true_possessions"])})
        held = np.array([f["held_out_spell_f1"] for f in folds[family]])
        print("{}: held-out spellF1 {:.3f} +- {:.3f}; v3 {:.3f}; better than v3 on {}/{}".format(
            family, held.mean(), held.std(ddof=1), np.mean(list(v3_score.values())),
            int((held > np.array([v3_score[m] for m in matches])).sum()), len(matches)), flush=True)
    held_mean = {f: float(np.mean([x["held_out_spell_f1"] for x in folds[f]])) for f in FAMILIES}
    family = "trees" if held_mean["trees"] > held_mean["logistic"] + FAMILY_MARGIN else "logistic"
    for fold in folds[family]:
        print("  fold {held_out}: {setting} -> {held_out_spell_f1:.3f} (v3 {v3_spell_f1:.3f}), "
              "possessions {possessions} / {true_possessions}".format(**fold))

    # The submission's setting, on all 10.
    chosen = pick(summarise(grid, matches, family))
    setting = tuple(chosen[SETTING])
    plain = summarise(grid, matches, family).sort_values("spell_f1", ascending=False, kind="stable").iloc[0]
    print("family {}; chosen on all {}: {} mean spellF1 {:.3f}, possessions {:.0f}; plain best {} {:.3f}".format(
        family, len(matches), setting, chosen["spell_f1"], chosen["possessions"],
        tuple(plain[SETTING]), plain["spell_f1"]), flush=True)

    # Checks: dead-ball report, per set, merged labels, other damage levels.
    chosen_cost = {m: v4_cost(inputs[m][2], p[family][m], setting[1]) for m in matches}
    v4_letters = {m: decode(chosen_cost[m], data[m]["halves"], (v3.SWITCH_TEAM, setting[2], v3.MIN_POSSESSION, setting[3]))
                  for m in matches}
    checks = {"dead_ball": {}, "per_set": {}, "merged_labels": {}, "levels": {}}
    for name in ("idsse", "metrica"):
        ms = [m for m in matches if sets.source(m) == name]
        checks["dead_ball"][name] = {
            "v3": summed_report([dead_ball_report(truth[m]["raw"], v3_letters[m]) for m in ms]),
            "v4": summed_report([dead_ball_report(truth[m]["raw"], v4_letters[m]) for m in ms])}
        checks["per_set"][name] = {"v4": float(np.mean([lookup(grid, setting, m)["spell_f1"] for m in ms])),
                                   "v3": float(np.mean([v3_score[m] for m in ms]))}
    checks["merged_labels"] = {
        "v4": float(np.mean([lookup(grid, setting, m, "merged")["spell_f1"] for m in matches])),
        "v3": float(np.mean([score(truth[m]["merged"], v3_letters[m])["spell_f1"] for m in matches]))}
    del inputs
    for level in CHECK_LEVELS:
        other = level_inputs(level, matches, data, stats)
        v4s, v3s = [], []
        for m in matches:
            _, X, base, v3cost = other[m]
            pm_ = dead_ball.predict(fold_models[family][m], X)
            letters = decode(v4_cost(base, pm_, setting[1]), data[m]["halves"],
                             (v3.SWITCH_TEAM, setting[2], v3.MIN_POSSESSION, setting[3]))
            v4s.append(score(truth[m]["raw"], letters)["spell_f1"])
            v3s.append(score(truth[m]["raw"], decode(v3cost, data[m]["halves"], v3_setting))["spell_f1"])
        checks["levels"][str(level)] = {"v4": float(np.mean(v4s)), "v3": float(np.mean(v3s))}
        del other
    print("checks: {}".format(json.dumps(checks, indent=1)), flush=True)

    # The model for the submission: chosen family, all 10 matches.
    final = dead_ball.fit(family, np.concatenate([rows[m][0] for m in matches]),
                          np.concatenate([rows[m][1] for m in matches]), names)
    path = dead_ball.save(final)
    choice = {"family": family, "weight": float(setting[1]), "dead_price": float(setting[2]),
              "min_dead": int(setting[3]), "bench_mean_spell_f1": float(chosen["spell_f1"]),
              "bench_mean_possessions": float(chosen["possessions"]),
              "plain_best": {"setting": [plain[k] if k == "family" else float(plain[k]) for k in SETTING],
                             "spell_f1": float(plain["spell_f1"])},
              "held_out_mean": held_mean, "classifier_quality": quality, "folds": folds,
              "v3_spell_f1": v3_score, "checks": checks, "model_file": os.path.relpath(path, ROOT),
              "features": names}
    with open(os.path.join(RESULTS, "dead_ball_choice.json"), "w") as handle:
        json.dump(choice, handle, indent=1)
    print("wrote bench/results/dead_ball_grid.csv, bench/results/dead_ball_choice.json and {}".format(
        os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
