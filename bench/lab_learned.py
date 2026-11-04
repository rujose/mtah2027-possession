"""Lab candidate `learned` (idea 10c): a learned per-frame P(H / A / D) feeding the decoder (docs/lab/learned.md).

    python -m bench.lab_learned [--set all|idsse]   # ~18 min (~13 with idsse); after `python -m bench.lab --build`;
                                                    # writes bench/results/lab_learned[_idsse]*

One classifier (scikit-learn's gradient-boosted trees) in place of v1's hand weights: P(H / A / D)
per frame from every signal a match offers at prediction time (possession/lab_learned_features.py),
trained on the other bench matches only (leave one match out: the held-out match is never in its own
training). Its -log P, optionally blended with the earlier decoder costs, is decoded with each
decoder setting in GRID; then v9's restart snap and v11's touch move, exactly as bench/lab.py `build`
does; then scored. The grid was fixed before any result was seen and is judged as one (lab.judge).

For mixing: `fit(matches, **setting)` -> model, `decode(match, model, **setting)` -> letters at v7's
level (before the snap and the move), `finish(match, letters)` -> letters at v11's level. Without
refitting: `decode_held_out(match)` decodes the held-out P that `main` saved (CHOSEN by default).
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from bench import chain, lab, sets
from bench.tune_decoder import decode as decode_costs
from possession import lab_learned_features as llf
from possession import v9, v11
from possession.data import ROOT

# Params (fixed before the run; the page gives the why of each)
STAGE = "decode"
CACHE = os.path.join(ROOT, "data", "external", "lab_learned")    # gitignored (data/*)
EVERY = 10              # train on every 10th graded frame (0.4 s apart): neighbours are near-copies
# Learn only from IDSSE's matches (decision 3, ideas.md row 2): Metrica's samples blank the ball in
# dead time (on the bench its ball is missing on 94-99.5 % of dead frames, 9-14 % of live ones), so a
# model fitted on them would learn "no ball = dead", too strong for our match. IDSSE's damaged ball
# (missing on 51-58 % of dead frames, 27-34 % of live ones) was built to carry the hint ours does, a
# little weaker (bench.md). Metrica's matches are still predicted and scored.
TRAIN_SOURCES = ("idsse",)
CLIP = (0.01, 0.99)     # as v4's dead-ball model: one frame can't cost more than -log(0.01) = 4.6
DECIMALS = 3            # P rounded before it becomes a cost, as v4
TREES = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 200,
         "l2_regularization": 1.0, "early_stopping": False, "random_state": 0}
VARIANTS = ("all", "no_pathcrf")         # every input / nothing PathCRF touched
BLENDS = (0.0, 0.5)                      # share of the earlier costs (v7's; v6's without PathCRF)
SWITCH_TEAM = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SWITCH_DEAD = (50.0, 25.0, 10.0)         # v4's 50 first
MIN_POSSESSION = (20, 10)                # frames; v3's 20 first
MIN_DEAD = 1                             # v4's
COLUMNS = ["variant", "blend", "switch_team", "switch_dead", "min_possession"]
GRID = [dict(zip(COLUMNS, s)) for s in itertools.product(VARIANTS, BLENDS, SWITCH_TEAM, SWITCH_DEAD, MIN_POSSESSION)]
# Chosen on all 10 by `python -m bench.lab_learned` (bench/results/lab_learned.json). 5 of 10 folds
# picked exactly this; all 10 the same variant, blend and team price (dead price 50 or 10: in sample
# 0.6633 against 0.6631).
CHOSEN = {"variant": "all", "blend": 0.0, "switch_team": 8.0, "switch_dead": 50.0, "min_possession": 20}
GROUPS = {   # for the permutation importances: feature name -> family
    "ball": lambda k: k.startswith("ball_") or k in ("since_ball_seen", "until_ball_seen"),
    "players at the ball": lambda k: (k.startswith(("d_home", "d_away", "d_diff", "n_", "on_"))
                                      and "_last_seen" not in k and "_next_seen" not in k),
    "players at the last / next seen ball": lambda k: "_last_seen" in k or "_next_seen" in k,
    "teams' shape and speed": lambda k: k.startswith(("speed_", "spread_", "up_")),
    "touches": lambda k: "touch" in k,
    "passes": lambda k: "pass" in k,
    "P(dead)": lambda k: k == "p_dead",
    "restart": lambda k: k in ("p_start", "restart_clue"),
    "PathCRF": lambda k: k.startswith("p_pathcrf"),
    "v6 costs": lambda k: k.startswith(("v6_cost", "c6_")),
    "v7 costs": lambda k: k.startswith(("v7_cost", "c7_")),
}
SWAP = np.array([1, 0, 2])      # H <-> A on class columns and codes


def folder(which):
    return CACHE + chain.suffix(which)


def evidence(match, which="all"):
    """The match's seen passes (v9) and touches (v11, v11's ΔV), computed once and kept on it."""
    if getattr(match, "_learned_evidence", None) is None:
        match._learned_evidence = {"passes": v9.seen_passes(match.track),
                                   "touches": v11.seen_touches(match.track, v11.SETTINGS[which]["dv"])}
    return match._learned_evidence


def cached(m, which="all", match=None):
    """{names, X, X_swapped, truth} for match id m, from disk or computed (and saved) once."""
    path = os.path.join(folder(which), "{}.npz".format(m))
    if not os.path.exists(path):
        match = match or lab.load(m, which)
        ev = evidence(match, which)
        names, X = llf.frame_features(match.track, match.halves, match.pitch, match.pred, ev["touches"], ev["passes"])
        _, Xs = llf.frame_features(match.track, match.halves, match.pitch, match.pred, ev["touches"], ev["passes"], swap=True)
        os.makedirs(folder(which), exist_ok=True)
        np.savez(path, names=np.array(names), X=X, X_swapped=Xs, truth=match.truth.astype(np.int8))
    with np.load(path) as z:
        return {"names": [str(k) for k in z["names"]], "X": z["X"], "X_swapped": z["X_swapped"], "truth": z["truth"]}


def fit(matches, variant="all", which="all", sources=TRAIN_SOURCES, **_):
    """The classifier, trained on these matches only (ids or lab matches; of them, those from
    `sources`): every EVERY-th graded frame, each also with the teams exchanged (labels H <-> A)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs, ys, names = [], [], None
    used = [getattr(m, "id", m) for m in matches if sets.source(getattr(m, "id", m)) in sources]
    for m in used:
        c = cached(m, which)
        names = c["names"]
        cols = llf.columns(names, variant)
        rows = np.flatnonzero(np.isin(c["truth"], (0, 1, 2)))[::EVERY]
        y = c["truth"][rows].astype(np.int64)
        Xs += [c["X"][rows][:, cols], c["X_swapped"][rows][:, cols]]
        ys += [y, SWAP[y]]
        del c
    X, y = np.concatenate(Xs), np.concatenate(ys)
    clf = HistGradientBoostingClassifier(**TREES).fit(X, y)
    return {"variant": variant, "which": which, "names": [names[i] for i in cols], "cols": cols, "clf": clf,
            "rows": len(y), "trained_on": used, "_p": {}}


def proba(match, model, symmetric=True):
    """P(H, A, D) per frame [frames, 3]: the mean of the match as it is and with the teams exchanged,
    clipped and rounded as v4's."""
    key = (getattr(match, "id", match), symmetric)
    if key not in model["_p"]:
        from threadpoolctl import threadpool_limits
        c = cached(key[0], model["which"])
        with threadpool_limits(2):
            p = model["clf"].predict_proba(c["X"][:, model["cols"]])
            if symmetric:
                p = (p + model["clf"].predict_proba(c["X_swapped"][:, model["cols"]])[:, SWAP]) / 2
        model["_p"][key] = np.round(np.clip(p, *CLIP), DECIMALS)
    return model["_p"][key]


def held_out_proba(match, which="all", variant="all"):
    """The P(H, A, D) `main` saved for this match, from the model trained without it (so mixing
    needn't refit): data/external/lab_learned[_idsse]/p_<variant>_<match>.npy."""
    return np.load(os.path.join(folder(which), "p_{}_{}.npy".format(variant, getattr(match, "id", match)))).astype(np.float64)


def cost_from(match, p, variant="all", blend=0.0):
    """The decoder's cost from P: -log P, blended with the earlier costs (v7's; v6's for "no_pathcrf")."""
    cost = -np.log(p)
    if blend:
        base = match.pred["v6_cost" if variant == "no_pathcrf" else "v7_cost"].astype(np.float64)
        cost = (1.0 - blend) * cost + blend * base
    return cost


def costs(match, model, blend=0.0):
    return cost_from(match, proba(match, model), model["variant"], blend)


def decode_cost(match, cost, switch_team=CHOSEN["switch_team"], switch_dead=CHOSEN["switch_dead"],
                min_possession=CHOSEN["min_possession"], **_):
    return decode_costs(cost, match.halves, (switch_team, switch_dead, int(min_possession), MIN_DEAD))


def decode(match, model, blend=0.0, **setting):
    """Letters at v7's level (before v9's snap and v11's move); settings default to CHOSEN."""
    return decode_cost(match, costs(match, model, blend), **setting)


def decode_held_out(match, which="all", **setting):
    """`decode` from the saved held-out P (no refit), settings default to CHOSEN."""
    setting = {**CHOSEN, **setting}
    p = held_out_proba(match, which, setting["variant"])
    return decode_cost(match, cost_from(match, p, setting["variant"], setting["blend"]), **setting)


def finish(match, letters, which="all"):
    """v9's restart snap, then v11's touch move, with the harness's settings (bench/lab.py `build`)."""
    ev = evidence(match, which)
    back, ahead = chain.restart_snap(which)
    s = v11.SETTINGS[which]
    snapped, _ = v9.snap(np.asarray(letters), match.track, match.halves, back, ahead, starts=ev["passes"])
    moved, _ = v11.move(snapped, match.track, match.halves, s["dv"], s["back"], s["ahead"], found=ev["touches"])
    return moved


def importances(match, model, which):
    """Rise in held-out log loss when one family of features is shuffled (every 25th graded frame)."""
    c = cached(match.id, which)
    rows = np.flatnonzero(np.isin(c["truth"], (0, 1, 2)))[::25]
    X, y = c["X"][rows][:, model["cols"]], c["truth"][rows].astype(np.int64)

    def loss(Z):
        p = np.clip(model["clf"].predict_proba(Z), 1e-6, 1)
        return float(-np.log(p[np.arange(len(y)), y]).mean())

    base, rng, out = loss(X), np.random.default_rng(0), {}
    for group, test in GROUPS.items():
        idx = [i for i, k in enumerate(model["names"]) if test(k)]
        if not idx:
            continue
        Z = X.copy()
        Z[:, idx] = Z[rng.permutation(len(Z))][:, idx]
        out[group] = loss(Z) - base
    return base, out


def frame_accuracy(match, letters):
    graded = np.isin(match.truth, (0, 1, 2))
    return float((np.asarray(letters)[graded] == match.truth_letters[graded]).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=chain.SETS, default="all", dest="which")
    parser.add_argument("--train-all-sources", action="store_true",
                        help="for information only: let the classifier learn from Metrica's matches too")
    args = parser.parse_args()
    which, sources = args.which, (("idsse", "metrica") if args.train_all_sources else TRAIN_SOURCES)
    name = "learned_trainall" if args.train_all_sources else "learned"
    tag = chain.suffix(which)
    ms = lab.matches(which)
    for m in ms:       # the features, once per match
        if not os.path.exists(os.path.join(folder(which), "{}.npz".format(m))):
            started = time.time()
            cached(m, which)
            print("{}: features cached ({:.0f} s)".format(m, time.time() - started), flush=True)
    rows, extra = [], []
    for held in ms:
        match = lab.load(held, which)
        others = [m for m in ms if m != held]
        for variant in VARIANTS:
            started = time.time()
            model = fit(others, variant=variant, which=which, sources=sources)
            p = proba(match, model)
            np.save(os.path.join(folder(which), "p_{}_{}{}.npy".format(variant, held, "_trainall" if args.train_all_sources else "")),
                    p.astype(np.float32))      # the held-out P, for later checks
            base, imp = importances(match, model, which)
            letters_argmax = np.array(["H", "A", "D"])[p.argmax(axis=1)]
            extra.append({"match": held, "variant": variant, "train_rows": model["rows"], "features": len(model["cols"]),
                          "trained_on": " ".join(model["trained_on"]),
                          "frame_acc_argmax": frame_accuracy(match, letters_argmax),
                          "frame_acc_v7": frame_accuracy(match, match.v7), "frame_acc_v11": frame_accuracy(match, match.v11),
                          "log_loss": base, **{"imp " + k: v for k, v in imp.items()}})
            for setting in (s for s in GRID if s["variant"] == variant):
                at_v7 = decode(match, model, **setting)
                final = finish(match, at_v7, which)
                s = lab.score(match, final)
                rows.append({"match": held, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                             "possessions": s["possessions"], "true_possessions": s["true_possessions"],
                             "d_f1": s["d_f1"], "team_acc_alive": s["team_acc_alive"],
                             "spell_f1_decoded": lab.score(match, at_v7)["spell_f1"],
                             "frame_acc": frame_accuracy(match, final), "frame_acc_decoded": frame_accuracy(match, at_v7)})
            print("{} {}: fit on {} rows x {} features, {:.0f} s; frame accuracy argmax {:.3f} (v7 {:.3f}); "
                  "best spellF1 here {:.4f} (v11 {:.4f})".format(
                      held, variant, model["rows"], len(model["cols"]), time.time() - started,
                      extra[-1]["frame_acc_argmax"], extra[-1]["frame_acc_v7"],
                      max(r["spell_f1"] for r in rows if r["match"] == held and r["variant"] == variant),
                      lab.score(match, match.v11)["spell_f1"]), flush=True)
            del model
        del match
    grid = pd.DataFrame(rows)
    report = lab.judge(grid, COLUMNS, which)
    print(lab.describe(report))
    lab.save(name, grid, report, which)
    pd.DataFrame(extra).to_csv(os.path.join(lab.RESULTS, "lab_{}_frames{}.csv".format(name, tag)), index=False, float_format="%.5f")
    # Each variant judged on its own, for the page (the bar is judged on the whole grid above).
    for variant in VARIANTS:
        sub = grid[grid["variant"] == variant].reset_index(drop=True)
        print("[{} only] ".format(variant) + lab.describe(lab.judge(sub, COLUMNS, which)).replace("\n", "\n  "))
    print("wrote bench/results/lab_{0}_grid{1}.csv, lab_{0}{1}.json, lab_{0}_frames{1}.csv".format(name, tag))


if __name__ == "__main__":
    main()
