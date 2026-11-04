"""The lab's mixer: chains of the kept candidates, judged as one version (docs/lab.md, "Mixing").

    python -m bench.lab_mix [--set all|idsse]    # ~15 min; writes bench/results/lab_mix[_idsse].*
    python -m bench.lab_mix --save               # the kept chain's out-of-fold letters, for the next round

A chain is an optional decode stage (v7's costs decoded again, then v9's snap and v11's move) and
post stages in pipeline order: stoppage -> short -> segments. Each stage scores a match with the
setting its own judging picked for that match (chosen without it). A stage that trains (short,
segments; stoppage's offset) is fitted on the other matches, given as input what the stages before
it wrote on them out of fold: their own models trained without that match (stacking). Each winner
is added only if the chain with it clears the bar against the same chain without it.
"""

import argparse
import ast
import json
import os

import numpy as np
import pandas as pd

from bench import chain as bench_chain
from bench import (lab, lab_dead, lab_decoder, lab_learned, lab_learned_check, lab_recall, lab_robust, lab_segments,
                   lab_short, lab_stoppage, lab_timing)

# Params
RESULTS = lab.RESULTS
# Each chain, and the chain it must beat (None: v11). Winners added in order of their own gain.
CHAINS = [
    (("short",), None),
    (("short", "segments"), ("short",)),
    (("decoder", "short", "segments"), ("short", "segments")),
    (("stoppage", "short", "segments"), ("short", "segments")),
    (("decoder", "stoppage", "short", "segments"), ("decoder", "short", "segments")),
    (("decoder", "short"), ("short",)),
    (("stoppage", "short"), ("short",)),
]
KEPT = ("stoppage", "short", "segments")       # round 1's mix (docs/lab.md): what round 2 builds on
# Round 2: stages built on the mix, each chain against the same chain without its last stage.
ROUND2 = [
    (KEPT + ("recall",), KEPT),
    (KEPT + ("timing",), KEPT),
    (KEPT + ("recall", "timing"), KEPT + ("recall",)),
    (KEPT + ("timing", "recall"), KEPT + ("timing",)),
]
# Round 2, with the dead-ball stage: after the kept round-2 order, or before it.
ROUND2_DEAD = [
    (KEPT + ("timing", "recall"), KEPT),
    (KEPT + ("timing", "recall", "dead"), KEPT + ("timing", "recall")),
    (KEPT + ("dead", "timing", "recall"), KEPT + ("timing", "recall")),
]
ROUNDS = {"1": CHAINS, "2": ROUND2, "2-dead": ROUND2_DEAD}
# Round 3: the learned per-frame model as the decoder's input (all of it, or only its team half),
# then each post stage in pipeline order, kept only if it beats the chain without it (greedy).
STARTS = ("learned", "learned_team", "robust")    # robust: learned with a ball standing still >= 1 s blanked (round 4)
GREEDY = ("stoppage", "short", "segments", "timing", "recall", "dead")
LETTERS = os.path.join(lab.ROOT, "data", "external", "lab_mix")
REPORTS = {"stoppage": "lab_stoppage", "short": "lab_short", "segments": "lab_segments", "decoder": "lab_decoder_round1",
           "recall": "lab_recall", "timing": "lab_timing", "dead": "lab_dead"}


def cache_short_evidence():
    """lab_short.evidence once per match: it reads only the match, never the letters."""
    original, seen = lab_short.evidence, {}

    def cached(match):
        if match.id not in seen:
            seen[match.id] = original(match)
        return seen[match.id]
    lab_short.evidence = cached


def literal(value):
    """The judge's picks, read back from JSON (some candidates stringified their columns)."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def fold_settings(which):
    """{stage: {match: setting}}: each stage's pick for each match, from its own judging."""
    out = {}
    split = pd.read_csv(os.path.join(RESULTS, "lab_learned_split_grid{}.csv".format(bench_chain.suffix(which))))
    split = split[split.part == "team_only"]
    report = lab.judge(split, ["switch_team", "switch_dead"], which)
    out["learned_team"] = {f["match"]: f["picked"] for f in report["folds"]}
    for stage, name in {**REPORTS, "learned": "lab_learned", "robust": "lab_robust"}.items():
        path = os.path.join(RESULTS, "{}{}.json".format(name, bench_chain.suffix(which)))
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            report = json.load(handle)
        out[stage] = {f["match"]: {k: literal(v) for k, v in f["picked"].items()} for f in report["folds"]}
    return out


def start(match, use_decoder, setting, which):
    """The letters before the post stages: v11, or the decoder's re-decoding + v9's snap + v11's move."""
    if not use_decoder:
        return match.v11.copy()
    return lab_decoder.finish(match, lab_decoder.decode(match, which=which, **setting), which)


def start_learned(match, first, setting, which):
    """The learned model's held-out P decoded (all of it, or v7's dead-ball evidence with only its team
    half), then v9's snap and v11's move."""
    if first == "learned":
        letters = lab_learned.decode_held_out(match, which, **setting)
    elif first == "robust":
        return lab_robust.finish(match, lab_robust.decode_held_out(match, which, **setting), which)
    else:
        p = lab_learned.held_out_proba(match, which, "all")
        cost = lab_learned_check.split_costs(match, p, "team_only")
        letters = lab_learned.decode_cost(match, cost, setting["switch_team"], setting["switch_dead"], 20)
    return lab_learned.finish(match, letters, which)


def run_chain(stages, matches, settings, which):
    """{match id: letters}, every match out of fold: each trained stage fitted without it. Post stages
    run in the order given."""
    use_decoder = "decoder" in stages
    post = [s for s in stages if s != "decoder"]
    if tuple(stages[:len(KEPT)]) == KEPT and os.path.exists(LETTERS + bench_chain.suffix(which)):
        letters = {m.id: kept_letters(m.id, which) for m in matches}     # the mix, saved by --save
        post = post[len(KEPT):]
    else:
        letters = {m.id: start(m, use_decoder, settings["decoder"][m.id], which) for m in matches}
    for stage in post:
        letters = add_stage(stage, letters, matches, settings)
        print("  {}: done".format(stage), flush=True)
    return letters


def add_stage(stage, previous, matches, settings):
    """One post stage on every match out of fold: fitted on the other matches' `previous` letters."""
    letters = dict(previous)
    for m in matches:
        others = [o for o in matches if o.id != m.id]
        setting = settings[stage][m.id]
        if stage == "stoppage":
            model = lab_stoppage.fit(others)
            letters[m.id] = lab_stoppage.apply(m, previous[m.id], setting["back"], setting["ahead"], model=model)
        elif stage == "short":
            model = lab_short.fit(others, letters=lambda o: previous[o.id], **setting)
            letters[m.id] = lab_short.apply(m, previous[m.id], model=model, **setting)
        elif stage == "segments":
            model = lab_segments.fit(others, lambda o: previous[o.id], features=setting["features"])
            letters[m.id] = lab_segments.apply(m, previous[m.id], model, **setting)
        elif stage == "recall":
            fitted = lab_recall.fit(others, letters_of=lambda o: previous[o.id], **setting)
            rest = {k: v for k, v in setting.items() if k != "model"}   # its "model" is the classifier's kind
            letters[m.id] = lab_recall.apply(m, previous[m.id], fitted, **rest)
        elif stage == "timing":
            model = lab_timing.fit(others, letters_of=lambda o: previous[o.id], **setting)
            letters[m.id] = lab_timing.apply(m, previous[m.id], model, **setting)
        elif stage == "dead":
            model = lab_dead.fit(others, letters_of=lambda o: previous[o.id], features=setting["features"])
            letters[m.id] = lab_dead.apply(m, previous[m.id], model, **setting)
        else:
            raise ValueError(stage)
    return letters


def scores_of(letters, matches):
    return {m.id: {**{k: v for k, v in lab.score(m, letters[m.id]).items() if k in ("spell_f1", "macro_f1", "possessions")},
                   "source": m.source, "unseen": m.unseen} for m in matches}


def greedy(first, matches, settings, which):
    """Round 3: start from `first`, then add each GREEDY stage only if it clears the bar against the
    chain so far. Returns [(chain, report against the chain before, report against v11)]."""
    letters = {m.id: start_learned(m, first, settings[first][m.id], which) for m in matches}
    current, scores = (first,), scores_of(letters, matches)
    steps = [(current, None, judge_scores(scores, None, which))]
    for stage in GREEDY:
        trial = add_stage(stage, letters, matches, settings)
        trial_scores = scores_of(trial, matches)
        report = judge_scores(trial_scores, scores, which)
        steps.append((current + (stage,), report, judge_scores(trial_scores, None, which)))
        print("{} + {}: {:+.4f} against the chain so far, better on {}/{}; passes: {}".format(
            "+".join(current), stage, report["all"]["held_out"] - report["all"]["base"], report["all"]["better"],
            report["all"]["matches"], report["passes"]), flush=True)
        if report["passes"]:
            current, letters, scores = current + (stage,), trial, trial_scores
    return current, letters, steps


def judge_scores(scores, base, which):
    """lab.judge on one chain's per-match scores, against `base` (scores) or v11."""
    grid = pd.DataFrame([{"match": m, "setting": "chain", **v} for m, v in scores.items()])
    return lab.judge(grid, ["setting"], which, base=base)


def kept_letters(m, which="all"):
    """The kept chain's out-of-fold letters on one bench match (after `--save`)."""
    return np.load(os.path.join(LETTERS + bench_chain.suffix(which), "{}.npy".format(m)))


def kept_base(which="all"):
    """The kept chain per match, in lab.baseline's format: pass it to lab.judge(..., base=)."""
    with open(os.path.join(RESULTS, "lab_mix_base{}.json".format(bench_chain.suffix(which)))) as handle:
        return json.load(handle)


def save(which, matches, settings):
    """Write the kept chain's out-of-fold letters and its scores per match."""
    folder = LETTERS + bench_chain.suffix(which)
    if os.path.exists(folder):
        os.rename(folder, folder + "_previous")       # so run_chain builds the chain from scratch
    letters = run_chain(KEPT, matches, settings, which)
    os.makedirs(folder, exist_ok=True)
    base = {}
    for m in matches:
        np.save(os.path.join(folder, "{}.npy".format(m.id)), letters[m.id])
        s = lab.score(m, letters[m.id])
        base[m.id] = {"spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "source": m.source, "unseen": m.unseen}
    with open(os.path.join(RESULTS, "lab_mix_base{}.json".format(bench_chain.suffix(which))), "w") as handle:
        json.dump(base, handle, indent=1)
    print("kept chain {}: mean {:.4f}; letters in {}".format(
        "+".join(KEPT), np.mean([b["spell_f1"] for b in base.values()]), os.path.relpath(folder, lab.ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--save", action="store_true", help="only the kept chain: save its letters")
    parser.add_argument("--round", choices=tuple(ROUNDS) + ("3",), default="1", help="which round's chains to judge")
    parser.add_argument("--starts", nargs="+", choices=STARTS, default=list(STARTS), help="round 3's starting points")
    args = parser.parse_args()
    which = args.which
    settings = fold_settings(which)
    cache_short_evidence()
    matches = list(lab.each(which))
    if args.save:
        return save(which, matches, settings)
    if args.round == "3":
        out = {}
        for first in args.starts:
            print("\nstarting from {}:".format(first), flush=True)
            chain_kept, letters, steps = greedy(first, matches, settings, which)
            for stages, against, v11_report in steps:
                print("  {}: {:.4f}, against v11 {:+.4f} (better on {}/{})".format(
                    "+".join(stages), v11_report["all"]["held_out"], v11_report["all"]["held_out"] - v11_report["all"]["base"],
                    v11_report["all"]["better"], v11_report["all"]["matches"]))
            print("kept: {}\n{}".format(" -> ".join(chain_kept), lab.describe(judge_scores(scores_of(letters, matches), None, which))))
            out[first] = {"kept": chain_kept, "steps": [{"chain": "+".join(s), "against_previous": a, "against_v11": v}
                                                         for s, a, v in steps]}
        name = "lab_mix_round3" + ("" if args.starts == list(STARTS) else "_" + "_".join(args.starts))
        with open(os.path.join(RESULTS, "{}{}.json".format(name, bench_chain.suffix(which))), "w") as handle:
            json.dump(out, handle, indent=1, default=str)
        return
    base = lab.baseline(which)
    chains = ROUNDS[args.round]
    scores, rows = {}, []
    if args.round != "1":
        scores[KEPT] = kept_base(which)
    for stages, _ in chains:
        name = "+".join(stages)
        print("{}:".format(name), flush=True)
        letters = run_chain(stages, matches, settings, which)
        scores[stages] = {}
        for m in matches:
            s = lab.score(m, letters[m.id])
            scores[stages][m.id] = {"spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "source": m.source,
                                    "unseen": m.unseen, "possessions": s["possessions"],
                                    "true_possessions": s["true_possessions"]}
            rows.append({"chain": name, "match": m.id, **scores[stages][m.id]})
    grid = pd.DataFrame(rows)
    reports = {}
    for stages, against in chains:
        name = "+".join(stages)
        one = grid[grid.chain == name].assign(setting=name)
        reference = scores[against] if against else base
        report = lab.judge(one, ["setting"], which, base=reference)
        report["against"] = "+".join(against) if against else "v11"
        report["possessions"] = float(one.possessions.mean())
        reports[name] = report
        print("\n{} (against {}), {:.0f} possessions per match (true {:.0f}):\n{}".format(
            name, report["against"], report["possessions"], one.true_possessions.mean(), lab.describe(report)))
        v11_report = lab.judge(one, ["setting"], which)
        print("  against v11: {:+.4f}, better on {}/{}".format(
            v11_report["all"]["held_out"] - v11_report["all"]["base"], v11_report["all"]["better"], v11_report["all"]["matches"]))
    tag = bench_chain.suffix(which)
    name = "lab_mix" if args.round == "1" else "lab_mix_round" + args.round.replace("-", "_")
    grid.to_csv(os.path.join(RESULTS, "{}_grid{}.csv".format(name, tag)), index=False, float_format="%.5f")
    with open(os.path.join(RESULTS, "{}{}.json".format(name, tag)), "w") as handle:
        json.dump(reports, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
