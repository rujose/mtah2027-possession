"""Lab round 2 `polish`: round 1's mix with each stage's settings chosen again inside the chain (docs/lab/polish.md).

    python -m bench.lab_polish [--set all|idsse]          # ~20 min: the search, judged against the mix, both nested checks
    python -m bench.lab_polish --save [--set all|idsse]   # ~2 min, after the run: the polished chain's out-of-fold letters

Round 1's mix (bench/lab_mix.py) is stoppage -> short -> segments, each stage on the setting it picked
on v11's letters. Here each stage's settings are chosen again with the stages before it in place, and
two stages are tried again at the end (stoppage2: the stoppage rule on segments' new team -> D
changes; short2: short possessions in the runs segments cleaned). The grids, the search, the number of
passes and the verdict rule below were fixed before any run.

The chain is lab_mix.run_chain's: every trained stage on a match (short's regression, segments'
classifiers, stoppage's offset) is fitted on the bench's other matches, given as input what the
stages before it wrote there out of fold. A configuration is one value per stage; `Chain` scores any
configuration on every match of a bench, sharing each stage's work between configurations that agree
up to it.

The search is coordinate-wise: from the mix, each stage in pipeline order tries every value of its
grid with the others at their current values and keeps the best mean spellF1 (ties: the current
value, then the first in grid order); a second pass if the first changed anything, then stop.

Three ways of scoring it against the mix, each match on settings chosen without it:
1. judge: lab.judge over every configuration the search on all the matches scored;
2. nested selection: for each match, the whole search rerun with its score left out of every mean;
3. fully nested: for each match, the whole search rerun on a bench without it, where every trained
   stage, the stoppage offset and round 1's per-match segments setting are also fitted without it.
Kept only if 1 and 3 both clear lab.BAR against the mix (and on --set idsse too).

What a stage reads to predict a match: its damaged tracking, the out-of-sample outputs on it and the
letters before it. Truth: for training on the other matches and for scoring only.
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd

from bench import chain as bench_chain
from bench import lab, lab_mix, lab_segments, lab_short, lab_stoppage

# Params (fixed before any run; docs/lab/polish.md gives the why of each)
NAME = "polish"
STAGES = ("stoppage", "short", "segments", "stoppage2", "short2")
GRIDS = {
    # (s back / s ahead) to look for the exit; round 1's 4/2 was its widest and still rising
    "stoppage": ["4/2", "6/2", "8/2", "4/3", "6/3", "8/3"],
    # the learned rule's p_min (round 1's CHOSEN otherwise); 0.5 was its lowest and still rising
    "short": ["0.5", "0.4", "0.3"],
    # round 1's per-match pick (on v11's letters), then features / flip / merge, mode "dead" (it won
    # every comparable pair in round 1), around round 1's chosen all / 0.6 / 0.7
    "segments": ["round1"] + ["{} {} {}".format(f, flip, merge) for f in ("all", "no_pathcrf")
                              for flip in ("0.6", "0.7", "0.8") for merge in ("0.6", "0.7", "0.8", "0.9")],
    # the stoppage rule again after segments, with the stoppage stage's window
    "stoppage2": ["off", "on"],
    # the learned short rule again after segments (and stoppage2), fitted on their output; cautious first
    "short2": ["off", "0.8", "0.65", "0.5"],
}
CURRENT = {"stoppage": "4/2", "short": "0.5", "segments": "round1", "stoppage2": "off", "short2": "off"}   # the mix
PASSES = 2
MODE = "dead"
TAGS = np.array(lab.LETTERS)       # letters stored as int8 codes into this (H, A, D, X)
LETTERS = os.path.join(lab.ROOT, "data", "external", "lab_polish")
# The search's result on all the matches of each set (bench/results/lab_polish[_idsse].json, "polished"):
# what a version for our match would run, every trained stage fitted on the whole bench.
POLISHED = {
    "all": {"stoppage": "6/3", "short": "0.5", "segments": "all 0.6 0.7", "stoppage2": "on", "short2": "0.65"},
    "idsse": {"stoppage": "6/2", "short": "0.4", "segments": "no_pathcrf 0.8 0.6", "stoppage2": "on", "short2": "0.5"},
}


# --------------------------------------------------------------------------------------------
# The matches, loaded once
# --------------------------------------------------------------------------------------------

def load_all(which):
    """{id: match} with what every stage reads precomputed (short's evidence, segments' signals,
    the truth's exit lags for stoppage's offset); the players' tracks are dropped once those are in
    (~50 MB a match), so any later read of them fails loudly."""
    out, ev, lags = {}, {}, {}
    for m in lab.each(which):
        ev[m.id] = lab_short.evidence(m)
        lab_segments.signals(m)                     # cached on the match object
        lags[m.id] = lab_stoppage.lags(m)           # the truth: only for fitting other matches' offset
        m.track = {"ball": m.track["ball"], "team": m.track["team"]}
        out[m.id] = m
        gc.collect()
    return out, ev, lags


def encode(letters):
    letters = np.asarray(letters).astype(str)
    codes = np.full(len(letters), -1, np.int8)
    for i, t in enumerate(TAGS):
        codes[letters == t] = i
    assert (codes >= 0).all()
    return codes


def decode(codes):
    return TAGS[codes]


def round1_segments(which, exclude=()):
    """Round 1's per-match segments setting (lab.judge's pick over bench/results/lab_segments_grid*.csv),
    with the matches in `exclude` left out of every choice. Without exclusions it is
    lab_mix.fold_settings(which)["segments"] (checked)."""
    grid = pd.read_csv(os.path.join(lab.RESULTS, "lab_segments_grid{}.csv".format(bench_chain.suffix(which))))
    cols = lab_segments.COLUMNS
    grid = grid[~grid.match.isin(exclude)].reset_index(drop=True)
    grid["_order"] = np.arange(len(grid))
    settings = grid.drop_duplicates(cols)[cols + ["_order"]]
    out = {}
    for m in sorted(grid.match.unique()):
        others = grid[grid.match != m].groupby(cols, sort=False).spell_f1.mean().reset_index()
        others = others.merge(settings, on=cols).sort_values("_order", kind="stable")
        best = others.loc[others["spell_f1"].round(6).idxmax()]
        out[m] = {c: (best[c].item() if hasattr(best[c], "item") else best[c]) for c in cols}
    return out


def window(label):
    back, ahead = label.split("/")
    return float(back), float(ahead)


def short_setting(label):
    return dict(lab_short.CHOSEN, p_min=float(label))


def segments_setting(label):
    features, flip, merge = label.split()
    return {"features": features, "flip": float(flip), "merge": float(merge), "mode": MODE}


def describe_setting(config):
    """{stage: the setting each stage's code takes}, for rebuilding the chain."""
    out = {}
    back, ahead = window(config["stoppage"])
    out["stoppage"] = {"back": back, "ahead": ahead}
    out["short"] = short_setting(config["short"])
    out["segments"] = "round 1's per-match pick" if config["segments"] == "round1" else segments_setting(config["segments"])
    out["stoppage2"] = None if config["stoppage2"] == "off" else {"back": back, "ahead": ahead}
    out["short2"] = None if config["short2"] == "off" else short_setting(config["short2"])
    return out


# --------------------------------------------------------------------------------------------
# The chain on a bench
# --------------------------------------------------------------------------------------------

class Chain:
    """Any configuration of the chain on the bench `ids`: each match's trained stages fitted on the
    bench's other matches, given their out-of-fold letters from the stages before (lab_mix.run_chain).
    A prefix is the tuple of the first k stages' values; its letters, short's training rows and
    segments' tables are cached per prefix, so configurations that agree up to a stage share it."""

    def __init__(self, matches, ev, lags, ids, round1):
        self.m, self.ev, self.ids, self.round1 = matches, ev, list(ids), round1
        self.offset = {j: int(round(float(np.median(np.concatenate([lags[o] for o in self.ids if o != j])))))
                       for j in self.ids}                                     # lab_stoppage.fit, without j
        self._letters, self._rows, self._short, self._tables, self._probs, self._scores = {}, {}, {}, {}, {}, {}

    def others(self, j):
        return [o for o in self.ids if o != j]

    def letters(self, prefix, keep=True):
        """{id: codes} after the stages of `prefix`."""
        if prefix in self._letters:
            return self._letters[prefix]
        if not prefix:
            out = {j: encode(self.m[j].v11) for j in self.ids}
        else:
            parent = self.letters(prefix[:-1])
            stage, label = STAGES[len(prefix) - 1], prefix[-1]
            if label == "off":
                out = parent
            else:
                out = {j: encode(self.stage(stage, label, prefix, j, parent)) for j in self.ids}
        if keep:
            self._letters[prefix] = out
        return out

    def stage(self, stage, label, prefix, j, parent):
        match, before = self.m[j], decode(parent[j])
        if stage in ("stoppage", "stoppage2"):
            back, ahead = window(label if stage == "stoppage" else prefix[0])
            return lab_stoppage.apply(match, before, back, ahead, model={"offset": self.offset[j]})
        if stage in ("short", "short2"):
            model = self.short_model(prefix[:-1], j, parent)
            return lab_short.apply(match, before, model=model, ev=self.ev[j], **short_setting(label))
        if stage == "segments":
            setting = self.round1[j] if label == "round1" else segments_setting(label)
            tab, _ = self.table(prefix[:-1], j, parent)
            p_wrong, p_spurious = self.seg_probs(prefix[:-1], setting["features"], j, parent)
            return lab_segments.act(before, tab, p_wrong, p_spurious, setting["flip"], setting["merge"], setting["mode"])[0]
        raise ValueError(stage)

    def short_model(self, prefix, j, parent):
        """lab_short.fit on the bench's other matches, on their letters after `prefix`."""
        key = (prefix, j)
        if key not in self._short:
            c = lab_short.CHOSEN
            for o in self.others(j):
                if (prefix, o) not in self._rows:
                    self._rows[(prefix, o)] = lab_short.training_rows(self.m[o], decode(parent[o]), c["length"], c["dead"], ev=self.ev[o])
            self._short[key] = lab_short.fit_rows([self._rows[(prefix, o)] for o in self.others(j)])
        return self._short[key]

    def table(self, prefix, o, parent):
        """(segments' table of o's possessions after `prefix`, the truth's answers: for training only)."""
        key = (prefix, o)
        if key not in self._tables:
            tab = lab_segments.table(self.m[o], decode(parent[o]))
            self._tables[key] = (tab, lab_segments.targets(self.m[o], tab))
        return self._tables[key]

    def seg_probs(self, prefix, features, j, parent):
        """(P(wrong), P(spurious)) on j's possessions, from lab_segments.fit on the other matches."""
        key = (prefix, features, j)
        if key not in self._probs:
            train = [pd.concat(self.table(prefix, o, parent), axis=1) for o in self.others(j)]
            model = lab_segments.train(train, features)
            self._probs[key] = lab_segments.probabilities(self.table(prefix, j, parent)[0], model)
        return self._probs[key]

    def scores(self, config):
        """{id: spell_f1, macro_f1, possessions, true_possessions} for a configuration."""
        key = tuple(config[s] for s in STAGES)
        if key not in self._scores:
            final = self.letters(key[:-1])
            last = self.letters(key, keep=False) if key[-1] != "off" else final
            out = {}
            for j in self.ids:
                s = lab.score(self.m[j], decode(last[j]))
                out[j] = {k: s[k] for k in ("spell_f1", "macro_f1", "possessions", "true_possessions")}
            self._scores[key] = out
        return self._scores[key]

    def visited(self):
        return list(self._scores)


def search(chain, select, log=None):
    """The coordinate search, each mean over `select` (ids of the bench). (config, trace)."""
    config, trace = dict(CURRENT), []
    for p in range(PASSES):
        changed = False
        for stage in STAGES:
            means = {}
            for value in GRIDS[stage]:
                s = chain.scores(dict(config, **{stage: value}))
                means[value] = round(float(np.mean([s[j]["spell_f1"] for j in select])), 6)
            best = max(means.values())
            pick = config[stage] if means[config[stage]] == best else next(v for v in GRIDS[stage] if means[v] == best)
            trace.append({"pass": p + 1, "stage": stage, "means": means, "from": config[stage], "pick": pick})
            if log:
                log("  pass {} {}: {} -> {} ({})".format(p + 1, stage, config[stage], pick, ", ".join(
                    "{} {:.4f}".format(v, means[v]) for v in GRIDS[stage])))
            changed |= pick != config[stage]
            config[stage] = pick
        if not changed:
            break
    return config, trace


# --------------------------------------------------------------------------------------------
# Scoring against the mix
# --------------------------------------------------------------------------------------------

def bar(folds):
    """lab.judge's summary and BAR checks for folds already scored (one held-out row per match)."""
    f = pd.DataFrame(folds)

    def means(rows):
        return {"held_out": float(rows.held_out.mean()), "base": float(rows.base.mean()),
                "better": int((rows.held_out > rows.base).sum()), "matches": len(rows)} if len(rows) else None

    by_source = {s: means(f[f.source == s]) for s in sorted(f.source.unique())}
    unseen = means(f[f.unseen])
    report = {"folds": folds, "all": means(f),
              "macro": {"held_out": float(f.macro_held_out.mean()), "base": float(f.macro_base.mean())},
              "by_source": by_source, "unseen": unseen,
              "chosen": {" | ".join("{}={}".format(k, v) for k, v in t): int(n) for t, n in
                         pd.Series([tuple(sorted(x["picked"].items())) for x in folds]).value_counts().head(3).items()},
              "possessions": float(f.possessions.mean()), "true_possessions": float(f.true_possessions.mean())}
    checks = {"mean": report["all"]["held_out"] > report["all"]["base"],
              "share": report["all"]["better"] >= int(np.ceil(lab.BAR["better_share"] * len(f) - 1e-9))}
    if lab.BAR["each_source"]:
        checks["each_source"] = all(v["held_out"] >= v["base"] for v in by_source.values())
    if lab.BAR["unseen"] and unseen:
        checks["unseen"] = unseen["held_out"] >= unseen["base"]
    report["checks"], report["passes"] = checks, all(checks.values())
    return report


def fold(m, config, outer, base):
    s = outer.scores(config)[m]
    return {"match": m, "source": base[m]["source"], "unseen": base[m]["unseen"], "picked": dict(config),
            "held_out": s["spell_f1"], "base": base[m]["spell_f1"], "macro_held_out": s["macro_f1"],
            "macro_base": base[m]["macro_f1"], "possessions": s["possessions"], "true_possessions": s["true_possessions"]}


def grid_of(chain):
    """One row per (match, configuration scored), the mix first (lab.judge's ties keep it)."""
    rows = []
    for key in chain.visited():
        for j, s in chain._scores[key].items():
            rows.append({"match": j, **dict(zip(STAGES, key)), **s})
    return pd.DataFrame(rows)


def stage_counts(chain, config):
    """Possessions per match (scorer's count) after each stage of `config`, on the bench."""
    key = tuple(config[s] for s in STAGES)
    out = {}
    for k in range(len(STAGES) + 1):
        letters = chain.letters(key[:k], keep=k < len(STAGES))
        name = "v11" if k == 0 else STAGES[k - 1]
        out[name] = float(np.mean([lab.score(chain.m[j], decode(letters[j]))["possessions"] for j in chain.ids]))
    return out


def run(which):
    started = time.time()

    def log(text):
        print("[{:5.0f} s] {}".format(time.time() - started, text), flush=True)

    matches, ev, lags = load_all(which)
    ids = list(matches)
    base = lab_mix.kept_base(which)
    round1 = round1_segments(which)
    assert round1 == lab_mix.fold_settings(which)["segments"], "round 1's segments picks don't reproduce"
    log("{} matches loaded".format(len(ids)))

    outer = Chain(matches, ev, lags, ids, round1)
    mix = outer.scores(CURRENT)
    gap = max(abs(mix[j]["spell_f1"] - base[j]["spell_f1"]) for j in ids)
    same = all(np.array_equal(decode(outer.letters(tuple(CURRENT[s] for s in STAGES[:3]))[j]), lab_mix.kept_letters(j, which))
               for j in ids)
    log("the mix rebuilt: largest spellF1 gap to lab_mix {:.2e}; letters identical: {}".format(gap, same))
    assert gap < 1e-9 and same, "the chain doesn't reproduce round 1's mix"

    # 1. The search on all the matches, and lab.judge over every configuration it scored
    log("search on all {} matches:".format(len(ids)))
    polished, trace = search(outer, ids, log)
    grid = grid_of(outer)
    log("{} configurations scored; polished: {}".format(grid.groupby(list(STAGES)).ngroups, polished))
    report = lab.judge(grid, list(STAGES), which, base=base)
    for f in report["folds"]:
        s = outer.scores(f["picked"])[f["match"]]
        f["possessions"], f["true_possessions"] = s["possessions"], s["true_possessions"]
    print("\njudge, against the mix:\n" + lab.describe(report).replace("vs v11", "vs the mix"), flush=True)
    against_v11 = lab.judge(grid, list(STAGES), which)
    print("  against v11: {:.4f} vs {:.4f} ({:+.4f}), better on {}/{}".format(
        against_v11["all"]["held_out"], against_v11["all"]["base"], against_v11["all"]["held_out"] - against_v11["all"]["base"],
        against_v11["all"]["better"], against_v11["all"]["matches"]), flush=True)

    # 2. Nested selection: the search rerun with each match's score left out
    log("nested selection:")
    nested = []
    for m in ids:
        config, _ = search(outer, [j for j in ids if j != m])
        nested.append(fold(m, config, outer, base))
        log("  {}: {} -> {:+.4f}".format(m, config, nested[-1]["held_out"] - nested[-1]["base"]))
    nested_report = bar(nested)
    print("nested selection, against the mix:\n" + lab.describe(nested_report).replace("vs v11", "vs the mix"), flush=True)

    # 3. Fully nested: the search on a bench without the match, every fit without it too
    log("fully nested:")
    full = []
    for m in ids:
        inner_ids = [j for j in ids if j != m]
        inner = Chain(matches, ev, lags, inner_ids, round1_segments(which, exclude=(m,)))
        config, inner_trace = search(inner, inner_ids)
        full.append(fold(m, config, outer, base))
        full[-1]["inner_trace"] = inner_trace
        log("  {}: {} -> {:+.4f}".format(m, config, full[-1]["held_out"] - full[-1]["base"]))
        del inner
        gc.collect()
    full_report = bar(full)
    print("fully nested, against the mix:\n" + lab.describe(full_report).replace("vs v11", "vs the mix"), flush=True)

    # Diagnostics: possessions after each stage, the mix and the polished chain (all matches' fits)
    counts = {"mix": stage_counts(outer, CURRENT), "polished": stage_counts(outer, polished)}
    log("possessions per match after each stage: {}".format(counts))

    keep = report["passes"] and full_report["passes"]
    report.update({"against": "the mix (lab_mix.kept_base)", "grids": GRIDS, "current": CURRENT, "passes_max": PASSES,
                   "polished": polished, "polished_settings": describe_setting(polished), "trace": trace,
                   "configurations": int(grid.groupby(list(STAGES)).ngroups),
                   "against_v11": {k: against_v11[k] for k in ("all", "by_source", "unseen", "macro", "checks", "passes")},
                   "nested_selection": nested_report, "fully_nested": full_report,
                   "possessions_after_stage": counts, "keep": keep})
    lab.save(NAME, grid, report, which)
    log("verdict on {}: judge {}, fully nested {}: {}".format(which, report["passes"], full_report["passes"],
                                                             "KEEP" if keep else "discard"))
    return report


def save(which):
    """The polished chain's out-of-fold letters, for building on it (as lab_mix --save): each match
    with the configuration its fully nested fold picked (chosen without it) and every trained stage
    fitted on the other matches; their scores are the fully nested check's. Writes
    data/external/lab_polish[_idsse]/<match>.npy and bench/results/lab_polish_base[_idsse].json."""
    tag = bench_chain.suffix(which)
    with open(os.path.join(lab.RESULTS, "lab_{}{}.json".format(NAME, tag))) as handle:
        picks = {f["match"]: f["picked"] for f in json.load(handle)["fully_nested"]["folds"]}
    matches, ev, lags = load_all(which)
    outer = Chain(matches, ev, lags, list(matches), round1_segments(which))
    folder = LETTERS + tag
    os.makedirs(folder, exist_ok=True)
    base = {}
    for m, config in picks.items():
        key = tuple(config[s] for s in STAGES)
        letters = decode(outer.letters(key)[m])
        np.save(os.path.join(folder, "{}.npy".format(m)), letters)
        s = lab.score(matches[m], letters)
        base[m] = {"spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "source": matches[m].source, "unseen": matches[m].unseen}
    with open(os.path.join(lab.RESULTS, "lab_{}_base{}.json".format(NAME, tag)), "w") as handle:
        json.dump(base, handle, indent=1)
    print("polished chain: mean {:.4f}; letters in {}".format(np.mean([b["spell_f1"] for b in base.values()]),
                                                            os.path.relpath(folder, lab.ROOT)))


def polished_letters(m, which="all"):
    """The polished chain's out-of-fold letters on one bench match (after `--save`)."""
    return np.load(os.path.join(LETTERS + bench_chain.suffix(which), "{}.npy".format(m)))


def polished_base(which="all"):
    """The polished chain per match, in lab.baseline's format (after `--save`): lab.judge(..., base=)."""
    with open(os.path.join(lab.RESULTS, "lab_{}_base{}.json".format(NAME, bench_chain.suffix(which)))) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    parser.add_argument("--save", action="store_true", help="after the run: the polished chain's out-of-fold letters")
    args = parser.parse_args()
    if args.save:
        return save(args.which)
    run(args.which)


if __name__ == "__main__":
    main()
