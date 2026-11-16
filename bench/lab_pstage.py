"""Night 2026-11-14, builder "output": `pstage`, branch B's post stages given B's own start P (docs/lab/night/pstage.md).

    python -m bench.lab_pstage reproduce                       # R1, R2: trial 1015 through this file's code path
    python -m bench.lab_pstage prepare                         # the 4 variants' out-of-fold chains on the training copies; the choices
    python -m bench.lab_pstage confirm [--variants p25 p50 stressed]   # B's bench (the judge); --variants normal p10: information
    python -m bench.lab_pstage skillcorner                     # deployment models on SkillCorner's 20
    python -m bench.lab_pstage skillcorner-report              # SkillCorner's judge again from skillcorner.csv

Trial 1015 (v15's B) stays fixed: its start P, decoder, `short` and `segments`. What changes:
- `timing` (B's 3rd stage) gets 5 columns from the start P at each candidate, and per-change context columns;
- `recall` (lab_recall, never in B) runs as a 4th stage after `timing`, with 6 columns from the start P per proposal.
Every stage is trained as 1015's: on the 10 matches' copies under augment's training stress, the chain out of fold
on those copies, a held-out match's models trained on the other 9 copies' tables. The judged option is chosen for
each held-out match on the other 9 copies' out-of-fold scores, never on the bench and never on the held-out match.

Writes only to data/external/lab_pstage/ and bench/results/lab_pstage/ (a guard refuses any other write). Reads
search_b's caches (P on the training copies, the reference tables and CSVs) without writing them. Nothing reads our
match.
"""

import argparse
import builtins
import functools
import hashlib
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from bench import lab, lab_branch_b, lab_carried, lab_gate, lab_recall, lab_robust, lab_segments, lab_short, lab_timing
from bench import search_b, sets
from bench.run import score as bench_score

# Params
WHICH = "all"
L = lab_robust.L
B = lab_branch_b
C = lab_carried
R = lab_recall
T = lab_timing
CACHE = os.path.join(lab.ROOT, "data", "external", "lab_pstage")
RESULTS = os.path.join(lab.RESULTS, "lab_pstage")
TRIAL = 1015
REF_TABLES = os.path.join(search_b.CACHE, "tables_trial1015.joblib")          # read only (R1)
REF_CONFIRM = os.path.join(search_b.RESULTS, "confirm_1015_554_586_1133_1269_-1_-2.csv")
REF_NORMAL = os.path.join(search_b.RESULTS, "confirm_1015_-1_normal_p10.csv")
REF_SC = os.path.join(search_b.RESULTS, "skillcorner_1015_554_586_1133_1269.csv")
TOL = 1e-5
B_VARIANTS = ("p25", "p50", "stressed")
P_MINS = (0.7, 0.6, 0.5, 0.4)          # recall's p_min, most conservative first (ties go to it)
SLOPE = 5                              # frames either side for tp_slope_k
CROSS = 75                             # frames either side of g where the margin's crossing is looked for (3 s)
NO_CROSS = 9.0                         # s: tp_cross_dk when there's none
PT = ["tp_new_k", "tp_gap_k", "tp_slope_k", "tp_gap_g", "tp_cross_dk"]
PR = ["sp_n", "sp_o", "sp_d", "sp_nmax", "sp_arg", "sp_len"]
NO_CONTEXT = ("dk", "adk", "is_g")
T_BASE = T.FEATURES["all"] + PT
T_CONTEXT = [c + s for c in T_BASE if c not in NO_CONTEXT for s in ("_dmax", "_dmean")] + ["n_cand"]
T.FEATURES["allP_ctx"] = T_BASE + T_CONTEXT       # registered in this process only (lab_timing.py is not edited)
R.FEATURES["allP"] = R.FEATURES["all"] + PR
RECALL_MODEL = "hgb"                   # lab_recall's CHOSEN (features "all", hgb)
# The options: (a) timing+P; (c) B's timing + recall+P; (d) timing+P + recall+P; (b) B's timing + recall without P
# (the control, reported only). Order = tie order.
CHOICE = ["a"] + ["c{}".format(p) for p in P_MINS] + ["d{}".format(p) for p in P_MINS]
REPORTED = ["b{}".format(p) for p in P_MINS]
OPTIONS = ["1015"] + CHOICE + REPORTED


# --------------------------------------------------------------------------------------------
# Writes only here
# --------------------------------------------------------------------------------------------

def _allowed(path):
    p = os.path.realpath(os.fspath(path))
    return any(p == d or p.startswith(d + os.sep) for d in (os.path.realpath(CACHE), os.path.realpath(RESULTS)))


def guard():
    """Refuse every write outside this file's two folders (the shared caches are read, never written)."""
    def check(path):
        if isinstance(path, (str, bytes, os.PathLike)) and not _allowed(path):
            raise PermissionError("lab_pstage: refusing to write {}".format(path))

    def wrap(fn, arg=0):
        @functools.wraps(fn)
        def inner(*a, **k):
            check(a[arg] if len(a) > arg else next(iter(k.values())))
            return fn(*a, **k)
        return inner
    np.save, np.savez, np.savez_compressed = wrap(np.save), wrap(np.savez), wrap(np.savez_compressed)
    joblib.dump = wrap(joblib.dump, 1)
    pd.DataFrame.to_csv = wrap(pd.DataFrame.to_csv, 1)
    os.replace = wrap(os.replace, 1)
    original_makedirs = os.makedirs

    def guarded_makedirs(name, *a, **k):
        if not os.path.isdir(name):          # an existing folder is no write
            check(name)
        return original_makedirs(name, *a, **k)
    os.makedirs = guarded_makedirs
    original_open = builtins.open

    def guarded_open(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+"):
            check(file)
        return original_open(file, mode, *a, **k)
    builtins.open = guarded_open


def own_dead_cache():
    """lab_gate.own_dead_cache, writing (if ever needed) only here."""
    from bench import lab_dead

    def frame_features(match):
        key = hashlib.sha1(match.pred["p_dead"].tobytes() + match.track["ball"].tobytes()).hexdigest()[:12]
        name = "{}_{}.npy".format(match.id, key)
        for base in (lab_dead.CACHE, os.path.join(C.CACHE, "lab_dead"), os.path.join(lab_gate.CACHE, "lab_dead"),
                     os.path.join(CACHE, "lab_dead")):
            if os.path.exists(os.path.join(base, name)):
                return lab_dead.FEATURES["ball"], np.load(os.path.join(base, name))
        names, X_ = lab_dead.compute(match)
        os.makedirs(os.path.join(CACHE, "lab_dead"), exist_ok=True)
        np.save(os.path.join(CACHE, "lab_dead", name), X_)
        return names, X_
    lab_dead.frame_features = frame_features


def setup():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    guard()
    search_b.cache_short_evidence()          # keyed by (id, variant)
    own_dead_cache()


_TRIAL = {}


def trial():
    if not _TRIAL:
        _TRIAL.update({**search_b.every_trial()[TRIAL], "post": "own"})
    return dict(_TRIAL)


def settings():
    return search_b.settings()


# --------------------------------------------------------------------------------------------
# The new columns
# --------------------------------------------------------------------------------------------

def with_dead(p, f_dead):
    """search_b.start_p's P(dead) factor on a mixed P."""
    if f_dead == 1.0:
        return p
    p = p.copy()
    p[:, 2] *= f_dead
    return np.round(np.clip(p / p.sum(axis=1, keepdims=True), *L.CLIP), L.DECIMALS)


def start_p(m, data):
    t = trial()
    return search_b.start_p(m, data, t["w_gru"], t["f_dead"])


def timing_columns(tab, p):
    """timing's table with the 5 start-P columns and the per-change context."""
    if len(tab) == 0:
        return tab
    tab = tab.copy()
    n = len(p)
    ni = np.where(tab.new.to_numpy() == "H", 0, 1)
    k, g = tab.k.to_numpy().astype(int), tab.g.to_numpy().astype(int)
    gap = p[:, 0] - p[:, 1]
    sgn = np.where(ni == 0, 1.0, -1.0)
    tab["tp_new_k"] = p[k, ni]
    tab["tp_gap_k"] = sgn * gap[k]
    tab["tp_slope_k"] = sgn * (gap[np.clip(k + SLOPE, 0, n - 1)] - gap[np.clip(k - SLOPE, 0, n - 1)])
    tab["tp_gap_g"] = sgn * gap[g]
    cross = []
    for kk, gg, s in zip(k, g, sgn):
        lo, hi = max(gg - CROSS, 1), min(gg + CROSS, n - 1)
        w = s * gap[lo - 1:hi]
        c = np.flatnonzero((w[:-1] <= 0) & (w[1:] > 0)) + lo
        cross.append(abs(kk - c[np.argmin(np.abs(c - gg))]) / 25.0 if len(c) else NO_CROSS)
    tab["tp_cross_dk"] = cross
    grouped = tab.groupby("change")
    extra = {}
    for c in T_BASE:
        if c in NO_CONTEXT:
            continue
        extra[c + "_dmax"] = tab[c] - grouped[c].transform("max")
        extra[c + "_dmean"] = tab[c] - grouped[c].transform("mean")
    extra["n_cand"] = grouped["k"].transform("size")
    return pd.concat([tab, pd.DataFrame(extra, index=tab.index)], axis=1)


def recall_columns(tab, p):
    """recall's table with the 6 start-P columns per proposal."""
    tab = tab.copy()
    f = {k: [] for k in PR}
    for a, b, team in zip(tab.start, tab.stop, tab.team):
        n = 0 if team == "H" else 1
        w = p[a:b] if b > a else p[a:a + 1]
        f["sp_n"].append(w[:, n].mean())
        f["sp_o"].append(w[:, 1 - n].mean())
        f["sp_d"].append(w[:, 2].mean())
        f["sp_nmax"].append(w[:, n].max())
        f["sp_arg"].append(np.mean(w.argmax(1) == n))
        f["sp_len"].append(b - a)
    for k in PR:
        tab[k] = np.asarray(f[k], float)
    return tab


# --------------------------------------------------------------------------------------------
# The stages, with the P
# --------------------------------------------------------------------------------------------

def timing_table(match, letters, p, with_p):
    """timing's training table (with the truth's answers); with_p adds the P and context columns."""
    tab = T.table(match, letters)
    tab = tab.assign(right=T.targets(match, tab)) if len(tab) else tab
    return timing_columns(tab, p) if with_p else tab


def timing_model(tables, with_p):
    return T.train(tables, "allP_ctx" if with_p else "all")


def timing_apply(match, letters, p, model):
    """lab_timing.apply with 1015's window and p_min, the table carrying the P columns when the model reads them;
    returns (letters, moves)."""
    s = {**settings()["timing"], "p_min": trial()["timing_p_min"]}
    letters = np.asarray(letters).astype("<U1")
    tab = T.table(match, letters, max(s["window"], 0.0))
    if model["features"] == "allP_ctx":
        tab = timing_columns(tab, p)
    out, moves = T.act(letters, tab, T.probabilities(tab, model), s["window"], s["p_min"])
    return np.asarray(out).astype("<U1"), len(moves)


def recall_table(match, letters, p, truth=True):
    """recall's table on `letters` (features "all": its three generators), with the P columns (and the truth's answers)."""
    tab = R.table(match, np.asarray(letters).astype("<U1"), "all")
    if truth:
        tab = pd.concat([tab, R.targets(match, tab)], axis=1)
    return recall_columns(tab, p)


def recall_model(tables, with_p):
    return R.train(tables, "allP" if with_p else "all", RECALL_MODEL)


def recall_acts(letters, tab, model):
    """{p_min: (letters, written)} for every p_min of the grid."""
    probs = R.probabilities(tab, model)
    out = {}
    for p_min in P_MINS:
        new, written = R.act(letters, tab, probs, p_min)
        out[p_min] = (np.asarray(new).astype("<U1"), len(written))
    return out


def option_letters(match, seg, p, models):
    """Every option's final letters (and its moves / written) from `seg` (1015's letters after `segments`)."""
    out = {}
    tb, moves_b = timing_apply(match, seg, p, models["timing"])     # = lab_timing.apply (features "all": no P read)
    tp, moves_p = timing_apply(match, seg, p, models["timingP"])
    out["1015"] = (tb, moves_b)
    out["a"] = (tp, moves_p)
    tab_b = recall_table(match, tb, p, truth=False)
    tab_p = recall_table(match, tp, p, truth=False)
    for p_min, v in recall_acts(tb, tab_b, models["recall"]).items():
        out["b{}".format(p_min)] = v
    for p_min, v in recall_acts(tb, tab_b, models["recallP_on_B"]).items():
        out["c{}".format(p_min)] = v
    for p_min, v in recall_acts(tp, tab_p, models["recallP_on_P"]).items():
        out["d{}".format(p_min)] = v
    return out


def chain_1015(match, p, models):
    """1015's letters after `segments` (letters_of up to there), and its final letters."""
    s, t = settings(), trial()
    letters = L.decode_cost(match, -np.log(p), switch_team=t["switch_team"], switch_dead=t["switch_dead"],
                            min_possession=int(t["min_possession"]))
    if t["snap"]:
        letters = lab_robust.finish(match, letters, WHICH)
    letters = np.asarray(letters).astype("<U1")
    letters = lab_short.apply(match, letters, model=models["short"], **{**s["short"], "p_min": t["short_p_min"]})
    letters = lab_segments.apply(match, np.asarray(letters).astype("<U1"), models["segments"],
                                 **{**s["segments"], "flip": t["segments_flip"], "merge": t["segments_merge"]})
    return np.asarray(letters).astype("<U1")


# --------------------------------------------------------------------------------------------
# The out-of-fold chain on the training copies
# --------------------------------------------------------------------------------------------

def copies():
    return [search_b.train_copy(m) for m in lab.matches(WHICH)]


def oof_1015(matches):
    """search_b.oof_tables for trial 1015 on the training copies, also returning each stage's out-of-fold letters."""
    s, t = settings(), trial()
    letters = {x.id: search_b.trial_start(x, "train", t) for x in matches}
    steps = {"start": letters}
    tables = {}
    for stage in search_b.CHAIN:
        tabs = {x.id: search_b.stage_table(stage, x, letters[x.id], s[stage]) for x in matches}
        new = {}
        for x in matches:
            model = search_b.stage_model(stage, [tabs[o.id] for o in matches if o.id != x.id], s[stage])
            new[x.id] = np.asarray(search_b.apply_stage(stage, x, letters[x.id], model,
                                                        search_b.trial_setting(t, stage))).astype("<U1")
        tables[stage], letters = tabs, new
        steps[stage] = letters
    return tables, steps


def same_table(a, b):
    if isinstance(a, tuple):
        return all(same_table(x, y) for x, y in zip(a, b)) and len(a) == len(b)
    if isinstance(a, pd.DataFrame):
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            return False
        for c in a.columns:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            if x.dtype.kind in "fc" or y.dtype.kind in "fc":
                if not np.array_equal(x.astype(float), y.astype(float), equal_nan=True):
                    return False
            elif not np.array_equal(x, y):
                return False
        return True
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "fc":
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def models_1015(tables, without):
    s = settings()
    return {stage: search_b.stage_model(stage, [t for i, t in tabs.items() if i.split("~")[0] != without], s[stage])
            for stage, tabs in tables.items() if stage in search_b.CHAIN}


def all_models(tables, without):
    """Every stage model, trained on the copies' tables but `without`'s (None: all 10)."""
    keep = lambda tabs: [t for i, t in tabs.items() if i.split("~")[0] != without]
    models = models_1015(tables, without)
    models["timingP"] = timing_model(keep(tables["timingP"]), True)
    models["recall"] = recall_model(keep(tables["recall_B"]), False)
    models["recallP_on_B"] = recall_model(keep(tables["recall_B"]), True)
    models["recallP_on_P"] = recall_model(keep(tables["recall_P"]), True)
    return models


def tables_path(name):
    return os.path.join(CACHE, "tables_{}.joblib".format(name))


# --------------------------------------------------------------------------------------------
# R1, R2: 1015 reproduced
# --------------------------------------------------------------------------------------------

def reproduce():
    setup()
    started = time.time()
    matches = copies()
    tables, _ = oof_1015(matches)
    print("1015's out-of-fold chain on the training copies ({:.0f} s)".format(time.time() - started), flush=True)
    ref = joblib.load(REF_TABLES)
    r1 = {stage: {i: bool(same_table(tables[stage][i], ref[stage][i])) for i in ref[stage]} for stage in ref}
    r1_ok = set(ref) == set(tables) and all(all(v.values()) for v in r1.values()) and \
        all(set(ref[s]) == set(tables[s]) for s in ref)
    print("R1 (every table equal to search_b's tables_trial1015): {}".format(r1_ok), flush=True)
    if not r1_ok:
        print(json.dumps(r1, indent=1))
    joblib.dump(tables, tables_path("1015"), compress=3)
    del matches
    t = trial()
    reference = pd.read_csv(REF_CONFIRM)
    reference = reference[reference.trial == TRIAL].set_index(["match", "variant"]).spell_f1
    rows = []
    for m in lab.matches(WHICH):
        models = models_1015(tables, m)
        for variant in B_VARIANTS:
            match = lab_gate.variant_match(m, WHICH, variant)
            letters = search_b.letters_of(match, variant, t, {"own": models})
            mine = bench_score(match.truth, letters)["spell_f1"]
            rows.append({"match": m, "variant": variant, "spell_f1": mine, "reference": float(reference.loc[(m, variant)])})
            print("{} {}: {:.5f} (reference {:.5f})".format(m, variant, mine, rows[-1]["reference"]), flush=True)
            del match
    df = pd.DataFrame(rows)
    df["diff"] = (df.spell_f1.round(5) - df.reference).abs()
    r2_ok = bool((df["diff"] <= TOL).all())
    print("R2 (30 units equal to the confirm CSV to 1e-5): {}; B's bench {:.4f} (reference {:.4f})".format(
        r2_ok, df.groupby("match").spell_f1.mean().mean(), df.groupby("match").reference.mean().mean()), flush=True)
    df.to_csv(os.path.join(RESULTS, "reproduce.csv"), index=False, float_format="%.6f")
    with open(os.path.join(RESULTS, "reproduce.json"), "w") as handle:
        json.dump({"R1": r1_ok, "R1_tables": r1, "R2": r2_ok, "max_diff": float(df["diff"].max()),
                   "b_bench": float(df.groupby("match").spell_f1.mean().mean())}, handle, indent=1)


# --------------------------------------------------------------------------------------------
# The variants' chains on the training copies, and the choices
# --------------------------------------------------------------------------------------------

def prepare():
    setup()
    started = time.time()
    matches = copies()
    tables, steps = oof_1015(matches)
    ref = joblib.load(tables_path("1015"))
    assert all(same_table(tables[s][i], ref[s][i]) for s in ref for i in ref[s]), "the 1015 chain moved since reproduce"
    print("1015's chain again, equal to reproduce's ({:.0f} s)".format(time.time() - started), flush=True)
    p = {x.id: start_p(search_b.base_id(x), "train") for x in matches}
    seg, tb = steps["segments"], steps["timing"]
    # timing with P: its tables on the segments letters (out of fold), the models on the other 9, the letters
    tables["timingP"] = {x.id: timing_table(x, seg[x.id], p[x.id], True) for x in matches}
    tp, moves = {}, {}
    for x in matches:
        model = timing_model([tables["timingP"][o.id] for o in matches if o.id != x.id], True)
        tp[x.id], moves[x.id] = timing_apply(x, seg[x.id], p[x.id], model)
    print("timing with P out of fold ({:.0f} s)".format(time.time() - started), flush=True)
    # recall on B's timing letters (with and without P) and on timing+P's
    tables["recall_B"] = {x.id: recall_table(x, tb[x.id], p[x.id]) for x in matches}
    tables["recall_P"] = {x.id: recall_table(x, tp[x.id], p[x.id]) for x in matches}
    print("recall tables ({:.0f} s)".format(time.time() - started), flush=True)
    rows = []
    for x in matches:
        others = [o.id for o in matches if o.id != x.id]
        out = {"1015": (tb[x.id], 0), "a": (tp[x.id], moves[x.id])}
        for name, key, with_p, letters in (("b", "recall_B", False, tb[x.id]), ("c", "recall_B", True, tb[x.id]),
                                           ("d", "recall_P", True, tp[x.id])):
            model = recall_model([tables[key][o] for o in others], with_p)
            for p_min, v in recall_acts(letters, tables[key][x.id], model).items():
                out["{}{}".format(name, p_min)] = v
        for option, (letters, n) in out.items():
            rows.append({"copy": search_b.base_id(x), "option": option, "spell_f1": bench_score(x.truth, letters)["spell_f1"],
                         "actions": n})
    print("every option out of fold ({:.0f} s)".format(time.time() - started), flush=True)
    joblib.dump(tables, tables_path("pstage"), compress=3)
    oof = pd.DataFrame(rows)
    oof.to_csv(os.path.join(RESULTS, "copies_oof.csv"), index=False, float_format="%.6f")
    choices = choose(oof)
    with open(os.path.join(RESULTS, "choices.json"), "w") as handle:
        json.dump(choices, handle, indent=1)
    print("choices: {}".format(json.dumps(choices)), flush=True)


def choose(oof):
    """For each held-out match, the option with the best mean on the other 9 copies (rounded to 6, ties to the
    first in CHOICE's order); 'deployment': the same on all 10."""
    per = oof.pivot_table(index="option", columns="copy", values="spell_f1")
    out = {}
    for m in list(per.columns) + ["deployment"]:
        others = [c for c in per.columns if c != m]
        means = per.loc[CHOICE, others].mean(axis=1).round(6)
        out[m] = str(means.index[int(np.argmax(means.to_numpy()))])
    return out


# --------------------------------------------------------------------------------------------
# The judge: B's bench (and the normal bench, for information)
# --------------------------------------------------------------------------------------------

def confirm(variants=B_VARIANTS):
    setup()
    tables = joblib.load(tables_path("pstage"))
    with open(os.path.join(RESULTS, "choices.json")) as handle:
        choices = json.load(handle)
    rows = []
    for m in lab.matches(WHICH):
        started = time.time()
        models = all_models(tables, m)
        for variant in variants:
            match = lab_gate.variant_match(m, WHICH, variant)
            p = start_p(m, variant)
            seg = chain_1015(match, p, models)
            for option, (letters, n) in option_letters(match, seg, p, models).items():
                rows.append({"match": m, "variant": variant, "option": option, "actions": n,
                             "spell_f1": bench_score(match.truth, letters)["spell_f1"]})
            mine = {r["option"]: r["spell_f1"] for r in rows if r["match"] == m and r["variant"] == variant}
            print("{} {}: 1015 {:.4f}  pstage ({}) {:.4f}  {}  ({:.0f} s)".format(
                m, variant, mine["1015"], choices[m], mine[choices[m]],
                " ".join("{}={:.4f}".format(o, mine[o]) for o in OPTIONS[1:]), time.time() - started), flush=True)
            del match
    df = pd.DataFrame(rows)
    tail = "" if tuple(variants) == B_VARIANTS else "_" + "_".join(variants)
    df.to_csv(os.path.join(RESULTS, "confirm{}.csv".format(tail)), index=False, float_format="%.6f")
    return judge(df, choices, variants)


def reference_rows(variants):
    if tuple(variants) == B_VARIANTS:
        ref = pd.read_csv(REF_CONFIRM)
    else:
        ref = pd.read_csv(REF_NORMAL)
    ref = ref[(ref.trial == TRIAL) & ref.variant.isin(variants)]
    return ref.set_index(["match", "variant"]).spell_f1


def judge(df, choices, variants=B_VARIANTS):
    ref = reference_rows(variants)
    base = df[df.option == "1015"].set_index(["match", "variant"]).spell_f1
    diff = (base.round(5) - ref.reindex(base.index)).abs()
    r_ok = bool((diff <= TOL).all())
    print("1015 reproduced on these {} units to 1e-5: {} (max {:.1e})".format(len(base), r_ok, float(diff.max())))
    picked = df[df.option == df.match.map(choices)].set_index(["match", "variant"]).spell_f1
    per_match = pd.DataFrame({"1015": base.groupby(level="match").mean(), "pstage": picked.groupby(level="match").mean()})
    per_match["gain"] = per_match.pstage - per_match["1015"]
    per_match["option"] = [choices[m] for m in per_match.index]
    per_match["source"] = [sets.source(m) for m in per_match.index]
    by_variant = pd.DataFrame({"1015": base.groupby(level="variant").mean(), "pstage": picked.groupby(level="variant").mean()})
    by_variant["gain"] = by_variant.pstage - by_variant["1015"]
    by_source = per_match.groupby("source")[["1015", "pstage", "gain"]].mean()
    g = per_match.gain
    report = {"variants": list(variants), "reproduced": r_ok, "max_diff": float(diff.max()),
              "bench_1015": float(per_match["1015"].mean()), "bench_pstage": float(per_match.pstage.mean()),
              "gain": float(g.mean()), "better": int((g > 0).sum()), "worse": int((g < 0).sum()),
              "per_match": per_match.reset_index().to_dict(orient="records"),
              "by_variant": by_variant.reset_index().to_dict(orient="records"),
              "by_source": by_source.reset_index().to_dict(orient="records")}
    print(per_match.round(4).to_string())
    print(by_variant.round(4).to_string())
    print(by_source.round(4).to_string())
    if tuple(variants) == B_VARIANTS:
        parts = {"gain": bool(g.mean() >= B.GAIN), "better": bool((g > 0).sum() >= B.BETTER),
                 "variants": bool(by_variant.gain.min() >= -B.VARIANT_LOSS), "sources": bool(by_source.gain.min() >= 0)}
        report["bar_bench"] = parts
        print("B's bench: {:.4f} against 1015's {:.4f}: {:+.4f}, better on {}/10, worst variant {:+.4f}, sources {} -> {}".format(
            per_match.pstage.mean(), per_match["1015"].mean(), g.mean(), int((g > 0).sum()), by_variant.gain.min(),
            ", ".join("{} {:+.4f}".format(s, v) for s, v in by_source.gain.items()), parts))
    # every option, in sample, information only
    per_option = df.groupby(["option", "match"]).spell_f1.mean().unstack()
    info = pd.DataFrame({"bench": per_option.mean(axis=1), "gain": per_option.sub(per_option.loc["1015"]).mean(axis=1),
                         "better": (per_option.sub(per_option.loc["1015"]) > 0).sum(axis=1),
                         "actions_per_unit": df.groupby("option").actions.mean()}).loc[OPTIONS]
    print("every option, in sample (information, never judged):\n" + info.round(4).to_string())
    report["options_in_sample"] = info.reset_index().to_dict(orient="records")
    tail = "" if tuple(variants) == B_VARIANTS else "_" + "_".join(variants)
    with open(os.path.join(RESULTS, "report{}.json".format(tail)), "w") as handle:
        json.dump(report, handle, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    return report


# --------------------------------------------------------------------------------------------
# SkillCorner: deployment models
# --------------------------------------------------------------------------------------------

def skillcorner():
    from bench import lab_skillcorner
    from bench import skillcorner as sc
    setup()                                 # search_b's evidence cache, keyed (id, None) there
    t = trial()
    tables = joblib.load(tables_path("pstage"))
    with open(os.path.join(RESULTS, "choices.json")) as handle:
        option = json.load(handle)["deployment"]
    models = all_models(tables, None)
    aug = C.deployment_augment()
    gru = lab_gate.gru_model(B.used_for(B.held_list()[-1]), WHICH, B.net_path(B.held_list()[-1]))
    rows = []
    for m in sc.match_ids():
        started = time.time()
        gt = sc.load(m)["labels"]
        match, X, Xs = C.sc_prepared(m)
        p_aug = C.proba(aug, {"X": X, "X_swapped": Xs})
        p_gru = lab_gate.gru_proba(gru, {"X": X, "X_swapped": Xs}, match.halves)
        p = with_dead(B.mix(p_gru, p_aug, t["w_gru"]), t["f_dead"])
        seg = chain_1015(match, p, models)
        for o, (letters, n) in option_letters(match, seg, p, models).items():
            rows.append({"match": m, "option": o, "actions": n, "spell_f1": lab_skillcorner.score(gt, letters)["spell_f1"]})
        mine = {r["option"]: r["spell_f1"] for r in rows if r["match"] == m}
        print("{}: 1015 {:.4f}  pstage ({}) {:.4f}  ({:.0f} s)".format(m, mine["1015"], option, mine[option],
                                                                      time.time() - started), flush=True)
        del match, X, Xs
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "skillcorner.csv"), index=False, float_format="%.6f")
    return sc_judge(df, option)


def sc_judge(df, option):
    """SkillCorner's part of the bar, and R3 there: 1015's 20 matches against search_b's CSV (5 decimals there; ids
    compared as strings; |mine - reference| unrounded within 1e-5)."""
    ref = pd.read_csv(REF_SC)
    ref = ref[ref.trial == TRIAL]
    ref = pd.Series(ref.spell_f1.to_numpy(), index=ref.match.astype(str))
    df = df.assign(match=df.match.astype(str))
    per = df.pivot_table(index="match", columns="option", values="spell_f1")
    diff = (per["1015"] - ref.reindex(per.index)).abs()
    g = per[option] - per["1015"]
    report = {"option": option, "reproduced": bool((diff <= TOL).all()), "max_diff": float(diff.max()),
              "sc_1015": float(per["1015"].mean()), "sc_pstage": float(per[option].mean()), "gain": float(g.mean()),
              "better": int((g > 0).sum()), "pass": bool(per[option].mean() >= per["1015"].mean() - B.SC_LOSS),
              "options": {o: float(per[o].mean()) for o in per.columns}}
    print("1015 reproduced on SkillCorner to 1e-5: {} (max {:.1e})".format(report["reproduced"], report["max_diff"]))
    print("SkillCorner: pstage ({}) {:.4f} against 1015's {:.4f}: {:+.4f}, better on {}/20; within 0.005: {}".format(
        option, report["sc_pstage"], report["sc_1015"], report["gain"], report["better"], report["pass"]))
    print("every option (information): " + " ".join("{}={:.4f}".format(o, v) for o, v in report["options"].items()))
    with open(os.path.join(RESULTS, "report_skillcorner.json"), "w") as handle:
        json.dump(report, handle, indent=1)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("reproduce", "prepare", "confirm", "skillcorner", "skillcorner-report"))
    parser.add_argument("--variants", nargs="+", default=list(B_VARIANTS), choices=lab_gate.VARIANTS)
    args = parser.parse_args()
    if args.step == "reproduce":
        return reproduce()
    if args.step == "prepare":
        return prepare()
    if args.step == "confirm":
        return confirm(tuple(args.variants))
    if args.step == "skillcorner-report":        # the judge again from skillcorner.csv (no model run)
        setup()
        with open(os.path.join(RESULTS, "choices.json")) as handle:
            return sc_judge(pd.read_csv(os.path.join(RESULTS, "skillcorner.csv")), json.load(handle)["deployment"])
    return skillcorner()


if __name__ == "__main__":
    main()
