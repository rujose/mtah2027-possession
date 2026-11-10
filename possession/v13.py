"""Model v13: v12 with the lab's temporal `average` start (docs/models/v13-temporal.md).

    python -m possession.v13           # writes evaluation/results/v13.csv; prints only its path and SHA-256

v12 (possession/v12.py) starts its chain from robust, a per-frame classifier. v13 starts it from the
mean of robust's P and a small bidirectional GRU's P over the same 97 blanked features, read every
5th frame with +-60 s of context (bench/lab_temporal.py, docs/lab/temporal.md, lab round 7):

  P = round(clip((P_gru + P_robust) / 2)), decoded with `average`'s chosen setting, then v9's snap
  and v11's move, then v12's post stages of the bench set in v10.BENCH_SET:
    all:   short -> segments -> timing
    idsse: short -> segments -> recall -> dead
  each on the setting chosen on all the set's matches, with models `python -m bench.train_v13`
  trained on the bench matches (possession/models/v13_<set>.joblib). Last, as v12, the second half's
  graded frames before its kick-off are D.

The inputs are v12's (`v12.inputs`), robust is v12's own classifier, and every stage is the lab's code.
The GRU runs in float64 (as v12 runs PathCRF), so another CPU can't move the file by float32's last
bit. `python -m bench.check_v13` shows the port gives the lab's letters on bench matches.

The start is one function, `start_proba(match, models, which)`, and the rest one more,
`run_chain_from(match, p, models, which)`, so a different start can plug in.

The stoppage switch (docs/lab/gate.md, lab round 7). v13's start, like v12's, learned "dead <=> no ball"
from the bench, whose dead ball is frozen or missing; where a match's ball stays tracked and moving
through stoppages (SkillCorner's broadcast tracking) it misses them. Before the chain, v13 reads one
number from the tracking and v4's players-only P(dead) (bench/lab_gate.py `signal`): how much more
often the ball is seen and moving where the players say dead than where they say live. Above
THRESHOLD (fixed on the bench and its stressed copies) it runs branch B: `augment`, robust retrained
with SkillCorner-like stoppage balls, then v12's post stages refitted on it
(possession/models/v13_augment_all.joblib). Otherwise branch A, as above. Nothing prints the signal
or the branch unless asked (`--show-branch`, for Jose; not to be overridden by hand).
"""

import argparse
import hashlib
import os

import joblib
import numpy as np

from bench import lab_gate, lab_recall, lab_robust, lab_segments, lab_short, lab_temporal, lab_timing
from possession import v1, v9, v10, v11, v12
from possession import lab_learned_features as llf
from possession.data import PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v13.csv")
MODELS = os.path.join(ROOT, "possession", "models", "v13_{}.joblib")
CHAIN = v12.CHAIN                   # the post stages: v12's, per bench set
L = lab_robust.L                    # CLIP (0.01, 0.99), DECIMALS 3, SWAP, decode_cost
# The GRU's arithmetic. float64 makes it the same on any CPU to ~1e-16; float32 is what the lab ran
# (bench/check_v13.py measures the difference). The weights are the float32 ones it was trained with.
GRU_FLOAT64 = True
# What the network's shape and reading depend on (bench/lab_temporal.py): saved with the weights and
# checked on load, so a changed lab file can't silently change v13.
GRU_PARAMS = ("STEP", "CHUNK", "STRIDE", "Z_CLIP", "MISSING_SHARE", "PROJ", "HIDDEN", "LAYERS", "DROPOUT")
# The stoppage switch: on, for the bench set(s) whose branch B was built (lab_carried built `augment` for `all`
# only; on `idsse` v13 always runs branch A).
SWITCH = True
SWITCH_SETS = ("all",)
AUGMENT = os.path.join(ROOT, "possession", "models", "v13_augment_{}.joblib")


# --------------------------------------------------------------------------------------------
# The GRU
# --------------------------------------------------------------------------------------------

def gru_bundle(state, norm, names, **extra):
    """What a model file keeps of one network: its weights as float32 arrays, the normaliser (mean,
    sd, flagged features, from its training matches), the 97 feature names it reads and the constants."""
    state = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)).copy() for k, v in state.items()}
    return {"state": state, "norm": {k: np.asarray(v).copy() for k, v in norm.items()}, "names": list(names),
            "n_in": len(names) + len(norm["flagged"]), "params": {k: getattr(lab_temporal, k) for k in GRU_PARAMS},
            **extra}


def gru_network(gru, dtype=None):
    """The network rebuilt from a bundle, in eval mode, in float64 (default: GRU_FLOAT64) or float32."""
    dtype = dtype or (np.float64 if GRU_FLOAT64 else np.float32)
    current = {k: getattr(lab_temporal, k) for k in GRU_PARAMS}
    if gru["params"] != current:
        raise ValueError("the saved GRU's constants differ from bench/lab_temporal.py: {} vs {}".format(gru["params"], current))
    torch = lab_temporal.torch_ready()
    net = lab_temporal.network(gru["n_in"])
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in gru["state"].items()})
    net.eval()
    return net.double() if dtype == np.float64 else net


def blanked_features(match, which):
    """(names, X, X_swapped): lab_robust.cached's features (all 97 columns), built as v12.robust_proba
    builds robust's: the still ball blanked, v11's touches and v9's passes on that tracking."""
    b = lab_robust.blanked(match)
    found = v11.seen_touches(b.track, v11.SETTINGS[which]["dv"])
    passes = v9.seen_passes(b.track)
    names, X = llf.frame_features(b.track, b.halves, b.pitch, b.pred, found, passes)
    _, Xs = llf.frame_features(b.track, b.halves, b.pitch, b.pred, found, passes, swap=True)
    return names, X, Xs


def run_half(net, Z, norm, dtype):
    """lab_temporal.run_half with the network's inputs in `dtype`: P [frames, 3] for one half, each
    phase its own 5 Hz sequence, read in CHUNK-step windows every STRIDE steps, overlaps averaged with
    triangular weights."""
    torch = lab_temporal.torch_ready()
    step, chunk, stride = lab_temporal.STEP, lab_temporal.CHUNK, lab_temporal.STRIDE
    out = np.zeros((len(Z), 3))
    for o in range(step):
        idx = np.arange(o, len(Z), step)
        x = lab_temporal.inputs(Z[idx], norm).astype(dtype, copy=False)
        k = len(idx)
        w = min(chunk, k)
        starts = list(range(0, k - w + 1, stride))
        if starts[-1] + w < k:
            starts.append(k - w)
        with torch.no_grad():
            prob = torch.softmax(net(torch.from_numpy(np.stack([x[s:s + w] for s in starts]))), dim=-1).numpy()
        tri = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1)).astype(np.float64)
        acc, weight = np.zeros((k, 3)), np.zeros(k)
        for s, p in zip(starts, prob):
            acc[s:s + w] += p * tri[:, None]
            weight[s:s + w] += tri
        out[idx] = acc / weight[:, None]
    return out


def gru_raw(match, gru, which, dtype=None, features=None):
    """The network's P(H, A, D) per frame before clipping and rounding: the mean of the match as it is
    and with the teams exchanged; 1/3 each outside the halves (lab_temporal.proba, without the cache)."""
    dtype = dtype or (np.float64 if GRU_FLOAT64 else np.float32)
    names, X, Xs = features or blanked_features(match, which)
    if list(names) != gru["names"]:
        raise ValueError("the GRU's features differ from possession/lab_learned_features.py")
    net, norm = gru_network(gru, dtype), gru["norm"]
    p = np.full((len(X), 3), 1 / 3)
    for _, first, last in match.halves:
        p[first:last + 1] = 0.0
    for M, cols in ((X, np.arange(3)), (Xs, L.SWAP)):
        for _, first, last in match.halves:
            Z = lab_temporal.standardised_half(M, first, last, norm)
            p[first:last + 1] += run_half(net, Z, norm, dtype)[:, cols] / 2
    return p


def clipped(p):
    """P clipped to 0.01-0.99 and rounded to 3 decimals, as robust's (one frame can't cost more than 4.6)."""
    return np.round(np.clip(p, *L.CLIP), L.DECIMALS)


def gru_proba(match, gru, which, dtype=None, features=None):
    """The network's P, clipped and rounded as robust's."""
    return clipped(gru_raw(match, gru, which, dtype, features))


# --------------------------------------------------------------------------------------------
# The start and the chain
# --------------------------------------------------------------------------------------------

def average(p_gru, p_robust):
    """`average`'s P: the mean of both, clipped and rounded. The lab averaged the P it had saved as
    float32 (lab_temporal.held_out), so both go through float32 first: about half the means fall
    exactly on a 0.0005 tie, and this makes them round as the lab's did."""
    return clipped((p_gru.astype(np.float32).astype(np.float64) + p_robust.astype(np.float32).astype(np.float64)) / 2)


def start_proba(match, models, which, dtype=None):
    """P(H, A, D) per frame from v13's start: the GRU's and robust's (v12's own classifier) averaged."""
    return average(gru_proba(match, models["gru"], which, dtype), v12.robust_proba(match, models["robust"], which))


def post_stages(match, letters, models, which, settings):
    """v12's post stages of the set, in order, each with its model and setting (as v12.run_chain)."""
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
            letters = v12.dead_apply(match, letters, model, **s)
        else:
            raise ValueError(stage)
    return np.asarray(letters).astype("<U1")


def run_chain_from(match, p, models, which, settings=None):
    """Letters for every frame from a start's P: -log P decoded with the start's setting, v9's snap and
    v11's move, then v12's post stages of the set."""
    settings = settings or models["settings"]
    decoded = L.decode_cost(match, -np.log(p), **settings["start"])
    letters = np.asarray(lab_robust.finish(match, decoded, which)).astype("<U1")
    return post_stages(match, letters, models, which, settings)


def load_models(which):
    path = MODELS.format(which)
    if not os.path.exists(path):
        raise FileNotFoundError("{} is missing: run `python -m bench.train_v13 --set {}`".format(path, which))
    models = joblib.load(path)
    if models["which"] != which or tuple(models["chain"]) != CHAIN[which]:
        raise ValueError("{} was trained for another set or chain".format(path))
    return models


# --------------------------------------------------------------------------------------------
# The stoppage switch (docs/lab/gate.md)
# --------------------------------------------------------------------------------------------

def load_branch_b(which):
    path = AUGMENT.format(which)
    if not os.path.exists(path):
        raise FileNotFoundError("{} is missing: run `python -m bench.train_v13 --switch`".format(path))
    bundle = joblib.load(path)
    if bundle["which"] != which or tuple(bundle["chain"]) != CHAIN[which]:
        raise ValueError("{} was trained for another set or chain".format(path))
    if bundle["threshold"] != lab_gate.THRESHOLD:
        raise ValueError("{}'s threshold {} differs from bench/lab_gate.py's {}".format(path, bundle["threshold"], lab_gate.THRESHOLD))
    return bundle


def branch(fixed, pred, halves, pitch, which):
    """"B" when the switch is on for this set and the tracking's signal is above the threshold, else "A"."""
    if not SWITCH or which not in SWITCH_SETS:
        return "A"
    return "B" if lab_gate.picks_b(lab_gate.signal(fixed, pred, halves, pitch)) else "A"


def augment_proba(match, bundle, which, features=None):
    """Branch B's start: `augment`'s P(H, A, D) on robust's 97 blanked features, the mean of the match as it is
    and with the teams exchanged, clipped and rounded (lab_carried.proba, without the cache)."""
    names, X, Xs = features or blanked_features(match, which)
    if list(names) != list(bundle["names"]):
        raise ValueError("augment's features differ from possession/lab_learned_features.py")
    clf = bundle["augment"]
    return clipped((clf.predict_proba(X) + clf.predict_proba(Xs)[:, L.SWAP]) / 2)


def predict(tracking, tags=None, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH), bench_set=None):
    """(labels for every video frame, {"pred": the inputs, "p_start": the start's P, "branch": "A" or "B"}). As
    v12: Smart Tagging feeds no model; its second-half kick-off tag only marks the frames before kick-off as D."""
    halves = halves or graded_halves()
    which = bench_set or v10.BENCH_SET
    fixed, pred = v12.inputs(tracking, halves, pitch, which)
    match = v12.as_match(fixed, pred, halves, pitch)
    chosen = branch(fixed, pred, halves, pitch, which)
    models = load_branch_b(which) if chosen == "B" else load_models(which)
    p = augment_proba(match, models, which) if chosen == "B" else start_proba(match, models, which)
    labels = run_chain_from(match, p, models, which)
    labels[v12.before_kickoff(tags, len(labels), halves)] = "D"
    return labels, {"pred": pred, "p_start": p, "branch": chosen}


def main():
    """Only what Jose needs to check the file, as v10-v12: no statistics about the match. --show-branch adds the
    branch the switch took (Jose's to read; Claude doesn't run it on our match)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--show-branch", action="store_true", help="also print the branch the stoppage switch took")
    args = parser.parse_args()
    labels, info = predict(load_tracking(), load_tags())
    v1.write_submission(OUTPUT, labels)
    with open(OUTPUT, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    path = os.path.relpath(OUTPUT, ROOT)
    print("wrote {} (bench set {})".format(path, v10.BENCH_SET))
    print("SHA-256 {}".format(digest))
    if args.show_branch:
        print("stoppage switch: branch {}".format(info["branch"]))
    print("format check: .venv/bin/python evaluation/validate_submission.py {}".format(path))


if __name__ == "__main__":
    main()
