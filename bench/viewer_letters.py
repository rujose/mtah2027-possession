"""The latest versions' letters on every bench match, for the viewer (viewer/build_bench_cache.py reads them).

    python -m bench.viewer_letters v15 --variant normal|stressed       # v15's branches A and B and its switch
    python -m bench.viewer_letters netmet5 --variant normal|stressed   # night 2026-11-14's candidate (not adopted)

One variant per process (lab_short's cache is keyed by match id), and `netmet5` in a process of its own (the mirror
repoints module-level caches). Writes data/external/viewer_letters/<variant>/<lane>/<match>.npy (gitignored):

- `a`: v15's branch A (= v14's), each match out of fold exactly as bench/check_v14_switch.py scores it;
- `b`: v15's branch B (search trial 1015), each match's stage models trained without it, as search_b confirm runs it;
- `v15`: what v15 outputs: the switch (lab_gate.THRESHOLD) picks A or B per match from the match's own signal;
- `netmet5`: branch B with night 2026-11-14's `netmet5` network (Metrica's games in training, 4 seeds averaged),
  trial 1015's settings, from the network builder's mirror (bench/lab_netmirror.py, docs/lab/night/network.md).

Each lane's spellF1 is printed beside the score the lab recorded, as a check. Nothing reads our match.
"""

import argparse
import json
import os
import time

import numpy as np

from bench import lab, lab_carried, lab_gate, lab_mix, search_b
from bench.run import score as bench_score
from possession import v13, v14

# Params
WHICH = "all"
OUT = os.path.join(lab.ROOT, "data", "external", "viewer_letters")
TRIAL = 1015


def path(variant, lane, m):
    return os.path.join(OUT, variant, lane, "{}.npy".format(m))


def save(variant, lane, letters):
    os.makedirs(os.path.join(OUT, variant, lane), exist_ok=True)
    for m, x in letters.items():
        np.save(path(variant, lane, m), np.asarray(x).astype("<U1"))


def branch_a(variant, matches):
    """bench/check_v14_switch.chain, keeping the letters."""
    picks = lab_mix.fold_settings(WHICH)
    start = lab_carried.average_picks(WHICH)
    letters = {}
    for m in matches:
        p = v14.mix(lab_gate.held_out_p("gru", variant, m.id, WHICH), lab_gate.held_out_p("robust", variant, m.id, WHICH))
        decoded = v13.L.decode_cost(m, -np.log(p), **start[m.id])
        if v14.SNAP:
            decoded = v13.lab_robust.finish(m, decoded, WHICH)
        letters[m.id] = np.asarray(decoded).astype("<U1")
    for stage in v13.CHAIN[WHICH]:
        letters = lab_mix.add_stage(stage, letters, matches, picks)
    return letters


def branch_b(variant, matches):
    """search_b.confirm for one trial, keeping the letters."""
    trial = {**search_b.every_trial()[TRIAL], "post": "own"}
    out = {}
    for x in matches:
        models = search_b.trial_models(search_b.every_trial()[TRIAL], without=x.id)
        out[x.id] = search_b.letters_of(x, variant, trial, {"own": models})
    return out


def report(name, variant, matches, letters):
    f1 = [bench_score(x.truth, letters[x.id])["spell_f1"] for x in matches]
    print("{} on {}: spellF1 {:.4f} ({})".format(name, variant, float(np.mean(f1)),
                                                ", ".join("{} {:.3f}".format(x.id, s) for x, s in zip(matches, f1))), flush=True)
    return dict(zip([x.id for x in matches], f1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=("v15", "netmet5"))
    parser.add_argument("--variant", choices=("normal", "stressed"), required=True)
    args = parser.parse_args()
    started = time.time()
    if args.what == "netmet5":
        from bench import lab_netmirror
        lab_netmirror.use_named("netmet", "ens4")
    search_b.cache_short_evidence()
    lab_gate.own_dead_cache()
    matches = [lab_gate.variant_match(m, WHICH, args.variant) for m in lab.matches(WHICH)]
    scores = {}
    if args.what == "v15":
        a = branch_a(args.variant, matches)
        b = branch_b(args.variant, matches)
        sig = lab_gate.signal_table(WHICH)[args.variant]
        picks = {x.id: lab_gate.picks_b(sig[x.id]) for x in matches}
        v15 = {x.id: b[x.id] if picks[x.id] else a[x.id] for x in matches}
        for lane, letters in (("a", a), ("b", b), ("v15", v15)):
            save(args.variant, lane, letters)
            scores[lane] = report(lane, args.variant, matches, letters)
        scores["picks_b"] = picks
    else:
        b = branch_b(args.variant, matches)
        save(args.variant, "netmet5", b)
        scores["netmet5"] = report("netmet5", args.variant, matches, b)
    with open(os.path.join(OUT, args.variant, "scores_{}.json".format(args.what)), "w") as handle:
        json.dump(scores, handle, indent=1)
    print("done in {:.0f} s".format(time.time() - started), flush=True)


if __name__ == "__main__":
    main()
