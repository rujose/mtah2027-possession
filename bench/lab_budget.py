"""The error budget of a version on the bench: what each kind of error costs (docs/lab.md).

    python -m bench.lab_budget                 # on round 1's mix (bench/lab_mix.py --save)
    python -m bench.lab_budget --labels v11    # on v11

Each "oracle" fixes one kind of error with the truth and nothing else, so its gain is that error's
share of what's left. Exploration only: the truth is read on every match, so nothing here may pick
a setting. It says where the next round should look.
"""

import argparse

import numpy as np
import pandas as pd

from bench import lab, lab_mix
from possession.data import FPS

# Params
IOU = 0.3          # a true and a predicted possession of the same team "match" from here (the metric's lowest)
NEAR = 2 * FPS     # a predicted change this close to a true change of the same kind is "the same change"
SHORT = 2 * FPS    # a missed possession under this is "short"


def runs(letters):
    """[(start, stop, letter)] maximal runs."""
    edges = np.flatnonzero(letters[1:] != letters[:-1]) + 1
    bounds = np.r_[0, edges, len(letters)]
    return [(int(bounds[i]), int(bounds[i + 1]), letters[bounds[i]]) for i in range(len(bounds) - 1)]


def spells(letters):
    return [r for r in runs(letters) if r[2] in ("H", "A")]


def iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return inter / float(max(a[1], b[1]) - min(a[0], b[0])) if inter > 0 else 0.0


def best_iou(spell, others):
    """The best IoU of `spell` against the same-team spells in `others` (sorted by start)."""
    best = 0.0
    for o in others:
        if o[0] >= spell[1]:
            break
        if o[2] == spell[2] and o[1] > spell[0]:
            best = max(best, iou(spell, o))
    return best


def oracles(truth, pred):
    """{kind: letters with that kind of error fixed}, plus counts and the missed possessions' details."""
    t_spells, p_spells = spells(truth), spells(pred)
    out, info = {}, {}

    fixed, missed = pred.copy(), []
    for s in t_spells:
        if best_iou(s, p_spells) < IOU:
            fixed[s[0]:s[1]] = s[2]
            missed.append(s)
    out["missed possessions"] = fixed
    info["missed"] = missed

    fixed = pred.copy()
    for i, s in enumerate(p_spells):
        frames = truth[s[0]:s[1]]
        teams = frames[np.isin(frames, ["H", "A"])]
        if len(teams) and np.mean(teams == s[2]) < 0.5:
            fixed[s[0]:s[1]] = "A" if s[2] == "H" else "H"
    out["team of each possession"] = fixed

    fixed = pred.copy()
    for s in p_spells:
        if best_iou(s, t_spells) < IOU and s[0] > 0:
            fixed[s[0]:s[1]] = fixed[s[0] - 1]
    out["spurious possessions (merged into what precedes)"] = fixed

    def boundaries(letters, kind):
        a, b = letters[:-1], letters[1:]
        if kind == "team change":
            hit = np.isin(a, ["H", "A"]) & np.isin(b, ["H", "A"]) & (a != b)
        elif kind == "stoppage start":
            hit = np.isin(a, ["H", "A"]) & (b == "D")
        else:
            hit = (a == "D") & np.isin(b, ["H", "A"])
        return np.flatnonzero(hit) + 1

    for kind in ("team change", "stoppage start", "restart"):
        fixed = pred.copy()
        true_b = boundaries(truth, kind)
        for g in boundaries(pred, kind):
            before, after = pred[g - 1], pred[g]
            near = true_b[(np.abs(true_b - g) <= NEAR) & (truth[np.minimum(true_b, len(truth) - 1)] == after)
                          & (truth[true_b - 1] == before)]
            if len(near) == 0:
                continue
            f = near[np.argmin(np.abs(near - g))]
            if f < g:
                fixed[f:g] = after
            elif f > g:
                fixed[g:f] = before
        out["{} timing".format(kind)] = fixed

    fixed = pred.copy()
    for s in runs(truth):
        if s[2] == "D" and not np.any(pred[s[0]:s[1]] == "D"):
            fixed[s[0]:s[1]] = "D"
    out["missed stoppages"] = fixed

    fixed = pred.copy()
    for s in runs(pred):
        if s[2] == "D" and not np.any(truth[s[0]:s[1]] == "D"):
            fixed[s[0]:s[1]] = truth[s[0]:s[1]]
    out["false stoppages"] = fixed
    return out, info


def where(missed, truth):
    """Shares of the missed possessions: short, between two possessions of the other team, next to D."""
    if not missed:
        return {}
    r = runs(truth)
    index = {(a, b): i for i, (a, b, _) in enumerate(r)}
    between = next_to_d = 0
    for a, b, k in missed:
        i = index.get((a, b))
        prev = r[i - 1][2] if i else None
        nxt = r[i + 1][2] if i is not None and i + 1 < len(r) else None
        other = "A" if k == "H" else "H"
        between += prev == other and nxt == other
        next_to_d += prev == "D" or nxt == "D"
    lengths = np.array([(b - a) / FPS for a, b, _ in missed])
    return {"count": len(missed), "median_s": float(np.median(lengths)), "short": float(np.mean(lengths < SHORT / FPS)),
            "between_other": between / len(missed), "next_to_dead": next_to_d / len(missed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", choices=("mix", "v11"), default="mix")
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    args = parser.parse_args()
    rows, details = [], []
    for match in lab.each(args.which):
        pred = lab_mix.kept_letters(match.id, args.which) if args.labels == "mix" else match.v11
        pred = np.asarray(pred).astype("<U1")
        truth = match.truth_letters.astype("<U1")
        base = lab.score(match, pred)["spell_f1"]
        fixed, info = oracles(truth, pred)
        for kind, letters in fixed.items():
            rows.append({"match": match.id, "kind": kind, "gain": lab.score(match, letters)["spell_f1"] - base})
        details.append(where(info["missed"], truth))
        print("{}: {:.4f}".format(match.id, base), flush=True)
    table = pd.DataFrame(rows).groupby("kind", sort=False).gain.mean().sort_values(ascending=False)
    print("\nspellF1 gain if each kind of error were fixed (mean over the matches):")
    print(table.to_string(float_format="{:+.4f}".format))
    d = pd.DataFrame(details)
    print("\nmissed possessions per match: {:.0f}; median {:.1f} s; under {:.0f} s {:.0%}; between two of the other team {:.0%}; "
          "next to a stoppage {:.0%}".format(d["count"].mean(), d.median_s.median(), SHORT / FPS, d.short.mean(),
                                             d.between_other.mean(), d.next_to_dead.mean()))


if __name__ == "__main__":
    main()
