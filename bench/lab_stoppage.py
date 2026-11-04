"""Lab candidate `stoppage`: each stoppage starts when the ball is seen leaving the pitch (docs/lab/stoppage.md).

    python -m bench.lab_stoppage [--set all|idsse]   # ~1 min; after `python -m bench.lab --build`

Idea 10e, v9's mirror. After any version's labels, each change from a team to D moves to a frame
OFFSET frames after the ball is seen crossing a pitch line (the truth's convention, measured here),
when such an exit is seen from BACK seconds before the change to AHEAD seconds after, and inside the
two runs the change separates. The team never changes, no possession or stoppage is created or
removed, and with no exit seen nothing moves.

1. The truth's convention: at every true team -> D change, the first seen exit near it and the lag
   (frames from the exit to the change); the offset is that lag's median, fitted leave one match out.
2. Every setting of GRID on every match, from match.v11; one row per (match, setting).
3. lab.judge against v11; the report is printed and saved (bench/results/lab_stoppage[_idsse].*).

What the rule reads on a match: only its damaged ball and the letters it is given. The truth is read
only in `fit` and `convention`, on the other matches, and for scoring.
"""

import argparse

import numpy as np
import pandas as pd

from bench import lab
from possession.data import FPS

# Params
STAGE = "post"
# (s before, s after) the change to look for the exit; the no-op first, then from narrow to wide.
GRID = [{"back": 0.0, "ahead": 0.0},
        {"back": 1.0, "ahead": 1.0},
        {"back": 2.0, "ahead": 1.0},
        {"back": 2.0, "ahead": 2.0},
        {"back": 4.0, "ahead": 2.0}]
COLUMNS = ["back", "ahead"]
STAY = int(0.4 * FPS)   # frames after an exit with the ball never seen back inside: not jitter on the line
NEAR = (FPS, FPS // 2)  # frames before / after a true change to look for its exit when measuring the lag
OFFSET = 3              # frames from the exit to the change, measured on all 10 bench matches (for our match)


def outside(ball, pitch):
    """bool [frames]: the ball is seen beyond a pitch line (no margin: the truth's D starts at the line)."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < 0) | (x > pitch[0]) | (y < 0) | (y > pitch[1])


def seen_exits(ball, pitch):
    """bool [frames]: the ball is seen outside at k, was seen inside at k - 1, and isn't seen inside
    again in the STAY frames after k."""
    seen = ~np.isnan(ball[:, 0])
    out = outside(ball, pitch) & seen
    inside = seen & ~out
    exits = np.zeros(len(ball), bool)
    exits[1:] = out[1:] & inside[:-1]
    ahead = np.r_[np.convolve(inside.astype(np.int16), np.ones(STAY, np.int16), "full"), 0]
    back_in = ahead[STAY:STAY + len(ball)] > 0     # seen inside somewhere in k + 1 .. k + STAY
    return exits & ~back_in


def team_to_dead(letters, first, last):
    """Frames g in (first, last] where the letters go from a team to D."""
    a, b = letters[first:last], letters[first + 1:last + 1]
    return np.flatnonzero(np.isin(a, ["H", "A"]) & (b == "D")) + first + 1


def lags(match):
    """For each true team -> D change with a seen exit from NEAR[0] frames before to NEAR[1] after:
    frames from the first such exit to the change (the truth's convention, on the damaged ball)."""
    exits = seen_exits(match.track["ball"], match.pitch)
    out = []
    for _, first, last in match.halves:
        for g in team_to_dead(match.truth_letters, first, last):
            found = np.flatnonzero(exits[max(g - NEAR[0], first):g + NEAR[1] + 1])
            if len(found):
                out.append(int(g - (max(g - NEAR[0], first) + found[0])))
    return out


def fit(matches, **setting):
    """{"offset": median lag (frames) over `matches`}: lab match objects or ids (loaded one at a time)."""
    pooled = []
    for m in matches:
        pooled += lags(lab.load(m) if isinstance(m, str) else m)
    return {"offset": int(round(float(np.median(pooled))))}


def run_bounds(letters, g, first, last):
    """(first frame of the team run ending at g - 1, end (exclusive) of the D run starting at g)."""
    before = np.flatnonzero(letters[first:g] != letters[g - 1])
    start = first + (before[-1] + 1 if len(before) else 0)
    after = np.flatnonzero(letters[g:last + 1] != "D")
    end = g + (after[0] if len(after) else last + 1 - g)
    return start, end


def apply(match, letters, back=0.0, ahead=0.0, offset=None, model=None):
    """Letters with each team -> D change moved to OFFSET frames after the first exit seen from `back`
    s before it to `ahead` s after, inside the two runs it separates. `model` (from fit) or `offset`
    overrides OFFSET."""
    return move(match, letters, back, ahead, offset, model)[0]


def move(match, letters, back=0.0, ahead=0.0, offset=None, model=None):
    """(letters, [(change, new change)])."""
    letters = np.asarray(letters).astype("<U1")
    if back <= 0 and ahead <= 0:
        return letters.copy(), []
    offset = model["offset"] if model else (OFFSET if offset is None else offset)
    exits = seen_exits(match.track["ball"], match.pitch)
    out = letters.copy()
    moves = []
    for _, first, last in match.halves:
        for g in team_to_dead(letters, first, last):
            team = letters[g - 1]
            start, end = run_bounds(letters, g, first, last)
            lo, hi = max(g - int(back * FPS), start + 1), min(g + int(ahead * FPS), end - 1)
            found = np.flatnonzero(exits[lo:hi + 1])
            if len(found) == 0:
                continue
            t = lo + found[0] + offset           # the earliest: the ball leaves play once per stoppage
            if start < t < g and np.all(out[t:g] == team):
                out[t:g] = "D"                   # the team keeps at least its first frame
            elif g < t < end and np.all(out[g:t] == "D"):
                out[g:t] = team                  # the stoppage keeps at least its last frame
            else:
                continue
            moves.append((int(g), int(t)))
    return out, moves


def timing(moves, letters, truth):
    """For moves whose change matches a true team -> D change (same team before, within 5 s): the
    distance (s) to it before and after the move."""
    true = team_to_dead(truth, 0, len(truth) - 1)
    before, after = [], []
    for g, t in moves:
        near = true[(np.abs(true - g) <= 5 * FPS) & (truth[true - 1] == letters[g - 1])]
        if len(near):
            f = near[np.argmin(np.abs(near - g))]
            before.append(abs(g - f) / FPS)
            after.append(abs(t - f) / FPS)
    return before, after


def run(which):
    ms = lab.matches(which)
    lag = {}
    for m in ms:                                 # pass 1: the convention on each match
        lag[m] = lags(lab.load(m, which))
    pooled = np.concatenate([lag[m] for m in ms])
    print("truth: {} team -> D changes with a seen exit nearby; frames from the exit to the change: median {:.0f}, "
          "quartiles {:.0f} / {:.0f}, 10-90 % {:.0f} / {:.0f}".format(len(pooled), np.median(pooled),
                                                                     *np.percentile(pooled, [25, 75, 10, 90])))
    rows = []
    diag = {(g["back"], g["ahead"]): {"moves": 0, "earlier": 0, "later": 0, "before": [], "after": []} for g in GRID}
    offsets = {}
    for m in ms:                                 # pass 2: every setting, with the offset fitted without m
        match = lab.load(m, which)
        offsets[m] = int(round(float(np.median(np.concatenate([lag[o] for o in ms if o != m])))))
        for setting in GRID:
            letters, moves = move(match, match.v11, offset=offsets[m], **setting)
            s = lab.score(match, letters)
            rows.append({"match": m, **setting, "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"],
                         "d_f1": s["d_f1"], "moved": len(moves)})
            d = diag[(setting["back"], setting["ahead"])]
            d["moves"] += len(moves)
            d["earlier"] += sum(t < g for g, t in moves)
            d["later"] += sum(t > g for g, t in moves)
            b, a = timing(moves, match.v11, match.truth_letters)
            d["before"] += b
            d["after"] += a
        print("{}: offset {} frames; {}".format(m, offsets[m], ", ".join(
            "{:g}/{:g} s {:.4f} ({} moved)".format(r["back"], r["ahead"], r["spell_f1"], r["moved"])
            for r in rows[-len(GRID):])), flush=True)
    grid = pd.DataFrame(rows)
    for (back, ahead), d in diag.items():
        if d["moves"]:
            b, a = np.array(d["before"]), np.array(d["after"])
            print("{:g}/{:g} s: {} moves, {} earlier, {} later; of the {} matching a true change, within 0.2 s {:.0%} -> {:.0%}, "
                  "median {:.2f} s -> {:.2f} s".format(back, ahead, d["moves"], d["earlier"], d["later"], len(b),
                                                       np.mean(b <= 0.2), np.mean(a <= 0.2), np.median(b), np.median(a)))
    means = grid.groupby(COLUMNS, sort=False)[["spell_f1", "macro_f1", "d_f1"]].mean()
    print("means over the matches (not held out):\n" + means.to_string(float_format="%.4f"))
    report = lab.judge(grid, COLUMNS, which)
    report["offsets"] = offsets
    report["d_f1"] = {"held_out": float(np.mean([grid[(grid.match == f["match"]) & np.all(
        [grid[c] == f["picked"][c] for c in COLUMNS], axis=0)].d_f1.iloc[0] for f in report["folds"]])),
        "base": float(grid[(grid.back == 0) & (grid.ahead == 0)].d_f1.mean())}
    print(lab.describe(report))
    print("  D's F1 held out {:.4f} vs v11 {:.4f}".format(report["d_f1"]["held_out"], report["d_f1"]["base"]))
    lab.save("stoppage", grid, report, which)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("all", "idsse"), default="all", dest="which")
    args = parser.parse_args()
    report = run(args.which)
    if args.which == "all" and report["passes"]:
        print("\npasses on all: the same on IDSSE alone")
        run("idsse")


if __name__ == "__main__":
    main()
