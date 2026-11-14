"""What does "owning the ball" mean in Metrica's corrected tracking? Read from its 3 public sample games.

    python -m bench.owner_rule      # ~10 min; bench/results/owner_rule.json, docs/owner-rule.md

The organisers' truth is "derived mechanically from corrected tracking": D where it marks the ball dead,
otherwise the team it names as owning it (CHALLENGE.md). We don't get the corrected tracking of our match,
but Metrica's sample games are corrected Metrica tracking. This script measures, on those 3 games only
(external data; nothing of our match is read):

1. holds: while a player keeps the ball between receiving a pass and making his next one (2 s or more),
   how far is the corrected ball from him, and does it bend with him?
2. a touch rule: the team of the player within `r` m of the ball owns it from that frame (a run of at least
   `k` frames), a loose ball stays with the last team. Scored against the games' event labels (our recipe,
   bench/metrica.py), whose dead ball it keeps, so only the owner convention differs.
3. a confirmed-touch rule: a touch takes the ball only if that team keeps it `hold` frames or its next touch
   is its own too.
4. v13 (branch A, held out, bench/lab_gate.py's letters) against each version, with possession counts on
   every bench match.
"""

import csv
import json
import os

import numpy as np

from bench import lab, metrica, sets
from bench.run import score as bench_score
from possession.data import FPS, HOME

# Params
GAMES = ("metrica1", "metrica2", "metrica3")
LET = np.array(["H", "A", "D", "X"])
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
HOLD_MIN = 2 * FPS                  # frames: a hold between a reception and the same player's next pass
TOUCH_GRID = [(r, k) for r in (0.2, 0.3, 0.5) for k in (1, 3, 5, 8, 12, 25)]
CONFIRM_GRID = [(r, h) for r in (0.2, 0.3, 0.5) for h in (5, 10, 25)]
V13_LETTERS = os.path.join(lab.ROOT, "data", "external", "lab_gate", "letters", "A_normal_{}.npy")
OUT = os.path.join(lab.RESULTS, "owner_rule.json")


def runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def nearest(ball, players, team):
    """(distance to the nearest player, his side 0 = home / 1 = away) per frame."""
    dist = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    dist = np.where(np.isnan(dist), np.inf, dist)
    return dist.min(1), np.where(np.asarray(team)[dist.argmin(1)] == HOME, 0, 1)


def contacts(close, side, a, b):
    """[(start, end, side)]: runs of frames in [a, b) with the ball within reach of one side's player."""
    out, t = [], a
    while t < b:
        if close[t]:
            s, tm = t, side[t]
            while t < b and close[t] and side[t] == tm:
                t += 1
            out.append((s, t, tm))
        else:
            t += 1
    return out


def relabel(truth, dmin, side, r, takes):
    """truth's D and X kept; inside each live stretch the owner changes at each contact `takes` accepts."""
    out = truth.copy()
    close = dmin < r
    for a, b in runs(np.isin(truth, (0, 1))):
        cs = contacts(close, side, a, b)
        changes = [(s, tm) for i, (s, e, tm) in enumerate(cs) if takes(cs, i)]
        cur, ci = (changes[0][1] if changes else 0), 0
        for t in range(a, b):
            while ci < len(changes) and changes[ci][0] <= t:
                cur = changes[ci][1]
                ci += 1
            out[t] = cur
    return out


def touch_rule(truth, dmin, side, r, k):
    return relabel(truth, dmin, side, r, lambda cs, i: cs[i][1] - cs[i][0] >= k)


def confirmed_rule(truth, dmin, side, r, hold):
    def takes(cs, i):
        s, e, tm = cs[i]
        return e - s >= hold or (i + 1 < len(cs) and cs[i + 1][2] == tm)
    return relabel(truth, dmin, side, r, takes)


def deviation(path):
    """The largest distance from the straight line between a path's ends (NaN if they're under 2 m apart)."""
    s, e = path[0], path[-1]
    v = e - s
    n = np.linalg.norm(v)
    if n < 2:
        return np.nan
    return float(np.max(np.abs((path - s) @ (np.array([-v[1], v[0]]) / n))))


def holds(game, ball, players):
    """Per hold of 2 s+ between a reception and the same player's next pass (games 1-2: CSV events with names)."""
    raw = os.path.join(metrica.RAW, os.path.basename(metrica.FILES[game][0]))
    rows = list(csv.DictReader(open(raw)))
    gap, bend_ball, bend_holder = [], [], []
    for r1, r2 in zip(rows, rows[1:]):
        if (r1["Type"].strip(), r2["Type"].strip()) != ("PASS", "PASS") or r1["Team"] != r2["Team"] or r1["To"] != r2["From"]:
            continue
        f1, f2 = int(r1["End Frame"]) - 1, int(r2["Start Frame"]) - 1
        if f2 - f1 < HOLD_MIN or np.isnan(ball[f1:f2 + 1, 0]).any():
            continue
        dist = np.linalg.norm(players[f1] - ball[f1], axis=1)
        k = int(np.argmin(np.where(np.isnan(dist), np.inf, dist)))
        if np.isnan(players[f1:f2 + 1, k, 0]).any():
            continue
        b, p = ball[f1:f2 + 1], players[f1:f2 + 1, k]
        gap.append(float(np.median(np.linalg.norm(b - p, axis=1))))
        bend_ball.append(deviation(b))
        bend_holder.append(deviation(p))
    gap, bb, bh = map(np.array, (gap, bend_ball, bend_holder))
    ok = ~np.isnan(bb) & ~np.isnan(bh)
    return {"holds": len(gap), "ball_to_holder_median_m": float(np.median(gap)), "ball_to_holder_p90_m": float(np.percentile(gap, 90)),
            "bend_ball_median_m": float(np.median(bb[ok])), "bend_holder_median_m": float(np.median(bh[ok])),
            "holder_bends_1m_more_share": float(np.mean(bh[ok] - bb[ok] > 1))}


def lengths(codes):
    """Possession lengths (s): runs of one team, a D or X ending them."""
    out, prev, n = [], None, 0
    for c in list(codes) + [None]:
        if c in (0, 1) and c == prev:
            n += 1
            continue
        if prev in (0, 1):
            out.append(n / FPS)
        prev, n = c, 1
    return np.array(out)


def main():
    report = {"games": {}, "v13": {}}
    for m in GAMES:
        d = sets.load(m)
        truth = np.asarray(sets.truth(m, d)).astype(int)
        ball, players = d["ball"][:, :2].astype(float), d["players"][:, :, :2].astype(float)
        dmin, side = nearest(ball, players, d["team"])
        live_seen = np.isin(truth, (0, 1)) & ~np.isnan(ball[:, 0])
        g = {"possessions_events": int(bench_score(truth, LET[truth])["possessions"]),
             "ball_within_m": {str(x): float(np.mean(dmin[live_seen] < x)) for x in (0.01, 0.05, 0.1, 0.3)}}
        if m != "metrica3":
            g["holds"] = holds(int(m[-1]), ball, players)
        live = np.isin(truth, (0, 1))
        g["touch_rule"] = []
        for r, k in TOUCH_GRID:
            lab_ = touch_rule(truth, dmin, side, r, k)
            s = bench_score(truth, LET[lab_])
            g["touch_rule"].append({"r": r, "k": k, "spell_f1": s["spell_f1"], "possessions": s["possessions"],
                                    "frames_agree": float(np.mean(lab_[live] == truth[live]))})
        g["confirmed_rule"] = []
        for r, h in CONFIRM_GRID:
            lab_ = confirmed_rule(truth, dmin, side, r, h)
            s = bench_score(truth, LET[lab_])
            g["confirmed_rule"].append({"r": r, "hold": h, "spell_f1": s["spell_f1"], "possessions": s["possessions"]})
        report["games"][m] = g
        print("{}: done".format(m), flush=True)
    for m in lab.matches("all"):
        match = lab.load(m, "all")
        v13 = np.load(V13_LETTERS.format(m)).astype(str)
        truth = np.asarray(match.truth).astype(int)
        tl, pl = lengths(truth), lengths([CODE[c] for c in v13])
        row = {"spell_f1": bench_score(truth, v13)["spell_f1"], "possessions_true": len(tl), "possessions_v13": len(pl),
               "under_2s_true": int((tl < 2).sum()), "under_2s_v13": int((pl < 2).sum())}
        if m in GAMES:
            d = sets.load(m)
            dmin, side = nearest(d["ball"][:, :2].astype(float), d["players"][:, :, :2].astype(float), d["team"])
            touch = touch_rule(truth, dmin, side, 0.2, 1)
            row["spell_f1_against_touch_rule"] = bench_score(touch, v13)["spell_f1"]
            row["possessions_touch_rule"] = len(lengths(touch))
        report["v13"][m] = row
        print("{}: {}".format(m, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print("wrote {}".format(os.path.relpath(OUT, lab.ROOT)))


if __name__ == "__main__":
    main()
