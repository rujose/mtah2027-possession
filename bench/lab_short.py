"""Lab candidate `short`: short possessions at clear touches by the other team (docs/lab/short.md).

    python -m bench.lab_short [--set all|idsse]     # ~1 min: the grid on every bench match, judged

Inside a possession of team O, where team N clearly touches the ball and O touches it again shortly
after, N gets the frames from its touch to O's (the truth's convention: a possession changes hands at
the new team's touch, docs/models/v11-touch-line.md). Optionally (`dead`), where N's clear touch is
the last one before the ball goes dead, N gets the frames from its touch to the stoppage.

"Clearly" is the whole problem: a loose version cost -0.14 to -0.20. A candidate is any group of N
touches (possession/touches.py, ΔV as v11) after a touch by O in O's possession, ended by O's next
touch (or by the stoppage). Two ways to decide which candidates are clear:
- "touch": N's player is at the ball (within REACH, tighter than the detector's 2 m), O's nearest is
  at least MARGIN further, and optionally PathCRF gives N at least PC over the stretch;
- "learned": a logistic regression on the candidate's evidence (distances, the ball's change of
  speed and direction, PathCRF, how long N stays nearest), trained on the other bench matches'
  candidates and their truth, and a minimum probability.

A post stage: `apply(match, letters, **setting)` works on any letters (H/A/D/X per frame) and reads
only the match's damaged tracking and PathCRF's out-of-sample probabilities, never its truth.
"""

import argparse

import numpy as np
import pandas as pd

from bench import chain, lab
from possession import touches
from possession.data import FPS

# Params (by reasoning; docs/lab/short.md gives the why of each)
STAGE = "post"
NAME = "short"
DV = 3.0               # m/s: v11's ΔV, so a "touch" means the same thing in both stages
MARGIN = 1.0           # m: O's nearest at least this much further than N's at N's touch (v12's "clearly")
LOOK = touches.LOOK    # frames either side of a touch for the ball's velocity (the detector's)
C = 1.0                # the logistic regression's inverse regularisation (sklearn's default)
OTHER = {"H": "A", "A": "H"}
SIDE = {"H": 0, "A": 1}
FEATURES = ["d_new", "margin", "dv", "cos", "v_after", "pc", "near_frac", "seen_frac", "length", "dead"]


def _grid():
    """Simplest first, the no-op first (a tie keeps v11)."""
    out = [{"rule": "none", "reach": None, "pc": None, "p_min": None, "length": None, "dead": False}]
    for dead in (False, True):
        for length in ((1.0, 3.0), (0.5, 5.0)):
            for pc in (None, 0.3):
                for reach in (1.0, 0.5):
                    out.append({"rule": "touch", "reach": reach, "pc": pc, "p_min": None, "length": length, "dead": dead})
    for dead in (False, True):
        for p_min in (0.5, 0.65, 0.8):
            out.append({"rule": "learned", "reach": None, "pc": None, "p_min": p_min, "length": (0.5, 5.0), "dead": dead})
    return out


GRID = _grid()
COLUMNS = ["rule", "reach", "pc", "p_min", "length", "dead"]
# Chosen on all 10 matches, and by every fold of both sets (bench/results/lab_short[_idsse].json)
CHOSEN = {"rule": "learned", "reach": None, "pc": None, "p_min": 0.5, "length": (0.5, 5.0), "dead": True}


def evidence(match):
    """What every setting reads, once per match: each team's nearest distance to the ball, each
    team's touches, the ball's velocity before and after each frame, and PathCRF's P(home)."""
    tr = match.track
    ball = tr["ball"]
    n = len(ball)
    before, after = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    before[LOOK:] = (ball[LOOK:] - ball[:-LOOK]) / (LOOK / FPS)
    after[:-LOOK] = (ball[LOOK:] - ball[:-LOOK]) / (LOOK / FPS)
    return {"near": {s: touches.nearest(tr["players"], tr["team"], ball, i) for s, i in SIDE.items()},
            "touch": {s: touches.touches(tr["players"], tr["team"], ball, i, DV) for s, i in SIDE.items()},
            "before": before, "after": after, "seen": ~np.isnan(ball[:, 0]), "p_home": match.pred["p_pathcrf"]}


def team_touches(ev, a, b):
    """[(frame, team)]: the touches seen in [a, b), a frame that qualifies for both teams (a duel)
    given to the team whose nearest player is closer."""
    near, found = ev["near"], ev["touch"]
    out = []
    for k in np.flatnonzero(found["H"][a:b] | found["A"][a:b]) + a:
        if found["H"][k] and found["A"][k]:
            out.append((int(k), "H" if near["H"][k] <= near["A"][k] else "A"))
        else:
            out.append((int(k), "H" if found["H"][k] else "A"))
    return out


def candidates(letters, halves, ev, length, dead):
    """[(first, end, team, dead)]: each group of `team`'s touches inside the other team's possession,
    after a touch by that team, to be given frames first..end-1. It ends at the other team's next
    touch, or (with `dead`) at the stoppage that follows. Only stretches `length` (s) long."""
    lo, hi = int(round(length[0] * FPS)), int(round(length[1] * FPS))
    out = []
    for _, first, last in halves:
        seg = letters[first:last + 1]
        bounds = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)] + first
        runs = [(int(bounds[i]), int(bounds[i + 1]), seg[bounds[i] - first]) for i in range(len(bounds) - 1)]
        for r, (a, b, own) in enumerate(runs):
            if own not in OTHER:
                continue
            new = OTHER[own]
            seen = team_touches(ev, a, b)
            i = 1                                    # N's stretch must follow a touch by O in this run
            while i < len(seen):
                if seen[i][1] != new:
                    i += 1
                    continue
                j = i
                while j < len(seen) and seen[j][1] == new:
                    j += 1
                k1 = seen[i][0]
                if j < len(seen):
                    k2, to_dead = seen[j][0], False  # O touches it again: O-N-O
                elif dead and r + 1 < len(runs) and runs[r + 1][2] == "D":
                    k2, to_dead = b, True            # N's touch is the last before the stoppage: O-N-D
                else:
                    break
                if lo <= k2 - k1 <= hi:
                    out.append((k1, k2, new, to_dead))
                i = j
    return out


def features(ev, cand):
    """One row of evidence per candidate (FEATURES)."""
    rows = []
    for k1, k2, new, to_dead in cand:
        old = OTHER[new]
        near = ev["near"]
        vb, va = ev["before"][k1], ev["after"][k1]
        p = ev["p_home"][k1:k2] if new == "H" else 1 - ev["p_home"][k1:k2]
        seen = ev["seen"][k1:k2]
        with np.errstate(invalid="ignore"):
            nearer = near[new][k1:k2] < near[old][k1:k2]
        rows.append({
            "d_new": near[new][k1],                                   # N's player: at the ball, or 2 m off
            "margin": near[old][k1] - near[new][k1],                  # how much further O's nearest is
            "dv": float(np.hypot(*(va - vb))),                        # the touch's change of velocity
            "cos": float(np.dot(vb, va) / (np.linalg.norm(vb) * np.linalg.norm(va) + 1e-9)),  # its change of direction
            "v_after": float(np.linalg.norm(va)),                     # a reception stops the ball
            "pc": float(np.nanmean(p)) if np.any(~np.isnan(p)) else np.nan,
            "near_frac": float(nearer[seen].mean()) if seen.any() else np.nan,   # N stays nearest
            "seen_frac": float(seen.mean()),
            "length": (k2 - k1) / FPS, "dead": float(to_dead)})
    return pd.DataFrame(rows, columns=FEATURES)


def design(f):
    """Features as the regression sees them: PathCRF as log-odds, missing values neutral."""
    x = f.copy()
    pc = x["pc"].fillna(0.5).clip(1e-3, 1 - 1e-3)
    x["pc"] = np.log(pc / (1 - pc))
    x["near_frac"] = x["near_frac"].fillna(0.5)
    x["cos"] = x["cos"].fillna(0.0)
    return x[FEATURES].to_numpy(float)


def training_rows(match, letters, length, dead, ev=None):
    """(features, label) for one match's candidates; label = the truth gives most of the stretch to N.
    For training on matches other than the one predicted only."""
    ev = evidence(match) if ev is None else ev
    cand = candidates(np.asarray(letters).astype(str), match.halves, ev, length, dead)
    label = np.array([np.mean(match.truth_letters[k1:k2] == new) > 0.5 for k1, k2, new, _ in cand], bool)
    return features(ev, cand), label


def fit_rows(parts):
    """A logistic regression on [(features, label), ...]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x = np.vstack([design(f) for f, _ in parts])
    y = np.concatenate([lab_ for _, lab_ in parts])
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000)).fit(x, y)


def fit(matches, rule="learned", length=(0.5, 5.0), dead=False, letters=None, **_):
    """The model for rule "learned", trained on `matches` (never the one it will predict).
    letters: match -> the letters the stage will see on it (default match.v11)."""
    if rule != "learned":
        return None
    letters = letters or (lambda m: m.v11)
    return fit_rows([training_rows(m, letters(m), length, dead) for m in matches])


def keep(ev, cand, rule, reach, pc, p_min, model):
    """bool per candidate: clear enough to write in."""
    if not cand:
        return np.zeros(0, bool)
    f = features(ev, cand)
    if rule == "touch":
        ok = (f["d_new"] <= reach) & (f["margin"] >= MARGIN)
        if pc is not None:
            ok &= f["pc"].fillna(0.0) >= pc
        return ok.to_numpy()
    if rule == "learned":
        return model.predict_proba(design(f))[:, 1] >= p_min
    raise ValueError(rule)


def apply(match, letters, rule="none", reach=None, pc=None, p_min=None, length=None, dead=False, model=None, ev=None):
    """letters with each clear short possession by the other team written in (rule "none": no-op)."""
    letters = np.asarray(letters).astype(str)
    if rule == "none":
        return letters.copy()
    ev = evidence(match) if ev is None else ev
    cand = candidates(letters, match.halves, ev, length, dead)
    out = letters.copy()
    for (k1, k2, new, _), ok in zip(cand, keep(ev, cand, rule, reach, pc, p_min, model)):
        if ok:
            out[k1:k2] = new
    return out


def run(which):
    """The grid on every match of the set, each "learned" model trained without the match it predicts."""
    learned = sorted({(s["length"], s["dead"]) for s in GRID if s["rule"] == "learned"})
    parts = {}
    for match in lab.each(which):
        ev = evidence(match)
        parts[match.id] = {key: training_rows(match, match.v11, *key, ev=ev) for key in learned}
    rows = []
    for match in lab.each(which):
        ev = evidence(match)
        models = {key: fit_rows([parts[m][key] for m in parts if m != match.id]) for key in learned}
        for setting in GRID:
            model = models.get((setting["length"], setting["dead"])) if setting["rule"] == "learned" else None
            letters = apply(match, match.v11, model=model, ev=ev, **setting)
            s = lab.score(match, letters)
            rows.append({"match": match.id, **{c: str(setting[c]) for c in COLUMNS},
                         "spell_f1": s["spell_f1"], "macro_f1": s["macro_f1"], "possessions": s["possessions"],
                         "true_possessions": s["true_possessions"], "frames_changed": int(np.sum(letters != match.v11))})
        print("{}: done".format(match.id), flush=True)
    grid = pd.DataFrame(rows)
    report = lab.judge(grid, COLUMNS, which)
    print(lab.describe(report))
    lab.save(NAME, grid, report, which)
    return grid, report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=chain.SETS, default="all", dest="which")
    args = parser.parse_args()
    run(args.which)


if __name__ == "__main__":
    main()
