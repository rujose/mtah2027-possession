# Lab candidate `short`: short possessions at clear touches

| | |
|---|---|
| **status** | **built 2026-11-04, passes the bar on both sets: keep** (held out 0.5909 against v11's 0.5727, better on 10/10; IDSSE only 0.5834 against 0.5625, 7/7) |
| **what** | inside a possession of team O, where team N clearly touches the ball and O touches it again 0.5–5 s later, N gets those frames; also where N's clear touch is the last before the ball goes dead |
| **stage** | `post`: runs on any letters, after v11 here |
| **new code** | `bench/lab_short.py` (`apply`, `fit`, the grid, `python -m bench.lab_short [--set all \| idsse]`, ~10 s) |
| **results** | `bench/results/lab_short[_idsse].json`, `lab_short_grid[_idsse].csv` |
| **rules** | bench only, through `bench.lab`; to predict a match it reads only that match's damaged tracking and PathCRF's out-of-sample probabilities; the learned rule is trained on the other matches |

## Why

The biggest error left is **missed possessions** (+0.250 if all fixed, measured on v9). Re-measured
on v11: 1,425 of 4,114 true possessions (35 %) have no predicted one of their team at IoU ≥ 0.3;
740 of them sit between two possessions of the other team. Each one costs a true possession, and
the decoder can't produce them: its minimum possession is 0.8 s and its team switch has a price.

v11's page proposed v12: N's player within 2 m, O's nearest at least 1 m further, 1–3 s. A probe
gave about +0.003 (7/10). **Why so little, found here:** a "touch" is the frame where the ball's
velocity changes most while a player is within 2 m, and that frame is usually not where the player
is at the ball. Of the 897 loose candidates inside v11's possessions (any touch by N after one by O,
ended by O's next touch):

| N's player at N's touch | candidates | the truth gives most of the stretch to N |
|---|---|---|
| within 0.3 m | 89 | 70 % |
| 0.3–1 m | 97 | 61 % |
| 1–1.5 m | 64 | 23 % |
| **1.5–2 m** | **647** | **6 %** |

Most candidates are a ball passing an opponent at arm's length while O plays it. A margin over O's
nearest can't separate those: with N's player 1–2 m off, the truth gives N the stretch 13 % of the
time even when O's nearest is over 2 m further. **The distance of N's own player to the ball is the
clarity signal**; the margin adds little once it's in.

**Exploration before the grid was fixed** (all 10 matches, truth used, not leave-one-out; it shaped
the features and the grid, so it is in-sample in that sense):

| | per match | spellF1 change |
|---|---|---|
| every loose candidate written in (O-N-O, any length) | 89.7 | **−0.081** |
| only those the truth agrees with (oracle, O-N-O) | 17.5 | +0.030 |
| oracle, O-N-O and O-N-D | 23.1 of 103 | +0.035 |

So the ceiling of this candidate generator is about +0.035, and precision is everything.

## What it does

**Candidates** (`candidates`), for each run of O (H or A) in the input letters:
1. The touches in it (`possession/touches.py`, ΔV 3 m/s as v11), each given to its team; a frame
   that qualifies for both teams goes to the team whose nearest player is closer.
2. A group of consecutive touches by N, **after at least one touch by O in the run**, is a candidate.
   It runs from N's first touch to O's next touch (**O-N-O**), or, with `dead`, to the end of the run
   when N's touch is the last and a stoppage follows (**O-N-D**).
3. Only stretches 0.5–5 s long (or 1–3 s).

Writing a candidate in gives N frames *first..end−1*: N's possession starts on its touch and O's
resumes on its own touch, the truth's convention (v11's page: a possession changes hands at the new
team's touch). O always keeps at least the frames up to its touch before, so no possession is
swallowed; the number of possessions grows by 2 (O-N-O) or 1 (O-N-D) per candidate.

**Which candidates are clear** (`keep`), two rules:
- **touch**: N's player within **reach** of the ball at N's touch, O's nearest at least 1 m further,
  and optionally PathCRF's mean P(N) over the stretch at least **pc**.
- **learned**: a logistic regression (standardised features, sklearn defaults) predicting "the truth
  gives most of the stretch to N", trained on the candidates of the *other* bench matches (on their
  v11 letters), kept when its probability is at least **p_min**. Its features:

| feature | what it says | weight (standardised, mean of the 10 folds; min–max) |
|---|---|---|
| `d_new` | N's player's distance to the ball at the touch | **−0.95** (−1.05 to −0.86) |
| `v_after` | the ball's speed over 0.2 s after: a reception or interception stops it | **−0.78** (−0.85 to −0.65) |
| `dv` | the touch's change of velocity | **+0.70** (+0.55 to +0.78) |
| `pc` | PathCRF's mean P(N) over the stretch (log-odds; 0.5 where it didn't answer) | **+0.65** (+0.55 to +0.74) |
| `near_frac` | share of the stretch's seen frames where N's player is nearer than O's | +0.42 (+0.38 to +0.46) |
| `dead` | the stretch ends at a stoppage (O-N-D) | +0.33 (+0.27 to +0.41) |
| `seen_frac` | share of the stretch with the ball seen | +0.27 (+0.14 to +0.34) |
| `length` | the stretch's length | +0.15 (+0.04 to +0.22) |
| `cos` | cosine of the ball's change of direction at the touch | +0.12 (+0.03 to +0.20) |
| `margin` | O's nearest minus N's nearest at the touch | +0.04 (−0.02 to +0.12) |

The weights are stable across folds: the same story on every match.

## Every constant

| constant | value | why |
|---|---|---|
| ΔV | 3 m/s | v11's, so "a touch" means the same in both stages |
| touch look | 0.2 s either side | the detector's (`touches.LOOK`), also used for the ball's velocity before and after |
| O's touch first | required | a stretch with no O touch before it in the run is a timing error at the change (v11's job), not a missed possession |
| MARGIN (touch rule) | 1 m | v12's "clearly", unchanged: in a duel "the nearest team" is a coin toss |
| reach (grid) | 1, 0.5 m | "at the ball": half the detector's 2 m, and a quarter |
| pc (grid) | none, 0.3 | PathCRF at least not confidently against N. It already shaped v7's letters (live frames), so its P(N) inside O's possessions is low (median 0.1 over candidates) |
| length (grid) | 1–3 s, 0.5–5 s | 1–3 s is the missed possessions' quartiles (0.9 / 3.0 s); 0.5–5 s covers 91 % of them, and under 0.5 s the stretch is shorter than the touch detector's own look either side |
| dead (grid) | off, on | 48 % of missed possessions sit next to a stoppage; O-N-D is the one shape the touches can see (N's clearance or deflection out) |
| p_min (grid) | 0.5, 0.65, 0.8 | more likely right than wrong; about 2 to 1; 4 to 1 |
| label for training | most of the stretch is N's in the truth | the plain question "is this a possession of N?" |
| C | 1.0 | sklearn's default; ~680 rows (9 matches' candidates), 10 features: no tuning |

## The grid

23 settings, simplest first: the no-op; 16 touch-rule settings (dead × length × pc × reach); 6
learned settings (dead × p_min, length 0.5–5 s). Run once; not widened afterwards.

## Results (`python -m bench.lab_short`, held out)

| | held out | v11 | change | matches better |
|---|---|---|---|---|
| **all 10** | **0.5909** | 0.5727 | **+0.0182** | **10/10** |
| IDSSE (7) | 0.5832 | 0.5673 | +0.0159 | 7/7 |
| Metrica (3) | 0.6090 | 0.5855 | +0.0234 | 3/3 |
| **PathCRF-unseen (4)** | **0.6038** | 0.5842 | **+0.0196** | **4/4** |
| macroF1 | 0.8727 | 0.8705 | +0.0022 | |
| **`--set idsse` (7)** | **0.5834** | 0.5625 | **+0.0209** | **7/7** |
| idsse: macroF1 | 0.8785 | 0.8757 | +0.0028 | |
| idsse: PathCRF-unseen (J03WR9) | 0.5939 | 0.5790 | +0.0149 | 1/1 |

**Chosen on all, and by every fold of both sets:** learned, p_min 0.5, 0.5–5 s, dead on. All four
checks of the bar pass on both sets.

**Nested check** (the learned rule's setting chosen with models that never saw the held-out match
either, 90 fits): the same pick in every fold, the same 0.5909 (+0.0182, 10/10).

**Per match** (chosen setting; each match predicted by a model trained on the other 9):

| match | v11 | short | change | kept (O-N-O / O-N-D) | of them right | possessions v11 → short | true |
|---|---|---|---|---|---|---|---|
| J03WMX | 0.5100 | 0.5161 | +0.0061 | 11 / 2 | 54 % | 340 → 364 | 511 |
| J03WN1 | 0.5282 | 0.5526 | +0.0245 | 8 / 4 | 100 % | 405 → 425 | 500 |
| J03WOH | 0.5676 | 0.5811 | +0.0135 | 6 / 3 | 78 % | 345 → 360 | 402 |
| J03WOY | 0.6045 | 0.6263 | +0.0217 | 6 / 3 | 89 % | 381 → 396 | 456 |
| J03WPY | 0.6181 | 0.6462 | +0.0281 | 9 / 6 | 80 % | 351 → 375 | 421 |
| J03WQQ | 0.5624 | 0.5718 | +0.0094 | 11 / 5 | 44 % | 338 → 365 | 401 |
| J03WR9 (unseen) | 0.5801 | 0.5881 | +0.0081 | 1 / 4 | 60 % | 332 → 338 | 370 |
| metrica1 (unseen) | 0.5934 | 0.6197 | +0.0263 | 12 / 7 | 79 % | 314 → 345 | 354 |
| metrica2 (unseen) | 0.5811 | 0.6082 | +0.0271 | 10 / 3 | 85 % | 318 → 341 | 318 |
| metrica3 (unseen) | 0.5821 | 0.5990 | +0.0169 | 14 / 3 | 59 % | 366 → 397 | 381 |
| **mean** | **0.5727** | **0.5909** | **+0.0182** | **8.8 / 4.0** | **72 %** | **349 → 371** | **411** |

- **About 13 short possessions a match**, 1.5 s long (median), ~570 frames (0.4 % of a match).
- **True possessions matched** (IoU ≥ 0.3, same team): 269 → 281 a match. About one more true
  possession for each one written in: the right ones count, and the O-halves around them often
  match their own true possessions better.
- **It catches 62 % of the candidates the truth agrees with**, at 72 % precision.
- **Possessions per match:** 349 → 371 against 411 true. It closes a third of the gap, but on
  metrica2 and metrica3 it now predicts more possessions than the truth (341 vs 318, 397 vs 381),
  and still gains there.

**The whole grid, mean change against v11** (in sample, all 10; the `idsse` grid has the same order):

| rule | 1–3 s | 1–3 s + dead | 0.5–5 s | 0.5–5 s + dead |
|---|---|---|---|---|
| touch, reach 1 m | +0.0096 (9/10) | +0.0110 | +0.0130 | **+0.0150 (9/10)** |
| touch, reach 0.5 m | +0.0079 (10/10) | +0.0093 | +0.0112 | +0.0130 (10/10) |
| touch, reach 1 m, pc 0.3 | +0.0025 | +0.0028 | +0.0024 | +0.0028 |
| touch, reach 0.5 m, pc 0.3 | +0.0022 | +0.0025 | +0.0020 | +0.0023 |

| rule | O-N-O only | + dead |
|---|---|---|
| learned, p_min 0.8 | +0.0071 (9/10) | +0.0102 (8/10) |
| learned, p_min 0.65 | +0.0094 (10/10) | +0.0134 (9/10) |
| **learned, p_min 0.5** | +0.0143 (10/10) | **+0.0182 (10/10)** |

- **Every setting gains.** The reach is what was missing from the old probe.
- **A PathCRF threshold hurts:** it keeps about a sixth as many frames (51 against 311 a match at
  reach 1 m, 0.5–5 s, dead), good ones with the bad. As one feature among others it helps.
- **Longer windows and O-N-D both add,** in every row.
- **The simple rule passes too.** Judged on the touch rows alone (the same grid without the learned
  rule): touch, reach 1 m, 0.5–5 s, dead, chosen by every fold: held out 0.5877 (+0.0150), 9/10
  (J03WR9 −0.0005, one wrong insertion); IDSSE +0.0117 (6/7), Metrica +0.0226, unseen +0.0168
  (3/4); `--set idsse` +0.0145, 7/7. No training, no PathCRF: the fallback if the learned rule
  can't be mixed.
- **p_min 0.5 is the grid's lowest:** the gain still rises as p_min falls, so the best threshold may
  be lower. Not tried: the grid was fixed before the run.

## Failure modes

1. **Wrong insertions cost two possessions.** A wrong O-N-O splits O's possession in three: one
   spurious N, and O's halves may each lose its match. J03WQQ (44 % right) and J03WMX (54 %) still
   gained; a match where precision falls well under half would lose.
2. **Duels and passes past an opponent.** A ball played past N's player within 1 m with a change of
   speed looks like N's touch. The regression's `v_after` and `dv` weigh against it (a reception
   stops the ball), but it stays the main source of the 28 % wrong.
3. **High balls** (the dot is the ball's shadow): a ball landing next to N's player looks like a
   touch, exactly as in v11.
4. **Unseen touches:** a stretch needs N's touch and O's next touch seen (or the stoppage). Missed
   possessions with the ball missing, or with only one touch seen, are untouched: 38 % of the
   oracle's good candidates aren't kept, and most missed possessions never become candidates.
5. **PathCRF on our match** is weaker than on the 6 IDSSE matches it was trained on. The unseen
   matches gain +0.0196 (4/4), and the touch rule (no PathCRF) is the fallback.
6. **The other 48 % next to stoppages:** only O-N-D is handled. D-N-O (N plays the restart, O
   intercepts at once) had 18 loose candidates on the bench: not worth a rule.

## Mixing

```python
from bench import lab_short
model = lab_short.fit(training_matches, letters=lambda m: previous_stage(m), **lab_short.CHOSEN)
letters = lab_short.apply(match, letters, model=model, **lab_short.CHOSEN)
# CHOSEN = {"rule": "learned", "reach": None, "pc": None, "p_min": 0.5, "length": (0.5, 5.0), "dead": True}
# fallback, no model: apply(match, letters, rule="touch", reach=1.0, pc=None, length=(0.5, 5.0), dead=True)
```

- `fit` takes an iterable of `lab` matches (a generator is fine: one in memory at a time), never
  including the match it will predict. Train it on the letters the stage will actually see: the
  candidates come from the input's possession runs.
- **Expected interactions:** a decoder with a shorter minimum possession (the `decoder` candidate)
  goes after the same missed possessions, so the gains won't simply add; a stoppage stage changes
  which runs end in D (the O-N-D half); a stage that moves team changes moves the runs the
  candidates are cut from. Put this stage last, or refit it on each new input.
- **Not checked:** 30 % and 55 % damage (the lab caches only 41 %).

## Verdict

**Keep.** It clears every check of the bar on both sets, with all folds choosing the same setting,
and a larger gain than v9's or v11's post steps (+0.018, about twice v11's +0.0098). The exploration that
found the reach signal used the same 10 matches, so the true gain on our match is likely smaller;
the per-match and unseen checks (10/10, 4/4) and the training-free fallback (+0.015, 9/10) say it
isn't only the winner's curse.
