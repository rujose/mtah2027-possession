# Lab candidate `recall`: missed possessions, proposed broadly and kept by a classifier

| | |
|---|---|
| **status** | **built 2026-11-04, lab round 2, on the mix. Passes the bar on `all`** (held out 0.6329 against the mix's 0.6156, **+0.0172, better on 9/10**; nested +0.0162, 9/10). **On `--set idsse` it fails one check:** +0.0078, 5/7, but its one unseen match, J03WR9, loses 0.0008. Verdict: keep, flagged (below) |
| **what** | inside each predicted possession of team O, three generators propose stretches for the other team N (N on the ball, round 1's touch candidates without their clarity check, PathCRF); a classifier trained on the other matches says which are true possessions, and those are written in |
| **stage** | `post`: runs on any letters, after the mix (stoppage → short → segments) here |
| **new code** | `bench/lab_recall.py` (`fit`, `apply`, the grid; `python -m bench.lab_recall [--set all \| idsse]`, ~1 min; `--nested`, ~1.5 min, 0.7 GB) |
| **results** | `bench/results/lab_recall[_idsse].json`, `lab_recall_grid[_idsse].csv`, `lab_recall_nested[_idsse].json` |
| **rules** | bench only, through `bench.lab`; to predict a match it reads that match's damaged tracking, the out-of-sample outputs on it (`match.pred`) and its input letters (the mix's, out of fold); the classifier is trained on the other matches' proposals and truth |

## Why

The mix's error budget (`python -m bench.lab_budget`): **missed possessions, +0.252 if all were
found**, five times the next error. 147 true possessions a match have no predicted possession of
their team at IoU ≥ 0.3. Round 1's `short` writes in only one shape (a touch by N between two by
O, or before a stoppage) through a clarity check tuned for precision, and `segments` then merges
some possessions away. What's left, measured first.

### What's missed (exploration: the mix's letters against the truth, all 10 matches)

| what the mix says over the missed possession | per match | median | N's player within 1 m and nearer than O's, ≥ 3 seen frames | a touch by N seen (ΔV 3 m/s, 2 m) | N within 2 m at any seen frame |
|---|---|---|---|---|---|
| **the other team's possession** (O-N-O, O-N-D inside O's run) | 65 | 1.4 s | 34 % | 60 % | 66 % |
| **a stoppage** (the mix's D covers it) | 42 | 2.0 s | 34 % | 41 % | 51 % |
| **its own team, but merged with or split from another** | 40 | 1.8 s | 51 % | 63 % | 69 % |

- The third row is mostly the shadow of the first: when N O N in the truth is one N in the mix,
  the O between is missed (row 1) and each N half may miss too (row 3). Writing the O in fixes all
  three, so a generator that looks inside both teams' possessions reaches both rows.
- **What's visible:** 38.5 % of all missed possessions have N on the ball (1 m, nearer than O, 3
  seen frames); 62 % have N within 2 m at some seen frame. The rest are invisible in this damage
  (ball or players not seen). So the realistic ceiling is well under the +0.252.

### What proposals can reach (oracle: write only the proposals that match a true possession)

| generator (inside the mix's possessions) | proposals per match | of them right (IoU ≥ 0.3) | ceiling |
|---|---|---|---|
| **N on the ball** (1 m, ≥ 1 frame) | 123 | 19.3 | +0.047 |
| N on the ball, ≥ 3 frames | 89 | 16.0 | +0.040 |
| N on the ball, 1.5 m | 197 | 20.9 | +0.044 |
| short's touch candidates, loose (0.2–8 s, O-N-O and O-N-D) | 84 | 13.4 | +0.032 |
| PathCRF > 0.5 | 41 | 6.4 | +0.013 |
| **all three** | **249** | **27.2** | **+0.064** |

The base rate is low (11 % right), so precision is everything; but the union's ceiling is almost
twice short's (+0.035, on v11) and it reaches possessions short can't (no touch seen, D-N-O). By
shape (N on the ball, 1 m): O-N-D proposals are right 36 % of the time, O-N-O 13 %, D-N-O 4 %.
This exploration used the truth of all 10 matches and chose the generators' constants (1 m, ≥ 1
frame, the union): in-sample in that sense, as short's was.

## What it does

### Proposals (`proposals`), inside each run of O (H or A) in the input letters, per half

1. **`ctl`, N on the ball.** A frame is "N on the ball" when the ball is seen, N's nearest player
   is within **1 m** and nearer than O's. Over the run, take the frames where either team is on the
   ball, in order; each group of consecutive N frames is a proposal from N's first frame to O's next
   one (**O-N-O**); to the run's end if O never comes back and a stoppage follows (**O-N-D**); from
   the run's start if N is first and the run follows a stoppage (**D-N-O**: N restarts, O wins it).
2. **`touch`:** `lab_short.candidates` (a group of N touches after an O touch, to O's next touch or
   the stoppage), with any length from 0.2 to 8 s and no clarity check.
3. **`pc`:** a stretch of at least 0.3 s where PathCRF gives N more than half (feature set `all` only).

Common to all: 0.2–8 s long; at the head only after a stoppage and at the tail only before one (at
a change of team it's the change's timing, v11's job); never the whole run (a wrong team,
segments' job). The same stretch from several generators is one proposal with several sources.

### Features (`table`; prediction-time inputs only)

| feature | what |
|---|---|
| `log_len`, `head`, `tail` | its length; D-N-O, O-N-D (else O-N-O) |
| `src_ctl`, `src_touch`, `src_pc` | which generators proposed it |
| `left_rem`, `right_rem`, `log_run` | O's frames left before and after it; O's run length |
| `ctl_n`, `ctl_o`, `ctl_share` | frames with N on the ball, with O on the ball; N's share of the seen frames |
| `tight_n`, `min_n`, `med_n`, `med_o`, `near_share` | frames with N within 0.5 m; N's closest and median distance; O's median; share of seen frames N is nearer |
| `seen` | share of frames with the ball seen |
| `touch_n`, `touch_o`, `touch_first`, `dv`, `v_after`, `cos` | touches by each team; one by N at its start; N's first touch's change of velocity, the ball's speed after, the change of direction |
| `pass_n`, `pass_o` | pass starts by each team (v9's detector) |
| `speed` | the ball's mean speed |
| `pdead_in`, `pdead_after` | P(dead) inside it and in the 1 s after |
| `edge_dist` | the last seen ball's distance to the nearest pitch line |
| `pc_mean`, `pc_share` | PathCRF's P(N): mean, share over 0.5 (`all` only) |
| `margin_mean`, `margin_sum` | the decoder's cost of O minus N (v7's for `all`, v6's, which has no PathCRF clue, for `no_pathcrf`) |

**Label** (truth of the training matches only): a true possession of N matches the proposal at
IoU ≥ 0.3, the question the metric asks of it.

### The classifier and the action

- **`hgb`:** a `HistGradientBoostingClassifier` with segments' fixed settings (200 rounds, rate
  0.05, 8 leaves, ≥ 40 rows a leaf, L2 1.0, seed 0); **`logistic`:** median imputation, standard
  scaling, a logistic regression at sklearn's defaults (as short). Neither tuned. Each fold trains
  on ~2,150 proposals (~340 right) from the other 9 matches' input letters.
- **Writing in** (`act`): from the most likely down, every proposal at or above **p_min** whose
  frames are all still O's and that keeps 1 frame from one already written becomes N's. A proposal
  inside O's run leaves O both halves (the number of possessions grows by 2; by 1 at a stoppage).

## Every constant

| constant | value | why |
|---|---|---|
| on the ball (`CTL`) | 1 m | short's finding: with N's player within 1 m at the touch the truth agrees 61–70 %, at 1.5–2 m 6 %; the exploration: 1.5 m adds 60 % more proposals for 1.6 more right a match and a lower ceiling |
| frames on the ball | ≥ 1 | ≥ 3 loses 3 right a match (ceiling +0.040 against +0.047); the classifier sees the count |
| "at the feet" (`TIGHT`) | 0.5 m | a feature only: short's tighter reach |
| length | 0.2–8 s | under 0.2 s is shorter than the touch detector's own look either side; 90 % of the missed possessions are under 4.8 s |
| PathCRF stretch | ≥ 0.3 s | shorter is flicker |
| touch candidates | 0.2–8 s, O-N-D on | short's, loosened: the classifier does the clarity check |
| P(dead) after | 1 s | an O-N-D ends at the stoppage, so P(dead) should rise right after it |
| label IoU | 0.3 | the lowest the metric counts: below it the written possession never matches |
| gap between writes | 1 frame | two N's with a 1-frame O between them would make a blip |
| distance cap | 30 m | segments' |
| **p_min (grid)** | **0.8, 0.7, 0.6, 0.5, 0.4** | break-even, roughly, by the metric's arithmetic (segments.md): a right O-N-O gains about (2·1.5 − 0.6·2) and a wrong one costs about (2·0.5 + 0.6·2), so about 0.55; an O-N-D adds one possession, not two, so about 0.4 |
| **model (grid)** | **logistic, hgb** | the simple one and segments' |
| **features (grid)** | **no_pathcrf, all** | PathCRF is better on the 6 IDSSE matches it was pretrained on |

## The grid

21 settings, simplest first: the no-op, then for `no_pathcrf` and `all`, for `logistic` and `hgb`,
p_min from 0.8 down to 0.4. Fixed before any held-out run; run once, not widened afterwards.

## Results (`python -m bench.lab_recall`, held out, against the mix)

### All 10 matches

| | held out | the mix | change | matches better |
|---|---|---|---|---|
| **all 10** | **0.6329** | 0.6156 | **+0.0172** | **9/10** |
| IDSSE (7) | 0.6123 | 0.5967 | +0.0156 | 6/7 |
| Metrica (3) | 0.6809 | 0.6599 | +0.0210 | 3/3 |
| **PathCRF-unseen (4)** | **0.6690** | 0.6499 | **+0.0190** | **4/4** |
| macroF1 | 0.8784 | 0.8775 | +0.0009 | |

**Chosen on all, and by every fold:** features `all`, `hgb`, p_min 0.4. All four checks of the
bar pass. Against v11 (0.5727) that is +0.0601, better on 10/10.

**Nested check** (`--nested`: for each held-out match m and each other match j, the classifiers
trained without m and j, j scored for every setting, m given the setting best over those; 360 fits,
~1.5 min, 0.66 GB):

| | nested | judge |
|---|---|---|
| **all 10** | **0.6318 (+0.0162), 9/10** | +0.0172, 9/10 |
| IDSSE | 0.6108 (+0.0141), 6/7 | +0.0156, 6/7 |
| Metrica | 0.6809 (+0.0210), 3/3 | +0.0210, 3/3 |
| PathCRF-unseen | 0.6690 (+0.0190), 4/4 | +0.0190, 4/4 |
| macroF1 | 0.8783 | 0.8784 |
| picks | `all`, hgb, 0.4 on 8 folds; `no_pathcrf`, hgb (0.5 on J03WMX, 0.4 on J03WQQ) | `all`, hgb, 0.4 on every fold |

It would pass the bar too.

**Per match** (chosen setting; each match predicted by a classifier trained on the other 9):

| match | the mix | recall | change | written (right) | possessions mix → recall | true |
|---|---|---|---|---|---|---|
| J03WMX | 0.5025 | 0.5202 | +0.0177 | 30 (12) | 285 → 341 | 511 |
| J03WN1 | 0.5770 | 0.6011 | +0.0241 | 30 (17) | 334 → 387 | 500 |
| J03WOH | 0.5992 | 0.6254 | +0.0262 | 12 (9) | 309 → 331 | 402 |
| J03WOY | 0.6277 | 0.6526 | +0.0248 | 30 (19) | 308 → 358 | 456 |
| J03WPY | 0.6640 | 0.6725 | +0.0086 | 21 (12) | 317 → 351 | 421 |
| J03WQQ | 0.5863 | 0.5811 | **−0.0052** | 15 (6) | 316 → 341 | 401 |
| J03WR9 (unseen) | 0.6202 | 0.6332 | +0.0131 | 11 (6) | 275 → 292 | 370 |
| metrica1 (unseen) | 0.6662 | 0.6962 | +0.0299 | 25 (15) | 283 → 324 | 354 |
| metrica2 (unseen) | 0.6671 | 0.6722 | +0.0051 | 22 (11) | 290 → 330 | 318 |
| metrica3 (unseen) | 0.6463 | 0.6743 | +0.0281 | 25 (14) | 306 → 348 | 381 |
| **mean** | **0.6156** | **0.6329** | **+0.0172** | **22.1 (12.1: 55 %)** | **302 → 340** | **411** |

- **What's written** (221 in all): 159 O-N-O (54 % right, median 1.8 s), 61 O-N-D (57 %, 1.5 s),
  1 D-N-O. By source: N on the ball 118 (59 % right), touch 62 (55 %), both 4, PathCRF alone 37
  (41 %). "Right" is IoU ≥ 0.3 with a true possession of N.
- **Missed possessions: 147 → 128 a match** (−18), more than the 12 right writes: splitting O's
  possession around a right one lets O's halves match their own true possessions.
- **Possessions per match 302 → 340** against 411 true: most of the way back to v11's 349 after
  segments took them down, and more of them right. metrica2 now predicts more than the truth (330 against 318)
  and still gains.
- **Held-out AUC** (right vs not, per match): `all` hgb 0.865, logistic 0.827; `no_pathcrf` hgb
  0.847, logistic 0.848.
- **Calibration:** written proposals are right 46 % at P 0.4–0.5, 57 % at 0.5–0.8, 67 % above 0.8.

**The whole grid, mean change against the mix** (in sample, all 10; matches better in brackets):

| p_min | `no_pathcrf` logistic | `no_pathcrf` hgb | `all` logistic | **`all` hgb** |
|---|---|---|---|---|
| 0.8 | +0.0028 (10) | +0.0016 (7) | +0.0052 (8) | +0.0015 (6) |
| 0.7 | +0.0054 (9) | +0.0057 (8) | +0.0063 (8) | +0.0060 (8) |
| 0.6 | +0.0091 (9) | +0.0094 (10) | +0.0059 (8) | +0.0119 (10) |
| 0.5 | +0.0108 (9) | +0.0127 (10) | +0.0100 (7) | +0.0159 (10) |
| 0.4 | +0.0118 (8) | +0.0147 (9) | +0.0098 (7) | **+0.0172 (9)** |

- **Every setting gains** on the mean; the gain rises as p_min falls in every column but `all`
  logistic.
- **Without PathCRF** (the no-op plus the `no_pathcrf` rows alone, judged the same way; a
  diagnostic, the verdict is the full grid's): held out 0.6304 (+0.0147), 9/10; IDSSE +0.0132
  (6/7), Metrica +0.0184, unseen +0.0174 (4/4); chosen `no_pathcrf`, hgb, 0.4. **Passes**: the
  fallback if PathCRF's probabilities aren't available or trusted on our match.

### IDSSE only (`--set idsse`, the mix's IDSSE letters, 6 training matches a fold)

| match | the mix | recall | change | picked | written (right) |
|---|---|---|---|---|---|
| J03WMX | 0.5148 | 0.5254 | +0.0105 | all, hgb, 0.5 | 21 (12) |
| J03WN1 | 0.5757 | 0.5926 | +0.0169 | all, hgb, 0.4 | 21 (11) |
| J03WOH | 0.5953 | 0.6154 | +0.0201 | all, hgb, 0.4 | 9 (7) |
| J03WOY | 0.6534 | 0.6643 | +0.0109 | all, hgb, 0.4 | 17 (10) |
| J03WPY | 0.6686 | 0.6701 | +0.0015 | all, hgb, 0.4 | 14 (7) |
| J03WQQ | 0.5704 | 0.5660 | −0.0045 | all, hgb, 0.4 | 15 (6) |
| **J03WR9 (unseen)** | 0.6223 | 0.6215 | **−0.0008** | all, hgb, 0.4 | 10 (4) |
| **mean** | **0.6001** | **0.6079** | **+0.0078, 5/7** | | **15.3 (8.1: 53 %)** |

- **Bar:** mean ✓, 5/7 ✓ (the minimum), each source ✓, **unseen ✗** (J03WR9 −0.0008). **Fails.**
- macroF1 0.8823 against 0.8816. Held-out AUC lower with 6 training matches: `all` hgb 0.816,
  `no_pathcrf` hgb 0.783.
- **Nested:** 0.6072 (+0.0071), 5/7; J03WR9 −0.0008 again. Fails the same check.
- **J03WR9 is noise around zero here:** across the 20 settings it moves from −0.0068 to +0.0041;
  `all` hgb at 0.5–0.8 gains +0.003 to +0.004 on it. On `all` (classifier trained on 9 matches,
  Metrica's 3 among them) the same match gains +0.0131.
- `no_pathcrf` alone on IDSSE: +0.0073, 4/7, J03WR9 −0.0046: fails too. With IDSSE alone,
  PathCRF's sharper probabilities help, not hurt.

## Failure modes

1. **Wrong insertions split a correct possession.** 45 % of what's written doesn't match a true
   possession at IoU ≥ 0.3. J03WQQ, the one
   match that loses (−0.005), has the fewest right proposals (17 of 197) and its writes are right
   6 of 15.
2. **p_min 0.4 is the grid's lowest,** and every fold picks it: the gain still rises as p_min falls
   (in sample +0.0159 at 0.5, +0.0172 at 0.4), and the classifier isn't sharp at the top (right
   46 % at 0.4–0.5, 67 % above 0.8). The best threshold may be lower; not tried, the grid was fixed.
3. **What the damage hides stays missed.** 128 missed possessions a match remain; most have the
   ball or N's players unseen. Missed possessions inside the mix's stoppages (42 a match) get no
   proposals at all: none of the generators looks inside D.
4. **PathCRF's proposals are the weakest** (written 37 times, 41 % right). The `no_pathcrf` rows
   alone pass the bar too (below), and the unseen matches gain as much as the others.
5. **Stacking leak, as in the mix:** the input letters of each training match were made by short
   and segments models trained with the held-out match among their 9. lab_mix's chains have the
   same second-order leak; the nested check removes only this stage's.
6. **Only 41 % damage** (the lab caches no other level).

## Mixing

```python
from bench import lab_recall as R
model = R.fit(training_matches, letters_of=lambda m: previous_stage(m), **R.CHOSEN)   # never the predicted match
letters = R.apply(match, letters, model, **R.CHOSEN)
# CHOSEN = {"features": "all", "model": "hgb", "p_min": 0.4}
# without PathCRF: R.fit(..., features="no_pathcrf", model="hgb"); R.apply(..., p_min=0.4)
```

- `STAGE = "post"`, `GRID` (21 settings, no-op first), `COLUMNS = ["features", "model", "p_min"]`.
- `fit(matches, letters_of, features, model)`: any iterable of `lab` matches, each read once; pass
  `letters_of` explicitly (its default is the mix's out-of-fold letters on `all`). Train it on the
  letters the stage will actually see: the proposals are cut from the input's possessions, and
  short's candidates already written in are no longer proposals.
- `apply(match, letters, model, p_min)`: reads the match's tracking, `pred` (`p_pathcrf`, `p_dead`,
  `v7_cost`, or `v6_cost` for `no_pathcrf`) and the letters, never its truth. Per-frame evidence is
  cached on the match object (`match._recall_evidence`).
- **Expected interactions:** it goes after the same possessions as short (its touch generator is
  short's candidates) and undoes some of segments' merges, so it belongs after both; putting
  segments after it again would merge its writes away unless refitted on its letters. A decoder
  with a shorter minimum possession would compete for the same frames.
- **Our match needs `v7_cost`** (as segments does): `possession.v7.predict` doesn't return it yet.
  Only 41 % damage was tried.

## Verdict

**Keep, flagged.** On `all` it clears every check of the bar with all folds choosing the same
setting, the nested check holds (+0.0162, 9/10), the gain is the largest of any single stage so far
(+0.017 on top of the mix), and it holds where it matters for our match: Metrica +0.021 (3/3) and
the PathCRF-unseen matches +0.019 (4/4); the PathCRF-free version passes too (+0.0147, 9/10).
**It does not clear the bar on `--set idsse`:** +0.0078 and 5/7, but the set's only unseen match
loses 0.0008, which fails the unseen check (nested: the same). That check rests on one match and
the loss is tiny, but round 1 kept nothing that failed a check on either set, so the lead should
decide whether `idsse` binds for a round-2 stage; if it does, this is a discard. The exploration
that chose the generators used the same 10 matches, and p_min 0.4 is the grid's edge, so expect
less on our match.
