# Lab candidate `segments`: possession-level clean-up (wrong team, spurious possessions)

| | |
|---|---|
| **status** | **built 2026-11-04, lab round 1; passes the bar on `all` and on `idsse`** (verdict: keep) |
| **what** | each predicted possession is judged as one object: two classifiers, trained on the other bench matches, give P(its team is wrong) and P(it's spurious); possessions above a threshold are flipped to the other team or merged into their neighbours |
| **stage** | `post`: after v11, on any input letters |
| **code** | `bench/lab_segments.py` (new file only; nothing else changed) |
| **run** | `python -m bench.lab_segments [--set all\|idsse]` (~30 s); `--oracle` (~10 s, the ceiling with the truth's answers); `--nested` (~10 min, a stricter check of the same grid) |
| **results** | `bench/results/lab_segments[_idsse].json` (the judge's report, plus flips/merges alone, held-out AUCs and actions), `lab_segments_grid[_idsse].csv` (every match × setting) |
| **new dependencies** | none (sklearn's `HistGradientBoostingClassifier`, already installed) |

## Results (summary)

| | **all 10, held out** | IDSSE only (`--set idsse`, 7) |
|---|---|---|
| **spellF1** | **0.5873 against v11's 0.5727 (+0.0146), better on 7/10** | **0.5725 against 0.5625 (+0.0100), 6/7** |
| IDSSE | 0.5708 against 0.5673 (+0.0035), 4/7 | (as above) |
| Metrica | 0.6259 against 0.5855 (**+0.0404**), 3/3 | — |
| **PathCRF-unseen** (J03WR9, metrica1–3) | 0.6211 against 0.5842 (**+0.0369**), 4/4 | J03WR9: 0.6018 against 0.5790 (+0.0228) |
| macroF1 | 0.8723 against 0.8705 | 0.8769 against 0.8757 |
| **chosen on all the matches** | **features `all`, flip 0.6, merge 0.7, mode `dead`** | features `no_pathcrf`, flip 0.8, merge 0.7, mode `dead` |
| possessions per match | 349 → **281** (true 411) | 348 → 310 (true 437) |
| **the bar** | mean ✓, share 7/10 ✓ (exactly the minimum), each source ✓, unseen ✓: **passes** | mean ✓, 6/7 ✓, unseen ✓: **passes** |

- **Merges carry it; flips alone fail the bar** (6/10). Merges alone pass on both sets (+0.0151,
  8/10; IDSSE only +0.0120, 6/7).
- **The gain sits on Metrica and the PathCRF-unseen matches.** On the 6 IDSSE matches PathCRF saw,
  `all`'s settings gain little and 3 of 7 IDSSE matches get worse (J03WMX −0.019). With `--set idsse`
  the picked settings are more cautious and 6 of 7 improve.

## Why

**The error budget** ([../ideas.md](../ideas.md), "Where v9's remaining spellF1 is"): two errors are
about whole possessions, not their edges:

| error | if fixed perfectly (on v9) |
|---|---|
| **team of each possession** (a predicted possession whose majority true team is the other one) | +0.054 |
| **spurious possessions** (predicted, no true match at IoU ≥ 0.3) | +0.043 |

v9 and v11 predict ~349 possessions per match against ~411 true. On v11 the counts per match are
39–61 wrong-team possessions and 66–102 spurious ones (of 314–405). **They overlap:** of 492 wrong-team
possessions, 428 are also spurious. They are short: median **1.8 s** (wrong team) and 2.2 s
(spurious, right team), against 7.2 s for the possessions that match.

**The metric's arithmetic** (evaluation/possession_metrics.py): predicted and true possessions are
matched one to one, same team only, by temporal IoU; F1 = 2·TP / (predicted + true), averaged over
IoU 0.3–0.7. At F1 ≈ 0.57, removing a spurious possession gains F1/(P+G); removing a matched one
costs (2 − F1)/(P+G), about **2.5 times more**. So a merge only pays when P(spurious) is above about
0.7. A flip of a wrong possession gains about what a flip of a right one loses: break-even near 0.5.

**The ceiling of these exact actions** (`--oracle`: the truth's own answers, P = 0 or 1, on v11;
exploration, not a candidate):

| oracle | spellF1 | gain | better | possessions |
|---|---|---|---|---|
| flip every wrong-team possession | 0.6294 | +0.057 | 10/10 | 272 |
| merge every spurious one, mode `neighbour` | 0.6182 | +0.045 | 10/10 | 256 |
| merge every spurious one, mode `dead` | 0.6265 | +0.054 | 10/10 | 253 |
| both, `neighbour` | 0.6287 | +0.056 | 10/10 | 248 |
| **both, `dead`** | **0.6369** | **+0.064** | 10/10 | 248 |

## What it does

### The possessions

A possession is a maximal run of H or A inside a half, read as the scorer reads it (X forward-filled
inside the half). Its neighbours are the runs either side: D, the other team, or the half's edge.

### Features (prediction-time inputs only)

From the match's damaged tracking, the out-of-sample outputs in `match.pred`, and the input letters.
T is the possession's team, O the other.

| feature | what | from |
|---|---|---|
| `log_len` | log of its length in frames | letters |
| `seen` | share of its frames with the ball seen | tracking |
| `near_share` | share of ball-seen frames where T's nearest player is closer to the ball than O's | tracking |
| `reach_t`, `reach_o` | share of ball-seen frames with a T (O) player within 2 m | tracking |
| `dist_t`, `dist_o` | median distance from the ball to T's (O's) nearest player, capped at 30 m | tracking |
| `touch_t`, `touch_o` | touches by each team inside it (v11's detector, ΔV 3 m/s) | `v11.seen_touches` |
| `pass_t`, `pass_o` | pass starts by each team inside it | `v9.seen_passes` |
| `touch_start` | a touch by T within 0.5 s of its first frame | `v11.seen_touches` |
| `pdead_in`, `pdead_before`, `pdead_after` | mean P(dead) inside it, in the 2 s before, in the 2 s after | `p_dead` (v4) |
| `left_d`, `left_o`, `right_d`, `right_o`, `sandwich` | what's either side (D, O; both O) | letters |
| `left_len`, `right_len` | log(1 + length) of each neighbour | letters |
| `x_toward`, `y_off` | mean ball x toward T's attacking goal, mean distance from the long axis | tracking |
| `restart_t` | the restart model's P(T restarts) at its first frame | `p_start` (v6) |
| `p_mean`, `p_share`, `p_cover` | PathCRF's mean P(T), share of frames > 0.5, share of frames where it answers | `p_pathcrf` (feature set `all` only) |
| `margin_o_mean`, `margin_o_sum`, `margin_d_mean` | the v7 decoder's cost of O minus T (mean and sum), of D minus T (mean) | `v7_cost` (`all` only) |
| `margin6_*` | the same from v6's costs, which have no PathCRF clue | `v6_cost` (`no_pathcrf` only) |

Attacking direction: in each half, a team attacks toward +x when its players' mean x is below the
other team's (its keeper and back line sit in its own half).

**The strongest single features** (pooled AUC, either direction): for both targets, the decoder's
margin (`margin_o_sum` 0.84–0.85), the length (0.82–0.84), PathCRF's mean P(T) (0.80–0.81), touches
and passes by T (0.72–0.77).

### Targets (truth of the training matches only)

- **wrong**: inside the possession, the truth has more frames of O than of T.
- **spurious**: no true possession of T overlaps it at IoU ≥ 0.3.

### The classifiers

Two `HistGradientBoostingClassifier`s (one per target), the same fixed settings: 200 rounds, rate
0.05, 8 leaves, at least 40 possessions per leaf, L2 1.0, no early stopping, seed 0. Small and
regularised because each fold trains on only ~3,100 possessions; not tuned. Trained on the input
letters of the other matches (`fit(..., letters_of)`), so it learns the errors of whatever stage
comes before it.

### The actions

1. **Flip** (all at once): every possession with P(wrong) > `flip` becomes O's. A flipped blip
   inside O's possession joins it.
2. **Merge** (from the most likely spurious down, reading the labels as they are by then): every
   possession not flipped with P(spurious) > `merge`:
   - both neighbours O: it becomes O (a blip inside O's possession disappears);
   - otherwise, mode **`neighbour`**: O if one neighbour is O, D if neither is;
     mode **`dead`**: D;
   - skipped if a flipped neighbour has joined it (it isn't on its own any more).

Each possession's frames change only through its own action; frames outside every acted-on
possession never change.

### Constants and the grid

| setting | value | why |
|---|---|---|
| touch ΔV | 3 m/s | v11's, both sets: a "touch" means the same thing as in v11 |
| on the ball | 2 m | the touch detector's `REACH` (v1's "on the ball") |
| distance cap | 30 m | a team with nobody near isn't more or less absent past that; keeps the feature bounded |
| P(dead) around | 2 s either side | a stoppage just before or after says whether it's a restart, a last touch before the ball goes out, or a blip |
| touch "at the start" | ±0.5 s | v11 moved changes onto touches within ±1 s; half of that says "this start is on a touch" |
| spurious IoU | 0.3 | the lowest IoU the metric counts: below it, the possession is never a match |
| **flip** | **off, 0.5, 0.6, 0.7, 0.8** | break-even near 0.5 (above); higher for caution |
| **merge** | **off, 0.6, 0.7, 0.8, 0.9** | break-even near 0.7 (above); 0.6 in case the classifier is under-confident |
| **mode** | **`neighbour`, `dead`** | where a merged possession's frames go when only one neighbour is O |
| **features** | **`all`, `no_pathcrf`** | PathCRF is better on the 6 IDSSE matches it was pretrained on; `no_pathcrf` drops it (and uses v6's costs, which don't include it) |

89 settings: the no-op first (ties keep v11), then for each feature set flips alone, merges alone,
both. All fixed before any held-out run; nothing was changed after.

## Results

### The procedure

`lab.judge`: for each held-out match, both classifiers are trained on the other 9 (6 with
`--set idsse`) and applied to it for every setting; the match is scored with the setting best on
the other matches. Against v11 on the same match.

### All 10 matches

| match | source | picked (features, flip, merge, mode) | spellF1 | v11 | Δ | possessions (true) | flips (right) | merges (right) |
|---|---|---|---|---|---|---|---|---|
| J03WMX | IDSSE | all, 0.6, 0.6, dead | 0.4914 | 0.5100 | **−0.0186** | 268 (511) | 8 (6) | 37 (21) |
| J03WN1 | IDSSE | all, 0.6, 0.7, dead | 0.5420 | 0.5282 | +0.0139 | 311 (500) | 23 (13) | 38 (25) |
| J03WOH | IDSSE | all, 0.6, 0.7, dead | 0.5757 | 0.5676 | +0.0081 | 281 (402) | 16 (7) | 29 (22) |
| J03WOY | IDSSE | all, 0.6, 0.6, dead | 0.6056 | 0.6045 | +0.0010 | 297 (456) | 14 (6) | 43 (30) |
| J03WPY | IDSSE | no_pathcrf, 0.6, 0.7, dead | 0.6135 | 0.6181 | −0.0047 | 291 (421) | 17 (8) | 23 (17) |
| J03WQQ | IDSSE | all, 0.6, 0.6, dead | 0.5606 | 0.5624 | −0.0018 | 279 (401) | 11 (9) | 31 (21) |
| J03WR9 | IDSSE, unseen | all, 0.6, 0.7, dead | 0.6065 | 0.5801 | +0.0265 | 275 (370) | 12 (10) | 27 (21) |
| metrica1 | Metrica, unseen | all, 0.8, 0.6, dead | 0.6302 | 0.5934 | +0.0368 | 256 (354) | 2 (2) | 37 (31) |
| metrica2 | Metrica, unseen | all, 0.6, 0.8, dead | 0.6228 | 0.5811 | +0.0417 | 269 (318) | 19 (15) | 11 (11) |
| metrica3 | Metrica, unseen | all, 0.8, 0.6, dead | 0.6248 | 0.5821 | +0.0427 | 281 (381) | 3 (3) | 50 (42) |
| **mean** | | | **0.5873** | **0.5727** | **+0.0146** | **281 (411)**; v11 349 | **125 (79: 63 %)** | **326 (241: 74 %)** |

- **Bar:** mean ✓; 7 of 10 better ✓ (the minimum); IDSSE +0.0035 ✓ and Metrica +0.0404 ✓; unseen
  +0.0369 ✓. **Passes.**
- **macroF1** 0.8723 against 0.8705 (+0.0018): a D in place of a spurious blip is as often right as
  the blip was.
- **Held-out AUC** (each match, its classifiers trained without it): wrong 0.87, spurious 0.90
  (`no_pathcrf`: 0.86, 0.90). Higher on Metrica (0.91–0.95) than on IDSSE (0.81–0.89).
- **In sample** (one setting on all 10, not a held-out number): the chosen setting gives 0.5939
  (+0.0212), better on 8/10; IDSSE +0.0086 (5/7), Metrica +0.0505 (3/3).

**Flips and merges alone** (the same judge over the no-op plus only those settings; diagnostics, the
verdict is the full grid's):

| | held out | better | unseen | chosen on all | bar |
|---|---|---|---|---|---|
| **flips alone** | 0.5850 (+0.0123) | 6/10 | +0.0306 | all, flip 0.5 | **fails** (share) |
| **merges alone** | 0.5878 (+0.0151) | **8/10** | +0.0319 | all, merge 0.6, dead | **passes** |
| both (the candidate) | 0.5873 (+0.0146) | 7/10 | +0.0369 | all, flip 0.6, merge 0.7, dead | passes |

Mode `dead` beats `neighbour` in every comparable pair (in sample, best 0.5939 against 0.5906), as in
the oracle: a spurious possession next to a stoppage is more often dead ball than the neighbour's.

**What the chosen setting acts on** (held out, all 10): 150 flips, 69 % of them wrong-team and 76 %
blips inside the other team's possession, median 1.2 s; 157 merges to D (78 % spurious, median
1.2 s); 121 merges into a surrounding possession of the other team (74 % spurious, median 1.0 s).
Flipping a blip and merging it into the possession around it give the same letters, which is why
the two actions overlap so much.

### IDSSE only (`--set idsse`)

| match | picked | spellF1 | v11 | Δ | possessions (true) | flips (right) | merges (right) |
|---|---|---|---|---|---|---|---|
| J03WMX | no_pathcrf, 0.8, 0.7, dead | 0.5072 | 0.5032 | +0.0040 | 299 (511) | 0 | 21 (15) |
| J03WN1 | no_pathcrf, 0.8, 0.7, dead | 0.5234 | 0.5002 | +0.0232 | 336 (500) | 0 | 43 (34) |
| J03WOH | no_pathcrf, 0.8, 0.7, dead | 0.5787 | 0.5667 | +0.0120 | 301 (402) | 2 (2) | 19 (17) |
| J03WOY | no_pathcrf, 0.8, 0.7, dead | 0.6131 | 0.6146 | **−0.0014** | 321 (456) | 1 (0) | 31 (23) |
| J03WPY | all, 0.7, 0.8, dead | 0.6247 | 0.6197 | +0.0050 | 300 (421) | 9 (5) | 13 (11) |
| J03WQQ | all, 0.8, 0.8, dead | 0.5589 | 0.5543 | +0.0046 | 314 (401) | 2 (2) | 15 (12) |
| J03WR9 | no_pathcrf, 0.8, 0.8, dead | 0.6018 | 0.5790 | +0.0228 | 302 (370) | 5 (5) | 18 (15) |
| **mean** | | **0.5725** | **0.5625** | **+0.0100**, 6/7 | **310 (437)**; v11 348 | **19 (14)** | **160 (127: 79 %)** |

(v11's own settings differ per set, so its per-match scores differ from `all`'s; see
`bench/results/lab_baseline_idsse.json`.) Flips alone fail here too (+0.0054, 4/7); merges alone pass
(+0.0120, 6/7, chosen `no_pathcrf`, merge 0.7, dead).

### The stricter check (`--nested`)

`lab.judge` picks match m's setting from the other matches' scores, but each of their classifiers
was trained with m among its 9 training matches. `--nested` removes that: for each m and each other
match j, the classifiers are trained without m and j, j is scored for every setting, and m gets
the setting best over those scores. m's own score is the same as in the main run (classifiers trained
without m). Same grid, nothing changed; run after the main run, ~10 min.

| | nested | judge (above) |
|---|---|---|
| **all 10** | **0.5915 against 0.5727 (+0.0187), better on 8/10** | +0.0146, 7/10 |
| IDSSE | 0.5739 against 0.5673 (+0.0067), 5/7 | +0.0035, 4/7 |
| Metrica | 0.6324 against 0.5855 (+0.0469), 3/3 | +0.0404, 3/3 |
| PathCRF-unseen | 0.6249 against 0.5842 (+0.0407), 4/4 | +0.0369, 4/4 |
| picks | merge 0.7 on 9 of 10 folds (0.6 on metrica2), flip 0.6–0.8, all `dead` | merge 0.6 on 5 of 10 |

The stricter check comes out **better**, not worse: without the leak the folds pick merge 0.7, not
the aggressive 0.6 that cost J03WMX. It would pass the bar too. Worse: J03WMX (−0.003) and J03WPY
(−0.002).

With `--set idsse`: **0.5733 against 0.5625 (+0.0108), better on 5/7** (the judge: +0.0100, 6/7);
J03WR9 +0.0228. Worse: J03WMX (−0.003) and J03WOY (−0.001). Merge 0.7–0.8 and mode `dead` on every
fold; `no_pathcrf` on 5 of 7.

## Failure modes

1. **IDSSE's PathCRF-seen matches gain little under `all`.** Metrica's three matches gain +0.04 each
   in sample and pull the picked thresholds toward aggressive ones (merge 0.6), which cost J03WMX
   −0.019 (its merges were right only 57 % of the time). Only 7 of 10 improve, the minimum the bar
   allows. With IDSSE alone the picks are cautious (flip 0.8, merge 0.7) and 6 of 7 improve.
2. **A merge removes a true match about 1 time in 4** (241 of 326 right). Each wrong one costs about
   2.5 times what a right one gains; the thresholds already price that in, but the margin is thin on
   matches where the classifier is less sure (J03WMX, J03WQQ).
3. **Possessions get fewer, not more right:** 349 → 281 per match against 411 true. This cleans up
   precision; the biggest error left (missed possessions, +0.250) is untouched. A later stage that
   inserts short possessions (v12, the `short` candidate) and this one pull in opposite directions:
   chained after it, this stage must be re-fitted on its letters (`letters_of`), or it will merge
   away what it inserted.
4. **Flips are fragile on their own** (6/10; 4/7 with IDSSE). Most flips are blips inside the other
   team's possession, the same letters a merge would give; long wrong-team possessions are rarely
   found with enough confidence.
5. **PathCRF.** `all` (with PathCRF and v7's costs) is picked on 9 of 10 folds, but `no_pathcrf` has
   the same held-out AUC, and the unseen matches gain the most (+0.037), so the gain does not rest
   on PathCRF's sharper IDSSE probabilities. With IDSSE alone, `no_pathcrf` is picked on 5 of 7.
6. **The threshold choice isn't fully nested** in `lab.judge`: match m's setting is picked from the
   other matches' scores, and their classifiers were trained with m's truth in (1 of 9 training
   matches). `--nested` removes that, and the result holds (+0.0187, 8/10, above).
7. **Our match needs the costs.** The chosen setting reads `v7_cost` (v7's decoder costs) and
   `p_pathcrf`, `p_dead`, `p_start`. `possession.v7.predict` builds the cost inside v4's decoder and
   doesn't return it, so running this on our match needs v7's cost (and v6's for `no_pathcrf`) to be
   exposed. Only the 41 % damage level was tried (the lab cache has no other).

## Verdict

**Keep.** Passes the bar on `all` (+0.0146, 7/10, both sources and the unseen matches up) and on
`idsse` (+0.0100, 6/7), and the stricter nested check holds (+0.0187, 8/10). The merges are the solid part (alone: 8/10 and 6/7); the flips add little
on top and fail alone. The gain is concentrated on Metrica and the PathCRF-unseen matches, which are
the ones most like our match, but it rests on 4 matches there; on IDSSE's PathCRF-seen matches it's
close to zero under `all`'s thresholds.

## For mixing

```python
from bench import lab_segments as S
model = S.fit(matches, letters_of, features="all")          # matches: any iterable; truth used
letters = S.apply(match, letters, model, flip=0.6, merge=0.7, mode="dead")   # chosen on `all`
# --set idsse: S.fit(..., features="no_pathcrf"); S.apply(..., flip=0.8, merge=0.7, mode="dead")
```

- `STAGE = "post"`, `GRID` (89 settings, no-op first), `COLUMNS = ["features", "flip", "merge", "mode"]`.
- `fit` reads each training match once and only its own letters, tracking, `pred` and truth; the
  thresholds don't affect it. `apply` reads the match's tracking, `pred` and the given letters,
  never its truth. Per-frame signals are cached on the match object (`match._segments_signals`,
  ~15 MB), since they don't depend on the letters.
- Off: `flip=1.0` / `merge=1.0` (no probability exceeds 1).
