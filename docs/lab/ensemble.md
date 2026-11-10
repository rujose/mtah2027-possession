# Lab candidate `ensemble`: the lab's per-frame models combined

| | |
|---|---|
| **status** | **discarded** (2026-11-09). v12 stays as it is |
| **what** | v12's start (robust) replaced by a combination of the four per-frame models the lab has built: robust, learned, weighted (alpha 0.5) and motion. The arithmetic or geometric mean of their P(H / A / D) over four combinations, or a logistic-regression stacker on all four; the combination chosen with the decoder's prices, each match on the others |
| **why** | averages of diverse models often beat each of them, and every member already has a held-out P on every bench match: the cheapest candidate of round 7 |
| **code** | `bench/lab_ensemble.py`; results in `bench/results/lab_ensemble[_idsse].json` (and `_grid`, `_starts`, `_frames`), `lab_ensemble_chains[_idsse].json`; each start's P in `data/external/lab_ensemble[_idsse]/` |

## What was fixed before the run

| setting | value | why |
|---|---|---|
| members | robust (blanked ball), learned (all inputs), weighted (alpha 0.5), motion; each match's P from the model trained without it | the four the lab has; PathCRF left out, as every member already reads it |
| combinations | robust + motion, robust + learned, robust + weighted, all four | v12's start with each other model, and everything |
| rules | arithmetic mean; geometric mean (the mean of log P); each renormalised, then clipped to 0.01–0.99 and rounded to 3 decimals, as the members are | the two usual ways to average probabilities |
| stacker | a multinomial logistic regression (C = 1) on the four members' log P (12 numbers), trained on the other matches' held-out P and truth: IDSSE's only, every 10th graded frame, each also with the teams swapped | as the members: IDSSE-only training, teams symmetric |
| decoder | robust's grid: team switch 0.5–16, dead switch 50, 25, 10, minimum 20 or 10 frames; then v9's snap and v11's move | the averages change how sure P is, so the prices are chosen again |
| the choice | the start and the prices together, 9 × 36 settings, each match on the other matches (`lab.judge`) | one candidate, one verdict |

A check on the harness: robust alone decoded over the same 36 settings gives 0.6732 on `all` and 0.6365 on
the IDSSE set, against its own judging's 0.6728 and 0.6365.

## The start, against robust

| | all 10: 0.6821 vs 0.6728 | IDSSE set: 0.6423 vs 0.6365 |
|---|---|---|
| overall | +0.0093, 8/10 | +0.0059, 5/7 |
| IDSSE's matches | +0.0109, 5/7 | +0.0059, 5/7 |
| Metrica's matches | +0.0054, 3/3 | — |
| PathCRF-unseen | +0.0038, 3/4 | **−0.0071, 0/1** (J03WR9) |
| bar | passes | **fails** (unseen) |
| macroF1 | 0.9170 vs 0.9145 | 0.9137 vs 0.9115 |

On `all` every fold picked the geometric mean of all four (team switch 4, dead switch 50, minimum 10 frames
in 9 folds). On the IDSSE set the folds split: the mean of robust and weighted 3 times, the geometric mean of
all four 4 times. Possessions per match barely move (343 against robust's 347 on J03WMX, the same ±10
elsewhere).

Each start judged on its own (for information; the bar is judged on the choice above):

| start | all 10 | IDSSE's 7 | Metrica's 3 | unseen 4 | bar | IDSSE set | unseen (J03WR9) | bar |
|---|---|---|---|---|---|---|---|---|
| mean, robust + motion | +0.0056, 7/10 | +0.0057 | +0.0054 | +0.0027 | passes | +0.0053, 4/7 | −0.0075 | fails |
| mean, robust + learned | +0.0011, 6/10 | +0.0064 | −0.0113 | −0.0095 | fails | +0.0036, 4/7 | −0.0027 | fails |
| mean, robust + weighted | +0.0048, 8/10 | +0.0079 | −0.0026 | +0.0011 | fails | +0.0110, 6/7 | +0.0103 | passes |
| mean, all four | +0.0053, 5/10 | +0.0081 | −0.0015 | −0.0024 | fails | +0.0082, 6/7 | −0.0071 | fails |
| geometric, robust + motion | +0.0054, 6/10 | +0.0068 | +0.0022 | +0.0001 | fails | +0.0037, 4/7 | −0.0075 | fails |
| geometric, robust + learned | +0.0014, 7/10 | +0.0070 | −0.0116 | −0.0079 | fails | +0.0055, 5/7 | −0.0010 | fails |
| geometric, robust + weighted | +0.0041, 8/10 | +0.0079 | −0.0047 | −0.0007 | fails | +0.0101, 6/7 | +0.0093 | passes |
| **geometric, all four** | **+0.0093, 8/10** | +0.0109 | +0.0054 | +0.0038 | passes | +0.0084, 6/7 | −0.0071 | fails |
| stacker, all four | +0.0058, 6/10 | **+0.0128** | −0.0105 | −0.0072 | fails | +0.0094, 5/7 | −0.0047 | fails |

## The chain, against v12

v12's chain from each start (short → segments → timing on `all`; short → segments → recall → dead on the
IDSSE set), every stage refitted on that start's output and run on its setting for that match.

| | all 10 | IDSSE's 7 | Metrica's 3 | unseen 4 | IDSSE set |
|---|---|---|---|---|---|
| v12 | 0.7033 | 0.6674 | 0.7870 | 0.7617 | 0.6699 |
| ensemble | 0.7076 | 0.6718 | 0.7913 | 0.7639 | 0.6660 |
| gain | **+0.0043**, 7/10 | +0.0043, 4/7 | +0.0043, 3/3 | +0.0022, 3/4 | **−0.0039**, 4/7; J03WR9 −0.0214 |
| bar | **fails**: the four checks pass, the gain is under +0.005 | | | | **fails** every check |

Per match on `all`: J03WN1 +0.027, J03WQQ +0.016, J03WMX +0.013, metrica2 +0.007, metrica1 +0.004,
metrica3 +0.002, J03WPY +0.002; J03WOY −0.015, J03WOH −0.010, J03WR9 −0.004.

## What it showed

1. **Averaging helps the start on every combination, mostly on IDSSE's matches.** All 9 starts beat robust's
   mean on both sets; on IDSSE's matches each gains +0.004 to +0.013. The members disagree with robust on only
   2–3 % of graded frames, so the gain comes from a few uncertain stretches; frame accuracy moves by under
   0.6 points.
2. **Members that read IDSSE's frozen ball hurt Metrica.** Both pairs with learned lose on Metrica
   (−0.011, 0/3), and so does the stacker (−0.011, 0/3): trained on IDSSE's truth, it
   weighs learned most (its log P(home) gets +0.41 for home, robust's −0.13) and so re-learns the
   frozen-ball convention robust was built to drop (robust.md). It's the best start on IDSSE's matches
   (+0.013) and the worst on Metrica's. Motion is the only member that helps both sources.
3. **The chain takes it back.** On `all` the start's +0.0093 becomes +0.0043 after short, segments and
   timing, which already repair much of what averaging repairs (IDSSE's matches from 5/7 better to 4/7). On
   the IDSSE set +0.0059 becomes −0.0039 (J03WR9 −0.021). The stages are refitted on the ensemble's
   letters, but their settings were chosen on v12's own chain; this run doesn't say which of the two costs it.
4. **Which average wins is noise-sized.** The arithmetic and geometric means of the same four models differ by
   0.004 and 3 matches on `all`, while their frame accuracy is the same. The folds agree on `all` and split on
   the IDSSE set, where the start fails on J03WR9 (the one PathCRF-unseen IDSSE match), the same match
   where the chain loses on `all`.

## Caveats

- **The stacker's training P isn't fully nested.** The members' P on another match j came from models trained
  on the held-out match among others. That's second-order, and the same structure as the mix's stacking
  (lab.md, "Mixing"); the stacker wasn't kept either way.
- **The frame log loss isn't comparable between members and averages** (`lab_ensemble_frames*.csv`): the
  averages are renormalised before the clip, so a frame at the 0.99 cap costs −log 0.98 instead of
  −log 0.99. The decoder only sees differences between the classes' costs, so it barely matters to it.
- Nine starts on one bench flatter the best (the winner's curse); the choice is nested in leave one match
  out, but the members and combinations were the brief's.
