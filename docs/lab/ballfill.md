# Lab candidate `ballfill`: the missing ball, reconstructed

| | |
|---|---|
| **status** | **discarded** (2026-11-09, round 7). v12 stays as it is |
| **what** | the ball estimated wherever it's missing (interpolation between the last and next seen positions, plus an offset learned from the players), then robust given 11 numbers about who is near the estimate |
| **why** | [inside-v12.md](../inside-v12.md): robust's team call is right 93 % of the time with the ball seen, 86 % with it missing, and the ball is missing on ~41 % of frames |
| **code** | `bench/lab_ballfill.py`; results in `bench/results/lab_ballfill_impute[_idsse].json` (the imputer's error), `lab_ballfill[_idsse].json` and `lab_ballfill_grid[_idsse].csv` (the start against robust), `lab_ballfill_chains[_idsse].json` (the chain against v12); caches in `data/external/lab_ballfill[_idsse]/` (gitignored) |
| **run** | `python -m bench.lab_ballfill --impute [--set idsse]` (~5 min), then `python -m bench.lab_ballfill [--set idsse]` (~25 min), then `--chains [--set idsse]` (~25 min) |

## How it works

1. **Robust's view of the ball.** Everything starts from `lab_robust.blanked`: a ball still for 1 s
   counts as missing, so nothing here can read a still ball. In that view the ball is missing on
   57–66 % of an IDSSE match's frames and 36–46 % of a Metrica one's.
2. **The imputer** (`impute_inputs`, `fit_imputer`). For each missing frame inside a half with a
   seen ball before or after it: plain interpolation between the last and next seen positions (the
   one that exists, at a half's edge), plus an (x, y) offset from two gradient-boosted regressors.
   Their 43 inputs are team-agnostic, so they can't learn which side the bench's home teams are:

   | family | inputs |
   |---|---|
   | the gap | length, s since and until a seen ball, how far through it, whether each end exists |
   | the ends | where (and the interpolated point; distance to the nearest line; how far each end is beyond a line), the move between them, the ball's velocity at each end, dead reckoning from each (at most 1 s) |
   | players at the interpolated point | the two nearest (distance, offset), the centroid's offset, how many within 5 m, everyone's mean speed |
   | the players who had it | the player nearest the ball when it was last seen (next seen): his distance then, where he is now, where the ball would be had it moved with him; whether the two are teammates |

   Trained leave one match out. The target is the clean ball (`bench.sets.load`) on the other
   matches' missing frames: IDSSE's everywhere (tracked through dead time), Metrica's where its clean
   ball exists (live play). On the `idsse` set, the other IDSSE matches only. Never the predicted
   match's clean ball.
3. **The features**, 11 per frame beside robust's 97, on the filled ball (the seen ball where seen,
   the estimate where missing): imputed or not; each team's nearest player and their difference,
   with the difference's past and next 1 s and 3 s means; where it is (x from home's goal, from the
   touchline, beyond a line). No speed of the filled ball: a reconstructed track's speed mostly
   restates the gap's length, and robust keeps speed out of anything blanked.
4. **Everything else is robust's**: its cached features, IDSSE-only training, every 10th graded
   frame, the teams swapped, the same trees, the same 36 decoder settings, then v9's snap and v11's
   move on the original tracking. Judged against robust (`lab_weighted.robust_base`), then as v12's
   chain start against v12 (each stage on its setting for that match, refitted on this input).

## What was fixed before the run

| setting | value | why |
|---|---|---|
| ball view | robust's (still ≥ 1 s blanked) | the model must not read a still ball through the imputer either |
| regressors | 300 trees, rate 0.1, 31 leaves, ≥ 100 rows a leaf, L2 1, squared error, every 4th missing frame | robust's kind of trees, a bit longer (two continuous targets) |
| dead reckoning | at most 1 s | a ball's course past 1 s is a guess |
| estimate kept within | the pitch ± 10 m | a guard; balls out of play lie a few metres outside |
| team-agnostic imputer | yes | the team signal belongs to the classifier, which sees both orientations |
| features | the 11 above | the brief's "small set": the nearest player to the filled ball, his team, the distance |
| fallback | interpolation instead of the imputer if the imputer's pooled held-out error (mean m) isn't lower | written down beforehand; not triggered (below) |
| everything downstream | robust's | so any difference is the filled ball |

## The imputer, held out

Mean error in metres against the clean ball, every missing frame with a clean target (634,734 on all
10, 97 % IDSSE; Metrica's only in live play). Each match from an imputer trained without it.

| | frames | interpolation, mean / median | **imputer**, mean / median | within 2 m: interp. / imputer | within 5 m |
|---|---|---|---|---|---|
| **all** | 634,734 | 15.7 / 10.6 | **12.1 / 8.5** | 19 % / **12 %** | 33 % / 32 % |
| IDSSE | 614,367 | 15.7 / 10.6 | 12.1 / 8.4 | 19 % / 12 % | 33 % / 33 % |
| Metrica (live only) | 20,367 | 15.2 / 10.8 | 12.7 / 10.4 | 21 % / 8 % | 32 % / 22 % |
| dead frames | 452,588 | 14.2 / 8.4 | 11.1 / 7.2 | 23 % / 14 % | 38 % / 38 % |
| **live frames** | 182,146 | 19.5 / 16.6 | **14.7 / 12.0** | 10 % / 6 % | 19 % / 20 % |

Live frames by the length of the gap they sit in:

| gap | share of live missing frames | interpolation, mean / median | imputer, mean / median |
|---|---|---|---|
| under 1 s | 0.2 % | **0.35 / 0.17** | 2.66 / 2.52 |
| 1–3 s | 3.4 % | **1.17 / 0.20** | 2.94 / 2.18 |
| 3–10 s | 8.8 % | **6.5 / 4.6** | 7.1 / 5.3 |
| 10–30 s | 30.4 % | 16.2 / 13.3 | **12.9 / 10.5** |
| 30 s and more | 56.2 % | 24.2 / 21.9 | **17.4 / 15.0** |
| at a half's edge (one end) | 1.1 % | 35.9 / 35.2 | 27.7 / 25.1 |

The IDSSE set's imputer (6 IDSSE matches each): 12.2 m against 15.7 m, the same shape.

By the rule fixed beforehand the features read the imputer (12.1 m against 15.7 m). Two things the
pooled mean hides:

- **87 % of the live missing frames sit in gaps of 10 s or more**, where neither estimate is
  anywhere near the ball (13–24 m). The damaged ball is lost in long stretches, as ours is (median gap
  10.7 s, [bench.md](../bench.md)). Where the ball *could* be pinned down (gaps under 3 s),
  interpolation is already within a metre, and the imputer makes it worse (2.7–3.1 m): a squared-error
  offset trained mostly on long dead-time gaps adds metres of noise to short ones.
- **The estimate's nearest player is rarely the team with the ball.** On live missing frames with a
  clean ball, the share where the nearest (damaged) player belongs to the team in possession:

  | the ball taken as | all | IDSSE | Metrica |
  |---|---|---|---|
  | where it was last seen | **64 %** | 64 % | 71 % |
  | interpolation | 56 % | 55 % | 64 % |
  | the imputer | 51 % | 50 % | 56 % |
  | the clean ball (a ceiling) | 73 % | 73 % | 73 % |

  The imputer pulls the estimate towards where players are, so its nearest player is close to a coin
  flip. Robust already reads the players at the last and next seen ball, the best of the three.

## The start alone, against robust

| | all 10: 0.6713 vs 0.6728 | IDSSE set: 0.6369 vs 0.6365 |
|---|---|---|
| overall | −0.0015, 3/10 | +0.0005, 3/7 |
| IDSSE's matches | −0.0051, 1/7 | +0.0005, 3/7 |
| Metrica's matches | +0.0068, 2/3 | — |
| PathCRF-unseen | +0.0035, 2/4 | J03WR9 −0.0007 |
| bar | **fails** (mean, share, IDSSE) | **fails** (share, unseen) |
| macroF1 | 0.9156 vs 0.9145 | 0.9125 vs 0.9115 |

| match | all 10: ballfill | robust | difference | IDSSE set: ballfill | robust | difference |
|---|---|---|---|---|---|---|
| J03WMX | 0.5722 | 0.5791 | −0.0069 | 0.5762 | 0.5774 | −0.0013 |
| J03WN1 | 0.5939 | 0.5907 | +0.0032 | 0.5786 | 0.5928 | −0.0142 |
| J03WOH | 0.6614 | 0.6774 | −0.0160 | 0.6559 | 0.6653 | −0.0094 |
| J03WOY | 0.6628 | 0.6638 | −0.0010 | 0.6575 | 0.6439 | +0.0136 |
| J03WPY | 0.6731 | 0.6790 | −0.0059 | 0.6794 | 0.6710 | +0.0084 |
| J03WQQ | 0.6406 | 0.6436 | −0.0030 | 0.6528 | 0.6458 | +0.0070 |
| J03WR9 (unseen) | 0.6555 | 0.6618 | −0.0063 | 0.6582 | 0.6589 | −0.0007 |
| metrica1 (unseen) | 0.7696 | 0.7563 | +0.0133 | — | | |
| metrica2 (unseen) | 0.7420 | 0.7349 | +0.0071 | — | | |
| metrica3 (unseen) | 0.7417 | 0.7417 | −0.0000 | — | | |

The same match moves by up to 0.03 between the two sets (J03WN1 +0.003 and −0.014, J03WOY −0.001 and
+0.014): the decoder picks differ (all 10: team 4, dead 25 in every fold; IDSSE set: team 8, dead 50
in 5 of 7), and that is most of what either set's difference measures.

What the classifier did with it (held out, mean of the folds):

| | all 10: ballfill | robust | IDSSE set: ballfill | robust |
|---|---|---|---|---|
| team right in play, ball missing | 85.0 % | 85.1 % | 88.1 % | 88.5 % |
| team right in play, ball seen | 93.0 % | 93.0 % | 92.1 % | 92.0 % |
| frames right (argmax) | 92.3 % | 92.2 % | 92.4 % | 92.4 % |
| rise in log loss with the 11 filled-ball features shuffled | 0.016 | — | 0.015 | — |
| … with the ball family shuffled | 1.08 | 1.04 (robust.md) | 1.05 | 1.03 (robust.md) |

**The team call with the ball missing doesn't move** (85.0 % against 85.1 %; per match within ±1.6
points either way; IDSSE set 88.1 % against 88.5 %). On all 10 the decoder's minimum possession moved
from robust's 10 frames to 20 (9 of 10 folds): 338 possessions a match against 341.

## The chain, against v12

v12's chain from each start, every stage refitted on that start's output, each on its setting for
that match. v12 reproduces at 0.7033 (all) and 0.6699 (IDSSE set).

| | all 10 | IDSSE's 7 | Metrica's 3 | PathCRF-unseen | IDSSE set |
|---|---|---|---|---|---|
| v12 | **0.7033** | 0.6674 | 0.7870 | 0.7617 | **0.6699** |
| ballfill chain | 0.6983 | 0.6577 | 0.7929 | 0.7643 | 0.6651 |
| difference | **−0.0050, 4/10** | −0.0097, 1/7 | +0.0059, 3/3 | +0.0026, 3/4 | **−0.0048, 1/7** (J03WR9 −0.0142) |
| bar | **fails** (mean, share, IDSSE) | | | | **fails** (every check) |
| macroF1 | 0.9182 → 0.9192 | | | | 0.9162 → 0.9166 |

| match | all 10: v12 | ballfill chain | IDSSE set: v12 | ballfill chain |
|---|---|---|---|---|
| J03WMX | 0.6000 | 0.5832 | 0.5849 | 0.6121 |
| J03WN1 | 0.6180 | 0.6330 | 0.6306 | 0.6066 |
| J03WOH | 0.7014 | 0.6775 | 0.7144 | 0.7005 |
| J03WOY | 0.6957 | 0.6847 | 0.7020 | 0.6979 |
| J03WPY | 0.7109 | 0.6910 | 0.7246 | 0.7239 |
| J03WQQ | 0.6603 | 0.6563 | 0.6483 | 0.6445 |
| J03WR9 | 0.6857 | 0.6782 | 0.6843 | 0.6701 |
| metrica1 | 0.7876 | 0.7905 | — | — |
| metrica2 | 0.7866 | 0.7910 | — | — |
| metrica3 | 0.7868 | 0.7972 | — | — |

## What it showed

1. **On this tracking the missing ball can't be recovered where it matters.** The ball is lost in
   long stretches: 87 % of the live frames without a ball sit in gaps of 10 s or more, and there the
   best estimate is still 13–17 m off. Where gaps are short, interpolation is already exact to a
   metre, and robust's "players at the last / next seen ball" carry most of it.
2. **A closer estimate isn't a more useful one.** The imputer cuts the mean error by a fifth (15.7 →
   12.1 m) by hedging towards the players, and that makes its nearest player *less* often the team in
   possession (51 % against 56 % for interpolation and 64 % for the last seen position). Even the clean
   ball's nearest damaged player is right only 73 % of the time on these frames.
3. **The classifier saw nothing new.** The 11 features weigh 0.016 in log loss against the ball's
   1.08, and the team call with the ball missing stays at 85 %. What moved the score was noise around
   the decoder (its minimum possession went from 10 to 20 frames).
4. **The same source split again** (lab.md, rounds 5–6): Metrica up (+0.006 in the chain, 3/3), IDSSE
   down (−0.010, 1/7; IDSSE set −0.005, 1/7). It isn't evidence for anything here: the features changed
   nothing measurable per frame, and the per-match swings follow the decoder's picks.
5. **The fallback rule measured the wrong thing.** "Lower pooled mean error" picked the imputer; by
   short gaps, "within 2 m" or the nearest player's team, interpolation was better. A pooled metric
   dominated by long dead-time gaps was a poor proxy for what the classifier needs. Not rescued here
   (the choice was fixed beforehand), and the nearest-team table says interpolation wouldn't have
   carried more than robust's last-seen features already do.

## Caveats

- **One level of stacking, as every stacked input in the lab.** Each match's filled ball comes from an
  imputer trained without it, but the classifier's training matches were filled by imputers that had
  the held-out match among their training matches. The held-out match's own clean ball never reaches its own
  prediction.
- **The imputer learns mostly from IDSSE's dead time** (71 % of the frames with a target; IDSSE tracks a
  frozen dead ball), and Metrica adds only its live frames. A ball that moves in our stoppages would
  meet an imputer trained where it doesn't.
- **The bench's damaged players** (fragmented slots, 20.5 of 22 seen) limit what any ball estimate's
  nearest player can say: 73 % even with the clean ball.
- 41 % damage only; 10 matches; the start's picks are flat (team 4, dead 25 in every fold).

## Verdict

**Discarded.** The start fails against robust (all 10: −0.0015, 3/10, IDSSE −0.0051; IDSSE set:
+0.0005, 3/7, J03WR9 −0.0007), and the chain fails against v12 on both sets (all 10: −0.0050, 4/10,
IDSSE −0.0097; IDSSE set: −0.0048, 1/7), far from the +0.005 the round asks for. The ball isn't recoverable from this tracking in the stretches where robust needs
it, and the classifier already reads what the ends of each gap say. `bench/lab_ballfill.py` stays as a
record; `impute_inputs` and `measure` are reusable if a better ball source (broadcast, event data)
ever narrows the gaps.
