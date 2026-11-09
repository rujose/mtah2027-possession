# Lab candidate `weighted`: robust trained possession by possession

| | |
|---|---|
| **status** | **discarded** (2026-11-06). v12 stays as it is |
| **what** | robust with the frames of a true possession of length n weighing n^-alpha, so short possessions count more; and v12's chain with and without v9's snap and v11's move |
| **why** | [inside-v12.md](../inside-v12.md): the classifier trains on frames, the metric counts possessions, and most of what's missed is short |
| **code** | `bench/lab_weighted.py`; results in `bench/results/lab_weighted[_idsse].json`, `lab_weighted_chains[_idsse].json` |

## What was fixed before the run

| setting | value | why |
|---|---|---|
| alpha | 0.5, 1 | 1 gives every possession the same total weight; 0.5 is halfway from frames |
| dead frames | weight 1 | and each match's live frames keep their total, so the balance between live and dead doesn't move |
| decoder | robust's grid: team switch 0.5–16, dead switch 50, 25, 10, minimum 20 or 10 frames | the weights change the probabilities, so the prices are chosen again |
| everything else | robust's | blanked features, IDSSE-only training, every 10th graded frame, teams swapped, the same trees |

## The start alone, against robust

| | all 10: 0.6801 vs 0.6728 | IDSSE set: 0.6487 vs 0.6365 |
|---|---|---|
| overall | +0.0072, 8/10 | +0.0122, 6/7 |
| IDSSE's matches | **+0.0140, 7/7** | +0.0122, 6/7 |
| Metrica's matches | **−0.0085, 1/3** | — |
| PathCRF-unseen | −0.0021, 2/4 | +0.0082, 1/1 |
| bar | **fails** (a source drops; unseen drops) | passes |
| macroF1 | 0.9126 vs 0.9145 | 0.9087 vs 0.9115 |

It adds possessions (357 a match; 411 true), as intended. Folds picked alpha 0.5 six times and 1 four
times on `all`, with a team switch of 8 or 16 and a minimum of 10 frames.

## The four chains

Every post stage retrained on each chain's output, each stage on its setting for that match.

| chain | all 10 | IDSSE's 7 | Metrica's 3 | unseen 4 | IDSSE set |
|---|---|---|---|---|---|
| **robust, snap and move (v12)** | **0.7033** | 0.6674 | 0.7870 | 0.7617 | **0.6699** |
| robust, no snap or move | 0.7055 | 0.6651 | 0.7998 | 0.7723 | 0.6683 |
| weighted, snap and move | 0.6966 | 0.6700 | 0.7588 | 0.7422 | 0.6634 |
| weighted, no snap or move | 0.7010 | 0.6734 | 0.7656 | 0.7477 | 0.6674 |

| comparison | all 10 | matches better | bar | IDSSE set | matches better | bar |
|---|---|---|---|---|---|---|
| no snap or move vs v12 | +0.0022 | 6/10 | **fails** (IDSSE −0.0024) | −0.0016 | 3/7 | **fails** |
| weighted vs v12 | −0.0067 | 4/10 | **fails** | −0.0064 | 2/7 | **fails** |
| weighted, no snap or move vs v12 | −0.0023 | 5/10 | **fails** | −0.0024 | 4/7 | **fails** |

## What it taught us

1. **Dropping snap and move isn't the gain it first looked like.** The first test gave +0.0045 (7/10),
   with one setting on every match. Judged with each match's own held-out settings, it's +0.0022
   (6/10), and it loses on the IDSSE set. **So the same comparison moved by 0.002 from settings alone:
   gains of that size on this bench are noise.** Snap and move are about neutral on top of robust.
2. **Weighting's gain is the same gain as `short`'s.** Alone it adds +0.014 on IDSSE; after short and
   the rest, +0.003. They go after the same possessions.
3. **The two sources pull apart.** Weighting helps IDSSE and hurts Metrica (−0.028 in the chain, 0/3);
   dropping snap and move helps Metrica (+0.013, 3/3) and hurts IDSSE. Metrica's truth has fewer short
   possessions (318–381 a match against 370–511), so pushing for more of them costs precision there.
   Our match is Metrica-tracked, so the direction of this risk matters more than the mean.
