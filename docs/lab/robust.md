# Lab candidate `robust`: `learned`, unable to read a still ball

| | |
|---|---|
| **status** | **built 2026-11-04, lab round 4; clears the bar against v11 on both bench sets**: held out spellF1 **0.6728** against v11's 0.5727 (**+0.1001**), better on **10/10**; IDSSE set **0.6365** against 0.5625 (**+0.0739**), **7/7**. **Against learned: +0.0119 (8/10), but IDSSE −0.0011 (5/7; IDSSE set −0.0012, 2/7) and Metrica +0.042 (a convention match, not robustness)**: the still ball's information moved into "ball missing", so robustness costs ~nothing on the bench, and the bench can't show what it buys |
| **what** | `learned` (one gradient-boosted P(H / A / D) per frame, trained on IDSSE, its −log P decoded) retrained so it can't use "a seen ball standing still means dead". Two variants in one grid, both fixed before any held-out run: **`blank`**, every run of a seen ball standing still for ≥ 1 s becomes "ball missing", on every match, in training and in prediction, and learned's 97 features are recomputed on that tracking; **`no_speed`**, learned's features on the tracking as it is, without the ball's 13 speed features |
| **stage** | `decode`: replaces v7's costs (v9 and v11 run after it, on the original tracking) |
| **code** | `bench/lab_robust.py` (new; imports `bench.lab_learned` and `possession.lab_learned_features`, changes neither) |
| **run** | `python -m bench.lab_robust --measure [--set idsse]` (seconds: what the blanking removes); `python -m bench.lab_robust [--set idsse]` (~20 / ~8 min, ~0.5 GB; after `python -m bench.lab --build` and `python -m bench.lab_learned`); `python -m bench.lab_robust --ball-parts` (~6 min, a diagnostic) |
| **results** | `bench/results/lab_robust[_idsse].json` (the judge's report against v11, plus `vs_learned`, `d_f1`, `variants`, `importances`), `lab_robust_grid[_idsse].csv`, `lab_robust_frames[_idsse].csv` (per fold: accuracy on the blanked frames and elsewhere, against learned's; importances), `lab_robust_blanking[_idsse].csv` (the measurement), `lab_robust_ball_parts.csv`; held-out P per match in `data/external/lab_robust[_idsse]/p_<variant>_<match>.npy`, blanked features in `data/external/lab_robust[_idsse]/<match>.npz` (gitignored) |
| **mix with** | `bench.lab_robust.decode_held_out(match, which)` (the saved held-out P decoded with `CHOSEN`, no refit) → letters at v7's level, then `bench.lab_robust.finish(match, letters, which)`. P itself: `held_out_proba(match, which, variant)`; a decoder-stage mix: `cost_from(match, held_out_proba(match, which))` |
| **chosen** | `CHOSEN = {"variant": "blank", "blend": 0.0, "switch_team": 4.0, "switch_dead": 25.0, "min_possession": 10}` on all 10 and on IDSSE alone (every fold picked `blank`); `CHOSEN_NO_SPEED` = `no_speed`, no blend, team 4, dead 25, min 20 |

## Why

`learned` ([learned.md](learned.md)) is the lab's strongest candidate (held out 0.6610 against
v11's 0.5727 on the 10 bench matches; round 3's best chain starts from it, 0.7019), and its main
input is the ball: shuffling the ball family raises its held-out log loss by **1.52**, six times
the next family (v6's costs, 0.26). The bench's two providers show dead time in ways our match may
not ([learned.md](learned.md), "The bench's dead-time ball"):

- **IDSSE freezes the ball in dead time**: when it's seen while dead, it stands still 99 % of the
  time (median 0.07 m/s, the jitter we add). In a real stoppage the ball is fetched, carried, thrown
  back.
- **Metrica's samples blank it** (missing on 94–99.5 % of dead frames).
- **Ours** (Metrica tracking): the ball is missing 83 % of the time players stand still against
  29.5 % while they move ([../bench.md](../bench.md), label-free), so "no ball" should transfer.
  Whether our *seen* ball stands still in stoppages is unknown, and is not looked at (the
  organisers' rules: no match data to AI services).

So the model should not be able to use "a seen ball standing still means dead". Three cases for our
match, and what each model does in them:

| our match | learned | `blank` | `no_speed` |
|---|---|---|---|
| **A.** the seen ball stands still in stoppages, like IDSSE's | reads it as dead (right) | blanked: "missing", as in its training (right) | judged by where it lies, the gaps, the players, the earlier models |
| **B.** the seen ball moves in stoppages (fetched, carried, thrown) | trained where a seen dead ball ~never moves: reads it as live (wrong) | **the same**: in its training a seen ball is almost never dead (0.3 % of dead frames keep one), so it reads it as live | can't see the speed, so a moving dead ball looks like IDSSE's frozen one; judged as in A (less exposed, not immune: touches and "a player on the ball" still say live) |
| **C.** a still ball in live play, more often than the bench's (a keeper holding it, a stuck detection) | false stoppages | blanked: "missing" in live play, common in its training (30 % of live frames) | can't see the speed |

`blank` is what the brief asked for: a still ball doesn't exist in its inputs, so its bench score
can't come from reading one, and case C can't hurt it. It does **not** protect against case B, and
it trades one convention for another (below, "The blanking, measured"). `no_speed`, the brief's
fallback, is measured beside it in the same grid because it is the one that bears on case B.

## How it works

1. **The blanking** (`still_frames`, `blanked`), label-free, the same on every match, in training
   and in prediction: a frame is *still* when the ball is seen and moves under 0.3 m/s over the
   0.2 s before it or the 0.2 s after it (either window, where known; unknown on both sides is not
   still). Every run of still frames, within a half, lasting at least **1 s** loses its ball (set to
   missing). Nothing else in the tracking changes.
2. **`blank`**: learned's 97 features (`possession/lab_learned_features.py`, unchanged) recomputed
   on the blanked tracking, **touches and passes included** (v11's and v9's detectors run on it too),
   so the model sees a feed in which a still ball never existed. Cached in
   `data/external/lab_robust[_idsse]/<match>.npz` with the mask.
3. **`no_speed`**: learned's own cached features, on the tracking as it is, without the 13 that
   measure the ball's speed: speed over the 0.2 s before and after, change of speed, ΔV, change of
   course, and the 1 s / 3 s means of the speed after and of ΔV. Kept: seen or not, s since / until
   seen, where it is (x from home's goal, distance from the touchline, how far outside), and every
   other family (touches and passes included).
4. **Everything else is learned's, unchanged**: IDSSE-only training, every 10th graded frame with
   the teams exchanged, the same trees and seed, P the mean of both orientations, clipped to
   0.01–0.99, rounded to 3 decimals; −log P (optionally blended with v7's costs) decoded with the
   same decoder grid; then **v9's restart snap and v11's touch move on the original tracking**
   (`finish` is learned's), as every post stage reads it. The earlier models' outputs (P(dead), the
   restart P, PathCRF, v6's and v7's costs) are the harness's, from the original tracking.
5. **Judged** by `lab.judge` over the whole grid against v11 (the bar), and for the record against
   learned's held-out score on each match.

## Every constant, and why

| constant | value | why |
|---|---|---|
| still speed | **0.3 m/s over 0.2 s** (`lab_learned_check.STILL`, `llf.LOOK`) | the measure learned's caveat was stated in, so the numbers compare: the jitter alone gives ~0.06 m/s, a ball being carried > 0.5 |
| either window | the 0.2 s before **or** after | covers the whole stretch the ball lies still, ends included: with "after" only, the last 0.2 s before a gap or a kick stay seen and still; with "both", 0.2 s at each end do. The kick that ends a still run is blanked too (its frame is still over the 0.2 s before it) |
| **minimum run** | **1 s** (25 frames) | by reasoning, before measuring: IDSSE's frozen dead ball lies still for seconds, so 1 s catches nearly all of it, while live play rarely holds a ball under 6 cm per 0.2 s for a whole second (a keeper standing with it; a ball stopped dead). The model's own context windows are 1 s and 3 s, so a shorter still fragment left seen says little. 2 s would leave 1–2 s fragments of frozen ball carrying the cue |
| fallback, fixed beforehand | 2 s if 1 s blanked > 2 % of either source's live frames | a rule that removes the cue mustn't also remove a visible share of live play. Not triggered (0.86 % IDSSE, 0.54 % Metrica) |
| within a half | runs never cross a half | the halves are separate in every other piece |
| touches, passes on the blanked feed | yes | the brief: the learned features recomputed on the preprocessed tracking. Cost: the kick that ends a still run is no longer seen as a touch (the detector needs the ball before the kick). `finish` keeps the original touches and passes |
| `no_speed`'s 13 features | the names starting `ball_speed`, `ball_accel`, `ball_dv`, `ball_turn` | the brief's fallback: drop the speed / stillness features, keep missingness, position and gaps |
| variants | `blank` first, then `no_speed` | ties go to the first. learned's `no_pathcrf` isn't repeated: it failed learned's bar and no fold of learned picked it, and keeping it would double the run |
| decoder grid | learned's: blend 0 / 0.5 (v7's costs), team price 0.5–16, dead price 50 / 25 / 10, minimum possession 20 / 10 | the brief: same procedure and grid. 72 settings per variant, 144 in all (learned's count) |
| everything else | learned's `EVERY`, `TRAIN_SOURCES`, `TREES`, `CLIP`, `DECIMALS`, `SWAP` | imported, not copied |

Fixed before any held-out run: the rule and its 1 s (measured first, below, with the fallback
written down beforehand), both variants, the grid, and that the judge runs over the whole grid.

## The blanking, measured (before any held-out run)

`python -m bench.lab_robust --measure` (the truth only to count; the IDSSE set's tracking is the
same, so its numbers are IDSSE's below). Shares of each source's frames, pooled; ranges over the
matches.

| minimum run 1 s (the rule) | IDSSE (7) | Metrica (3) |
|---|---|---|
| **live frames blanked** | **0.86 %** (0.36–1.56 %) | **0.54 %** (0.27–0.79 %) |
| **dead frames blanked** | **44.1 %** (41.4–49.0 %) | **2.3 %** (0–5.2 %) |
| … of the dead frames whose ball was seen (mean of matches) | 99.2 % | 42 % (0–87 %) |
| all frames blanked | 17–23 % | 0.4–2.6 % |
| ball missing on dead frames: before → after | 55.5 % → **99.7 %** | 96.8 % → 99.1 % |
| ball missing on live frames: before → after | 29.2 % → 30.1 % | 10.7 % → 11.3 % |
| seen ball still (< 0.3 m/s over the next 0.2 s), dead frames: before → after | 99.2 % → 87.9 % (of the 0.3 % of dead frames still seen) | 43 % → 1.3 % |
| … live frames: before → after | 1.8 % → 0.7 % | 1.1 % → 0.6 % |

Other minimum runs, for information (`bench/results/lab_robust_blanking.csv`):

| minimum run | live blanked, IDSSE / Metrica | dead blanked, IDSSE / Metrica | IDSSE dead frames still seen after | … of them still |
|---|---|---|---|---|
| 0.5 s | 1.22 % / 0.93 % | 44.4 % / 2.3 % | 0.1 % | 58 % |
| **1 s** | **0.86 % / 0.54 %** | **44.1 % / 2.3 %** | **0.3 %** | **88 %** |
| 2 s | 0.49 % / 0.32 % | 43.5 % / 2.3 % | 1.0 % | 94 % |
| 3 s | 0.32 % / 0.17 % | 42.1 % / 2.2 % | 2.4 % | 96.5 % |

What this says:

- **The still ball is gone**: after blanking, 0.3 % of IDSSE's dead frames keep a seen ball (short
  fragments, under 1 s), against 44.5 % before. Live play loses its ball on under 1 % of frames, so
  the fallback (2 s) wasn't needed.
- **It trades one convention for another.** IDSSE's dead time becomes 99.7 % "ball missing"
  against 30.1 % of live time: the near-perfect "no ball = dead" that Metrica's samples show (97 % /
  11 %), and that learned avoided by not training on Metrica. The frozen ball's information isn't
  removed, it's re-encoded as missingness. On ours, "no ball" while players are stopped is 83 %
  against 29.5 % while they move *before* blanking; after it, ours gets whatever its own still runs
  add: close to IDSSE's 99 % if our stopped ball stands still (case A), still ~83 % if it moves (B).

## Results (2026-11-04)

### The bar (against v11)

| | all 10 matches | IDSSE only (7, its own chain and folds) |
|---|---|---|
| **held out spellF1** | **0.6728** against 0.5727 (**+0.1001**), better on **10/10** | **0.6365** against 0.5625 (**+0.0739**), **7/7** |
| IDSSE | 0.6422 against 0.5673 (+0.0749), 7/7 | as above |
| Metrica | 0.7443 against 0.5855 (+0.1588), 3/3 | — |
| **PathCRF-unseen** | **0.7237** against 0.5842 (+0.1395), 4/4 | J03WR9: 0.6589 against 0.5790 (+0.0799) |
| macroF1 | 0.9145 against 0.8705 (learned 0.9107) | 0.9115 against 0.8757 (learned 0.9161) |
| D's F1 | **0.957** against 0.908 (learned 0.951); IDSSE 0.967 (learned 0.971), Metrica 0.933 (learned 0.905) | 0.966 against 0.928 (learned 0.972) |
| possessions a match | 341 (learned 319, v11 349, true 411) | 340 |
| chosen | `blank`, no blend, team 4, dead 25, min 10. **Every fold picked `blank`**, no blend, team 4, dead 25 (min 10 in 7, 20 in 3) | the same; folds: `blank`, team 4 or 8, dead 25, min 10 |
| **passes** | **yes** (all four checks) | **yes** |

The grid is flat at the top: `blank` at learned's own setting (team 8, dead 50, min 20) gives 0.671
in sample against 0.675 at its chosen one, so the result isn't the decoder's re-choice. Blending
v7's costs back in hurts again (best 0.662).

### Against learned, match by match (the record, not the bar)

| | all 10 | IDSSE set (7) |
|---|---|---|
| **`blank` (the candidate)** | **+0.0119**, better on 8/10 | **−0.0012**, better on 2/7 |
| … IDSSE | **−0.0011**, 5/7 | −0.0012, 2/7 |
| … Metrica | **+0.0420**, 3/3 | — |
| … PathCRF-unseen | +0.0317, 4/4 | J03WR9 +0.0132 |
| `no_speed` alone | 0.6508 (+0.0781 vs v11, 10/10, passes the bar); **−0.0101** vs learned, 4/10 | 0.6280 (+0.0655 vs v11, 7/7, passes); **−0.0096**, 1/7 |
| … IDSSE | **−0.0172**, 1/7 | −0.0096, 1/7 |
| … Metrica | +0.0064, 3/3 | — |
| … PathCRF-unseen | −0.0011, 3/4 | J03WR9 +0.0059 |
| `no_speed` macroF1 / D's F1 | 0.9047 / 0.944 | 0.9096 / 0.963 |

Per match (all 10; each with the setting chosen on the other 9):

| match | held out | v11 | learned | vs learned | macroF1 / learned | possessions (true) | D's F1 / learned / v11 |
|---|---|---|---|---|---|---|---|
| J03WMX | 0.5791 | 0.5100 | 0.5726 | +0.0065 | 0.918 / 0.920 | 340 (511) | 0.968 / 0.973 / 0.922 |
| J03WN1 | 0.5907 | 0.5282 | 0.5928 | −0.0021 | 0.886 / 0.892 | 385 (500) | 0.963 / 0.972 / 0.942 |
| J03WOH | 0.6774 | 0.5676 | 0.6691 | +0.0083 | 0.918 / 0.920 | 332 (402) | 0.969 / 0.975 / 0.923 |
| J03WOY | 0.6638 | 0.6045 | 0.6589 | +0.0048 | 0.904 / 0.904 | 375 (456) | 0.955 / 0.957 / 0.920 |
| J03WPY | 0.6790 | 0.6181 | 0.6746 | +0.0043 | 0.923 / 0.927 | 349 (421) | 0.975 / 0.978 / 0.937 |
| J03WQQ | 0.6436 | 0.5624 | 0.6738 | **−0.0302** | 0.915 / 0.921 | 369 (401) | 0.961 / 0.967 / 0.912 |
| **J03WR9** (unseen) | 0.6618 | 0.5801 | 0.6609 | +0.0009 | 0.928 / 0.929 | 310 (370) | 0.974 / 0.975 / 0.927 |
| metrica1 (unseen) | 0.7563 | 0.5934 | 0.7092 | +0.0471 | 0.928 / 0.911 | 324 (354) | 0.958 / 0.937 / 0.888 |
| metrica2 (unseen) | 0.7349 | 0.5811 | 0.6923 | +0.0426 | 0.897 / 0.879 | 284 (318) | 0.904 / 0.881 / 0.852 |
| metrica3 (unseen) | 0.7417 | 0.5821 | 0.7054 | +0.0363 | 0.928 / 0.902 | 339 (381) | 0.938 / 0.897 / 0.858 |
| **mean** | **0.6728** | 0.5727 | 0.6610 | +0.0119 | 0.915 / 0.911 | 341 (411) | **0.957** / 0.951 / 0.908 |

- **Per-frame accuracy** (argmax P, held out, mean of matches): 0.922 against learned's 0.919;
  IDSSE 0.924 / 0.928, **Metrica 0.917 / 0.897**. On the frames the rule blanks (20 % of IDSSE's
  graded frames), IDSSE accuracy falls from **0.994 to 0.978** (IDSSE set 0.993 to 0.977): learned read them from the still
  ball almost perfectly, `blank` reads them from the missing ball nearly as well. Held-out log loss
  0.274 against 0.273 (IDSSE set 0.199 against 0.183).
- **Fewer merges than learned** (341 possessions a match against 319; v11 349, true 411), from the
  lower team price (4 against 8).

### Importances (rise in held-out log loss when one family is shuffled, mean of the folds)

| family | learned | **`blank`** | `no_speed` | learned, IDSSE set | `blank`, IDSSE set | `no_speed`, IDSSE set |
|---|---|---|---|---|---|---|
| **the ball** | **1.518** | **1.043** | **0.052** | 1.600 | 1.032 | 0.078 |
| v6's costs | 0.263 | 0.149 | 0.153 | 0.502 | 0.305 | 0.557 |
| PathCRF | 0.034 | 0.032 | 0.029 | 0.056 | 0.065 | 0.062 |
| v7's costs | 0.031 | 0.049 | 0.052 | 0.024 | 0.027 | 0.023 |
| touches | 0.020 | 0.016 | **0.120** | 0.016 | 0.019 | 0.114 |
| players at the ball | 0.017 | 0.018 | 0.035 | 0.017 | 0.026 | 0.037 |
| passes | 0.002 | 0.006 | 0.033 | 0.002 | 0.006 | 0.038 |
| teams' shape and speed | 0.002 | 0.009 | 0.009 | 0.010 | 0.012 | 0.014 |
| players at the last / next seen ball | 0.002 | 0.005 | 0.004 | 0.003 | 0.006 | 0.005 |
| restart, P(dead) | ≤ 0.002 | ≤ 0.002 | ≤ 0.002 | ≤ 0.003 | ≤ 0.005 | ≤ 0.004 |

**Order, `blank`**: the ball (1.04), v6's costs, v7's costs, PathCRF, players at the ball, touches,
the rest. **The ball still dominates**, down a third. **`no_speed`**: v6's costs, **touches**,
v7's costs, the ball (0.05), players at the ball, passes, PathCRF: with the speed gone, the
touches and passes (v11's and v9's detectors, which read the ball's change of velocity) take up
part of it, 6× and 13× their weight in learned.

**Inside the ball family** (`--ball-parts`, a diagnostic chosen after the run, 3 refitted folds:
J03WMX, J03WR9, metrica1; `bench/results/lab_robust_ball_parts.csv`):

| shuffled | learned | `blank` | `no_speed` |
|---|---|---|---|
| seen, s since / until seen | 0.001 | 0.063 | 0.000 |
| the 13 speed features | **1.495** | 0.459 | (dropped) |
| where it is | 0.002 | 0.005 | 0.056 |
| the whole family | 1.581 | 1.091 | 0.051 |

Learned's reliance on the ball is its speed features. Read with care: every ball and
players-at-the-ball feature is NaN when the ball is missing, so a shuffled group also loses
"missing", and the trees pick whichever copy they like. So the split can't separate stillness
from missingness. What it does show is that `blank`'s ball still matters (1.09) once no still
ball is left to read: that is the missing ball (and the live ball's dynamics), now carrying the
dead-ball evidence the frozen ball carried.

## The cost of robustness, plainly

- **On IDSSE, nothing measurable: −0.0011 (5/7) on all 10, −0.0012 (2/7) on the IDSSE set.** But
  not because the model learned to do without the providers' dead-time ball: the blanking moves the
  frozen ball's information into "missing" (dead frames 55 % → 99.7 % missing), and the trees read it
  there almost as well (0.994 → 0.978 on those frames). **The bench can't measure what `blank`
  protects against**, because the bench has no dead-time ball that moves.
- **On Metrica, a gain (+0.042, 3/3) that isn't robustness.** After blanking, IDSSE's dead time
  looks like Metrica's samples (dead ⇒ no ball), so the model fits Metrica's convention better: D's
  F1 on Metrica 0.933 against 0.905, accuracy 0.917 against 0.897. The same thing happened, larger,
  when learned was trained on Metrica too (`--train-all-sources`: Metrica 0.7725, against `blank`'s
  0.7443 and learned's 0.7023). Count `blank`'s gain over learned as zero, not +0.012.
- **Dropping the ball's speed altogether (`no_speed`) costs −0.017 on IDSSE (1/7), −0.010 overall
  (4/10), −0.0096 on the IDSSE set (1/7)**, and still passes the bar against v11 on both sets
  (+0.078, +0.066). That is the honest price of a model that can't read the ball's speed at all; and
  even then the touches and passes read its movement indirectly.

## Failure modes and caveats

1. **`blank` doesn't protect against case B** (our ball moving in stoppages): in its training a seen
   ball is almost never dead, so a ball carried in a stoppage on ours reads as live, as in learned.
   Only `no_speed` bears on B, and only partly.
2. **It bets on "no ball in dead time" instead.** On blanked IDSSE, dead time is 99.7 % "missing"
   against 30.1 % of live time; on ours it is 83 % (players stopped) against 29.5 % (moving) before
   blanking, plus whatever our still runs add. That's closer to `blank`'s training than to learned's
   (55.5 % / 29.2 %), so on the missing-ball frames `blank` should be at least as well calibrated on
   ours as learned. By this reasoning (not measurable on the bench), in every case of the table in "Why" `blank`
   does no worse than learned, and it avoids case C. The share of our stopped frames that would still show a seen ball after blanking
   is the one unknown: it's learned.md's label-free check (the still share of the seen ball inside
   v1's "players stopped" stretches; bench IDSSE 98.0 %, Metrica 23.3 %), for Jose to run.
3. **The earlier models' outputs come from the original tracking.** v1's clues inside v6's and v7's
   costs read the ball (clue A: a slow ball with a player within 2 m counts against D; clue B: a ball
   beyond a line counts for D), so a frozen ball lying outside the pitch still reaches the model
   through v6's costs (0.15, down from 0.26), by *where* it is, not by its standing still. v4's
   P(dead), v6's restart P and PathCRF read the players only. Recomputing them on the blanked feed
   would mean re-running `bench.eval_pathcrf`, outside this candidate.
4. **A residue:** 0.3 % of IDSSE's dead frames keep a seen ball after blanking (fragments under 1 s),
   88 % of them still.
5. **The kick that ends a still run isn't a touch in the features** (the detector needs the ball
   before it); `finish` keeps the original touches, so v9's snap and v11's move are unaffected.
6. **Inputs on our match must be built as on the bench** (learned.md, caveat 3): the same blanking,
   the costs without tags, kick-off rule or keeper repair; trained on all 7 IDSSE matches.
7. **The setting choice leaks a little** (each match's setting chosen on scores of models trained
   with it, as learned); the choice is flat (every fold `blank`, no blend, dead 25).
8. **Only 41 % damage**, as every lab candidate; **winner's curse**: 144 settings, but against v11
   the gain is +0.10, 10/10.

## Verdict

**Keep: it clears the bar against v11 on both sets (+0.1001, 10/10; +0.0739, 7/7), every check, and
it can't read a still ball.** As the chain's start in place of learned: on IDSSE (the closer guide
to our match) it costs nothing measurable (−0.001), it is no worse than learned in any of the three
cases for our match and better in one (a still ball in live play), and its dead-ball evidence is the
missing ball, the one bench.md measured on ours. Its +0.042 on Metrica is a convention match, not a
gain to bank. It is **not** robust to a ball that moves in our stoppages; if Jose's label-free check
finds our seen ball rarely still while players stand (Metrica-like 23 %, not IDSSE's 98 %),
`no_speed` (−0.010 against learned, passes against v11) or learned's team half are the hedges.

For the mixer: `decode_held_out(match, which)` then `finish(match, letters, which)`; `no_speed` with
`decode_held_out(match, which, **CHOSEN_NO_SPEED)`. Round 3's post stages (stoppage, short, timing,
recall) should be re-checked on top: `blank` picks a lower team price than learned (4 against 8)
and predicts more possessions (341 against 319), so the stages that merge or add possessions may
pay differently.
