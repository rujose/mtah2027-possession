# Inside v12: how it works, what works, what doesn't

An X-ray of v12 on the 10 bench matches, to find what to improve. Every letter here comes from
models trained without the match (out of fold), with v12's settings. The truth is read on every
match, so none of this may pick a setting; it says where to look.

```bash
.venv/bin/python -m bench.inside_v12            # every table below (~3 min)
.venv/bin/python -m bench.lab_weighted --chains   # the chain with and without v9's snap and v11's move (~30 min)
.venv/bin/python -m bench.lab_budget --labels v12   # what each kind of error is worth
```

Tables: `bench/results/inside_v12.json`. Every error, one row each: `bench/results/inside_v12_errors.csv`.

## The chain, stage by stage

| # | stage | what it does | settings | code |
|---|---|---|---|---|
| 0 | **inputs** | the earlier models' answers on the match: keeper repair, v1's clues, P(dead), P(home restarts), PathCRF's P(home), v6's and v7's decoder costs | as on the bench | `v12.inputs` |
| 1 | **robust** | a classifier gives P(home), P(away), P(dead) for each frame from 97 numbers: the ball (seen, speed, course, where), each team's distance to it, players near it, teams' speed and shape, touches, passes, and stage 0's answers. A ball still for 1 s or more counts as missing | trained on IDSSE's matches, every 10th frame, each also with the teams swapped | `possession/lab_learned_features.py`, `bench/lab_robust.py` |
| 2 | **decoder** | turns the frame answers into one sequence of possessions: the cheapest path through −log P, paying a price for each switch, with a minimum length | team switch 4, dead switch 25, minimum possession 10 frames (0.4 s) | `possession/decoder.py` |
| 3 | **snap, move** | v9: a restart moves to the seen pass. v11: a change of team moves to the nearest touch by the new team | ±5/1 s, ±1 s | `possession/v9.py`, `v11.py` |
| 4 | **short** | looks at each stretch where the other team touches the ball inside a possession (or last before a stoppage). A logistic regression on 10 numbers decides if it's real; if so, that stretch becomes theirs | 0.5–5 s long, at least 50 % sure | `bench/lab_short.py` |
| 5 | **segments** | a classifier reads each possession as a whole (length, the ball, who's near, touches by each team, PathCRF, costs, neighbours) and gives P(wrong team), P(spurious). Flips the team, or merges it into a stoppage | flip at 60 %, merge at 70 % | `bench/lab_segments.py` |
| 6 | **timing** | for each change of team, lists candidate frames within 3 s (touches, the new team reaching the ball or becoming nearest); a ranker picks one. Never crosses another change | moves at 35 % or more | `bench/lab_timing.py` |

## Where the score comes from

Mean of the 10 matches. "Frames right" is the share of graded frames with the right label.

| after | spellF1 | precision | recall | possessions (true 411) | frames right |
|---|---|---|---|---|---|
| v11, for reference | 0.573 | 0.622 | 0.533 | 349 | 87.8 % |
| robust, frame by frame | **0.385** | 0.276 | 0.641 | **952** | 92.2 % |
| + decoder | **0.672** | 0.736 | 0.620 | 342 | 92.3 % |
| + snap and move | 0.675 | 0.739 | 0.622 | 342 | 92.3 % |
| + short | 0.689 | 0.733 | 0.653 | 362 | 92.5 % |
| + segments | 0.699 | 0.763 | 0.647 | 344 | 92.4 % |
| + timing (v12) | **0.704** | 0.768 | 0.652 | 344 | 92.6 % |

Three things to take from it:

1. **Almost all of v12's gain over v11 is better frames, then the decoder.** Frames right go from
   87.8 % to 92.2 %.
2. **Frames alone score badly.** At 92.2 % of frames right, robust scores 0.385: it flickers into
   952 possessions. The decoder doesn't change the frames (92.3 %) and takes the score to 0.672.
   The metric counts possessions, not frames.
3. **Everything after the decoder adds +0.032 in total.**

### What each stage changes (frames per match)

| stage | frames changed | made right | made wrong | neither | net |
|---|---|---|---|---|---|
| snap and move | 664 | 315 | **340** | 8 | **−25** |
| short | 501 | 362 | 124 | 16 | +238 |
| segments | 455 | 193 | 211 | 51 | −18 |
| timing | 439 | 332 | 95 | 13 | +237 |

Segments loses frames and still gains score: removing a wrong possession matters more than its
frames. Snap and move lose frames and gain almost nothing, but the chain doesn't reliably do better
without them: see "Tested".

## How the score is lost

| IoU | precision | recall | F1 |
|---|---|---|---|
| 0.3 | 0.864 | 0.731 | **0.790** |
| 0.4 | 0.827 | 0.701 | 0.757 |
| 0.5 | 0.778 | 0.660 | 0.713 |
| 0.6 | 0.720 | 0.612 | 0.660 |
| 0.7 | 0.652 | 0.555 | **0.598** |

spellF1 averages this column. So two separate losses:
- **Possessions not found at all:** even at the loosest bar, F1 is 0.790 (0.21 lost). 114 true
  possessions a match have no match, and 47 predicted ones have none.
- **Possessions found with loose edges:** with perfect edges every row would be 0.790, so edges cost
  **+0.086**.

Recall is the weaker side (0.652 against 0.768), and the misses are short:

| true length | true per match | found at IoU 0.3 | at 0.7 | missed per match | predicted right at 0.3 |
|---|---|---|---|---|---|
| under 1 s | 45.7 | **19 %** | 14 % | **36.9** | 53 % |
| 1–2 s | 68.6 | **51 %** | 34 % | **33.6** | 69 % |
| 2–5 s | 120.6 | 73 % | 47 % | **33.2** | 84 % |
| 5–10 s | 76.2 | 91 % | 67 % | 7.1 | 92 % |
| 10–20 s | 56.4 | 97 % | 86 % | 1.9 | 98 % |
| over 20 s | 43.9 | 97 % | 86 % | 1.5 | 99 % |

**Possessions over 5 s are solved at the loose bar. Under 2 s, most are missed.**

## Inside robust: the classifier alone

| truth ↓, its top answer → | H | A | D |
|---|---|---|---|
| H | 88.2 % | 7.0 % | 4.8 % |
| A | 7.5 % | 88.8 % | 3.7 % |
| D | 1.5 % | 1.3 % | 97.2 % |

- **The team is the hard part:** the dead ball is 97 % right. Which team, in play: 91.7 %.
- **The ball matters:** the team is right 93.3 % of the time with the ball seen, 86.3 % with it missing.
- **It knows when it's unsure, and is honest about it:**

| its confidence | share of frames | right |
|---|---|---|
| over 99 % | 59 % | 98.3 % |
| 90–99 % | 22 % | 94.1 % |
| 70–90 % | 11 % | 78.1 % |
| 50–70 % | 7 % | 58.7 % |
| under 50 % | 1 % | 43.8 % |

- **What it reads** (rise in held-out log loss when a family is shuffled): the ball **1.04**, v6's costs
  0.15, v7's costs 0.05, PathCRF 0.03, players at the ball 0.02, touches 0.02, everything else under
  0.01. Correlated families hide each other, so only the order means much. The players' own positions
  and shape add little by themselves.

## The missed possessions, up close

114 a match (the metric's one-to-one match at IoU 0.3), median 1.5 s:

| what | share |
|---|---|
| between two possessions of the other team | 62 % |
| next to a stoppage | 38 % |
| **a player of that team within 1 m of the seen ball for 3+ frames ("visible")** | **35 %** |
| a touch by that team is detected inside it | 39 % |
| frames with the ball seen (mean) | 58 % |
| v12 says, over its frames: the other team / dead / the right team | 50 % / 17 % / 33 % |

| if written in with the truth | spellF1 gain |
|---|---|
| the visible ones | **+0.058** |
| the rest | +0.135 |

**So the realistic ceiling here is about +0.06, not +0.19:** most missed possessions leave no player
within 1 m of a seen ball, in this damaged tracking.

**It depends on the source.** Visible: 25–36 % of the misses on IDSSE (104–208 a match), 57–62 % on
Metrica (56–67 a match). Metrica is the source closest to our tracking, so there the misses are fewer
and more of them can be reached. By length: 27 % under 1 s, 31 % at 1–2 s, 37 % at 2–5 s, 71 % over 5 s.

## The spurious possessions

47 a match, median 2.0 s. Over their frames, the truth says: the other team 52 %, the same team 37 %,
dead 10 %. 54 % are mostly the other team's (wrong team), 35 % mostly their own (a true possession
cut so badly neither piece matches), 10 % mostly dead.

## Match by match

| match | source | spellF1 | precision | recall | predicted | true |
|---|---|---|---|---|---|---|
| J03WMX | IDSSE | 0.599 | 0.740 | 0.504 | 348 | **511** |
| J03WN1 | IDSSE | 0.624 | 0.713 | 0.554 | 389 | **500** |
| J03WQQ | IDSSE | 0.648 | 0.679 | 0.620 | 366 | 401 |
| J03WR9 | IDSSE | 0.681 | 0.747 | 0.626 | 310 | 370 |
| J03WOH | IDSSE | 0.696 | 0.770 | 0.636 | 332 | 402 |
| J03WOY | IDSSE | 0.696 | 0.774 | 0.632 | 372 | 456 |
| J03WPY | IDSSE | 0.704 | 0.785 | 0.638 | 342 | 421 |
| metrica1 | Metrica | 0.796 | 0.826 | 0.768 | 329 | 354 |
| metrica2 | Metrica | 0.795 | 0.819 | 0.772 | 300 | 318 |
| metrica3 | Metrica | 0.797 | 0.826 | 0.770 | 355 | 381 |

The score falls as the truth has more possessions (correlation −0.82; −0.70 within IDSSE): **how many
short possessions a match has largely sets its score.** Our match's count isn't something we look
at, so its level stays uncertain. The metric's own note: on one match, gaps under 0.04 aren't
separable (`SPELL_NOISE_BAND`).

## One minute, stage by stage

J03WR9, frames 17,005–18,505. Each entry: label and seconds.

```
truth            D13.4 H4.7 A6.5 H20.5 A13.0 H0.2 A0.2 H1.5
+ decoder        D14.1 A6.6 H2.4 A2.3 H19.9 A10.1 H4.7
+ snap and move  D15.0 A4.8 H3.3 A2.3 H19.8 A9.9 H4.9
+ short          D15.0 A0.4 H2.8 A1.6 H3.3 A2.3 H19.8 A9.9 H3.2 A1.6
+ segments       D15.0 A0.4 H2.8 A1.6 H25.4 A9.9 H3.2 A1.6
+ timing (v12)   D15.0 A0.4 H2.7 A1.8 H25.4 A9.9 H3.2 A1.6
```

- **The restart:** the truth gives it to H at 13.4 s. The decoder says A, for 6.6 s.
- **Short rescues it:** it writes H for 2.8 s (15.4–18.2 s), nearly the true H (13.4–18.1 s), but
  leaves a 0.4 s A blip that nothing removes (spurious).
- **Then a wrong flip:** the true A runs 18.1–24.6 s. After short we have A 1.6, H 3.3, A 2.3: the
  middle is wrong. Segments flips the wrong piece: the A 2.3 becomes H, not the H 3.3 becoming A. So
  the true A 6.5 s keeps only 1.8 s (IoU 0.28: missed), though H 25.4 s still matches the true H
  20.5 s (IoU 0.81).
- **Edges:** the last A ends 2.9 s early (IoU 0.76): it counts at every bar, so nothing is lost.

## What's working

- **The classifier:** 92 % of frames, honest about its doubt, and it transfers to Metrica's matches
  without having trained on them.
- **The decoder:** the largest single step.
- **Long possessions:** over 10 s, 97 % found, 86 % even at IoU 0.7.
- **Short and timing:** three frames right for each one wrong.

## What isn't

- **Snap and move** on top of the learned model: about neutral now (see "Tested").
- **Possessions under 2 s:** 61 % of the misses, partly invisible in the tracking.
- **Which team, with the ball missing:** 86 %.
- **Segments' flips:** it loses frames overall, and flips the wrong piece when a true possession is cut
  into three (the minute above).

## Tested (2026-11-06): both failed the bar

[lab/weighted.md](lab/weighted.md) has the numbers. Against v12, with each match on its own held-out
settings:

| lead | all 10 | IDSSE's 7 | Metrica's 3 | IDSSE set |
|---|---|---|---|---|
| **drop snap and move** | +0.0022, 6/10 | −0.0024 | +0.0129 | −0.0016, 3/7 |
| **weight training by possession** | −0.0067, 4/10 | +0.0025 | −0.0282 | −0.0064, 2/7 |

**A first test of dropping snap and move gave +0.0045 (7/10), with one setting on every match; that gain
didn't hold.** Gains near 0.002 on this bench are noise. Weighting does find more short possessions on
its own, but `short` already gets most of them, and Metrica's matches pay for it.

**What this changes:** the two sources now pull in opposite directions. Metrica's truth has fewer short
possessions, so pushing recall costs precision there. Which source our match's labels resemble matters
more than the next small gain.

## Leads, by what I'd try first

| # | lead | why | size | status |
|---|---|---|---|---|
| 1 | **Which source to trust** | Changes that help IDSSE hurt Metrica and the reverse: see below | decides how to read every gain below | open: yours or the organisers' call |
| 2 | **A switch price that changes frame by frame** | The decoder charges a fixed 4 to change team anywhere. A model of "does the ball change hands here?" could make it cheap at real touches and dear elsewhere: more short possessions without flicker | unmeasured; part of the visible +0.058 and of the edges +0.086 | to test |
| 3 | **Timing for every edge** | It only moves changes between teams. Possessions also start and end at stoppages | part of +0.086 (stoppage start and restart timing +0.012 in the budget) | to test |
| 4 | **Settings at their grids' edge** | Timing (3 s, 35 %), short (50 %), recall (40 %) were all still rising; their settings were chosen before robust existed | small; recall on this chain +0.0044, 6/10 | to test |
| 5 | **Segments that sees a cut possession** | Judge three pieces together (the minute above) rather than one at a time | part of +0.038 and +0.039 | to test |
| 6 | **Let robust learn from Metrica too** | It trains on IDSSE because Metrica's blanked ball looked like cheating; blanking now makes IDSSE look the same | learned trained on all reached 0.7725 on Metrica; artifact risk, and Metrica's licence is open | to test |
| 7 | **Better team evidence while the ball is missing** | 86 % against 93 % | part of +0.039 | ideas needed |

**Lead 1, what we know from the organisers' text alone:** CHALLENGE.md says the truth is derived mechanically
from Metrica's corrected tracking, not from events, and that the public 5 minutes hold 17 possessions (3.4
a minute). Per graded minute, Metrica's samples have 3.67 (3.38–3.98) and IDSSE's matches 4.58 (3.78–5.30).
That leans to Metrica, but weakly: 5 minutes from one match, 3 Metrica matches, labels we built from events,
and no licence yet. It's stated here, not used: changing how the bar weighs the sources would need deciding
before the next test, not after, and it mustn't rest on the public truth (rules: no calibrating on it).

Not worth chasing: the other +0.135 of missed possessions, unless new evidence appears, and missed
stoppages (+0.004: robust solved them; it added false stoppages, +0.012).

**Be careful with small gains.** The bench is the same 10 matches every time, and a comparison moved by
0.002 from settings alone (see "Tested"). Below about +0.005, don't believe a gain without a reason.

## Look for yourself

Open a bench match in the viewer (the match picker, or `http://localhost:8765/?match=J03WMX`). It
shows the truth, so you can look at failures freely. New lanes:

| lane | what it shows |
|---|---|
| **errors: missed** | true possessions v12 misses (IoU 0.3), in the team's colour |
| **errors: spurious** | v12's possessions with no true match |
| **v12: robust, + short, + segments** | the chain after each stage |
| **v12: P(dead), v12: P(home)** | what robust says frame by frame, before the decoder |

**To walk through the failures:** under "Watch changes in", leave only **errors: missed** on, then
press **N** and **P**. Turn on "clean tracking" to compare what the models see with what happened.

**To sort them:** `bench/results/inside_v12_errors.csv` has each one: match, frame, seconds, team, how
much of it the ball was seen, whether it's visible, what v12 said. Paste the frame in the viewer's
jump box.

Your earlier observations (the keeper, the late restarts) came from watching like this. Two questions
worth asking while watching: what do the missed possessions that *are* visible have in common, and
when does robust's P(home) already flip but the decoder ignores it?
