# Lab candidate `stoppage`: each stoppage starts when the ball is seen leaving the pitch

| | |
|---|---|
| **status** | **built 2026-11-04, lab round 1; clears the bar** on both bench sets: held out spellF1 0.5777 against v11's 0.5727 (**+0.0050**), better on **10/10**; IDSSE only +0.0032, better on 5/7 |
| **what** | idea 10e, v9's mirror: each change from a team to **D** moves to **3 frames after the ball is seen crossing a pitch line**, when that exit is seen from 4 s before the change to 2 s after. Which team, and how many possessions and stoppages, never change |
| **stage** | `post`: works on any letters (judged on v11's) |
| **code** | `bench/lab_stoppage.py` (the rule, the convention measurement, the leave-one-out run) |
| **run** | `python -m bench.lab_stoppage` (~5 s for both sets, ~0.6 GB; after `python -m bench.lab --build`) |
| **results** | `bench/results/lab_stoppage[_idsse].json`, `lab_stoppage_grid[_idsse].csv` |
| **mix with** | `bench.lab_stoppage.apply(match, letters, back=4.0, ahead=2.0)` (the offset defaults to 3 frames; see [Settings](#every-setting-and-why)) |
| **not built** | idea 10h (restart timing with no seen pass) and a post-hoc rule for missed stoppages: measured, no clean rule (below) |

## Why

**The error budget** ([../ideas.md](../ideas.md), "Where v9's remaining spellF1 is"): stoppage
start timing (team → D) is worth +0.015 if fixed perfectly, missed stoppages +0.022, restart
timing left after v9 +0.013. A possession's end at a stoppage shapes its IoU just like its start.

**v11's stoppage starts are loose.** Of v11's team → D changes that match a true one (same team
before, within 5 s), the lag is centred (median −0.16 to +0.56 s per match) but wide: quartiles
from −0.24 to −1.32 s and from +0.40 to +1.48 s, and only 9–36 % per match land within 0.2 s.
As with restarts before v9, the change comes from the dead-ball model and the decoder, which
have no sharp moment to land on.

**The ceiling, measured here:** moving each matched v11 stoppage start exactly onto the true one
lifts the bench from 0.5727 to **0.5852 (+0.0125)**. Restricted to the true starts with a seen exit
nearby (what a ball-out rule can reach): **0.5784 (+0.0057)**.

## The truth's convention (measured first, as v9 did for restarts)

**IDSSE, clean ball (710 true team → D changes):**
- the ball is **outside the pitch at the first D frame in 469 (66 %)**, on average **3–4 frames
  (0.12–0.16 s) after it crossed the line**; in the other 241 the ball is inside: whistle stoppages
  (fouls, offsides, injuries). This repeats `bench/labels.py` check 2.
- **the last touch is no marker:** the last seen touch before the change is by the team labelled
  before it in only 384 of 702 (55 %). The truth ends the possession at the crossing, not at the
  last touch, whose distance from the line (and so its time before the crossing) varies.

**Metrica's samples, clean ball (blanked while dead, so read just before the change):** the ball's
last position seen within 1 s before the change is outside in 194 of 287; where the crossing is
seen (193), the change comes **2–6 frames after it** (median per match).

**The damaged ball (what the rule sees):** a seen exit (below) falls from 1 s before to 0.5 s after
**482 of 998 true changes (48 %; 33–72 % per match without the STAY check)**. From that exit to the true change:
**median 3 frames**, quartiles 2 / 6, 10–90 % 1 / 9. The rest: no ball in the 2 s before (≈30 % on
IDSSE, 6 % on Metrica), or the ball inside (whistle stoppages). A ball lost heading out and found
again outside is rare (0–4 per match), so extrapolating unseen crossings isn't worth a rule.

**Also seen, for other builders:** 13–28 % of true team → D changes come after a possession under
1 s by the other team, right after a possession of the first: the truth often gives the ball, for a
moment, to the team that touched it last before it went out. That's a short-possession pattern
(idea 10b), not timing. The team before the stoppage restarts in only 10–43 % per match.

## The rule (`bench/lab_stoppage.py`)

### 1. A seen exit

Frame *k* is **an exit** when:
1. the ball is seen beyond a pitch line at *k* (no margin) and seen inside at *k* − 1;
2. the ball isn't seen back inside in the 0.4 s after *k*.

### 2. The move

At each change from a team T to **D** at frame *g*, in the letters given:
1. **Look** for exits from *g* − **back** to *g* + **ahead**, and only between the first frame of
   T's run before *g* and the last frame of the D run after it: an exit outside those belongs to
   another stoppage.
2. **Take the earliest.** The ball leaves play once per stoppage; a later exit is the ball moved
   during the stoppage.
3. **Move the change to the exit + OFFSET** (the truth's lag):
   - **earlier than *g*:** T's frames in between become D. T keeps at least its first frame;
   - **later than *g*:** the D frames in between become T's. The stoppage keeps at least its last
     frame;
   - **no exit seen:** nothing changes.

### Every setting, and why

| setting | value | why |
|---|---|---|
| line margin | 0 m | the truth's D starts at the line (3–4 frames after the crossing, with no margin); the bench's ball jitter is under 1 cm. v1's 1 m margin is for its "ball out" clue, which must not fire on noise during play |
| STAY | 0.4 s | a ball that comes back in that fast is jitter on the line, not out. By reasoning; it removes 243 of 771 raw exits on the bench but **changes nothing in the score** (sensitivity: 0.5776 without it, 0.5777 with it). Insurance for our ball near the lines, which is untested |
| OFFSET | **3 frames** (0.12 s) | the median lag from the exit to the true change. **Fitted leave one match out** (`fit`): 3 in 9 of 10 folds, 4 in the fold without metrica3 (whose lags are shortest); 3 on all 10, and 3 in every IDSSE-only fold. `apply` defaults to 3 |
| back, ahead | grid: 1/1, 2/1, 2/2, 4/2 s (and the no-op) | chosen leave one match out; v11's lags reach ±1 s and beyond, and an exit is precise evidence, so the window only guards against another stoppage's exit (also bounded by the runs) |

The grid was fixed before the run and not extended after it.

## How it's chosen and judged

- **Input:** v11's letters on the 10 bench matches (`bench/lab.py` cache, 41 % damage).
- **Leave one match out** (`lab.judge`): each match scored with the window best on the other 9,
  and with the offset fitted on the other 9.
- **The bar** ([../lab.md](../lab.md)): better held-out mean than v11, ≥ 70 % of matches better, no
  drop on IDSSE nor on Metrica, no drop on the 4 PathCRF-unseen matches. Then the same on IDSSE
  alone (`--set idsse`, whose v11 is the IDSSE-only chain).

## Results

### All 10 matches

**Means over the 10 matches, every window** (not held out):

| back / ahead | spellF1 | macroF1 | D's F1 |
|---|---|---|---|
| no-op (v11) | 0.5727 | 0.8705 | 0.9080 |
| 1 / 1 s | 0.5743 | 0.8714 | 0.9095 |
| 2 / 1 s | 0.5750 | 0.8718 | 0.9101 |
| 2 / 2 s | 0.5763 | 0.8724 | 0.9112 |
| **4 / 2 s** | **0.5777** | **0.8731** | **0.9122** |

**Every fold picked 4 / 2 s.** Held out:

| match | source | v11 | held out | gain |
|---|---|---|---|---|
| J03WMX | IDSSE | 0.5100 | 0.5142 | +0.0042 |
| J03WN1 | IDSSE | 0.5282 | 0.5379 | +0.0097 |
| J03WOH | IDSSE | 0.5676 | 0.5746 | +0.0070 |
| J03WOY | IDSSE | 0.6045 | 0.6088 | +0.0043 |
| J03WPY | IDSSE | 0.6181 | 0.6223 | +0.0041 |
| J03WQQ | IDSSE | 0.5624 | 0.5656 | +0.0032 |
| J03WR9 | IDSSE, unseen | 0.5801 | 0.5835 | +0.0034 |
| metrica1 | Metrica, unseen | 0.5934 | 0.5964 | +0.0030 |
| metrica2 | Metrica, unseen | 0.5811 | 0.5849 | +0.0038 |
| metrica3 | Metrica, unseen | 0.5821 | 0.5890 | +0.0070 |
| **mean** | | **0.5727** | **0.5777** | **+0.0050, 10/10** |

| | v11 | held out | better |
|---|---|---|---|
| IDSSE (7) | 0.5673 | 0.5724 (+0.0051) | 7/7 |
| Metrica (3) | 0.5855 | 0.5901 (+0.0046) | 3/3 |
| PathCRF-unseen (4) | 0.5842 | 0.5885 (+0.0043) | 4/4 |
| macroF1 | 0.8705 | 0.8731 | |
| D's F1 | 0.9080 | 0.9122 | |

**All four checks pass.**

**Timing, at 4 / 2 s:** 292 changes move (125 earlier, 167 later), about 29 per match. Against the
nearest true team → D change (any team; offset 3 on every match), 258 of 291 get closer and 27
farther; **within 0.2 s: 24 % →
88 %**, median distance 0.60 s → 0.08 s. Of the 157 that match a true change with the same team
before, 29 % → 89 % within 0.2 s.

**Against the ceiling:** +0.0050 of the +0.0057 a perfect ball-out rule could reach (88 %), of
the +0.0125 for all stoppage starts.

### IDSSE only (`--set idsse`)

Means: no-op 0.5625, 1/1 0.5636, 2/1 0.5644, 2/2 0.5649, **4/2 0.5657**. Every fold picked 4 / 2 s.

| match | v11 (IDSSE chain) | held out | gain |
|---|---|---|---|
| J03WMX | 0.5032 | 0.5027 | −0.0005 |
| J03WN1 | 0.5002 | 0.5055 | +0.0053 |
| J03WOH | 0.5667 | 0.5716 | +0.0049 |
| J03WOY | 0.6146 | 0.6204 | +0.0058 |
| J03WPY | 0.6197 | 0.6234 | +0.0037 |
| J03WQQ | 0.5543 | 0.5537 | −0.0005 |
| J03WR9 (unseen) | 0.5790 | 0.5824 | +0.0034 |
| **mean** | **0.5625** | **0.5657** | **+0.0032, 5/7** |

macroF1 0.8757 → 0.8780; D's F1 0.9277 → 0.9311. Passes (5 of 7 needed). The two worse matches
lose 0.0005 each.

## Failure modes

1. **A whistle, then the ball kicked out.** If the stoppage is a whistle (the ball inside) and the
   ball is then played out within `ahead` (2 s), the possession is wrongly extended to that exit.
   The ahead limit bounds it; 27 of 292 moves end farther from the truth.
2. **The grid's edge.** 4 s back was the widest window and the score was still rising (2/2 → 4/2:
   +0.0014). Not extended, since the grid was fixed before the run; a wider window risks taking an
   exit from a stoppage the labels missed inside a long possession.
3. **Our ball near the lines is untested.** Our tracker loses the ball 5× more often beyond a line
   (the bench's damage copies that, so coverage should be similar), but a systematic offset of the
   lines in our coordinates would shift exits, or make false ones while the ball runs along a line.
   STAY guards against flicker only. A bias of 0.5 m moves a 10 m/s ball's exit by about 1 frame.
4. **Wrong team before the stoppage.** The rule moves the stoppage's start, never the team. 134 of
   the 292 moves are at changes whose team before differs from the truth's; they still land closer
   (18 % → 87 % within 0.2 s), which helps macroF1 more than spellF1.
5. **No exit seen** (half of the true changes: ball missing, or a whistle stoppage): nothing moves.

## What didn't make a rule

- **10h, restarts with no seen pass.** v9 finds no pass by the team at 45–57 of 67–85 v11 restarts
  per IDSSE match (20–23 of 61–74 on Metrica). A seen entry (the ball seen coming onto the pitch,
  a throw-in) falls in v9's window at only **0–2 of those per match**; v9's page already showed
  that snapping to the ball's reappearance hurts the IDSSE matches. Nothing built.
- **Missed stoppages from P(dead).** True stoppages of 2 s or more that v11 misses: 11–25 per match
  (median 5–8 s). Inside them P(dead) is low (median 0.21–0.45; above 0.7 on only 11–23 % of their
  frames), and stretches where P(dead) stays above 0.7 for 2 s while v11 says live are dead on
  anywhere from 19 % to 100 % of their frames depending on the match. No threshold separates them:
  the dead-ball model doesn't see these stoppages, so this is the dead-ball revision's job (10f).

## Verdict

**Keep.** It clears every check of the bar on both bench sets, with every fold choosing the same
window. The gain (+0.005 overall, +0.003 on IDSSE alone) is above the lab's ~+0.002 fragility line
but small; it's mostly correct timing where the ball is seen going out, which is what a
transparent rule should do. For the mix: `apply(match, letters, back=4.0, ahead=2.0)`.
