# Model v9: v8 + possessions start at the restart pass

| | |
|---|---|
| **status** | **built 2026-10-30** (Jose approved the page with the five recommended choices); **adopted** by the rule fixed below: held-out spellF1 0.561 against v7's 0.556, better on 10/10 |
| **change from v8** | one step after decoding: where the labels go from dead to a team, the change moves to the restart pass when a pass is seen nearby. Which team, and how many possessions, never change |
| **code** | `possession/passes.py` (the pass detector, reused later by step 5), `possession/v9.py`, `bench/restart_timing.py` (measurements and the leave-one-out choice) |
| **run** | `python -m bench.restart_timing` (~6 min, writes `bench/results/restart_timing_*`); `python -m possession.v9` (~1.5 min, writes `evaluation/results/v9.csv`) |
| **settings** | look for the pass from **5 s before** the change to **1 s after** (picked by all 10 folds) |
| **output** | SHA-256 `e87ea126…` (`e87ea12687fa373b01c8bb1acb906e6f563a25ecfed4c8287272af3b8c82332b`), identical over two runs |
| **new dependencies** | none |

## Results (summary)

- **Bench, held out:** spellF1 **0.5613** against v7's 0.5557 (**+0.0056**), **better on 10/10**;
  macroF1 0.8676 against 0.8639. The adoption bar (better mean, at least 7 of 10) is passed.
- **Timing:** of the 272 moved starts that match a true one, **83 %** now land within 0.2 s of it
  (20 % before); median distance 0.60 s → 0.00 s.
- **Our match:** 40 of v8's 78 restart starts move, **34 earlier and 6 later** (median −0.94 s):
  ours were late, as Jose saw. At the starts, the ball is now leaving 59 % of the time (was 15 %)
  and arriving 15 % (was 36 %).

Details in [Results](#results) below.

## Why

**Jose's observation, watching the viewer:** after a stoppage, our labels start the new possession
when the ball reaches the receiver. It should start **at the moment of the pass**: a throw-in, a
free kick, a kick-off, a goal kick ([../ideas.md](../ideas.md), idea 10).

**The truth agrees, exactly.** On the bench, a possession after a stoppage starts on the frame the
ball is played:
- **IDSSE, clean ball (698 restarts):** the ball is leaving at the true start 100 % of the time,
  arriving 0 %. It's never above 8 m/s in the 2 s before.
- **Metrica's samples (291 restarts):** the ball reappears in Metrica's own data within 0.2 s of the
  true start 97 % of the time.

**Our models don't.**
- **At v7's starts on the bench:** the ball was already above 8 m/s in the 2 s before 56 % of the
  time.
- **At v8's starts on our match:** 62 %.
- **Against the truth on the bench:** half of v7's starts are more than 0.64 s off, early as often
  as late.
- **Why:** the change from dead to live comes from the dead-ball model, which reads the players.
  Their speed rises gradually around a restart (1.1 m/s 1 s before, 1.3 at the pass, 1.7 1 s
  after), so it has no sharp moment to land on.

**The ceiling:** putting each of v7's 670 matched starts exactly on the true one lifts the bench's
mean spellF1 from 0.556 to **0.574 (+0.018)**, better on 10/10. No real rule gets all of that.

## Why the ball's reappearance alone isn't enough

Our tracking loses the ball during stoppages (83 % missing while players stand still). So "the ball
comes back" is a tempting marker. Three facts say it isn't safe on its own:

| | the ball comes back within 0.2 s of the true start | where it comes back, near starts |
|---|---|---|
| **Metrica's samples** (Metrica's own processing) | 97 % | at a restart spot (touchline, centre spot, goal area) 64–80 % |
| **the bench's damage** (IDSSE) | 51 %: our damage brings the ball back at random | — |
| **our match** (no labels: at v8's starts) | — | at a restart spot **37 %**, against 16 % for other comebacks |

On our match the comeback is tied to restarts more than chance, but much less than in Metrica's
samples. A probe that snapped to the comeback **hurt the IDSSE matches** (by up to −0.012) while
helping Metrica's. So the rule below snaps only to a **pass it can see**, which doesn't depend on
when the tracking finds the ball again.

## The rule

### 1. A seen pass (`possession/passes.py`)

Frame *k* is **the start of a pass by team T** when:
1. the ball is seen at *k* and 0.4 s later;
2. a player of T is within **2 m** of the ball at *k* (the taker);
3. 0.4 s later the ball is at least **1 m further** from that player than at *k* (it leaves him; at
   a reception it would come closer);
4. the ball covered those 0.4 s at more than **4 m/s** on average.

| setting | value | why |
|---|---|---|
| taker distance | 2 m | the ball is at the feet (or in the hands at a throw-in) of whoever plays it: under a metre, plus tracking error on both dots |
| look ahead | 0.4 s (10 frames) | long enough to see the ball leave through the tracking's jitter; short enough that a 5 m pass hasn't arrived yet |
| leaving | 1 m further | separates "leaving the player" from "arriving at the player", which is Jose's exact distinction |
| played | more than 4 m/s | throw-ins and short free kicks travel at 8–15 m/s; a ball being placed or rolled to the taker is slower, and a dribble stays within 2 m |

These four are fixed by reasoning, not searched. Searching them would multiply the settings for a
small effect.

### 2. The snap (`possession/v9.py`)

After v8 decodes, at each change from **D** to a team T at frame *g*:
1. **Look** for seen passes by T from *g* − **back** to *g* + **ahead**.
2. **Take the earliest one.** The restart is the first pass after the stoppage. The nearest one can
   be the receiver's next pass. In the probe, earliest beat nearest (+0.003/+0.004 against +0.001/+0.002).
3. **Move the change to that frame:**
   - **pass before *g*:** the dead frames in between become T's. Only if they're all dead: the snap
     never eats into an earlier possession.
   - **pass after *g*:** T's frames before the pass become dead. Only if T's possession lasts past
     the pass: the snap never deletes a possession.
   - **no pass seen:** nothing changes.

| setting | grid | chosen by |
|---|---|---|
| back | 1, 2, 3, 5 s | leave-one-out on the bench (below) |
| ahead | 1, 2, 3 s | leave-one-out on the bench |

## How it's chosen and judged (bench)

- **Input:** v7's out-of-sample predictions on the 10 bench matches (`data/external/bench_predictions`).
  v8 equals v7 there, since the keeper repair never fires.
- **Damage:** 41 % as usual, with the same check at 30 % and 55 %.
- **Leave-one-out:**
  - for each match, pick the (back, ahead) pair with the best mean spellF1 on the other 9, then
    score it on that match;
  - report the held-out mean against v7, how many of the 10 matches improve, and macroF1.
- **Timing check:** for snapped starts, the distance to the true start, before and after.
- **Adopted only if:** the held-out mean is above v7's **and** at least 7 of 10 matches improve.
  This is fixed now, before the results. If it fails, v9 isn't built, and the pass detector waits
  for step 5.
- **Our match, without labels:**
  - how many starts move, earlier or later, and the frames changed;
  - how often the ball is "leaving" at our starts (15 % now);
  - a list of moved starts to watch in the viewer.

## What a probe already shows (exploration, not the procedure above)

One setting at a time, all 10 matches, no leave-one-out:

| | mean spellF1 | matches better |
|---|---|---|
| v7 | 0.556 | — |
| earliest pass, back 2 s / ahead 1 s | 0.559 | 9 (1 equal) |
| earliest pass, back 3 s / ahead 2 s | 0.560 | 7 (2 equal, 1 worse) |
| ceiling (every matched start on the truth) | 0.574 | 10 |

- **The timing gets right where a pass is seen:** snapped starts within 0.2 s of the truth go from
  22 % to **68 %** (median distance 0.64 s → 0.00 s).
- **The score moves little, for three reasons:**
  - a pass is seen at only 27–45 % of the IDSSE starts (the ball is missing);
  - some snaps pick the wrong pass;
  - spellF1's overlap thresholds forgive sub-second shifts.

  Even with the ball always visible (clean IDSSE), a first probe (nearest pass) gains only +0.004
  against a +0.019 ceiling there.
- **Metrica's samples gain the most** (+0.004 to +0.011; a pass is seen at 72–81 % of their starts).
  Their ball behaves most like ours.
- **On our match:** a seen pass by the restarting team falls within −3/+2 s of 50 of v8's 78
  starts (64 %).

**Expected size: about +0.003 to +0.005,** a quarter of the ceiling and well under our match's
noise band (±0.04). The point is a correct convention, applied consistently, plus a pass detector
that step 5 needs anyway.

## Results

### The bench (`python -m bench.restart_timing`)

**The truth's convention, re-checked by the script:** at all 698 IDSSE starts with the clean ball,
the ball is leaving 100 %, arriving 0 %, and never above 8 m/s in the 2 s before.

**Mean over the 10 matches, every window** (v7 = 0.5557 spellF1, 0.8639 macroF1):

| back \ ahead | 1 s | 2 s | 3 s |
|---|---|---|---|
| 1 s | 0.5576 | 0.5569 | 0.5545 |
| 2 s | 0.5592 | 0.5587 | 0.5568 |
| 3 s | 0.5605 | 0.5601 | 0.5582 |
| **5 s** | **0.5613** | 0.5609 | 0.5591 |

- **Looking back helps; looking ahead hurts.** Looking back is where the late starts get fixed.
  Looking ahead lets the snap take a later pass, the receiver's (failure mode 2).
- **5 s is the grid's edge,** but the gain is flattening: 3 → 5 s adds only +0.0008. The grid
  was fixed on this page before the run, so it wasn't extended.
- **Leave one match out:** every fold picked back 5 s / ahead 1 s.

| match | v7 | v9 (held out) |
|---|---|---|
| J03WMX | 0.4954 | 0.4996 |
| J03WN1 | 0.5008 | 0.5056 |
| J03WOH | 0.5531 | 0.5542 |
| J03WOY | 0.5921 | 0.5931 |
| J03WPY | 0.5953 | 0.6016 |
| J03WQQ | 0.5597 | 0.5608 |
| J03WR9 | 0.5692 | 0.5698 |
| metrica1 | 0.5713 | 0.5796 |
| metrica2 | 0.5528 | 0.5704 |
| metrica3 | 0.5671 | 0.5783 |
| **mean** | **0.5557** | **0.5613 (+0.0056), 10/10** |

- **Metrica's samples gain the most** (+0.008 to +0.018). Their ball behaves most like ours, and a
  pass is seen at more of their starts.
- **The timing does what Jose asked:** 83 % of moved starts now within 0.2 s of the truth (20 %
  before).
- **Sensitivity:** with the pass detector reading tracking damaged at 30 % or 55 % (v7's labels
  stay those made at 41 %), 0.5624 (10/10 better) and 0.5589 (9/10). That's a cheaper version of
  the check than re-running v7 at each level (~30 min each). It answers the question that matters
  for the snap: does it still help when the ball is seen more or less often?
- **Gain against the ceiling:** +0.0056 of a possible +0.018, about a third.

### Our match, against v8 (label-free; `python -m possession.v9`)

- **Moved:** 40 of 78 dead → team changes (51 %): **34 earlier, 6 later**; median −0.94 s
  (quartiles −1.61 / −0.49 s).
- **The ball at the starts:**

| | leaving | arriving | already above 8 m/s in the 2 s before |
|---|---|---|---|
| v8 (53 starts with the ball seen) | 15 % | 36 % | 62 % |
| v9 (39) | **59 %** | **15 %** | **36 %** |

- **Unchanged:** possessions 351, as designed. Dead share 46.8 % → 46.0 %, since stoppages end
  earlier. 1,409 frames differ (0.89 % of graded).
- **The whistle check:** 74 → 73 of 86 whistles followed by dead within 3 s; counted ones still
  11/11. One restart now starts within 3 s of a whistle.
> **2026-11-03:** the organisers' clarification forbids checking our output against the video, so the
> moments listed here are not to be watched against it ([../limitations.md](../limitations.md), section 0).

- **The largest moves, to watch in the viewer.** They're near the 5 s limit, where a pass during
  the stoppage could be mistaken for the restart (failure mode 1):

| change (v8) | pass (v9) | shift | team |
|---|---|---|---|
| 124,711 | 124,591 | −4.8 s | Athletic |
| 16,416 | 16,308 | −4.3 s | Athletic |
| 18,522 | 18,419 | −4.1 s | Barça |
| 144,429 | 144,328 | −4.0 s | Barça |
| 172,049 | 171,963 | −3.4 s | Barça |

- **Public 5 minutes (pipeline check only, not used to choose):** spellF1 0.640 against v8's
  0.606; macroF1 0.856 against 0.857. Two of the moves above fall inside it.

## Failure modes

1. **Passes during the stoppage.** Players often knock the ball to the taker before the restart.
   If that pass is by the restarting team and inside the window, the possession starts too early.
   The "back" limit bounds it, and the bench measures it.
2. **An unseen restart.** If the restart pass isn't seen and the receiver passes again inside the
   window, "earliest" takes the second pass, which is later than the real start.
3. **Wrong team at the start.** The snap looks for the team the decoder chose. It never fixes the
   team; that's the restart model's and PathCRF's job.
4. **Throw-ins.** The ball dot sits at the thrower's hands, so the taker is within 2 m and the ball
   leaves fast. It's covered, but our tracking's ball near the touchline is untested.
5. **Kick-offs at the start of a half** aren't dead → team changes (the label goes from not graded
   to a team), so they're untouched. Kick-offs after goals are covered.

## Not covered

- **The other end of a stoppage:** a possession that should end when the ball crosses the line or
  the whistle blows. The same truth check could be run there; it's separate.
- **The 28 % of true restarts with no predicted start nearby:** those stoppages were missed or
  merged. That's the dead-ball revision's job (ideas 1, 2, 5, 6).

## Decisions (Jose, 2026-10-30)

| | choice | taken | why |
|---|---|---|---|
| 1 | Snap after decoding, rather than a decoder clue that makes dead → team cheap only at seen passes | **after decoding** | Transparent, can't change teams or the number of possessions, easy to measure; a decoder version fits better with step 5 |
| 2 | The earliest seen pass in the window, rather than the nearest | **earliest** | The restart is the first pass; the probe agrees (+0.003/+0.004 against +0.001/+0.002) |
| 3 | The pass detector's four settings fixed by reasoning; only the window chosen on the bench | **yes** | Fewer knobs for a small effect; less room to overfit 10 matches |
| 4 | Adopt only if held-out better than v7 on the mean and on at least 7 of 10 matches | **yes** | Set before seeing the result |
| 5 | Name it v9 (v8 + restart pass) | **yes** | One change per version |
