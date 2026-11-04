# Lab candidate `decoder`: the decoder re-chosen, with touches as evidence

| | |
|---|---|
| **status** | **built 2026-11-04. Discard by the protocol:** the whole grid (rounds 1 + 2) fails on `all` (held out 0.5793 against v11's 0.5727, but better on only 6/10, and IDSSE drops by 0.0009). **The pre-registered round 1 passes every check** (0.5855 against 0.5727, +0.0128, better on 10/10; IDSSE only +0.0052, 6/7). Accepting round 1 is the lead's call, not this page's: see [Verdict](#verdict) |
| **what** | v7's out-of-sample costs decoded again with other team settings (minimum possession, team switch price) and, optionally, each seen touch by a team making that team cheaper on the touch frame; then v9's snap and v11's move as before |
| **stage** | `decode`: replaces the decoding inside v7, before v9's snap and v11's move |
| **new code** | `bench/lab_decoder.py` (`decode`, `finish`, `predict`, the grid, `python -m bench.lab_decoder [--set all \| idsse]`, ~4 min on `all`, ~3 min on `idsse`) |
| **results** | `bench/results/lab_decoder[_idsse].json` (whole grid), `lab_decoder_round1[_idsse].json` (round 1), `lab_decoder_round1_part1[_idsse].json`, `lab_decoder_part1[_idsse].json`, `lab_decoder_grid[_idsse].csv` (every match × setting) |
| **rules** | bench only, through `bench.lab`; to predict a match it reads only that match's damaged tracking and v7's out-of-sample costs; settings chosen leave one match out |

## Results (summary)

| | held out | v11 | gain | better | IDSSE | Metrica | PathCRF-unseen | macroF1 | possessions / match (v11 349, true 411) |
|---|---|---|---|---|---|---|---|---|---|
| **round 1** (pre-registered, 48 settings) | **0.5855** | 0.5727 | **+0.0128** | **10/10** | +0.0093, 7/7 | +0.0209, 3/3 | +0.0187, 4/4 | 0.8730 vs 0.8705 | 329 |
| round 1, Part 1 only (no touches, 12) | 0.5858 | 0.5727 | +0.0130 | 9/10 | +0.0084, 6/7 | +0.0238, 3/3 | +0.0219, 4/4 | 0.8711 vs 0.8705 | 324 |
| **whole grid** (rounds 1 + 2, 84 settings) | 0.5793 | 0.5727 | +0.0065 | **6/10** | **−0.0009, 3/7** | +0.0238, 3/3 | +0.0193, 4/4 | 0.8714 vs 0.8705 | 301 |
| whole grid, Part 1 only (21) | 0.5837 | 0.5727 | +0.0109 | 7/10 | +0.0011, 4/7 | +0.0339, 3/3 | +0.0285, 4/4 | 0.8703 vs 0.8705 | 294 |

`--set idsse` (7 matches, v11 0.5625; true 437 possessions per match, v11 348):

| | held out | gain | better | macroF1 | chosen on all 7 | possessions / match |
|---|---|---|---|---|---|---|
| round 1 | 0.5678 | +0.0052 | 6/7 | 0.8762 vs 0.8757 | 0.5, 30, 2 | 333 |
| round 1, Part 1 only | 0.5672 | +0.0047 | 6/7 | 0.8755 vs 0.8757 | 1, 30, 0 | 331 |
| whole grid | 0.5714 | +0.0089 | 6/7 | 0.8767 vs 0.8757 | 0.25, 40, 4 | 318 |

(settings as switch team, minimum possession in frames, touch weight)

- **The idea's premise is wrong.** A **shorter** minimum never helps: every setting at 10 frames
  is below v11, and 15 frames beats it only with a dearer switch (1) and touches, by at most
  +0.0015 in sample. The gain is all in the other direction: a **longer** minimum (30 frames,
  1.2 s), which **removes** about 20 possessions per match, almost all of them spurious. The switch
  price barely matters at 30 frames and above.
- **Touches as decoder evidence add almost nothing to spellF1** (round 1 with and without them:
  0.5855 against 0.5858 on `all`, 0.5678 against 0.5672 on `idsse`). They add a little macroF1
  (+0.002) and make the gain more even (10/10 against 9/10).
- **The best minimum depends on the source.** IDSSE's matches peak at 30 frames; Metrica's keep
  rising to 60 (2.4 s). Round 2 let the choice go there, and on `all` some folds then picked 50–60,
  which costs IDSSE. That's what fails the bar.

## Why

The error budget ([../ideas.md](../ideas.md), "Where v9's remaining spellF1 is") says the largest
error left is **missed possessions**: 35 % of true possessions have no predicted one at IoU ≥ 0.3,
median 1.6 s. The decoder's team settings (minimum possession 20 frames = 0.8 s, team switch 0.5)
were chosen in v3 on v1's clues alone, before the dead-ball model, the restart model, PathCRF and
v11's touches existed ([../models/v3-decoder.md](../models/v3-decoder.md)). With PathCRF in the
costs and v11 placing the changes after decoding, a shorter minimum or a cheaper switch might now
let those short possessions through. That was the hypothesis; v11's page planned it as v13.

## What it does

**Part 1: the team settings.** `decode(match, switch_team, min_possession, touch_weight, which)`
runs the same min-length Viterbi (`bench/tune_decoder.py` `decode`) on `match.pred["v7_cost"]`, per
half, with the dead-ball settings unchanged (`chain.decoder(which)`: dead price 50, no minimum
stoppage). `finish(match, letters, which)` then applies v9's restart snap
(`chain.restart_snap(which)`) and v11's touch move (`v11.SETTINGS[which]`), exactly as
`bench/lab.py` builds v11.

**Part 2: touches as evidence.** Before decoding, each seen touch by team T (v11's detector,
`v11.seen_touches`, v11's ΔV of 3 m/s) makes T's cost lower by `touch_weight` **on the touch frame
only**:
- **one frame, not a window:** the minimum length already makes any T stretch that covers the touch
  at least `min_possession` long, and lets the decoder place it where the other evidence agrees;
  v11 then moves its start onto the touch. One knob, with a plain meaning: a seen touch is worth
  `touch_weight` against the switch prices (inserting a possession inside the other team's costs
  two switches).
- **T cheaper, rather than the other team dearer:** the literal reading of "the touch says T has
  the ball". Against D it hardly matters: leaving or entering D costs 50 each way, far more than
  any weight tried.

**The current settings reproduce v11 exactly.** v7's costs are saved as float32; decoding them
again with v3's settings differs from `match.v7` on 0–7 frames per match (ties), and on J03WOH that
moves spellF1 by +0.0005. So with the chain's own settings and no touch weight, `decode` returns
`match.v7` itself, and the grid run checks that this row equals v7 and v11 on every match.

## Every setting, and why

| setting | grid | why |
|---|---|---|
| minimum possession | **round 1:** 20 (v3's, first), 10, 15, 30 frames | the range v11's page planned for v13: 0.4–1.2 s around v3's 0.8 s |
| | **round 2:** 40, 50, 60 frames | added after round 1, because its best (30) sat on the grid's edge with the score still rising (+0.012 from 20 to 30). The one axis with a slope at the edge, three steps on |
| team switch price | 0.5 (v3's, first), 0.25, 1 | halving and doubling v3's price. At 30 frames and above it's nearly flat (within 0.002), so it wasn't extended |
| touch weight | 0 (Part 1), 1, 2, 4 | the price of inserting a possession is two switches (1 at v3's price) plus the other evidence; over the best 20-frame stretch covering a touch inside the other team's possession, that evidence is median ~21, lower quartile ~5 (J03WR9, costs only, no truth). So 1 flips only near-ties and 4 only the closest part; much more would be v12's loose rule, which was a disaster. Flat from 2 to 4 at 30 frames, so not extended |
| dead price, minimum stoppage | unchanged: chain's (50, 1) | chosen in v4 with the dead-ball model in the loop, i.e. after the evidence that matters to them; the budget for stoppage errors is small (+0.022 missed, +0.015 timing, +0.003 false) |
| touch detector | v11's (2 m, 0.2 s, ΔV 3 m/s on both sets) | no new knob: the same touches v11 moves changes onto |
| grid order | current first | `lab.judge` breaks ties towards the first setting |

Round 1 was fixed before any score was seen and judged first; round 2 was added after, so the
whole grid is judged again (docs/lab.md: "a second grid round is reported as such and the whole
grid is judged").

## Results

### Round 1, held out per match (`all`, 48 settings)

Each match scored with the setting best on the other 9.

| match | source | held out | v11 | gain | picked |
|---|---|---|---|---|---|
| J03WMX | IDSSE | 0.5125 | 0.5100 | +0.0025 | 1, 30, 4 |
| J03WN1 | IDSSE | 0.5345 | 0.5282 | +0.0064 | 1, 30, 4 |
| J03WOH | IDSSE | 0.5841 | 0.5676 | +0.0165 | 1, 30, 4 |
| J03WOY | IDSSE | 0.6141 | 0.6045 | +0.0095 | 1, 30, 2 |
| J03WPY | IDSSE | 0.6338 | 0.6181 | +0.0157 | 1, 30, 2 |
| J03WQQ | IDSSE | 0.5647 | 0.5624 | +0.0023 | 1, 30, 4 |
| J03WR9 | IDSSE, unseen | 0.5921 | 0.5801 | +0.0121 | 1, 30, 4 |
| metrica1 | Metrica, unseen | 0.6129 | 0.5934 | +0.0195 | 1, 30, 2 |
| metrica2 | Metrica, unseen | 0.6053 | 0.5811 | +0.0241 | 1, 30, 4 |
| metrica3 | Metrica, unseen | 0.6011 | 0.5821 | +0.0190 | 0.25, 30, 4 |

Every fold picks a minimum of 30 frames; 9 of 10 a switch of 1. Chosen on all 10: **switch 1,
minimum 30, touch weight 4** (in sample 0.5872, macroF1 0.8732, 328 possessions per match).

### The whole grid, held out per match (`all`, 84 settings)

| match | held out | v11 | gain | picked |
|---|---|---|---|---|
| J03WMX | 0.4913 | 0.5100 | **−0.0186** | 0.5, 60, 0 |
| J03WN1 | 0.5186 | 0.5282 | **−0.0096** | 0.5, 60, 0 |
| J03WOH | 0.5802 | 0.5676 | +0.0126 | 1, 50, 0 |
| J03WOY | 0.6003 | 0.6045 | **−0.0043** | 1, 50, 0 |
| J03WPY | 0.6286 | 0.6181 | +0.0104 | 0.5, 40, 4 |
| J03WQQ | 0.5599 | 0.5624 | **−0.0025** | 0.5, 40, 4 |
| J03WR9 | 0.5859 | 0.5801 | +0.0059 | 0.5, 40, 4 |
| metrica1 | 0.6175 | 0.5934 | +0.0241 | 0.5, 40, 4 |
| metrica2 | 0.6053 | 0.5811 | +0.0241 | 1, 30, 4 |
| metrica3 | 0.6052 | 0.5821 | +0.0231 | 0.5, 40, 4 |

The picks scatter over 30–60 frames: past 30 the mean over 9 matches is flat (best per minimum,
in sample: 30 → 0.5872, 40 → 0.5880, 50 → 0.5878, 60 → 0.5875), so which minimum wins a fold is
decided by which matches are left in.

### The landscape, in sample, by source (`all`; switch 0.5, touch weight 0)

| minimum (frames) | 10 | 15 | 20 (v11) | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|---|
| IDSSE (7) | 0.5563 | 0.5615 | 0.5673 | **0.5755** | 0.5734 | 0.5712 | 0.5700 |
| Metrica (3) | 0.5624 | 0.5763 | 0.5855 | 0.6056 | 0.6124 | 0.6242 | **0.6282** |
| possessions / match, IDSSE (true 437) | 376 | 367 | 356 | 336 | 317 | 304 | 291 |
| possessions / match, Metrica (true 351) | 371 | 350 | 333 | 312 | 292 | 276 | 261 |

Metrica's truth has fewer, longer possessions (351 per match against IDSSE's 437), and the best
minimum follows that convention. 30 frames beats 20 in every one of the 24 (source, switch, touch
weight) rows on `all`, and in all 12 rows on `idsse`.

### Why a longer minimum helps: it removes spurious short possessions

Predicted possessions per match by length, and how many match a true one at IoU 0.5 (`all`,
`bench.lab` scorer's matching):

| | < 1 s | 1–2 s | 2–4 s | 4–10 s | > 10 s | total | matched |
|---|---|---|---|---|---|---|---|
| IDSSE, v11 | 25 (3 matched) | 45 (17) | 82 (47) | 105 (76) | 99 (83) | 356 | 227 |
| IDSSE, round 1's choice (1, 30, 4) | 6 (1) | 46 (17) | 80 (47) | 102 (76) | 101 (84) | 336 | 226 |
| IDSSE, minimum 60 | 0 | 6 (4) | 87 (51) | 96 (71) | 102 (84) | 291 | 211 |
| Metrica, v11 | 24 (1) | 45 (11) | 60 (28) | 83 (60) | 120 (102) | 333 | 202 |
| Metrica, round 1's choice | 4 (1) | 45 (11) | 59 (28) | 81 (60) | 121 (104) | 309 | 204 |
| Metrica, minimum 60 | 0 | 7 (4) | 61 (31) | 74 (56) | 118 (104) | 261 | 195 |

- **v11 predicts ~25 possessions per match under 1 s, and only 1–3 of them are real.** They come
  from the decoder's 0.8 s stretches shortened afterwards by v9's snap and v11's move. A 1.2 s
  minimum removes about 20 of them and loses one match at most: precision up, recall flat.
- **At 60 frames** IDSSE also loses real possessions (−16 matched per match, mostly 1–2 s ones);
  Metrica loses fewer (−7) and has fewer true possessions to find, so its precision gain still wins.
- **The missed possessions stay missed.** A shorter minimum (10 frames) adds ~25 predictions per
  match, and spellF1 falls (0.5581 against 0.5727 at switch 0.5): from these costs the decoder
  can't tell the real short possessions from the spurious ones.

### Touches as evidence

| | in sample, best at each minimum (`all`) | |
|---|---|---|
| minimum 20, switch 1 | touch weight 0: 0.5772 | 4: 0.5790 |
| minimum 30, switch 1 | 0: 0.5858 | 4: 0.5872 |
| minimum 10, switch 1 | 0: 0.5693 | 4: **0.5618** |

With a long minimum the touches help a little (+0.001 to +0.002, and +0.002 macroF1); with a short
one they insert spurious possessions, as v12's loose rule did.

## Failure modes

1. **The convention of our match.** The best minimum follows the truth's convention: IDSSE peaks at
   1.2 s, Metrica keeps rising to 2.4 s. Our match is Metrica-tracked, but whose possession
   convention the organisers' labels follow is unknown. A longer minimum is a bet on fewer, longer
   possessions; round 1's 1.2 s is the setting both sources agree improves on 0.8 s.
2. **It moves further from the true count.** v11 predicts 349 possessions per match against 411
   true; round 1 predicts 329, the whole grid 301. spellF1 rewards that (fewer spurious ones), but a
   later candidate that adds short possessions (`short`) works in the other direction, so the two
   should be mixed and judged together, not assumed to add up.
3. **Short real possessions under 1.2 s are given up.** On the bench they were almost never
   matched anyway (3 of 55 per IDSSE match under 1 s, with v11).
4. **Touch evidence is fragile.** Duels and high balls give false touches; at short minimums the
   weight hurts. Its spellF1 gain is under the ~0.002 the lab calls fragile.
5. **Float32 costs.** Settings other than the current one are decoded from float32 costs; that
   moves 0–7 frames per match (ties) and is the same for every setting.
6. **Two rounds on one bench.** Round 2 was added after seeing round 1 (reported as such). The
   whole grid is what the protocol judges, and it fails on `all`.
7. **Untested here:** 30 % and 55 % damage (the lab's cache is at 41 % only).

## Verdict

- **By the protocol: discard.** The whole grid (both rounds) fails on `all`: better on 6/10, and the
  IDSSE mean drops (0.5664 against 0.5673). On `idsse` it passes (+0.0089, 6/7).
- **Round 1, pre-registered, passes every check on both sets** (`all` +0.0128, 10/10, every source
  and the unseen matches up; `idsse` +0.0052, 6/7). It was measured before round 2 existed. Taking
  it now means choosing a grid after seeing round 2 fail, which is the lead's decision, not this
  page's. What would justify it: round 2 failed because the choice beyond 1.2 s splits by source,
  not because 1.2 s stopped working (every round-1 fold picked 30 frames, and 30 beats 20 on both
  sources in every row of the grid, on both sets).
- **Touches as decoder evidence: not worth keeping on their own** (spellF1 within ±0.0006 of
  Part 1 alone; macroF1 +0.002).

## For mixing (if round 1 is accepted)

```python
from bench import lab_decoder
letters = lab_decoder.decode(match, which="all", **lab_decoder.CHOSEN["all"])   # v7's level
letters = lab_decoder.finish(match, letters, "all")                             # + v9 snap, v11 move
```

- `CHOSEN["all"]` = switch 1.0, minimum 30 frames, touch weight 4.0 (round 1, chosen on all 10).
- `CHOSEN["idsse"]` = switch 0.5, minimum 30 frames, touch weight 2.0 (round 1, chosen on 7).
- Without touches (Part 1 only): switch 1.0, minimum 30, touch weight 0, on both sets.
- Dead-ball settings, the restart snap and v11's move stay the chain's for each set.
