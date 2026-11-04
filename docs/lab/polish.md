# Lab candidate `polish`: the mix's own settings chosen again inside the chain

| | |
|---|---|
| **status** | **built 2026-11-04, lab round 2; passes the bar against the mix on both sets, under the judge and under a fully nested check: keep (narrowly on the PathCRF-unseen matches)** |
| **what** | round 1's mix (stoppage → short → segments), with each stage's settings chosen again with the stages before it in place, plus two stages run again at the end: **stoppage2** (the stoppage rule on the team → D changes segments creates) and **short2** (short possessions in the runs segments cleaned) |
| **result, `all`** | judge 0.6207 against the mix's 0.6156 (**+0.0051, 8/10**); fully nested **+0.0072, 7/10**; unseen +0.0003 / +0.0012 (2/4) |
| **result, `idsse`** | judge 0.6046 against 0.6001 (**+0.0046, 5/7**); fully nested **+0.0085, 6/7**; J03WR9 +0.0063 / +0.0135 |
| **polished, `all`** | stoppage **6 s back / 3 s ahead**; short **p_min 0.5** (unchanged); segments **`all`, flip 0.6, merge 0.7, `dead`**; stoppage2 **on** (6 / 3 s); short2 **p_min 0.65** |
| **code** | `bench/lab_polish.py` (new file only; nothing else changed) |
| **run** | `python -m bench.lab_polish [--set all\|idsse]` (~20 min on `all` with the machine shared, ~14 min on `idsse`; under 0.75 GB resident); `--save` (~2 min, after the run: the polished chain's out-of-fold letters) |
| **results** | `bench/results/lab_polish[_idsse].json` (the judge's report, plus `trace`, `nested_selection`, `fully_nested`, `polished_settings`, `possessions_after_stage`, `keep`), `lab_polish_grid[_idsse].csv` (every match × configuration scored), `lab_polish_base[_idsse].json` and `data/external/lab_polish[_idsse]/` (after `--save`) |

## Why

Round 1 built each stage on v11's letters and chose its settings there; the mix then chained them
with those settings ([../lab.md](../lab.md), "Mixing"). Three of them sat where a re-choice in the
chain could pay:

- **stoppage**: the widest window tried (4 s back, 2 s ahead) was best and still rising
  ([stoppage.md](stoppage.md), failure mode 2);
- **short**: p_min 0.5 was its lowest and the gain still rose as it fell ([short.md](short.md));
  with segments after it, extra insertions might now be cleaned up;
- **segments**: its thresholds were picked on v11's letters, not on short's output
  ([segments.md](segments.md)). The mix predicts ~302 possessions a match against ~411 true and
  loses only on J03WMX (−0.0075 against v11), the match with the most short possessions.

And two stages may be worth running again:

- **stoppage2**: segments' merges (mode `dead`) create new team → D changes at the start of the
  merged possession; if the ball is seen going out just after, the stoppage rule moves that change
  onto the exit (the team before keeps the ball until it leaves), exactly as it does for v11's.
- **short2**: segments merges blips into the surrounding possession, which makes longer runs with new
  O-N-O candidates. A second short stage, fitted on segments' output, learns which of those to trust.

## What was searched (all fixed before any run)

### The chain

Stages in this order: **stoppage → short → segments → stoppage2 → short2**. Every trained stage on
a match is fitted on the bench's other matches, given as input what the stages before it wrote on
them out of fold (`lab_mix.run_chain`'s stacking): short's regression, segments' two classifiers,
and stoppage's offset (the median exit-to-change lag of the other matches, 3 frames). The first
configuration (the mix) rebuilds round 1's mix **exactly**: the same letters on every match, the same
scores (checked at every run).

### The grids (the current value first)

| stage | values | why |
|---|---|---|
| stoppage (back / ahead, s) | **4/2**, 6/2, 8/2, 4/3, 6/3, 8/3 | round 1's 4/2 was its widest and still rising; both edges pushed. The look stays bounded by the two runs the change separates |
| short (p_min) | **0.5**, 0.4, 0.3 | round 1's lowest, and lower; the rest of round 1's `CHOSEN` (learned rule, 0.5–5 s, O-N-D on) |
| segments (features, flip, merge) | **round 1's per-match pick**, then {`all`, `no_pathcrf`} × flip {0.6, 0.7, 0.8} × merge {0.6, 0.7, 0.8, 0.9}; mode `dead` | around round 1's chosen `all` / 0.6 / 0.7. Merge up to 0.9 because the mix merges a lot; mode `dead` won every comparable pair in round 1. 25 values |
| stoppage2 | **off**, on | on: the stoppage rule with the stoppage stage's window and offset |
| short2 (p_min) | **off**, 0.8, 0.65, 0.5 | cautious first: a candidate segments has just merged away needs more evidence to come back |

Round 1's per-match segments pick is kept as a value (not replaced by one setting) so that the
first configuration is the mix itself: a fold that can't beat it keeps it.

### The search (coordinate-wise)

1. Start from the mix: 4/2, 0.5, round 1's pick, off, off.
2. A pass: each stage in pipeline order tries every value of its grid, the other stages at their
   current values, and keeps the value with the best mean spellF1 (rounded to 6 decimals; ties keep
   the current value, then the first in grid order).
3. A second pass if the first changed anything; then stop (**at most 2 passes**).

40 configurations a pass at most; the search on all 10 matches scored 68 distinct ones (71 on
`idsse`). No full product (6 × 3 × 25 × 2 × 4 = 3,600).

### How it's judged (fixed before the run)

Against the mix (`lab_mix.kept_base`), same bar ([../lab.md](../lab.md)), three ways, each match
scored on settings chosen without it:

1. **judge**: `lab.judge` over every configuration the search on all the matches scored. Each
   match gets the configuration best on the other matches. Flattering in one way: *which*
   configurations were visited depended on every match's scores, the held-out one included.
2. **nested selection**: for each match, the whole search rerun with its score left out of every
   mean. It gets the configuration that search ends on.
3. **fully nested**: for each match, the whole search rerun on a bench of the other matches, where
   every trained stage, the stoppage offset and round 1's per-match segments pick are also fitted
   without it (each of the other matches predicted by models trained on the remaining 8, or 5). The
   held-out match is then scored with that configuration, its own stages trained on the others.
   Nothing the held-out match's truth touched is used to choose or fit anything.

**Kept only if 1 and 3 both clear the bar, on `all` and on `idsse`.** 2 is reported too.

## Results

### The search on all 10 matches (in sample)

| stage | pass 1 | pass 2 (others polished) |
|---|---|---|
| stoppage | 4/2 0.6156, 6/2 0.6158, 8/2 0.6153, 4/3 0.6145, **6/3 0.6186**, 8/3 0.6172 | 4/2 0.6235, 6/2 0.6246, 8/2 0.6227, 4/3 0.6246, **6/3 0.6269**, 8/3 0.6264 |
| short | **0.5 0.6186**, 0.4 0.6172, 0.3 0.6173 | **0.5 0.6269**, 0.4 0.6265, 0.3 0.6252 |
| segments (best four; round 1's pick) | **all 0.6 0.7 0.6224**, all 0.7 0.7 0.6218, all 0.8 0.7 0.6213, no_pathcrf 0.6 0.7 0.6194; round 1 0.6186 | **all 0.6 0.7 0.6269**, all 0.7 0.7 0.6267, all 0.8 0.7 0.6264, no_pathcrf 0.6 0.7 0.6257; round 1 0.6234 |
| stoppage2 | off 0.6224, **on 0.6230** | off 0.6250, **on 0.6269** |
| short2 | off 0.6230, 0.8 0.6255, **0.65 0.6269**, 0.5 0.6264 | off 0.6230, 0.8 0.6255, **0.65 0.6269**, 0.5 0.6264 |

Pass 2 changed nothing: the search converged. **Polished on `all`: 0.6269 in sample (+0.0113 on
the mix), macroF1 0.8802.** Merging less doesn't pay: for each feature set and flip, merge 0.8 is
below 0.7 and 0.9 is the worst (0.605–0.613), in both passes. The mix's ~302 possessions are not
segments merging too much on short's output.

The steps along the path, per match (in sample):

| step | mean | better / worse (of 10) |
|---|---|---|
| stoppage 4/2 → 6/3 | +0.0029 | 7 / 3 |
| segments round 1's pick → `all` 0.6 / 0.7 | +0.0038 | 5 / 2 (3 already had it) |
| stoppage2 on | +0.0007 | 5 / 1 |
| short2 0.65 | +0.0039 | 6 / 3 |

### All 10 matches, held out

| | judge | nested selection | **fully nested** |
|---|---|---|---|
| **spellF1** | **0.6207 against the mix's 0.6156 (+0.0051), 8/10** | 0.6246 (+0.0090), 9/10 | **0.6229 (+0.0072), 7/10** |
| IDSSE (7) | 0.6021 against 0.5967 (+0.0054), 6/7 | +0.0112, 7/7 | 0.6044 (+0.0077), 5/7 |
| Metrica (3) | 0.6641 against 0.6599 (+0.0042), 2/3 | +0.0037, 2/3 | 0.6659 (+0.0060), 2/3 |
| **PathCRF-unseen (4)** | 0.6503 against 0.6499 (**+0.0003**), 2/4 | +0.0037, 3/4 | 0.6512 (**+0.0012**), 2/4 |
| macroF1 | 0.8797 against 0.8775 | 0.8801 | 0.8784 |
| possessions per match (true 411) | 318 (mix 302) | 319 | 313 |
| against v11 (0.5727) | +0.0480, 10/10 | | +0.0501, 10/10 |
| **the bar** | mean ✓, 8/10 ✓, each source ✓, unseen ✓: **passes** | passes | mean ✓, **7/10 ✓ (the minimum)**, each source ✓, unseen ✓: **passes** |

**Per match** (v11; the mix; the polished configuration in sample; the three held-out scores;
possessions):

| match | source | v11 | mix | polished, in sample | judge | nested selection | fully nested | possessions mix → polished (true) | fully nested's pick |
|---|---|---|---|---|---|---|---|---|---|
| J03WMX | IDSSE | 0.5100 | 0.5025 | 0.5272 | 0.5195 | 0.5327 | **0.5233** | 285 → 313 (511) | 6/2, 0.5, all 0.7 0.6, on, 0.5 |
| J03WN1 | IDSSE | 0.5282 | 0.5770 | 0.5831 | 0.5776 | 0.5898 | 0.5887 | 334 → 341 (500) | 6/3, 0.5, all 0.6 0.7, on, 0.5 |
| J03WOH | IDSSE | 0.5676 | 0.5992 | 0.6109 | 0.6109 | 0.6109 | 0.6107 | 309 → 315 (402) | 8/3, 0.4, all 0.6 0.6, on, 0.65 |
| J03WOY | IDSSE | 0.6045 | 0.6277 | 0.6462 | 0.6462 | 0.6462 | 0.6450 | 308 → 332 (456) | 6/3, 0.5, all 0.6 0.6, on, 0.5 |
| J03WPY | IDSSE | 0.6181 | 0.6640 | 0.6643 | 0.6643 | 0.6643 | 0.6714 | 317 → 319 (421) | 4/3, 0.4, all 0.6 0.6, on, 0.5 |
| J03WQQ | IDSSE | 0.5624 | 0.5863 | 0.5877 | 0.5877 | 0.5877 | **0.5848** | 316 → 330 (401) | 4/2, 0.5, all 0.6 0.6, on, 0.65 |
| J03WR9 | IDSSE, unseen | 0.5801 | 0.6202 | 0.6287 | **0.6088** | 0.6238 | **0.6071** | 275 → 279 (370) | 6/3, 0.3, no_pathcrf 0.6 0.6, on, 0.65 |
| metrica1 | Metrica, unseen | 0.5934 | 0.6662 | 0.6789 | 0.6709 | 0.6713 | 0.6815 | 283 → 300 (354) | 4/2, 0.4, all 0.6 0.7, on, 0.65 |
| metrica2 | Metrica, unseen | 0.5811 | 0.6671 | 0.6760 | **0.6669** | **0.6649** | **0.6586** | 290 → 285 (318) | 4/2, 0.4, no_pathcrf 0.6 0.7, on, 0.5 |
| metrica3 | Metrica, unseen | 0.5821 | 0.6463 | 0.6659 | 0.6545 | 0.6545 | 0.6575 | 306 → 332 (381) | 6/2, 0.4, all 0.8 0.7, on, 0.65 |
| **mean** | | **0.5727** | **0.6156** | **0.6269** | **0.6207** | **0.6246** | **0.6229** | **302 → 314 (411)** | |

(Bold: worse than the mix.)

- **J03WMX is fixed:** the mix's one loss against v11 (0.5025 against 0.5100) becomes 0.5195–0.5327
  held out, above v11 under every scoring. Its possessions go 285 → 313 (true 511): segments at
  merge 0.7 instead of the 0.6 its round-1 fold picked, and short2's insertions.
- **What loses:** J03WR9 (−0.011 judge, −0.013 fully nested), where the folds without it prefer
  `no_pathcrf` or other settings while J03WR9 itself gains in sample (+0.0085); metrica2 (−0.0002,
  −0.0085). Both are PathCRF-unseen, so **the unseen check passes by a hair** (+0.0003 / +0.0012,
  2 of 4).
- **Judge picks:** 6/3 in 9 folds (8/3 on J03WMX), p_min 0.5 in 9, segments `all` 0.6 / 0.7 in
  8, stoppage2 on in all 10, short2 0.65 in 8 (0.5 in 2).
- **Fully nested picks scatter:** the stoppage window is 4/2 in 3 folds, 6/3 in 3, 6/2 in 2, 8/3 and 4/3 in
  one each; short's p_min 0.5 in 4, 0.4 in 5, 0.3 in 1; segments merge 0.6 in 5 and 0.7 in 5,
  `no_pathcrf` in 2. **Stable in every fold: stoppage2 on, short2 on (0.5 or 0.65), segments mode `dead`.**
- **Possessions after each stage** (polished, in sample; the mix in brackets): v11 349 → stoppage
  349 → short 372 (372) → segments **309** (302) → stoppage2 309 → short2 **314** (302).

### IDSSE only (`--set idsse`)

The search, in sample: pass 1 stoppage 4/2 → 6/3 (0.6001 → 0.6021), short 0.5 → **0.4** (0.6039),
segments round 1's pick → `no_pathcrf` 0.8 / 0.7 (0.6043), stoppage2 on (0.6045), short2 0.65
(0.6063); pass 2 stoppage → **6/2** (0.6067), short 0.4 kept, segments → `no_pathcrf` **0.8 / 0.6**
(0.6083), stoppage2 on kept, short2 → **0.5** (0.6088). **Pass 2 still changed three stages; the
search stopped at its 2-pass cap without converging.** Polished on `idsse`: 6/2, p_min 0.4,
`no_pathcrf` flip 0.8 merge 0.6, stoppage2 on, short2 0.5; 0.6088 in sample (+0.0087).

| | judge | nested selection | **fully nested** |
|---|---|---|---|
| **spellF1** | **0.6046 against the mix's 0.6001 (+0.0046), 5/7** | 0.6084 (+0.0083), 6/7 | **0.6086 (+0.0085), 6/7** |
| PathCRF-unseen (J03WR9) | 0.6286 against 0.6223 (+0.0063) | +0.0162 | 0.6358 (+0.0135) |
| macroF1 | 0.8831 against 0.8816 | 0.8838 | 0.8842 |
| possessions per match (true 437) | 327 (mix 341) | 330 | 326 |
| against v11 (0.5625) | +0.0421, 7/7 | | +0.0461, 7/7 |
| **the bar** | mean ✓, **5/7 ✓ (the minimum)**, unseen ✓: **passes** | passes | mean ✓, 6/7 ✓, unseen ✓: **passes** |

| match | v11 | mix | polished, in sample | judge | nested selection | fully nested | possessions mix → polished (true) |
|---|---|---|---|---|---|---|---|
| J03WMX | 0.5032 | 0.5148 | 0.5285 | 0.5216 | 0.5216 | 0.5275 | 325 → 317 (511) |
| J03WN1 | 0.5002 | 0.5757 | 0.5887 | 0.5811 | 0.5802 | 0.5757 (+0.00003) | 381 → 374 (500) |
| J03WOH | 0.5667 | 0.5953 | 0.6106 | 0.6106 | 0.6159 | 0.6268 | 323 → 314 (402) |
| J03WOY | 0.6146 | 0.6534 | 0.6494 | **0.6494** | 0.6540 | 0.6540 | 349 → 333 (456) |
| J03WPY | 0.6197 | 0.6686 | 0.6701 | **0.6647** | **0.6647** | **0.6618** | 337 → 320 (421) |
| J03WQQ | 0.5543 | 0.5704 | 0.5765 | 0.5765 | 0.5838 | 0.5787 | 350 → 322 (401) |
| J03WR9 (unseen) | 0.5790 | 0.6223 | 0.6378 | 0.6286 | 0.6385 | 0.6358 | 321 → 300 (370) |
| **mean** | **0.5625** | **0.6001** | **0.6088** | **0.6046** | **0.6084** | **0.6086** | **341 → 323 (437)** |

On IDSSE the polished chain predicts **fewer** possessions than the mix (341 → 323): round 1's
IDSSE segments picks merged at 0.7–0.8 and the chain prefers 0.6. On `all` it predicts more
(302 → 314). So "segments merges too much" isn't what the chain says: the merge threshold it
wants is 0.6–0.7 on both sets, and the extra possessions on `all` come from short2.

## Failure modes

1. **The winner's curse is real here.** In sample the polished chain gains +0.0113 on `all`; held
   out +0.0051 (judge) to +0.0072 (fully nested). About half of what the search found survives
   leaving a match out. On `idsse`: +0.0087 in sample, +0.0046 to +0.0085 held out.
2. **The PathCRF-unseen matches barely move on `all`** (+0.0003 judge, +0.0012 fully nested, 2 of
   4), under the lab's ~+0.002 fragility line. The gain sits on the six PathCRF-seen IDSSE matches
   and on metrica1 and metrica3; J03WR9 and metrica2 lose. On `idsse` J03WR9 gains (+0.006 /
   +0.014), so its loss on `all` comes from Metrica's matches pulling the picks, not from the stages.
   Our match is Metrica-tracked and PathCRF-unseen: the gain to expect there is small.
3. **The match counts sit at the minimum:** fully nested 7/10 on `all`, judge 5/7 on `idsse`; one of
   the fully nested 6/7 on `idsse` is J03WN1 at +0.00003.
4. **The settings within a stage are a coin toss.** The fully nested folds pick five different
   stoppage windows, three short thresholds and two merge thresholds. The bench can't tell them
   apart; only "run stoppage2 and short2" and mode `dead` are picked everywhere. Treat the exact
   values (6/3, p_min 0.5, merge 0.7) as one draw from a flat top, not as measured optima.
5. **Wider stoppage window: stoppage's failure mode 1 grows** (a whistle stoppage, then the ball
   played out within `ahead`: the possession is wrongly extended to that exit). Ahead is now 3 s.
6. **The two sets want different settings** (`all`: 6/3, 0.5, `all` 0.6 / 0.7, on, 0.65;
   `idsse`: 6/2, 0.4, `no_pathcrf` 0.8 / 0.6, on, 0.5), and the `idsse` search hadn't converged
   at its 2-pass cap. For our match, `all`'s are the ones to use (they include Metrica).
7. **Round 2's other stages were built on the mix with round 1's settings** (recall, timing, dead
   in [../lab.md](../lab.md)). On top of the polished chain they would need refitting on its
   letters (`lab_polish.polished_letters`), and their gains won't simply add: recall also inserts
   possessions, like short2. Round 3's chain starts from the learned model and drops segments;
   these settings were chosen on the v11-start chain and don't transfer to it.
8. **Only 41 % damage was tried** (the lab cache has no other level).

## Verdict

**Keep, narrowly.** By the rule fixed before the run, it clears the bar against the mix under the
judge and under the fully nested check, on `all` (+0.0051 / +0.0072) and on `idsse` (+0.0046 /
+0.0085), and the nested selection agrees (+0.0090, 9/10; +0.0083, 6/7). It also turns J03WMX, the
mix's one loss against v11, into a gain. But it is a small gain on one bench after a search:
the unseen matches barely move on `all` (+0.0003 / +0.0012), the match counts sit at the bar's
minimum, and the per-stage values aren't stable across folds. The robust part is running the stoppage
rule and the short stage again after segments, and segments at flip 0.6, merge 0.6–0.7 on short's
output.

## Rebuilding the chain

The polished configuration chosen on all 10 matches (`lab_polish.POLISHED["all"]`), stage by stage
with the round-1 modules:

```python
from bench import lab_segments, lab_short, lab_stoppage
SHORT = lab_short.CHOSEN                       # learned rule, 0.5-5 s, O-N-D on, p_min 0.5
SHORT2 = dict(lab_short.CHOSEN, p_min=0.65)
stop = lab_stoppage.fit(training)              # {"offset": 3} frames on the bench
L = lab_stoppage.apply(match, v11_letters, 6.0, 3.0, model=stop)                         # stoppage
L = lab_short.apply(match, L, model=lab_short.fit(training, letters=<stoppage's out-of-fold letters>, **SHORT), **SHORT)
seg = lab_segments.fit(training, <short's out-of-fold letters>, features="all")
L = lab_segments.apply(match, L, seg, flip=0.6, merge=0.7, mode="dead")                  # segments
L = lab_stoppage.apply(match, L, 6.0, 3.0, model=stop)                                   # stoppage2
L = lab_short.apply(match, L, model=lab_short.fit(training, letters=<stoppage2's out-of-fold letters>, **SHORT2), **SHORT2)
# --set idsse: 6/2 s; short p_min 0.4; segments "no_pathcrf", flip 0.8, merge 0.6; stoppage2 6/2; short2 p_min 0.5
```

Each trained stage is fitted on the training matches' **out-of-fold** output of the stages before
it (for our match: all 10 bench matches, each predicted by the chain trained on the other 9). In
code this is `lab_polish.Chain(matches, ev, lags, ids, round1).letters(config_tuple)`.

**For the next round:** `python -m bench.lab_polish --save [--set ..]` writes each bench match's
letters under its fully nested fold's configuration (chosen without it, every stage fitted
without it): `lab_polish.polished_letters(match_id, which)`, judged against with
`lab.judge(..., base=lab_polish.polished_base(which))`. Their means are the fully nested numbers
(0.6229 on `all`, 0.6086 on `idsse`).
