# Model v3: v2's clues, a decoder that knows how short a possession can be, settings from the bench

| | |
|---|---|
| **status** | **built 2026-10-30**, as proposed, with the four recommended choices |
| **output** | SHA-256 `2474718693efe940906a74d4e1d2c89e227214d97109d20dccc1c470bbb4a9aa`; byte-identical on re-run. The search (`python -m bench.tune_decoder`, ~5 min on 4 processes) also gives identical files on re-run |
| **change from v2** | the decider only: lower switch prices, a minimum length per label, and every decoder setting chosen on the [bench](../bench.md). All of v2's clues and their weights stay, except two weights that are rescaled so they keep their meaning (see "Tags and whistle") |
| **new code** | `possession/decoder.py` gains a minimum-length decoder next to the current one; `possession/v3.py`; `bench/tune_decoder.py` (the search) |
| **run** | `python -m possession.v3` → `evaluation/results/v3.csv`. `python -m bench.tune_decoder` re-does the search (~30 min); the submission doesn't need it, because the chosen values are written into `v3.py` |
| **new dependencies** | none |

## Why

The bench showed that v1 ties the organisers' floor (spellF1 0.224 against 0.217 at our damage
level) because it **merges**: it finds ~148 possessions per match where there are ~437
([../bench.md](../bench.md)). Two things are behind that:

1. **The team switch price is high.** v1 charges 15 to change team, which needs about 0.6 s of
   clear, one-sided evidence. The evidence for a team (clue A, "who is on the ball") only fires
   when the ball is seen, slow, and within 2 m of a player. On the bench, half of all real
   possessions last under 3.7 s and a quarter under 1.7 s, so many never collect enough evidence
   to pay the price.
2. **The price is v1's only defence against flicker.** Lower it, and a single frame of misleading
   evidence (two players from both teams near the ball) can create a 1-frame possession. Each fake
   possession costs one false positive on spellF1, the same as missing a real one.

Splitting that one knob into two fixes the conflict:
- **a minimum length per label** stops flicker by itself;
- so **the switch price can drop** to whatever lets real possessions through.

## What we measured first (bench only: other matches)

**How long things really last** on the 7 bench matches (3,061 possessions, 713 stoppages):

| | 1 % | 5 % | 10 % | 25 % | median | 75 % | 90 % |
|---|---|---|---|---|---|---|---|
| possession | 0.12 s | 0.48 s | 0.84 s | 1.68 s | 3.68 s | 8.52 s | 18.6 s |
| stoppage | 1.93 s | 3.88 s | 5.76 s | 10.6 s | 19.6 s | 32.1 s | 49.4 s |

- A minimum possession length of 0.4 s (10 frames) would erase 3.8 % of real possessions; 0.8 s,
  9 %. So the minimum has to stay short.
- A minimum stoppage of 2 s (50 frames) erases about 1 % of real stoppages. It can be longer.

**What lowering only the team price does** to v1's clues on the bench (41 % damage, all 7
matches, everything else unchanged). This was exploration, to know whether the idea is worth a
page. The final settings are chosen properly, below.

| team switch price | spellF1 | possessions found (true 437) | macroF1 | D F1 |
|---|---|---|---|---|
| 15 (v1) | 0.224 | 148 | 0.683 | 0.732 |
| 8 | 0.256 | 177 | 0.682 | 0.727 |
| 4 | 0.279 | 207 | 0.681 | 0.725 |
| 2 | 0.298 | 239 | 0.681 | 0.725 |
| 1 | 0.307 | 268 | 0.680 | 0.725 |
| 0.5 | 0.310 | 300 | 0.677 | 0.723 |

Changing the dead-ball price from 25 to 10 moved nothing beyond 0.001.

**What that tells us:**
- **Most of the gain may come from the price alone:** +0.08, twice the noise band.
- **Even at 0.5 there is no flicker yet.** It still finds fewer possessions than there are,
  because clue A is silent most of the time and, while it is, nothing pushes the decoder anywhere.
- So the minimum length is a **safety net** that may or may not add score. The search decides
  that; if it doesn't help, it will choose a minimum of 1 frame, which is v1's decoder.

## How it works

### Piece 1: a decoder with a minimum length per label (`possession/decoder.py`)

Today's decoder finds the cheapest label sequence given per-frame costs and a price per switch.
The new one adds one rule: **once a label starts, it lasts at least `k` frames**, with its own `k`
for possessions (H, A) and for dead ball (D).

- **How:** each label becomes a short chain of `k` steps. Entering a label puts the decoder on step
  1, every frame moves it one step, and only from the last step can it stay or leave. Same exact
  search, with about `k` times the states per frame; expected to stay around a second per half.
- **Edges of a half:** the first and last stretch of a half may be shorter than `k`, because the
  half can cut them.
- **`k = 1` is exactly today's decoder.** The new function will be tested to give identical output
  to the current one on the full match with `k = 1`.

**Worked example** (to be checked on the real decoder when built). Cost 1 per frame that disagrees
with the clue, team price 0.3, clues `H H A H H`:
- with `k = 1`: `H H A H H`, two switches (0.6) are cheaper than ignoring the blip (1), so it
  invents a 1-frame possession;
- with `k = 3`: an A stretch would have to cover frames 3–5, paying for frames 4 and 5 (2) plus a
  switch (0.3), so `H H H H H` (1) wins and the blip is ignored.

### Piece 2: the four settings, chosen on the bench

| setting | candidates | why these |
|---|---|---|
| team switch price | 0.25, 0.5, 1, 2, 4, 8, 15 | from far below where the exploration flattens out up to v1's value |
| dead-ball switch price (into and out of D, one value as in v1) | 5, 10, 25, 50 | around v1's 25 |
| minimum possession length `k_poss` | 1, 5, 10, 20 frames (0.04–0.8 s) | above 20 frames, more than 9 % of real possessions would be erased |
| minimum stoppage length `k_dead` | 1, 25, 50 frames (up to 2 s) | about 1 % of real stoppages are shorter than 2 s |

That's 336 combinations. v1's clue weights stay as they are; choosing them is step 6 of the plan.

**How a combination is chosen: leave-one-match-out.**
1. Compute v1's clues once per bench match at 41 % damage, without the tag clues, which the bench
   can't feed. The clues don't depend on the decoder, so only the decoding is repeated.
2. Score every combination on every match (336 × 7 spellF1 values, saved to
   `bench/results/decoder_grid.csv`).
3. **Held-out estimate:** for each match in turn, pick the best combination on the other 6 (mean
   spellF1) and score it on the one left out. The 7 held-out scores are the honest estimate of
   what the procedure is worth on an unseen match. They're compared with v1 (0.224) and the floor
   (0.217) on the same matches.
4. **The setting used for the submission:** chosen the same way on all 7 matches.

**Tie rule (decision 1 below).** Bench scores within ~0.01 of each other are a tie. Among the
combinations within 0.01 of the best, take the one that **finds the fewest possessions**. The bench
follows DFL's possession convention; if Metrica's truth has fewer, longer possessions, the more
cautious setting loses less.

**Checks on the choice, reported but not used to choose:**
- the chosen setting's score at 30 % and 55 % damage (does it only work at one level?);
- the score with the merged labels (does the ranking against v1 hold?);
- how many of the 7 folds pick the same setting (is the choice stable?).

### Piece 3: tags and whistle keep their meaning (`possession/v3.py`)

The bench can't see Smart Tagging or the whistle. Their weights were set by reasoning, **relative
to the switch prices**, so lowering the prices would silently make them much stronger:

- **Tag team clue:** v1 gives it 0.2 per frame against a team price of 15, meaning "about 3 s of
  tag-only evidence is needed to flip possession". At a team price of 1, the same 0.2 would flip it
  in 5 frames, so a fixed-length tag window would start creating possessions. **New weight =
  0.2 × (new team price / 15)**, which keeps "3 s of tag-only evidence".
- **Set-piece window:** 0.3 per frame against a dead-ball price of 25. **New weight = 0.3 × (new
  dead price / 25)**, which keeps the same meaning.
- **Whistle** (v2): 2.0 per frame for 3 s, set to beat "into D + out of D" (150 vs 50). Any
  dead-ball price up to 50 keeps that true, so it **stays as it is**. The same holds for the
  "before the 2nd-half kick-off" rule (1000 per frame).

These rescalings are reasoning, not fitting. They're listed in the settings table below as such.

### Piece 4: everything else

v2's clues, weights and whistle detector are unchanged.

## Every setting, and where its value comes from

| setting | value | source |
|---|---|---|
| team switch price | chosen from the grid | bench, leave-one-match-out, tie rule |
| dead-ball switch price | chosen from the grid | bench, same |
| minimum possession length | chosen from the grid | bench, same |
| minimum stoppage length | chosen from the grid | bench, same |
| tag team weight | 0.2 × team price / 15 | reasoning: keep v1's "3 s of tag-only evidence" |
| set-piece weight | 0.3 × dead price / 25 | reasoning: keep v1's meaning |
| everything else | as in v1 and v2 | see [v1](v1-clues-and-decider.md), [v2](v2-whistle.md) |

## What to expect on our match

- **More possessions than v2's 177.** On the bench, a team price of 1 roughly doubled v1's count.
  We'll report v3's count and length distribution on the full match next to v2's. Those are
  unlabelled numbers, so they're allowed.
- **Dead ball roughly unchanged:** on the bench the dead-ball price barely mattered, and the
  whistle and tag rules keep their meaning.
- **Public 5 minutes:** run as a pipeline check only. Its score is recorded and not used to choose
  anything (EVALUATION.md).

## Expected failures and risks

1. **The bench's convention may not be ours.** DFL possessions are short (median 3.7 s). If
   Metrica's truth has longer possessions, a low team price over-splits on our match. Mitigation:
   the tie rule (decision 1), plus comparing v3's full-match possession count with v2's.
2. **Over-fitting the bench.** Four settings chosen on 7 matches is modest, and the held-out
   estimate measures the procedure honestly, but it's still one league and one damage model.
   The 30 % and 55 % checks guard against fitting the damage level.
3. **Duels.** With a low price, two players from both teams near a slow ball can make the decoder
   flip on each frame. The minimum possession length is there for this; the search tells us
   whether it is needed.
4. **The tag rescaling is untested.** It is reasoned and keeps v1's stated meaning, but nothing
   measures it. If v3's full-match count jumps far more than the bench predicts, the tags are the
   first suspect.
5. **Runtime of the search:** ~336 × 7 decodings, about 30 minutes, run once in the background.

## Results (2026-10-30)

### Tests of the decoder

- **Exhaustive:** on 400 random small problems (up to 7 frames, minimum lengths 1–4) the new decoder
  finds the same lowest total as trying every possible label sequence, and every stretch it returns
  respects its minimum.
- **Minimum length 1 = today's decoder:** identical output on 200 random problems full of ties, and
  on the full match: v3 run with v1's prices and minimum 1 writes a file byte-identical to v2.csv.
- **Worked example:** `HHAHH` with a minimum of 1, `HHHHH` with a minimum of 3, as written above.
- **Speed:** ~0.2 s per 90-minute match, so the whole 336-setting search takes about 5 minutes.

### The search on the bench (41 % damage, raw labels)

**Every one of the 7 folds picked the same setting:** team price **0.5**, dead-ball price **50**,
minimum possession **20 frames (0.8 s)**, minimum stoppage **1 frame (none)**.

| held-out match | v3's setting, chosen on the other 6 | v1 | possessions found / true |
|---|---|---|---|
| J03WMX | 0.365 | 0.217 | 292 / 511 |
| J03WN1 | 0.318 | 0.151 | 297 / 500 |
| J03WOH | 0.375 | 0.232 | 291 / 402 |
| J03WOY | 0.383 | 0.234 | 281 / 456 |
| J03WPY | 0.369 | 0.242 | 282 / 421 |
| J03WQQ | 0.370 | 0.256 | 270 / 401 |
| J03WR9 | 0.405 | 0.238 | 245 / 370 |
| **mean ± sd** | **0.369 ± 0.026** | **0.224** | 280 / 437 |

Better than v1 on all 7 held-out matches, by +0.145 on average: more than three times the noise
band. The floor scores 0.217 on the same matches.

**What each setting does** (best spellF1 reachable with that setting fixed, mean over 7 matches):

| setting | values → best spellF1 |
|---|---|
| minimum possession | 1 frame 0.312 · 5 frames 0.368 · 10 frames 0.371 · **20 frames 0.376** |
| team price | **0.25 0.376** · 0.5 0.370 · 1 0.359 · 2 0.343 · 4 0.311 · 8 0.283 · 15 0.263 |
| dead-ball price | 5–50 within 0.0014 of each other |
| minimum stoppage | no effect at all (1, 25 and 50 frames score the same) |

- **Both halves of the idea mattered.** The minimum length is not just a safety net. At the lowest
  prices, going from no minimum to 5 frames is worth +0.06, because without it the decoder does
  start to flicker. And with it, the lower the price the better.
- **The tie rule moved the choice.** The plain best is team 0.25, dead 25 (0.376, 303 possessions);
  the rule took team 0.5, dead 50 (0.370, 280 possessions), the most cautious setting within 0.01.
- **Two settings sit on the edge of the grid.** The best minimum possession is the largest candidate
  (20 frames), and the plain best team price is the smallest (0.25). A wider grid might score a
  little higher on the bench. The grid's limits were reasoned (a longer minimum erases more real
  possessions), so widening it is a decision, listed below, not something done here.

**Checks, not used to choose:**

| | chosen setting | v1 |
|---|---|---|
| 30 % damage | 0.384 | 0.224 |
| 41 % damage (where it was chosen) | 0.370 | 0.224 |
| 55 % damage | 0.310 | 0.178 |
| merged labels (possessions under 0.5 s merged) | 0.382 | 0.238 |

The gain holds at every damage level and with both label sets.

### On our match (no ground truth involved)

| | v2 | v3 |
|---|---|---|
| possessions | 177 | **429** |
| median / mean length | 16.1 s / 22.9 s | 4.6 s / 8.9 s |
| possessions under 1 s | 0 | 74 (none under 0.8 s, the minimum) |
| dead ball, share of graded frames | 36.3 % | 39.9 % |
| Barcelona's share of live frames | 70.1 % | 65.1 % |
| frames that differ from v2 | — | 18,539 (11.7 % of graded) |

The rise in possessions is a bit larger than on the bench: ×2.4 here against ×1.9 there.

### On the public 5 minutes (a pipeline check, not a ranking)

| | macroF1 | spellF1 | possessions (truth: 17) | mean length (truth: 14.2 s) |
|---|---|---|---|---|
| v2 | 0.789 | 0.427 | 13 | 18.1 s |
| **v3** | 0.704 | **0.335** | 20 | 10.6 s |
| floor | 0.621 | 0.326 | 26 | 10.8 s |

- **What it proves:** the pipeline is wired correctly (the file is valid, frames line up, the
  scorer reads it).
- **What it doesn't:** that v3 is worse than v2. One possession moves this window's spellF1 by about
  6 points, and the difference is about 1.5 possessions out of 17. EVALUATION.md is explicit that
  this window must not choose between versions.
- **Recorded, not acted on:** v3 finds more possessions than the truth in this window, where v2 found
  fewer. That fits risk 1 below (Metrica's possessions may run longer than DFL's), and it fits the
  ×2.4 above. Nothing here can settle it without tuning on the window.

## Not in this step (later, if the bench says it's needed)

- a full duration model (each label gets a length distribution, not just a minimum);
- a second pass that finds boundaries and takes a majority vote inside each stretch;
- decoding for the best expected spellF1 rather than the cheapest path.

## Decisions for Jose before building

| | choice | recommended | why |
|---|---|---|---|
| 1 | Tie rule: plain best bench score, or the most cautious (fewest possessions) within 0.01 of the best | **most cautious within 0.01** | 0.01 is inside the noise, and caution hedges against DFL's convention having shorter possessions than Metrica's |
| 2 | Choose on 41 % damage only, or on the average of 30/41/55 | **41 % only**, with 30/55 as reported checks | 41 % is what our tracking measures; averaging would let levels we don't have pull the choice |
| 3 | Rescale the tag and set-piece weights with the new prices | **yes** | Without it, lowering the price silently makes untestable clues 15× stronger |
| 4 | Name: v3 (v2's clues + new decoder) | **yes** | Each version changes one thing on top of the last, as v2 did |

## Open after building

1. **Widen the grid?** The bench's best minimum possession (20 frames) and plain-best team price
   (0.25) sit on the grid's edges. Trying 30/40/50 frames and 0.1 would show whether the bench keeps
   improving. The limits were reasoned (each extra frame of minimum erases more real short
   possessions), and the convention risk below argues against pushing further on DFL's data alone.
2. **Answered by the [convention check](../convention-check.md) (2026-10-30):** Metrica's events
   change hands about 20 % less often than DFL's, but on Metrica's own labels v3 scores 0.512 against
   v1's 0.384, finds 344 possessions where there are 351, and the search on all 10 matches picks
   exactly v3's setting in every fold. The original note follows.
   **The convention risk is now the main unknown.** The bench (DFL's convention) says more, shorter
   possessions score better; our match's output shows a larger jump than the bench predicts, and the
   public window, which can't rank anything, leans the other way. What would settle it is labelled
   data with Metrica's own convention on another match: Metrica's sample games, with labels built
   from their events (deferred as decision 3 in [../bench.md](../bench.md)), or a match we label
   ourselves (allowed: #8, #10).
