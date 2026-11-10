# Lab candidate `hsmm`: a decoder that knows how long possessions last

| | |
|---|---|
| **status** | **discarded** (2026-11-09). v12 stays as it is |
| **what** | robust's decoder replaced by an explicit-duration one (a hidden semi-Markov decoder): a stretch of label s lasting d frames costs its frames' −log P, a switch price into it, and λ·(log g_max − log g_s(d)), where g_s is how long true possessions (H and A pooled) and dead balls last on the other matches |
| **why** | [inside-v12.md](../inside-v12.md): the decoder is the largest single step (0.385 → 0.672), and what it misses is short possessions. Its flat price and hard 10-frame minimum make a 0.5 s possession and a 50 s one cost the same; this one should prefer realistic lengths |
| **code** | `bench/lab_hsmm.py`; results in `bench/results/lab_hsmm[_idsse].json`, `lab_hsmm_grid[_idsse].csv`, `lab_hsmm_check.csv`, `lab_hsmm_chains[_idsse].json` |

## How it works

- **The length cost.** For each held-out match, the true stretches of the other matches of the set (the
  first and last of each half left out: cut by the edges): a Gaussian kernel density of log length
  (Silverman's bandwidth), divided by the length, gives g(d). The cost is 0 at the likeliest length and
  grows away from it; the level of the price stays the switch price's job. On `all` (J03WMX's fold) the likeliest
  possession lasts about 1.4 s (cost 1.0 at 5 s, 1.9 at 10 s, 3.0 at 20 s, 5.9 at 60 s, 0.6 at 0.2 s);
  the likeliest dead ball about 9 s.
- **The decoder** (`segmental`): a segmental Viterbi over steps of 5 frames (5 Hz; a step costs its
  frames' sum). Lengths are enumerated up to 60 s for possessions and 150 s for dead balls, vectorised
  over lengths with cumulative sums; a stretch that reaches the cap continues step by step at the cap's
  cost. The first and last stretch of a half get no minimum and no length cost, as in
  `viterbi_min_length`. Then each boundary is moved within ±5 frames to the frame the 25 Hz costs like
  best (`refine`), and v9's snap and v11's move run as for robust (`L.finish`).
- **λ = 0 is robust's decoder.** With the cap at the minimum, `segmental` is `viterbi_min_length`'s
  recursion step for step, ties included.

## What was fixed before the run

| setting | value | why |
|---|---|---|
| λ | 0, 0.5, 1, 2, 4, 8 | 0 is robust's decoder. 4 and 8 were added before any score was seen: on one half of J03WMX, λ = 2 moved the decoded possessions by 2 of 165, so the term looked weak |
| team switch | 4, 2, 1, 8 | around robust's 4 (chosen in every fold); lower ones because λ > 0 adds a cost to every possession but the likeliest |
| dead switch | 25 | robust's, in every fold of both sets |
| minimum possession | 10 frames (robust's) or none | the length cost is meant to replace the hard minimum |
| dead balls' length cost | off or on (same λ) | |
| step, caps, refinement | 5 frames; 60 s and 150 s; ±5 frames | the brief's |
| input | robust's held-out P (`blank`, no blend), −log P | as v12 |

88 settings (λ = 0 doesn't need the dead-ball switch), each match on the setting best on the others
(`lab.judge`), against robust's own held-out score.

## The check: λ = 0 against the min-length decoder

`python -m bench.lab_hsmm --check` (`bench/results/lab_hsmm_check.csv`), robust's held-out P on all 10
matches, both halves:

| | result |
|---|---|
| 25 Hz, cap = minimum (team 4 / min 10, team 8 / min 20, team 2 / min 1) | **identical path on 60 of 60 halves × settings** |
| 25 Hz, lengths enumerated to the caps with zero cost (team 4, min 10) | the same total cost on 20 of 20 halves (gap 0.0); the same path on 11, the other 9 differ on 1–2 frames where the two labels' costs are exactly equal (rounded P): ties broken the other way |
| 5 Hz + refinement, λ = 0 (team 4, min 10) | 99.95 % of frames agree (lowest half 99.89 %); 221 stretches a half against 222; its path costs 2.1 more a half (the price of the coarse steps) |
| the same, scored after snap and move | 0.6736 against the exact decoder's 0.6746 in sample (−0.0009; per match −0.0049 to +0.0022) |

A brute-force check (every segmentation of random 5–40-step problems, 2,000 of them, with and without
enumerated lengths and caps) found the same optimum every time, after one fix: the backtrack forgot to
leave a long stretch when stepping out of a dead ball.

**Decode time** (one half, ~70k frames): `viterbi_min_length` 0.14 s; this decoder at 5 Hz 0.04 s with
λ = 0, 0.24 s with λ > 0 and dead lengths on; at 25 Hz with every length enumerated, 2.1 s.

## The start alone, against robust

| | all 10: 0.6722 vs 0.6728 | IDSSE set: 0.6367 vs 0.6365 |
|---|---|---|
| overall | −0.0006, 3/10 | +0.0003, 3/7 |
| IDSSE's matches | −0.0001, 3/7 | +0.0003, 3/7 |
| Metrica's matches | −0.0019, 0/3 | — |
| PathCRF-unseen | −0.0009, 1/4 | −0.0038, 0/1 |
| bar | **fails** every check | **fails** (share, unseen) |
| macroF1 | 0.9145 vs 0.9145 | 0.9115 vs 0.9115 |

Folds' picks on `all`: λ = 0 with team 4 and the 10-frame minimum (robust's decoder, at 5 Hz) in 7 of
10, λ = 1 or 2 with the same price and minimum in 3. On the IDSSE set: λ = 0.5, no minimum, team 4 in
4 of 7; λ = 0 in 3. Possessions per match with each fold's pick: 340 on `all` (robust 342), 346 on
`idsse`.

**In sample the grid is flat**, best setting at each λ (mean of all 10, after snap and move):

| λ | 0 | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|
| `all` | **0.6736** | 0.6729 | 0.6734 | 0.6733 | 0.6721 | 0.6708 |
| `idsse` | 0.6400 | **0.6401** | 0.6395 | 0.6383 | 0.6364 | 0.6360 |

Every λ's best keeps team 4 on `all`, and possessions per match hardly move (341 at λ = 0, 337 at
λ = 8). The 5 Hz approximation alone moves single matches by up to ±0.005 (0.05 % of frames), which is
as large as anything λ does.

## Why the length cost can't move much

robust's per-frame costs, not the decoder's prices, decide nearly every possession
(robust's held-out P, its own decoder, all 10 matches; truth read for this description only):

- **What the decoder keeps is not marginal.** Of the 1,665 decoded possessions sitting between two of the
  other team's, dropping one would cost a median 292 nats of frame evidence beyond the two switch prices;
  0.9 % are within 2 nats and 3.7 % within 6. A length cost of 0–6 × λ reaches only those.
- **What it misses has no evidence to find.** Over the frames of the 1,141 true possessions under 2 s,
  robust's P favours the other team in 43 % (no decoder finds those), and only 13 % have between 0 and
  10 nats in their favour: the ones a cheaper short possession could pick up, along with a similar
  number of false blips. The grid's low team prices with λ > 0 (short cheap, long dear) do just that and
  score lower in sample.

So the decoder already does what a length prior could do on these probabilities. Finding the short
possessions needs better frame evidence (robust, or a stage after the decoder that adds its own, as
`short` and `recall` do), not a better prior on lengths.

## v12's chain from this start, against v12

`python -m bench.lab_hsmm --chains [--set idsse]`: each match's start decoded with its own fold's pick,
then v12's post stages, each refitted on this start's output and on its setting for that match. robust's
chain reproduces v12 (0.7033, 0.6699).

| | all 10 | IDSSE's 7 | Metrica's 3 | PathCRF-unseen | IDSSE set |
|---|---|---|---|---|---|
| v12 | 0.7033 | 0.6674 | 0.7870 | 0.7617 | 0.6699 |
| hsmm start | 0.7037 | 0.6671 | 0.7890 | 0.7646 | 0.6649 |
| gain, matches better | +0.0004, 4/10 | −0.0003, 3/7 | +0.0020, 1/3 | +0.0030, 2/4 | **−0.0049, 1/7** (J03WR9 −0.0053) |
| bar | **fails** (share, IDSSE drops; and under +0.005) | | | | **fails** every check |

macroF1 0.9185 against 0.9182 on `all`, 0.9153 against 0.9162 on `idsse`. Possessions per match after the
chain: 341 against 340 on `all`, 374 against 374 on `idsse`.

## What it taught us

1. **On robust's probabilities, the min-length decoder is already the right decoder.** A prior on
   lengths, at any weight from 0.5 to 8, with or without the minimum, with or without dead balls, never
   beat λ = 0 in sample on `all`, and folds chose λ = 0 7 times of 10. The decoder's big step comes from
   smoothing robust's flicker; what's left is decided by the frames, not the prices (see "Why").
2. **The misses need evidence, not a prior.** 43 % of the true possessions under 2 s have robust's
   P favouring the other team over their frames. That's work for the classifier or for stages that bring
   their own evidence (`short`, `recall`), not the decoder.
3. **A noise floor for chain comparisons.** In 7 of 10 folds on `all` this start is robust's own decoder
   at 5 Hz, agreeing with it on 99.95 % of frames. Those matches still move by −0.0094 to +0.0082 after
   the chain (the post stages are refitted on slightly different input). Single matches moving by
   ±0.009 from near-identical starts is another reason to hold to the +0.005 bar and the share checks.
4. **The IDSSE set's −0.0049**: 5 of its 7 folds picked no minimum (4 of them with λ = 0.5). At the
   start that was +0.0003; after the chain, −0.0049, so the post stages gained 0.005 less on top of it,
   probably because their settings were chosen on robust's chain, which starts with the minimum.

If a decoder change is tried again, it should come with new evidence per frame or per boundary (lab.md
round 6's `switch` was the closest: a price from a boundary model, +0.0045 at the start, 6/10), not a
better prior on the shapes of the stretches.
