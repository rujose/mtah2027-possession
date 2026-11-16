# Night, builder "netsc2": SkillCorner's even 10 in `gru_b`'s training, for dead or live only

| | |
|---|---|
| **status** | **fails the robustness bar** (2026-11-15 05:50): B's bench **−0.0074** (0.7254 against v15's 0.7329), better on 4/10, p25 −0.0063, p50 −0.0070, stressed −0.0089, IDSSE −0.0071, Metrica −0.0081 (part 1 fails on all three clauses); SkillCorner's untouched odd 10 **+0.0180**, better on **10/10** (part 2 passes). The round-8 B bar fails too. Against `netsc`: B's bench +0.0058 (7/10), SkillCorner's odd 10 −0.0413. Dead or live only removes most of `netsc`'s timing loss, but SkillCorner's "dead" still costs live play on the bench, and the stoppage balls alone are worth about a third of `netsc`'s SkillCorner gain. Nothing adopted; seed 1 not trained (a fail). Plan written 03:50, before any score |
| **question** | `netsc` ([netsc.md](netsc.md)) taught `gru_b` real moving stoppage balls (SkillCorner's odd 10 **+0.0593, 10/10**) but also SkillCorner's timing of a change of team, which cost B's bench **−0.0132** (2/10; the team right ~3 points less often within 1 s of a true change). If SkillCorner's frames teach the network **only dead or live** (not which team, so not when the team changes), does B keep most of the stoppage-ball gain on SkillCorner without losing B's bench? |
| **proposal** | `netsc`'s closing note ([netsc.md](netsc.md), "What Jose decides", point 2); picked in [../night.md](../night.md) at 03:50 |
| **code** | `bench/lab_netsc2.py` (new file). It imports `bench/lab_netsc.py` (SkillCorner's segments, the per-frame diagnostics) and `bench/lab_netmirror.py` (the mirror, confirm, SkillCorner); neither is edited. Its training loop is `lab_netmirror.fit` copied line for line (no owner head), with the SkillCorner frames' loss changed |
| **outputs** | `data/external/lab_netsc2/s0/` (nets, the held-out P, trial 1015's stage tables recomputed there; started empty; 106 MB), `bench/results/lab_netsc2/s0/` (`check.json`, `judge.json`, `confirm_1015.csv`, `skillcorner_1015.csv`, `frames.csv`, `scframes.csv`, `confirm_1015_normal_p10.csv`). No seed 1. Everything finished 05:55; no `netsc2` job or slot is left running. Logs in the scratchpad's `logs/netsc2/` |
| **our match** | never read, nothing run on it |
| **licence** | SkillCorner open data ([github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata)), **MIT**; "we kindly ask that you credit SkillCorner". Data by SkillCorner. If this is ever adopted, the credit goes into the README and the model page |

## What changes, exactly

`netsc`'s recipe, one thing changed: the loss on SkillCorner's frames.

| | `gru_b` (v15) | `netsc` | **`netsc2`** |
|---|---|---|---|
| training matches | the fold's DFL matches (6, or all 7 for `metrica1`), as they are and under `augment`'s training stress, both orientations | the same + SkillCorner's even 10 (`lab_carried.sc_pool('train')`), as they are, each half in both orientations | **the same as `netsc`** (the same segments in the same order) |
| loss on DFL frames | 3-way cross-entropy (H / A / D) | the same | **the same** |
| loss on SkillCorner frames | | 3-way cross-entropy against SkillCorner's own H / A / D | **dead or live only**: binary cross-entropy of P(D) against P(H) + P(A) = 1 − P(D). Target: D where SkillCorner says D ("no phase of play"), live where it says H or A |
| cut-away frames (SkillCorner's `no_data`) | | out of the loss | **out of the loss** |
| weighting | each graded frame counts once (the batch's mean) | the same: each graded frame of either source counts once | **the same**: (DFL frames' summed 3-way loss + SkillCorner frames' summed 2-way loss) / graded frames in the batch. No weight between the two losses is tuned |
| normaliser, seed, epochs, chunks, optimiser | the fold's DFL matches; seed 0; 20 epochs; 60 s chunks at 5 Hz; AdamW, cosine | the same | **the same** |
| folds | 7 DFL folds + `metrica1` (all 7 DFL; Metrica's 3 share it) | the same 8 | **the same 8** |
| SkillCorner's deployment net | `gru_b`'s `metrica1` net | `netsc`'s `metrica1` net | **`netsc2`'s `metrica1` net** (all 7 DFL + the even 10: it is the "all" network, as in v15 and `netsc`, so no ninth training) |

**How the loss is implemented (the simplest faithful way).** The network's head is unchanged (3 logits, a softmax).
SkillCorner's labels are recoded before training: D → code 10 ("dead"), H or A → code 11 ("live"), cut-aways stay
3 (ignored), in both orientations (so the swapped copy carries the same targets; only its inputs differ). In a batch:

- DFL frames (codes 0-2) give −log P(true class), as in `gru_b`;
- SkillCorner frames give −log P(D) if dead, −log(P(H) + P(A)) if live (computed as a log-sum-exp of the log-softmax);
- the batch's loss is the sum of both over the number of graded frames.

On a SkillCorner **dead** frame this is exactly `netsc`'s loss (−log P(D)); only the **live** frames change: they no
longer say which team has the ball, so nothing of SkillCorner's possession timing (its change of team that doesn't
sit on a touch) reaches the network. A batch with no SkillCorner frame takes `gru_b`'s exact code path
(`CrossEntropyLoss`), so with no SkillCorner match added the wrapper must give `gru_b`'s recipe's weights to the bit.

Everything after the network is trial 1015 as v15 deploys it, settings fixed (w_gru 0.915, f_dead 0.909, switch
prices 2.82 / 48.7, minimum 7, snap and move off, short 0.340, segments 0.910 / 0.661, timing 0.450, post stages
trained on the training-stress copies). Nothing is searched again.

**SkillCorner's odd 10 stay untouched**: nothing trains or tunes on them (the code refuses to build their segments),
and they are the source of the bench's test variants' stoppage balls, so nothing leaks into B's bench either.

## The bar (fixed before any score; `netsc`'s robustness bar, copied)

**The robustness bar** (from [ideas_outside.md](ideas_outside.md), idea 4, as [netsc.md](netsc.md) used it; against
**v15's B**, seed 0, trial 1015's settings fixed). **Accepting a robustness bar instead of the B bar is Jose's call**
(as round 7 did for `carried`).

1. B's bench (each match's mean of p25, p50, stressed): **no loss beyond 0.002** on the mean (≥ **0.7309**; v15's B
   0.7329); **no variant down more than 0.003** (p25 ≥ 0.7259, p50 ≥ 0.7289, stressed ≥ 0.7349; v15's 0.7289 /
   0.7319 / 0.7379); **neither source (IDSSE's 7, Metrica's 3) down more than 0.005** on its own mean.
2. **SkillCorner's odd 10: +0.010 or more over v15's B** (v15's B 0.3992, so ≥ **0.4092**), **better on 7 of 10**.
   The even 10 are reported as in sample, never judged.

Against v15's B as confirmed: trial 1015 in `bench/results/search_b/confirm_1015_554_586_1133_1269_-1_-2.csv` and
`skillcorner_1015_554_586_1133_1269.csv`, per match. The deployment models score SkillCorner as `search_b
skillcorner` does (augment's deployment trees, this recipe's `metrica1` net, trial 1015's stage models in the
mirror).

**The round-8 B bar, for information only** ([../night.md](../night.md)): B's bench **+0.005** or more (0.7379),
**better on 7 of 10**, no variant down more than 0.002, neither source down on the mean, SkillCorner down no more than
0.005 (on the 20: ≥ 0.4035; the even 10 are in sample here, so the odd 10 are shown beside it).

**Seed noise and the noise rule.** `gru_b`'s seed alone (seed 1 against seed 0) moves B's bench by **−0.0035**
(4/10) and SkillCorner's odd 10 by **−0.0040** (3/10) ([network.md](network.md), [netsc.md](netsc.md)). s ≥ 0.003,
so **a seed-0 pass counts only if the same recipe with seed 1 also passes both parts**. Seed 1 is trained **only if
seed 0 passes and there is time** (it can add a condition to a pass; it can't rescue a fail). If seed 0 passes and
seed 1 can't finish by 08:45, the verdict is "passes, pending the noise rule".

## Also reported (information, truth only to report)

As `netsc` did, where it moves (the GRU alone, argmax): on SkillCorner's odd 10, dead frames called dead, the real
stoppage ball (dead with the ball tracked and moving) called dead, live frames called dead, the team right away from
and near a change; on B's bench (held-out P, B's 3 variants), **the team right within 1 s of a true change** and the
switch offset, `gru_b` → `netsc` → `netsc2`. Also `netsc2` against `netsc` itself on B's bench and SkillCorner (the
same segments, the same seed: the cleanest reading of what the 2-way loss changes).

## The run plan

Every heavy job goes through the scratchpad's `slot.py` (one slot, 2 threads), **at most two at a time** (the network
builder's runs come first). `df -h /private/tmp` is checked before each big write; under 4 GB free, everything stops
and the reason goes here.

| step | jobs | cost |
|---|---|---|
| checks | (a) with no SkillCorner match the training loop gives `lab_netmirror.fit`'s weights to the bit (= `gru_b`'s recipe, which the network builder checked against `lab_branch_b.fit`), 1 epoch on 2 matches; (b) with one SkillCorner match and its 3-way labels, it gives `lab_netsc.fit`'s weights to the bit (the same plumbing as `netsc`); (c) the 2-way loss against a hand computation on a mixed batch; (d) the recoded labels' shares on one fold's normaliser | ~5 min |
| seed 0 training | 8 folds, each followed by its held-out P (5 bench variants + the training-stress copy) into the mirror | 8 × ~18 min, two at a time |
| B rebuilt | trial 1015's stage tables in the mirror (computed there, never reused), then `confirm` (trials 0 and 1015, B's 3 variants; trial 0 must equal v15's rows) and SkillCorner's 20 with the `metrica1` net | ~25 min |
| judge | the robustness bar; the B bar (information); per source, per variant, per match; SkillCorner odd and even apart | |
| information | the per-frame diagnostics above (`frames.csv`, `scframes.csv`) | ~10 min |
| seed 1 | only if seed 0 passes (the noise rule) | as seed 0 |

## Results

### The checks (03:52): pass, no score involved

`bench/results/lab_netsc2/s0/check.json`; 1 epoch on J03WN1 + J03WOH for (a), (b), (b').

| check | result |
|---|---|
| (a) no SkillCorner match: `netsc2`'s loop against `lab_netmirror.fit` (`gru_b`'s recipe) | max \|difference\| over the weights **0.0**; the same loss (0.4182) |
| (b) one SkillCorner match (1874553) with its 3-way labels: against `lab_netsc.fit` | max \|difference\| **0.0**; the same loss (0.4137). So the segments, their order and the batches are `netsc`'s |
| (b') the same match, dead / live only | the weights move (max \|difference\| 0.026): the 2-way loss is what differs. Epoch loss 0.3653 (DFL 3-way 0.393, SkillCorner 2-way 0.248 in the mixed batches) |
| (c) the loss on a mixed batch (3-way, dead, live, ignored codes) against a hand computation in float64 | difference 8e-9; on SkillCorner's dead frames alone it equals the 3-way loss exactly (0.0); a batch with no SkillCorner frame takes `gru_b`'s call |
| (d) the recoded labels (1874553, both orientations) | H 80,679 + A 80,679 → live 161,358; D 59,854 → dead 59,854; cut-aways 80,192 stay out; the inputs identical to `netsc`'s |

The seed-0 queue (8 folds, two at a time) started at 03:53.

### Seed 0 (05:48): fails the robustness bar on part (1); part (2) passes

8 folds trained 03:53–05:37 (16–19 min each; final training loss 0.112–0.118: the DFL frames' 3-way part 0.124–0.132,
`gru_b`'s ~0.125; SkillCorner's 2-way part 0.093–0.096). Trial 1015's stage tables computed in the empty mirror (46
s), then `confirm` and SkillCorner. **Trial 0 reproduces v15's rows exactly** (max |difference| 0.0 on B's bench and
on SkillCorner's 20), so every difference below is the network's.

| | v15's B | `netsc2` | difference | bar | |
|---|---|---|---|---|---|
| **B's bench** (mean of p25, p50, stressed) | 0.7329 | **0.7254** | **−0.0074**, better on 4/10 | ≥ −0.002 | **fails** |
| p25 / p50 / stressed | 0.7289 / 0.7319 / 0.7379 | 0.7225 / 0.7248 / 0.7290 | −0.0063 / −0.0070 / −0.0089 | each ≥ −0.003 | **fails** (all three) |
| IDSSE's 7 / Metrica's 3 | | | −0.0071 / −0.0081 | each ≥ −0.005 | **fails** (both) |
| **SkillCorner's odd 10** (the guard) | 0.3992 | **0.4172** | **+0.0180**, better on **10/10** | ≥ +0.010, 7/10 | passes |
| SkillCorner's even 10 (in sample, not judged) | 0.4179 | 0.4364 | +0.0186, 10/10 | | |
| SkillCorner's 20 | 0.4085 | 0.4268 | +0.0183, 20/20 | | |

**The robustness bar: fails** (part 1 fails on every clause; part 2 passes). **The round-8 B bar (information):
fails** (−0.0074 against +0.005, 4/10; its SkillCorner part passes). The bench loss is **2.1 times the seed noise**
(s = 0.0035) and is on all three variants and both sources, so seed 1 isn't trained: the noise rule can add a
condition to a pass, never rescue a fail. Against B as built (trial 0, no GRU), `netsc2`'s B gains +0.0279 on B's
bench (v15's B +0.0353, `netsc` +0.0222) and +0.0141 on SkillCorner's odd 10.

Per match on B's bench (v15's B → `netsc2`; p25 / p50 / stressed), and `netsc` beside it:

| match | source | v15's B | `netsc2` | difference | p25 | p50 | stressed | `netsc` − v15 | `netsc2` − `netsc` |
|---|---|---|---|---|---|---|---|---|---|
| J03WMX | IDSSE | 0.6254 | 0.6213 | −0.0041 | −0.0047 | +0.0003 | −0.0079 | −0.0180 | +0.0139 |
| J03WN1 | IDSSE | 0.6691 | 0.6568 | −0.0123 | −0.0171 | −0.0109 | −0.0089 | −0.0190 | +0.0067 |
| J03WOH | IDSSE | 0.7199 | 0.7173 | −0.0025 | +0.0023 | −0.0006 | −0.0092 | −0.0160 | +0.0134 |
| J03WOY | IDSSE | 0.6942 | 0.6979 | +0.0037 | +0.0111 | +0.0025 | −0.0026 | −0.0062 | +0.0099 |
| J03WPY | IDSSE | 0.7589 | 0.7227 | **−0.0362** | −0.0325 | −0.0388 | −0.0371 | −0.0449 | +0.0087 |
| J03WQQ | IDSSE | 0.6939 | 0.6940 | +0.0001 | −0.0026 | −0.0055 | +0.0086 | +0.0045 | −0.0043 |
| J03WR9 | IDSSE | 0.7128 | 0.7143 | +0.0015 | +0.0054 | −0.0003 | −0.0006 | −0.0004 | +0.0019 |
| metrica1 | Metrica | 0.8346 | 0.8367 | +0.0021 | +0.0037 | +0.0013 | +0.0014 | −0.0160 | +0.0181 |
| metrica2 | Metrica | 0.7965 | 0.7954 | −0.0011 | −0.0007 | +0.0046 | −0.0073 | +0.0013 | −0.0024 |
| metrica3 | Metrica | 0.8233 | 0.7979 | **−0.0254** | −0.0283 | −0.0227 | −0.0252 | −0.0173 | −0.0081 |

Two matches carry most of the loss (J03WPY −0.036, metrica3 −0.025); J03WPY was `netsc`'s worst too (the same seed
and the same batches, so their noise is shared). The other eight sit within −0.012 to +0.004. A single match moves
by up to ±0.02 with `gru_b`'s seed alone ([network.md](network.md)), but the bar reads the mean, and the mean fails.

SkillCorner's odd 10 (never trained on), per match:

| | 1886347 | 1925299 | 1953632 | 1986691 | 1996436 | 2006363 | 2007721 | 2011166 | 2015213 | 2017461 |
|---|---|---|---|---|---|---|---|---|---|---|
| B as built | 0.4283 | 0.4006 | 0.3914 | 0.4128 | 0.3847 | 0.4236 | 0.4512 | 0.3793 | 0.4237 | 0.3356 |
| v15's B | 0.4102 | 0.4108 | 0.3888 | 0.4436 | 0.3740 | 0.4243 | 0.4532 | 0.3799 | 0.3972 | 0.3102 |
| `netsc` | 0.4597 | 0.4320 | 0.4323 | 0.4924 | 0.4407 | 0.4688 | 0.5376 | 0.4416 | 0.4933 | 0.3866 |
| **`netsc2`** | 0.4225 | 0.4333 | 0.3964 | 0.4576 | 0.3897 | 0.4369 | 0.4948 | 0.4012 | 0.4079 | 0.3318 |
| `netsc2` − v15 | +0.0123 | +0.0226 | +0.0076 | +0.0139 | +0.0158 | +0.0126 | +0.0415 | +0.0213 | +0.0107 | +0.0216 |

The +0.018 is 4.5 times what `gru_b`'s seed alone moves the odd 10 (−0.0040; [netsc.md](netsc.md), `ens1`), so it is
real. The even 10 (in sample): +0.001 to +0.030 a match, 10/10.

### `netsc2` against `netsc` (the same segments, seed and batches; only the loss on SkillCorner's live frames differs)

| | `netsc` − v15 | `netsc2` − v15 | `netsc2` − `netsc` |
|---|---|---|---|
| B's bench | −0.0132 (2/10) | −0.0074 (4/10) | **+0.0058** (7/10) |
| p25 / p50 / stressed | −0.0101 / −0.0140 / −0.0155 | −0.0063 / −0.0070 / −0.0089 | +0.0038 / +0.0070 / +0.0066 |
| IDSSE's 7 / Metrica's 3 | −0.0143 / −0.0107 | −0.0071 / −0.0081 | +0.0072 / +0.0025 |
| SkillCorner's odd 10 | +0.0593 (10/10) | +0.0180 (10/10) | **−0.0413** (1/10) |

Dropping SkillCorner's team from the loss gives back 44 % of `netsc`'s bench loss and costs 70 % of its SkillCorner
gain.

### Where it moves (the GRU alone, argmax; truth only to report)

`scframes.csv` (SkillCorner, deployment nets, cut-aways ungraded) and `frames.csv` (B's bench, held-out P, B's 3
variants). "Near a change": live frames within 1 s of a direct change of team in the truth.

| GRU alone | SkillCorner odd 10: `gru_b` → `netsc` → **`netsc2`** | B's bench, IDSSE 7: `gru_b` → `netsc` → **`netsc2`** | B's bench, Metrica 3: `gru_b` → `netsc` → **`netsc2`** |
|---|---|---|---|
| log loss | 0.542 → 0.345 → **0.452** | 0.174 → 0.182 → **0.175** | 0.297 → 0.316 → **0.301** |
| frames right | 0.823 → 0.862 → **0.842** | 0.936 → 0.932 → **0.934** | 0.917 → 0.910 → **0.913** |
| dead frames called dead | 0.894 → 0.947 → **0.947** | 0.981 → 0.981 → **0.981** | 0.968 → 0.974 → **0.974** |
| … dead with the ball tracked and moving (the real stoppage ball) | 0.883 → 0.941 → **0.941** (10/10 matches up) | | |
| live frames called dead | 0.043 → 0.042 → **0.041** | 0.025 → 0.029 → **0.027** | 0.075 → 0.088 → **0.086** |
| team right, live, away from changes | 0.853 → 0.891 → **0.856** | 0.958 → 0.959 → **0.958** | 0.960 → 0.960 → **0.961** |
| **team right, live, within 1 s of a change** | 0.598 → 0.599 → **0.602** | 0.744 → 0.713 → **0.739** | 0.826 → 0.796 → **0.819** |
| changes the GRU switches at within 2 s | | 0.76 → 0.72 → **0.76** | 0.87 → 0.85 → **0.86** |
| \|switch offset\|, frames | 19.2 → 17.8 → **19.1** | 10.2 → 12.0 → **10.8** | 5.7 → 6.9 → **6.0** |

(B's bench, all 10, team right within 1 s of a change, per variant: p25 0.768 → 0.740 → 0.764, p50 0.769 → 0.739 →
0.763, stressed 0.769 → 0.734 → 0.763.)

- **The recipe does what it was built for.** On SkillCorner the real moving stoppage ball is read exactly as `netsc`
  reads it (0.883 → 0.941, all 10 odd matches up), and nothing of SkillCorner's team is learned (away from changes
  0.853 → 0.856, against `netsc`'s 0.891). On the bench, about **80 % of `netsc`'s timing loss is gone**: within 1 s
  of a change the team is right 0.739 / 0.819 (IDSSE / Metrica) against `gru_b`'s 0.744 / 0.826 and `netsc`'s
  0.713 / 0.796; switches land about where `gru_b`'s do.
- **What still costs on the bench is SkillCorner's "dead".** Live play called dead on Metrica rises 0.075 → 0.086
  (`netsc` 0.088), on IDSSE 0.025 → 0.027: the 2-way loss still teaches SkillCorner's D, which is "no phase of play",
  not Metrica's or DFL's dead ball. The bench's own stoppage balls were already read well (dead called dead 0.98 /
  0.97, unchanged), so on the bench the network has only the cost of that convention to show, and a small residue at
  changes (−0.004 / −0.007).
- **Most of `netsc`'s SkillCorner gain was not the stoppage ball.** `netsc2` reads SkillCorner's stoppage balls as
  well as `netsc` does but keeps only +0.018 of `netsc`'s +0.059: the rest came from learning SkillCorner's team on
  its own broadcast tracking (away from changes 0.853 → 0.891), which a bench that follows Metrica's and DFL's
  convention can't reward. The moving stoppage ball is worth about +0.014 over B as built on SkillCorner.

### The switch's table (information): down, half as much as `netsc`

The stoppage switch sends none of the 10 bench matches to B on `normal` and `p10`, so this only says how B would do
there: trial 1015 with `netsc2`'s P, **normal 0.7194 (v15's B 0.7233, −0.0039), p10 0.7210 (0.7256, −0.0046)**
(`netsc`: −0.0091, −0.0098).

### What was checked that nothing else moved

- Trial 0 (B as built, no GRU) equals v15's rows to 0.0 on B's bench and on SkillCorner's 20.
- The mirror started empty; trial 1015's tables were computed there (the code refuses to reuse them). No dead-ball
  feature had to be computed into the mirror. The mirror holds 106 MB. `netsc`'s P and nets were read, never written.
- The SkillCorner odd 10 were never read in training (`lab_netsc.sc_segments` refuses them).

## What Jose decides

Nothing to adopt: `netsc2` fails its own bar on B's bench (−0.0074, 2.1 times the seed noise, every variant and both
sources), though it passes the SkillCorner part (+0.018, 10/10). Two things are his call, only if he wants to take the
idea further (not run tonight):

1. **Whether the stoppage-ball gain is worth a bench loss.** Reading real moving stoppage balls is now isolated: about
   +0.018 on SkillCorner's odd 10 (10/10) for about −0.007 on B's bench. The bar was written so that the bench keeps
   the last word (our match is Metrica-tracked, the answer key follows Metrica's convention), and the diagnosis puts
   the remaining loss on SkillCorner's own definition of dead ("no phase of play"), which calls more of Metrica's live
   play dead. Accepting that trade would be a new kind of bar.
2. **A narrower recipe still** (a new candidate, needing its own bar before any score; not measured here, so a guess):
   the same 2-way loss, but SkillCorner's D frames counted **only where the ball is tracked and moving** (the real
   stoppage ball) and their other D frames left out of the loss, live frames kept, so the network learns the moving
   stoppage ball without the rest of SkillCorner's "no phase of play". Whether the extra live-called-dead comes from
   those other D frames isn't shown here. The plumbing is here (`lab_netsc2.recode` and `loss_of`); 8 folds, ~1 h
   45 min with two slots. Its ceiling on SkillCorner is likely the +0.014 to +0.018 seen here, and the bench's own
   stoppage balls are already read 0.97-0.98 dead, so the room is small.

SkillCorner open data, MIT; data by SkillCorner.
