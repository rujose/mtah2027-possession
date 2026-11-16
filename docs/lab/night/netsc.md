# Night, builder "netsc": SkillCorner's even 10 in `gru_b`'s training

| | |
|---|---|
| **status** | **fails the robustness bar** (2026-11-15 03:40): B's bench **−0.0132** (0.7197 against v15's 0.7329), better on 2/10, p25 −0.0101, p50 −0.0140, stressed −0.0155, IDSSE −0.0143, Metrica −0.0107 (part 1 fails on all three clauses); SkillCorner's untouched odd 10 **+0.0593**, better on **10/10** (part 2 passes, six times its +0.010; `gru_b`'s own seed moves them −0.0040). The round-8 B bar fails too. The network does learn real stoppage balls, but it also learns SkillCorner's timing of a change of team, which costs every bench match that follows DFL's or Metrica's. Nothing adopted. Plan written 01:40, before any score |
| **question** | Branch B is 92 % one network (`gru_b`), which never saw a real moving stoppage ball: only SkillCorner's balls pasted into DFL stoppages. If SkillCorner's 10 even-indexed matches (real broadcast tracking, real stoppage balls, SkillCorner's own labels) join every fold's training, does B read real stoppage balls better (SkillCorner's untouched odd 10) without losing B's bench? |
| **proposal** | [ideas_outside.md](ideas_outside.md), idea 4 (its guess: B's bench −0.002, −0.008 to +0.004; SkillCorner's odd 10 +0.012; ~25 % to pass the robustness bar) |
| **code** | `bench/lab_netsc.py` (new file). It imports the network builder's verified mirror, `bench/lab_netmirror.py`, and wraps it; no existing file is edited |
| **outputs** | `data/external/lab_netsc/s0/` (nets, the held-out P, trial 1015's stage tables recomputed there; started empty), `bench/results/lab_netsc/s0/` (`judge.json`, `confirm_1015.csv`, `confirm_1015_normal_p10.csv`, `skillcorner_1015.csv`, `frames.csv`, `scframes.csv`, `check.json`); `ens1/` (`netens` seed 1 on SkillCorner, the seed yardstick). No second seed of `netsc` was trained (below). Logs in the scratchpad's `logs/netsc/` |
| **our match** | never read, nothing run on it |
| **licence** | SkillCorner open data ([github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata)), **MIT**: training is allowed; "we kindly ask that you credit SkillCorner". Data by SkillCorner. Thanks to SkillCorner for publishing it. If this is ever adopted, the credit goes into the README and the model page |

## What changes, exactly

Only the network behind B's GRU P. Everything else is trial 1015 as v15 deploys it, its settings fixed (w_gru 0.915,
f_dead 0.909, switch prices 2.82 / 48.7, minimum 7, snap and move off, short 0.340, segments 0.910 / 0.661, timing
0.450, post stages trained on the training-stress copies). No setting is searched again: they were picked on `gru_b`'s
own P, which slightly handicaps the new network.

| | `gru_b` (v15) | `netsc` |
|---|---|---|
| training matches | the fold's DFL matches (6, or all 7 for the `metrica1` fold), each as it is and under `augment`'s training stress, both orientations | **the same, plus SkillCorner's 10 even-indexed matches** (`lab_carried.sc_pool('train')`: 1874553, 1899585, 1927964, 1959846, 1996435, 2006229, 2007448, 2010085, 2013725, 2016236) |
| SkillCorner's inputs | | `lab_carried.sc_prepared(m)`'s features (robust's 97 blanked features, the arrays the SkillCorner check already feeds the GRU), **as they are** (no stressed copy: their stoppages are already real), each half in both orientations (H and A exchanged on the swapped one) |
| SkillCorner's labels | | SkillCorner's own, as `bench/skillcorner.py` reads them (H / A from `possession.group`, D where no phase of play covers the frame). **Cut-away frames are left out of the loss** (the loader's `no_data`: no tracking at all on the nearest 10 fps frame; ~26 % of frames, 96 % of them dead): our match has none. The network still reads through them |
| normaliser and missing flags | the fold's DFL matches (`lab_temporal.normaliser`) | **the same** (SkillCorner isn't in it), so the inputs read as `gru_b`'s |
| seed, epochs, chunks, optimiser | seed 0, 20 epochs, 60 s chunks at 5 Hz, AdamW, cosine | **the same** (each epoch has more chunks: the 10 matches add about 80 % to a fold's data) |
| folds | 7 DFL folds + `metrica1` (all 7 DFL; Metrica's 3 share it) | **the same 8** |
| SkillCorner's deployment network | `gru_b`'s `metrica1` net (all 7 DFL) | **this recipe's `metrica1` net** (all 7 DFL + the even 10), as v15 does it |

How it's built: `lab_netmirror.fit` (checked by the network builder to give `lab_branch_b.fit`'s weights to the bit)
with `lab_branch_b.stressed_segments` wrapped for the length of the call, so that after the last training match's
stressed copy the even 10's segments are appended (sorted ids; per match: as is, then swapped; per half). With no
SkillCorner match the wrapper must give `lab_netmirror.fit`'s weights to the bit (checked before training).

**SkillCorner's odd 10 stay untouched**: nothing trains or tunes on them, and they are the source of the bench's test
variants' stoppage balls (`p25`, `p50`, `stressed`), so nothing leaks into B's bench either. The even 10 already give
`augment`'s training stress its balls; now the network also sees the matches themselves.

**SkillCorner's label convention differs** (from [../skillcorner.md](../skillcorner.md) and the idea page): at a true
change of team the new team's nearest player is **2.1 m** from the ball (DFL 0.6 m, Metrica 0.0 m; within 1.5 m only
33 % of the time against 97 % on DFL), so its change moments are blurred against a touch; and **D means "no phase of
play"**. The network may learn that timing, which is not Metrica's convention (our match's). That's the risk the
bench's part of the bar guards.

## The robustness bar (copied verbatim from [ideas_outside.md](ideas_outside.md), fixed there before any score)

**Bar: a robustness bar, not the B bar** (fixed here; it needs Jose to accept that kind of bar, as round 7 did for
`carried`): (1) B's bench **no loss beyond 0.002** on the mean, no variant down more than 0.003, neither source down more
than 0.005; (2) **SkillCorner's odd 10: +0.010 or more, better on 7 of 10**; the even 10 reported as in sample, never
judged.

How it's read here (nothing added to it):

- **Against v15's B as confirmed**: trial 1015 in `bench/results/search_b/confirm_1015_554_586_1133_1269_-1_-2.csv`
  (B's bench, each match's mean of p25, p50, stressed: **0.7329**; p25 0.7289, p50 0.7319, stressed 0.7379) and
  `skillcorner_1015_554_586_1133_1269.csv`, per match.
- Part (1): mean over the 10 matches ≥ **0.7309**; each of p25, p50, stressed ≥ its v15 value − 0.003; IDSSE's 7 and
  Metrica's 3, each on its own mean, ≥ v15's − 0.005.
- Part (2): **v15's B on SkillCorner's odd 10 is 0.3992** (below), so the odd 10 must reach **0.4092 or more** on the
  mean and beat v15's B on **7 or more** of the 10 matches. The deployment models score them, as `search_b
  skillcorner` does (augment's deployment trees, this recipe's `metrica1` net, trial 1015's stage models on all 10
  bench matches in the mirror).

**The round-8 B bar, for information only** (from [../night.md](../night.md)): B's bench +0.005 or more (0.7379),
better on 7 of 10, no variant down more than 0.002, neither source down on the mean, SkillCorner down no more than
0.005 (on the 20: 0.4035; the even 10 are in sample here, so the odd 10 are shown beside it).

**Seed noise.** A single network moves with its seed. The network builder measures it: s = |B's bench of `netens`
seed 1 − v15's B| ([network.md](network.md)). The night's noise rule, applied here: **if s ≥ 0.003, a pass counts
only when the same recipe with seed 1 also passes both parts**; until then it is "pending the noise rule". Declared
now: **seed 1 of this recipe is trained whatever seed 0 gives, if seed 0's trainings end by 05:15** (so seed 1 can be
judged by ~08:15). Seed 1 is also compared, for information, with `netens` seed 1 (the same seed without
SkillCorner): the cleanest reading of what the 10 matches add.

**02:20: the network builder measured s = 0.0035** (`netens` seed 1: B's bench 0.7294, −0.0035, better on 4/10; a
single match moves by up to ±0.02 with the seed alone). **So the noise rule applies here: a seed-0 pass counts only if
seed 1 of this recipe passes both parts too.** SkillCorner wasn't scored for `netens` seed 1, and part (2) asks
+0.010 of 10 matches, so its seed noise matters as much. Added now, before any `netsc` score, for information only:
`netens` seed 1's P cloned (read only) into `data/external/lab_netsc/ens1/`, trial 1015's tables recomputed there, and
SkillCorner's 20 scored with `netens` seed 1's `metrica1` net: **how far SkillCorner's odd 10 move with `gru_b`'s seed
alone**, the yardstick for part (2).

### The baseline: v15's B on SkillCorner, odd and even apart

From the existing per-match results (`bench/results/search_b/skillcorner_1015_554_586_1133_1269.csv`, which the
network builder's mirror reproduced to 0.0); trial 0 is B as built (`augment` alone):

| SkillCorner | trial 0 (B as built) | **v15's B (trial 1015)** | 1015 − 0 | better |
|---|---|---|---|---|
| **odd 10** (the guard; never trained on) | 0.4031 | **0.3992** | −0.0039 | 5/10 |
| even 10 (in sample for `netsc`) | 0.4123 | 0.4179 | +0.0055 | 6/10 |
| all 20 | 0.4077 | 0.4085 | +0.0008 | 11/20 |

v15's B is already **behind B as built on the odd 10** (−0.0039): the GRU's bench gain doesn't carry to real stoppage
balls there. Part (2) asks `netsc` for +0.010 over v15's B on those 10, i.e. about +0.006 over B as built.

| odd 10 | 1886347 | 1925299 | 1953632 | 1986691 | 1996436 | 2006363 | 2007721 | 2011166 | 2015213 | 2017461 |
|---|---|---|---|---|---|---|---|---|---|---|
| v15's B | 0.4102 | 0.4108 | 0.3888 | 0.4436 | 0.3740 | 0.4243 | 0.4532 | 0.3799 | 0.3972 | 0.3102 |

## The run plan

Every heavy job goes through `slot.py` (one slot, 2 threads), two at a time at most (the network builder holds the
rest); logs in the scratchpad's `logs/netsc/`. `df -h /private/tmp` is checked before each big write; under 4 GB free,
everything stops and the reason goes here.

| step | jobs | cost |
|---|---|---|
| checks | (a) the wrapper with no SkillCorner match gives `lab_netmirror.fit`'s weights to the bit (1 epoch, 2 matches); (b) SkillCorner's segments: frames, share left out of the loss, label shares, the standardised inputs' range, on one fold's normaliser | ~5 min |
| seed 0 training | 8 folds, each followed by its held-out P (5 bench variants + the training-stress copy) into the mirror | 8 × ~16 min |
| B rebuilt | trial 1015's stage tables in the mirror (never reused), then `confirm` (trials 0 and 1015, B's 3 variants; trial 0 must equal v15's rows, which it reads from the shared cache) and SkillCorner's 20 with the `metrica1` net | ~10 + ~25 min |
| judge | the robustness bar; the B bar (information); per source, per variant, per match; SkillCorner odd and even apart | |
| information | the GRU's own held-out log loss per variant; on SkillCorner, its frame log loss on the odd and even 10 against `gru_b`'s deployment net; the switch's table (normal, p10) if slots allow | |
| seed 1 | the same, seed 1 (see the noise rule) | as seed 0 |

## Results

### The checks (01:37): pass, no score involved

- **The wrapper adds nothing when empty**: `lab_netsc.fit` with no SkillCorner match against `lab_netmirror.fit`, 1
  epoch on J03WN1 + J03WOH: max |difference| over the weights **0.0**, the same loss (0.4182). So `netsc` differs from
  `gru_b` only by the 10 matches' segments.
- **SkillCorner's segments** (the even 10, `bench/results/lab_netsc/s0/check.json`):

| | per match (range over the 10) |
|---|---|
| frames in the halves | 142 k – 151 k |
| left out of the loss (cut-aways) | 21 % – 33 % |
| graded frames left | 102 k – 114 k (1.10 M over the 10) |
| of those: H / A / D | 0.29–0.43 / 0.28–0.44 / 0.18–0.35 (D is low: 96 % of the cut-aways were dead) |
| ball missing on graded frames | 0.1 % |

On metrica1's normaliser (the 7 DFL matches) the first match's standardised inputs sit near 0 (mean |feature mean|
0.11) with a median sd of 1.01, and 0.02 % of values reach the ±5 clip: the DFL normaliser reads SkillCorner's
features without trouble. Four segments per match (2 halves × 2 orientations); H in the first orientation equals A in
the swapped one.

The first fold (`metrica1`, 24 training copies against `gru_b`'s 14) runs at ~53 s an epoch on the shared machine:
~18 min a fold.

### Seed 0 (03:26): fails the robustness bar on part (1); part (2) passes six times over

8 folds trained 01:37–03:09 (17–18 min each; final training loss 0.199–0.206 against `gru_b`'s ~0.125: SkillCorner's
labels are harder to fit). Trial 1015's stage tables recomputed in the mirror from this recipe's P (46 s), then
`confirm` and SkillCorner. **Trial 0 reproduces v15's rows exactly** (max |difference| 0.0 on B's bench and on
SkillCorner's 20), so the mirror is sound and every difference below is the network's.

| | v15's B | `netsc` | difference | bar | |
|---|---|---|---|---|---|
| **B's bench** (mean of p25, p50, stressed) | 0.7329 | **0.7197** | **−0.0132**, better on 2/10 | ≥ −0.002 | **fails** |
| p25 / p50 / stressed | 0.7289 / 0.7319 / 0.7379 | 0.7188 / 0.7178 / 0.7224 | −0.0101 / −0.0140 / −0.0155 | each ≥ −0.003 | **fails** (all three) |
| IDSSE's 7 / Metrica's 3 | | | −0.0143 / −0.0107 | each ≥ −0.005 | **fails** (both) |
| **SkillCorner's odd 10** (the guard) | 0.3992 | **0.4585** | **+0.0593**, better on **10/10** | ≥ +0.010, 7/10 | passes |
| SkillCorner's even 10 (in sample, not judged) | 0.4179 | 0.4941 | +0.0763, better on 10/10 | | |
| SkillCorner's 20 | 0.4085 | 0.4763 | +0.0678, better on 20/20 | | |

**The robustness bar: fails** (part 1 fails on every one of its three clauses; part 2 passes). **The round-8 B bar
(information): fails** (−0.0132 against +0.005; SkillCorner's part passes). The loss on B's bench is **3.8 times the
seed noise** (s = 0.0035), on all three variants and both sources, so no second seed can turn it into a pass (the
noise rule only adds conditions to a pass). Against B as built (trial 0, no GRU), `netsc`'s B still gains +0.0222 on
B's bench (v15's B: +0.0353): the 10 matches cost about 37 % of the GRU's bench gain.

Per match on B's bench (v15's B → `netsc`; p25 / p50 / stressed):

| match | source | v15's B | `netsc` | difference | p25 | p50 | stressed |
|---|---|---|---|---|---|---|---|
| J03WMX | IDSSE | 0.6254 | 0.6074 | −0.0180 | −0.0167 | −0.0172 | −0.0201 |
| J03WN1 | IDSSE | 0.6691 | 0.6501 | −0.0190 | −0.0234 | −0.0209 | −0.0128 |
| J03WOH | IDSSE | 0.7199 | 0.7039 | −0.0160 | −0.0106 | −0.0174 | −0.0199 |
| J03WOY | IDSSE | 0.6942 | 0.6880 | −0.0062 | +0.0053 | −0.0074 | −0.0165 |
| J03WPY | IDSSE | 0.7589 | 0.7140 | **−0.0449** | −0.0393 | −0.0517 | −0.0437 |
| J03WQQ | IDSSE | 0.6939 | 0.6984 | +0.0045 | +0.0066 | +0.0041 | +0.0027 |
| J03WR9 | IDSSE | 0.7128 | 0.7124 | −0.0004 | +0.0070 | −0.0064 | −0.0019 |
| metrica1 | Metrica | 0.8346 | 0.8186 | −0.0160 | −0.0162 | −0.0151 | −0.0166 |
| metrica2 | Metrica | 0.7965 | 0.7978 | +0.0013 | +0.0026 | +0.0072 | −0.0060 |
| metrica3 | Metrica | 0.8233 | 0.8060 | −0.0173 | −0.0164 | −0.0157 | −0.0197 |

SkillCorner's odd 10 (never trained on), per match:

| | 1886347 | 1925299 | 1953632 | 1986691 | 1996436 | 2006363 | 2007721 | 2011166 | 2015213 | 2017461 |
|---|---|---|---|---|---|---|---|---|---|---|
| B as built | 0.4283 | 0.4006 | 0.3914 | 0.4128 | 0.3847 | 0.4236 | 0.4512 | 0.3793 | 0.4237 | 0.3356 |
| v15's B | 0.4102 | 0.4108 | 0.3888 | 0.4436 | 0.3740 | 0.4243 | 0.4532 | 0.3799 | 0.3972 | 0.3102 |
| `netsc` | 0.4597 | 0.4320 | 0.4323 | 0.4924 | 0.4407 | 0.4688 | 0.5376 | 0.4416 | 0.4933 | 0.3866 |
| **difference** | +0.0495 | +0.0212 | +0.0435 | +0.0488 | +0.0667 | +0.0445 | +0.0844 | +0.0617 | +0.0961 | +0.0764 |

The even 10 (in sample): +0.023 to +0.104 a match, 10/10.

**Seed 1 is not trained (a change to the plan, 03:30, after seeing seed 0).** It was declared above "whatever seed 0
gives", for the noise rule. But the noise rule can only add a condition to a pass, and seed 0 fails part (1) by 3.8
times s on every clause, so seed 1 can't change the verdict; its 8 folds (~2.4 slot-hours) would only slow the
network builder's own noise-rule runs (`netmet` seed 1), which can. `netens` seed 1 on SkillCorner (`ens1`, below)
still runs: it is the yardstick for how much of the +0.059 a seed could explain.

### How far a seed moves SkillCorner (`ens1`, 03:32): about a tenth of the gain

`netens` seed 1 (`gru_b`'s recipe, seed 1, no SkillCorner in training) through the same SkillCorner run (its P cloned
read only, trial 1015's tables recomputed in `ens1/`; trial 0 again identical to v15's rows):

| SkillCorner | v15's B (seed 0) | `gru_b` seed 1 | seed 1 − seed 0 | `netsc` − seed 1 |
|---|---|---|---|---|
| odd 10 | 0.3992 | 0.3952 | −0.0040, better on 3/10 (per match: mean \|diff\| 0.006, largest 0.022) | +0.0632, 10/10 |
| even 10 | 0.4179 | 0.4122 | −0.0057, 4/10 | +0.0819, 10/10 |

So `netsc`'s +0.059 on the odd 10 is about 15 times what `gru_b`'s seed alone does there: **the SkillCorner gain is
real**, as the bench loss is.

### Where the gain and the loss come from (the GRU alone, argmax; truth only to report)

`scframes.csv` (SkillCorner, deployment nets, cut-aways ungraded) and `frames.csv` (B's bench, held-out P, B's 3
variants). "Near a change": live frames within 1 s of a direct change of team (H to A or A to H) in the truth;
"switch offset": the GRU's own switch to the new team nearest the truth's, within 2 s.

| GRU alone | SkillCorner odd 10: `gru_b` → `netsc` | B's bench, IDSSE 7: `gru_b` → `netsc` | B's bench, Metrica 3: `gru_b` → `netsc` |
|---|---|---|---|
| log loss | 0.542 → **0.345** | 0.174 → 0.182 | 0.297 → 0.316 |
| frames right | 0.823 → **0.862** | 0.936 → 0.932 | 0.917 → 0.910 |
| dead frames called dead | 0.894 → **0.947** | 0.981 → 0.981 | 0.968 → 0.974 |
| … dead with the ball tracked and moving (the real stoppage ball; 27 % of graded frames) | 0.883 → **0.941** (10/10 matches up) | | |
| live frames called dead | 0.043 → 0.042 | 0.025 → 0.029 | 0.075 → **0.088** |
| team right, live, away from changes | 0.853 → **0.891** (10/10) | 0.958 → 0.959 | 0.960 → 0.960 |
| **team right, live, within 1 s of a change** | 0.598 → 0.599 | 0.744 → **0.713** | 0.826 → **0.796** |
| changes the GRU switches at within 2 s | 0.45 → 0.37 | 0.76 → 0.72 | 0.87 → 0.85 |
| \|switch offset\|, frames | 19.2 → 17.8 | 10.2 → **12.0** | 5.7 → **6.9** |

- **The idea's mechanism works.** On SkillCorner's own matches the network now reads a real moving stoppage ball as
  dead far more often (0.88 → 0.94, every one of the odd 10), and reads the team better away from changes (0.85 →
  0.89: most likely SkillCorner's coarse broadcast positions, learned). That is the +0.059.
- **The cost is the timing of a change.** On B's bench the network loses 3 points of team accuracy within 1 s of a
  true change (all three variants: −0.028 / −0.030 / −0.034), switches farther from the truth's change (+1.2 to +1.8
  frames on average) and calls a little more live play dead on Metrica (+1.3 points). Away from changes nothing
  moves. This is the label convention the idea page warned about: SkillCorner's change of team doesn't sit on a touch
  (the new team's player is 2.1 m from the ball when its label changes), and its D is "no phase of play". On
  SkillCorner itself the network gains nothing at changes (0.598 → 0.599): that timing isn't learned from its
  tracking, so the network only gets blurrier there.
- **The bench's stoppage balls were already read well** (dead called dead 0.98 / 0.97 before and after): the pasted
  balls of `augment`'s stress had taught `gru_b` what SkillCorner's real ones teach `netsc`, as far as the bench can
  show. Where B runs on our match (a Metrica-tracked match, Metrica's convention), the timing is what counts.

### The switch's table (information): down as well

The stoppage switch sends none of the 10 bench matches to B on `normal` and `p10`, so this only says how B would do
there: trial 1015 with `netsc`'s P, **normal 0.7142 (v15's B 0.7233, −0.0091), p10 0.7157 (0.7256, −0.0098)**.

### What was checked that nothing else moved

- Trial 0 (B as built, no GRU) equals v15's rows to 0.0 on B's bench and on SkillCorner's 20, in both mirrors.
- `find -newer` on `data/external` and `bench/results` since 01:37: only the night builders' own folders
  (`lab_netsc`, and the network builder's `lab_netens`, `lab_netmet`, `lab_netavg`). No shared cache was written;
  no dead-ball feature had to be computed into the mirror. The mirror holds 209 MB.

## What Jose decides

Nothing to adopt: `netsc` fails its own bar on B's bench, by 3.8 times the seed noise, on every variant and both
sources. Two things are his call, only if he wants to take the idea further (not run tonight):

1. **Whether B's bench should keep the last word over SkillCorner.** SkillCorner's odd 10 say +0.059 (10/10) and the
   bench says −0.013 (2/10). The bar was written so that the bench wins (our match is Metrica-tracked and its labels
   follow Metrica's convention, which the bench's DFL and Metrica matches share and SkillCorner doesn't), and the
   diagnosis above says the loss sits exactly at the changes of team, where conventions differ. Accepting SkillCorner
   as the judge instead would be a new kind of bar, and it would favour SkillCorner's convention.
2. **A narrower recipe, if he wants the stoppage balls without the timing** (a new candidate, needing its own bar
   before any score): SkillCorner's even 10 in training **for dead or live only**: on their frames a two-way loss
   (P(H) + P(A) against P(D)) instead of the three-way one, so the network learns the real stoppage ball and nothing
   of SkillCorner's team or its change timing (simply ignoring their H / A labels would not do: the network would
   only ever see D on them). A lighter cousin: keep the three-way loss but ignore SkillCorner's frames within ~1 s of
   its changes of team. Same plumbing (`lab_netsc.sc_segments` and a small change to the loss), 8 folds, ~2.5 h with
   two slots.

SkillCorner open data, MIT; data by SkillCorner.
