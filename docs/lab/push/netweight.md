# Push, track 2 "netweight": branch B's network leaning harder on Metrica's games

| | |
|---|---|
| **status** | **finished 03:40, 2026-11-16. The finalist W4 (Metrica's training matches at loss weight 4; seeds 0-3 averaged) passes the bar on both parts: against v15's B (Metrica's rule) and against `netmet5`.** Its four deployment nets are trained. Under v17's broadcast merge it fails track 3's pessimistic floor and ties `netmet5` on B's bench (04:15); **not adopted** (Jose, 08:45: "Keep v17") |
| **question** | Does branch B's GRU gain more on Metrica's games if its training leans harder on them, beyond the night's `netmet5` (netmet's recipe, seeds 0-3 averaged)? |
| **brief** | [../push.md](../push.md), track 2. Builds on [../night/network.md](../night/network.md) (`netmet`, `ens2`, `netmet5` = `ens4`) and [../night/audit_netmet.md](../night/audit_netmet.md) (the folds' cleanliness, which this track keeps) |
| **code** | `bench/lab_netweight.py` (new; imports `lab_netmet`, `lab_netmirror`, `lab_branch_b`, `lab_carried`; no existing file edited) |
| **outputs** | `data/external/lab_netweight/` (nets, P mirrors, D3's extra stress draws); `bench/results/lab_netweight/` |
| **our match** | never read, nothing run on it |

## The candidates (fixed 10:28, before any score)

Each is **netmet's recipe exactly** (`lab_netmirror.fit` as `lab_netmet.train` calls it) with one change. What stays
netmet's: the folds (leave one match out over all 10: a DFL fold trains on the other 6 DFL + Metrica's 3, a Metrica
fold on the 7 DFL + the other 2 Metrica; `all` on the 10); a held-out match never enters its own fold's training, in
any copy, draw or weight; the normaliser and missing flags (the fold's DFL matches, `lab_netmet.norm_for`); gru_b's
network, 20 epochs, both orientations, 5 Hz, 60 s chunks, batch 16, AdamW with cosine decay, gradient clip; the
seeds (torch and numpy, as netmet); the held-out P (`lab_netmirror.predict_into`: 5 bench variants and the
training-stress copy `TRAIN_SEED`); B rebuilt in the mirror with trial 1015's settings fixed, its post stages
retrained on the mirror's training-stress P (`lab_netmirror.confirm`).

| candidate | the change |
|---|---|
| **W2** | every frame of a **Metrica training match** (as it is and its training-stress copy, both orientations) carries **loss weight 2**; DFL frames 1 |
| **W4** | the same with **weight 4** |
| **D3** | each Metrica training match under **3 independent training-stress draws** instead of netmet's 1: netmet's own copy (`lab_carried.TRAIN_SEED` = 1, read from the shared cache) plus two new draws with stress seeds **11 and 12**; loss weight 1 everywhere (as netmet). DFL matches keep their one copy |

**Loss weight, not sampling weight, and why.** netmet's loop draws a random 5 Hz phase and chunk shift per segment
per epoch, permutes the chunks and takes the mean cross-entropy over a batch's labelled frames. A loss weight changes
one line: each frame's cross-entropy is multiplied by its match's weight and the batch loss is the **weighted mean**
(Σ w·l / Σ w over the labelled frames; in expectation the same as drawing Metrica's frames W times as often). Every
other draw stays netmet's: the same chunks, the same batch order, the same initial weights and dropout masks, the same
number of steps. So W2 and W4 with seed S are **paired** with netmet's seed S (only the loss differs), and they cost
what netmet costs. Sampling weight would add segments, change every random draw after the first Metrica match (a new
seed in effect) and make W4 twice as slow. When every weight is 1 the code takes netmet's own loss line (so the
reproduction below is netmet's loop to the bit).

**D3's draws.** netmet's training-stress copy of a match is `lab_carried.stressed_features(m, "all", TRAIN_SEED)`:
the lab cache's damaged tracking (the 41 % damage is the cache's and fixed; it is the same in every copy) with the
ball in every true stoppage replaced by a SkillCorner stoppage's ball (`lab_carried.stress_ball`: one of the 5
same-kind templates nearest in length, drawn by `default_rng([seed, crc32(match)])`), costs and robust's 97 features
recomputed. D3's new draws run the same code with seeds 11 and 12 and **the training pool only** (SkillCorner's
even-indexed 10, `templates("train")`, as `TRAIN_SEED`; the test variants' odd-indexed pool is never touched). They
are written to `data/external/lab_netweight/draws/`, never to the shared cache. A Metrica match then trains as it is
plus 3 stressed copies (4 copies; netmet 2): D3 gives Metrica's training matches twice netmet's frames, the same
total as W2, but as new stress draws rather than repeated weight. Checked before use: the same code with seed 1 must
give the shared `features_1_<m>.npz` exactly.

**The combination (rule fixed now, before the screen).** Let W\* be the better of W2 and W4 on Metrica's 3 mean (their
2-seed averages). **If W\* and D3 both beat `ens2` on Metrica's 3 mean, W\*+D3 is built** (W\*'s loss weight on every
copy of a Metrica training match, D3's three draws included), seeds 0 and 1, judged as the others, and it joins the
screen as a fourth candidate. Otherwise it is not built. If it can't start by 21:00 (the clock), it is dropped.

**The screen.** Each candidate with seeds 0 and 1 (10 folds each); B's GRU P = the mean of the two seeds' P exactly as
`ens2` averages (`lab_netmirror.average`: each through float32, mean, clipped to 0.01-0.99, rounded to 3 decimals), on
every bench variant and the training-stress copies; B rebuilt in the mirror; judged on B's bench (30 units) against
**`ens2`** (netmet's seeds 0+1, 2 seeds against 2) and against v15's B.

**The finalist.** The screen's best on **Metrica's 3 mean** (all candidates built), **if it is strictly above `ens2`'s
Metrica's 3 mean**, gets seeds 2 and 3 → the mean of seeds 0-3, judged against **`netmet5` (`ens4`)** and v15's B,
plus SkillCorner's 20 with its four `all` nets (seeds 0-3) averaged as `netmet5`'s were. If no candidate beats `ens2`
on Metrica's 3, there is no finalist and `netmet5` stays.

## The bar for replacing `netmet5` as v16's network (fixed before any score; the brief's words)

> **(a)** the Metrica rule against v15's B — Metrica's 3 +0.010 or more on the mean of their 3 variants and better on
> 3/3; DFL's 7 no loss beyond 0.005; no variant down more than 0.002; SkillCorner no loss beyond 0.005;
> **(b)** against netmet5 — Metrica's 3 mean +0.005 or more; DFL's 7 no lower than netmet5's by more than 0.003; B's
> bench mean no lower than netmet5's.
> Report the full B bar (+0.005, 7/10, …) too, as information.

How the parts are read here: every number is on B's bench (each match's mean of p25, p50, stressed), trial 1015's
settings, the reference rows v15's B as confirmed (`search_b/confirm_1015_554_586_1133_1269_-1_-2.csv`, 0.7329;
SkillCorner `skillcorner_1015_554_586_1133_1269.csv`, 0.4085) and `netmet5` as judged
(`bench/results/lab_netmet/ens4/`: B's bench 0.7468, Metrica's 3 +0.0400 and DFL's 7 +0.0027 over v15's B,
SkillCorner 0.4090). "Metrica's 3 mean +0.005 against `netmet5`" is the candidate's Metrica's 3 mean minus
`netmet5`'s; "DFL's 7 no lower by more than 0.003" the same on DFL's 7. The finalist is a 4-seed mean, so the noise
rule's single-network clause doesn't apply.

**Information, never judged:** the full B bar against v15's B (+0.005, 7 of 10, variants, sources, SkillCorner);
each candidate's seeds' held-out log loss; the finalist's **fresh seeds 2+3 averaged against netmet's seeds 2+3
averaged** (neither pair was used to pick the finalist, so this comparison carries no winner's curse from the screen);
SkillCorner split into its even 10 (whose stoppage balls are in every training stress) and odd 10 (untouched).

## The reproduction (before any candidate trains)

netmet seed 0 through this module's code path, fold `metrica1`: the net's weights and its held-out P (5 variants and
the training-stress copy) must equal `data/external/lab_netmet/main/`'s to the bit. Then a 1-epoch check on 2 matches
that the weighted loss line with every weight 1 equals netmet's loss line (to float rounding: information). And D3's
draw code with seed 1 must reproduce the shared training-stress features of a Metrica match exactly.

## Run plan

Every heavy job through `slot.py` (1 slot, 2 threads), at most 2 at a time; logs in the scratchpad's
`logs/netweight/`. A fold costs ~15 min (D3's ~1.3 times that). Order: reproduction and D3's draws in parallel; then
W2 (seeds 0 ∥ 1), W4, D3, each candidate's `ens2` combined and confirmed as soon as its 20 folds are in; then the
combination (if its rule fires); then the finalist's seeds 2-3 (20 folds) and the four `all` nets.

## Log (machine clock, UTC−5)

- 10:28 page and bar written; no score of any candidate exists.
- 10:30 queue started (2 slots): the reproduction (`metrica1`, seed 0) and D3's draws first; W2, W4, D3 wait on the reproduction.
- 10:31 **D3's draws: the check passes exactly** (the draw code with seed 1 on metrica1 gives the shared training-stress
  copy: ball, X, X_swapped, truth, blanked and feature names all identical). The 6 new draws (Metrica's 3 × seeds 11,
  12; 95-97 stoppages stressed each) are written compressed to `data/external/lab_netweight/draws/` (290 MB). Each
  stoppage takes one of its 5 nearest same-kind templates, so a new draw repeats netmet's ball on part of the dead
  frames: 41-62 % of dead frames' ball differ from netmet's draw (`draws/draws.json`).
- 10:48 **The reproduction passes, to the bit**: netmet seed 0's fold `metrica1` through this module (`W1`: weight 1,
  netmet's one draw) gives netmet's net (max |difference| over all weights 0) and its held-out P on normal, p10, p25,
  p50, stressed and the training-stress copy identical (`bench/results/lab_netweight/repro/repro.json`). 1056 s of
  training (the night's ~800 s: the machine is shared by three tracks). Candidates may now train.
- 10:49 all 4 slots held by tracks 1 and 3; the queue waits for its slots.
- 10:54 the loss check (information): 1 epoch on J03WN1 + metrica1, the weighted loss line with every weight 1 against
  netmet's loss line: the epoch's loss agrees to 7e-9 and the trained weights are identical (max |difference| 0)
  (`repro/losscheck.json`). So W2 and W4 differ from netmet's same seed only by the weights.
- 11:03 W2's first folds start (both slots). ~15 min a fold.
- 11:22 the queue's courtesy pause (30 s after every job) had handed both slots to the other tracks; now it pauses
  only after a job whose sibling still runs (the track keeps one slot, yields the second). Queue restarted; nothing lost.
- 11:43 **the coordinator: track 1 (v16) is starved; at most 1 slot until it says track 1 is done (~1-2 h).** The
  two running folds (W2 J03WN1, seeds 0 and 1) finish; no new job starts until both have, then one at a time. The
  queue now reads its limit from a file (`netweight/limit`, now 1) and adopted the two running jobs. **Timing:** about
  7 folds lost to the limit; the screen's W2 and W4 should still be in by ~18:00, D3 by ~21:00-21:30 (the clock
  rule for the combination, 21:00, may drop it).
- 11:54 the coordinator: track 1 is done; back to at most 2 slots (limit file set to 2). The 1-slot spell cost about
  10 minutes of one slot.
- 14:26 **W2's screen in** (20 folds, 11:03-14:22; seeds 0+1 averaged, B rebuilt, trial 0 equal to v15's on all 30
  units): B's bench 0.7492, **+0.0164 over v15's B, +0.0045 over `ens2`**; Metrica's 3 **+0.0105 over `ens2`** (3/3);
  DFL's 7 +0.0019 over `ens2` (5/7). Table below. W4 started 14:26.
- 17:05 **W4's screen in** (20 folds, 14:26-17:01): B's bench 0.7481, +0.0152 over v15's B (6/10), **+0.0033 over
  `ens2`**; Metrica's 3 **+0.0192 over `ens2`** (3/3); DFL's 7 **−0.0034 under `ens2`** (2/7; −0.0008 under v15's
  B). So W\* (the better W on Metrica's 3) is **W4**. D3 started 17:05.
- 20:16 **D3's screen in** (20 folds, 16:56-20:13): B's bench 0.7476, +0.0148 over v15's B (8/10), **+0.0029 over
  `ens2`**; Metrica's 3 **+0.0149 over `ens2`** (3/3); DFL's 7 **−0.0023 under `ens2`** (3/7; +0.0004 over v15's B).
- 20:21 **the combination's rule fires**: W\* = W4 (+0.0192) and D3 (+0.0149) both beat `ens2` on Metrica's 3 mean,
  and it starts before 21:00. **W4D3** (loss weight 4 on every copy of a Metrica training match, D3's three draws
  included) queued with seeds 0 and 1; it joins the screen as a fourth candidate. The finalist is chosen when it is in.
- 21:57-01:18 **the machine stalled.** From ~22:00 macOS swap filled (9.8 of 10 GB, ~64 MB of RAM free); W4D3's two
  J03WQQ folds went from ~50 s to ~3,600 s an epoch (epochs 4-5 of 20 after 3 hours). The swapped memory was not
  this track's: with both folds killed and no slot held by anyone, swap stayed at 9.8 GB and the load at ~20 (macOS
  and desktop apps working through swap; no research job ran). Killed at 01:18: W4D3's two J03WQQ folds (mine, no
  other process touched). 8 of W4D3's 20 folds were trained; it was never combined or confirmed.
- 01:20 **W4D3 is dropped by the clock** (decided now; no score of W4D3 exists): its 12 remaining folds (~2 h on a
  healthy machine) and then the finalist's 24 trainings (~3.5-4 h) leave no margin before the finalist's 09:00
  deadline on a machine that just lost 3 hours. Its 8 trained fold nets stay in `data/external/lab_netweight/W4D3_s*/`
  unscored. **The finalist is chosen from the three built candidates by the rule written at 10:28: W4** (Metrica's 3
  +0.0192 over `ens2`, above W2's +0.0105 and D3's +0.0149). Queued: W4's `all` nets (seeds 0-3), its folds with
  seeds 2-3, then `final W4` (seeds 0-3 averaged, B, SkillCorner, the bar) and `fresh W4` (seeds 2+3 against
  netmet's 2+3, information).
- 01:19-03:22 W4's 24 trainings (its `all` nets for seeds 0-3, its folds with seeds 2-3), 2 slots, ~10-12 min each
  on the recovered machine.
- 03:25 `fresh W4` (information): W4's seeds 2+3 against netmet's seeds 2+3.
- 03:30 **`final W4`: the bar passes, both parts** (results below).
- 03:40 `lab_netweight.fit_recipe(..., cand="W4")`, the interface for `bench/train_v16.py`, rebuilds W4's seed-0
  deployment net to the bit (max |difference| 0; `W4_ens4/recipe_check.json`). `DEPLOY = "W4"` set in the module;
  the four nets cloned under clear names into `data/external/lab_netweight/deploy_W4/` (manifest with SHA-256).

## Results

### The screen (2 seeds each, against `ens2` and v15's B)

B's bench = each match's mean of p25, p50, stressed (trial 1015, B rebuilt in the mirror); v15's B 0.7329.

| B's GRU | B's bench | vs v15's B | better on | p25 / p50 / stressed vs v15 | DFL's 7 / Metrica's 3 vs v15 | Metrica better | vs `ens2`: bench / DFL's 7 / **Metrica's 3** (better on) |
|---|---|---|---|---|---|---|---|
| `ens2` (netmet, seeds 0+1) | 0.7448 | +0.0119 | 8/10 | +0.0102 / +0.0106 / +0.0149 | +0.0027 / +0.0334 | 3/3 | |
| **W2** (seeds 0+1) | 0.7492 | +0.0164 | 8/10 | +0.0155 / +0.0163 / +0.0172 | +0.0046 / +0.0439 | 3/3 | +0.0045 / +0.0019 / **+0.0105** (8/10) |
| **W4** (seeds 0+1) | 0.7481 | +0.0152 | 6/10 | +0.0149 / +0.0145 / +0.0163 | −0.0008 / +0.0526 | 3/3 | +0.0033 / −0.0034 / **+0.0192** (5/10) |
| **D3** (seeds 0+1) | 0.7476 | +0.0148 | 8/10 | +0.0118 / +0.0128 / +0.0197 | +0.0004 / +0.0483 | 3/3 | +0.0029 / −0.0023 / **+0.0149** (6/10) |

| per match, vs v15's B | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | J03WQQ | J03WR9 | metrica1 | metrica2 | metrica3 |
|---|---|---|---|---|---|---|---|---|---|---|
| `ens2` | +0.003 | +0.001 | −0.014 | +0.019 | −0.009 | +0.006 | +0.014 | +0.028 | +0.046 | +0.027 |
| W2 | +0.010 | +0.003 | −0.017 | +0.020 | −0.017 | +0.013 | +0.020 | +0.030 | +0.057 | +0.045 |
| W4 | +0.014 | −0.002 | −0.033 | +0.017 | −0.013 | −0.006 | +0.019 | +0.029 | +0.077 | +0.052 |
| D3 | +0.000 | +0.002 | −0.033 | +0.017 | −0.011 | +0.011 | +0.016 | +0.037 | +0.066 | +0.042 |

Against `ens2` per variant: W2 p25 +0.0053, p50 +0.0058, stressed +0.0023; W4 +0.0047, +0.0039, +0.0014; D3 +0.0016,
+0.0022, +0.0048.
Trial 0 (B as built) equals v15's on all 30 units in every mirror (max |difference| 0).

### The finalist: W4, seeds 0-3 averaged (against `netmet5` and v15's B)

B's GRU P = the mean of W4's seeds 0-3 (`lab_netmirror.average`, as `netmet5`); SkillCorner's 20 with the mean of
W4's four `all` nets, each read through the 7 DFL matches' normaliser (as `netmet5`'s). `bench/results/lab_netweight/W4_ens4/`
(`confirm_1015.csv`, `skillcorner_1015.csv`, `judge.json`, `compare_netmet5.json`, `bar.json`).

| | v15's B | `netmet5` (netmet, seeds 0-3) | **W4 (seeds 0-3)** | W4 − v15's B | W4 − `netmet5` |
|---|---|---|---|---|---|
| **B's bench** | 0.7329 | 0.7468 | **0.7493** | +0.0165 (7/10) | **+0.0026** (5/10) |
| p25 / p50 / stressed | 0.7289 / 0.7319 / 0.7379 | | | +0.0152 / +0.0158 / +0.0184 | +0.0011 / +0.0028 / +0.0037 |
| **Metrica's 3** | | 0.8582 | **0.8706** | **+0.0525** (3/3) | **+0.0125** (3/3) |
| **DFL's 7** | | 0.6990 | **0.6973** | +0.0010 (4/7) | **−0.0017** (2/7) |
| **SkillCorner (20)** | 0.4085 | 0.4090 | **0.4113** | +0.0028 (14/20) | +0.0024 (10/20) |
| SkillCorner even 10 / odd 10 | | | | +0.0015 / +0.0041 | +0.0032 / +0.0015 |

| per match | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | J03WQQ | J03WR9 | metrica1 | metrica2 | metrica3 |
|---|---|---|---|---|---|---|---|---|---|---|
| `netmet5` − v15's B | +0.009 | −0.001 | −0.012 | +0.018 | −0.012 | −0.001 | +0.018 | +0.034 | +0.052 | +0.034 |
| **W4 − v15's B** | +0.009 | −0.001 | −0.026 | +0.025 | −0.015 | +0.002 | +0.012 | +0.036 | +0.071 | +0.050 |
| **W4 − `netmet5`** | −0.000 | −0.000 | −0.014 | +0.007 | −0.002 | +0.003 | −0.005 | +0.002 | +0.019 | +0.016 |

**The bar** (fixed at 10:28):

| part | criterion | W4 | |
|---|---|---|---|
| (a) | Metrica's 3 +0.010 or more over v15's B | +0.0525 | passes |
| (a) | better on 3/3 of Metrica's | 3/3 | passes |
| (a) | DFL's 7 no loss beyond 0.005 | +0.0010 | passes |
| (a) | no variant down more than 0.002 | worst +0.0152 (p25) | passes |
| (a) | SkillCorner no loss beyond 0.005 | +0.0028 | passes |
| (b) | Metrica's 3 mean +0.005 or more over `netmet5` | +0.0125 | passes |
| (b) | DFL's 7 no lower than `netmet5`'s by more than 0.003 | −0.0017 | passes (margin 0.0013) |
| (b) | B's bench mean no lower than `netmet5`'s | +0.0026 | passes |
| **the bar** | (a) and (b) | | **passes** |

**The full B bar, information:** +0.0165 over v15's B (≥ +0.005), better on **7/10** (≥ 7: passes, where `netmet5` had 6),
every variant up, neither source down (DFL +0.0010, Metrica +0.0525), SkillCorner +0.0028: **W4 passes the full B bar
too.** The noise rule's single-network clause doesn't apply (4 seeds).

### Information: the winner's curse, and DFL's cost

W4 was picked as the best of three on Metrica's 3, on seeds 0+1. Its seeds 2+3 played no part in the choice, so
**W4's seeds 2+3 averaged against netmet's seeds 2+3 averaged** is a comparison with no selection in it
(`W4_s23/compare_netmet_s23.json`):

| W4 − netmet | seeds | B's bench | Metrica's 3 | DFL's 7 |
|---|---|---|---|---|
| the screen (selected on this) | 0+1 | +0.0033 (5/10) | +0.0192 (3/3) | −0.0034 (2/7) |
| **fresh pair (no selection)** | **2+3** | **+0.0006** (6/10) | **+0.0109** (3/3) | **−0.0038** (3/7) |
| the finalist (bar) | 0-3 | +0.0026 (5/10) | +0.0125 (3/3) | −0.0017 (2/7) |

Per match, fresh pair: J03WMX −0.018, J03WN1 −0.018, J03WOH −0.002, J03WOY +0.004, J03WPY +0.012, J03WQQ +0.004,
J03WR9 −0.008, metrica1 +0.004, metrica2 +0.016, metrica3 +0.013.

- **Metrica's gain over `netmet5` is real and survives out of selection:** +0.011 on the fresh pair (3/3), against
  the screen's +0.019. About 0.008 of the screen's number was the selection; what remains is still twice (b)'s +0.005.
- **The price is DFL:** −0.0017 on four seeds, −0.0038 on the fresh pair (that pair alone would miss (b)'s −0.003;
  it is information, not the bar). Leaning on Metrica costs the DFL matches 0.002-0.004 against `netmet5`. On the bar
  (b) passes this part with a margin of 0.0013.
- **B's bench against `netmet5`** is +0.0026 on four seeds and +0.0006 on the fresh pair: level within seed noise.
  The gain is Metrica's, paid partly by DFL.
- The same trade appears across the screen: the more weight on Metrica (W2 → W4), the more Metrica gains and the more
  DFL gives (W2's DFL +0.0019 over `ens2`, W4's −0.0034). D3 (more stress draws, not more weight) gains less on
  Metrica (+0.0149) for a similar DFL cost (−0.0023). J03WOH loses the most with every candidate (−0.033 for W4 and D3
  at 2 seeds).

**Held-out frame log loss** on B's 3 variants (`bench/results/lab_netweight/frames.csv`, information):

| | DFL's 7 | Metrica's 3 |
|---|---|---|
| netmet seeds 0 / 1 / `ens2` | 0.1745 / 0.1746 / **0.1709** | 0.2609 / 0.2646 / **0.2578** |
| W2 `ens2` | 0.1752 | 0.2492 |
| W4 `ens2` / W4 seeds 0-3 | 0.1817 / **0.1793** | 0.2413 / **0.2355** |
| D3 `ens2` | 0.1731 | 0.2486 |

The frames say what the chain says: two seeds against two, W4 cuts Metrica's held-out log loss by 6 % against `ens2`
(0.2578 → 0.2413) and raises DFL's by 6 % (0.1709 → 0.1817).

## Verdict

**W4 passes the bar for replacing `netmet5` as v16's network, on both parts.** Against v15's B it passes Metrica's
rule with room (Metrica's 3 +0.0525, 3/3; DFL's 7 +0.0010; worst variant +0.0152; SkillCorner +0.0028), and also the
full B bar (+0.0165, 7/10), which `netmet5` missed on "7 of 10". Against `netmet5` it gains on Metrica's 3 (+0.0125,
3/3) and on B's bench (+0.0026), and loses 0.0017 on DFL's 7 (allowed: 0.003).

**What Jose should weigh** (nothing is adopted):

1. **The trade is Metrica for DFL.** Our match is Metrica's tracker under Metrica's convention, so the Metrica gain
   is the one that counts, if our match behaves like Metrica's sample games. Out of selection, W4 gains about +0.011
   on Metrica's 3 and loses about 0.002-0.004 on DFL's 7 against `netmet5`.
2. **The DFL margin in (b) is thin** (−0.0017 against −0.003 allowed). The fresh seed pair alone shows −0.0038.
3. **Metrica's 3 are the provider W4 is weighted towards.** Each held-out Metrica game is scored by nets trained on
   the other two at weight 4, the same arrangement as `netmet`'s (audit point 1), only stronger. SkillCorner (no
   training on it) is level to slightly up (+0.0024 over `netmet5`; odd 10 +0.0015).
4. **Chosen after seeing scores, all declared above:** the finalist by the 10:28 rule (best of the built candidates
   on Metrica's 3); the combination W4D3 dropped by the clock at 01:20, unscored, after the machine stalled. Had it
   finished, it might have been the finalist.
5. **Not done:** the stoppage switch's check with W4 as B (as the night ran for `netmet5`); no re-search of trial
   1015's settings for the new P.

**For v16 (track 1):** the recipe is `bench/lab_netweight.py`'s `W4`; one more `RECIPES` entry in `bench/train_v16.py`
(not edited here):

```python
"netweight_W4": {
    "about": "bench/lab_netweight.py, candidate W4: netmet's recipe with every frame of a Metrica training match at "
             "loss weight 4 (batch loss the weighted mean); the `all` net on all 10 bench matches, normaliser on the "
             "7 DFL; seeds 0-3, P averaged (docs/lab/push/netweight.md)",
    "seeds": (0, 1, 2, 3),
    "fit": lambda seed, log: lab_netweight.fit_recipe(lab_netmet.used_for("all"), lab_netmet.norm_for("all"),
                                                      seed=seed, cand="W4", log=log),
    "lab_net": lambda seed: lab_netweight.net_path("all", seed, "W4"),
    "member": lambda seed: lab_netweight.root("W4", seed),
    "mirror": ("netweight", "W4_ens4"),
    "fold_net": lambda held, seed: lab_netweight.net_path(held, seed, "W4"),
    "fold_norm": lambda held: lab_netmet.norm_for(held),
},
```

The deployment nets: `data/external/lab_netweight/W4_s{0,1,2,3}/nets/net_all.pt`, cloned as
`data/external/lab_netweight/deploy_W4/netweight_W4_all_seed{0,1,2,3}.pt` with `manifest.json` (SHA-256 of each).
`fit_recipe` rebuilds seed 0's net to the bit.

## What ran, what didn't, and the files

| | ran | not run, and why |
|---|---|---|
| checks | reproduction of netmet seed 0 (`metrica1`, to the bit); the weighted loss line at weight 1; D3's draw code against netmet's copy; `fit_recipe` against W4's seed-0 net | |
| screen | W2, W4, D3: seeds 0-1 × 10 folds, `ens2` confirmed | each seed alone through the confirm (not needed for the rule) |
| combination | W4D3 seeds 0-1, 8 of 20 folds | dropped by the clock at 01:20 (machine stall); never scored |
| finalist | W4 seeds 2-3 × 10 folds, four `all` nets, `ens4` confirm, SkillCorner, the bar; fresh pair (seeds 2+3); frame log loss | the switch's check |
| our match | nothing | the push's rule |

Disk: `data/external/lab_netweight/` holds 1.8 GB (float32 P mirrors, ~100 MB each; D3's draws 290 MB; nets ~0.5 MB
each). Nothing was written into a shared cache. If space is needed, the losers' mirrors (`W2_*`, `D3_*`, `W4D3_*`,
`W4_ens2`) can go. Keep `W4_s0-3`, `W4_ens4`, `deploy_W4`, `repro` and `draws` (D3's reproduction).

## After the track: W4 under v17's broadcast merge (the coordinator, 2026-11-16 04:15)

v17 (the upload since 11-15 21:30) is v16 plus the broadcast merge (rule M2) in branch B's input, so W4 was also run
under the merge before Jose is asked (`bench/lab_broadcast.py --tag _w4`; `bench/results/lab_broadcast/w4_vs_v16.json`).
`off_w4` reproduces this page's `W4_ens4` (30 units, max 4.97e-6).

| W4 − `netmet5`, B's bench | B's bench | Metrica's 3 | DFL's 7 |
|---|---|---|---|
| no merge (this page's bar) | +0.0026 | +0.0125 (3/3) | −0.0017 |
| merge, central simulation | **−0.0000** | +0.0119 (3/3) | **−0.0051** |
| merge, pessimistic simulation | −0.0002 | +0.0183 (3/3) | −0.0081 |

The merge's gain on W4 is smaller than on `netmet5` (central +0.0051 against +0.0076) and its pessimistic run fails
track 3's robustness floor (−0.0032 against −0.002; `netmet5`: −0.0005). With the merge on, W4's bench gain is gone
and its DFL cost is past (b)'s 0.003; the Metrica gain stays.

**Decision (Jose, 2026-11-16 08:45, "Keep v17"):** W4 is not adopted; v17 keeps `netmet5` as B's network. W4's nets
and results stay where this page lists them ([decisions.md](../../decisions.md)).
