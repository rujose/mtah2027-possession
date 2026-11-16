# Night, builder "network": the network at the heart of branch B

| | |
|---|---|
| **status** | **finished 06:55, 2026-11-15. No idea passes the bar; `netmet5` misses it on one part (6 of 10 better) and passes Metrica's rule.** Plan written 23:35 before any new score. Nothing adopted; Jose decides |
| **question** | Branch B is 92 % one network (`gru_b`: one seed, trained on DFL's 7 matches only). Does it get better, on B's bench and the round-8 B bar, if it is (1) five seeds averaged, (2) trained with Metrica's 3 games too, or (3) trained with a training-only clean-owner head? |
| **proposals** | [ideas_network.md](ideas_network.md) (`netens`, `netmet`, the shared plumbing), [ideas_outside.md](ideas_outside.md) idea 2 (the clean-owner head, Barca-Hackathon's recipe) |
| **code** | `bench/lab_netmirror.py` (the mirror and the judge), `bench/lab_netens.py`, `bench/lab_netmet.py`, `bench/lab_netown.py`, `bench/lab_netavg.py` (new files; no existing file edited) |
| **outputs** | `data/external/lab_netmirror/`, `lab_netens/`, `lab_netmet/`, `lab_netown/`, `lab_netavg/`; `bench/results/` folders of the same names |
| **our match** | never read, nothing run on it |

## Morning summary (06:55)

**No idea passes the night's bar.** The lead is `netmet` (B's GRU trained with Metrica's 3 games too): with four
seeds averaged it is the night's strongest result (+0.0139 on B's bench, every variant and both sources up,
SkillCorner level) but **better on only 6 of 10 matches** (7 needed). With one seed it passed every part (+0.0130,
8/10) and then failed the noise rule (seed 1: +0.0069, 6/10, DFL's 7 down). The five-seed ensemble of `gru_b`, the
clean-owner head and `gru_b` + A's GRU all fail.

| idea | B's bench vs v15's B (0.7329) | better on | p25 / p50 / stressed | DFL's 7 / Metrica's 3 | SkillCorner vs v15's B (0.4085) | the bar |
|---|---|---|---|---|---|---|
| **`netmet5`**: `netmet`'s seeds 0-3 averaged (K by the clock) | **+0.0139** | **6/10** | +0.0140 / +0.0130 / +0.0147 | +0.0027 / +0.0400 | +0.0004 (12/20) | **fails** (7 of 10); passes every other part; passes Metrica's rule |
| `netmet`, seed 0 | +0.0130 | 8/10 | +0.0108 / +0.0120 / +0.0163 | +0.0047 / +0.0324 | −0.0017 (7/20) | passes, **but fails the noise rule** |
| `netmet`, seed 1 (the noise rule's run) | +0.0069 | 6/10 | +0.0046 / +0.0051 / +0.0110 | −0.0025 / +0.0288 | not run | fails (7 of 10; DFL down) |
| `netens`: 5 seeds of `gru_b` averaged | +0.0005 | 6/10 | ±0.0000 / +0.0001 / +0.0012 | +0.0001 / +0.0012 | −0.0005 (10/20) | fails (gain; 7 of 10) |
| `netown`: training-only clean-owner head | +0.0036 | 7/10 | +0.0037 / +0.0033 / +0.0038 | +0.0012 / +0.0091 | −0.0031 (6/20) | fails (gain) |
| `netavg`: `gru_b` + A's GRU averaged | −0.0112 | 2/10 | −0.0004 / −0.0095 / −0.0236 | −0.0134 / −0.0060 | −0.0164 (0/20) | fails every part |
| `gru_b` with seed 1 (the noise yardstick) | −0.0035 | 4/10 | −0.0035 / −0.0037 / −0.0032 | −0.0052 / +0.0006 | not run | (s = 0.0035 ≥ 0.003: the noise rule is on) |

What the night says:

1. **Metrica's games in training are the one lever that moves B** (+0.010 to +0.014 over `gru_b` with every seed and
   ensemble tried; Metrica's 3 up in every run, +0.015 to +0.052 a match). Averaging seeds removes `netmet`'s seed noise
   without losing the gain. What keeps it under the bar is two DFL matches, J03WOH and J03WPY (−0.012 each), that
   lose with every `netmet` seed; two more are level (−0.001).
2. **Seed noise is real: 0.0035 on B's bench with one second seed** (±0.02 on single matches). v15's own +0.0353 was measured
   on the seed its settings were tuned on: with seed 1 the same recipe gives +0.0319 over B as built.
3. **Averaging `gru_b`'s seeds improves the frame P (log loss −3 to −4 %) but not the chain** (+0.0005): the decoder and
   post stages absorb it, as in round 7.
4. With `netmet5` as branch B, the stoppage switch keeps parts 1, 2 and 4 and fails part 3 at 10 % and 25 % (as v15
   did, by more): B now beats A on the normal bench too (+0.0153, 7/10; all 3 of Metrica's normal games).

**What Jose must decide** (nothing was adopted):

- **Whether `netmet5` goes further**, under the precedent of v14 (a pass on Metrica's rule, not the bar): it passes
  Metrica's rule (Metrica +0.040, 3/3; DFL +0.003; no variant down; SkillCorner level) and fails the bar only on
  "7 of 10". It was registered at 03:37, after `netmet`'s seed 1 came in (said on the page); its K (4) was set by
  the clock at 06:44, before its score.
- **Whether Metrica's sample games may be used to train a deployed network at all** (no licence file,
  [limitations.md](../../limitations.md); the night's rules leave this to Jose). Deploying `netmet5` would also mean four
  networks in the model file and four GRU passes on our match.
- If `netmet5` is taken further: the switch's threshold (part 3 fails because B improved; our match takes B either
  way) and whether trial 1015's settings should be re-searched on the new P (not done tonight: forking paths).

## What each idea changes

Only the file behind B's GRU P changes: the held-out P of `gru_b` on B's bench variants (p25, p50, stressed; normal
and p10 for the switch's table) and on each match's training-stress copy (the letters B's post stages learn from).
Everything else is trial 1015 as v15 deploys it, its settings fixed (w_gru 0.915, f_dead 0.909, switch prices
2.82 / 48.7, minimum 7, snap and move off, short 0.340, segments 0.910 / 0.661, timing 0.450, post stages trained on
the stressed copies). No setting is searched again tonight (forking paths); they were picked on `gru_b`'s own P, so
this slightly handicaps every new network.

| idea | the network(s) | training data | deployment (SkillCorner) |
|---|---|---|---|
| `netens` | `gru_b`'s recipe, folds and settings with seeds 0-4 (seed 0 is `gru_b` itself, its files reused); **P = round(clip(mean of the five P, 0.01, 0.99), 3)** on every variant and on the training-stress copies | as `gru_b` (DFL's 6 or 7, as they are and under `augment`'s training stress) | the five seeds' `metrica1` fold nets (each on all 7 DFL), P averaged |
| `netmet` | `gru_b`'s recipe, seed 0, **10 folds, leave one match out over all 10** (a DFL fold trains on 6 DFL + 3 Metrica; a Metrica fold on 7 DFL + the other 2 Metrica) | every training match as it is and under `augment`'s training stress (Metrica's stressed copies exist: `lab_carried/features_1_metrica*.npz`); the normaliser and missing flags stay the fold's IDSSE matches' (`lab_temporal.normaliser`), so the inputs read as `gru_b`'s | one network on all 10 |
| `netown` | `gru_b`'s recipe, seed 0, its 8 folds, plus a **training-only second head**: for each frame, the team (home / away / nobody) of the nearest player within **1.5 m** of the ball **in the clean tracking** (`sets.load`), ignored where the clean ball is missing, home and away exchanged on the swapped orientation, the same target on the stressed copy. Loss = main + **0.5** × head. The head is built after `T.network(n_in)` with torch's random state saved and restored around it, so the main net's initial weights, its dropout draws and the batch order are v15's: the only difference is the head's gradient. P is the main head's, so the deployed program and its inputs don't change | as `gru_b` | the `metrica1` fold net (all 7 DFL), as v15 |

**Added 01:40, before any score of it (the coordinator's suggestion, from F3 in [ideas_network.md](ideas_network.md)):**

| idea | the network(s) | training data | deployment (SkillCorner) |
|---|---|---|---|
| `netavg` | no new training: B's GRU P becomes **round(clip(mean of `gru_b`'s P and A's GRU's P, 0.01, 0.99), 3)** (A's GRU: lab_temporal's network, trained on the normal DFL matches only, its own folds), each member through float32 as stored; on every variant and on the training-stress copies (A's GRU's P on those is computed tonight with its fold nets, inference only). No weight between the two (plain mean), trial 1015's settings untouched | as each member | `gru_b`'s and A's GRU's deployment nets (both on all 7 DFL), P averaged |

`netavg` is judged with the same bar. It averages two networks, so the noise rule's "single network" clause doesn't
apply as written; but both members are single seeds, so if seed noise s ≥ 0.003 its margin is read against s on this
page, and its pass is flagged.

## The bar (copied from [night.md](../night.md), fixed before any score)

**The bar for every candidate tonight** (the round-8 B bar, against v15's B as confirmed, trial 1015's settings
fixed): on B's bench (each match's mean of p25, p50, stressed) **+0.005 or more** (0.7379), **better on 7 of 10**,
**no stress variant down more than 0.002**, **neither source (IDSSE, Metrica) down on the mean**, **SkillCorner down
no more than 0.005** (v15's B 0.4085, so 0.4035). **Noise rule:** if a second seed of the single network moves B's
bench by 0.003 or more, a single-network pass must repeat with a second seed. `netmet` also reports Metrica's 3 and
DFL's 7 apart; a pass on Metrica only is a decision for Jose (as v14 was), and so is using Metrica's sample games in
training at all (they carry no licence file, [limitations.md](../../limitations.md)).

The reference rows are v15's B as confirmed: trial 1015 in `bench/results/search_b/confirm_1015_554_586_1133_1269_-1_-2.csv`
(B's bench 0.7329; p25 0.7289, p50 0.7319, stressed 0.7379) and `skillcorner_1015_554_586_1133_1269.csv` (0.4085).

How the parts apply here:

- **The noise yardstick** is `netens`'s seed 1 alone, through the same confirm: s = |B's bench of seed 1 − v15's B|.
  If s ≥ 0.003, a pass by a single network (`netmet`, `netown`) counts only when the same recipe with seed 1 also
  passes the bench parts; until that run exists it is reported as "pending the noise rule". The ensemble is not a
  single network, so the rule doesn't apply to it.
- **`netmet`'s secondary rule** (from [ideas_network.md](ideas_network.md), on round 8's precedent for Metrica,
  information for Jose, not a pass): Metrica's 3 **+0.010 or more** on the mean of their 3 variants and **better on
  3/3**; DFL's 7 no loss beyond 0.005; no variant down more than 0.002; SkillCorner no loss beyond 0.005. A result that
  passes only this is written up as "passes the Metrica rule, not the bar".
- **If two ideas pass, their product** (e.g. five seeds of `netmet`) replaces the better single passer only if it
  beats it on B's bench (mean higher, better on 6 of 10) and still passes the rest of the bar against v15.
- **`netens` uses K = 5** (seeds 0-4), fixed now. If the slots can't give the 32 trainings in time, K = 3 (seeds 0-2),
  declared here before any ensemble score is seen.

Reported, never judged: each seed's held-out log loss and B's bench through the full confirm where the slots allow;
the ensemble with 2, 3 and 4 members; possessions per match; the switch's table (normal, p10).

## The plumbing (before any new P is scored)

`bench/lab_netmirror.py` points B's caches at a small folder of the idea's own: `lab_branch_b.CACHE` (B's GRU P per
variant), `search_b.CACHE` (the GRU's P on the training-stress copies, and the trial's stage tables, recomputed
there: `tables_trial1015.joblib` is never reused from the shared cache), `search_b.RESULTS`, and lab_gate's dead-ball
feature cache (read from the shared ones, written only in the mirror). `augment`'s P and trial 0's tables (B as built:
no GRU in them) are read from the shared cache, never written. P files are copied in with APFS clones (no disk cost).

**The check:** the mirror filled with `gru_b`'s own P must reproduce trial 1015 (and trial 0) on all 30 units of the
confirm CSV above to 1e-5, and SkillCorner's 20 matches to 1e-5. If it doesn't, the reason goes here and nothing else
runs.

## The run plan

Every heavy job goes through `slot.py` (one slot, 2 threads); logs in the scratchpad's `logs/network/`.

| step | jobs | cost |
|---|---|---|
| plumbing check | confirm (trial 0, 1015) + SkillCorner in the mirror with `gru_b`'s P | ~15 min, 1 slot |
| `netens` training | seeds 1-4 × 8 folds, each fold followed by its held-out P (5 variants + training-stress copy) | 32 × ~8 min |
| `netmet` training | 10 folds + the deployment net, each followed by its held-out P | 11 × ~11 min |
| order | seed 1 and `netmet` first (the noise yardstick and the second idea), then seeds 2-4 | |
| judge | per candidate: confirm (B's 3 variants), confirm (normal, p10), SkillCorner | ~15 min each |
| then | `netown` (8 folds), if slots and time allow; then a product of two passers | |

## Results

### The plumbing check (23:45): passes, exactly

| check | units | max \|difference\| | |
|---|---|---|---|
| confirm, trials 0 and 1015, B's 3 variants × 10 matches, against round 8's CSV | 60 | **0.00000** | passes |
| SkillCorner, trials 0 and 1015 (gru_b's deployment net), 20 matches | 40 | **0.00000** | passes |

B's bench in the mirror: trial 0 0.6975, trial 1015 0.7329; SkillCorner 0.4077 / 0.4085, as round 8. Trial 1015's
stage tables were recomputed in the mirror (not reused). Nothing outside the mirror was written (`find -newer` on
`data/external` and `bench/results` after the run: only other builders' own folders). `bench/results/lab_netmirror/plumb/plumbing.json`.

Two more checks of the new code, before any new network was scored:

- **Prediction** (`lab_netens check`): `lab_netmirror.predict_into` with `gru_b`'s own fold net on J03WMX gives `gru_b`'s
  saved P exactly (max |difference| 0 on normal, p10, p25, p50, stressed and the training-stress copy).
- **Training** (`fit_check.py`, 1 epoch on 2 matches): `lab_netmirror.fit` without hooks gives `lab_branch_b.fit`'s
  weights to the bit (max |difference| 0); with the owner head attached but every target ignored, the main net is
  again identical (0): saving and restoring torch's random state around the head works, so `netown` differs from
  `gru_b` only by the head's gradient.
- **A second reproduction, independent of this mirror** (the lead's `stage3.sh c0`, a full no-change rebuild of B in
  the scratchpad, finished 00:48: `gru_b`'s 8 folds retrained from scratch, then confirm and SkillCorner): trial 0 and
  1015 on all 60 units **identical** to round 8 (max |difference| 0), SkillCorner 0.4085. So `gru_b`'s training is
  bit-reproducible on this machine, and the spread between seeds below is the seed's alone.

### `netavg` (01:44): fails the bar, clearly

The plain mean of `gru_b` and A's GRU as B's GRU P, trial 1015's settings fixed (`bench/results/lab_netavg/main/`):

| | v15's B | `netavg` | difference |
|---|---|---|---|
| **B's bench** | 0.7329 | 0.7217 | **−0.0112**, better on 2/10 |
| p25 / p50 / stressed | 0.7289 / 0.7319 / 0.7379 | 0.7285 / 0.7223 / 0.7143 | −0.0004 / −0.0095 / **−0.0236** |
| DFL's 7 / Metrica's 3 | | | −0.0134 / −0.0060 |
| SkillCorner (20) | 0.4085 | 0.3921 | **−0.0164**, better on 0/20 |

Fails every part of the bar. The loss grows with the stress (p25 level, stressed −0.024): A's GRU never saw a
moving stoppage ball (it was trained on the normal bench only), so on B's bench it pulls the mean the wrong way exactly
where B runs. F3's −7.5 % log loss was measured on the **normal** variant, the one variant B never sees. Only J03WR9
(+0.002) and metrica1 (+0.002) gain. SkillCorner's real stoppage balls say the same (−0.016): A's GRU reads a
moving ball in a stoppage as live play.

### Seed noise (01:52): seed 1 alone moves B's bench by −0.0035, so the noise rule is on

`gru_b`'s recipe with seed 1 instead of 0 (same folds, data, settings; trial 1015's settings fixed), through the same
confirm (`bench/results/lab_netens/seed1/`):

| | v15's B (seed 0) | seed 1 alone | difference |
|---|---|---|---|
| **B's bench** | 0.7329 | 0.7294 | **−0.0035**, better on 4/10 |
| p25 / p50 / stressed | 0.7289 / 0.7319 / 0.7379 | 0.7254 / 0.7282 / 0.7346 | −0.0035 / −0.0037 / −0.0032 |
| DFL's 7 / Metrica's 3 | | | −0.0052 / +0.0006 |
| per match | | | J03WMX +0.004, J03WN1 +0.003, J03WOH −0.020, J03WOY −0.011, J03WPY −0.012, J03WQQ +0.007, J03WR9 −0.007, metrica1 +0.009, metrica2 −0.004, metrica3 −0.004 |

- **s = 0.0035 ≥ 0.003: the noise rule applies.** A pass by a single network (`netmet`, `netown`) counts only if the
  same recipe with seed 1 also passes the bench parts.
- Read the other way: a single match moves by up to ±0.02 with the seed alone, and v15's +0.0353 was measured on the
  seed its settings were tuned on; with seed 1 the same recipe and settings give +0.0319 over B as built. Part of
  v15's margin is seed 0's luck (about 0.003, the size of the bar's +0.005 within a factor of two).

### `netmet` (02:20): passes every part of the bar with seed 0 (the noise rule then fails with seed 1, below)

`gru_b`'s recipe, seed 0, trained with Metrica's 3 games too (leave one match out over all 10), trial 1015's settings
fixed (`bench/results/lab_netmet/main/`):

| | v15's B | `netmet` | difference |
|---|---|---|---|
| **B's bench** | 0.7329 | **0.7459** | **+0.0130**, better on **8/10** |
| p25 / p50 / stressed | 0.7289 / 0.7319 / 0.7379 | 0.7396 / 0.7438 / 0.7542 | +0.0108 / +0.0120 / +0.0163 |
| **DFL's 7** / **Metrica's 3** | | | **+0.0047** / **+0.0324** (3/3) |
| **SkillCorner** (20, the `all` net) | 0.4085 | 0.4068 | **−0.0017**, better on 7/20 (allowed: −0.005) |

| match | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | J03WQQ | J03WR9 | metrica1 | metrica2 | metrica3 |
|---|---|---|---|---|---|---|---|---|---|---|
| `netmet` − v15's B | +0.007 | +0.007 | −0.015 | +0.019 | −0.008 | +0.006 | +0.017 | +0.015 | +0.050 | +0.032 |

- **Bench parts of the bar: all pass** (+0.0130 ≥ 0.005; 8/10 ≥ 7; every variant up; neither source down). The
  margin is 3.7 times the seed noise (0.0035).
- **The gain is not only Metrica's.** DFL's 7 gain +0.0047 on the mean (5 of 7 better): three more training matches
  help them more than Metrica's convention hurts. J03WOH (−0.015) is the match that also lost most with seed 1
  (−0.020): it moves with anything that changes the network.
- Frame level (held out, `lab_netmet report`): Metrica's log loss 0.303 → 0.267 on p25 (accuracy 91.4 → 92.9 %);
  DFL's 0.186 → 0.189 (level). F4's pocket (live called dead, in runs of 3 s or more) shrinks only a little on
  Metrica (4.3 → 4.0 % of graded frames): the gain comes mostly from elsewhere (team calls), not from the convention
  pocket the proposal aimed at.
- **SkillCorner passes** (−0.0017 against −0.005 allowed), but leans down (7/20): Metrica's convention is not
  SkillCorner's, and every B scores ~0.41 there.
- **Metrica's rule** (information): passes too (Metrica +0.032, 3/3; DFL +0.005).
- **The noise rule:** `netmet` with seed 1 (10 folds, queued first at 02:22) had to pass the bench parts too. It
  didn't (03:35, below).
- **Queued behind everything else (02:27), for the product only:** `netmet` with seeds 2-4 (and each seed's `all` net),
  so that five seeds of `netmet` can be averaged if `netens` also passes (the product rule above). If `netens` fails,
  these runs only measure `netmet`'s seed spread and nothing is judged on them. **The product's K, fixed now (02:47),
  before any score of it:** the mean of `netmet`'s seeds 0-4; if not all are trained (with their `all` nets) by
  07:00, the seeds that are (at least 0 and 1), chosen by the clock, not by any score. It is judged only if `netens`
  passes (changed at 03:37: judged whatever `netens` gives, below).

### The stoppage switch with `netmet` as branch B (03:03, information; the coordinator's request)

Round 8's check (`bench/check_v15_switch.py`, threshold 0.1445 unchanged, branch A v14's) with `netmet`'s trial 1015
as B: `python -m bench.lab_netmirror switch netmet main` (the script's module paths pointed at the mirror's results;
checked first: pointed at round 8's own files it reproduces `check_v15_switch/report_1015.json` exactly). Normal and
p10 come from `confirm --variants normal p10` in the mirror. Nothing runs on our match. Run before `netmet`'s seed 1
was in (it costs one confirm); the noise rule still applies to `netmet` itself.

| bench variant | v14's A | v15's B | **`netmet`'s B** | `netmet`'s B better than A on | the switch picks B on | switch | switch − the better branch |
|---|---|---|---|---|---|---|---|
| normal | 0.7241 | 0.7233 | **0.7355** | 7/10 | 0 | 0.7241 | **−0.0113** |
| 10 % stressed | 0.7124 | 0.7255 | 0.7368 | 10/10 | 0 | 0.7124 | **−0.0244** |
| 25 % | 0.6979 | 0.7289 | 0.7396 | 10/10 | 5 | 0.7126 | **−0.0270** |
| 50 % | 0.6699 | 0.7319 | 0.7438 | 10/10 | 10 | 0.7438 | ±0 |
| stressed | 0.6107 | 0.7379 | 0.7542 | 10/10 | 10 | 0.7542 | ±0 |
| SkillCorner (20) | 0.3196 | 0.4085 | 0.4068 | 20/20 | 20 | 0.4068 | ±0 |

1. **Part 1 (normal: A everywhere, or no loss beyond 0.002): passes** (A on 10/10).
2. **Part 2 (stressed: B on 9+, +0.02): passes** (B on 10/10, +0.143).
3. **Part 3 (10 / 25 / 50 %: within 0.005 of the better branch): fails at 10 % (−0.024) and 25 % (−0.027)**, passes
   at 50 %: the same failure as v15's (−0.013, −0.017), larger, for the same reason: B improved, and the threshold keeps
   A where B is now better.
4. **Part 4 (SkillCorner: +0.02, 15/20): passes** (+0.087, 20/20; B on 20/20).

- **`netmet`'s B now beats A on the normal bench too** (+0.0113, 7/10), including **all 3 of Metrica's normal games**
  (+0.004, +0.053, +0.017), where v15's B lost all 3 (−0.014, −0.001, −0.019) and so "B everywhere" was refused in
  round 8. The round-7 threshold rule, recomputed with this B, has no crossing (B ahead of A at every stress level:
  the rule returns −∞, on all 10 and on every leave-one-out).
- Caveat on that comparison: `netmet`'s B trained on Metrica's other games, A never did; on Metrica the comparison
  favours B for reasons that are not about the ball.
- **None of this changes our match's file**: our match takes B under the deployed threshold (round 8), whatever B is.
  The threshold matters only for other matches; whether to lower it (or send everything to B) is a decision for
  Jose, not taken here. `bench/results/lab_netmet/main/switch/report_1015.json`.

### `netmet` with seed 1 (03:35): the noise rule is not met

The same recipe with seed 1 (10 folds), through the same confirm, against v15's B (`bench/results/lab_netmet/seed1/`):

| | v15's B | `netmet` seed 0 | **`netmet` seed 1** | the bench parts, seed 1 |
|---|---|---|---|---|
| B's bench | 0.7329 | 0.7459 (+0.0130, 8/10) | **0.7397 (+0.0069, 6/10)** | gain passes; **7 of 10 fails** |
| p25 / p50 / stressed | | +0.0108 / +0.0120 / +0.0163 | +0.0046 / +0.0051 / +0.0110 | passes |
| DFL's 7 / Metrica's 3 | | +0.0047 / +0.0324 | **−0.0025** / +0.0288 (3/3) | **DFL down: fails** |

**So `netmet` does not pass under the noise rule**: seed 1 fails two of the bench parts (6/10 better; DFL's 7 down
−0.0025). Read within each seed, the effect of training on Metrica is steadier than that verdict suggests:

| `netmet` − `gru_b`, same seed | mean | better on | DFL's 7 | Metrica's 3 |
|---|---|---|---|---|
| seed 0 | +0.0130 | 8/10 | +0.0047 | +0.0324 |
| seed 1 | +0.0103 | 6/10 | +0.0027 | +0.0282 |

- Metrica's 3 gain +0.02 to +0.05 each with both seeds (6 of 6). DFL's 7 move ±0.02 a match with the seed alone;
  J03WOY (+0.019, +0.019) and J03WR9 (+0.017, +0.017) gain with both seeds, J03WOH and J03WPY lose with both.
- Seed 1 starts lower (`gru_b` seed 1 is −0.0035 under seed 0, above), and v15's B is seed 0, the seed its settings
  were tuned on. The bar compares a seed-1 network with that tuned seed-0 one, which is what the noise rule intends.
- **What this points to: the variance, not the direction.** Averaging seeds is what removes it, which is the product
  queued at 02:27 (`netmet` with seeds 0-4). **Changed at 03:37, before any score of it, and said here because it was
  decided after seeing this result:** the five-seed `netmet` (`netmet5`, K fixed as at 02:47) is judged with the bar
  whatever `netens` gives (it is an ensemble, so the single-network noise rule doesn't apply to it). Jose should
  weigh it knowing it was chosen after `netmet`'s seed 1 came in.
- **Order changed at 03:40:** `netmet`'s seeds 2-4 now train before `netown` (the coordinator's priority: `netens`,
  then `netmet`'s seeds); `netown` runs last, if the night allows. `netavg`'s switch table (normal, p10) dropped: it
  failed every part, and the slot is worth more elsewhere.

### `netens` (04:30): fails the bar; five seeds averaged are level with v15's B

Five seeds of `gru_b` (0-4; seed 0 is `gru_b`), P averaged, trial 1015's settings fixed
(`bench/results/lab_netens/ens5/`):

| | v15's B (seed 0) | `netens` (5 seeds) | difference |
|---|---|---|---|
| **B's bench** | 0.7329 | 0.7333 | **+0.0005**, better on 6/10 |
| p25 / p50 / stressed | 0.7289 / 0.7319 / 0.7379 | 0.7289 / 0.7320 / 0.7391 | ±0.0000 / +0.0001 / +0.0012 |
| DFL's 7 / Metrica's 3 | | | +0.0001 / +0.0012 |
| SkillCorner (20, five deployment nets) | 0.4085 | 0.4081 | −0.0005, better on 10/20 |

| match | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | J03WQQ | J03WR9 | metrica1 | metrica2 | metrica3 |
|---|---|---|---|---|---|---|---|---|---|---|
| `netens` − v15's B | +0.005 | −0.006 | −0.010 | +0.003 | −0.008 | +0.011 | +0.006 | −0.006 | +0.006 | +0.004 |

- **Fails parts 1** (+0.0005 against +0.005; 6/10 against 7). Passes the guards: no variant down, neither source
  down, SkillCorner −0.0005.
- Against the seed-1 network (0.7294) the ensemble is +0.004: averaging buys back what a single seed can lose, but it
  doesn't beat the seed whose P trial 1015's settings were tuned on. The ensemble's value is robustness (it removes
  the ±0.02-a-match seed swings), not a higher score on this bench.
- **At frame level the ensemble is clearly better, and the chain absorbs it** (held-out log loss, mean of the 10
  matches; `bench/results/lab_netens/frames.csv`):

  | | normal | p25 | stressed | training-stress copy |
  |---|---|---|---|---|
  | single seeds 0-4 (range) | 0.2278-0.2321 | 0.2197-0.2225 | 0.1948-0.1976 | 0.1948-0.1971 |
  | 2 / 3 / 4 / 5 seeds averaged | 0.2250 / 0.2226 / 0.2222 / 0.2221 | 0.2166 / 0.2145 / 0.2138 / 0.2136 | 0.1926 / 0.1909 / 0.1903 / 0.1899 | 0.1932 / 0.1913 / 0.1905 / 0.1898 |

  Log loss −3 to −4 %, argmax accuracy +0.1 point (92.65 → 92.76 % on p25), most of it with 3 seeds. The decoder's
  switch prices and the post stages turn that into +0.0005 on spellF1: the round-7 lesson again (a better P at
  frame level is not a better chain unless it is better over time).
- **With 2, 3, 4 and 5 members** (information, full confirm each, 06:45): B's bench −0.0013 (4/10), +0.0029 (6/10),
  +0.0011 (5/10), +0.0005 (6/10) against v15's B. No member count reaches the bar; the ensemble sits where seed 0 is,
  within ±0.003.
- The switch's table (normal, p10), information: 0.7239 / 0.7256 against v15's B 0.7233 / 0.7255: level.
- **Order changed at 04:33** (the clock, not a score): `netmet`'s seed 2, then `netown`'s 8 folds, then `netmet`'s
  seeds 3-4. With another builder on two slots, all of `netmet`'s seeds 3-4 may not finish by 07:00; the 02:47 rule
  then sets `netmet5`'s K to the seeds trained by 07:00 (at least 0-2). `netens` with 2-4 members and its switch
  table run last, for information.

### `netmet` seeds 0 and 1 averaged (04:59, information; not the registered candidate)

Scored while seeds 2-4 train, so the morning has a number if the night runs short; `netmet5`'s K is still the
clock's (02:47 rule), not chosen from this (`bench/results/lab_netmet/ens2/`):

| | v15's B | `netmet`, seeds 0+1 averaged | difference |
|---|---|---|---|
| B's bench | 0.7329 | 0.7448 | **+0.0119**, better on 8/10 |
| p25 / p50 / stressed | | | +0.0102 / +0.0106 / +0.0149 |
| DFL's 7 / Metrica's 3 | | | +0.0027 / +0.0334 (3/3) |

Per match: J03WMX +0.003, J03WN1 +0.001, J03WOH −0.014, J03WOY +0.019, J03WPY −0.009, J03WQQ +0.006, J03WR9 +0.014,
metrica1 +0.028, metrica2 +0.046, metrica3 +0.027. Two seeds averaged keep almost all of seed 0's gain (+0.0130) and
every bench part passes; DFL's 7 stay positive (5 of 7 better).

### `netown` (06:08): fails the bar (+0.0036, under +0.005 and within the seed noise)

`gru_b`'s recipe, seed 0, its 8 folds, with the training-only clean-owner head (1.5 m, weight 0.5); P from the main
head only; trial 1015's settings fixed (`bench/results/lab_netown/main/`). A paired comparison with v15's B: same
initial weights, dropout draws and batch order, only the head's gradient differs.

| | v15's B | `netown` | difference |
|---|---|---|---|
| **B's bench** | 0.7329 | 0.7365 | **+0.0036**, better on 7/10 |
| p25 / p50 / stressed | | | +0.0037 / +0.0033 / +0.0038 |
| DFL's 7 / Metrica's 3 | | | +0.0012 / +0.0091 (3/3) |
| SkillCorner (20, the `metrica1` fold net) | 0.4085 | 0.4054 | −0.0031, better on 6/20 |

Per match: J03WMX +0.003, J03WN1 +0.006, J03WOH −0.004, J03WOY −0.003, J03WPY −0.012, J03WQQ +0.001, J03WR9 +0.018,
metrica1 +0.008, metrica2 +0.014, metrica3 +0.005.

- **Fails part 1** (+0.0036 against +0.005); passes 7 of 10, every variant, both sources, SkillCorner (−0.0031,
  allowed −0.005). The gain is the size of the seed noise (0.0035), so even a pass would have needed seed 1.
- It points the same way as Barca-Hackathon's result (Metrica up most: +0.009 here, their +0.003 to +0.011 per seed),
  but smaller than the bar. Not run with a second seed (it failed with the first).
- **06:44: K = 4.** Seed 3 (with its `all` net) finished at 06:44, before 07:00, so `netmet5` is the mean of
  `netmet`'s seeds 0-3 (`bench/results/lab_netmet/ens4/`), fixed by the clock before any score of it.
- **06:12, the clock:** `netmet`'s seed 4 can't finish by 07:00 (seed 3 is half done), so its 11 trainings are
  dropped from the queue; under the 02:47 rule `netmet5`'s K is seeds 0-3 if seed 3 (with its `all` net) is done by
  07:00, else seeds 0-2.

### `netmet5` = `netmet`'s seeds 0-3 averaged (06:50): misses the bar on one part (better on 6 of 10)

The registered product (K = 4 by the clock: seeds 0-3, each with its `all` net), trial 1015's settings fixed
(`bench/results/lab_netmet/ens4/`):

| | v15's B | `netmet5` (4 seeds) | difference | the bar |
|---|---|---|---|---|
| **B's bench** | 0.7329 | **0.7468** | **+0.0139** | +0.005: passes |
| better on | | | **6/10** | 7 of 10: **fails** |
| p25 / p50 / stressed | 0.7289 / 0.7319 / 0.7379 | | +0.0140 / +0.0130 / +0.0147 | no variant down: passes |
| DFL's 7 / Metrica's 3 | | | +0.0027 / **+0.0400** (3/3) | neither down: passes |
| **SkillCorner** (20, four `all` nets averaged) | 0.4085 | 0.4090 | +0.0004, better on 12/20 | −0.005: passes |

| match | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | J03WQQ | J03WR9 | metrica1 | metrica2 | metrica3 |
|---|---|---|---|---|---|---|---|---|---|---|
| `netmet5` − v15's B | +0.009 | −0.001 | −0.012 | +0.018 | −0.012 | −0.001 | +0.018 | +0.034 | +0.052 | +0.034 |

- **It fails the bar on one part: 6 of 10 matches better, 7 needed.** Two of the four losses are level (J03WN1
  −0.0007, J03WQQ −0.0011); the other two, J03WOH and J03WPY (−0.012 each), lose with every `netmet` seed (and J03WOH
  with `gru_b`'s seed 1 too).
- Everything else is the strongest of the night: the largest mean gain (+0.0139), all three variants +0.013 to
  +0.015, DFL's 7 up (+0.0027), Metrica's 3 up +0.034 to +0.052, SkillCorner level (+0.0004, 12/20, where seed 0
  alone was −0.0017, 7/20). Averaging four seeds removed the seed problem (seed 0 +0.0130, seed 1 +0.0069; the mean
  of 4 +0.0139) without removing the gain.
- **Metrica's rule** (information, round 8's precedent): passes (Metrica +0.040, 3/3; DFL +0.003; no variant down;
  SkillCorner level). So `netmet5` is "passes the Metrica rule, not the bar": for Jose, as v14 was.

### The stoppage switch with `netmet5` as branch B (06:53, information)

As for `netmet` above (`bench/results/lab_netmet/ens4/switch/report_1015.json`):

| bench variant | v14's A | v15's B | **`netmet5`'s B** | B better than A on | the switch picks B on | switch − the better branch |
|---|---|---|---|---|---|---|
| normal | 0.7241 | 0.7233 | **0.7394** | 7/10 | 0 | −0.0153 |
| 10 % | 0.7124 | 0.7255 | 0.7405 | 10/10 | 0 | −0.0281 |
| 25 % | 0.6979 | 0.7289 | 0.7429 | 10/10 | 5 | −0.0267 |
| 50 % | 0.6699 | 0.7319 | 0.7449 | 10/10 | 10 | ±0 |
| stressed | 0.6107 | 0.7379 | 0.7525 | 10/10 | 10 | ±0 |
| SkillCorner | 0.3196 | 0.4085 | 0.4090 | 20/20 | 20 | ±0 |

Parts 1, 2 and 4 pass; part 3 fails at 10 % (−0.028) and 25 % (−0.027). On the normal bench B − A per match:
J03WMX −0.003, J03WN1 +0.011, J03WOH −0.007, J03WOY +0.018, J03WPY +0.026, J03WQQ −0.009, J03WR9 +0.020, metrica1
+0.018, metrica2 +0.054, metrica3 +0.024 (with the caveat above: B trained on Metrica's other games, A didn't). The
round-7 threshold rule again finds no crossing. Our match's branch doesn't change.

`netown`'s switch table (information): normal 0.7275, p10 0.7307 (v15's B 0.7233, 0.7255).

## What ran and what didn't

| | ran | not run, and why |
|---|---|---|
| plumbing | mirror check (60 bench units + 40 SkillCorner, exact), prediction and training checks (exact) | |
| `netens` | seeds 1-4 × 8 folds; confirm of seed 1 alone and of the 2-, 3-, 4-, 5-member means; SkillCorner and switch table for 5 | seeds 2-4 alone through the confirm (slots went to `netmet`) |
| `netmet` | seeds 0-3 × 10 folds + `all`; confirm of seeds 0 and 1 alone, of 0+1 and of 0-3 averaged; SkillCorner for seed 0 and for 0-3; switch check for seed 0 and for 0-3 | seed 4 (dropped at 06:12: couldn't finish by 07:00); SkillCorner for seed 1 alone (the noise rule needs the bench parts only) |
| `netown` | 8 folds, seed 0; confirm, SkillCorner, switch table | seed 1 (it failed with seed 0) |
| `netavg` | confirm, SkillCorner | switch table (dropped: failed every part) |
| our match | nothing | the night's rule |

Slots: four, shared with the stage-3 rebuild until 00:48 and with the ghost, pstage, netsc and netsc2 builders; the
queue runner paused 30 s between jobs so others' waiting jobs could take freed slots. Disk: about 1.8 GB in
`data/external/lab_net{mirror,ens,met,own,avg}/` (mostly float32 P files, ~100 MB per mirror; nets ~0.6 MB each);
nothing written into a shared cache.
