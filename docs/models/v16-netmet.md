# Model v16: v15 with branch B's network trained on Metrica's games too (`netmet5`)

| | |
|---|---|
| **status** | **built and checked 2026-11-15 11:49 (track 1 of [the push](../lab/push.md)); not the upload.** The four networks retrain bit-identical to the night's; the port gives the lab's letters on 3 bench units (0 frames in float32 and in float64); the built file reproduces the lab's SkillCorner score on all 20 matches; on our match three runs give one file. Building v16 was approved (Jose, 10:10: "Yes, all three"). **Adopted as the upload 2026-11-15 (Jose: "Yes, adopt v16")**; track 2's recipe may still replace the network if it passes its bar ([push.md](../lab/push.md)) |
| **what** | v15 ([v15](v15-branch-b.md)) with one change in branch B: the GRU P that weighs 0.915 in B's start is **`netmet5`'s**, the mean of four networks (seeds 0-3) of `gru_b`'s recipe trained on **all 10 bench matches, Metrica's 3 sample games included** (v15's `gru_b`: one network, the 7 DFL matches). B's post stages are retrained on that chain's letters. Branch A, the stoppage switch, its threshold (0.1445), the kick-off rule and trial 1015's decoder and post-stage settings are v15's |
| **why** | Our match is Metrica's tracker under Metrica's answer convention, and the switch sends it to branch B. The night's one lever that moved B was what its network trains on ([night/network.md](../lab/night/network.md)): on Metrica's 3, each scored by networks that never saw it, `netmet5` gains +0.040, 3/3 |
| **bench** (B's bench: 25 %, 50 % and fully stressed; each match scored by nets that never saw it) | **0.7468 against v15's B 0.7329 (+0.0139), better on 6/10**: Metrica's 3 **+0.040, 3/3**; DFL's 7 +0.0027; every variant up. **Fails the B bar's "7 of 10", passes Metrica's rule**: the precedent of v14 |
| **SkillCorner** (20 real matches, deployment models) | 0.4090 against 0.4085 (+0.0004, 12/20): level |
| **our match** | SHA-256 `90599088419d812cbb341168b27ff0ff767549b75c8dfa12a51de290ccc0523c` (3 runs agree; format check OK); v15's file unchanged (`209a2081…7f5e`) |
| **new code** | `possession/v16.py`, `possession/models/v16_b_all.joblib` (3.4 MB, four networks), `bench/train_v16.py`, `bench/check_v16.py` (port, SkillCorner, switch); the lab: `bench/lab_netmet.py`, `bench/lab_netmirror.py` |
| **run** | `OMP_NUM_THREADS=2 python -m bench.train_v16 nets --seeds 0 1` and `… --seeds 2 3` (~40 min side by side), `python -m bench.train_v16` (seconds once the nets exist), `python -m bench.check_v16 port` (~4 min), `… skillcorner` (~17 min), `… switch`; then `python -m possession.v16` on our match: path, SHA-256 and format check only (`--show-branch` for Jose) |

## What changes, and what doesn't

| | v15's branch B | v16's branch B |
|---|---|---|
| the GRU P in B's start | `gru_b`: one network (seed 0), trained on the 7 DFL matches as they are and under `augment`'s training stress | **`netmet5`: four networks (seeds 0-3) of the same recipe trained on all 10 bench matches** (the 7 DFL + Metrica's 3), each as it is and under `augment`'s training stress; normaliser and missing flags from the 7 DFL matches (as `gru_b`'s). Each network's P clipped and rounded, **through float32, averaged, clipped and rounded** |
| the start | 0.9152 P_GRU + 0.0848 P_`augment` (both through float32, clipped, rounded); P(dead) × 0.9095, rows renormalised | the same, with `netmet5`'s P as P_GRU |
| decoder, snap and move, post-stage settings | trial 1015: switch prices 2.82 / 48.7, minimum 7; snap and move off; short 0.340, segments 0.910 / 0.661, timing 0.450 | unchanged (checked equal to v15's file when `train_v16` builds) |
| what the post stages are trained on | trial 1015's own chain's letters on the 10 bench matches under `augment`'s training stress, from `gru_b`'s out-of-fold P | the same chain from **`netmet5`'s** out-of-fold P (each match's P from the four fold nets that never saw it, averaged): the lab's `netmet/ens4` mirror, the stage tables the night's confirm used |
| `augment`, branch A, the switch, threshold 0.1445, kick-off rule, bench set `idsse` | | unchanged (v15's `augment` checked equal) |
| model file | `v15_b_all.joblib`: one network | `v16_b_all.joblib`: **four networks**, so four GRU passes (float64) on our match |

### What the lab did, read in its code (each point checked before building)

1. **How `netmet5` averages** (`lab_netmet.combine` → `lab_netmirror.average`): each seed's held-out P is
   `lab_gate.gru_proba`'s (clipped to 0.01-0.99 and rounded to 3 decimals), stored as float32; the four are read as
   float32 → float64, averaged, **then** clipped and rounded, and stored as float32. The night's SkillCorner run
   (`lab_netmirror.skillcorner`, the four `all` nets) calls the same `average` on the four nets' P without storing
   it; both routes then pass the mean through float32 inside the mix (`lab_branch_b.mix`), so they give the same
   start. (The audit's nit, that the two routes could round a tie differently, does not arise: both average the same
   float32 members and both cast the rounded mean to float32 before the mix.) v16 does exactly this
   (`v16.average`), with each network run in float64 as v13-v15 run their GRU.
2. **The mix** is `lab_branch_b.mix` with trial 1015's w_gru 0.9152 (= `v14.mix`, which v15 uses), then P(dead) ×
   0.9095, renormalised, clipped, rounded (as `v15.b_start_proba`).
3. **The post stages**: the night's `netmet5` confirm ran `search_b.confirm` in the `ens4` mirror, where
   `search_b.trial_tables` rebuilt trial 1015's out-of-fold stage tables from `ens4`'s P on the training-stress
   copies, and scored each match with stage models trained without it. Its SkillCorner run used
   `search_b.trial_models` on all 10 tables. `train_v16` does what `train_v15` did (`trial_models` on all 10 tables),
   pointed at the `ens4` mirror (`lab_netmirror.use`), after checking the mirror's training-copy P is the mean of the
   four seeds' (max |difference| 0).
4. **The deployment nets**: the night trained `netmet`'s `all` net per seed with
   `lab_netmirror.fit(all 10 matches, norm_used=the 7 DFL, seed=S)` (`lab_netmet.train --fold all --seed S`, one
   slot, 2 threads); `train_v16` calls the same and compares the weights with the night's `net_all.pt` of each seed.

## The bench (the night's `netmet5`, trial 1015's settings fixed; [night/network.md](../lab/night/network.md))

B's bench is the 10 bench matches with SkillCorner's stoppage balls pasted into 25 %, 50 % and all of their true
stoppages; each match is scored by the four fold nets that never saw it, averaged, and stage models trained without
it. Against v15's B as confirmed in round 8 (`search_b/confirm_1015_…csv`, `skillcorner_1015_…csv`):

| | v15's B | `netmet5`'s B (v16) | difference | the round-8 B bar |
|---|---|---|---|---|
| **B's bench** (mean of 25 %, 50 %, stressed) | 0.7329 | **0.7468** | **+0.0139** | +0.005: passes |
| better on | | | **6/10** | 7 of 10: **fails** |
| 25 % / 50 % / stressed | 0.7289 / 0.7319 / 0.7379 | | +0.0140 / +0.0130 / +0.0147 | no variant down: passes |
| DFL's 7 / Metrica's 3 | | | +0.0027 / **+0.0400 (3/3)** | neither source down: passes |
| SkillCorner (20, deployment nets) | 0.4085 | 0.4090 | +0.0004, better on 12/20 | down ≤ 0.005: passes |

| match | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | J03WQQ | J03WR9 | metrica1 | metrica2 | metrica3 |
|---|---|---|---|---|---|---|---|---|---|---|
| v16's B − v15's B | +0.009 | −0.001 | −0.012 | +0.018 | −0.012 | −0.001 | +0.018 | +0.034 | +0.052 | +0.034 |

- **It fails the B bar on one part, "7 of 10"** (6 better; two of the four losses are level, J03WN1 −0.0007 and
  J03WQQ −0.0011). **It passes Metrica's rule** (Metrica's 3 +0.040, 3/3; DFL's 7 +0.003; no variant down; SkillCorner
  level): the same standing v14 had when Jose took it ("a pass on Metrica's rule, not the bar").
- **Read the +0.040 as a provider the networks trained on.** Each Metrica game is scored by nets that trained on the
  other two (different matches, same provider and convention; [night/audit_netmet.md](../lab/night/audit_netmet.md)
  found no leak). For a provider no net saw, the nearer guides are DFL's 7 (+0.0027, under the seed noise 0.0035)
  and SkillCorner's odd 10 (in no training: +0.0025, 7/10).
- The audit's second-order check: `netmet5`'s P decoded with **v15's own stage models** (no net in them saw a
  Metrica game) still gives Metrica +0.041, so the gain is the network's, not the post stages'.
- The stoppage switch with this B (information; `check_v16 switch` reproduces the night's report exactly): parts 1,
  2 and 4 pass, part 3 fails at 10 % (−0.028) and 25 % (−0.027), as v15's did by less; B now beats A on the normal
  bench too (+0.0153, 7/10). **The threshold stays 0.1445**: our match takes B under it either way.

| bench variant | v14's A | v15's B | v16's B | the switch picks B on | switch − the better branch |
|---|---|---|---|---|---|
| normal | 0.7241 | 0.7233 | 0.7394 | 0 | −0.0153 |
| 10 % | 0.7124 | 0.7255 | 0.7405 | 0 | −0.0281 |
| 25 % | 0.6979 | 0.7289 | 0.7429 | 5 | −0.0267 |
| 50 % | 0.6699 | 0.7319 | 0.7449 | 10 | ±0 |
| stressed | 0.6107 | 0.7379 | 0.7525 | 10 | ±0 |
| SkillCorner | 0.3196 | 0.4085 | 0.4090 | 20 | ±0 |

## Built: what the checks say

| check | result |
|---|---|
| the four deployment nets retrained (`train_v16 nets`, seeds 0-3, 2 threads) | **bit-identical** to the night's `lab_netmet/{main,seed1,seed2,seed3}/nets/net_all.pt` (each ~18-19 min) |
| the rest of B (`train_v16`) | trial 1015 equals v15's file; `augment`, names, w_gru, f_dead, snap, stages, settings, post data and threshold hash equal to v15's; the `ens4` mirror's training-copy P is the mean of the four seeds' (0 difference) |
| **the port with fold models** (`check_v16 port`: J03WR9 stressed, metrica1 50 %, J03WOH 25 %) | float32: the averaged GRU P and the start P **equal the lab's**, **0 frames** differing on all 3; float64 (as deployed): 3-4 frames of the averaged GRU P one rounding step (0.001) off, **0 frames** differing, the same scores (v15's port had 1 frame in float64) |
| **the built file on SkillCorner's 20** (`check_v16 skillcorner`, float64) | **0.4090**, every match within 5e-6 of the lab's `netmet5` run (`lab_netmet/ens4/skillcorner_1015.csv`); the switch picks B on 20/20 |
| the post stages' training data (`train_v16 tables`) | trial 1015's out-of-fold stage tables rebuilt in memory from `netmet5`'s P on the 10 training-stress copies: **identical** to the lab mirror's `tables_trial1015.joblib` (short, segments, timing), the tables the file's post stages are trained on |
| the switch's check with this B (`check_v16 switch`, information) | the night's `ens4/switch/report_1015.json` reproduced exactly |

Results: `bench/results/check_v16/` (`port.csv`, `skillcorner_float64.csv`, `switch/report_1015.json`),
`bench/results/train_v16/tables_netmet5.json`.

## Our match

Run 2026-11-15 by Claude under Jose's rule (only the path, the SHA-256 and the format check's pass line printed; the
validator's output piped straight into `grep '^OK '`; the file and its counts not looked at; no `--show-branch`; the
switch not overridden): **three runs give the same file**, SHA-256
**`90599088419d812cbb341168b27ff0ff767549b75c8dfa12a51de290ccc0523c`**, and the format check passes (`OK v16.csv`).

| run | threads | SHA-256 | format check |
|---|---|---|---|
| v16, 1 | `slot.py`'s default (`OMP_NUM_THREADS=2`) | `90599088…0523c` | OK |
| v16, 2 | `slot.py`'s default | `90599088…0523c` | OK |
| v16, 3 | `OMP_NUM_THREADS=1` | `90599088…0523c` | OK |
| v15, rerun | `slot.py`'s default | `209a2081…7f5e`, **unchanged** from 2026-11-14 | OK |

(On this shared machine every heavy job runs through `slot.py`, so "default threads" means its 2; the GRUs run on
torch's 2 threads in every run, as in v13-v15.) **The file differs from v15's**, as it should: v16 changes only
branch B, and v15's own page already showed the switch takes B on our match. Nothing else was run on it.

## Failure modes

1. **Trained on Metrica's sample games, which ship with no licence file.** Using them to train a deployed network
   is Jose's decision of 2026-11-15 ([push.md](../lab/push.md); until then they were for checking and choosing
   settings only, 2026-10-30). Metrica publishes them and runs the challenge, but there is no written permission;
   [limitations.md](../limitations.md) row D and G8 apply now to the network our match runs, not only to v4, v6, v7
   and v9. The technical document should say so.
2. **The gain is mostly Metrica's, on a provider the nets trained on.** Of +0.0139, +0.0120 is Metrica's 3. Our match
   is Metrica's tracker under Metrica's convention, which is the case for it; but DFL's 7 (+0.0027) are within the
   seed noise, and SkillCorner is level (+0.0004). If our match's tracking behaves unlike Metrica's sample games, the
   honest expectation is "about v15".
3. **It fails the B bar's "7 of 10"** (6/10) and passes on Metrica's rule, the precedent of v14. `netmet5` was
   registered at 03:37, after `netmet`'s seed 1 had failed the noise rule (said on the night's page); its K = 4 was set
   by the clock.
4. **J03WOH and J03WPY lose with every seed** (−0.012 each in `netmet5`; −0.008 to −0.023 per seed). Something about
   these two DFL matches gets worse when Metrica's games enter training; we don't know what.
5. **Seed sensitivity.** Single `netmet` seeds give B's bench +0.0069 to +0.0141 (DFL's 7 from −0.0025 to +0.0047).
   Averaging four removes most of it (+0.0139), but a different set of four seeds would land somewhere in that
   range; trial 1015's settings were tuned on `gru_b` seed 0, not on this network (not re-searched: forking paths).
6. **Four networks in the model file** (3.4 MB against v15's 1.9 MB) and four float64 GRU passes on our match:
   slower, and four times the places a CPU's arithmetic can move a rounding (the port: 3-4 frames of the averaged P
   one step off in float64, no letter changed).
7. **The switch's in-between zone** (10-25 % of stoppages stressed) keeps A where this B is now better by more
   (−0.028); on our match the switch takes B, so this concerns other matches. B now also beats A on the normal bench
   (+0.015, 7/10), but that comparison favours B on Metrica (B trained on Metrica's other games, A never did).
8. **v15's failure modes apply** ([v15](v15-branch-b.md)): B's bench is a synthetic ball; the post stages learned
   from stressed copies.

## Run

```
OMP_NUM_THREADS=2 .venv/bin/python -m bench.train_v16 nets --seeds 0 1   # ~19 min a seed; with 2 3 in another slot
OMP_NUM_THREADS=2 .venv/bin/python -m bench.train_v16                    # the file (nets kept in data/external/train_v16/)
.venv/bin/python -m bench.check_v16 port                                 # ~4 min
.venv/bin/python -m bench.check_v16 skillcorner                          # ~17 min (four float64 GRUs a match)
.venv/bin/python -m bench.check_v16 switch                               # seconds (the lab's files)
.venv/bin/python -m bench.train_v16 tables                               # optional: the stage tables rebuilt from netmet5's P
.venv/bin/python -m possession.v16                                       # our match: path, SHA-256 (and the format check's line)
```

`train_v16 --recipe R` takes another network recipe (`RECIPES` in `bench/train_v16.py`: its seeds, how its `all`
net trains, the lab's nets to match bit for bit, its lab mirror for the post stages, its fold nets for the port), so
track 2's recipe, if it wins, is one more entry; `possession/v16.py` reads whatever networks the file holds.

## Log (machine clock, UTC−5)

- 10:21 started: read the push plan, v15's page and code, the night's network page, its audit and code.
- 10:25 `possession/v16.py`, `bench/train_v16.py` written; the four deployment nets (`netmet`'s `all` net, seeds
  0-3) training in 2 slots (seeds 0, 1 in one; 2, 3 in the other), each to be compared bit for bit with the night's.
- 10:28 `bench/check_v16.py switch` (the lab's files only): the night's switch report reproduced exactly.
- 10:44 seeds 0 and 2 trained (~19 min each, two at a time): **bit-identical** to the night's `net_all.pt` of
  `main` and `seed2`. Seeds 1 and 3 training.
- 11:03 seeds 1 and 3: **bit-identical** to the night's `seed1` and `seed3` nets (~18 min each). All four
  deployment nets reproduce the lab's. The build, then the port and SkillCorner checks, queued behind the other
  tracks' jobs (4 of 4 slots busy).
- 11:10 built `possession/models/v16_b_all.joblib` (3.37 MB, four networks). Trial 1015 read from `search_b` equals
  v15's file; the `ens4` mirror's training-copy P is the mean of the four seeds' (max |difference| 0); `augment`, the
  feature names, w_gru, f_dead, snap, the stages and their settings, the post data and the threshold hash equal to
  v15's file. The port check running; SkillCorner waiting for a slot.
- 11:14 **the port passes**: on J03WR9 stressed, metrica1 50 % and J03WOH 25 %, float32 gives the lab's start P
  exactly and **0 frames** differing; float64 (as deployed) puts P one rounding step (0.001) off on some frames and
  still **0 frames** differ; the lab's letters score as its confirm CSV on all 3. (One diagnostic column, the
  averaged GRU P against the lab's, compared a float64 mean with the lab's float32-stored one; fixed to compare
  through float32, rerun queued.) SkillCorner running.
- 11:31 **SkillCorner passes**: the built file (four float64 GRUs) gives **0.4090** on the 20 matches, each match
  within 5e-6 of the lab's `netmet5` run (largest 4.5e-6: the CSV's 5 decimals); the switch picks B on 20/20. The
  port's rerun with the fixed column: the averaged GRU P is the lab's exactly in float32 (0 frames), 3-4 frames one
  step off in float64; letters 0 frames differing either way.
- 11:32 our match: v15's file backed up (APFS clone in the scratchpad), then v16 three times in a row (default
  threads under `slot.py` twice, `OMP_NUM_THREADS=1` once) and v15 once, queued behind the other tracks' slots.
- 11:49 our match done: v16's three runs agree (`90599088…0523c`), format check OK; v15's rerun gives its
  2026-11-14 SHA-256 (`209a2081…7f5e`). `possession/v15.py`, `possession/models/v15_b_all.joblib`,
  `bench/train_v15.py`, `bench/check_v15.py` hash as before this track started. `train_v16 tables` (optional check)
  queued.
- 11:51 `train_v16 tables`: the three stage tables rebuild **identically** from `netmet5`'s P (~40 s). Its first
  run (11:50) reported `short` different: my comparison took every table for a DataFrame, and short's is a
  (DataFrame, labels) tuple; fixed, rerun, identical. **Track 1 built and checked.** Nothing committed; v16 is not the
  upload until Jose says so.
