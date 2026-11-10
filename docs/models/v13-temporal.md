# Model v13: v12 with the lab's temporal start, and a stoppage switch

| | |
|---|---|
| **status** | **built 2026-11-09, Jose's go-ahead** (v13 plus the switch). The port reproduces the lab on every frame of two bench matches (float32 and float64); the deployment GRU is bit-identical to the lab's network trained on the same matches; the switch reproduces the lab's signal and branch B on SkillCorner's 20 matches (0 frames differ) |
| **what** | v12's chain with its start replaced: the mean of a small bidirectional GRU's P and robust's P, decoded, v9's snap and v11's move, then v12's post stages: **average → short → segments → timing** (bench set `all`), or **average → short → segments → recall → dead** (`idsse`). **Before it, a switch** (`all` only): where the tracking's ball is seen and moving through what the players say are stoppages, it runs branch B instead, **augment → short → segments → timing** |
| **why** | [lab round 7](../lab.md): `average` is the first start since robust whose gain survives v12's chain, and it passes every check on both sets ([temporal](../lab/temporal.md)). The switch guards against the failure SkillCorner exposed ([skillcorner](../lab/skillcorner.md), [carried](../lab/carried.md)) and passed its own four-part bar ([gate](../lab/gate.md)) |
| **bench** | **0.7162 held out against v12's 0.7033 (+0.0129), better on 8/10**; IDSSE set 0.6760 against 0.6699 (+0.0061), 5/7. Against v11: +0.1435 and +0.1135. The switch picks branch A on every normal bench match, so it changes none of this |
| **SkillCorner** (20 broadcast matches) | **0.408 with the switch** (branch B on 20/20), against 0.312 without it, v12's 0.308 and v11's 0.366 (`bench/check_v13_switch.py`, v13's deployment models) |
| **new code** | `possession/v13.py`, `possession/models/v13_all.joblib` and `v13_idsse.joblib` (1.8 and 2.0 MB), `v13_augment_all.joblib` (branch B, 1.4 MB), `bench/train_v13.py` (trains them), `bench/check_v13.py` (the port reproduces the lab), `bench/check_v13_switch.py` (the switch reproduces the lab); the lab's code it reuses: `bench/lab_temporal.py` (the network, its inputs and reading), `bench/lab_gate.py` (the signal and its threshold), `bench/lab_carried.py` (branch B's training), `bench/lab_robust.py`, the post stages' `bench/lab_*.py`, `possession/v12.py` (inputs, robust, kick-off) |
| **run** | `OMP_NUM_THREADS=2 python -m bench.train_v13 [--set idsse]` (~6 min each, 1.2 GB), `python -m bench.train_v13 --switch` (bundles branch B from `lab_carried`'s models), `OMP_NUM_THREADS=2 python -m bench.check_v13` (~11 min, 3 GB peak; `--set idsse` ~10 min), `python -m bench.check_v13_switch [--with-a]`, then `python -m possession.v13` on our match: it prints only the path, the SHA-256 and the format check (`--show-branch` adds the branch, for Jose) |

## What it does, stage by stage

Only the start is new; everything after it is v12's ([v12](v12-lab-chain.md) has each stage's page).

| stage | what it does | settings (`all`; `idsse` where different) | page |
|---|---|---|---|
| **inputs** | v12's, unchanged (`v12.inputs`): keeper repair, v1's clues without Smart Tagging, the dead-ball and restart models, PathCRF in float64; v6's and v7's costs, P(dead), P(home restarts), the restart clue, PathCRF's P(home) | v10's `SETTINGS[BENCH_SET]` | [v12](v12-lab-chain.md) |
| **robust** | v12's classifier, unchanged: P(home / away / dead) per frame from 97 features, any seen ball still for ≥ 1 s blanked | v12's deployment model (the 7 IDSSE matches) | [robust](../lab/robust.md) |
| **GRU** (new) | the same 97 blanked features, standardised (the training matches' mean and sd, clipped at ±5 sd, missing → 0 plus 44 missing flags), read every 5th frame by a 2-layer bidirectional GRU (64 units each way), each of the 5 phases its own sequence, in 60 s windows every 30 s with triangular weights; P is the mean of the match as is and with the teams exchanged, clipped to 0.01–0.99, rounded to 3 decimals. **Run in float64** | trained on the 7 IDSSE matches of the set, seed 0, 20 passes | [temporal](../lab/temporal.md) |
| **average** (new) | P = round(clip((P_GRU + P_robust) / 2)); −log P is the decoder's cost | | [temporal](../lab/temporal.md) |
| decoder, snap, move | the min-length decoder, then v9's restart snap and v11's touch move | team switch 2, dead switch 25 (`idsse`: 50), minimum possession 10 frames (robust's were 4, 25, 10: the GRU's P changes team less often, so the decoder charges less) | [v9](v9-restart-pass.md), [v11](v11-touch-line.md) |
| **short** | short stretches of the other team inside a possession, from touch evidence | p_min 0.5 | [short](../lab/short.md) |
| **segments** | each possession merged into the stoppage or its team flipped, by a classifier | flip 0.6, merge 0.7 (`idsse`: no PathCRF features, flip 0.8) | [segments](../lab/segments.md) |
| **timing** (`all`) | each change of team moved to the candidate frame within 3 s a ranker trusts most | p_min 0.35 | [timing](../lab/timing.md) |
| **recall** (`idsse`) | short stretches for the other team written in by a classifier | p_min 0.4 | [recall](../lab/recall.md) |
| **dead** (`idsse`) | stretches a dead-ball classifier is sure about become D | threshold 0.3, at least 4 s | [dead](../lab/dead.md) |
| kick-off | the second half's graded frames before its kick-off tag are D (`v12.before_kickoff`; FAQ) | | [v12](v12-lab-chain.md) |

The post stages' settings are v12's, chosen on robust's letters (`train_v12.chosen`), as `lab_temporal.chains`
judged them. They were not chosen again on the new start's letters (see failure mode 4).

## The bench (leave one match out)

From the lab's own runs (`bench/results/lab_temporal_average[_idsse].json`, `lab_temporal_chains[_idsse].json`;
the chain's per-match scores recomputed by `bench/check_v13.py`, below).

| | `all`: held out | against | matches better | `idsse` | against |
|---|---|---|---|---|---|
| the start alone (vs robust 0.6728; 0.6365) | 0.7030 | +0.0301 | 10/10 | 0.6635 | +0.0271, 7/7 |
| **the chain (vs v12 0.7033; 0.6699)** | **0.7162** | **+0.0129** | **8/10** | **0.6760** | **+0.0061, 5/7** |
| the chain against v11 (0.5727; 0.5625) | | +0.1435 | 10/10 | | +0.1135, 7/7 |

By source (`all`, chain against v12): IDSSE 0.6765 (+0.0090, 5/7), Metrica 0.8088 (+0.0219, 3/3), the 4
matches PathCRF never saw 0.7804 (+0.0188, 4/4). macroF1 0.9222 against 0.9182 (`idsse` 0.9194 against 0.9162).

Per match, chain against v12:

| | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | J03WQQ | J03WR9 | metrica1 | metrica2 | metrica3 |
|---|---|---|---|---|---|---|---|---|---|---|
| `all` | +0.027 | +0.021 | +0.005 | **−0.007** | −0.001 | +0.009 | +0.010 | +0.027 | +0.009 | +0.030 |
| `idsse` | +0.045 | −0.005 | +0.007 | **−0.024** | +0.003 | +0.016 | +0.000 | | | |

**With the settings chosen on all the set's matches** (what v13 runs; `bench/train_v13.py`, every model
out of fold), the chain scores **0.7195** on `all` (v12 the same way: 0.7036) and **0.6833** on `idsse`
(v12: 0.6688). Those are a little flattered: the settings saw every match. The held-out numbers above are
the ones to quote.

| `train_v13`, out of fold, chosen settings | start | + short | + segments | + timing / + recall, dead |
|---|---|---|---|---|
| `all` (v12 the same way) | 0.7063 (0.6747) | 0.7148 (0.6894) | 0.7171 (0.6988) | **0.7195** (0.7036) |
| `idsse` | 0.6649 | 0.6774 | 0.6809 | 0.6827, **0.6833** |

## Porting it to our match

1. **The deployment GRU** (`bench/train_v13.py`): `lab_temporal.fit` on the set's 7 IDSSE matches in
   `lab.matches` order, seed 0, 2 threads, deterministic algorithms. On `all` that is exactly the lab's
   metrica1 fold (Metrica's folds were trained on all 7 IDSSE matches), and the weights came out
   **bit-identical** to the lab's `net_metrica1.pt`, all 20 tensors, twice (a scratch fit and
   `train_v13`). The IDSSE set has no such fold to compare with (each of its folds left an IDSSE match
   out). The bundle keeps the weights as float32 arrays, the normaliser (mean, sd, the 44 flagged
   features), the 97 feature names and the constants the network's shape and reading depend on; v13
   rebuilds the network with `lab_temporal.network` and refuses a bundle whose names or constants differ
   from the code. Nothing outside the repo and the organisers' data is read at run time.
2. **Robust is v12's deployment classifier.** The lab averaged robust's held-out P; on Metrica's folds
   that came from a model trained on the same 7 IDSSE matches as v12's. Checked: v12's classifier gives
   the lab's held-out P on metrica1 and metrica2 exactly (after the lab's float32 save).
3. **The post stages**, trained once on every bench match of the set, each on the out-of-fold letters of
   the stages before it, as v12's: the start's letters are the lab's held-out `average` P decoded with the
   chosen setting, so what each stage learns from looks like what it will see on our match.
4. **`possession/v13.py`**: `predict(tracking, tags, halves, pitch, bench_set)` builds v12's inputs,
   wraps them like a lab match (`v12.as_match`), computes the start's P (`start_proba`) and runs the rest
   (`run_chain_from`), then the kick-off rule. The GRU's features are built as `v12.robust_proba` builds
   robust's (`blanked_features`): exactly what the lab's GRU read from `lab_robust.cached` (all 97
   columns; robust's `cols` for `blank` are all 97 too). `main` writes `evaluation/results/v13.csv` and
   prints only the path, the SHA-256 and the format check.
5. **One detail of the average.** The lab averaged the two P it had saved as float32. Half the means of
   two 3-decimal numbers fall exactly on a 0.0005 tie, and the float32 round trip decides which way those
   round. `v13.average` does the same round trip, so the ties break as the lab's did (deterministic on any
   CPU: it's a cast of a 3-decimal number). Without it, 32,710 of J03WR9's 440,433 P entries (28,693 of
   metrica1's 435,021) would be 0.001 off the lab's, and 2 (3) letters would differ
   (on the IDSSE set: 33,430 entries and 0 letters on J03WR9, 32,004 and 1 on J03WMX).

### Functions

| function | what |
|---|---|
| `predict(tracking, tags=None, halves=None, pitch=..., bench_set=None)` | labels for every frame and `{"pred", "p_start", "branch"}` |
| `branch(fixed, pred, halves, pitch, which)` | `"B"` when `SWITCH` is on for the set and `lab_gate.signal` is above `lab_gate.THRESHOLD`, else `"A"` |
| `augment_proba(match, bundle, which, features=None)`, `load_branch_b(which)` | branch B's start P (the same 97 features, `augment`'s classifier); its bundle, refused if its threshold differs from `lab_gate.py`'s |
| `start_proba(match, models, which, dtype=None)` | the start's P(H, A, D): `average(gru_proba(...), v12.robust_proba(...))` |
| `run_chain_from(match, p, models, which, settings=None)` | any start's P → decoded with `settings["start"]`, snap, move, `post_stages` |
| `post_stages(match, letters, models, which, settings)` | v12's post stages of the set, as `v12.run_chain` runs them |
| `gru_proba` / `gru_raw(match, gru, which, dtype=None, features=None)` | the network's P, clipped and rounded / raw |
| `blanked_features(match, which)` | (names, X, X_swapped): robust's 97 features on the blanked tracking |
| `average(p_gru, p_robust)`, `clipped(p)` | the `average` rule; clip to 0.01–0.99 and round to 3 decimals |
| `gru_bundle(state, norm, names, **extra)`, `gru_network(gru, dtype=None)`, `run_half(...)` | the network saved and rebuilt; one half read (`lab_temporal.run_half` with a dtype) |
| `load_models(which)`, `main()` | |

`models` is the bundle: `{"robust", "gru", <post stages>, "settings": {"start", <post stages>}, "chain", "which"}`;
branch B's is `{"augment", "names", <post stages>, "settings", "chain", "which", "threshold"}`.

## Built: what the checks say

Run 2026-11-09 on this machine (macOS arm64, torch 2.15.0), 2 threads, another builder's jobs alongside;
`check_v13` on `all` was run twice and printed the same lines both times.

| check | `all` | `idsse` |
|---|---|---|
| **the deployment GRU against the lab's network on the same matches** | **bit-identical**: 20 of 20 tensors, largest difference 0, twice (final training loss 0.15861, the lab's log 0.1586) | no lab fold on the same 7 matches; trained (loss 0.15791) |
| robust: v12's classifier against the lab's held-out P on Metrica's folds | equal on metrica1 and metrica2 (after the lab's float32 save) | (no Metrica) |
| the lab's `average`-start chain, recomputed by `check_v13` | 0.7162, as recorded; every match within 5·10⁻⁶ (the record's rounding) | 0.6760, as recorded; every match within 5·10⁻⁶ |
| **the port, float32: letters differing from the lab's chain** | **0 of 146,811 (J03WR9), 0 of 145,007 (metrica1)** | **0 of 146,811 (J03WR9), 0 of 145,968 (J03WMX)** |
| … its pieces against the lab's | GRU's P 3·10⁻⁸, robust's 3·10⁻⁸ (the lab's float32 save), the start's P exactly equal, the start's letters 0 differ | the same: 3·10⁻⁸, 3·10⁻⁸, equal, 0 |
| **float64 against float32 (fold models)**: the GRU's raw P | largest 4.3·10⁻⁷ / 4.5·10⁻⁷ (median 2·10⁻⁹ / 1·10⁻⁹) | largest 3.8·10⁻⁷ / 4.0·10⁻⁷ (median 2·10⁻⁹) |
| … P entries changed after rounding: the GRU's / the start's | 5 / 5 of 440,433; 6 / 4 of 435,021 | 1 / 1 of 440,433; 4 / 0 of 437,904 |
| … **letters changed** | **0 / 0**: in float64 the port still gives the lab's letters on every frame | **0 / 0** |
| float64 against float32, deployment models (in sample) | J03WR9: largest 5.7·10⁻⁷, 1 GRU entry, 0 start entries, 0 letters; metrica1: 4.5·10⁻⁷, 6, 4, 0 letters | J03WR9: 5.0·10⁻⁷, 1, 0, 0 letters; J03WMX: 4.5·10⁻⁷, 2, 2, 0 letters |
| **`v13.predict` end to end on J03WR9's damaged tracking, twice** | **the same letters**; they agree with the deployment chain run on the lab's inputs on 99.1 % of graded frames (not 100 %: `v12.inputs` uses dead-ball and restart models trained on every bench match, the lab's are out of fold) | **the same letters**; 98.9 % agreement |
| run time, memory | `train_v13` 5.6 min, 1.2 GB; `check_v13` 11 min, 3.0 GB peak | `train_v13` 6.7 min, 0.8 GB; `check_v13 --set idsse` 10 min, 2.5 GB peak |

**Float64.** The GRU's float32 and float64 outputs differ by at most ~6·10⁻⁷, which moves a handful of
3-decimal roundings in ~440,000 and no letter on any of the matches tried. Between two CPUs running float64
the differences are ~10⁻¹⁶, so a rounding flip needs a value within 10⁻¹⁶ of a 0.0005 boundary: the file
should come out the same on another CPU ([reproduce.md](../reproduce.md)'s argument for PathCRF, which v13
inherits). Not tried on Linux.

**Our match** (2026-11-09, with the switch, run by Claude under Jose's rule: only the path, the SHA-256 and the
format check's pass line were printed; the file, its counts and the branch were not looked at):
`python -m possession.v13` twice gives the same file, SHA-256
`3efbef53e4394c318815fa33d8284af9ab6f675667c09008dfdba2e51a54a7c4`, and the format check passes (199,319 rows).

## Failure modes

1. **Branch A shares robust's stoppage weakness** ([skillcorner](../lab/skillcorner.md), [carried](../lab/carried.md)).
   The GRU reads robust's blanked features, so a still ball doesn't exist for it either. On SkillCorner branch A
   scores 0.312 (v12 0.308); on the stressed bench 0.598. The switch (below) is the guard: it sends such a match
   to branch B. What it can't guard against is a match in between (failure mode 7).
2. **Narrow on the IDSSE set**: +0.0061 against the +0.005 floor, 5 of 7 against the 5-of-7 floor,
   J03WR9 (the PathCRF-unseen match) +0.0002. And `average` is the better of two variants tried there,
   so it's slightly flattered (lab.md, "Many tries, one bench").
3. **J03WOY loses in every chain from it** (−0.007 on `all`, −0.024 on `idsse`) though its start is
   level: the post stages undo something there. Not looked into.
4. **The post stages' settings were chosen on robust's letters.** The GRU already does part of short's
   and segments' job (v12's stages add +0.013 on top of `average`, against +0.031 on top of robust; `idsse` +0.013 against +0.033); a greedy chain
   from this start might drop a stage. Not tried.
5. **v12's other failure modes apply unchanged** (its 2–7): the bench flatters it, PathCRF is in sample
   on 6 IDSSE matches, our inputs come from models trained on every bench match, Smart Tagging is off
   except the kick-off, Metrica's samples train the `all` chain, and only 41 % damage was tested.
6. **PyTorch at run time.** v12 already ran PathCRF in PyTorch; v13 adds the GRU. Version pinned
   (torch 2.15.0); float64 (measured above); the weights are in the bundle, so nothing is downloaded.
7. **The switch's in-between zone.** Its partly stressed checks passed narrowly (−0.0046 against −0.005 at
   25 %), both losses on Metrica's metrica1, and Metrica's samples cross later than IDSSE's. A match near the
   threshold may get the slightly worse branch; the cost measured there is ~0.005–0.015, against ~0.1 for the
   wrong branch at either end.
8. **Branch B trains on SkillCorner's stoppage balls** (MIT), only through `augment`'s stressed training
   copies; its post stages and settings are v12's. The switch's threshold was set on the bench's partly
   stressed copies, which use SkillCorner's own balls, so its SkillCorner result is a kind test
   ([gate](../lab/gate.md)).

## The switch

**Why.** The bench's dead ball is frozen (IDSSE) or missing (Metrica), so every start learned "dead ⇔ no
ball". On SkillCorner's broadcast tracking the ball stays tracked and moving through stoppages, and v12 fell
below v11. A fix that helps there costs on the normal bench (`augment`: −0.0095), so no single model serves
both. The switch lets the tracking say which world a match is in.

**The signal** (`bench/lab_gate.py`, `signal(track, pred, halves, pitch)`): from the repaired tracking and v4's
players-only P(dead) (`v12.inputs`), nothing else. Among frames where the players say dead (P ≥ 0.5, the last
2 s of each run left out: that's the restart), the share where the ball is seen, not blanked by robust's
still rule, and moving ≥ 0.3 m/s over 0.2 s; divided by the same share where the players say live (P < 0.2).
Frames inside the halves with a player tracked. Normal bench 0.02–0.10, stressed bench 0.36–0.63,
SkillCorner 0.25–0.49.

**The threshold**: `THRESHOLD = 0.1445`, fixed on the bench before SkillCorner's signal was computed: leave one
match out, the other matches' mean signal at the stress level where `augment`'s chain starts beating
`average`'s. Above it, branch B.

**Branch B** (`possession/models/v13_augment_all.joblib`, `python -m bench.train_v13 --switch`): `augment`,
robust's classifier retrained on the 7 IDSSE matches as they are and with SkillCorner-like balls in their
stoppages (`lab_carried.deployment_augment`), decoded with v12's robust setting (team 4, dead 25, minimum
10), snap and move, then v12's post stages refitted on its out-of-fold letters (`lab_carried.deploy_post`),
with v12's settings. Built for `all` only; on `idsse` v13 always runs branch A.

**What the lab measured** ([gate](../lab/gate.md); against branch A):

| | `all` | `idsse` |
|---|---|---|
| normal bench | A on 10/10: ±0 | A on 7/7: ±0 |
| stressed bench | B on 10/10: +0.108 (0.598 → 0.706) | B on 7/7: +0.057 |
| 10 / 25 / 50 % of stoppages stressed: gap to the better branch | −0.004 / −0.005 / 0 | 0 / −0.004 / 0 |
| SkillCorner | B on 20/20: +0.093, 20/20 | |

**The port** (`bench/check_v13_switch.py --with-a`, SkillCorner's 20 matches, v13's own code and committed
models): the signal equals the lab's to 5·10⁻⁶ on every match; branch B on 20/20; branch B's letters equal
`lab_carried`'s deployment path on every frame (0 differ); **v13 0.408 with the switch against 0.312 without
it, better on 20/20** (v12 0.308).

**What it can't know.** Which branch our match takes: nobody reads the signal on it (`--show-branch` is
Jose's, and the choice is never overridden by hand). The risky case is in between: a Metrica-tracked match
whose dead ball moves in about a quarter of its stoppages reads 0.14–0.18 and gets B, where on Metrica's
samples A was still ~0.01 better at that level. Metrica's crossing point is later than IDSSE's (signal ~0.18
against ~0.12).
