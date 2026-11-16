# Night, builder "output": `pstage`, branch B's post stages given B's own start P

| | |
|---|---|
| **status** | **fails the bar** (2026-11-15 01:10): B's bench **+0.0011** (0.7339 against 1015's 0.7329), better on 8/10, worst variant +0.0007, IDSSE +0.0008, Metrica +0.0016; SkillCorner 0.4051 (**−0.0035**, 3/20, inside the 0.005 allowance). Short of +0.005 by 0.0039; the best of the 13 options, picked in hindsight, is +0.0019. Nothing adopted; nothing for Jose to decide. Plan written 23:50, before any score |
| **question** | B's learned post stages never see the start P that B decodes (0.915 × `gru_b` + 0.085 × `augment`). If `timing` (B's 3rd stage) and `recall` (never in B, added as a 4th stage after `timing`) get a few numbers from that P as extra ranker columns, does B get better on B's bench, by the night's bar? |
| **proposal** | [ideas_output.md](ideas_output.md), idea 1 (in-sample screens there: `timing` with P + per-change context +0.0028 against B's own +0.0018; `recall` with P +0.0017 to +0.0026; expected +0.002 to +0.004, ~15 % to clear the bar) |
| **code** | `bench/lab_pstage.py` (new file; no existing file edited; it refuses to write anywhere but its own two folders) |
| **outputs** | `data/external/lab_pstage/` (tables, chains), `bench/results/lab_pstage/` (unit CSVs, report JSON); logs in the scratchpad's `logs/pstage/` |
| **our match** | never read, nothing run on it |

## What stays fixed: trial 1015, v15's B, as `search_b confirm` scores it

Start P = `search_b.start_p(m, data, 0.91518…, 0.90946…)`: 0.915 × `gru_b` + 0.085 × `augment`, P(dead) × 0.909,
renormalised, clipped and rounded, exactly what B's decoder reads. Decoder: switch prices 2.82 / 48.7, minimum 7
frames, v9's snap and v11's move off. Stages: `short` (p_min 0.340), `segments` (flip 0.910, merge 0.661), `timing`
(window 3 s, p_min 0.4495). **No setting of 1015 is searched again.** `short` and `segments` are left exactly as
in 1015.

**How every stage is trained (as trial 1015's `post = stressed`):** on the 10 matches' copies under `augment`'s
training stress (`search_b.train_copy`: SkillCorner's even matches' stoppage balls, never the test's templates).
The chain is built **out of fold** on those copies: each copy's letters after a stage come from that stage's model
trained on the other 9 copies, and the next stage's table is built on them (`search_b.oof_tables`, extended to the
4th stage). **A held-out match's stage models are trained on the other 9 copies' tables, never on its own.** The P
columns on a copy come from that match's held-out P on its training copy (`data = "train"`, cached by `search_b
stress`); on a bench unit, from the variant's held-out P; on SkillCorner, from the deployment models' P (as
`search_b skillcorner`).

## What is added, to which stage

**`timing`** (B's own candidates, unchanged; its classifier, settings, window 3 s and p_min 0.4495 unchanged). Five
columns from the start P, at each candidate frame k of a change at g from team O to team N:

| column | what |
|---|---|
| `tp_new_k` | P(N) at k |
| `tp_gap_k` | P(N) − P(O) at k |
| `tp_slope_k` | that margin at k + 5 frames minus at k − 5 |
| `tp_gap_g` | the margin at g (where the change sits now) |
| `tp_cross_dk` | seconds from k to the frame nearest g (within ±3 s) where the margin crosses to N; 9 if none |
| **per-change context** | for each of timing's 42 `all` columns and these 5 (but `dk`, `adk`, `is_g`): the value minus the change's max and minus its mean over the change's candidates (88 columns); and the change's number of candidates |

**`recall`, a 4th stage after `timing`** (lab_recall as round 2 built it: its three generators, its 34 `all`
columns, `hgb`, its settings; never in B before). Six columns from the start P over each proposal [k1, k2) for team
N: mean P(N), mean P(O), mean P(dead), max P(N), the share of frames where N is the argmax, the length in frames.
A proposal is written in when its P reaches **p_min, chosen from {0.4, 0.5, 0.6, 0.7}** (below).

## The variants and how the one judged is chosen (fixed now)

| option | `timing` | `recall` (4th stage) | in the choice? |
|---|---|---|---|
| **(a)** | with P + context | none | yes |
| **(c)** | B's own | with P, p_min 0.7 / 0.6 / 0.5 / 0.4 | yes |
| **(d)** | with P + context | with P, p_min 0.7 / 0.6 / 0.5 / 0.4 | yes |
| (b) | B's own | without P (lab_recall as built), p_min 0.7 / 0.6 / 0.5 / 0.4 | **no**: the control that says how much of recall's effect is the P; reported only |

**`pstage` is one rule, not one setting.** For each held-out match m, each of the 9 options in the choice is scored
by the **mean spellF1 of the other 9 matches' training copies**, out of fold (each copy's letters from stage models
trained on the other 9 copies, as above); the option with the best mean wins (rounded to 6 decimals; ties to the
first in the order a, c 0.7 → 0.4, d 0.7 → 0.4, the fewest written first). **The choice never reads the held-out
match's copy and never reads a bench unit**: B's bench is only the judge, and the variants' bench totals are never
used to choose. For SkillCorner (deployment) the same rule runs on all 10 copies and the stage models are trained on
all 10 copies' tables. The second-order leak search_b accepted is here too: the other copies' out-of-fold models
saw the held-out match's training copy (its labels), said so, not removed.

`timing`'s p_min stays 1015's (0.4495), not re-chosen: the screen was flat over 0.30–0.50.

## The bar (copied from [night.md](../night.md), fixed before any score)

**The bar for every candidate tonight** (the round-8 B bar, against v15's B as confirmed, trial 1015's settings
fixed): on B's bench (each match's mean of p25, p50, stressed) **+0.005 or more** (0.7379), **better on 7 of 10**,
**no stress variant down more than 0.002**, **neither source (IDSSE, Metrica) down on the mean**, **SkillCorner down
no more than 0.005** (v15's B 0.4085, so 0.4035). **Noise rule:** if a second seed of the single network moves B's
bench by 0.003 or more, a single-network pass must repeat with a second seed.

How it applies here: the reference rows are v15's B as confirmed, trial 1015 in
`bench/results/search_b/confirm_1015_554_586_1133_1269_-1_-2.csv` (B's bench 0.7329; p25 0.7289, p50 0.7319,
stressed 0.7379) and `skillcorner_1015_554_586_1133_1269.csv` (0.4085). **The noise rule doesn't apply**: no
network is trained; the stage classifiers are deterministic (gradient boosting without early stopping or
subsampling, `random_state` 0; logistic regressions), so a rerun gives the same letters.

## The reproduction gates (before any new score)

| gate | what | needed |
|---|---|---|
| R1 | my out-of-fold chain of 1015 on the 10 training copies, built in my folder | every table equal to `search_b`'s `tables_trial1015.joblib` (read, never written) |
| R2 | 1015 through my code on B's bench | every one of the 30 units equal to the confirm CSV above to 1e-5 (0.7329) |
| R3 | 1015's column in the SkillCorner and normal-bench runs | equal to `skillcorner_1015_….csv` (0.4085) and `confirm_1015_-1_normal_p10.csv` to 1e-5; otherwise those runs' numbers don't count |

If R1 or R2 fails, the reason goes here and nothing else runs.

## Reported, never judged

- Every option (a), (b), (c), (d) at every p_min on B's bench, all 30 units (in sample: picking among them on these
  numbers would be the winner's curse, so they only explain the judged number).
- The normal bench (`normal`, `p10`; the switch sends none of the 10 there) with the same picks, for information.
- Per unit: moves `timing` makes, possessions `recall` writes; the training copies' out-of-fold scores behind each
  choice.

## The run plan

All through the scratchpad's `slot.py` (one slot, 2 threads), one job at a time.

| step | command (`.venv/bin/python -m bench.lab_pstage …`) | planned | measured |
|---|---|---|---|
| R1, R2 | `reproduce` | ~5 min | 115 s, 1.7 GB |
| the chains | `prepare` (the 4 variants' out-of-fold chains on the copies, the choices) | ~5–10 min | 80 s, 2.0 GB |
| the judge | `confirm` (B's 3 variants, 30 units) | ~10 min | 140 s, 0.9 GB |
| SkillCorner | `skillcorner` (20 matches, deployment models) | ~10 min | 354 s, 1.6 GB |
| information | `confirm --variants normal p10` | ~8 min | 110 s, 0.8 GB |

The last four ran in one slot in order (`logs/pstage/run_all.sh`); the slot wait was ~1 h (B's no-change rebuild
and the network builder held them). `skillcorner-report` re-judges SkillCorner from `skillcorner.csv` without a
model run (added after the run: the first check compared ids as numbers to strings and reported a false mismatch).

## Results

### The reproduction (00:48, `reproduce`: 115 s, 1.7 GB peak)

| gate | result |
|---|---|
| **R1** | **passes**: all 30 tables (10 copies × `short`, `segments`, `timing`) equal to `tables_trial1015.joblib`, every value |
| **R2** | **passes**: all 30 units equal to the confirm CSV to 1e-5 (largest difference 0); B's bench **0.7329** |

No file was written outside `data/external/lab_pstage/` and `bench/results/lab_pstage/` (the guard, and a `find
-newer` over `data/external/` after the run).

### The choice (00:58, `prepare`: 80 s, 2.0 GB peak; training copies only, no bench unit read)

The out-of-fold chain on the 10 training copies, each option against 1015's chain on the same copies (the data the
choice reads; the choice per held-out match uses 9 of these 10 columns):

| option | mean of the 10 copies | against 1015 | better on | actions per copy |
|---|---|---|---|---|
| 1015 (B as it deploys) | 0.7346 | | | |
| (a) timing+P | 0.7348 | +0.0002 | 4/10 | 108 moves |
| (c) recall+P at 0.7 / 0.6 / 0.5 / 0.4 | | +0.0009 / +0.0019 / +0.0013 / −0.0002 | 5 / 5 / 5 / 4 | 1.2 / 2.8 / 5.8 / 10.5 written |
| **(d) timing+P + recall+P** at 0.7 / **0.6** / 0.5 / 0.4 | 0.7356 / **0.7371** / 0.7361 / 0.7351 | +0.0010 / **+0.0024** / +0.0015 / +0.0005 | 6 / **7** / 6 / 5 | 1.3 / 2.9 / 6.0 / 9.6 |
| (b) recall without P (control, not in the choice) at 0.7 / 0.6 / 0.5 / 0.4 | | +0.0003 / +0.0016 / +0.0009 / +0.0028 | 1 / 5 / 5 / 6 | 0.4 / 1.9 / 4.6 / 7.3 |

**Picked (leave one match out on the copies): (d) at p_min 0.6 for 9 of the 10 held-out matches** (J03WN1: (c) at
0.6); **for deployment (all 10): (d) at 0.6.** Already here the size is small: +0.0024 on the copies, with recall
writing about 3 possessions a match, and `timing`'s P columns adding +0.0002. The control without P is as good on
the copies (+0.0028 at 0.4): on this data the P isn't what makes `recall` work.

### The judge: B's bench (01:00, `confirm`: 140 s, 0.9 GB peak)

1015 reproduced on all 30 units in this run too (largest difference 0). `pstage` = each match with its option
picked on the other 9 copies.

| match | v15's B (1015) | `pstage` | gain | option |
|---|---|---|---|---|
| J03WMX | 0.6254 | 0.6255 | +0.0001 | (d) 0.6 |
| J03WN1 | 0.6691 | 0.6704 | +0.0013 | (c) 0.6 |
| J03WOH | 0.7199 | 0.7212 | +0.0013 | (d) 0.6 |
| J03WOY | 0.6942 | 0.6978 | +0.0036 | (d) 0.6 |
| J03WPY | 0.7589 | 0.7601 | +0.0012 | (d) 0.6 |
| J03WQQ | 0.6939 | 0.6935 | −0.0004 | (d) 0.6 |
| J03WR9 | 0.7128 | 0.7115 | −0.0014 | (d) 0.6 |
| metrica1 | 0.8346 | 0.8357 | +0.0011 | (d) 0.6 |
| metrica2 | 0.7965 | 0.7998 | +0.0033 | (d) 0.6 |
| metrica3 | 0.8233 | 0.8238 | +0.0005 | (d) 0.6 |
| **B's bench** | **0.7329** | **0.7339** | **+0.0011, better on 8/10** | |
| p25 / p50 / stressed | 0.7289 / 0.7319 / 0.7379 | 0.7299 / 0.7332 / 0.7386 | +0.0011 / +0.0014 / +0.0007 | |
| IDSSE's 7 / Metrica's 3 | 0.6963 / 0.8181 | 0.6971 / 0.8198 | +0.0008 / +0.0016 | |

### SkillCorner (01:06, `skillcorner`: 354 s, 1.6 GB peak; deployment models, option (d) at 0.6)

1015 reproduced on all 20 matches (largest difference 5e-6, the reference CSV's 5-decimal rounding).
**`pstage` 0.4051 against 1015's 0.4085: −0.0035, better on 3/20** (allowance −0.005: inside it, by 0.0015).

### The bar

| part | needed | `pstage` | |
|---|---|---|---|
| B's bench, mean gain | +0.005 or more | **+0.0011** | **fails (short by 0.0039)** |
| better on | 7 of 10 | 8 of 10 | passes |
| no variant down more than 0.002 | | worst +0.0007 (stressed) | passes |
| neither source down | | IDSSE +0.0008, Metrica +0.0016 | passes |
| SkillCorner | down no more than 0.005 (≥ 0.4035) | 0.4051 (−0.0035) | passes, barely |
| noise rule | | doesn't apply (no network trained; deterministic stages) | |

**`pstage` fails the bar: +0.0011 on B's bench against the +0.005 needed.** It is a small, consistent gain (8 of 10
matches, every variant, both sources up) that costs 0.0035 on SkillCorner (17 of 20 down).

### Information (never judged): every option in sample, the normal bench

Every option on all 30 units of B's bench, against 1015 (choosing among these numbers would be the winner's curse;
they show the judged number isn't a bad pick):

| option | B's bench | gain | better on | actions per unit | SkillCorner (20) |
|---|---|---|---|---|---|
| 1015 | 0.7329 | | | 105 timing moves | 0.4085 |
| (a) timing+P | 0.7326 | −0.0003 | 6/10 | 110 timing moves | 0.4076 (−0.0009) |
| (c) recall+P at 0.7 / 0.6 / 0.5 / 0.4 | 0.7347 / 0.7348 / 0.7332 / 0.7327 | +0.0019 / +0.0019 / +0.0003 / −0.0001 | 6 / 7 / 4 / 4 | 1.4 / 3.1 / 5.9 / 10.3 written | 0.4071 / 0.4060 / 0.4040 / 0.3995 |
| (d) both at 0.7 / 0.6 / 0.5 / 0.4 | 0.7340 / 0.7346 / 0.7346 / 0.7325 | +0.0011 / +0.0017 / +0.0018 / −0.0003 | 8 / 8 / 7 / 6 | 1.5 / 2.7 / 5.9 / 9.6 | 0.4059 / 0.4051 / 0.4020 / 0.3995 |
| (b) recall without P at 0.7 / 0.6 / 0.5 / 0.4 | 0.7337 / 0.7345 / 0.7341 / 0.7343 | +0.0009 / +0.0017 / +0.0012 / +0.0014 | 3 / 6 / 6 / 5 | 0.5 / 2.1 / 4.9 / 7.4 | 0.4079 / 0.4071 / 0.4055 / 0.4019 |

**The best of the 13 options in hindsight is +0.0019**: no choice rule could have reached +0.005 with these columns.

The normal bench (`normal`, `p10`; 20 units; 1015 reproduced on all 20, largest difference 0), same picks:
**+0.0012, better on 6/10** (normal +0.0012, p10 +0.0012; IDSSE +0.0001, Metrica +0.0037).

## Why it fails

1. **`timing` gains nothing from the P.** With the five P columns and the per-change context, it is −0.0003 on
   B's bench (6/10) and +0.0002 on the training copies, against the proposal's screen of +0.0028 vs +0.0018. The
   screen trained on the same variant's letters and chose p_min in sample; trained as B deploys (on the training
   copies, 1015's p_min), the P adds nothing B's own columns (PathCRF, the decoder's costs) don't already carry. Not
   a shift in the P: on live frames the start P scores the same on the training copies as on the stressed units
   (log loss 0.202 against 0.202; p25 0.225, p50 0.217; accuracy 0.935 against 0.935).
2. **`recall` is worth +0.001 to +0.002, with or without the P.** It writes about 3 possessions a unit at p_min
   0.6, against the ~98 true possessions a unit B misses; written more freely (0.4) it loses. The P columns don't
   change that (+0.0019 with them at 0.6, +0.0017 without). This is the proposal's own anatomy: on 97 % of the
   missed possessions the start P is below 0.5 for the right team, so P can't tell the true proposals from the
   false ones; the evidence isn't in it.
3. **On SkillCorner every recall setting loses** (−0.0006 to −0.0090), more as it writes more: what it writes
   under the bench's convention isn't a possession under SkillCorner's. That is the risk the proposal named for
   our match too.

**What it would take:** the ranker headroom the proposal measured is real (+0.021 for `timing`, +0.035 for
`recall` with a perfect ranker over the same candidates), but the start P doesn't open it. Only new evidence of who
is on the ball in short contested moments can (the stacking section below).

## Stacking with the other builders (what the stages would need; nothing built for it tonight)

`pstage` reads only one thing besides 1015's chain: **the start P B decodes**, on three kinds of data. Its P
columns are computed on the fly from `search_b.start_p` (bench units, training copies) and from the deployment
models (SkillCorner), so it stacks with any builder whose output is a new start P, provided these exist:

| what | where `pstage` reads it | network builder (`netens`, `netmet`, `netown`) | ghost builder (only if B is rebuilt) |
|---|---|---|---|
| the GRU's held-out P on every bench variant (p25, p50, stressed; normal, p10 for information) | `lab_branch_b.gru_b` (`lab_branch_b.CACHE`) | in its mirror, `data/external/lab_<name>/<sub>/lab_branch_b/` | the rebuilt copy's `lab_branch_b/` |
| the GRU's held-out P on each match's **training-stress copy** (the stages train on these; the P columns there must be held out too) | `search_b.train_p_path("gru_b", m)` | in its mirror, `…/search_b/p_gru_b_train_<m>.npy` | the copy's `search_b/` (and `augment`'s, which ghost changes too) |
| the deployment network(s) for SkillCorner | `lab_gate.gru_model(…, B.net_path(metrica1))` | the `metrica1` fold net(s) of each seed (ensemble: P averaged as `lab_netmirror.average`) | the copy's nets and `augment_all.joblib` |
| the new 1015 as confirmed (the reference rows and the R1 tables) | `REF_TABLES`, `REF_CONFIRM`, `REF_SC` | the mirror's own confirm / SkillCorner CSVs and `tables_trial1015.joblib` | the copy's |

So stacking is a run of `lab_pstage` **after** `lab_netmirror.use_named(name, sub)` has pointed the two caches at the
mirror (the module reads them through `search_b`'s and `lab_branch_b`'s module attributes, so the patch reaches it),
with three small changes: its own `CACHE` / `RESULTS` under the mirror's folders, the reference paths taken from the
mirror (not v15's), and the SkillCorner GRU replaced by the mirror's deployment net(s). The bar for a product
would then be against v15's B **and** against the network candidate alone (the product must beat its better
parent, as network.md's rule for two passers).

**A new signal as ranker columns** (the proposal's idea 3, not built here): the ghost builder's per-frame columns
(`gd_X`, `gt_X`, `gn_X_10m`, ghost touches) would enter `timing` and `recall` the way the P does, if they exist as
per-frame arrays (frames × home/away, NaN outside the halves) **on the 10 training-stress copies, the 30 + 20 bench
units and SkillCorner's 20**, from prediction-time inputs only (the damaged, blanked tracking each unit carries),
keyed by match id and variant, in a cache of their own. `timing` would read them at the candidate frame and its
±0.5 s; `recall` would read their mean, max and count over the proposal.
