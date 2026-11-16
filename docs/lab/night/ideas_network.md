# Night ideas: the network at the heart of branch B

| | |
|---|---|
| **written** | 2026-11-14, 23:10 to 00:15, by the "network" idea agent. Nothing built; nothing adopted |
| **question** | `gru_b` carries most of round 8's gain (+0.028 to +0.030 of B's +0.035). What would most likely make it better tonight, judged on B's bench with the round-8 B bar? |
| **read** | [branch-b.md](../branch-b.md), [temporal.md](../temporal.md), [v13](../../models/v13-temporal.md), [v15](../../models/v15-branch-b.md), [lab.md](../../lab.md) rounds 7-9, [cleaning.md](../cleaning.md), [field.md](../../field.md), [owner-rule.md](../../owner-rule.md) (head), `bench/lab_temporal.py`, `bench/lab_branch_b.py`, `bench/search_b.py`, `bench/lab_gate.py`, `bench/lab_carried.py`, the round-8 logs in `data/external/search_b/` |
| **light analyses** | bench caches only (`lab_robust/<m>.npz` truth and features, `lab_branch_b/p_gru_b_*`, `lab_temporal/p_*`, `lab.load` tracking of 7 bench matches); under 10 s of one core each. Scripts: `/private/tmp/claude-501/-Users-joseruiz-Code-more-than-a-hack-2027/39e3effe-3bf8-4611-8584-0402cfeb00b7/scratchpad/net_ideas/` (`errors.py` … `pocket_team.py`). **Nothing about our match was read or run** |

## What the bench caches say about `gru_b` (frame level, held out)

| | finding | what it points to |
|---|---|---|
| F1 | **Error types by variant** (share of graded frames; normal / p25 / p50 / stressed): log loss 0.229 / 0.221 / 0.213 / **0.197**; dead→live 1.37 / 1.15 / 0.99 / **0.62 %**; live→dead 2.68 / 2.56 / 2.47 / 2.17 %; H↔A **3.61 / 3.63 / 3.63 / 3.70 %**. The fully stressed variant is the *easiest*: the network reads the pasted SkillCorner stoppage ball well (and on SkillCorner's real balls v15's B gains +0.0008) | dead/live is no longer the main error on B's bench; team errors are. More stress tuning has little left to win *on the bench* |
| F2 | **H↔A errors sit at the edges**: 41 % within 0.5 s of a true label change, 64 % within 1 s, 84 % within 2 s, 4 % beyond 5 s (live frames there: 11 / 21 / 36 / 38 %) | sharper edges (idea 4); smoother, better-averaged P (idea 1) |
| F3 | **Two GRUs averaged** (A's GRU from lab_temporal and `gru_b`, normal variant, both held out): log loss **0.2293 → 0.2120** (A's alone 0.2145); better than `gru_b` on **10/10** matches, better than both on 8/10; argmax disagreement 2.8 %; accuracy 92.3 → 92.7 % | averaging networks pays, even two that each learned something the other didn't (idea 1) |
| F4 | **Metrica's live→dead pocket**: live called dead on **4.8 %** of Metrica's graded frames (both variants) against 1.1-1.8 % on DFL's; **87-92 % of it in 11-15 runs of 3 s or more per game**, 50-56 % of Metrica's dead/live errors more than 10 s from any true dead edge. In those runs the players look stopped (v4's players-only P(dead) 0.69-0.90, speed 0.9-1.2 m/s, as in true dead time), the ball is mostly unseen, `gru_b` says P(dead) **0.95**, and the team under it is right only 66-75 % | a convention the network never saw (idea 2) |
| F5 | **The convention, label-free side**: where the players look stopped (v4 P(dead) > 0.7) and the ball is unseen, DFL labels **1.1-2.6 %** live, Metrica **5.5-15.7 %**; stretches of 3 s or more that are mostly live: DFL 0-4 a game, Metrica 5-11. `gru_b` learned its dead convention from DFL's 7 only; the answer key follows Metrica's owner field ([owner-rule.md](../../owner-rule.md)) | idea 2 |
| F6 | **A "ghost near the ball" signal** (label-free: a team's player whose piece ended in the last 10 s within 15 m of the ball's last-seen spot; 28 % of live frames): H↔A errors 5.3 % without a ghost, 6.1 % with the owner's, 6.5 % with the other team's, 8.1 % with both | it marks crowded moments, it doesn't point to a team: weak lever (idea 5) |
| F7 | **A network-only change rebuilds B fast.** The tracking, the trees and the stress don't change, so only the GRU's P (B's 3 variants, the training-stress copies), trial 1015's stage tables, the confirm and SkillCorner are redone. Round 8's logs: train-copy P ~15 s a match, trial tables ~25 s, confirm ~5 min, SkillCorner ~15 s a match. `gru_b`'s folds took **405-504 s each** at 2 threads (< 1.5 GB) | everything below fits tonight; training is the cost |

## The ranking

| rank | idea | guess on B's bench | chance it passes the B bar | carries to our match? | slot-minutes | judged after start (2 slots / 4 slots) |
|---|---|---|---|---|---|---|
| **1** | **`netens`: five seeds of `gru_b`, averaged** | **+0.004** (0 to +0.008) | ~35 % | **high** (variance reduction, no new inputs or labels) | ~320 | ~3 h / ~1.5 h |
| **2** | **`netmet`: Metrica's labels in `gru_b`'s training** | **+0.005** (−0.004 to +0.012); Metrica's 3 +0.01 to +0.03 | ~25 % (B bar); ~40 % (the Metrica rule, below) | the convention yes (the answer key's); Metrica's tracking quirks no | ~165 | ~2 h / ~1.3 h |
| 3 | `netdraw`: five members, each with its own training-stress draw | +0.003 | ~25 % | better than 1 on real balls, if anything | ~360 | ~3.5 h / ~1.7 h |
| 4 | `netedge`: boundary-aware training (two auxiliary heads, edge-weighted loss) | +0.002 | ~15 % | medium | ~120 | ~1.5 h / ~1 h |
| 5 | `netghost`: visibility inputs (visible and vanished players near the ball) | +0.001 | ~10 % | medium-low (SkillCorner can't check it) | ~200 + ~2 h of building | not tonight |

**Run order:** start 1 and 2 at once (they share nothing but the plumbing below); give 3 or 4 the slots that free
up. If 1 and 2 both pass, their product (five seeds of `netmet`, 44 more trainings) is the morning's follow-up.
Idea 1 also measures **seed noise**, which every single-network comparison tonight (2, 4) must be read against.

## Shared plumbing: rebuilding B for a new network (all ideas)

What changes is only the file behind `lab_branch_b.gru_b()` and `search_b.p_of("gru_b", m, "train")`. So:

1. **An isolated mirror**, written only under the idea's own folder: `data/external/lab_<name>/b/lab_branch_b/`
   (`p_gru_b_<variant>_<m>.npy` for p25, p50, stressed, and normal, p10 for the switch's table) and
   `data/external/lab_<name>/b/search_b/` (`p_gru_b_train_<m>.npy`; **copies** of the GRU-independent
   `p_augment_train_*.npy` and `tables_trial0.joblib`). The new script sets `lab_branch_b.CACHE`,
   `search_b.CACHE` and `search_b.RESULTS` (to `bench/results/lab_<name>/`) **before** calling `search_b.confirm`;
   every other cache is only read (all of them exist: `lab_gate`, `lab_carried`, `lab_robust`, the stage-model
   inputs). Nothing is written into a shared cache.
2. **Plumbing check, before any new P is scored:** the mirror filled with `gru_b`'s own P
   (`lab_branch_b/p_gru_b_*`, `search_b/p_gru_b_train_*`) must reproduce trial 1015 on all 30 units of
   `bench/results/search_b/confirm_1015_554_586_1133_1269_-1_-2.csv` to 1e-5 (and trial 0 at 0.6975). If not, stop.
3. **Then the new P**: `search_b.confirm([1015])` in the mirror (~5-10 min, tables included), the same with
   `variants=("normal", "p10")` for the switch's table (information), and SkillCorner: a copy of
   `search_b.skillcorner` whose GRU P is the idea's deployment network(s) on `lab_carried.sc_prepared(m)`, trial
   1015's stage models from the mirror's tables (~6-10 min).
4. **Trial 1015's settings stay fixed** (w_gru 0.915, f_dead 0.909, switch prices 2.82 / 48.7, minimum 7, snap and
   move off, short 0.340, segments 0.910 / 0.661, timing 0.450, post stages trained on the stressed copies). They
   were picked on `gru_b`'s P, so this slightly handicaps a new network; no re-search tonight (forking paths).

**The bar, fixed now, for every idea: the round-8 B bar against v15's B** (trial 1015 as confirmed):

1. B's bench (each match's mean of p25, p50, stressed): **+0.005 or more** over v15's B (0.7329, so 0.7379 or
   more), **better on 7 or more of the 10** (per match from the confirm CSV above);
2. **no variant down more than 0.002** (v15's B: p25 0.7289, p50 0.7319, stressed 0.7379);
3. **SkillCorner** (20 matches, deployment models): **no loss beyond 0.005** (v15's B 0.4085, from
   `bench/results/search_b/skillcorner_1015_554_586_1133_1269.csv`).

The switch's table (normal, p10) is reported, not a gate: on our match the switch takes B whatever B's network is.

## 1. `netens`: five seeds of `gru_b`, averaged

**What changes.** B's GRU P becomes the mean of five networks with `gru_b`'s exact recipe, data and folds, seeds
0-4 (seed 0 is `gru_b` itself, its files reused): **P = round(clip(mean of the five P, 0.01, 0.99), 3)**, each
member's P as `lab_gate.gru_proba` gives it. The same definition on the bench variants, the training-stress copies
(so the post stages learn from the ensemble's letters) and SkillCorner (the five seeds' `net_metrica1.pt`, each
trained on the 7 DFL matches).

**Why it should help.** A 140k-weight network trained on 6-7 matches, with random chunk offsets, phases and
dropout, carries seed-specific quirks; averaging cancels them and smooths P in time, which is the very property
that made the GRU worth +0.03 ("its P doesn't flicker", round 7). Evidence: F3 (two GRUs averaged: log loss −7.5 %,
better on 10/10); round 8's own search put 8.5 % of `augment` into the start (diversity paid); the field's strongest
entries average 5-8 temporal networks (zka2038 5 BiGRUs, TEKAB2 5 TCNs, Barca-Hackathon 8 TCNs; [field.md](../../field.md)).
Five seeds are less diverse than F3's pair, so expect less than F3's −7.5 %.

**Guess:** +0.004 on B's bench (0 to +0.008), 7-9 matches of 10 better, SkillCorner within ±0.004. **~35 %** to
clear +0.005. The main way it loses: the ensemble's P is less extreme, and trial 1015's switch prices were set on
a single network's sharpness, so it may find fewer short possessions (report possessions per match).

**The test** (new files: `bench/lab_netens.py`, `docs/lab/netens.md`, `bench/results/lab_netens/`,
`data/external/lab_netens/`):

| step | command (through `slot.py`, `OMP_NUM_THREADS=2`) | jobs × time | slots |
|---|---|---|---|
| train | `python -m bench.lab_netens train --seed S --fold HELD` for S in 1-4 and the 8 folds of `lab_branch_b.held_list()`: sets `lab_temporal.SEED = S` (read at call time by `lab_branch_b.fit` for `torch.manual_seed` and the numpy rng), calls `lab_branch_b.fit(lab_branch_b.used_for(HELD))`, saves `seedS/net_HELD.pt` + json | 32 × 7-8.5 min (~240 slot-min) | 1 each, < 1.5 GB |
| predict | `predict --seed S` once a seed's 8 folds are done: each match's P with the net that never saw it (`lab_branch_b.net_for` logic) on p25, p50, stressed, normal, p10 (`lab_gate.variant_features`) and the training-stress copy (`lab_carried.stressed_features(m, "all", TRAIN_SEED)`) | 4 × ~15 min | 1 |
| combine | `combine`: the mean above into the mirror (seed 0 = the existing `gru_b` files) | ~2 min | 1 |
| check | `confirm --plumbing` (mirror with seed 0 alone; must reproduce trial 1015) | ~8 min | 1 |
| judge | `confirm` (B's 3 variants; then normal, p10), `skillcorner` (5 deployment nets averaged) | ~20 min | 1 |

**Reported, not judged:** each seed alone through trial 1015's start (decoded with 1015's settings, no post
stages) and its held-out log loss, so the **seed-to-seed spread** of `gru_b` is known; the ensemble's score with
2, 3 and 4 members. If one seed alone moves B's bench by ±0.005, round 8's single-network +0.0296 carried seed luck of
that size, and ideas 2 and 4 (single networks) must be read against that spread.

**K is fixed now at 5.** If the machine can't give ~240 slot-minutes, K = 3 (seeds 1-2), declared before any
ensemble score is seen.

**Carry risk: low.** No new inputs, labels or data; the same synthetic-ball caveat as round 8 (F1), checked on
SkillCorner. Deploying it means five networks in v16's file (~0.6 MB each) and five GRU passes on our match
(each must retrain bit-identically, as `train_v15` checked for one).

**Fits tonight:** yes; judged ~1.5 h after start on 4 slots, ~3 h on 2.

## 2. `netmet`: Metrica's labels in `gru_b`'s training

**What changes.** `gru_b` trains on **every** other bench match, Metrica's included: 10 folds (each match held out;
Metrica's 3 no longer share one network) plus one deployment network on all 10 for SkillCorner. Each training match
gives both of `gru_b`'s copies (as it is, and under `augment`'s training stress; Metrica's training-stress features
already exist, `lab_carried/features_1_metrica*.npz`). The normaliser and missing flags stay the IDSSE training
matches' (`lab_temporal.normaliser(idsse_used)`), so the inputs read as `gru_b`'s; seed 0; every other setting
`gru_b`'s.

**Why it should help.** The answer key follows Metrica's convention, and `gru_b` has only ever seen DFL's. F4 and
F5 show where that costs: Metrica labels live about five times as much of the "players stopped, ball unseen" time as
DFL does (5.5-15.7 % against 1.1-2.6 %), in 5-11 long stretches a game, and `gru_b` calls them dead with P 0.95 (and gets the team right only 66-75 % under
it). These are whole stretches inside long true possessions (50-56 % of Metrica's dead/live errors are more than 10 s
from a true dead edge), so each one splits a possession and adds a false dead spell. No decoder price or P(dead)
factor can fix it (round 8's search: f_dead weak); only a model that has seen the convention can learn when "stopped"
is still someone's ball. TEKAB2 trains on the other Metrica games ([field.md](../../field.md)).

**Why both copies of Metrica.** The convention lives in live, stopped stretches, which the stress doesn't touch (it
replaces the ball only in true stoppages). With the stressed copy alone, Metrica would teach "no ball and stopped =
live, moving ball and stopped = dead", a cue made by the pasting. Decision 3 (`learned.md`: Metrica blanks its dead
ball, so a model learns "no ball = dead") predates robust's blanking, which now removes IDSSE's still dead ball too;
the cue is no longer Metrica's alone, though it is stronger there.

**Guess:** Metrica's 3 **+0.01 to +0.03** each; DFL's 7 −0.005 to +0.003 (they gain three matches of data but meet
a conflicting convention); B's bench **+0.005** (−0.004 to +0.012). **~25 %** to pass the B bar (7 of 10 needs 4 DFL
matches better), **~40 %** to pass the Metrica rule below.

**The test** (new files: `bench/lab_netmet.py`, `docs/lab/netmet.md`, `bench/results/lab_netmet/`,
`data/external/lab_netmet/`):

| step | command | jobs × time | slots |
|---|---|---|---|
| train | `python -m bench.lab_netmet train --fold HELD` for the 10 matches, and `--fold all` (deployment): a copy of `lab_branch_b.fit` with the rows' matches (all others) and the normaliser's (the IDSSE ones among them) separated | 11 × ~10-12 min (~120 slot-min) | 1 each, ~2 GB |
| predict | each match with its own fold net: B's 3 variants, normal, p10, the training-stress copy | ~15 min | 1 |
| check, judge | as idea 1 (plumbing check with `gru_b`'s P, then confirm, then SkillCorner with `--fold all`'s net) | ~30 min | 1 |

**Reported, not judged:** F4 and F5's measures before and after (live→dead in long runs, per source); possessions
per match; per-source gains.

**The bar.** Primary: **the B bar above.** Secondary, fixed now on round 8's precedent for Metrica (decisions,
2026-11-12: "Metrica's 3 games are the main judge; DFL's 7 a guard (at most −0.005)"), applied to B's bench:
**Metrica's 3 +0.010 or more on the mean of their 3 variants and better on 3/3; DFL's 7 no loss beyond 0.005; no
variant down more than 0.002; SkillCorner no loss beyond 0.005.** A result that passes only the secondary is written
up as "passes the Metrica rule, not the B bar", for Jose to weigh.

**Carry risk: two-sided.** For: the convention is the answer key's, and F5 is a label fact, not a tracking quirk.
Against: (a) Metrica's 3 stop being an unseen provider, so the bench loses its provider-transfer test for this
change (DFL's 7 still are one: they meet Metrica's data in training, not in testing); (b) Metrica's damaged ball is
seen far more in live play (missing 9-14 % against DFL's 27-35 %; ours 29.5 % while players move), so a network
taught by Metrica may lean on a cue our match lacks; (c) SkillCorner is another convention and can't speak for this
one. If it shows "Metrica up, DFL down", the next step is a convention input (1 on Metrica's rows, 0 on DFL's; 1 on
our match, whose labels are Metrica's convention), not tonight.

**Fits tonight:** yes; ~1.3 h on 4 slots, ~2 h on 2.

## 3. `netdraw`: five members, each with its own training-stress draw

**What changes.** As idea 1, but member S (1-4) also trains on its own training-stress draw: stoppage balls from
SkillCorner's even-indexed matches drawn with seed S + 1 (`lab_carried.stressed_ball` takes the training pool for any
seed but `TEST_SEED` 0), torch seed S. The post stages keep draw 1 (v15's), so only the network changes.

**Why.** F1: `gru_b` finds the pasted stoppage ball *easier* than the normal one (dead→live 0.62 % against 1.37 %),
so it has learned one draw's templates well; on SkillCorner's real balls B gains +0.0008. Members that saw different
templates should lean less on any one. On B's bench this likely gains a little less than idea 1 (its test balls come
from the odd-indexed pool either way); on SkillCorner a little more.

**The test** (`bench/lab_netdraw.py`): 28 new stressed copies (7 DFL × 4 draws, ~1-2 min and ~110 MB each: 3 GB of
the 17 GB free), written under `data/external/lab_netdraw/` by setting `lab_carried.CACHE` there first (ball and
features both follow it); plumbing check: draw 1 rebuilt there must equal `lab_carried/features_1_<m>.npz` on one
match. Then 32 trainings (a copy of `lab_branch_b.fit` taking the draw), predict, combine, confirm, SkillCorner as idea
1. ~360 slot-min. **Bar:** the B bar. It is a second try at the same lever, so if both 1 and 3 run, both are reported
and the first to pass is not crowned on its margin over the other.

**Fits tonight:** only if idea 1's slots are done by ~04:00, or in place of idea 1 if the lead prefers the
SkillCorner side.

## 4. `netedge`: boundary-aware training

**What changes.** The same network with two training-only heads on the BiGRU's output, the main head unchanged for
P: (a) "the label changes within ±2 steps (0.4 s)", (b) dead against live. Loss = cross-entropy × 2 on frames within
1 s (5 steps) of a label change, × 1 elsewhere, + 0.3 × each auxiliary binary loss. All weights fixed now.

**Why.** F2: 64 % of the team errors sit within 1 s of a true change, where only 21 % of live frames are; 39-42 % of
DFL's dead/live errors within 1 s of a dead edge. spellF1 averages IoU 0.3-0.7, so short spells' edges count. The
field: TEKAB2's "heads for possession edges", Barca-Hackathon's training-only clean-owner head.

**Guess:** +0.002 (−0.003 to +0.006), ~15 %. The timing stage already moves edges, and trial 1015's settings were
tuned on `gru_b`'s calibration at the edges. **Test:** `bench/lab_netedge.py`, 8 folds (~8.5 min each), predict, the
plumbing and judging as idea 1; ~120 slot-min. **Bar:** the B bar. **Fits tonight:** yes, on spare slots.

## 5. `netghost`: visibility inputs (not tonight)

Eight label-free inputs for the GRU only: per team, players visible; players visible within 15 m of the ball's
last-seen spot; "ghosts" (a player whose piece ended in the last 10 s within 15 m of that spot); seconds since the
last ghost appeared. The players-only parts are the same on every variant (the stress changes only the ball).
Round 9's ceiling (+0.022 from missing players near the ball) is about *where* they are; this gives the network only
*that* they are missing, and F6 says that signal marks crowds more than it points to a team. SkillCorner's broadcast
tracking misses players for other reasons (the camera), so its guard would judge a different thing. ~2 h to build, ~200
slot-min; **+0.001, ~10 %**: below the line tonight.

## Not proposed, and why

- **Calibration before the decoder (temperature).** The decoder's costs are −log P, so a temperature T multiplies
  every emission cost by 1/T: the same as multiplying both switch prices by T (up to the 3-decimal rounding).
  Round 8's search drew the two prices independently over 0.5-16 and 8-100, plus a P(dead) factor: covered.
- **Test-time shifts and flips.** The team swap is already averaged; every frame already gets two overlapping
  windows (stride 150 of 300). Denser windows cost only inference but should move B's bench by under 0.001. The
  97 inputs are mostly mirror-invariant distances; a pitch flip would need the features rebuilt.
- **A bigger network, a TCN or a transformer, longer context.** ~7 matches of data; round 7 surveyed MS-TCN++ and
  ASRF and found the GRU covers that direction. Longer context would serve long stoppages (57 % of dead time sits in
  stoppages over 30 s), but dead→live errors are only 0.5-0.9 % of frames on the stressed variants (F1). A TCN as a
  sixth ensemble member is a later step.
- **More stress strength or partial-stress training copies** for the bench's sake: F1 says the stressed stoppages are
  already the easiest part of B's bench. Idea 3 keeps the one version that serves real balls.
- **Training on SkillCorner's labels**: it would end SkillCorner's role as the untouched check of real balls.
