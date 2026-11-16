# Night ideas from outside: other teams, the literature, unexplored leads

| | |
|---|---|
| **written** | 2026-11-14, 23:10–00:15, by the "outside-in" idea agent ([../night.md](../night.md), plan step 2) |
| **read** | [../../field.md](../../field.md), [../../research/README.md](../../research/README.md) and its four reports, `reports/Possession from tracking advances.md`, [../../ideas.md](../../ideas.md), [../../limitations.md](../../limitations.md), [../../lab.md](../../lab.md), [../../decisions.md](../../decisions.md), [../../bench.md](../../bench.md), [../branch-b.md](../branch-b.md), [../skillcorner.md](../skillcorner.md), [../cleaning.md](../cleaning.md), [../../models/v15-branch-b.md](../../models/v15-branch-b.md); the competitor notes saved in the session scratchpad during the 11-14 survey (Barca-Hackathon's plan, the source of most numbers below); the code of `bench/lab_branch_b.py`, `bench/lab_temporal.py`, `bench/search_b.py`, `bench/lab_gate.py` (to cost the tests) |
| **nothing here reads our match** | no file, statistic or output of BAR-ATH was opened; every test below runs on the 10 bench matches and SkillCorner's open data |
| **status** | proposals only: nothing built, nothing scored |

## The one fact that shapes every idea

**Branch B is 92 % one network, and that network was trained on 6 or 7 matches, all DFL, one seed.** `gru_b` never
saw a Metrica-tracked frame in training (Metrica's 3 share the net trained on the 7 DFL matches, and the deployed net is
that same net); it never saw a real moving stoppage ball, only SkillCorner's balls pasted into DFL stoppages; and its
seed (0) was never varied. Our match is Metrica-tracked and its labels follow Metrica's convention. Outside, the
strongest entries say the same thing about the same kind of model: Barca-Hackathon measured that **the number of
labelled games is what limits a sequence network** (training on one Metrica game instead of two costs −0.026; its nets
fit their training games at 0.94–0.97), that **one net moves ±0.005 with its training seed**, and that averaging seeds
gains +0.005 to +0.010 over the mean member. So the ideas below are about what the network learns from (more of the
right data, a privileged target, more members), not about new decoders or new stages: the lab has shown the chain
absorbs start gains unless the start itself is better over time (round 7), and the decoder is not the limit
(hsmm, round 7).

## The ideas, ranked (expected gain × chance it passes × fits tonight)

| rank | idea | B's bench, my guess | chance it passes the B bar | fits tonight | why it's here |
|---|---|---|---|---|---|
| **1** | **`gru_b` trained on Metrica's games too** (leave one match out over all 10) | **+0.008** (−0.004 to +0.020) | ~35 % | yes: ~2.5–3 h wall, ~2 slot-hours | the data closest to our match, never used to train the net B leans on |
| **2** | **a clean-owner head in training** (Barca-Hackathon's, the cheapest form of distillation from clean tracking) | +0.004 (−0.003 to +0.010) | ~25–30 % | yes: ~3 h wall, ~1.5 slot-hours | round 9 found the ceiling in what the damage hides near the ball; this teaches the net to infer it |
| **3** | **a 4-seed ensemble of `gru_b`** (seeds 0–3, P averaged) | +0.003 (0 to +0.006) | ~20–25 % (no loss: ~85 %) | yes: ~3–3.5 h wall, ~3.5 slot-hours | the cheapest robustness fix for a real match; and the noise yardstick ideas 1–2 need |
| 4 | **SkillCorner's even-indexed 10 in `gru_b`'s training** (real stoppage balls, real labels) | −0.002 (−0.008 to +0.004); SkillCorner's odd 10: +0.012 | ~25 % on a robustness bar | borderline: ~3.5 h wall | the only way tonight to make B learn real, not synthetic, stoppage balls |
| 5 | **the passers combined** (e.g. 1 + 2, as a 4-seed ensemble) | the sum, less overlap | depends on 1–3 | only if two pass by ~04:30 | the lab's mixing rule |

Detail for each follows. Ideas 1–4 share one harness (below), so a builder writes it once.

---

## 1. Train `gru_b` on Metrica's three games too

**What changes.** Today `gru_b`'s folds are the 7 DFL matches (each trained on the other 6 DFL) plus one fold for all of
Metrica's 3 (trained on the 7 DFL); the deployed net is that last one. The candidate: **10 folds, leave one match out
over all 10** (a DFL fold trains on 6 DFL + 3 Metrica; a Metrica fold on 7 DFL + the other 2 Metrica), each match as it
is **and** under `augment`'s training stress, exactly `lab_branch_b.fit` otherwise (seed 0, 20 epochs, the fold's own
normaliser). The deployment net trains on all 10. `augment` (8.5 % of the start), the decoder, trial 1015's settings
and the post stages' recipe stay as they are.

**Why it should help.**
- **Barca-Hackathon** (field.md's serious entries; their plan's "Clean-owner head" section): "the number of labelled
  games is" the limit; training on one Metrica game instead of two costs **0.026**; adding DFL to Metrica games
  +0.015–0.019. **TEKAB2**'s Metrica 0.805 trains on the other two Metrica games. Every serious entry above us on
  Metrica trains on Metrica.
- **Our own lead 6** in [../../inside-v12.md](../../inside-v12.md) ("Let robust learn from Metrica too", status "to
  test", never tested): the reason Metrica was kept out of training (its ball vanishes in dead time, "no ball = dead")
  went away when robust's blanking made DFL's dead time look the same (99.7 % missing); and `gru_b` additionally trains
  on stressed copies, where a stoppage ball moves. `learned --train-all-sources` (information only) reached 0.7725 on
  Metrica against 0.7023.
- Our match is Metrica-tracked and its answer key follows Metrica's convention (decisions 2026-11-12). Of every
  candidate here, a gain that comes from Metrica's games is the one most likely to carry to our file.

**Risk.** Round 5-6's pattern: what helps Metrica can cost DFL. Here the DFL folds get +50 % training data, which
should hold them level or better, but the convention shift (Metrica's possessions change hands ~21 % less often) could
cost DFL's short possessions. The bar's "neither source down" part catches that.

**Licence / rules.** Metrica's sample data has **no licence file** (limitations G8). Jose's 2026-10-30 decision says
"checking only", but the `all` set's trained post stages already learn from Metrica's letters (v12 to v15; v15's stage
models are trained on all 10), with IDSSE-only twins behind one switch since v10. Training `gru_b` on them is the same kind of use; **it needs
Jose's explicit OK before any adoption**, and the technical document must credit Metrica (their README asks for it).
The test itself is internal and uses external data only.

**The test** (new files only: `bench/lab_night_gru.py` with `--recipe metrica`, results in
`bench/results/lab_night_gru_metrica_*`, page `docs/lab/night_metrica.md`):
1. Train 10 fold nets + 1 deployment net (all 10). ~9 min each at `OMP_NUM_THREADS=2` (v15's folds took 405–504 s on
   6–7 matches; 9 matches are ~30 % more chunks): **~100 slot-minutes**, 2 slots ≈ 50 min.
2. P on B's three variants and on the training-stress copies, each match from the net that never saw it (~15 min).
3. In the isolated copy (harness below): `search_b confirm --trials 1015`, then `search_b skillcorner --trials 1015`
   with the all-10 deployment net (~25 + 15 min, ~3 GB at peak).
- **Total ~2.5–3 h wall** from the start of coding (~45 min of plumbing). Judged by ~03:00 if started at 00:00.

**Bar** (the B bar, fixed here before any score; "Common bar" below): +0.005 or more on B's bench, better on 7/10, no
variant down more than 0.002, neither source down on the mean, SkillCorner loss ≤ 0.005; plus the noise rule.

---

## 2. A clean-owner head in training (distillation from clean tracking, its cheapest form)

**What changes.** `gru_b` gets a second output, used **in training only**: for each frame, which team's player is
within 1.5 m of the ball **in the bench's clean tracking** (`sets.load(m)`, level 0: home / away / nobody; ignored
where the clean ball is missing; home ↔ away exchanged on the swapped orientation; the same target on the stressed
copy, which only changes the ball). Loss = main + **0.5** × head. Prediction reads the main head only, so the deployed
program and its inputs don't change. Everything else is `gru_b`'s recipe (DFL training as now, seed 0). **Build the
extra head after `T.network(n_in)`**, so the main net's initial weights and the batch order are exactly v15's: the only
difference from v15's net is the head's gradient (a paired comparison).

**Why it should help.**
- **Barca-Hackathon, 2026-11-12, their "Clean-owner head" section:** 4 seeds per arm, held out: **+0.011, +0.011,
  +0.003, +0.010 per seed on Metrica**; the 4-net ensemble +0.009 (90 % interval +0.003 to +0.015); confirmed on their
  shipped recipe's own held-out runs (+0.007, every one of 7 seed nets up); DFL unchanged (−0.002). Recipe copied
  exactly (1.5 m, weight 0.5; their weight 1.5 leaned to DFL and cost Metrica −0.005). **Against:** Alberstation's
  similar head failed its screen (field.md).
- **It aims at round 9's finding** ([../cleaning.md](../cleaning.md)): 90 % of the +0.024 ceiling is the true
  positions of missing players within 15 m of the ball, and no fill built from the tracking recovers any of it. A
  target computed from what the clean tracking shows at the ball forces the net to infer that state from context
  (teammates' shape, the block's movement) instead of filling coordinates. This is "learning using privileged
  information" / generalized distillation (Lopez-Paz, Bottou, Schölkopf, Vapnik, ICLR 2016), where training-only
  information shapes the student's representation.
- **2b, if 2 passes and time is left (not tonight's first choice):** a teacher GRU trained on `o_players` features (the
  true players, damaged ball: +0.024 for the trees), its soft P distilled into `gru_b` (KL at weight 0.5) on the normal
  copies. `o_players`' features exist only for the normal bench (`data/external/lab_clean/o_players/`), so the stressed
  copies would keep the hard labels only. Twice the cost of 2, same evidence class.

**Licence / rules.** The targets come from IDSSE's clean tracking (CC BY 4.0, approved #16). The idea is Barca-
Hackathon's (public repo, read 11-14): credit it on the page; no code copied.

**The test** (`bench/lab_night_gru.py --recipe owner`; page `docs/lab/night_owner.md`):
1. Targets: one small array per DFL match from `sets.load` (frames aligned with the lab cache, as `lab_clean`'s `o_*`
   steps use them). ~5 min.
2. 8 nets (as v15's folds), ~8 min each: **~65 slot-minutes**.
3. P, confirm, SkillCorner as idea 1 (the deployment net is the `metrica1` fold, as in v15).
- **Total ~3 h wall** (~1 h of plumbing: the two-head subclass, the targets).

**Bar:** the common bar, plus the noise rule. Report also the head's held-out accuracy (does it learn anything beyond
the damaged nearest player?) and the main head's log loss against v15's.

---

## 3. A 4-seed ensemble of `gru_b` (and how much of v15's gain is seed luck)

**What changes.** `gru_b` trained with seeds 1, 2, 3 (`lab_temporal.SEED` set before `fit`; nothing else changes) on
v15's folds; the start's GRU P is the **mean of the 4 nets' P** (seeds 0–3, seed 0 = v15's net), then mixed with
`augment` at 0.915 exactly as trial 1015. Deployment: the 4 `metrica1`-fold nets (each trained on all 7 DFL), averaged.
All 4 members, not a chosen subset (Barca: choosing subsets is selection bias).

**Why it should help.**
- **Every serious public entry with a sequence model ships an ensemble:** Barca-Hackathon 8 nets (the ensemble beat
  its mean member by +0.005 to +0.010, its best by up to +0.006; single nets ±0.005 with the seed), TEKAB2 5 CNNs,
  zka2038 5 BiGRUs, nuggetofwisdom 3 TCNs. We ship one.
- **The literature's most robust choice under dataset shift:** Ovadia et al., NeurIPS 2019 ("Can you trust your
  model's uncertainty?") found that methods marginalising over models, deep ensembles first, keep accuracy and
  calibration best as inputs drift, while post-hoc calibration degrades. Our match *is* a drift from the bench (real
  damage, real stoppage balls).
- **It answers a question v15 left open:** trial 1015 was picked as the best of 1,500 settings **on one seed's P**.
  Scoring seed 1 alone through the same confirm says how much of the +0.035 is the seed. If seed 0 is the luckiest of
  4, v15's estimate is inflated, and the ensemble is the honest deployment even at ±0.

**The test** (`bench/lab_night_gru.py --recipe seeds`; page `docs/lab/night_seeds.md`):
1. 24 nets (3 seeds × 8 folds), ~8 min each: **~190 slot-minutes** (2 slots ≈ 95 min; if other ideas need the slots,
   seeds 1–2 only, a 3-member ensemble: decided before training starts, never after a score).
2. P: each seed's P per match, the 4-mean per match, on B's variants and the training-stress copies (~25 min).
3. In the isolated copy: confirm for **the ensemble** (the candidate) and for **seed 1 alone** (the yardstick), each
   with its own stage models (~25 min each); seeds 2 and 3 alone if a slot is free (information). SkillCorner for the
   ensemble.
- **Total ~3–3.5 h wall.** Start it first: it needs no new model code, and its seed-1 result sets the noise rule for
  ideas 1–2.

**Bar:** the common bar (an ensemble that can't reach +0.005 doesn't pass, however safe). **For Jose, information
next to it:** each seed's B bench; the ensemble against the mean of the single seeds; the ensemble's worst match.

**Licence / rules.** No new data.

---

## 4. SkillCorner's even-indexed 10 matches in `gru_b`'s training

**What changes.** The 10 even-indexed SkillCorner matches (`lab_carried.sc_pool('train')`, the ones `augment`'s
training stress already draws its balls from) join every fold's training, **as they are** (their real broadcast
tracking, real stoppage balls, SkillCorner's own labels as `bench/skillcorner.py` reads them), once each (no stressed
copy: their stoppages are already real), both orientations, read through the fold's DFL normaliser. Inputs:
`lab_carried.sc_prepared(m)` (robust's features on SkillCorner, already cached for the SkillCorner check). **Cut-away
frames (no tracking at all) are left out of the loss**: our match has none, and they're 96 % dead. The odd-indexed 10
stay untouched: the guard, and the source of the test variants' stoppage balls, so nothing leaks into the bench.

**Why.** It's the only change tonight that makes B learn **real** moving stoppage balls instead of balls pasted into DFL
stoppages, and B's gain is exactly where the evidence is synthetic (v15: +0.035 on the bench, **+0.0008 on
SkillCorner**). Outside: Barca-Hackathon added SkillCorner's 20 and gained +0.002 to +0.004 on Metrica; Ryu2804 trains
on it. **Against:** Alberstation tried it (v7) and reverted; SkillCorner's labels differ (the new team's player is
2.1 m from the ball at a change, against 0.6 m on DFL; D is "no phase of play"), which may blur the timing Metrica's
convention wants.

**Licence / rules.** SkillCorner open data, MIT: training is allowed; credit SkillCorner. External data only.

**The test** (`bench/lab_night_gru.py --recipe skillcorner`; page `docs/lab/night_sc.md`): ~1 h of plumbing (SkillCorner
segments for the training loop), 8 nets at ~12 min (**~100 slot-minutes**), then P, confirm and SkillCorner (scored on
the odd 10). **~3.5 h wall**; start only when a slot frees (~01:30–02:00).

**Bar: a robustness bar, not the B bar** (fixed here; it needs Jose to accept that kind of bar, as round 7 did for
`carried`): (1) B's bench **no loss beyond 0.002** on the mean, no variant down more than 0.003, neither source down more
than 0.005; (2) **SkillCorner's odd 10: +0.010 or more, better on 7 of 10**; the even 10 reported as in sample, never
judged.

---

## 5. The passers combined

If two or more of 1–3 pass, the obvious mix is one recipe (Metrica in training + the owner head) trained as a 4-seed
ensemble. **Rule (fixed now):** it replaces the best single passer only if it beats it on B's bench (mean higher, better
on 6 of 10) and still passes parts 2–4 of the common bar against v15. Cost: as idea 3 with idea 1's folds
(~4.5 slot-hours); start only if both inputs have passed by ~04:30, so it's judged by ~08:30.

---

## The harness ideas 1–4 share (written once, `bench/lab_night_gru.py`)

- **Isolated copy**, as `stage3.sh`: `rsync` the code (no `.git`, `.venv`, `data`, `evaluation/results`,
  `evaluation/reports`) to `$SCRATCH/night_<recipe>`; in the copy's `data/external/` **symlink every directory of the
  repo's `data/external/` except `lab_branch_b` and `search_b`**, which are created empty; copy only
  `data/external/search_b/p_augment_train_*.npy` into the new `search_b` (`augment` doesn't change). **Never** copy
  `tables_trial*.joblib`: `search_b.trial_tables` would reuse v15's stage tables silently.
- The module trains the recipe's nets into the copy's `lab_branch_b/`, writes `p_gru_b_<variant>_<m>.npy` (B's 3
  variants) and `search_b/p_gru_b_train_<m>.npy` (the training-stress copies) from the net that never saw `m`, then
  runs `search_b confirm --trials 1015` and `search_b skillcorner --trials 1015` **in the copy**, patching only the
  copy-side functions it needs (`lab_branch_b.net_for` / `used_for` for new folds; the SkillCorner GRU for an
  all-10 or ensemble deployment net) from inside the new module. No existing file is edited, in the repo or the copy.
- **Two checks before any candidate is scored** (shared, ~30 min, one slot): in a copy with v15's own nets and P,
  (a) trial 0 (augment alone) reproduces `confirm_1015_554_586_1133_1269_-1_-2.csv`'s trial-0 rows (B's bench 0.6975),
  (b) trial 1015 reproduces its rows (0.7329) to 1e-4. If (b) fails, the copy's fresh caches differ from the shared
  ones and every comparison below waits.
- `lab_gate.own_dead_cache` writes into lab_gate's `lab_dead/` under a content hash of P(dead) and the ball, neither of
  which these recipes change: the files already exist, so nothing new is written there. Check once that no other write
  reaches a symlinked directory (`find -newer` on them after the null run).
- Heavy steps go through `slot.py` at `OMP_NUM_THREADS=2`; one GRU training < 1.5 GB; at most two `confirm`s at once
  (~3 GB each).

## The common bar (fixed before any score; ideas 1, 2, 3, 5)

Against **v15's B as deployed** (trial 1015's rows in `bench/results/search_b/confirm_1015_554_586_1133_1269_-1_-2.csv`
and `skillcorner_1015_554_586_1133_1269.csv`), the candidate's trial 1015 in its copy, its stage models trained on its
own chain's out-of-fold letters (as `confirm` does):

1. **B's bench** (the mean of p25, p50 and stressed per match): **+0.005 or more, better on 7 of 10**.
2. **No variant down more than 0.002; neither source** (IDSSE's 7, Metrica's 3) **down on the mean.**
3. **SkillCorner's 20** (deployment models): **no loss beyond 0.005** on the mean.
4. **The noise rule:** let s = |seed 1's B bench − seed 0's B bench| from idea 3. **If s ≥ 0.003**, a pass by idea 1 or 2
   counts only if the same recipe with seed 1 also passes parts 1–2 (one more set of folds, ~1–1.5 slot-hours). If
   idea 3's seed-1 confirm isn't done when idea 1 or 2 finishes, its pass is reported as "pending the noise rule".
5. As round 9: the switch reads the raw tracking, so its choice on our match can't move; its bar's part 3 is rerun for
   information only. A pass gets a page and goes to Jose; nothing is adopted tonight, and nothing runs on our match.

Idea 4 has its own robustness bar (above).

## Licence and rules, in one table

| idea | data used to train or tune | licence | rules |
|---|---|---|---|
| 1 | IDSSE 7 (CC BY 4.0), **Metrica's 3 sample games (no licence file, G8)** | Metrica: already in the `all` set's trained post stages (v12 to v15), IDSSE-only twins behind a switch since v10; **Jose's OK needed before adoption** | external only |
| 2 | IDSSE 7 and their clean tracking | CC BY 4.0 (#16) | external only; credit Barca-Hackathon's idea |
| 3 | as v15 | as v15 | external only |
| 4 | IDSSE 7 + SkillCorner's even 10 | MIT; credit SkillCorner | external only; SkillCorner's odd 10 stay a pure test |

## Generalisation risk of v15's B, and the cheap fixes

What the evidence says about B on a real match:

- **The gain is real on a provider the net never trained on** (Metrica's 3: +0.023 to +0.073), so it isn't only a DFL
  fit. But it rests on **one seed, 6–7 training matches, and stress from one provider's balls**. On SkillCorner (real
  stoppage balls, another convention) the new B is level with the old (+0.0008, 11/20): the bench gain comes from
  steadier team calls over time, not from reading real stoppage balls better.
- **The cheapest change that makes B more robust rather than better on the bench is idea 3** (an ensemble of seeds:
  no new data, no new knob, the literature's most robust choice under shift, and it tells Jose how much of v15's
  +0.035 is one seed's luck). **The cheapest change that brings B's training closer to our match is idea 1** (the only
  Metrica-tracked, Metrica-convention data that exists). Idea 4 is the only one aimed at the real-stoppage-ball gap,
  and the riskiest for the bench.
- **Not proposed, though tempting:** lowering the GRU's weight or switching the post stages' training data to `both` as
  a hedge. SkillCorner hints that way (`mix_b` +0.0069, trial 554 +0.0097), but choosing on it would make SkillCorner
  part of the tuning, which round 8 ruled out; and the search found both choices flat on the bench.

## Considered and not proposed tonight

| idea | source | why not tonight |
|---|---|---|
| more damage draws per training match (Barca uses 6 per Metrica game) | Barca-Hackathon | each draw needs a full lab cache rebuild (v4–v11's out-of-fold outputs, PathCRF) per match; unmeasured cost; tomorrow's candidate if idea 1 shows data is the limit |
| damage matched to our "ball missing while players stopped" (83 % against the bench's 65.8 %, documented in bench.md) | report rank 2; bench.md's known gap | rebuilds the whole bench (the judge and every upstream model); and robust's blanking already makes the bench's dead time 99.7 % ball-missing, so the gap's effect after blanking is unclear |
| player holes biased to crowds near the ball | cleaning.md's caveat; Jose's 10-30 answer ("when a player has the ball close to another player") | no documented statistic of ours to calibrate it; an uncalibrated stress would win on a test variant built the same way |
| SoccerTrack v2 (10 matches, CC BY 4.0) | research README, Alberstation | needs a Hugging Face account (an account: Jose's call); its ball is reconstructed between events |
| PFF World Cup 2022 | TEKAB2, MikolajEkiert | its terms forbid benchmarking use (G8) |
| left-right mirrored copies | Barca-Hackathon | no gain with 4 seeds there; team swap is already in |
| heads for the latest / next clean owner, or possession edges | Barca-Hackathon, TEKAB2 | Barca: no gain on Metrica (−0.002), +0.007 on DFL with 2 seeds; would fork idea 2 |
| ball-exit snap | jekkiechan | our `stoppage` stage is the same rule (out on top of robust); B runs with snap and move off |
| MBR decoding, semi-Markov decoding | Pompeu, TEKAB2 | round 7: the frames limit recall, not the decoder (hsmm discarded) |
| a masked-ball set encoder (TranSPORTmer-like) | the 2026 report's rank 1 | 1–3 weeks |
| test-time adaptation | report | fails on single correlated streams (TTAB); and it would read our match |

## Sources

- Barca-Hackathon's public plan (read 2026-11-14 for [../../field.md](../../field.md); notes in the session
  scratchpad): ensemble and seed numbers, clean-owner head results, "more labelled games" measurements.
- [../../field.md](../../field.md): TEKAB2, zka2038, nuggetofwisdom, Alberstation, Ryu2804 recipes.
- Ovadia et al., NeurIPS 2019, [arXiv 1906.02530](https://arxiv.org/abs/1906.02530) (deep ensembles under shift).
- Lopez-Paz, Bottou, Schölkopf, Vapnik, ICLR 2016, [arXiv 1511.03643](https://arxiv.org/abs/1511.03643) (generalized
  distillation / privileged information).
- `reports/Possession from tracking advances.md` (training-time masking, structured faults, the shift literature).
