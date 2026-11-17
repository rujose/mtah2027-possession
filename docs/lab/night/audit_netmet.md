# Audit of `netmet` and `netmet5` (2026-11-15, 06:55 to 07:25)

| | |
|---|---|
| **verdict** | **No problem found. The numbers on [network.md](network.md) are right, and the comparison with v15's B is fair.** No held-out match, Metrica's included, reaches anything trained or fitted for scoring it: not its GRU fold, its normaliser, `augment`'s share, or the post stages. Every number re-derives exactly from the cached P. Four points change how the gain should be read (none is an error): see the next section |
| **question** | Is `netmet`'s gain over v15's B (seed 0 +0.0130; `netmet5`, seeds 0-3 averaged, +0.0139) real and fairly measured, especially with Metrica's 3 games newly in training? |
| **auditor** | independent of the builder. Read the code and the results, then ran only in-memory checks (at most one slot). No retraining. Nothing written outside this page and the scratchpad's `audit/` |
| **our match** | not read; nothing run on it |

## Problems, and what to keep in mind

**No problem that makes a number wrong or unfair.** Questions 1-4 below give the evidence. Four points change how the
numbers should be read. Jose should weigh them:

1. **Most of the gain is Metrica's own, and Metrica is no longer an unseen provider.** Of `netmet5`'s +0.0139,
   +0.0120 is Metrica's 3 (0.3 × +0.0400) and +0.0019 is DFL's 7 (0.7 × +0.0027). Each held-out Metrica game is
   scored by a network that trained on the other two Metrica games: the same provider, tracking system and labelling.
   That is not leakage. Each game is a different match (checked, Q4), and the idea page named this risk before any
   score ([ideas_network.md](ideas_network.md), "Carry risk (a)"). But +0.040 measures how well the network does on
   a provider it has trained on. For a match from a provider the network never trained on, the nearer guides are:
   - **DFL's 7: +0.0027 for `netmet5`.** That is under the seed noise (0.0035). Single seeds range from −0.0025 to
     +0.0047 (point 4).
   - **SkillCorner's odd 10: +0.0025, better on 7/10.** None of these matches is in any training. On the even 10,
     whose stoppage balls are in every B GRU's training stress, the change is −0.0017 (Q3).
2. **One second-order path now reaches Metrica as well. I measured it, and it does not carry the gain.** The post
   stages for a held-out match m are trained on the other matches' P. In `netmet`, every one of those P comes from a
   net that trained on m. In v15 this was already true for the DFL matches, but not for Metrica, because `gru_b`
   never trained on Metrica. To test it, I decoded `netmet5`'s P with **v15's own stage models**, in which no net
   ever saw a Metrica game. Metrica's 3 still gain **+0.0410** (the judged number is +0.0400), and B's bench gains
   +0.0143 (judged: +0.0139). For seed 0: Metrica +0.0331 against +0.0324 judged. Script `a7_v15stages.py`.
3. **The reference is flattered; `netmet` is not.** Trial 1015 was picked as the best of 1500 trials on these 10
   matches, using `gru_b` seed 0's P ([branch-b.md](../branch-b.md): +0.0353 in sample, +0.0334 nested, so about
   0.002 of winner's curse). `netmet` runs on settings tuned for another network. Both effects work against
   `netmet`. Neither inflates its gain.
4. **Seed spread, for information.** Seeds 2 and 3 alone were never scored overnight. I scored them in memory
   through the same trial 1015, with the procedure that reproduces the judged CSVs exactly (Q1). The four seeds give
   B's bench +0.0069 to +0.0141 (mean +0.0109); `netmet5`'s +0.0139 is inside that range. Within that:
   - **Metrica's 3** gain +0.029 to +0.039 with every seed, 3/3 each time.
   - **DFL's 7** move −0.0025 to +0.0047 (mean +0.0013), below zero for 2 of the 4 seeds. The DFL part is the size
     of the seed noise.
   - **J03WOH and J03WPY** lose with all four seeds (−0.008 to −0.023).

   | `netmet` − v15's B | B's bench | better on | p25 / p50 / stressed | DFL's 7 | Metrica's 3 |
   |---|---|---|---|---|---|
   | seed 0 (judged) | +0.0130 | 8/10 | +0.0108 / +0.0120 / +0.0163 | +0.0047 | +0.0324 |
   | seed 1 (judged, the noise rule) | +0.0069 | 6/10 | +0.0046 / +0.0051 / +0.0110 | −0.0025 | +0.0288 |
   | seed 2 (audit, information) | +0.0141 | 7/10 | +0.0129 / +0.0115 / +0.0179 | +0.0033 | +0.0392 |
   | seed 3 (audit, information) | +0.0096 | 7/10 | +0.0103 / +0.0084 / +0.0101 | −0.0002 | +0.0325 |
   | `netmet5` = mean of seeds 0-3 (judged) | +0.0139 | 6/10 | +0.0140 / +0.0130 / +0.0147 | +0.0027 | +0.0400 |

   Per match, seed 2: J03WMX +0.016, J03WN1 +0.006, J03WOH −0.014, J03WOY +0.018, J03WPY −0.019, J03WQQ −0.001,
   J03WR9 +0.016, metrica1 +0.028, metrica2 +0.062, metrica3 +0.028.

   Per match, seed 3: J03WMX +0.000, J03WN1 +0.002, J03WOH −0.019, J03WOY +0.021, J03WPY −0.019, J03WQQ −0.001,
   J03WR9 +0.014, metrica1 +0.031, metrica2 +0.033, metrica3 +0.033.

   These two rows are information, not a verdict. They were scored after the night, by the auditor, and they don't
   change the registered candidate.

The page's declared choices are confirmed as declared (Q2). `netmet5` was registered after seed 1 was seen, and it
was judged whatever `netens` gave. Its K = 4 was set by the clock. I found no other choice made after a score.

## Q1. Is each held-out match absent from everything trained or fitted to score it?

**Verdict: clean.**

| piece | what I checked | evidence |
|---|---|---|
| **GRU folds** (all 4 seeds) | Each fold's training list, normaliser list and seed. Then each stored P re-predicted from the net that should have made it | `lab_netmet.py:38-43` (`used_for`: the other 9; `norm_for`: the IDSSE matches among them), `:50-60`, `lab_netmirror.py:317-327`. All 44 net `.json` files (10 folds + `all`, seeds 0-3): `trained_on` is the other 9 (all 10 for `all`), `norm_on` their DFL matches, and the seed is right (`a1_nets.py`). **All 80 stored held-out P re-predicted in memory** from the fold net that never saw the match (10 matches × seeds 0-3 × p25 and the training-stress copy): max \|difference\| **0** (`a8_provenance_all.py`). As a control, the `all` net, which did see the match, gives a mean \|difference\| of 0.014-0.038 against the same files, so a fold mix-up would show (`a5_predict.py`). No P file equals `gru_b`'s, no two seeds' P are identical, and every P is newer than its fold net |
| **input normaliser** | Whose statistics it reads | `lab_temporal.normaliser` (`lab_temporal.py:92-102`) sums the per-match statistics of the matches it is given, and nothing else. Every `netmet` fold, `all` included, gets only its training DFL matches. The same list is used at prediction (`lab_netmirror.py:424`). `data/external/lab_temporal/stats_metrica{1,2,3}.npz` **do not exist**: no normaliser has ever computed Metrica's statistics. The held-out DFL match is never in its own fold's list (json above) |
| **training-stress copies** | Whether a held-out match's stressed copy, or its stress draw, reaches its training | Each copy is built from that match alone (`lab_carried.py:285-342`): a draw seeded by (seed, crc32 of the match id), the match's own truth and clean ball, and templates from SkillCorner (even 10 for training, `TRAIN_SEED`; odd 10 for the test, `TEST_SEED`). The two pools are disjoint, and **0 templates are shared** between the training and test draws over the 10 matches. Each match's training-stress features carry that match's own truth (all 10 checked). A held-out match's copy is used only by the other folds and in the other matches' stage models (one-liners, below) |
| **`augment`'s share** (weight 0.085) | Which fold model scores m | Unchanged from v15. On the bench variants: `lab_gate.held_out_p` reads lab_gate's fold P (`folds_<held>.joblib`; Metrica's 3 use `folds_metrica1`, trained on the 7 DFL). On the training-stress copies: the shared `p_augment_train_*`, made with `search_b.fold_models` (`search_b.py:123-143`). The mirror reads both and never writes them (`lab_netmirror.py:71-80`) |
| **post stages** (trial 1015, trained on the training-stress copies) | That m's own copy is left out, and that the tables are this network's and are fresh | `search_b.trial_models(trial, without=m)` drops `m~train` (`search_b.py:382-386`). Each mirror recomputed `tables_trial1015.joblib`: in `main`, `seed1`, `ens2` and `ens4`, all 10 tables of each stage differ from the shared (`gru_b`) ones, and each file is newer than every P in its mirror. **Rebuilt from scratch in memory from the mirror's P, they equal the cached tables exactly, and all 30 confirm units equal the builder's CSV exactly** (`ens4` and seed 0; `a4_confirm.py`) |
| **trial 1015's stage tables, second order** | The path in point 2 above | Present (new for Metrica, as it already was for DFL in v15), and measured not to carry the gain (`a7_v15stages.py`) |
| **caches built once** | Anything fitted on all 10 together | The lab cache (11-04), robust's features (11-04), the stressed features (11-09; Metrica's training-stress copies 11-13), the p25/p50 features (11-09) and `augment`'s training-stress P (11-13) are all from before the night, and all per match. The mirrors' `lab_dead/` folders are empty, and the trial-1015 chain (short, segments, timing) doesn't read `lab_dead`. Inputs made before the bench (v4/v6 left out one match at a time; PathCRF trained or validated on the 6 DFL matches other than J03WR9) are shared with v15 and outside this audit |

## Q2. Is the comparison with v15's B fair?

**Verdict: clean.** The choices the page declares are confirmed as declared. The reference is flattered (point 3), not
the candidate.

- **Same trial 1015.** `search_b.every_trial()[1015]`: w_gru 0.9152, f_dead 0.9095, switch prices 2.817 / 48.70,
  minimum 7, snap off, short 0.340, segments 0.910 / 0.661, timing 0.450, post stages on the stressed copies. It comes
  from the code that wrote the reference: `search_b.py` was last changed 11-13 19:00, and the reference CSV was
  written at 11-13 19:05. The reference CSV is tracked in git at `f17aa87` and is unmodified.
- **Same pipeline, only the GRU P swapped.** `lab_netmirror.use` (`lab_netmirror.py:83-108`) points only
  `lab_branch_b.CACHE`, `search_b.CACHE` and the training-stress P path at the mirror. A search of `bench/` and
  `possession/` finds no other reader of B's GRU P. The plumbing check passed exactly: with `gru_b`'s P, 60 bench
  units and 40 SkillCorner units match the reference (`bench/results/lab_netmirror/plumb/plumbing.json`), and the
  rebuilt tables equal the shared ones. The uncommitted edits in `bench/lab.py`, `bench/lab_carried.py` and
  `possession/v12.py` are round 9's. They are inert without a cleaned cache or `MTAH_LAB_CLEAN`, and they predate the
  plumbing check (11-14 20:01).
- **Same units, same truth.** Every candidate CSV has the reference's 30 units (p25, p50, stressed × 10). Trial 0
  (B as built), rerun in each mirror, equals the reference on all 30 bench units and all 20 SkillCorner units (max
  \|difference\| 0). The bench and the truth are therefore unchanged.
- **The bar, recomputed from the CSVs alone** (`a3_judge.py`): every row of the page's table, to the 4th decimal
  (gains, better-on counts, per variant, per source, SkillCorner).
- **Choices after scores.** The declared ones are true as written. Checked beyond them:
  - `netmet5`'s K: `network/metk.py` decides from the existence of files alone (a seed's 10 fold P files and its
    `all` net). Its log reads "K = 4 … by the clock" at 06:44.
  - Before K was fixed, the only `netmet` scores were seed 0, seed 1 and seeds 0+1 (`done.txt`: `conf_met`,
    `conf_netmet_seed1`, `conf_metens2`). There was no score of seed 2 or 3 alone, nor of a 3-seed mean, so there was
    nothing to choose K from.
  - The 04:33 reorder (`netown` before `netmet`'s seeds 3-4) cost seed 4. It came after seeds 0 and 1 were seen but
    before any `netmet` ensemble score, so it can't have selected on one.
  - Metrica's rule and its thresholds, the DFL-only normaliser and the `all` net for SkillCorner are all in
    [ideas_network.md](ideas_network.md), last modified at 23:24, before the first score (02:20).

## Q3. SkillCorner: which network, trained on what, as v15?

**Verdict: clean, and the same arrangement as v15's B.**

- **`netmet`** is scored by seed 0's `net_all`. **`netmet5`** is scored by the mean of the four `net_all` (seeds
  0-3; `ens4/members.json`, `metk.py`). Each was trained on the 10 bench matches only (json `trained_on`). Each was
  read through the 7 DFL matches' normaliser (`lab_netmirror.py:174`), as in its training. The stage models are
  trained on all 10 bench training-stress tables from the mirror (`search_b.trial_models(trial)`).
- **v15's B** is scored by `gru_b`'s `net_metrica1`, which was trained on all 7 DFL matches. In both cases the
  deployment net is the recipe trained on every one of its training matches, with the same normaliser and the same
  kind of stage models.
- **No SkillCorner label trains any of these networks.** All of them, and `augment`, do see the stoppage ball paths
  of SkillCorner's even 10 matches, pasted into bench stoppages by the training stress. The same holds for v15. So
  the even 10 are not untouched for either network.
  - Split for `netmet5`: even 10 −0.0017 (5/10), **odd 10 +0.0025 (7/10)**.
  - Split for seed 0: −0.0025 / −0.0009.
- **Recomputed in memory** (`a6_sc.py`) on 3 of the 20 SkillCorner matches: 1874553, 1986691 and 2007721, with four
  `all` nets averaged and the mirror's stage models. All three equal the CSV to 5 decimals.

## Q4. Anything else that would inflate the gain?

**Verdict: clean.**

- **Metrica's 3 are three different games.** Their frame counts differ (145,007 / 141,157 / 143,762). No run of 8 consecutive
  label-run lengths appears in any two of them. So the held-out game is not a copy of a training game.
- **Shared stressed copy, normalisation statistics or stress draws:** none (Q1). The draws are seeded per match, and
  the training and test template pools are disjoint. The normaliser never read a Metrica game.
- **Seed or fold mix-up:** none. 44 json files, 80 P re-predicted exactly, no cross-seed duplicates. `ens2` and
  `ens4` recompute exactly from their member seeds (max \|difference\| 0; `a2_mirrors.py`).
- **A stale cache from v15's run:** none. No mirror P equals `gru_b`'s. Each mirror's tables are newer than its P,
  differ from the shared tables, and equal a from-scratch rebuild. Each confirm CSV is newer than its tables.
- **Nit, no effect on any verdict.** The bench ensemble averages the float32-stored P. SkillCorner's ensemble
  averages float64 P (`lab_netmirror.py:183`). A mean of four 3-decimal values can sit exactly on a rounding tie, and
  the two routes may then round it differently, by 0.001 in P on those frames. A deployed `netmet5` would follow
  SkillCorner's route.

## What I ran (scratchpad `audit/`, nothing written elsewhere)

| script | what |
|---|---|
| `a1_nets.py` | the 44 net json files; the P files are present, newer than their nets, unlike `gru_b`'s and unlike other seeds' |
| `a2_mirrors.py`, `a2b_tables.py` | the ensembles recomputed from their seeds; the mirrors' file times; the tables against the shared ones |
| `a3_judge.py` | the bar, recomputed from the CSVs |
| `a4_confirm.py` | trial 1015 on all 30 units, in memory, with the tables rebuilt from scratch (`ens4`, seed 0) |
| `a5_predict.py`, `a8_provenance_all.py` | stored P re-predicted from the fold nets (6 spot checks with the `all`-net control; then all 80) |
| `a6_sc.py` | `netmet5` on 3 SkillCorner matches, in memory |
| `a7_v15stages.py` | `netmet`'s P with v15's own stage models (the second-order check) |
| `a9_seeds23.py` | seeds 2 and 3 alone through trial 1015 (information) |
| one-liners | the Metrica games' distinctness, the stress pools and templates, the caches' dates, the SkillCorner even/odd split |

**Not checked:**

- The training itself was read, not re-run. The builder's bit-for-bit fit check stands for it.
- The switch tables (information only on the page).
- Inputs built before the night, shared with v15.

Where [ideas_network.md](ideas_network.md) quotes a figure about our match's tracking, I read past it. Nothing here
uses it.
