# Night ideas: the output side (where branch B still loses spellF1)

| | |
|---|---|
| **who** | one of four idea agents, 2026-11-14, 23:10–00:15. No model built, nothing adopted, nothing read or run on our match |
| **angle** | branch B's decoder and post stages: where B's spellF1 is lost on B's bench, and which output-side changes can still reach it |
| **what was run** | B as it deploys (search trial 1015, the `search_b confirm` protocol, every stage model trained without the held-out match on the stressed training copies) recomputed on all 30 units of B's bench (10 matches × p25 / p50 / stressed) from the cached P and stage tables: **0.7329, every unit equal to `confirm_1015_…csv`**. Then truth oracles and cheap screens on those letters. ~3 min to rebuild the letters (5 more for the four other nested picks), ~15 min for every screen together, one process at a time, no shared cache written |
| **scripts** (throwaway) | `/private/tmp/claude-501/-Users-joseruiz-Code-more-than-a-hack-2027/39e3effe-3bf8-4611-8584-0402cfeb00b7/scratchpad/ideas/`: `dump_b.py` (B's letters at each stage, `unit_<match>_<variant>.npz`), `anatomy.py`, `anatomy2.py`, `bias.py` (the anatomy), `screen_p.py`, `owner_screen.py`, `fp_screen.py`, `seg_auc.py`, `seg_act.py`, `recall_screen.py`, `timing_screen.py`, `timing_ctx.py`, `short_screen.py`, `ceilings.py`, `dump_trials.py` + `consensus.py` (the screens) |

**The short answer.** B misses short possessions. The truth has 411 possessions per unit; B predicts 351 and
matches about 310 of the true ones at IoU 0.3 and 280 at 0.5. **The largest single pot is short true possessions sitting inside a
predicted possession of the other team (+0.138 if all were written in).** The output side can't reach most of
it: on 97 % of them B's start P is below 0.5 for the right team, 40 % never see the ball, and only 11 % ever show a
player of the right team clearly on it. Every output-side screen run tonight gains between +0.000 and +0.003 (in
sample). **So my ranking is short: one cheap run worth a slot tonight (`recall` as a 4th stage and `timing`, both
given B's start P: about +0.002 to +0.004, ~15 % to clear the bar), one speculative refiner, a hook for the input
side's new signals, and a list of ideas measured dead (MBR, consensus, segments+P, clean owner, temperature,
ball-exit snap). The big remaining pot is input evidence for contested short possessions (the other agents'
side).**

## 1. The error anatomy (B's bench, trial 1015's final letters, mean of 30 units)

**Each oracle fixes one kind of error with the truth and nothing else; its gain is that kind's share of what's
left.** The oracles overlap, so they don't add up. (Exploration: nothing is picked from them.)

| error kind (oracle) | per unit | spellF1 if fixed | gain |
|---|---|---|---|
| **missed true possessions** (no predicted one of its team at IoU ≥ 0.3), written in with the truth's bounds | 98 of 411 | 0.897 | **+0.164** |
| … of them **inside a predicted possession of the other team** (a short interception, a duel) | 51 | 0.871 | **+0.138** |
| … of them inside a predicted stoppage | 15 | 0.759 | +0.026 |
| … by length: under 1 s / 1–2 s / 2–5 s / 5 s and over | | | +0.049 / +0.056 / +0.054 / +0.011 |
| **team-change timing**: each predicted H ↔ A change moved onto the true one of the same kind within 2 s (1 s) | | 0.768 (0.748) | **+0.035** (+0.015) |
| **spurious possessions** (no true match at IoU 0.3) merged into a neighbour | | 0.768 | +0.035 |
| **wrong team** on a whole possession (flipped to the truth's majority) | 30 | 0.767 | +0.034 |
| **the dead-ball side, every frame** (D vs live made right, team left as predicted) | | 0.774 | +0.041 |
| … stoppage starts moved onto the truth within 3 s / restarts within 3 s | | | +0.006 / +0.004 |
| … missed stoppages written in / false stoppages removed | | | +0.005 / +0.005 |
| (upper bound) every live frame's team made right | | 0.945 | +0.212 |

**At the headline IoU 0.5** (131 true possessions unmatched and 71 predicted ones per unit):

| unmatched true possessions (FN) | per unit | median length | | unmatched predicted possessions (FP) | per unit | median length |
|---|---|---|---|---|---|---|
| missed, under the other team | 46 | 1.3 s | | **wrong team** | 30 | 2.2 s |
| **merged**: a predicted possession spans two true ones of the team (the separator between them was missed) | 42 | 2.2 s | | fragment of a true one matched elsewhere | 14 | 3.5 s |
| boundary: one-to-one, IoU 0.1–0.5 | 24 | 2.2 s | | boundary | 11 | 2.0 s |
| missed, under a stoppage | 12 | 1.3 s | | merger of two true ones | 9 | 9.1 s |
| split in two by a predicted blip | 4.5 | 10.4 s | | inside a true stoppage | 6 | 2.0 s |

- **"Merged" is mostly the same error as "missed under the other team".** Of the 228 true A–B–A patterns per unit
  (a short other-team possession between two of the same team), 38 have B erased in the output (median 1.4 s);
  each costs B and one of the two A's. Dead separators (A–D–A, 27 per unit) are erased only 1.8 times.
- **By IoU**: F1 0.813 / 0.784 / 0.740 / 0.694 / 0.635 at 0.3 / 0.4 / 0.5 / 0.6 / 0.7. 22 % of the possessions
  matched at 0.3 have IoU < 0.7 (69 per unit); where their start or end is an H ↔ A change, it is more than 0.5 s
  off in 59 % (starts) and 69 % (ends): the timing pot again.
- **Boundaries have no bias to correct**: the median offset of matched possessions is 0 frames at every kind of
  boundary (team change, stoppage, restart) and every length. A constant shift has nothing to learn.
- **Boundary offsets**: of true H ↔ A changes (311 per unit), 49 % have a predicted change of the same kind within
  0.2 s, 13 % 0.2–1 s, **19 % more than 1 s off, 20 % none within 5 s** (the missed possessions). Restarts are good
  (70 % within 0.2 s, 12 % none within 5 s); stoppage starts 48 % within 0.2 s, 31 % none within 5 s.
- **Where the stages stand**: decoder 0.7223 → short 0.7293 (+0.0070) → segments 0.7306 (+0.0013) → timing 0.7329
  (+0.0023). 99 % of the missed possessions are already missed by the decoder; the stages recover 7 per unit.
- **The wrong-team possessions come from the decoder** (27 of 30 per unit); `short`'s insertions are 12 per unit,
  66 % matched.
- By match, the missed-under-the-other-team pot is +0.13 to +0.27 on DFL's 7 and +0.06 to +0.08 on Metrica's 3; the
  timing pot +0.03 to +0.05 on DFL, +0.01 to +0.02 on Metrica.

### Is the evidence for the missed possessions there? (the question that decides the output side)

| missed true possessions (≤ 5 s), stressed variant | per match | start P for the right team, mean over it | ball ever seen | right team's player within 1 m of the ball | a **clean owner** (right team within 0.5 m, no opponent within 1.5 m) on ≥ 1 / ≥ 3 frames |
|---|---|---|---|---|---|
| **inside the other team's predicted possession** | 50 | ≥ 0.5: 3 %; 0.2–0.5: 50 %; < 0.2: 47 % | 60 % | 40 % | **11 % / 5 %** |
| inside its own team's (merged) | 30 | ≥ 0.5: 90 % | 60 % | 52 % | 26 % / 17 % |
| found short ones (≤ 5 s), for comparison | 147 | ≥ 0.5: 92 % | 79 % | 73 % | 46 % / 28 % |

**The start P can't find them.** Every stretch inside a predicted possession where the other team's P reaches 0.2
(527 per unit) was scored as a candidate: only **3–9 % match a true possession** at IoU 0.3, against a break-even
of about 40–60 % for writing one in (the metric: a right insertion adds about 2 matches, a wrong one splits a
matched possession). This is hsmm.md's finding again on B's much better P: the decoder already does what the P
allows. What's missing is evidence, and the "clean owner" signal is rare exactly where it would be needed.

## 2. The ideas, ranked (expected gain × chance it passes × fits tonight)

### 1. `recall` added as a 4th stage, and `timing`, both given B's own start P ("P-aware rankers")

**What changes.** B's three stages were designed on robust's chain and read PathCRF, v6/v7's hand costs, v4's
P(dead) and touches, **never the start P that B decodes** (0.915 × `gru_b` + 0.085 × `augment`, the best per-frame
evidence there is: log loss 0.151 on J03WMX stressed against `augment`'s 0.179). The idea gives each candidate
a few numbers from that P (over the candidate: P(new team) mean and max, P(old team), P(D); at a timing candidate:
P(new) at the frame, the margin and its slope ±5 frames, the distance to where the P crosses to the new team), and
adds `lab_recall` (proposals from "the other team on the ball", loose touch candidates and PathCRF, each scored by
a classifier) after `timing`. `recall` was never in B's chain; it targets the biggest pot with its own evidence.

**Why, tied to the anatomy.** The missed short possessions (+0.138) and the team-change timing (+0.035) are what
`recall` and `timing` attack. **Their candidate sets already hold real headroom; their rankers don't get it**
(stressed variant, the truth's answers used as the ranker's P; exploration):

| stage, on B's letters | B now | a perfect ranker over the same candidates |
|---|---|---|
| `timing` (B's own, input: B's `seg` letters) | +0.0019 (~110 moves a match) | **+0.021** (+0.013 to +0.032 per match) |
| `recall` after timing (190–380 proposals a match, 10–33 right) | not in B | **+0.035** (+0.012 to +0.058) |

The rankers are information-limited, and the start P is information they don't have. **Screened tonight** (each
stage's own classifier, leave one match out on the stressed variant's own letters, threshold in sample; a screen,
not the B bar):

| stage | as B has it | with the start-P numbers | + per-change context (each candidate against the change's best and mean) |
|---|---|---|---|
| `timing` (window 3 s) | +0.0018 (p_min 0.40), 7/10 | **+0.0025**, 8/10 (+0.0022 to +0.0025 at p_min 0.30–0.50) | **+0.0028**, 9/10 (p_min 0.40) |
| `recall` after timing (new stage) | +0.0010 at best (p_min 0.55), 5/10 | **+0.0017 to +0.0026** (p_min 0.40–0.70), 6–7/10 | |
| `short` (input: B's decoded letters), **left as is** | +0.0072 (p_min 0.34), 9/10 | +0.0066 at 0.34, +0.0078 at 0.40 (10/10): a wash | |
| `segments`, **left as is** | | AUC of P(wrong) 0.807 → 0.837, P(spurious) 0.820 → 0.838, but spellF1 ±0.001 at every threshold: almost no possession crosses the action thresholds | |

**Expected gain on B's bench: +0.002 to +0.004** (recall+P about +0.002, timing+P with context about +0.001 over
B's timing). Confidence that it's positive: moderate (both pieces positive on 6–9 of 10 in the screen). **Chance
of clearing the B bar (+0.005, 7/10): ~15 %.** It's the only output-side change I found worth a slot.

**The test (new files only):**
- `bench/lab_pstage.py`. Reuses `search_b`'s machinery unchanged: trial 1015's settings and start; the stage tables
  built on the **out-of-fold chain of the stressed training copies** (as `search_b.trial_tables`, with
  `post = stressed`), each held-out match's stage models trained on the other 9. The P numbers come from
  `search_b.start_p(base_id, data, 0.9151829…, 0.9094640…)`: for the training copies `data = "train"` (cached:
  `data/external/search_b/p_{augment,gru_b}_train_*.npy`), for the test units the variant (cached). Wrap
  `lab_timing.table` / `lab_recall.table` in the new file to append the columns, and add the new names to a
  feature list in the new file (no edit of the old files; `timing_screen.py`, `timing_ctx.py` and
  `recall_screen.py` in the scratchpad show the columns); `lab_recall`'s model with its `all` features and `hgb`.
  `short` and `segments` stay exactly as in 1015. One thing to add in the new file: `search_b.oof_tables` keeps
  each stage's tables but not the letters after `timing`, which `recall` trains on; copy it there returning the
  out-of-fold letters too (and the `timing` tables rebuilt with the P columns from the same `seg` letters).
- Variants, all fixed before the first score: **(a)** timing+P+context; **(b)** B + recall; **(c)** B + recall+P;
  **(d)** (a) + recall+P. Thresholds: 1015's for timing; recall's p_min picked nested (each match on the other 9)
  from {0.4, 0.5, 0.6, 0.7}. The pick among (a)–(d) is the best on all 10 (as `search_b report`), its nested score
  reported as the honest number.
- `python3 <scratchpad>/slot.py -- .venv/bin/python -m bench.lab_pstage prepare` (tables: ~5–10 min), `… confirm`
  (30 units × 5 chains: ~10 min), `… skillcorner` (deployment models on SkillCorner's 20: ~20 min, as
  `search_b skillcorner`). **One slot**, ~40 min of compute, ~1.5–2 h to build.
- **The bar, fixed before the first score: the B bar** against trial 1015 as confirmed (0.7329): +0.005 or more on
  the mean of p25 / p50 / stressed, better on 7 or more of the 10, no variant down more than 0.002, SkillCorner (20
  matches, deployment models) down no more than 0.005. Then `check_v15_switch`'s four parts, threshold unchanged,
  for information.
- **Cheap:** no GRU rebuild, no new feature cache; everything reads cached P and cached stage tables' inputs.

**Risk it doesn't carry to our match.** The new columns read `gru_b`'s calibration. On our match B's P comes from
the same network on real tracking; if it is less confident there, recall writes fewer possessions and timing moves
fewer changes (a conservative failure). If it's over-confident in a wrong direction, it now steers two more stages
besides the decoder. SkillCorner (real stoppage balls, another convention) is the guard; Metrica's 3 matter most
for the convention. `recall` was +0.0172 on round 2's mix but only +0.0044 (6/10) on robust's chain: its value
shrinks as the start gets better, which the screen above confirms.

**Fits tonight:** yes. Built by ~02:00, judged by ~03:00.

### 2. A learned refiner over B's output (speculative; only if a slot is idle)

**What changes.** One small bidirectional GRU (as `gru_b`, 5 Hz, ±60 s) whose inputs are B's final letters
(one-hot), the start P, and the per-frame evidence the stages compute (touch candidates by each team, the
nearest players' distances, `short`/`recall`/`timing` candidate scores), trained out of fold on the stressed
training copies to give P(H / A / D) again; decoded again with B's decoder and settings. It replaces three
hand-designed stages (+0.0106 together) with one model that sees structure (a possession's length, its neighbours)
and evidence together.

**Why.** The stages are greedy, one candidate at a time; a refiner can trade a merge against an insertion. But
the anatomy says the evidence for most misses isn't in these inputs, so the ceiling is modest.

**Expected gain: −0.003 to +0.006, low confidence. Chance of clearing the bar: ~10–15 %.** **Not testable on cached
outputs alone**: it needs the stages' per-frame evidence on the 10 training copies and the 30 test units (a new
cache, ~30 min), and 8 network folds (~2–3 h on 2 slots). Same B bar, plus SkillCorner with a deployment refiner.
**Risk on our match:** stacking learns the bench's own error patterns, including the synthetic stoppage ball's.
**Fits tonight:** only if started by ~01:30; judged by ~06:00 at best. I'd give the slots to the input-side
ideas first.

### 3. Whatever new evidence the input side builds, fed to `recall`/`timing` as ranker columns (a hook, not a build)

**What changes.** Nothing tonight on its own. If an input-side idea yields a per-frame signal of "who is on the
ball" (a touch/duel detector, a pass detector, a near-player estimate while the ball is unseen), the cheapest way
to cash it on B is as extra columns in `recall`'s and `timing`'s rankers, exactly as idea 1 adds the start P,
**before** retraining any network: the candidate sets hold +0.035 and +0.021 with a perfect ranker, and the
rankers now get +0.001 to +0.003.

**Expected gain:** whatever fraction of the ranker headroom the new signal opens; unknown until that signal exists.
**The test:** idea 1's file with one more column group; same B bar; ~20 min of compute once the signal is cached on
the 10 training copies, the 30 units and SkillCorner's 20. **Risk on our match:** the new signal's own. **Fits
tonight:** yes, if an input-side signal lands by ~05:00 and idea 1's file is built.

## 3. Weighed and not recommended (measured tonight or already tried)

| idea | verdict | why (numbers from tonight unless cited) |
|---|---|---|
| **MBR / expected-spellF1 decoding** from the start P (sampled paths, n-best with a spell scorer) | **don't build** | the paths differ only where the P allows: P-only candidates inside predicted possessions match a true one 3–9 % of the time, against a 40–60 % break-even; writing them in loses at every threshold (−0.004 at P ≥ 0.6, −0.039 at 0.4, −0.27 at 0.25), and flipping or merging low-P possessions loses too (−0.003 to −0.006) |
| **Consensus of the near-tied B chains** (MBR over outputs: the five nested picks 1015, 1133, 1269, 586, 554; the medoid by mean pairwise spellF1, or a 3-of-5 frame vote then runs under 7 frames merged) | **measured: fails** | against 1015 on B's bench (confirm protocol, 30 units): **medoid −0.0014 (4/10), vote −0.0007 (6/10)**, worst variant −0.0019 / −0.0009. Both beat the five's mean (0.7307) but not 1015, the in-sample best. The ceiling (best of five per unit with the truth) is only +0.0050 (7/10) |
| **Short given the start P** | **measured: a wash** | +0.0066 against +0.0072 at 1015's p_min; +0.0078 at 0.40 (in sample) |
| **A per-change context ranker for timing** | **folded into idea 1** | +0.0028 against +0.0025 (P alone) and +0.0018 (as B has it) |
| **HSMM / learned durations** instead of the minimum length | already tried ([../hsmm.md](../hsmm.md)): λ = 0 picked in 7/10 folds; on B the search found min length and switch prices flat (rank correlations −0.07 to +0.03, [../branch-b.md](../branch-b.md)) | the misses aren't a prior problem: the P is against them |
| **Temperature calibration** before the decoder | **a no-op** | Viterbi on −log P / T with switch prices s is the same path as −log P with prices s·T: a temperature is a rescaling of both switch prices, which the search already covered over a wide log range; a per-class bias is `f_dead` (searched, weak) |
| **Ball-exit snap** (a change moved to the frame the ball leaves the last owner) | **don't build** | the labels' convention puts an open-play change at the new team's deliberate touch; a loose ball stays with the last owner ([../../owner-rule.md](../../owner-rule.md)): the exit frame is the wrong target. At restarts (where the exit is right) B is already within 0.2 s on 70 %, and the whole restart pot is +0.004 |
| **Clean-owner head** (who has the ball when it's seen and one player clearly on it) | **low ceiling** | the signal appears in 11 % of the missed possessions under the other team (5 % for 3+ frames); in wrong-team possessions the right team is a clean owner in 7 %, the predicted team in 15 % (right-team short possessions: 5 % / 26 %): little to separate them |
| **Segments given the start P** (spell-level re-scoring of team / spurious) | **measured: nothing** | better AUCs (0.807 → 0.837, 0.820 → 0.838) but spellF1 ±0.001 at every flip/merge threshold: the flip pot (+0.034) is in short possessions the classifier never becomes sure about (9 % base rate) |
| **A boundary shift** (constant offset per boundary kind) | **nothing to learn** | median offset 0 frames at every boundary kind and length |
| **Dead-ball side** (stoppage start / restart / missed / false stoppages) | **small** | each piece +0.004 to +0.006 even with the truth; v9's snap and v11's move were switched off by the search for B (they read the stoppage ball) |

## 4. What this means for the night

- **The output side is close to saturated for B on the evidence B has.** The best I can find tonight is +0.002 to
  +0.004, worth one cheap run (idea 1). Don't spend several slots here.
- **The pot is evidence for short contested possessions**: +0.138 for the missed ones under the other team, +0.13 to
  +0.27 per DFL match. 40 % of them have a right-team player within 1 m of the ball at some frame, 60 % see the ball
  at all, and the start P is against them almost always. Anything on the input side that tells "who touched it"
  in a duel, or where the ball is when it's unseen for a second or two, is what moves this.
- Metrica's 3 matches (the labels' convention) have the smallest pots (+0.06 to +0.08 missed, +0.01 to +0.02
  timing): on the convention that matters most, B is already close to what its evidence allows.
