# Lab candidate `learned`: one learned P(H / A / D) per frame in place of the hand weights

| | |
|---|---|
| **status** | **built 2026-11-04, lab round 1; clears the bar on both bench sets, by far the largest gain so far**: held out spellF1 **0.6610** against v11's 0.5727 (**+0.0882**), better on **10/10**; IDSSE only **+0.0751**, **7/7**. **Read "The bench's dead-time ball" before mixing**: about half of the gain reads how the bench's two data providers show the ball in dead time, and our match may not show it that way |
| **what** | idea 10c: one classifier (gradient-boosted trees) gives P(H), P(A), P(D) for every frame from every signal the match offers; its −log P is the decoder's cost (instead of v1's clues + v4 + v6 + v7's weighted sum); decoder settings re-chosen; then v9's restart snap and v11's touch move, unchanged |
| **stage** | `decode`: replaces v7's costs (v9 and v11 run after it) |
| **code** | `possession/lab_learned_features.py` (the 97 features), `bench/lab_learned.py` (training, decoding, the leave-one-out run), `bench/lab_learned_check.py` (where the gain is; ball in dead time) |
| **run** | `python -m bench.lab_learned [--set idsse]` (~18 min / ~13 min, < 0.7 GB; after `python -m bench.lab --build`), then `python -m bench.lab_learned_check [--set idsse]` (~3 min) |
| **results** | `bench/results/lab_learned[_idsse].json`, `lab_learned_grid[_idsse].csv`, `lab_learned_frames[_idsse].csv` (per-frame accuracy, importances), `lab_learned_split_grid[_idsse].csv`; held-out P per match in `data/external/lab_learned[_idsse]/p_<variant>_<match>.npy` (gitignored) |
| **mix with** | `bench.lab_learned.decode_held_out(match, which)` (the saved held-out P decoded with `CHOSEN`, no refit), then `finish(match, letters, which)`; or `fit(matches, variant="all", which=...)` + `decode(match, model)`. For a decoder-stage mix: `cost_from(match, held_out_proba(match, which))` is the cost matrix |
| **chosen on all 10** | `CHOSEN = {"variant": "all", "blend": 0.0, "switch_team": 8.0, "switch_dead": 50.0, "min_possession": 20}`; the same on IDSSE alone |

## Why

v1's clues are hand-weighted ("frames of evidence"), and each later model (v4 dead ball, v6
restart, v7 PathCRF) was added on top with one weight. The weights were never fitted jointly, and
the ball itself was kept out of the dead-ball model on purpose (v4, decision 3). The error budget
([../ideas.md](../ideas.md), "Where v9's remaining spellF1 is") puts the team of each possession,
spurious and missed possessions at +0.054 / +0.043 / +0.250 if fixed. A classifier trained on the
bench's truth can weigh all the signals together, including their interactions (for example "the
ball is missing and PathCRF is sure").

## How it works

1. **Features** (`possession/lab_learned_features.py`), 97 per frame, from that match's damaged
   tracking and the out-of-sample outputs of v4–v7 on it (each trained without the match). Never
   its truth. All within each half.

   | family | features | why |
   |---|---|---|
   | the ball | seen; s since / until it was last / next seen; speed over the 0.2 s before and after; change of speed; change of velocity (the touch detector's ΔV); change of course (degrees, only above 1 m/s); its x measured from home's goal; distance from the nearer touchline; how far outside the pitch | where play is and whether it moves |
   | players at the ball | each team's nearest player (m), their difference, players within 2 m and 5 m per team; means of the difference and of "this team is on the ball" over the past and next 1 s and 3 s | v1's clue A, without the hand thresholds |
   | players at the last / next seen ball | each team's nearest to where the ball was last seen and will next be seen | who had it before a gap and who has it after |
   | the teams | mean speed, spread, how far up the pitch (from its own goal), per team; mean speed of all | the team with the ball spreads and pushes up |
   | touches, passes | per team: s since / until its last / next touch (v11's detector, ΔV 3) and pass (v9's), touches and passes in the past and next 3 s | v9's and v11's evidence, as inputs |
   | earlier models | P(dead) (v4), P(home restarts) and the restart clue (v6), PathCRF's P(home) and its 1 s / 3 s means, v6's and v7's three decoder costs, and their team and dead differences over 1 s / 3 s | everything the chain already knows |

2. **The classifier** (`bench/lab_learned.py` `fit`): scikit-learn's HistGradientBoostingClassifier,
   3 classes, trained on every 10th graded frame of the **IDSSE** matches among the training matches
   (6 per IDSSE fold, all 7 for a Metrica fold), each row also with the teams exchanged (labels
   H ↔ A, PathCRF's and the restart's P(home) → 1 − P, home's columns ↔ away's). At prediction, P is
   the mean of the match as it is and with the teams exchanged, then clipped to 0.01–0.99 and rounded
   to 3 decimals, as v4.
3. **The cost:** −log P per class (optionally blended with v7's costs), decoded by v3's min-length
   Viterbi with the grid's settings; then v9's snap and v11's move with the harness's settings.
4. **Judged** by `lab.judge` over the whole grid (leave one match out against v11).

## Every setting, and why

| setting | value | why |
|---|---|---|
| training frames | every 10th graded frame (0.4 s apart) | neighbours are near-copies; ~171k rows (with the exchanged copy) fit in ~45 s and ~0.6 GB |
| **training matches** | **IDSSE only** | Metrica's samples blank the ball in dead time: on the bench their ball is missing on **94–99.5 %** of dead frames against 9–14 % of live ones, so a model fitted on them learns "no ball = dead" (decision 3; ideas.md row 2: learn "no ball" on IDSSE damaged like ours). Decided after a one-fold timing test (J03WR9, trained on all 9 other matches) looked too good, before any leave-one-out run; on the rule, not on a score. `--train-all-sources` runs the other choice for information |
| teams exchanged | trained on both, answers the mean | a bench whose home teams happen to have the ball more mustn't teach "home has it" |
| trees | 200 rounds, learning rate 0.1, 31 leaves, ≥ 200 rows a leaf, L2 1.0, no early stopping, seed 0 | scikit-learn's defaults except: more rounds (200 against 100) for 97 inputs; bigger leaves (200 against 20) because rows of one match are correlated, so a leaf should describe many moments; no early stopping, so every row is used and the fit is repeatable (the first fold gave identical numbers on two runs) |
| clip, rounding | 0.01–0.99, 3 decimals | as v4: a frame costs at most 4.6, and a last-digit difference can't flip a label |
| distances | on the ball 2 m (v1), close 5 m, nobody seen = 60 m, difference clipped at ±20 m | v1's distance; beyond these the exact value says nothing more |
| since / until | touches and passes clipped at 10 s, ball seen at 30 s | longer than any possession's typical touch gap / most ball gaps (median 10.7 s) |
| velocity window | 0.2 s (5 frames), course only above 1 m/s | the touch detector's window; below 1 m/s the course is jitter |
| context windows | past and next 1 s and 3 s | about a touch and about a short possession |
| **variant** | `all`, `no_pathcrf` (no PathCRF P, no v7 costs: 81 features) | PathCRF was trained on 6 of the 7 IDSSE matches, so the model may over-trust it; the second variant shows what's left without it |
| blend | 0, 0.5 (share of v7's costs; v6's for `no_pathcrf`) | whether the old costs still add something |
| team switch price | 0.5, 1, 2, 4, 8, 16 | v3's 0.5 up: −log P differs between the teams by up to 4.6 a frame, far more than v1's clues, so a higher price may pay |
| dead switch price | 50, 25, 10 | v4's 50 first |
| minimum possession | 20, 10 frames | v3's 0.8 s, and half |
| minimum dead | 1 frame | v4's |

144 settings, fixed before any held-out result; simplest first (`all`, no blend, v3's prices).

## Results (2026-11-04)

### The bar

| | all 10 matches | IDSSE only (7, its own chain and folds) |
|---|---|---|
| **held out spellF1** | **0.6610** against 0.5727 (**+0.0882**), better on **10/10** | **0.6376** against 0.5625 (**+0.0751**), **7/7** |
| IDSSE | 0.6432 against 0.5673 (+0.0760), 7/7 | as above |
| Metrica | 0.7023 against 0.5855 (+0.1168), 3/3 | — |
| **PathCRF-unseen** | **0.6919 against 0.5842 (+0.1078), 4/4** | J03WR9: 0.6457 against 0.5790 (+0.0667) |
| macroF1 | 0.9107 against 0.8705 | 0.9161 against 0.8757 |
| chosen | all, no blend, team 8, dead 50, min 20 (every fold: `all`, no blend, team 8; dead 50 or 10; min 20, one fold 10) | the same (folds: team 4 or 8, min 20 or 10) |
| **passes** | **yes** (all four checks) | **yes** |

Per match (all 10; each with the setting chosen on the other 9):

| match | held out | v11 | gain | macroF1 / v11 | possessions (true) | D F1 / v11 | team right, both alive / v11 |
|---|---|---|---|---|---|---|---|
| J03WMX | 0.5726 | 0.5100 | +0.0626 | 0.920 / 0.878 | 321 (511) | 0.973 / 0.922 | 0.911 / 0.907 |
| J03WN1 | 0.5928 | 0.5282 | +0.0646 | 0.892 / 0.861 | 361 (500) | 0.972 / 0.942 | 0.879 / 0.876 |
| J03WOH | 0.6691 | 0.5676 | +0.1015 | 0.921 / 0.871 | 313 (402) | 0.975 / 0.923 | 0.914 / 0.907 |
| J03WOY | 0.6589 | 0.6045 | +0.0544 | 0.904 / 0.870 | 355 (456) | 0.957 / 0.920 | 0.912 / 0.906 |
| J03WPY | 0.6746 | 0.6181 | +0.0565 | 0.927 / 0.896 | 316 (421) | 0.978 / 0.937 | 0.923 / 0.927 |
| J03WQQ | 0.6738 | 0.5624 | +0.1114 | 0.921 / 0.876 | 344 (401) | 0.967 / 0.912 | 0.922 / 0.914 |
| **J03WR9** (unseen) | 0.6609 | 0.5801 | +0.0808 | 0.929 / 0.887 | 297 (370) | 0.975 / 0.927 | 0.929 / 0.923 |
| metrica1 (unseen) | 0.7092 | 0.5934 | +0.1158 | 0.911 / 0.858 | 307 (354) | 0.937 / 0.888 | 0.940 / 0.917 |
| metrica2 (unseen) | 0.6923 | 0.5811 | +0.1111 | 0.879 / 0.843 | 265 (318) | 0.881 / 0.852 | 0.943 / 0.924 |
| metrica3 (unseen) | 0.7054 | 0.5821 | +0.1233 | 0.902 / 0.865 | 308 (381) | 0.897 / 0.858 | 0.944 / 0.923 |
| **mean** | **0.6610** | 0.5727 | **+0.0882** | 0.911 / 0.871 | **319** (411; v11 349) | **0.951** / 0.908 | 0.922 / 0.912 |

- **Per-frame accuracy** (argmax P, held out): **0.919** against v7's 0.872 (v11 0.878); IDSSE
  0.928 / 0.883, Metrica 0.897 / 0.848. Without PathCRF: 0.904.
- **Fewer possessions, not more:** 319 a match against v11's 349 (true 411). The gain is cleaner
  boundaries (dead ball found: D F1 0.951 against 0.908) and fewer spurious possessions (team price
  8), not the missed short ones.
- **v9 and v11 add almost nothing on top:** at v7's level the learned decode is already 0.6599
  (v7: 0.5557); the snap and the move add +0.001 (on v7 they add +0.017). Post-decoder candidates
  may add less on top of this one than on v11.
- **Not the decoder's re-choice:** with v7's own decoder settings (team 0.5, dead 50, min 20) the
  learned costs already give 0.6518 in sample. Team price: 0.5 → 0.652, 2 → 0.656, 4 → 0.661,
  **8 → 0.663**, 16 → 0.645.
- **Blending v7's costs back in hurts** (best 0.657 at 0.5 against 0.663 at 0).
- **Without PathCRF it fails**: `no_pathcrf` alone 0.5870 (+0.0142, 6/10) on all 10, but −0.0122 on
  IDSSE (3/7) and −0.0114 on the IDSSE set (3/7). PathCRF is worth ~+0.07 inside the learned model,
  **and it holds on the 4 matches PathCRF never saw** (best setting of each: +0.02 to +0.06 on
  Metrica's 3, +0.09 on J03WR9, over `no_pathcrf`).
- **Importances** (rise in held-out log loss when one family is shuffled, mean of the 10 folds, `all`):
  the ball **1.52**; v6's costs 0.26; PathCRF 0.034; v7's costs 0.031; touches 0.020; players at the
  ball 0.017; the rest ≤ 0.003. Correlated families hide each other (v6/v7 costs, PathCRF), so only
  the order means much. **The ball dominates: see below.**

### The bench's dead-time ball (the main caveat)

`python -m bench.lab_learned_check`. Each provider's feed shows dead time in its own way, and the
model reads it:

| | ball missing, dead frames | … live frames | seen in dead time: still (< 0.3 m/s) | seen live: still |
|---|---|---|---|---|
| IDSSE (7) | 51–58 % | 27–35 % | **99.0–99.4 %** | 1.0–2.8 % |
| Metrica (3) | **94–99.5 %** | 9–14 % | (rarely seen) | 0.8–1.6 % |

- **IDSSE freezes the ball in dead time:** whenever it's seen while dead, it stands still (median
  0.07 m/s, the jitter we add), often outside the pitch (27–70 %). In a real stoppage the ball is
  fetched, carried, thrown back. **Metrica blanks it.** Either way, on the bench the ball stream
  says "dead" almost perfectly, and the trees learn it (the ball family's importance, 1.52).
- **Ours:** the ball is missing 83 % of the time players stand still against 29.5 % while they
  move (bench.md) — a stronger "no ball" hint than IDSSE's damaged 65.5 % / 34.6 %, so the "missing"
  half should transfer. **Whether our seen ball freezes in dead time is unknown.** A label-free check
  Jose could run on our tracking (not run by Claude): the share of seen-ball frames with the ball
  still (< 0.3 m/s over 0.2 s) inside v1's "players stopped" stretches and outside them. Bench
  reference: IDSSE **98.0 % / 25.9 %**, Metrica 23.3 % / 1.2 % (its stopped ball is mostly missing).
  If ours is near IDSSE's, the dead-ball half of this gain should carry over; if our ball moves in
  stoppages, trust only the team half below.

**The gain split in two** (diagnostic, built after round 1 and reported as such; `split_costs` in
`bench/lab_learned_check.py`; each judged leave one match out on its own small grid):

| | all 10 | … IDSSE | … Metrica | … PathCRF-unseen | macroF1 | IDSSE set (7) |
|---|---|---|---|---|---|---|
| **team only**: v7's dead vs live + the learned team | **+0.0515**, 10/10, passes | +0.0331, 7/7 | +0.0944 | +0.0843, 4/4 | 0.881 | **+0.0361**, 7/7, passes |
| **dead only**: the learned dead vs live + v7's team | +0.0473, 10/10, passes | +0.0454, 7/7 | +0.0518 | +0.0501, 4/4 | 0.901 | +0.0388, 7/7, passes |
| both (the candidate) | +0.0882, 10/10 | +0.0760, 7/7 | +0.1168 | +0.1078, 4/4 | 0.911 | +0.0751, 7/7 |

The team half leaves v7's dead-ball evidence untouched, so it can't read the frozen ball: its
+0.033 on IDSSE (7/7; +0.036 on the IDSSE set) is the part least exposed to the providers' conventions. Metrica's larger team
gain comes with its ball seen on 86–91 % of live frames (ours: ~70 %), so IDSSE's number is the
closer guide for our match.

## Failure modes and caveats

1. **The dead-time ball (above).** The biggest risk. Half the gain may not transfer.
2. **PathCRF in sample.** 5 of the 6 training matches of every IDSSE fold have PathCRF answers from
   its own training data, so the model learns to trust it more than it deserves on unseen matches.
   The 4 unseen matches hold (+0.108, 4/4), and J03WR9 alone (+0.081) is the only match where every
   input is honest *and* the ball is IDSSE's.
3. **Inputs on our match must be computed as on the bench.** The bench's v6/v7 costs have no Smart
   Tagging hints, no kick-off rule and no keeper repair (v8); v10 on our match has all three. A port
   should build the cost features exactly as `bench/eval_pathcrf.py` does (no tags), and the model
   should be trained on all 7 IDSSE matches.
4. **Fewer possessions.** It merges more than v11 (319 against 349, true 411); short possessions
   (idea 10b) are still missed.
5. **The setting choice leaks a little:** each match's setting is chosen on the other matches'
   scores, whose models were trained with that match (the same as v4's procedure). The choice is
   flat (all folds agree on the variant, blend and team price), so this can't explain the gain.
6. **Only 41 % damage:** the lab caches one level, so the 30 % / 55 % check isn't done.
7. **Winner's curse:** 144 settings, but the gain is 40× the "fragile" line (+0.002) and 10/10.

## Verdict

**Keep: it clears the bar on both sets, 10/10 and 7/7, and on the unseen matches.** Mix it first
(largest gain). But the bar can't see the bench providers' dead-time conventions: before it goes into
a submission, Jose's label-free "still ball while stopped" check on our tracking decides whether to
use the whole model or only its team half (`team_only`, +0.052 on all 10, +0.033 on IDSSE, 10/10).
The team half for mixing: `bench.lab_learned_check.split_costs(match, held_out_proba(match, which),
"team_only")`, decoded at team 8, dead 25, min 20 on all (team 4, dead 25 on IDSSE alone), then
`finish`.

**For information, not judged:** `python -m bench.lab_learned --train-all-sources` (Metrica's
matches in the training too; `bench/results/lab_learned_trainall.json`): 0.6875 (+0.1148), with
**Metrica jumping to +0.187** (from +0.117) and IDSSE to +0.084 (from +0.076). That jump is the
blanked ball being learned as "dead", the reason the classifier learns from IDSSE only.
