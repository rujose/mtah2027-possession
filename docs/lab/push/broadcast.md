# Push track 3, "broadcast": the broadcast's players near the ball, merged into the tactical tracking

| | |
|---|---|
| **status** | plan, simulator, merge rule and bar fixed 2026-11-15, 10:45 (machine clock, UTC−5), before any score. **16:00: on v16's B (the coordinator's 12:35 addition), the registered rule M passes every part of the bar without retraining: central +0.0081 (8/10), pessimistic −0.0018 (floor −0.002) and −0.0010 on a second draw.** On v15's B it fails robustness (−0.0047). The post-hoc M2 (a team never has more than 11) keeps central (+0.0076, 8/10) and is safe on both pessimistic draws (−0.0005, +0.0007). A deployable module exists (`possession/broadcast_merge.py`). The retrain on merged data (v15's recipe) adds nothing (+0.0026, 5/10; the no-retrain run on the same bench +0.0055). **Finished 19:20.** Nothing adopted |
| **question** | Would players seen by the broadcast tracking near the ball, merged into the tactical tracking, improve branch B (the chain our match runs)? The long shot of the push ([../push.md](../push.md)): a clear negative, well documented, is an acceptable outcome |
| **code** | `bench/lab_broadcast.py` (new: the simulator, the merge, the harness on lab_trackerr's machinery, the report) |
| **results** | `bench/results/lab_broadcast/`; data `data/external/lab_broadcast/` (gitignored) |
| **nothing here reads our match** | the broadcast is **simulated** on the 10 bench matches from their clean tracking, with only the statistics [../../data.md](../../data.md) ("6–9. The broadcast set") gives for our match's broadcast set. The merge module, if one is built, is never run on our match except to print paths, SHA-256 or row counts |

## What is known before this page

- **Where the room is** ([../cleaning.md](../cleaning.md), round 9): the missing players at their true positions within 15 m of
  the true ball (`o_near`) gain **+0.0216 on the start, 10/10** (trees retrained per cleaning; the true players
  everywhere +0.0241). Nothing built from the tactical tracking reaches it (24 link/fill settings −0.003 to −0.008).
- **Ghosts don't reach it either** ([../night/ghost.md](../night/ghost.md)): pointers to where a missing player was last seen
  reach about a tenth of it. 82 % of `o_near`'s gain is at moments the ball is seen, 79 % through robust's own
  features; it sits on frames whose possessor is missing (38 %) and on loose-ball frames (45 %).
- **Wrong teams near the ball are expensive** ([../tracking-errors.md](../tracking-errors.md)): 1 % of outfield frames in
  the wrong team costs B's bench −0.018 (0/10), 80 % of it where the wrong player is within 15 m of the ball. So a
  merge that adds a wrongly-labelled player beside the ball can lose more than the fill wins.
- **Our match's broadcast set** (data.md, the only facts used here): the same formats as the tactical files; 51
  slots with team labels (Barcelona 26, Athletic 25); **mean 11.5 players visible per frame** (tactical 20.5);
  **4,641 tracklets, mean 16 s** (tactical 68 s); **homography valid on 65.8 %** of frames (the rest replays,
  close-ups, crowd); the ball in the same pitch coordinates; lined up by `tools/align_broadcast.py` with one fixed
  shift per half (within ±2 frames in every 2-minute window of the 1st half, 96 % of the 2nd's, one window off by
  40 frames); the two balls sit a median **0.80 / 0.74 m** apart at the shift.

## The simulator (fixed before any score)

On each bench match, from its **clean** tracking (external data; never an input of the program), a broadcast view
as the broadcast files would give it after alignment. Seeded per match, setting and step.

1. **Time.** Each 2-minute window of a half shows the clean frame t + δ, δ uniform in [−dt, dt] frames (the
   alignment's residual); in the pessimistic setting 4 % of the windows are off by 40 frames.
2. **Invalid frames** (replays, close-ups, crowd): a share of the halves' frames. Most come in blocks anchored at
   stoppages: each true dead stretch (the bench's D labels) of at least 4 s may open a block from its start +
   U(1, 3) s to its end + U(−3, +1) s (the director usually cuts back to live just before the restart); blocks are
   drawn at random until the anchored share is reached. The rest are random blocks (lognormal length, median 5 s)
   anywhere in play. Nothing is reported on an invalid frame.
3. **The camera**: a window centred on the true ball followed with an exponential lag τ (held where the ball is
   missing), square on the pitch (x-extent = y-extent; a side camera's wide shot covers about as much depth as
   length), its size times a slow zoom (AR(1) of the log size, sd 0.15, 10 s). **Its base size W0 is calibrated** so
   the mean number of players inside per **valid** frame is the setting's target (pooled, 10 matches).
4. **Tracklets**: each player's in-window runs (gaps of up to 12 valid frames bridged: a player flickering across
   the picture's edge stays tracked); runs shorter than 1 s not reported; extra breaks to reach the target mean
   length (rate calibrated; 0 if the window already cuts them shorter), each followed by a geometric gap (mean 8
   frames). Tracklets are dealt per labelled team into 26 / 25 anonymous slots, each to the slot free longest
   (degrade's rule).
5. **Position noise** (per axis): the homography's common offset (AR(1), 2 s) sd_h, its scale/shear error sd_e (m
   per m from the camera's centre; AR(1), 5 s), a constant per tracklet sd_p (foot point, partial occlusion), a
   per-frame jitter sd_j. Pessimistic adds outlier episodes (3 % of valid frames, 2 s each, a common offset with
   sd 5 m per axis: "error up to 303.8" in data.md says the homography sometimes fails badly).
   **Why these sizes:** data.md's two balls sit a median 0.77 m apart at the shift; for two independent isotropic
   errors that is a combined sd of about 0.65 m per axis, and it includes the tactical ball's own error, the ball's
   height above the pitch and the ±2-frame residual. Central puts the broadcast player error at that bound
   (sqrt(0.55² + 0.30² + 0.15²) ≈ 0.65 m per axis, median distance ≈ 0.77 m), optimistic at about half, pessimistic
   at about twice, with outliers.
6. **Team-label errors**: each tracklet's label is the other team's with probability team_err.
7. **A false person** (pessimistic only): a referee-like track 10 m from the lagged ball, reported on 20 % of the
   valid frames when inside the window, team label random per tracklet. data.md doesn't say whether the referees
   are among the 51 slots; if they are, every merge will add them.

| setting | players inside per valid frame (W0) | invalid | of it anchored at stoppages | tracklet target | lag τ | sd_h / sd_p / sd_j / sd_e | time residual | team errors | other |
|---|---|---|---|---|---|---|---|---|---|
| **central** | **11.5** (44.5 m) | **34 %** | 75 % | 16 s | 0.5 s | 0.55 / 0.30 / 0.15 / 0.010 | ±2 frames | **1 %** | |
| optimistic | 17.0 (69.1 m) | 30 % | 85 % | 20 s | 0.3 s | 0.30 / 0.20 / 0.10 / 0.005 | 0 | 0.5 % | |
| **pessimistic** | **9.0** (34.9 m) | **40 %** | 50 % | 10 s | 1.0 s | 1.00 / 0.50 / 0.30 / 0.020 | ±2, 4 % of windows +40 | **2.5 %** | outliers 3 %; false person 20 %; zoom sd 0.25 |
| `ideal` (diagnostic) | central's camera, every frame valid | 0 | | | 0.5 s | 0 | 0 | 0 | |

**What "11.5 players visible" means here (decided 10:30, after one calibration and before any score).** Read as a
mean over *all* frames with nothing tracked on the invalid third, it means 17.4 players inside on every valid frame:
the calibration needs a 71 m square window for that, most of the pitch, which a camera that follows the ball does
not show. data.md's other numbers agree with either reading (4,641 tracklets × 16 s = 11.3 players per frame over
all frames, so the invalid frames would have to be nearly empty, as data.md says, *if* 11.5 counts all frames).
So **central takes 11.5 per valid frame** (the brief's "mean number inside ≈ 11.5"), and **optimistic covers the
other reading** (17 per valid frame, 11.9 over all frames). Pessimistic takes 9.

**The tracklets come out shorter than our broadcast's.** The window's edges and the invalid blocks alone cut the
simulated tracklets to a mean of 9.4 s (central), 16.9 s (optimistic), 7.1 s (pessimistic), so no extra breaks are
added (the calibrated rate is 0). Our broadcast's 16 s says its tracker holds players longer than this window
model does. The merge works frame by frame, so tracklet length only sets how often a team label is redrawn and the
per-tracklet offset: it should matter little; it is a difference, reported.

Calibration (`python -m bench.lab_broadcast calibrate`, `data/external/lab_broadcast/calibration.json`):

| setting | W0 | inside per valid frame | over all frames | valid frames | mean tracklet |
|---|---|---|---|---|---|
| central | 44.5 m | 11.5 | 7.6 | 66.0 % | 9.4 s |
| optimistic | 69.1 m | 17.0 | 11.9 | 69.9 % | 16.9 s |
| pessimistic | 34.9 m | 9.0 | 5.4 | 59.9 % | 7.1 s |

## The merge rule (fixed before any score)

**Rule M (the candidate)**, per valid broadcast frame, on the tactical tracking as delivered (before v8's repair,
which then runs on the merged tracking as v12.inputs runs it):

1. **Registration.** For each broadcast player, the nearest tactical player of either team; over those within
   5 m, if there are at least 3, the median of the vectors (tactical − broadcast) is the frame's shift, added to
   every broadcast player. (The homography's error is mostly common to the whole picture; 93 % of the broadcast's
   players are also in the tactical tracking, so they register it. This takes out the common offset and most of
   a time residual.)
2. **Gate.** If, after the shift, the median distance from those players to their nearest tactical player is above
   **1.5 m**, the frame is dropped (the homography or the alignment failed there).
3. **Add** each broadcast player with **no tactical player of his own label's team within 2.5 m** and **none of
   the other team within 1.5 m**. No distance limit to the ball (D = none): the window already follows it, and in
   round 9 the true players far from the ball still added a little (`o_players` +0.0241 against `o_near` +0.0216).
4. **Pieces.** Each run of added frames of one broadcast slot becomes a piece, dealt to a slot of its team that is
   empty from one frame before to one frame after it, the one free longest (degrade's rule; lab_trackerr's flips);
   a new slot is appended only if none is free.

**Why the radii and the other-team guard** (measured on the bench before any score: geometry, no score; the merge
never sees the truth): among missing players inside a 45 m window, a visible teammate is within 2.5 m of 7.7 % of
them (lost to the same-team test; 10 % at 3 m), an opponent within 1.5 m of 8.4 % (lost to the guard). Without the
guard, **every team-label error on a player the tactical tracking already has** becomes a player of the wrong team
beside him (the same-team test looks in the wrong team): at central's 1 % errors about 0.12 wrong players a valid
frame, all in the camera's window, the kind of error that cost −0.018 at 1 % in tracking-errors.md. After
registration the broadcast–tactical residual is about 0.4 m (central; measured on the simulation), so a wrongly
labelled seen player falls within 1.5 m of himself almost always, and his true duplicate within 2.5 m almost always.

**`literal` (a diagnostic, not a candidate): the brief's rule as written**: no registration, no gate, a broadcast
player is added if no tactical player of his team is within **3 m**. Screened only, to show what registration and
the guard are worth.

R, the guard and the gate are **fixed here**, not chosen on scores.

## The harness

`bench/lab_trackerr.py`'s machinery, in isolated copies of b16_c0 (the verified rebuild of v15's B), as the
tracking-errors page describes: the bench's damaged tracking (b16_c0's lab cache) → **the merge** → v8's keeper
repair → every input computed from the tracking recomputed out of fold (v4's P(dead), v6's restart P, cost and
letters, PathCRF's P, v7's cost, v9's and v11's letters; the fold dead-ball and restart models are lab_trackerr's,
trained on the other 9 matches' unmerged rows) → lab_gate's variants (p25, p50, stressed) → robust's 97 features →
`augment`'s fold trees and `gru_b`'s fold nets → trial 1015's decoder and post stages. **B's bench** = 10 matches ×
(p25, p50, stressed), the mean of a match's three, then of the 10 (v15's B: 0.7329).

- **Reproduction first:** `off` (merge switched off, the same copy and steps) must give every recomputed input
  bit-identical to b16_c0's lab cache and all 30 units within 1e-5 of round 8's confirm CSV (0.7329).
- **Near/far:** frames where an added piece is within 15 m of the true ball ("near"), frames with an added piece
  but none that near ("far"), frames with none; frame accuracy and the start's log loss in each, against `off`.
- **The view's statistics** per setting (players per frame, valid share, tracklets, error, team errors, what the
  merge added: missing players with the right team or the wrong one, duplicates, false persons; how many of the
  missing near-ball players `o_near` filled it covers). Counting uses the truth; the merge never does.

## Stage 1, the screen: v15's B as trained, on the merged bench

Settings `central`, `pessimistic`, `optimistic` (rule M), diagnostics `ideal` (rule M) and `literal` (central's
view). Every model is v15's (b16_c0's), never retrained. Reported against `off` with the B bar's parts 1-3.

**The gate to stage 2 (fixed now):** the retrain runs if `central` gains ≥ 0 on B's bench **or** `ideal` gains
≥ +0.005 (there is room a retrained B might reach). Otherwise the screen is the verdict.

## Stage 2, the judge: B retrained on merged training data

The central simulation and rule M applied to **every** bench match the same way; branch B rebuilt in an isolated copy
as the c0 rebuild was (stage3.sh's steps on b16_c0's code: the stressed copies, robust's features, `gru_b`'s 8
fold nets, `augment`'s trees, lab_gate's variants, the post-stage tables on the training-stress copies, trial 1015's
settings unchanged, no new search), every model trained leave one match out on merged matches. The upstream
dead-ball and restart models stay lab_trackerr's (they are not B's), as in the screen. If the retrain can't finish in
time, the no-retrain version (stage 1) is judged with the same bar, and the page says so.

**The bar (from the brief, fixed now):**

| | needed |
|---|---|
| 1. B's bench, central simulation | **+0.005 or more** against v15's B (≥ 0.7379), **better on 7 of 10** |
| 2. each variant (p25, p50, stressed) | **none down more than 0.002** |
| 3. each source (IDSSE's 7, Metrica's 3) | **neither down** on the mean |
| 4. robustness, pessimistic simulation | the **same** retrained models on the pessimistic merged bench: **no worse than v15's B by more than 0.002** (≥ 0.7309) |
| 5. robustness, broadcast absent | the result **equals v15's**: the deployed module falls back to the delivered tracking and v15's models when the broadcast can't be merged (files missing, alignment failed), so this holds by construction and is checked by `off`. For information: the retrained models on the unmerged bench |
| 6. SkillCorner's 20 (no tactical counterpart), merge off | **0.4085** (v15's models, by the fallback). For information: the retrained deployment models there |
| noise rule | seed noise on B's bench is 0.0035: a pass by the single retrained `gru_b` counts only if `gru_b` retrained with seed 1 passes parts 1-4 too |

## Run plan (as planned at 10:45; what actually ran, and why the order changed, is in the log)

| when (machine clock) | what | slots |
|---|---|---|
| 10:45 | `off` (reproduction), then the view's statistics per setting | 2 |
| ~11:15-13:00 | screen `central`, `ideal`, `pessimistic`, `literal`, `optimistic` (each ~15 min) | 2 |
| after the screen, if its gate passes | the merged lab cache, then the rebuild (~2-2.5 h on 2 slots), then its scores on central, pessimistic, unmerged, SkillCorner | 2 |
| if it passes | seed 1 of `gru_b`; the deployable module (`possession/broadcast_merge.py`) | 1-2 |
| 11-16 12:00 | hard stop, page finished either way | |

## Log

- 10:20 started; read push.md, cleaning.md, ghost.md, tracking-errors.md, branch-b.md, data.md's broadcast section.
- 10:30 geometry on the bench (no score): dead share 42 % of the halves' frames; players in a square window vs size;
  distances from missing in-window players to visible teammates and opponents (the radii above).
- 10:33 first calibration on "11.5 over all frames": a 71 m window. Re-read as 11.5 per valid frame (above).
- 10:36 calibration: central 44.5 m, optimistic 69.1 m, pessimistic 34.9 m; tracklets 9.4 / 16.9 / 7.1 s.
- 10:37 functional test on J03WR9 and metrica1 (counts only, no score): central adds 0.51 players a frame (0.77 a
  valid frame), all but 0.003 of them missing players with the right team, no duplicates; error 0.77 m median
  before registration, 0.42 m after; covers 70 % of the missing near-ball players `o_near` filled.
- 10:45 page written; everything above fixed before any score.
- 10:39-11:03 **`off` reproduces v15's B**: every recomputed input bit-identical to b16_c0's lab cache on 10/10 matches;
  all 30 units within **4.8e-6** of round 8's confirm CSV; B's bench **0.7329**. (Slots: one free until 10:55.)
- 10:48 screen queued: central, ideal, pessimistic, literal, optimistic.
- 11:18 **`central` screened (v15's models, no retrain): B's bench 0.7383, +0.0055, better on 7/10**; p25 / p50 /
  stressed +0.0053 / +0.0053 / +0.0058; IDSSE +0.0020, Metrica +0.0135. The screen's gate (central ≥ 0) passes:
  the retrain runs. Upstream inputs checked on J03WPY and J03WQQ (lab caches, off against central): v4's P(dead)
  slightly better (log loss −0.002 to −0.004), PathCRF −0.2 / −0.7 points, v7-v11 letters ±0.5 points: the effect
  is in B's own models.
- 11:25 queue reordered (no setting changed): ideal (running), **pessimistic**, then the retrain (stage 2), then the
  retrained B through the harness (central, pessimistic, off), then the screen's noise check, then literal, optimistic.
- 11:25 **the noise rule applied to the screen (fixed before it runs):** the night's `netens` trained `gru_b` with
  seed 1 (same recipe, folds, data) and refitted trial 1015's post-stage tables on it
  (`data/external/lab_netens/seed1/`). The harness runs `off_s1` and `central_s1` with those nets and tables and
  everything else v15's. `off_s1` should give the network page's seed-1 B (0.7294, −0.0035). The screen's result
  counts as a pass only if `central_s1` also clears parts 1-3 against `off_s1` (+0.005, 7 of 10, no variant down
  more than 0.002, neither source down).
- 11:45 the coordinator: track 1 (v16) is starved; **at most 1 slot** until it is done (1-2 h). The queue now takes
  tokens up to a cap read live from a file (1 now, 2 when released) and runs the screen's verdict first:
  `pessimistic`, `off_s1`, `central_s1` (~25 min each on one slot), then the retrain (~4 h on one slot, ~2.5 h on
  two), then the retrained B through the harness, then `literal`, `optimistic`. The screen's verdict moves to about
  13:30; the judge stays before 09:00 tomorrow.
- 11:39 **`ideal` (diagnostic: central's window, every frame valid, no noise, no team errors): +0.0109, 10/10**; p25 /
  p50 / stressed +0.0115 / +0.0108 / +0.0103; IDSSE +0.0085, Metrica +0.0163. The room for a camera that follows the
  ball, with v15's models as trained; central reaches half of it.
- 11:50 my background waiters (`pgrep -f "run_setting.sh ideal"`) matched themselves and held the queue ~5 min;
  killed. No run affected.
- 11:40 `pessimistic`'s first launch failed at setup (an empty `--tag` dropped by zsh; fixed); requeued after the
  noise check, before the retrain. `off_s1` started.
- 11:54 the coordinator: track 1 done, back to 2 slots (cap raised).
- 12:05 **`off_s1` = the network page's seed-1 B exactly: 0.7294 (−0.0035 against v15's, 4/10)**; p25 / p50 /
  stressed 0.7254 / 0.7282 / 0.7346. `central_s1` started.
- 12:21 **`central_s1`: 0.7419 against `off_s1` 0.7294: +0.0125, better on 7/10**; p25 / p50 / stressed +0.0125 /
  +0.0115 / +0.0134; IDSSE +0.0079, Metrica +0.0233. The screen's pass holds on seed 1 (noise rule met).
  `pessimistic` (v15's B) started.
- 12:25 the coordinator: **Jose adopted v16** (v15 with B's GRU P replaced by `netmet5`'s: the mean of four netmet
  networks, seeds 0-3, leave one match out over all 10; trial 1015's stage tables refitted on it,
  `data/external/lab_netmet/ens4`). The merge, if it goes in, goes in on top of v16.

- 12:39 **`pessimistic` (v15's B, no retrain): 0.7282, −0.0047, better on 3/10**; p25 / p50 / stressed −0.0054 /
  −0.0046 / −0.0040; IDSSE −0.0089, Metrica +0.0050. **The robustness part (≥ −0.002) fails** for the no-retrain v15
  version. Of the 0.48 players it adds a frame, 0.31 are missing players with the right team; **0.096 the false
  person** (the referee-like track), **0.070 duplicates** of players the tactical tracking has (its noise, p90 2.7 m,
  beats the 2.5 m same-team radius), 0.016 wrong-team.

### Added 12:35, before any v16 score: the no-retrain harness with v16's B

**Runs:** `off_v16` (merge off), `central_v16`, `pessimistic_v16`, each with everything v15's harness has except B's
GRU P: on each match and variant, the four netmet seeds' fold nets for that match (`lab_netmet/main/nets`,
`seed1..3/nets`, each read through its fold's DFL normaliser as `lab_netmet.norm_for`) give their P, averaged as
`lab_netmirror.average` (each member through float32, the mean, clipped, rounded); trial 1015's stage tables are
`ens4`'s. Models read only, never retrained.

**Reproduction:** `off_v16` must give `netmet5`'s B: **0.7468, all 30 units within 1e-5** of
`bench/results/lab_netmet/ens4/confirm_1015.csv`.

**The bar for v16 + merge (the same parts as above, against `off_v16`):** B's bench +0.005 or more and better on 7 of
10 (`central_v16`); no variant down more than 0.002; neither source down; `pessimistic_v16` no worse than `off_v16` by
more than 0.002; merge off = v16 exactly (by `off_v16`, and by the module's fallback). The noise rule is about passes
by a single network; v16's GRU is a four-seed mean, and the screen's seed-1 check above already covers the merge's own
dependence on the seed. SkillCorner: no broadcast there, the merge is off, v16's own number holds.

**Order:** these three runs first (2 slots); the v15-recipe retrain (stage 2) after them, if slots and time allow,
for what retraining adds; **a retrain on v16's recipe (4 seeds × 10 folds of netmet nets, then the tables) is not
attempted**: about 40 network fits on 2 shared slots plus the rest of B's rebuild is 6-8 h with no slack before
09:00, and v16's own netmet pipeline would have to be rebuilt on merged features in a copy first (my call).

### Added 12:50, after `pessimistic` (v15) was scored, before they run: two diagnostics on v16's B

Not candidates: they say which part of the pessimistic loss comes from what. Each uses **pessimistic's own random
draws** (same seeds: the same invalid blocks, camera, tracklets, offsets) with one element reverted:

| diagnostic | change from `pessimistic` |
|---|---|
| `pess_noref` | no false person |
| `pess_noref_cnoise` | no false person, and central's noise (sd_h / sd_p / sd_j / sd_e 0.55 / 0.30 / 0.15 / 0.010, no outliers) |

So `pessimistic` − `pess_noref` is the false person's cost, `pess_noref` − `pess_noref_cnoise` the extra noise's, and
`pess_noref_cnoise` − `central` the rest (fewer players, more invalid frames, 2.5 % team errors, the lag, the 40-frame
windows). Queue: `off_v16`, `central_v16`, `pessimistic_v16`, these two (`_v16`), then the v15-recipe retrain, then
`optimistic_v16`, `literal`.

- 12:40-12:58 **`off_v16` reproduces v16's B: all 30 units within 4.8e-6** of netmet5's confirm; B's bench **0.7468**.

### Added 13:05, after `pessimistic` (v15) and before any v16 merge score: rule M2 (post hoc, labelled so)

**M2 = M plus a cap: a team never has more than 11.** On each valid frame, a team's added players are limited to
11 minus its players in the tactical tracking on that frame; if more pass M's tests, the ones farthest from any
tactical teammate are kept first (the least likely to be duplicates). A genuinely missing player is never blocked by
it (his team then has at most 10 visible); a false person or a duplicate is blocked whenever his labelled team is
complete. It is aimed at what `pessimistic` showed (0.096 false person and 0.070 duplicates a frame); it is chosen
after seeing that, so a pass by M2 carries that caveat. Screened as `central_M2_v16` and `pessimistic_M2_v16` with
the same bar against `off_v16`; queued right after `pessimistic_v16`, before the two diagnostics.

- 13:24 **`central_v16`: 0.7549 against `off_v16` 0.7468: +0.0081, better on 8/10**; p25 / p50 / stressed +0.0065 /
  +0.0086 / +0.0093; IDSSE +0.0067, Metrica +0.0114. Parts 1-3 pass on v16, with more margin than on v15.
- 13:20 the deployable module's draft (scratchpad), tested end to end on J03WR9 (bench data only): the central
  simulation written in data.md's formats (EPTS-like meta, raw text, homography CSV) on its own timeline (shifted
  −500 and −2,500 frames, 2,000 frames of other footage between the halves), read back, aligned by
  `align_broadcast`'s method (both shifts found exactly; rms 0.81 / 0.85 m, next best 33.8 / 31.6 m), merged: **the
  same slot-frames as the bench merge (0 of 3.06 M differ), positions within 1.5e-5 m** (the text round trip); with
  the teams in the other order in the metadata, the same; a missing file or an unrelated ball → the tracking returned
  unchanged (`fallback-missing`, `fallback-alignment`). Its `merge()` is the bench's, bit for bit.
- 13:44 **`pessimistic_v16`: 0.7449 against 0.7468: −0.0018, better on 4/10**; p25 / p50 / stressed −0.0022 / −0.0017 /
  −0.0015; IDSSE −0.0031, Metrica +0.0011. Inside the −0.002 floor by 0.0002, so **rule M on v16 passes every
  registered part**, the robustness part only just (single matches swing −0.028 to +0.017; on v15 the same run failed,
  −0.0047). `central_M2_v16` started.
- 13:50 since M on v16 passes, **`possession/broadcast_merge.py` written** (new file; the scratchpad draft with the
  bench's current `merge()` copied in: rule M by default, `RULE_M2` available). `python -m bench.lab_broadcast port`
  added: the end-to-end check above, for M and M2, into `bench/results/lab_broadcast/port.json`.
- 13:52 the only contact with our match: the module's default paths for files 6-8, with **SHA-256 and row counts only**:
  `data/broadcast/BAR-ATH_broadcast_atd.xml` 53d387bc…dbe9 (1,676 lines), `…_atd.txt` a71c1c93…e458 (164,488),
  `…_homography.csv` 9ed07e70…a4afb (164,489 with the header). The module was not run on it.
- 14:10 **`central_M2_v16`: +0.0076, better on 8/10** (p25 / p50 / stressed +0.0061 / +0.0080 / +0.0088; IDSSE
  +0.0058, Metrica +0.0119): the cap costs almost nothing on a clean broadcast (M: +0.0081).
- 14:12 **`port` passes** (`bench/results/lab_broadcast/port.json`): the module's `merge()` is the bench's for M and M2;
  through the files both shifts are found exactly and 0 slot-frames differ from the bench merge (positions within
  1.5e-5 m), with either team order in the metadata; a missing file and an unrelated ball fall back unchanged.
- 14:33 **`pessimistic_M2_v16`: 0.7463 against 0.7468: −0.0005, better on 6/10**; p25 / p50 / stressed −0.0006 /
  −0.0009 / +0.0001; IDSSE −0.0015, Metrica +0.0020. M2 passes the robustness part with 0.0015 to spare (M: 0.0002).
  The diagnostics (`pess_noref_v16`, `pess_noref_cnoise_v16`) started.
- 14:40 **added before it runs: a second draw of the pessimistic view** (`pessimistic2`: the same settings and
  window, other random streams), with M and with M2 on v16, ahead of the retrain. Both robustness margins (0.0002 for
  M, 0.0015 for M2) are small next to the per-match spread (±0.03), so whether they hold on another draw matters for
  the decision. Read as information beside the registered run, not as a replacement for it. Then `optimistic` with M
  and M2 on v16 (after the retrain).
- 14:52 **`pess_noref_v16` (pessimistic without the false person): +0.0035, better on 6/10**; p25 / p50 / stressed
  +0.0031 / +0.0027 / +0.0047; IDSSE −0.0001, Metrica +0.0119. **The false person costs 0.0053** of the pessimistic
  view's result (−0.0018 → +0.0035).
- 15:14 **`pess_noref_cnoise_v16` (also central's noise): +0.0026, better on 5/10**; p25 / p50 / stressed −0.0000 /
  +0.0025 / +0.0053; IDSSE +0.0009, Metrica +0.0065. Pessimistic's larger noise costs nothing measurable (+0.0035
  with it, +0.0026 without: inside the noise); the rest of the pessimistic view (fewer players, more invalid frames, more
  of them in play, 2.5 % team errors, the lag, the 40-frame windows) takes central's +0.0081 to about +0.003.
- 15:32 **`pessimistic2_v16` (M, the second draw): −0.0010, better on 3/10**; p25 / p50 / stressed −0.0010 / −0.0007 /
  −0.0012; IDSSE +0.0005, Metrica −0.0045. M stays inside the −0.002 floor on both draws (−0.0018, −0.0010).
- 15:48 **`pessimistic2_M2_v16`: +0.0007, better on 4/10**; p25 / p50 / stressed +0.0015 / +0.0007 / −0.0001; IDSSE
  +0.0009, Metrica +0.0002. M2 is safe on both draws (−0.0005, +0.0007). The v15-recipe retrain (stage 2, central, M)
  started in `scratchpad/broadcast/b16_bc_central` (`stage2b.sh`: stage3.sh's steps on b16_c0's code, the lab cache
  from `lab_broadcast build central`, robust's held-out P saved by a one-line patch in that copy only).
- 16:00 results, verdict and the section for Jose written; the retrain's numbers to follow.
- 17:18 **the retrain (v15's recipe on merged central data; every B model refitted leave one match out: robust's
  held-out P, `augment`'s trees, `gru_b`'s 8 fold nets, lab_gate's fits, the post-stage tables) is in:
  B's bench 0.7354 against v15's 0.7329: +0.0026, better on 5/10** (p25 / p50 / stressed +0.0013 / +0.0027 / +0.0038;
  IDSSE −0.0001, Metrica +0.0089). **Part 1 fails.** Against the no-retrain run on the same merged bench (`central`,
  0.7383) it is −0.0029, better on 2/10, inside one `gru_b` seed's noise (0.0035): **retraining on merged data adds
  nothing measurable**, and the noise rule's seed 1 was therefore not run. (Its trial 0, B as built with no GRU, gains
  +0.0062, 7/10: the trees alone do take the merged players in.) Its SkillCorner run and the retrained B through the
  harness (central, pessimistic, unmerged) follow, for information.
- 17:27 the retrained deployment models on SkillCorner's 20 (no broadcast, merge off): **0.4033 against v15's 0.4085,
  −0.0052** (just past the old 0.005 allowance). With the fallback design SkillCorner would run v15's / v16's own
  models, so this only says the retrained models are a little worse where there is nothing to merge.
- 17:44 `central_rt` (the harness with the retrained copy's models): 0.7354, **all 30 units within 5e-6 of the copy's
  own confirm**: the harness and the rebuild agree. `pessimistic_rt`, `off_rt` follow.
- 18:03 **`pessimistic_rt`: 0.7239, −0.0089 against v15's B, better on 3/10** (IDSSE −0.0122, Metrica −0.0013): worse
  than the untrained models on the same view (−0.0047). Trained on a clean merge, B leans on the added players more and
  is hurt more by false ones: one more reason not to retrain.
- 18:21 **`off_rt` (the retrained models with no broadcast at all): 0.7231, −0.0098, better on 3/10** (IDSSE −0.0087,
  Metrica −0.0122; frames with nothing added lose 0.3 points). Models trained on merged data depend on the merge;
  without the broadcast they are a point below v15. The deployable design (fall back to the unretrained models and the
  delivered tracking) avoids this. The retrain copy's bulky caches removed (its trained models kept, 263 MB).
  `optimistic_v16`, `optimistic_M2_v16`, `literal` follow.
- 18:40 **`optimistic_v16` (M; 17 players per valid frame, 30 % invalid, half central's noise): +0.0087, better on
  9/10**; p25 / p50 / stressed +0.0080 / +0.0085 / +0.0095; IDSSE +0.0076, Metrica +0.0111. A wider, cleaner view adds
  little beyond central (+0.0081): the players that matter are the ones near the ball, which central's window has.
- 18:58 `optimistic_M2_v16`: +0.0091, better on 9/10 (M2 ≥ M on every view tried).
- 19:15 **`literal` (v15, central's view, the brief's rule as written): +0.0004, better on 4/10** (IDSSE −0.0066,
  Metrica +0.0165): registration and the guard are worth +0.005.
- 19:20 all runs done; page finished. Every run's copy cleaned; the retrain copy keeps only its trained models (263 MB).

## Results

Tables from `python -m bench.lab_broadcast report` (`bench/results/lab_broadcast/report.json`, `scores_<run>.csv`;
per-match infos and zones in `data/external/lab_broadcast/<run>/`). **Every run is v15's or v16's B as trained, on
the merged bench: nothing retrained** (the v15-recipe retrain is below, separately).

### The reproductions (merge off)

| run | B's bench | against | units within |
|---|---|---|---|
| `off` | **0.7329** | v15's B, round 8's confirm | 30 / 30 within **4.8e-6**; every recomputed input bit-identical to b16_c0's on 10/10 |
| `off_s1` | **0.7294** | the network page's seed-1 B | its B's bench exactly (−0.0035, 4/10) |
| `off_v16` | **0.7468** | v16's B, netmet5's confirm (`lab_netmet/ens4`) | 30 / 30 within **4.8e-6** |

### On v15's B (the screen as first registered)

| run | base | B's bench | gain | better / worse | p25 / p50 / stressed | IDSSE / Metrica |
|---|---|---|---|---|---|---|
| **`central`** (M) | `off` 0.7329 | 0.7383 | **+0.0055** | 7/3 | +0.0053 / +0.0053 / +0.0058 | +0.0020 / +0.0135 |
| `central_s1` (noise rule) | `off_s1` 0.7294 | 0.7419 | **+0.0125** | 7/3 | +0.0125 / +0.0115 / +0.0134 | +0.0079 / +0.0233 |
| **`pessimistic`** (M) | `off` 0.7329 | 0.7282 | **−0.0047** | 3/7 | −0.0054 / −0.0046 / −0.0040 | −0.0089 / +0.0050 |
| `ideal` (diagnostic) | `off` 0.7329 | 0.7437 | **+0.0109** | 10/0 | +0.0115 / +0.0108 / +0.0103 | +0.0085 / +0.0163 |
| `literal` (diagnostic: the brief's rule as written) | `off` 0.7329 | 0.7332 | **+0.0004** | 4/6 | −0.0002 / +0.0001 / +0.0012 | −0.0066 / +0.0165 |

`literal` against `central` (the same view): **registration and the other-team guard are worth +0.005** (+0.0004 →
+0.0055); without them the same-team test alone lets mislabelled and misregistered players through.

### On v16's B (the coordinator's addition, 12:35)

| run | base | B's bench | gain | better / worse | p25 / p50 / stressed | IDSSE / Metrica |
|---|---|---|---|---|---|---|
| **`central_v16`** (M) | `off_v16` 0.7468 | 0.7549 | **+0.0081** | **8/2** | +0.0065 / +0.0086 / +0.0093 | +0.0067 / +0.0114 |
| **`pessimistic_v16`** (M) | `off_v16` 0.7468 | 0.7449 | **−0.0018** | 4/6 | −0.0022 / −0.0017 / −0.0015 | −0.0031 / +0.0011 |
| `pessimistic2_v16` (M, 2nd draw) | `off_v16` 0.7468 | 0.7458 | **−0.0010** | 3/7 | −0.0010 / −0.0007 / −0.0012 | +0.0005 / −0.0045 |
| **`central_M2_v16`** (M2, post hoc) | `off_v16` 0.7468 | 0.7544 | **+0.0076** | **8/2** | +0.0061 / +0.0080 / +0.0088 | +0.0058 / +0.0119 |
| **`pessimistic_M2_v16`** (M2) | `off_v16` 0.7468 | 0.7463 | **−0.0005** | 6/4 | −0.0006 / −0.0009 / +0.0001 | −0.0015 / +0.0020 |
| `pessimistic2_M2_v16` (M2, 2nd draw) | `off_v16` 0.7468 | 0.7475 | **+0.0007** | 4/6 | +0.0015 / +0.0007 / −0.0001 | +0.0009 / +0.0002 |
| `pess_noref_v16` (diagnostic) | `off_v16` 0.7468 | 0.7502 | +0.0035 | 6/4 | +0.0031 / +0.0027 / +0.0047 | −0.0001 / +0.0119 |
| `pess_noref_cnoise_v16` (diagnostic) | `off_v16` 0.7468 | 0.7493 | +0.0026 | 5/5 | −0.0000 / +0.0025 / +0.0053 | +0.0009 / +0.0065 |
| `optimistic_v16` (M) | `off_v16` 0.7468 | 0.7554 | +0.0087 | 9/1 | +0.0080 / +0.0085 / +0.0095 | +0.0076 / +0.0111 |
| `optimistic_M2_v16` (M2) | `off_v16` 0.7468 | 0.7559 | +0.0091 | 9/1 | +0.0084 / +0.0090 / +0.0100 | +0.0079 / +0.0119 |

**Where the pessimistic loss comes from** (v16, rule M; the diagnostics share pessimistic's random draws):
central **+0.0081** → the pessimistic camera, validity, team errors, lag and 40-frame windows **≈ +0.003**
(`pess_noref_cnoise`) → pessimistic's larger noise **+0.0035** (no measurable cost) → **the false person −0.0018**
(it costs 0.0053). M2's cap takes the same view to −0.0005 / +0.0007 on two draws.

Per match (gain against the run's base):

| match | `central` | `central_s1` | `pessimistic` | `ideal` | `central_v16` | `pessimistic_v16` | `pessimistic2_v16` | `central_M2_v16` | `pessimistic_M2_v16` | `pessimistic2_M2_v16` |
|---|---|---|---|---|---|---|---|---|---|---|
| J03WMX | +0.0198 | +0.0195 | +0.0268 | +0.0219 | +0.0194 | +0.0168 | +0.0156 | +0.0180 | +0.0155 | +0.0177 |
| J03WN1 | −0.0028 | −0.0049 | −0.0192 | +0.0055 | +0.0175 | +0.0083 | −0.0021 | +0.0175 | +0.0139 | −0.0017 |
| J03WOH | +0.0042 | +0.0204 | −0.0176 | +0.0104 | +0.0046 | −0.0081 | −0.0009 | +0.0016 | −0.0060 | +0.0109 |
| J03WOY | +0.0063 | +0.0117 | −0.0058 | +0.0045 | −0.0020 | −0.0072 | +0.0007 | −0.0020 | −0.0076 | −0.0029 |
| J03WPY | −0.0107 | −0.0081 | −0.0248 | +0.0001 | −0.0138 | −0.0284 | −0.0044 | −0.0138 | −0.0302 | −0.0030 |
| J03WQQ | −0.0059 | −0.0031 | −0.0199 | +0.0101 | +0.0099 | −0.0030 | −0.0117 | +0.0076 | +0.0036 | −0.0111 |
| J03WR9 | +0.0034 | +0.0195 | −0.0017 | +0.0073 | +0.0115 | −0.0000 | +0.0066 | +0.0115 | +0.0001 | −0.0035 |
| metrica1 | +0.0167 | +0.0100 | +0.0099 | +0.0100 | +0.0086 | +0.0018 | −0.0027 | +0.0086 | +0.0003 | −0.0080 |
| metrica2 | +0.0190 | +0.0438 | +0.0067 | +0.0257 | +0.0165 | +0.0056 | −0.0068 | +0.0165 | +0.0122 | +0.0031 |
| metrica3 | +0.0048 | +0.0162 | −0.0015 | +0.0133 | +0.0090 | −0.0041 | −0.0039 | +0.0106 | −0.0066 | +0.0053 |

J03WPY loses in every non-ideal run (−0.003 to −0.030) and J03WMX gains in every one (+0.016 to +0.027): which
matches move is about the same from run to run, so part of the per-match spread is the match, not the draw.

Near / far from the ball (graded frames pooled over the 30 units; "near": an added piece within 15 m of the true ball):

| run | zone | share | accuracy base → run | change | letters changed | start's log loss |
|---|---|---|---|---|---|---|
| `central` | near | 18.9 % | 0.9130 → 0.9200 | +0.0071 | 3.5 % | 0.2507 → 0.2139 |
| `central` | far | 14.9 % | 0.9150 → 0.9174 | +0.0024 | 2.1 % | 0.2720 → 0.2671 |
| `central` | none | 66.2 % | 0.9418 → 0.9436 | +0.0018 | 0.7 % | 0.1911 → 0.1862 |
| `pessimistic` | near | 20.6 % | 0.9201 → 0.9239 | +0.0038 | 3.7 % | 0.2336 → 0.2172 |
| `pessimistic` | far | 10.0 % | 0.8994 → 0.8978 | −0.0016 | 3.0 % | 0.3237 → 0.3210 |
| `pessimistic` | none | 69.5 % | 0.9407 → 0.9427 | +0.0020 | 1.1 % | 0.1931 → 0.1884 |
| `ideal` | near | 24.9 % | 0.9229 → 0.9342 | +0.0113 | 2.8 % | 0.2237 → 0.1879 |
| `central_v16` | near | 18.9 % | 0.9142 → 0.9268 | **+0.0126** | 3.6 % | 0.2328 → 0.1976 |
| `central_v16` | far | 14.9 % | 0.9204 → 0.9205 | +0.0001 | 1.8 % | 0.2486 → 0.2437 |
| `central_v16` | none | 66.2 % | 0.9466 → 0.9472 | +0.0006 | 0.6 % | 0.1764 → 0.1724 |
| `pessimistic_v16` | near | 20.6 % | 0.9264 → 0.9306 | +0.0043 | 3.3 % | 0.2106 → 0.1929 |
| `pessimistic_M2_v16` | near | 17.3 % | 0.9238 → 0.9303 | +0.0065 | 3.5 % | 0.2157 → 0.1908 |

The gain is where it was meant to be: frames with an added player near the ball gain 0.7-1.3 points of frame accuracy
and lose 15 % of their log loss; frames with nothing added barely move (the decoder and the GRU's context carry a
little). The letters are what drops in the pessimistic runs (spellF1 down while frame accuracy near the ball is still
up): false persons and duplicates create short wrong possessions rather than wrong frames.

The simulated views and what the merge added (mean of 10 matches; the truth only counts, the merge never sees it):

| | `central` | `pessimistic` | `pessimistic` with M2 | `ideal` |
|---|---|---|---|---|
| players / valid frame (all frames) | 11.5 (7.6) | 9.2 (5.5) | 9.2 (5.5) | 11.1 (11.1) |
| valid frames (invalid ones in true dead time) | 66.0 % (80 %) | 59.9 % (64 %) | 59.9 % (64 %) | 100 % |
| tracklet mean | 9.4 s | 7.1 s | 7.1 s | 15.0 s |
| error median / p90 (after registration, median) | 0.78 / 1.42 m (0.42) | 1.43 / 2.71 m (0.78) | same | 0 |
| wrong team label / false person | 0.90 % / 0 | 2.53 % / 2.10 % | same | 0 / 0 |
| tactical: missing players a frame (within 15 m of the ball) | 1.59 (0.36) | 1.59 (0.36) | 1.59 (0.36) | 1.59 (0.36) |
| **added a frame** | **0.495** | **0.484** | **0.379** | **0.699** |
| … missing players, right team | 0.490 | 0.310 | 0.303 | 0.699 |
| … missing players, wrong team | 0.0045 | 0.0072 | 0.0032 | 0 |
| … duplicates (right / wrong team) | 0.0001 / 0.0001 | 0.0609 / 0.0093 | 0.0175 / 0.0048 | 0 |
| … the false person | 0 | 0.0961 | 0.0503 | 0 |
| **missing near-ball players covered** (`o_near`'s set) | **65.4 %** | 45.7 % | 44.5 % | 85.0 % |

### The judge against the bar

| part | v15 + M (registered) | **v16 + M** (registered, coordinator's addition) | v16 + M2 (post hoc) |
|---|---|---|---|
| 1. central: +0.005, 7 of 10 | +0.0055, 7/10 ✓ | **+0.0081, 8/10 ✓** | +0.0076, 8/10 ✓ |
| 2. no variant down more than 0.002 | all up ✓ | all up ✓ | all up ✓ |
| 3. neither source down | +0.0020 / +0.0135 ✓ | +0.0067 / +0.0114 ✓ | +0.0058 / +0.0119 ✓ |
| noise rule (a single network's pass) | seed 1: +0.0125, 7/10 ✓ | v16's GRU is a four-seed mean | as M |
| 4. pessimistic ≥ −0.002 | **−0.0047 ✗** | **−0.0018 ✓** (by 0.0002); 2nd draw −0.0010 | −0.0005 ✓; 2nd draw +0.0007 |
| 5. broadcast absent = the program without it | `off` = v15 ✓; module falls back | `off_v16` = v16 ✓; module falls back | as M |
| 6. SkillCorner (merge off) | v15's 0.4085 (no broadcast there) | v16's own (no broadcast there) | as M |
| **verdict** | **fails** (robustness) | **passes, robustness by a hair** | passes with margin, post hoc |

**The retrain (stage 2, v15's recipe, central, M):** B's bench 0.7354 against v15's 0.7329, **+0.0026, better on
5/10** (p25 / p50 / stressed +0.0013 / +0.0027 / +0.0038; IDSSE −0.0001, Metrica +0.0089): **part 1 fails**, and it
is below the no-retrain run on the same bench (−0.0029, 2/10, inside one seed's noise). Per match: J03WMX +0.0332,
J03WN1 −0.0050, J03WOH −0.0081, J03WOY −0.0002, J03WPY −0.0245, J03WQQ −0.0070, J03WR9 +0.0108, metrica1 +0.0103,
metrica2 +0.0122, metrica3 +0.0041. Its trial 0 (B as built, `augment` alone) gains +0.0062 (7/10). Its deployment
models on SkillCorner (no broadcast): 0.4033 (−0.0052 against v15's 0.4085). The same models through the harness:

| run (retrained B) | base | B's bench | gain | better / worse | p25 / p50 / stressed | IDSSE / Metrica |
|---|---|---|---|---|---|---|
| `central_rt` (= the copy's confirm, 30 units within 5e-6) | `off` 0.7329 | 0.7354 | +0.0026 | 5/5 | +0.0013 / +0.0027 / +0.0038 | −0.0001 / +0.0089 |
| `pessimistic_rt` | `off` 0.7329 | 0.7239 | **−0.0089** | 3/7 | −0.0100 / −0.0097 / −0.0071 | −0.0122 / −0.0013 |
| `off_rt` (no broadcast) | `off` 0.7329 | 0.7231 | **−0.0098** | 3/7 | −0.0110 / −0.0105 / −0.0077 | −0.0087 / −0.0122 |

Retrained on a clean merge, B leans on the added players: it loses more when they are wrong (pessimistic) and about a
point when there are none.

The retrain the brief asked for was done on v15's recipe only; v16's recipe was not retrained (see 12:35). The judge
above is therefore the **no-retrain** version, as the brief allows; on v15 the retrain was no better than not
retraining.

### The deployable module

`possession/broadcast_merge.py` (new): `merge_broadcast(tracking, halves, paths=None, rule=None)` reads files 6-8 by
path (default: `games.asset_paths(source="broadcast")`), finds the shift per graded half with `align_broadcast`'s
method (refuses it unless rms ≤ 12 m and the next-best shift is 3× worse), maps the broadcast onto the tactical frames,
checks the team order by position, and merges with rule M (`rule=broadcast_merge.RULE_M2` for M2). It prints nothing;
any failure returns the delivered tracking unchanged with a one-word status. `python -m bench.lab_broadcast port`
checks it against the bench's merge end to end (`bench/results/lab_broadcast/port.json`: identical for M and M2, both
team orders, both fallbacks). Not run on our match; integration is the coordinator's.

**How it would go in** (for the coordinator): call it on the delivered tactical tracking before `v12.inputs`, for
branch B only, exactly as the bench did (v8's repair, v4/v6's P, PathCRF, v7, v9, v11, robust's features, the GRUs and
the post stages then all read the merged tracking). The stoppage switch was not simulated here: it should keep reading
the delivered tracking, so its choice can't change.

## Verdict (16:00; the retrain added 17:30; final 19:20)

**Yes, on a broadcast like the one data.md describes, merging its players into the tactical tracking improves branch
B, and on v16 it clears the B bar without retraining anything. It is fragile in one specific way: people who aren't
players.**

- **v16 + rule M (registered before any score): passes every part.** Central +0.0081, better on 8 of 10, every
  variant and both sources up; pessimistic −0.0018 (the floor is −0.002) and −0.0010 on a second draw; merge off is v16
  exactly; SkillCorner has no broadcast, so v16's number stands. The robustness part passes by 0.0002 on the
  registered draw: a pass by the letter, not by much.
- **v15 + rule M fails** on robustness (pessimistic −0.0047), though central passes (+0.0055, 7/10; +0.0125 with
  seed 1). The stronger v16 network takes in the merged players better.
- **The pessimistic loss is mostly the false person** (a referee-like track near the ball on 20 % of frames:
  −0.0053). Without it the pessimistic view gains (+0.0035); its larger position noise costs nothing measurable.
- **M2** (M plus "a team never has more than 11"; post hoc, chosen after seeing `pessimistic`) keeps central
  (+0.0076, 8/10) and is safe on both pessimistic draws (−0.0005, +0.0007): it halves the false person's adds and cuts
  the duplicates by 70 %.
- **The room:** a perfect broadcast in the same window gives +0.0109 (10/10) with v15's models; central reaches half
  of it; a wider, cleaner view (`optimistic`) adds little more on v16 (+0.0087 / +0.0091 with M2, 9/10); round 9's
  truth near the ball (+0.022 on the start, retrained trees) is the far ceiling.
- **The merge rule matters:** the brief's rule as written (same team within 3 m, no registration) gives +0.0004 on
  v15 where M gives +0.0055.
- **Retraining doesn't help.** B retrained on merged training data (v15's recipe, every B model refitted) gets +0.0026
  (5/10) on the merged central bench, below the same models *not* retrained (+0.0055); the gap (−0.0029) is inside one
  `gru_b` seed's noise. The no-retrain version is the one to judge, and it needs no new models.
- **A deployable module exists** (`possession/broadcast_merge.py`), checked end to end on the bench; never run on our match.

**What I would recommend Jose decide on** (nothing is adopted here): v16 + M2, if he is willing to take a post-hoc rule
whose only change is a cap a real match can't violate; v16 + M if only registered rules count. Either way, the one
check that decides the sign is whether our broadcast tracking follows the referees (below).

## For Jose

1. **The question it answers:** would the broadcast's players near the ball, merged into the tactical tracking, make
   branch B better? On the bench, with a broadcast simulated from data.md's numbers: **yes, about +0.008 on v16's
   B's bench (8 of 10 matches)**, without retraining anything. With a much worse broadcast it goes from +0.008 to about
   −0.001 / −0.002; the cap rule (M2) keeps that worse case at zero.
2. **The one thing to check by eye, if you can:** on the broadcast video with its tracking drawn, **does the tracking
   follow the referee (or anyone who isn't a player) and give him a team?** That is what turns the result negative. If
   it doesn't, the pessimistic loss mostly goes away (+0.0035 without it). Also worth a glance: whether broadcast
   positions sit within about 2 m of the tactical ones, and whether team colours look right near the ball (Barcelona's
   and Athletic's kits both have red).
3. **What you'd be choosing between** (if it goes in at all, on top of v16): **M2** (my recommendation: the same as M
   plus "a team never has more than 11", chosen after I saw the pessimistic run, so it carries that caveat), or **M**
   (fixed before any score, passes the robustness part by 0.0002). Or not now: v16 as it is.
4. **What it costs to try:** the module is written and tested on the bench; the coordinator wires it in front of
   branch B's inputs (the switch keeps reading the delivered tracking). If the broadcast files can't be read or lined up,
   it falls back to the delivered tracking and the program is v16 exactly.

## Failure modes: how the real broadcast could differ from the simulation

The simulation uses only data.md's statistics; everything else about our broadcast is unknown, and some of it decides
the sign of the result (the pessimistic runs show that). In order of what the runs say matters most:

1. **People who aren't players.** If the broadcast tracking follows referees (or linesmen, coaches, ball boys near the
   touchline) and gives them a team label, the merge adds them: they are within a few metres of the ball and no
   tactical player is there to suppress them. In `pessimistic`, a referee-like track on 20 % of valid frames cost
   **0.0053** of B's bench (`pess_noref_v16` against `pessimistic_v16`); M2's cap blocks it only when its labelled
   team is complete (halves it). data.md lists 51 slots, all with a team (26 / 25), which neither confirms nor rules
   this out. **This is the one thing worth checking by eye** (Jose, on the broadcast video with its tracking drawn).
2. **Position error larger than the 2.5 m same-team radius.** Then players the tactical tracking already has come
   back as duplicates beside themselves (`pessimistic`: 0.070 a frame, p90 error 2.7 m). Registration takes out the
   error common to the picture; what's left is per player (foot point, occlusion, the far side's perspective).
   data.md's 0.77 m median ball distance at the alignment is the best evidence that central's 0.4 m (registered) is
   the right size, but it's the ball, not the players.
3. **Team-label errors near the ball.** The simulation flips whole tracklets at random (1 % central, 2.5 %
   pessimistic). A real classifier errs more in crowds, by the ball, and **Barcelona's and Athletic's kits share red**.
   The guard (no other-team tactical player within 1.5 m) suppresses most mislabelled *seen* players; a mislabelled
   *missing* player is added with the wrong team (0.004-0.007 a frame here). tracking-errors.md: a wrong team near
   the ball is the costliest error per frame.
4. **Where the invalid frames fall.** Central puts 80 % of them in true dead time; if the real director cuts away more
   during live play (replays over a restart, close-ups in build-up), fewer live frames get filled (pessimistic: 64 %).
5. **Fewer players in the window** (pessimistic: 9 per valid frame) mostly lowers the gain, it doesn't turn it negative.
6. **The tracklets** are shorter here (9.4 s) than our broadcast's 16 s, because the simulated window's edges cut them.
   The merge works frame by frame, so this changes little; longer real tracklets would only mean fewer, longer pieces.
7. **The alignment.** data.md: one shift per half holds within ±2 frames in every 2-minute window of the 1st half and
   96 % of the 2nd's (one window off by 40 frames). Registration absorbs the common motion; the gate drops frames whose
   residual stays above 1.5 m. The module checks the shift itself (rms ≤ 12 m and the next-best shift 3× worse; data.md
   gives 5.6 / 7.4 m against 36.9 / 37.8 m) and otherwise falls back to the delivered tracking.
8. **Our tactical tracking is not the bench's.** The bench's tactical positions are exact and its holes random; ours
   have their own error (so broadcast-to-tactical distances are larger than here, more duplicates at 2.5 m), more than
   11 players of a team on 8.1 % / 4.7 % of frames (data.md: someone counted twice; M2's cap then blocks every add for
   that team), and holes likelier in crowds near the ball (cleaning.md's caveat: more room for the merge, and harder
   for the broadcast too).
9. **Models never retrained.** Every number here is v15's or v16's B as trained on unmerged data. The merged inputs are
   in the range those models know (more players near the ball, at believable positions), but nothing was fitted to them.

## Files

- `bench/lab_broadcast.py` (new): the simulator (`SIM`, calibration), the merge (`RULES`: M, M2, literal), the harness
  on lab_trackerr's machinery (`setup`, `build`, `score`, with `--tag _s1/_v16/_rt`), `report`, `port`.
- `possession/broadcast_merge.py` (new): the deployable merge (reads files 6-8 by path, aligns, merges; falls back).
- `bench/results/lab_broadcast/`: `report.json` (every run: B's bench, per variant / source / match, zones, the view's
  statistics, the merge's counts), `scores_<run>.csv` (every unit), `reproduction.csv`, `reproduction_v16.csv`,
  `port.json`.
- `data/external/lab_broadcast/` (gitignored): `calibration.json`; per run, each match's letters, start log loss,
  zones and `info_<match>.json` (what the simulation and the merge did).
- Scratchpad (`scratchpad/broadcast/`): the run scripts (`run_setting2.sh`, `tokens.zsh`: my slot cap read live;
  `stage2b.sh`: the retrain; `chain*.sh`: the queues), `geo.py` (the radii's geometry), `tables.py`. Each run's copy
  (`wc_<run>`) had its caches removed once scored; the retrain copy `b16_bc_central` is removed when its runs are done.
