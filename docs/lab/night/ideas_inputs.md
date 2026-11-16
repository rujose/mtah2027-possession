# Night ideas: what the models see (the inputs)

| | |
|---|---|
| **who** | one of four idea agents, 2026-11-14, 23:10-00:15. No model built, nothing adopted, nothing run on our match |
| **angle** | round 9 ([../cleaning.md](../cleaning.md)) showed the room is "who was really near the ball while he was invisible" (`o_near` +0.022 on P2, 10/10) and that no fill made from the tracking gets any of it. How do we get some of it anyway, or stop the models being fooled by its absence? |
| **evidence added tonight** | a light probe on three bench matches (J03WR9, J03WMX, metrica1), damaged vs clean tracking, nothing trained: `/private/tmp/claude-501/-Users-joseruiz-Code-more-than-a-hack-2027/39e3effe-3bf8-4611-8584-0402cfeb00b7/scratchpad/ideas/inputs_probe.py` (and `probe2.py`, `probe3.py` beside it; ~6 s a match) |

## What the probe found (bench only)

The fills of round 9 put *positions* back. The probe asks a different question: when the player on the ball (clean
tracking: nearest to the clean ball, within 2 m) is invisible in the damaged tracking, what do our inputs say, and
what else in the damaged tracking points at him?

| live frames, ball seen | J03WR9 | J03WMX | metrica1 |
|---|---|---|---|
| on-ball player invisible (share of live, ball-seen frames) | 6.2 % | 4.4 % | 4.3 % |
| **visible-nearest player's team = team in possession**: on-ball player visible / **invisible** | 91.5 % / **33.8 %** | 90.3 % / **23.7 %** | 92.2 % / **33.7 %** |
| when invisible: only one team has a player missing (share of frames), that team right | 43 %, **89 %** | 42 %, **86 %** | 45 %, **94 %** |
| ... same cue when the on-ball player is visible (no signal) | 49 %, 46 % | 49 %, 49 % | 47 %, 46 % |
| when invisible: a "ghost" (a piece of either team that **ended in the last 10 s or starts in the next 10 s**, at its end / start point) within 5 m and nearer than every visible player | 47 %, team right **92 %** | 39 %, **85 %** | 46 %, **90 %** |
| ... same, on frames where the on-ball player is visible (false alarms) | 0.4 % | 0.6 % | 0.1 % |
| nearest-team call, visible + ghosts, on invisible frames (visible only) | 62 % (34 %) | 52 % (24 %) | 55 % (34 %) |
| invisible on-ball player: min(time since seen, until seen again) ≤ 3 s / ≤ 5 s | 42 % / 63 % | 52 % / 68 % | 42 % / 65 % |

Ball events (velocity change ≥ 3 m/s, robust's blanked ball, live; the events v11's touch detector reads):

| | J03WR9 | J03WMX | metrica1 |
|---|---|---|---|
| events whose true toucher is invisible | 8.7 % | 8.1 % | 6.4 % |
| ... of those: **orphans** (no visible player within 2 m: the touch is lost) / **given to a visible player within 2 m** (wrong team 94-100 %) | 86 % / 14 % | 82 % / 18 % | 86 % / 14 % |
| orphan events with an invisible true toucher that a ghost within 5 m (±10 s) finds; its team right | 48 %, **98 %** | 47 %, **93 %** | 53 %, **87 %** |

And what robust's held-out trees already make of it (`p_blank`): team right on ball-seen live frames with the
on-ball player visible 95.3 % / invisible 91.7 % (J03WR9), 95.6 % / 94.4 % (metrica1). The trees recover most of it
from context, so the frame-level room is small; the spell-level room is in **touches lost or given to the wrong
team** (6-9 % of all touches), which are what places possession changes.

Three things follow:
1. **The inputs never say that someone is missing.** Robust's 97 features use `nanmin`/`nanmean` over whoever is
   seen: "nearest home player 7 m away" reads the same whether home has 11 players seen or 9. Where the on-ball
   player is invisible, the nearest visible player is usually the opponent challenging him, so the feature points
   the wrong way two times in three.
2. **The damaged tracking does hold a cheap, local pointer to him:** the end (or start) point of his own piece,
   seconds away, and which team is short of players. Neither is a position estimate, so neither repeats round 9's
   failure (fills that the trees can't tell from real players, 1.5-4.7 m off in the long holes).
3. **Half the orphans have no ghost** (the player is deep in a long hole). Nothing local finds those; that part
   of `o_near` stays out of reach without another source (the broadcast, ruled out in cleaning.md).

## The ideas, ranked (expected gain × chance it passes × fits tonight)

### 1. Ghost features: tell the models who is missing, and where they were last / next seen

**What changes.** About 22 new columns beside robust's 97, computed from the damaged (blanked) tracking only, in
home/away pairs so `swap=True` works as for every other pair:

| family | per team X (home, away) | definition |
|---|---|---|
| ghosts | `gd_X`, `gd_X_last_seen`, `gd_X_next_seen` | distance from the ball (seen; last-seen; next-seen, as robust's `filled`) to X's nearest **pending ghost**; FAR (60) if none; NaN where that ball is NaN |
| | `gt_X`, `gt_X_last_seen`, `gt_X_next_seen` | that ghost's \|time gap\| (s, cap 10) |
| | `gn_X_10m` | pending ghosts of X within 10 m of the ball (cap 3) |
| ghost touches | `since_gtouch_X`, `until_gtouch_X`, `gtouches_X_past_3s`, `gtouches_X_next_3s` | **orphan ball events** (v11's touch rule, dv 3.0, but no visible player of either team within REACH 2 m) given to X when X's nearest ghost is within 5 m and nearer than the other team's; same encodings as robust's touch features |
| counts (variant b only) | `miss_X` | E_X − (X's players seen on the frame), clipped to 0-3; E_X = min(11, the 99th percentile of X's seen count in the half) so a red card doesn't read as a permanent hole |

A **pending ghost** of X at frame t: the last point of a piece of X that ended at e, t − 10 s ≤ e < t, with no piece of
X starting in (e, t] within 1 m + 7 m/s × (start − e) of that point (he hasn't reappeared nearby); or the first point
of a piece of X starting at a, t < a ≤ t + 10 s, with no piece of X ending in [t, a) within the same reach. Only
while X has fewer players seen than E_X. The reach is round 9's joiner rule (`link_g10_v7`), so no new constant.

**Why it should help.** The probe: on the frames and touches where the on-ball player is invisible, the current
inputs point to the wrong team 66-76 % of the time; ghosts point to the right one 85-98 % of the time on about half
of them, with almost no false alarms (0.1-0.6 %) where he is visible; "only one team is short" is right 86-94 %
there and pure noise elsewhere, which a tree can separate (nearest visible far + ball seen). Recovered ghost
touches fix about 3-4 % of all touches. Unlike round 9's fills, nothing replaces a position: the trees and the GRU
see a separate, time-stamped hint they can learn to trust or not.

**Gain guess.** P2 (robust's trees, held-out spellF1) **+0.003** (range −0.002 to +0.008); B's bench **+0.002**
(−0.003 to +0.006): the GRU already recovers part of this from context (robust: 91.7 % vs 95.3 % on those frames).
Chance of passing B's bar (+0.005, 7/10): **~20 %**. Confidence in the mechanism: good (consistent on 3 matches, 2
providers); in the size: low (P2's do-nothing noise is ±0.005).

**The test.**
- **New files** (the night's naming rule): `bench/lab_ghost.py`, results `bench/results/lab_ghost_*.json`, caches
  `data/external/lab_ghost/<variant>/`, page `docs/lab/ghost.md`.
  - `ghost_columns(track, halves, swap)` → (names, X_extra) from the blanked tracking (pieces as
    `lab_clean.pieces`; orphan events from `possession.touches.velocity_change`).
  - A screen that is `lab_clean.screen` with its own paths (copy `build_features`/`fit`/`proba`/`screen`, ~80 lines;
    **do not call `lab_clean.screen` with a borrowed variant name**: it would overwrite `lab_clean_c0.json` and the
    c0 cache). Features = `lab_clean.features_of("c0", match)` + `ghost_columns` (no PathCRF rerun: the tracking is
    unchanged), trees, decoder, snap/move exactly as P2. Base: `bench/results/lab_clean_c0.json`.
  - A unit check before any screen: on J03WR9, X_swapped's ghost columns equal X's with home/away exchanged; ghost
    touches found and right at about the probe's rates (~48 %, ~95 %).
- **Variants, fixed now:** `ghost_a` (ghosts + ghost touches, 22 columns), `ghost_b` (`ghost_a` + `miss_X`, 24).
- **Run:** `python3 $S/slot.py -- env OMP_NUM_THREADS=2 .venv/bin/python -m bench.lab_ghost screen ghost_a` and the
  same for `ghost_b`: **2 slots, ~25 min each, in parallel** (no PathCRF rerun, so likely under round 9's 25 min).
- **Gate to the B rebuild (round 9's stage-2 gate, fixed before the first score):** pick between `ghost_a` and
  `ghost_b` leave one match out (each match scored with the variant better on the other 9, ties to `ghost_a`, as
  `lab_clean.stage2`); held-out P2 gain **≥ +0.003, better on ≥ 6/10, Metrica's 3 not down on the mean**, log loss
  not up by more than 0.001. Below the gate: record and stop; no B rebuild.
- **B rebuild (only through the gate):** `stage3.sh`'s pattern in a new copy (`$S/b16_ghost`), with three changes:
  no `build-lab` (link the main repo's `data/external/lab`: the tracking doesn't change); the **copy's**
  `possession/lab_learned_features.py` appends `ghost_columns` at the end of `frame_features` (an edit inside the
  isolated copy only, so robust's cache, `lab_carried`'s stressed features, `gru_b` (its input width follows the
  data, `lab_branch_b.py:111`) and SkillCorner's `v12.inputs` all get the same columns); the copy's
  `data/external/lab_robust/p_blank_<m>.npy` = the screen's `data/external/lab_ghost/<picked>/p_<m>.npy`. Then the
  same steps (stress, gru_b folds, carried fit, gate, predict, `search_b stress`, `confirm --trials 1015`,
  `skillcorner --trials 1015`). **`--need 3`, ~2-2.5 h** (take the actual time from the c0 stage-3 run started at
  23:12, which is also the per-match reference to judge against).
- **The bar (B's, unchanged):** B's bench (mean of p25, p50, stressed per match) **+0.005 or more against v15's B,
  better on 7/10**; no variant down more than 0.002; neither source down on the mean; SkillCorner's 20 no loss
  beyond 0.005. The switch reads raw tracking, so its choices can't move.

**Risk that a bench gain doesn't carry to our match.** The bench's holes are random (`degrade.fragment`: geometric
holes regardless of play), so a ghost near the ball is "his own end point" ~90 % of the time. On real tracking:
- if players drop out in crowds (cleaning.md's caveat), invisible possessors are commoner than on the bench and the
  features matter more; the trees learned their weight on the bench's lower rate, so they under-use rather than
  misuse them;
- if pieces break in scrums with **identity swaps or brief false tracks** (the bench has neither), ghosts of both
  teams appear near the ball and the hint gets noisier than the trees learned. The "pending" rule (no reappearance
  nearby) removes the simplest case (a re-acquired player);
- `miss_X` is the riskiest column: a tracker that loses players **far** from the ball (camera edges) makes "one team
  short" mean nothing about the ball. SkillCorner (broadcast tracking: players out of the camera view are missing)
  is a direct test of that failure; if `ghost_b` passes on the bench but fails the SkillCorner part, take `ghost_a`.
  Ghost distances are local, so they fire only where a ghost really is near the ball, wherever holes come from.

**Fits tonight:** yes. Build + unit check ~1 h (to ~00:45), screens to ~01:20, B rebuild to ~04:00, judged by ~04:30.

### 2. A learned near-ball "denoiser": predict the clean near-ball features from the damaged ones

**What changes.** Round 9 kept the features of `o_players` (true players, the same damaged and blanked ball) on
disk: `data/external/lab_clean/o_players/<m>.npz`, frame-aligned with robust's own. So there are free, exact
targets for "what robust's near-ball features would read with every player seen". Fit small regressors (robust's
HistGB settings, squared error, every 10th graded frame, both orientations with targets swapped) from
**robust's 97 + idea 1's ghost columns** to the `o_players` values of the 6 near-ball columns that the missing
player corrupts most: `d_diff`, `d_diff_last_seen`, `d_diff_next_seen`, and `n_home_2m − n_away_2m` at the ball,
plus `d_home`, `d_away`. Append the 6 predictions as columns (`den_*`).

**Why it should help.** It is the "learned imputer" the brief asks for, aimed at the quantities the classifier
reads instead of at positions: positions can't be recovered to the ~2 m that decides who is on the ball (round 9:
3.6-4.7 m in the 10-30 s holes, even with perfect joins), but the *expected* nearest-player gap given every hint
(ghosts, counts, the ball's motion, PathCRF, the teams' shape) can be, and the regressors are supervised by clean
tracking, which the possession labels never show the trees. With 6 training matches that dense, privileged
target is the main hope of beating idea 1 alone. Ball-blanked frames (57-66 % of an IDSSE match in robust's view)
are where it can add most: `_last_seen`/`_next_seen` targets carry who really stood at the ball's last spot.

**Gain guess.** Over `ghost_a`: +0.000 to +0.004 on P2; over v15's B **+0.003** if idea 1's +0.002 holds. Chance to
pass B's bar: **~15 %**. Confidence: low (ballfill's imputer, round 7, shows a regressor pulled towards players can
end up a coin flip; the difference here is the target, the clean feature itself, not a ball position).

**The test.** Same file, variant `ghost_den`. Cross-fitting one level deep, as the bench's earlier models are
(`eval_pathcrf`'s "out-of-sample" outputs): each IDSSE match's `den_*` from regressors fitted on the other 6, each
Metrica match's from all 7. (A held-out match's clean tracking then shapes the training matches' `den_*` columns,
never its own: a second-order leak of the kind the bench already accepts; nested fitting would be 7× the cost.)
6 targets × 8 fits, ~15 s each at 2 threads: ~15 min, then the P2 screen (~25 min): **1 slot, ~45 min**, in the
same wave as `ghost_a`/`ghost_b` if built by then, else right after. **Gate:** as idea 1, and it goes to the B rebuild
instead of idea 1's pick only if it beats that pick by ≥ +0.002 on P2 held out (else the simpler wins). The B
rebuild then also needs the regressors saved and applied inside the copy's `frame_features` (fold-wise for the bench,
the all-7 fit for SkillCorner, the stressed copies and deployment): ~1 h more plumbing.

**Risk to our match:** as idea 1, plus the regressors' calibration rests on the bench's random holes; if our holes
cluster near the ball they will under-correct (the safe direction).

**Fits tonight:** the screen yes; a B rebuild only if it goes through the gate by ~02:30 (done ~05:30).

### 3. Soft targets from a teacher that saw the true players (generalised distillation)

**What changes.** Train the student (robust's trees for the screen; `gru_b` and `augment` for B) on its damaged
inputs, but against `(1 − α) × label + α × P_teacher`, where the teacher is robust's trees on the `o_near` (or
`o_players`) features: its held-out P is already on disk (`data/external/lab_clean/o_near/p_<m>.npy`, all 10).
Trees take soft targets by row replication (each row 3 times, labels H/A/D, weights = the soft target); the GRU by
a KL term. α = 0.5, fixed now.

**Why it might help.** Where the damaged inputs can't tell (the invisible-possessor frames, turnovers), the teacher
is confident and right more often (log loss −0.012); the student learns to hedge there in the way a model that saw
the players would, rather than in the way the labels' hard switches imply. Published results for distillation from
privileged information are mixed; the student never gets the information, only a better-shaped target.

**Gain guess.** P2 −0.003 to +0.004; B's bench ±0.003. Chance to pass B's bar: **~10 %**.

**The test.** P2 screen variant `distill_o_near` in the same file (c0's features, replicated rows: ~3× the fit
time, **1 slot, ~45 min**). Same leak note as idea 2 (the teacher's P for a training match came from a model that
saw the held-out match). Gate: as idea 1. B rebuild: a KL term in the copy's `lab_branch_b` training plus the
replicated-row `augment` fit, ~1.5 h plumbing + the 2.5 h rebuild.

**Fits tonight:** the screen yes; B only if everything else is out of the way.

### 4. Not for tonight (and why)

- **A learned position imputer for missing players** (from teammates, opponents, last/next seen): round 9's
  geometry lost even with perfect joins (`o_linkfill` −0.0035, `o_linkteam` −0.0049), and the probe says the
  decisive distance is the 2 m around the ball. A formation-based imputer might take 4.7 m to ~3 m in the long holes,
  still far from it, and it needs the joiner (88-92 % right). Idea 2 is the same idea aimed at the right target.
- **A learned ball estimate as an extra input:** already tried in round 7 as 11 extra columns beside robust's 97
  (ballfill.md): −0.0015, 3/10; its nearest player is the team in possession 51 % of the time against 64 % for the
  last-seen ball, which robust already reads.
- **Pitch-control / Voronoi fields:** a field loses the invisible possessor's contribution exactly as `d_home`
  does, and the probe shows what remains points at the challenger. A ghost-weighted field is idea 1 again.
- **Training-time dropout of near-ball players:** on the bench the holes are random, so a model taught to expect
  them near the ball is judged on a world where they aren't: it can only lose B's bar. It is a hedge for our match
  if its holes cluster near the ball, which can only be settled by a label-free measurement of our tracking (where
  its pieces end relative to the ball) and a bench variant built to match. That is Jose's call, not a night test.

## Cheap diagnostics worth a free slot (not candidates)

Both reuse `lab_clean.do_o_near` inside `bench/lab_ghost.py`, ~25 min on 1 slot each, and tell round 10 where the
+0.022 lives:
- `o_near_seen`: `o_near`'s fill only on frames where robust's blanked ball is seen. If it keeps most of +0.022,
  the room is at seen-ball moments (ideas 1-2's ghost touches and `gd_X`); if not, it's in the long ball gaps
  (`_last_seen`/`_next_seen`, idea 2).
- `o_near_nopc`: `o_near` with PathCRF's P and v7's cost kept as delivered. If the gain drops a lot, it flows
  through PathCRF (which pads missing players with zeros), and a ghost-aware PathCRF input becomes the next idea.

## Suggested order for the builder

| when | what | slots |
|---|---|---|
| 23:30-00:45 | `bench/lab_ghost.py`: ghost columns, unit check, own screen | 0 |
| 00:45-01:20 | screens `ghost_a`, `ghost_b` (+ `ghost_den` if ready) | 2-3 |
| 01:20 | gate (fixed above) → B rebuild or stop | |
| 01:30-04:00 | B rebuild in `$S/b16_ghost`, judged against the c0 rebuild and v15's 0.7329 | 3 |
| any free slot | `o_near_seen`, `o_near_nopc`, `distill_o_near` | 1 each |
| 04:30-08:30 | idea 2 into B if it went through its gate | 3 |
