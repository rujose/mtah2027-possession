# Night builder "ghost": inputs that point to players missing near the ball

| | |
|---|---|
| **status** | plan fixed 2026-11-14, 23:45, before any score. **Finished 11-15, 01:35: nothing passes the screen's gate, so branch B was not rebuilt.** `ghost_a` +0.0006 (5/10), `ghost_b` −0.0004 (7/10); held out −0.0031 (4/10, Metrica −0.0090). Idea 2 (`ghost_den`) −0.0005 (6/10, Metrica −0.0043). Nothing to adopt |
| **in one line** | the ghosts sharpen the frames whose possessor is missing (log loss −0.004 to −0.036 there) but reach about a tenth of what the truth gives on them and nothing on loose-ball frames, which hold 45 % of it; the letters don't move beyond the screen's noise |
| **question** | Does telling the models *who is missing and where he was last / will next be seen* (new input columns; positions untouched) make the start better on the bench, and then branch B? |
| **proposal** | [ideas_inputs.md](ideas_inputs.md), idea 1 (and its optional extras if time allows) |
| **code** | `bench/lab_ghost.py` (columns, unit check, screen, gate); results `bench/results/lab_ghost_*.json`; caches `data/external/lab_ghost/` |
| **nothing here reads our match** | the 10 bench matches, and SkillCorner's 20 if B is rebuilt |

## Why (from the proposal's probe, 3 bench matches)

- Where the player on the ball is invisible (4-6 % of live frames with the ball seen), the nearest *visible* player is
  on the team in possession only 24-34 % of the time: robust's 97 inputs point the wrong way there.
- The damaged tracking still holds a pointer to him: the end (or start) point of his own piece, seconds away. A
  "ghost" within 5 m and nearer than every visible player names the right team 85-92 % of the time, with 0.1-0.6 %
  false alarms where he is visible.
- 6-9 % of ball touches have an invisible toucher; 82-86 % of those are orphans (no visible player within 2 m), and a
  ghost within 5 m finds about half of them, team right 87-98 %.

## The columns (fixed now, before any score)

All from the **damaged, blanked** tracking robust's features read (`lab_robust.blanked`: the still ball blanked;
players as delivered), within each half, NaN outside the halves, in home/away pairs so `swap=True` exchanges them.

**Pieces**: every unbroken run of a slot within a half (`lab_clean.pieces`). **E_X** (per half, per team):
min(11, the 99th percentile of X's seen-player count over the half's frames, `method="nearest"`).

**A pending ghost of team X at frame t** (G = 10 s, reach(dt) = 1 m + 7 m/s × dt, round 9's `link_g10_v7` constants;
distances from the point itself, no velocity projection):
- *end ghost*: the last point of a piece of X whose last frame e satisfies t − G ≤ e < t, unless a piece of X starts
  at some a in (e, t] within reach((a − e)/25) of that point (he reappeared nearby); time gap (t − e)/25 s;
- *start ghost*: the first point of a piece of X whose first frame a satisfies t < a ≤ t + G, unless a piece of X has
  its last frame at some e in [t, a) within reach((a − e)/25) of that point; time gap (a − t)/25 s;
- only on frames where X's seen count is below E_X, and only within t's half.

| family | columns (per team X = home, away) | definition |
|---|---|---|
| ghosts (14) | `gd_X`, `gd_X_last_seen`, `gd_X_next_seen` | distance from the ball (seen; last seen; next seen, as robust's `filled`) to X's nearest pending ghost; 60 (FAR) if none; NaN where that ball is NaN |
| | `gt_X`, `gt_X_last_seen`, `gt_X_next_seen` | that ghost's time gap (s, cap 10; 10 if none); NaN where that ball is NaN |
| | `gn_X_10m` | X's pending ghosts within 10 m of the (seen) ball, cap 3; NaN where the ball is NaN |
| ghost touches (8) | `since_gtouch_X`, `until_gtouch_X`, `gtouches_X_past_3s`, `gtouches_X_next_3s` | robust's touch encodings (cap 10 s; 3 s counts) of **orphan ball events** given to X |
| counts (`ghost_b` only, 2) | `miss_X` | E_X − X's seen count on the frame, clipped to 0-3 |

**Orphan ball event**: a run of frames with v11's velocity change ≥ 3.0 m/s (`touches.velocity_change` on the blanked
ball, the same ΔV as v11 on `all`) on **none** of whose frames a visible player of either team is within 2 m (REACH)
of the ball, i.e. a run v11's detector gives to nobody. Its frame is the run's largest change. It is **given to X** when,
on that frame, X's nearest pending ghost is within 5 m of the ball and nearer than the other team's (strictly).

**Variants**: `ghost_a` = robust's 97 + the 22 ghost columns (119); `ghost_b` = `ghost_a` + `miss_X` (121).
`g0` = robust's 97 alone through my screen (the harness check, not a candidate).

**Unit check before any screen** (J03WR9): X_swapped's ghost columns equal X's with home/away exchanged (bit for
bit); and, with the clean tracking (only to count, never an input), orphan events with an invisible true toucher that
get a ghost touch, and its team right, near the probe's rates (~48 %, ~95 %).

## The screen (P2, round 9's)

`bench/lab_ghost.py screen VARIANT`: robust's trees (learned's settings) retrained leave one match out on the
variant's columns (IDSSE's 7 folds; Metrica's 3 from the model trained on all 7), both orientations, every 10th
graded frame; decoded with robust's chosen setting, then v9's snap and v11's move on the original tracking; held-out
spellF1 per match, against `bench/results/lab_clean_c0.json` (c0 = robust as delivered, **0.6747**, log loss 0.2266).
Base columns read from robust's cache (`data/external/lab_robust/<m>.npz`, read only); nothing is written into any
shared cache.

**Harness check first:** `g0` must give c0's held-out P bit for bit and 0.6747, before any ghost variant is scored.

## The gate to a B rebuild (round 9's stage-2 gate, fixed now)

Pick between `ghost_a` and `ghost_b` **leave one match out** (each match scored with the variant whose mean is better
on the other 9; ties, rounded to 6 decimals, to `ghost_a`), against c0:

| | needed |
|---|---|
| held-out P2 gain | **+0.003 or more** |
| matches better | **6 of 10 or more** |
| Metrica's 3 | **not down on the mean** |
| held-out log loss | **not up by more than 0.001** |

Below the gate: recorded here, no B rebuild.

## The B bar (night.md's, unchanged; only if the gate is passed)

Against v15's B as confirmed (trial 1015's settings fixed; B's bench 0.7329):

| | needed |
|---|---|
| B's bench (each match's mean of p25, p50, stressed) | **+0.005 or more (0.7379), better on 7 of 10** |
| each stress variant (p25, p50, stressed) | **none down more than 0.002** |
| sources (IDSSE's 7, Metrica's 3) | **neither down on the mean** |
| SkillCorner's 20 (deployment models) | **down no more than 0.005** (v15's B 0.4085, so 0.4035) |
| noise rule | if a second seed of the single network moves B's bench by 0.003 or more, a single-network pass repeats with a second seed |

## Run plan

| when | what | slots |
|---|---|---|
| 23:45-00:40 | `bench/lab_ghost.py`: columns, unit check, own screen | 0 |
| ~00:40 | `g0` (harness check, ~10 min) | 1 |
| after `g0` checks | `ghost_a`, `ghost_b` | 1 each |
| then | the gate (`python -m bench.lab_ghost gate`) | 0 |
| only through the gate | B rebuilt in an isolated copy (`$S/b16_ghost`, `stage3.sh`'s pattern, the bench's lab cache symlinked since the tracking doesn't change; the copy's `frame_features` appends the ghost columns, so robust's cache, the stressed copies, `gru_b`, the gate's fits and SkillCorner's inputs all get them; the copy's `p_blank` = the screen's P), judged with the B bar against v15's B and against the c0 rebuild | 3 |
| if nothing passes | optional diagnostics `o_near_seen`, `o_near_nopc`, or idea 2 (`ghost_den`), each with its gate written here first | 1 each |

## The optional extras (fixed 23:55, before any score)

Run on free slots, after `ghost_a`/`ghost_b`, whatever their gate says.

**Two diagnostics** (they read the clean tracking, so they are never candidates; they say where round 9's `o_near`
+0.0216 lives). Both are `o_near` (perfect joins, a missing player's true position where he is within 15 m of the
true ball), screened exactly as round 9 screened it, with one change each:

| diagnostic | the change | how it's read (fixed now) |
|---|---|---|
| `o_near_seen` | the fill only on frames where robust's blanked ball is seen | keeps ≥ 2/3 of `o_near`'s gain: the room is at seen-ball moments (ghost touches, `gd_X`); ≤ 1/3: in the ball gaps (`_last_seen`/`_next_seen`, idea 2); between: both |
| `o_near_nopc` | PathCRF's P and v7's cost kept as delivered (not rerun on the filled players) | keeps < 1/2 of `o_near`'s gain: it flows mostly through PathCRF (a ghost-aware PathCRF input is the next idea); ≥ 1/2: through robust's own features |

**Idea 2, `ghost_den`** (a candidate): 6 regressors (robust's tree settings, squared error, every 10th graded frame,
both orientations, the swapped targets for the swapped rows) from robust's 97 + `ghost_a`'s 22 to the values that
round 9's `o_players` features (true players, the same damaged and blanked ball; `data/external/lab_clean/o_players/`,
read only) hold for `d_diff`, `d_diff_last_seen`, `d_diff_next_seen`, `n_home_2m − n_away_2m`, `d_home`, `d_away`.
Rows whose target is NaN are left out of that target's fit; a prediction is NaN where the damaged column it shadows
is NaN (for the count: where `n_home_2m` is). Cross-fitted: each IDSSE match's 6 columns from regressors fitted on
the other 6 IDSSE matches, each Metrica match's from all 7 (a held-out match's clean tracking shapes the training
matches' columns, never its own: the second-order leak the bench already accepts). Appended as `den_*`: 125 columns.
**Gate:** idea 1's, against c0. If a ghost variant also passes, `ghost_den` replaces it for the B rebuild only if it
beats it by +0.002 or more on P2 held out (the picked ghost variant per fold).

## Results

### The checks (before any ghost variant was scored)

- **Unit check, J03WR9** (`python -m bench.lab_ghost unit`, 3 s): X_swapped's 24 columns equal X's with home and
  away exchanged, bit for bit. Against the probe (clean tracking only to count):

  | | here | probe |
  |---|---|---|
  | orphan events (live) with an invisible true toucher that get a ghost touch | 44.4 % (of 99) | ~48 % |
  | ... their team right | 100 % | ~98 % |
  | invisible on-ball player, ball seen: a ghost within 5 m and nearer than every visible player | 41.8 % | 47 % |
  | ... its team right | 91.9 % | 92 % |
  | false alarms (on-ball player visible) | 0.1 % | 0.4 % |

  The coverage is a little below the probe's because the pending rule (he hasn't reappeared nearby) and the E_X
  gate drop some ghosts; the false alarms drop with them. 13 of the 57 ghost touches fall on orphan events whose
  true toucher is visible (just beyond 2 m) or where nobody was within 2 m.
- **Harness check, `g0`** (robust's 97 through my screen, 12 min): **held-out P equal to round 9's c0 and to
  robust's own on all 10 matches (largest difference 0); spellF1 0.6747, log loss 0.2266.** The screen is c0's.

### The screen (P2, against c0 0.6747), 00:37-01:00

| variant | P2 spellF1 | gain | better / worse | IDSSE | Metrica | log loss |
|---|---|---|---|---|---|---|
| `ghost_a` (119 columns) | 0.6753 | **+0.0006** | 5/5 | +0.0016 | −0.0016 | +0.0003 |
| `ghost_b` (121) | 0.6743 | **−0.0004** | 7/3 | +0.0014 | −0.0046 | +0.0002 |

Per match (gain against c0):

| match | `ghost_a` | `ghost_b` | gate's pick (on the other 9) | held out |
|---|---|---|---|---|
| J03WMX | +0.0108 | −0.0047 | `ghost_b` | −0.0047 |
| J03WN1 | +0.0043 | +0.0145 | `ghost_a` | +0.0043 |
| J03WOH | −0.0227 | −0.0289 | `ghost_a` | −0.0227 |
| J03WOY | −0.0024 | +0.0064 | `ghost_a` | −0.0024 |
| J03WPY | +0.0080 | +0.0081 | `ghost_a` | +0.0080 |
| J03WQQ | +0.0146 | +0.0138 | `ghost_a` | +0.0146 |
| J03WR9 | −0.0014 | +0.0010 | `ghost_a` | −0.0014 |
| metrica1 | +0.0009 | +0.0022 | `ghost_a` | +0.0009 |
| metrica2 | −0.0044 | +0.0075 | `ghost_a` | −0.0044 |
| metrica3 | −0.0014 | −0.0234 | `ghost_b` | −0.0234 |

**The gate (`python -m bench.lab_ghost gate`): held out −0.0031, better on 4 of 10 (worse 6), IDSSE −0.0006,
Metrica −0.0090, log loss +0.0006. Needed +0.003, 6 of 10, Metrica not down, log loss ≤ +0.001: NOT PASSED.**
Chosen on all 10: `ghost_a`. So **no B rebuild** (results: `bench/results/lab_ghost_ghost_a.json`,
`lab_ghost_ghost_b.json`, `lab_ghost_gate.json`).

**Where it acts** (`python -m bench.lab_ghost where VARIANT`; live frames of all 10 matches pooled, the clean tracking
only to count who is on the ball):

| live frames | share | team right, c0 → `ghost_a` / `ghost_b` | log loss, c0 → `ghost_a` / `ghost_b` |
|---|---|---|---|
| ball seen, on-ball player seen | 44.5 % | 94.6 % → +0.0 / −0.0 pts | 0.1427 → −0.0009 / −0.0012 |
| **ball seen, on-ball player missing** | 3.6 % | 88.9 % → −0.3 / +0.1 pts | 0.2692 → **−0.0044 / −0.0076** |
| ball blanked, on-ball player seen | 12.4 % | 88.2 % → +0.0 / −0.1 pts | 0.5408 → **+0.0082 / +0.0086** |
| **ball blanked, on-ball player missing** | 1.1 % | 85.8 % → +0.8 / +0.7 pts | 0.5767 → **−0.0356 / −0.0316** |
| nobody within 2 m | 38.5 % | 87.1 % → +0.0 / +0.2 pts | 0.4565 → +0.0004 / +0.0001 |

- **The mechanism works where it was aimed, but the room there is small.** On the frames whose on-ball player is
  missing (4.7 % of live frames) the trees get sharper (log loss −0.004 to −0.036) and, with the ball blanked,
  right a little more often (+0.7-0.8 points). Robust's trees already had those frames right 86-89 % of the time
  from context, as the proposal's probe found (92-94 % on two matches).
- **It costs a little elsewhere:** frames with the ball blanked and the on-ball player seen (12.4 %) lose +0.008 of
  log loss: 22 more columns of mostly "no ghost" give the trees more ways to overfit 7 training matches.
- **On the letters, nothing moves beyond the screen's noise.** The per-match swings (J03WOH −0.023 / −0.029,
  metrica3 −0.023 with `ghost_b`, J03WQQ +0.014) are of the size round 9 saw for changes that do nothing (±0.005 on
  the mean, ±0.03 on a match): a different tree on 119 columns decodes some possessions differently.
- `miss_X` (`ghost_b`) helps IDSSE's letters on 5 matches and costs Metrica −0.0046 (metrica3 −0.023): the
  "one team is short" cue is the riskiest column, as the proposal warned, and here it doesn't carry to the
  provider the trees never saw.

### The extras (01:01-01:29), each with the rule fixed above before it ran

**Idea 2, `ghost_den`: 0.6742, −0.0005, better on 6 of 10 (worse 3), IDSSE +0.0012, Metrica −0.0043, log loss
−0.0001. The gate (+0.003, Metrica not down) is NOT passed.** (Against the ghost pick per fold: +0.0027, 6/10; moot,
neither passes.) Per match: J03WMX ±0, J03WN1 +0.0176, J03WOH −0.0245, J03WOY −0.0059, J03WPY +0.0007, J03WQQ
+0.0122, J03WR9 +0.0082, metrica1 +0.0043, metrica2 +0.0033, metrica3 −0.0206.

The regressors do what they were built for: on every held-out match the 6 near-ball values come out nearer the
clean tracking's than the damaged ones (mean absolute error, all frames where the value exists; damaged in brackets):

| | `d_diff` | `d_diff_last_seen` | `d_diff_next_seen` | `n_home_2m − n_away_2m` | `d_home` | `d_away` |
|---|---|---|---|---|---|---|
| mean of the 10 | 0.64 m (0.91) | 0.63 m (0.83) | 0.62 m (0.82) | 0.06 (0.06) | 0.33 m (0.51) | 0.35 m (0.46) |
| range | 0.57-0.77 (0.75-1.20) | 0.50-0.82 (0.68-1.05) | 0.54-0.69 (0.70-0.95) | | 0.26-0.40 (0.41-0.74) | 0.25-0.45 (0.34-0.72) |

and on the frames whose on-ball player is missing the trees get sharper than with `ghost_a` (ball seen: log loss
0.2692 → 0.2568, team right +0.5 points; ball blanked: −0.0233, +0.3 points), at a smaller cost elsewhere (ball
blanked, on-ball seen: +0.0066). The letters still don't move: a third of a metre nearer on average is far from the
2 m that decides who is on the ball.

**The two diagnostics** (round 9's `o_near`, +0.0216, 10/10, IDSSE +0.0168, Metrica +0.0329, log loss −0.0116):

| diagnostic | P2 gain | share of `o_near`'s | better on | IDSSE | Metrica | log loss | reading (rule fixed before it ran) |
|---|---|---|---|---|---|---|---|
| `o_near_seen` (fill only where robust's ball is seen) | **+0.0178** | **82 %** | 9/10 | +0.0129 (77 %) | +0.0290 (88 %) | −0.0098 | ≥ 2/3: **the room is at seen-ball moments** |
| `o_near_nopc` (PathCRF not rerun) | **+0.0170** | **79 %** | 8/10 | +0.0114 (68 %) | +0.0303 (92 %) | −0.0094 | ≥ 1/2: **it flows through robust's own features, not PathCRF** |

- `o_near_seen` fills only 40-50 % of `o_near`'s player-frames on IDSSE (the ball is blanked on 57-66 % of their
  frames) and keeps 77 % of its gain there; on Metrica (ball seen on most live frames) it fills 90-95 % and keeps 88 %.
  So the frames with the ball seen carry most of the value; the long ball gaps (`_last_seen`/`_next_seen`) little.
- PathCRF carries about a fifth of the gain (+0.0046 of +0.0216): a ghost-aware PathCRF input is not the lever.

**Where the ceiling's gain sits** (`where`, the same split for `o_near`'s held-out P, read only):

| live frames | share | team right, c0 → `o_near` | log loss, c0 → `o_near` | of the ceiling's log-loss gain* | `ghost_den` reached |
|---|---|---|---|---|---|
| ball seen, on-ball seen | 44.5 % | 94.6 → 94.9 % | −0.0063 | 15 % | −0.0017 |
| **ball seen, on-ball missing** | 3.6 % | **88.9 → 95.3 %** | **−0.1407** | **27 %** | −0.0124 (9 %) |
| ball blanked, on-ball seen | 12.4 % | 88.2 → 88.0 % | −0.0033 | 2 % | +0.0066 |
| ball blanked, on-ball missing | 1.1 % | 85.8 → 90.1 % | −0.1812 | 11 % | −0.0233 (13 %) |
| **nobody within 2 m (loose ball, passes)** | **38.5 %** | **87.1 → 88.2 %** | **−0.0215** | **45 %** | +0.0010 (none) |

\* share × change, as a share of the sum over live frames (−0.0186).

**What this says for the next round.** The truth near the ball is worth +0.022, four fifths of it at moments the ball
is seen, and almost all of it through robust's features. Two places hold it: the frames whose possessor is missing
(38 % of the gain; the ghosts reach about a tenth of it there, because "where his piece ended, seconds ago" is metres
from where he is), and the **loose-ball frames** (45 %; nobody seen within 2 m, the ball between players), where
the ghosts give nothing: there the invisible players `o_near` restores are 2-15 m from the ball and moving, and an
end point seconds old says little about who gets there first (an interpretation, not measured). Neither a pointer into the past (ghosts) nor a regression of the near-ball distances
(`ghost_den`) recovers it from the damaged tracking. That agrees with round 9: what's missing is the positions
themselves, which only a source that sees around the ball can give.

## For Jose

- **Nothing to adopt or decide from this page.** No variant passed the screen's gate, so B was never rebuilt and the
  B bar was never applied; v15 is untouched.
- **For the next round:** the two diagnostics narrow round 9's lead. The value is in the true positions of the
  players near a *seen* ball (82 %), read by robust's features (79 %), split between missing possessors and loose
  balls. Nothing built from the damaged tracking tonight (fills in round 9; ghosts, counts and a learned denoiser
  here) reaches it, which supports round 9's conclusion: only a source that sees around the ball (the broadcast
  tracking) could.
- **Reusable for any later B rebuild:** `$S/ghost/stage3_ghost.sh` takes one slot per step (it starts under a busy
  queue, unlike `--need 3`), and `$S/ghost/sc_fast.py` recomputes SkillCorner's features from main's cached v12
  inputs (read only, checked column for column) in seconds a match instead of `--sc-prepare`'s ~6 min (the c0
  rebuild's long pole). Neither was run tonight. (`$S` = the session's scratchpad.)

## Log

- 23:45 page written; features, gate and bar fixed before any score.
- 23:55 the optional extras (two diagnostics, idea 2) fixed on this page before any score.
- 23:59 `g0` reproduces c0 exactly; `ghost_a` and `ghost_b` queued (one slot each).
- 00:37-00:45 `ghost_b` screened (the slots were full until then: the c0 rebuild held 3, the network builder 1).
- 00:48 the c0 rebuild (b16_c0) finished: it reproduces v15's B exactly (0.7329, all 30 units ±0; SkillCorner
  0.4085; 1 h 36 min). A ghost rebuild, if any, is compared against it as against v15's B.
- 00:49 `ghost_a` started.
- 01:00 both screened; **the gate is not passed (held out −0.0031, 4/10, Metrica −0.0090): no B rebuild.**
  The rebuild script stays unused (`$S/ghost/stage3_ghost.sh`, with SkillCorner's features recomputed from main's
  cached inputs instead of a 2 h `--sc-prepare`).
- 01:01 the extras queued in the page's order: `ghost_den`, `o_near_seen`, `o_near_nopc` (one slot each).
- 01:26 `ghost_den` −0.0005 (6/10, Metrica −0.0043): gate not passed. `o_near_seen` +0.0178 (82 % of `o_near`).
- 01:29 `o_near_nopc` +0.0170 (79 %). The two diagnostics' feature files (2.2 GB, rebuildable in ~15 min) removed
  after scoring to spare the shared disk (11 GB free), as round 9's screen does; their held-out P are kept.
- 01:35 done. Nothing passed; nothing to rebuild; no shared cache written.
