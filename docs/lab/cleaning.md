# Lab round 9: cleaning the tracking before the program reads it

| | |
|---|---|
| **status** | **fixed 2026-11-14, 19:45 (Jose's time), before anything was scored.** Jose: "try different levels of cleaning tracking … find the ideal ones … if something doesn't affect or only makes it worse, take that into account" |
| **question** | Does repairing the tracking before PathCRF and our features read it make branch B (the chain our match runs) better, on matches and providers it has never seen? |
| **code** | `bench/lab_clean.py` (the cleaners, the ceilings, the screens); results in `bench/results/lab_clean_*.json` |
| **nothing here reads our match** | every number comes from the 10 bench matches and SkillCorner's 20 |
| **verdict (23:15)** | **no cleaner passes stage 2, so stage 3 doesn't run and v15 stays the upload.** Every real cleaner lost (24 link/fill settings: −0.003 to −0.008; held out −0.0098, 3 of 10). The ceiling (+0.024) is where the missing players near the ball really were; nothing built from the tracking recovers it |

## What there is to clean

Our tracking's faults were measured once for the bench ([../bench.md](../bench.md), label-free): the ball is
missing on 41 % of frames, mostly in long gaps (median 10.7 s); the players come in 1,219 pieces over 46
anonymous slots, 20.5 of 22 visible on average; 8 % of player speeds are blank. **The positions themselves are
not noisy** (the ball's spread around its 5-frame mean is 0.008 m). So "cleaning" here is mostly repairing:
joining pieces, filling holes, recomputing blank speeds, filling short ball gaps. Smoothing and removing false
detections are tried, but the bench has no position noise and no false detections, so it **can't measure**
them: they are reported and never adopted.

Round 7 (`ballfill`, [ballfill.md](ballfill.md)) already showed that a learned fill of the missing ball doesn't
help (87 % of live missing-ball frames sit in gaps of 10 s or more). Here the ball is only filled in short gaps,
where straight lines are exact to about a metre.

## Where the cleaned tracking goes

Only to **PathCRF** and to **robust's features** (the 97 inputs the trees and both GRUs read). The dead-ball and
restart models, v1's clues, the decoder's earlier costs, v9's snap, v11's move and **the stoppage switch** keep
reading the tracking as delivered (after v8's keeper repair). So the switch's choice on our match can't move,
and no earlier model meets inputs it never learned from. In a v16, branch A would be left as v15's; only
branch B would read the cleaned inputs.

*Added 20:05, before stage 2 had scored and before stage 3 was built (plumbing, not a choice made on results):*
in stage 3 branch B reads the cleaned tracking **everywhere it reads tracking**, including its post stages
(short, segments, timing; trial 1015 has snap and move off). On the bench this is a lab cache whose tracking is
cleaned (`lab_clean build-lab`), in an isolated copy of the repo, so every downstream cache is rebuilt from it;
the stressed copies' v1 clues still read the delivered players (`players_raw`), as they would on our match.
SkillCorner gets the same through `possession/v12.inputs(..., clean=...)` (default off: v12-v15 unchanged).

## Stage 1: the ceilings (how much each kind of cleaning could ever gain)

Each kind of repair is first done **perfectly**, from the bench's clean tracking (level 0), and measured. A
real cleaner can only gain less.

| ceiling | what the program is given |
|---|---|
| `o_players` | the true players (22 continuous tracks, every speed) with the damaged ball |
| `o_link` | the damaged pieces with the same holes, joined into the true players (perfect joining, no filling) |
| `o_speed` | the damaged pieces with the blank speeds restored |
| `o_ball1`, `o_ball3` | the damaged tracking with the true ball in every gap of 1 s or less (3 s or less) |

**Two quick measures** (the screen), on all 10 matches at 41 % damage:
- **P1, PathCRF:** team accuracy on live frames (where v6 says live) with the ball missing; the 4 matches
  PathCRF never saw (J03WR9, Metrica's 3) pooled, as in [../models/v7-pathcrf.md](../models/v7-pathcrf.md).
- **P2, the start:** robust's trees retrained leave one match out on the variant's features (IDSSE's matches
  only, as robust; Metrica's 3 from the model trained on all 7), decoded with robust's chosen setting, then
  v9's snap and v11's move: **held-out spellF1 per match**, against the same harness on the uncleaned tracking
  (`c0`); held-out log loss alongside. Metrica's 3 are never trained on, so their part is a test on a
  provider the trees never saw.

**A kind of cleaning goes on to stage 2 only if its ceiling gains +0.005 or more on P2** (mean of 10), **or
+2 points on P1 with +0.002 or more on P2.** Otherwise it's recorded as "no room" and dropped. The harness
must first reproduce robust's cached held-out P on `c0` (a check, not a score).

## Stage 2: real cleaners, several settings each

Only for the kinds with room. The settings (simplest first; a tie goes to the simpler):

| cleaner | settings |
|---|---|
| `link`: join pieces of the same team end-to-start | longest gap 1, 3 or 10 s; the start must be within 1 m + 7 m/s × gap (or 1 m + 10 m/s × gap) of where the piece ended, moved on by its last velocity (at most 1 s of it); greedy by that distance |
| `fill`: after `link`, straight lines across a joined piece's holes | holes of at most 1, 3 s, or the whole gap |
| `speed`: blank speeds recomputed from positions | ±2 frames, on or off |
| `ball`: straight lines across short ball gaps, both ends seen | gaps of at most 0.5, 1 or 2 s |
| `smooth` (can't be measured here) | Savitzky-Golay on the players' positions, 0.6 s |

Each setting is scored by P2 (and P1). The setting is picked **leave one match out** (each match scored with the
setting best on the other 9, as `bench/lab.py`'s `judge`). **A cleaner goes on to stage 3 only if its held-out
gain on P2 is +0.003 or more, better on 6 of 10, and Metrica's 3 don't drop on the mean.** Cleaners that pass
are then combined (the best of each kind together, scored the same way); the combination replaces its parts
only if it beats the best single one.

## Stage 3: the full branch B, against v15's

The finalist (at most two) is built into branch B as v15 builds it ([../models/v15-branch-b.md](../models/v15-branch-b.md)):
`augment`'s trees and `gru_b` retrained on the cleaned features (and their stressed copies), **trial 1015's
settings unchanged** (no new search), the post stages retrained on stressed copies.

**The bar (fixed now, the shape of round 8's):**
1. **B's bench** (the mean of p25, p50 and stressed, per match): **+0.005 or more against v15's B, better on 7
   of 10.**
2. **No variant** (p25, p50, stressed) **down more than 0.002**; **neither source** (IDSSE's 7, Metrica's 3)
   down on the mean.
3. **SkillCorner's 20** (deployment models): **no loss beyond 0.005** on the mean.
4. The switch reads the raw tracking, so its choices can't change; its bar's part 3 is rerun for information
   with the new B.
5. The port gives the lab's letters on bench units, and the file on our match comes out the same on three runs.

**Cut-off:** if nothing has passed stage 3 by **Nov 15, end of day**, v15 ships. A pass goes to Jose with this
page first; nothing is adopted without his go-ahead.

## Results

### Stage 1: the ceilings (2026-11-14, ~20:00)

**The harness reproduces robust exactly:** on the tracking as the bench gives it (`c0`), the held-out P equals
`data/external/lab_robust/p_blank_*.npy` to the last bit on all 10 matches, and PathCRF's accuracy with the ball
missing on the unseen matches is the v7 page's 85.7 %.

| ceiling | start's spellF1 (P2) | gain | better on | IDSSE / Metrica | log loss | PathCRF, ball missing, unseen (P1) |
|---|---|---|---|---|---|---|
| `c0` (as delivered) | 0.6747 | | | | 0.2266 | 85.7 % |
| **`o_players`** (the true players: joined, no holes) | **0.6987** | **+0.0241** | **10/10** | +0.0215 / +0.0301 | −0.0129 | 86.7 % (+1.1) |
| `o_link` (perfectly joined, the holes kept) | 0.6702 | −0.0045 | 4/10 | −0.0018 / −0.0108 | +0.0006 | 85.4 % (−0.2) |
| `o_ball3` (the true ball in every gap of ≤ 3 s) | 0.6663 | −0.0083 | 2/10 | −0.0069 / −0.0117 | −0.0011 | 85.7 % (±0) |

- **The players have room, and all of it is in the holes.** With every player seen on every frame the start
  gains +0.024 on 10 of 10 matches, Metrica's 3 (never trained on) the most. Joining the same pieces perfectly
  but leaving their holes gains nothing (−0.0045, within this measure's noise: a change that should do nothing
  moves it by about ±0.005). On an average frame 1.65 of 22 players are missing; robust's features (nearest
  players, counts, spreads) feel them, PathCRF a little (+1.1 points). Joining matters only because a hole can
  be filled once we know whose it is. **Goes on to stage 2** (gate: +0.005).
- **The ball has none.** Gaps of 3 s or less hold only 2.3 % of the live frames without a ball (0.5 % for 1 s),
  and filling them with the *true* ball makes the frames a little better (log loss −0.0011) and the letters
  worse (−0.0083, 2 of 10). `o_ball1` wasn't run (its gaps are a fifth of these). **Dropped**, with `ball`.
- **Blank speeds can't matter here:** robust's features and PathCRF compute speeds from positions and never
  read the speed channel. `o_speed` and `speed` dropped without a run.

### Stage 2: what changed while it ran (written down as it happened)

- **20:18, a packing change.** The first four settings (`fill all`) came back below `c0` (the first: −0.0073,
  3 of 10). Joined tracks were packed into as few slots as possible, reusing the first free slot, which puts two
  people back to back in one slot; the delivered tracking (and `degrade`) deals each piece to the slot free
  longest. Changed to that rule, over the delivered slot count. Counting slot-frames with a speed above 12 m/s
  on J03WR9 (delivered 415; old packing 422; new 418) shows it **wasn't** the cause of the loss; it's kept
  because it matches the delivered convention, and **all 24 settings were rerun with it** (the four early
  results discarded).
- **20:20, one more diagnostic** (needs the truth, so it can't be a candidate): `o_linkfill`, perfect joins with
  straight lines across every hole of each true player. Measured on the bench, the real joiner is right on
  88-92 % of its joins (gap 10 s), and straight lines across rightly joined holes are 0.0 m off under 1 s,
  0.2-0.3 m for 1-3 s and 1.3-1.5 m for 3-10 s. If `o_linkfill` gains and the real joiner doesn't, the wrong
  joins are what costs; if neither gains, straight lines aren't enough.
- **20:38, `o_linkfill`: −0.0035, 4 of 10** (log loss −0.0026, PathCRF +0.3 points). Perfect joins and every
  hole filled (21.99 of 22 players seen) gain nothing on the letters: **the ceiling's +0.024 is where the
  missing players really were, not that they are all there.** Where the holes' frames are, measured on 4 bench
  matches with perfect joins:

  | hole length | share of hole frames | straight line, mean error | line + the team's shift |
  |---|---|---|---|
  | < 1 s | 0.7 % | 0.03 m | 0.09 m |
  | 1-3 s | 4.8 % | 0.26 m | 0.42 m |
  | 3-10 s | 30.0 % | 1.51 m | 1.48 m |
  | 10-30 s | 54.4 % | 4.72 m | 3.59 m |
  | > 30 s | 10.1 % | 10.72 m | 6.52 m |

  Two thirds of the missing players' frames sit in holes longer than 10 s, which a 10 s joiner can't reach and a
  straight line misses by metres.
- **20:45, one more diagnostic, `o_linkteam`** (rule fixed before it ran): perfect joins, straight lines across
  holes of 3 s or less, and beyond that the straight line plus how far the visible teammates' centre moved off
  its own line over the hole (players move with their block). If even this stays far from `o_players`, a fill
  built from the tracking alone can't carry what the ceiling measured, and the family stops here.
- **21:04, `o_linkteam`: −0.0049, 4 of 10** (log loss −0.0011). It does stay far: **the family stops.**
- **21:05, a last diagnostic, for the next phase** (`o_near`, radius fixed before it ran): perfect joins, and a
  missing player's true position **only where he is within 15 m of the true ball**, roughly what a camera
  following the ball sees. It fills 0.37 of the 1.65 players missing on an average frame.

| diagnostic (all need the truth) | what fills the holes | P2 gain | better on | Metrica | log loss |
|---|---|---|---|---|---|
| `o_players` | every missing player, true position | **+0.0241** | 10/10 | +0.0301 | −0.0129 |
| **`o_near`** | **only the missing players within 15 m of the ball, true position** | **+0.0216** | **10/10** | **+0.0329** | −0.0116 |
| `o_linkfill` | every missing player, straight line between true pieces | −0.0035 | 4/10 | −0.0060 | −0.0026 |
| `o_linkteam` | the same, long holes following the team's centre | −0.0049 | 4/10 | −0.0066 | −0.0011 |
| `o_link` | nothing (pieces joined perfectly) | −0.0045 | 4/10 | −0.0108 | +0.0006 |

**Nine tenths of the ceiling is the players near the ball.** No fill made from the tracking itself gets any of
it; a source that *sees* the area around the ball might. The broadcast recording does exactly that (its own
tracking, 51 slots, homography usable on 65.8 % of its frames, not synchronised with the tactical feed). That's a
build of its own (synchronising the two feeds, a bench imitation of what the broadcast adds, every model
retrained), and it would need statistics of our match's broadcast to imitate it: not for this submission.

One caveat on the size: on the bench the holes are random (`degrade` cuts pieces regardless of play); on our
match players more likely go missing in crowds, which are near the ball. So the near-ball share of the missing
players, and what filling it is worth, may be larger on our match than here.

### Stage 2: the verdict (23:15)

All 24 `link`/`fill` settings and `smooth15` were screened on P2 (the robust trees retrained per setting, 8 folds)
with the longest-free packing. `speed` and `ball` weren't run: the speed channel is read by neither the robust
features nor PathCRF, and the best case for the ball (`o_ball3`, the true ball over gaps of 3 s or less) already
lost (−0.0083).

| setting | P2 gain | better/worse | IDSSE | Metrica | PathCRF (missing, unseen) |
|---|---|---|---|---|---|
| `link_g10_v7` (the best real cleaner) | −0.0032 | 4/6 | −0.0042 | −0.0011 | −0.3 |
| `link_g1_v10` | −0.0036 | 5/5 | −0.0017 | −0.0081 | +0.1 |
| `link_g3_v7_fill3` (= `fillall`: a 3 s join leaves no longer hole) | −0.0039 | 5/5 | −0.0025 | −0.0069 | 0.0 |
| … 19 more settings | −0.0048 to −0.0081 | 2-5 better | | | |
| `link_g10_v10` (the worst) | −0.0084 | 3/7 | −0.0060 | −0.0139 | −0.3 |
| `smooth15` (0.6 s Savitzky-Golay on the players) | −0.0028 | 4/6 | +0.0022 | −0.0146 | 0.0 |

**The registered judge** (`python -m bench.lab_clean stage2`, each match scored with the setting best on the
other 9): **held out −0.0098, better on 3 of 10** (IDSSE −0.0084, Metrica −0.0131); chosen on all 10:
`link_g10_v7`. The gate (+0.003, 6 of 10, Metrica not down) **is not passed**, so nothing goes to stage 3 and
nothing is combined. Results: `bench/results/lab_clean_stage2_link.json`, `lab_clean_report.json`.

- **`smooth15` measures only the cost of smoothing.** The bench's damage adds noise to the ball only
  (`degrade.py`), never to the players, so smoothing can't show a gain here; it costs −0.0028 on clean positions,
  most of it on Metrica. Whether our match's players are noisy enough for smoothing to pay is not measurable on
  this bench.
- **The settings differ from one another by about ±0.003**, about the screen's noise; none is above `c0`.

**What the round leaves:** the value is in the true positions of the missing players near the ball (+0.022 of
+0.024, 10/10), which no fill from the tracking provides. The next-phase lead is a source that sees around the
ball (the broadcast tracking), not cleaning.

