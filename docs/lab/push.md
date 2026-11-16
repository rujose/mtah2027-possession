# The two-day push (2026-11-15 to 11-16): three tracks toward v16

Jose, 2026-11-15 (~10:10, machine clock UTC−5): "two full days of work is a lot of work … if there is a moment to make
a difference i think is now". Asked "Start all three tracks, with netmet5 trained on Metrica's sample games for
v16?", Jose: **"Yes, all three"**.

**Decided with that answer:** Metrica's 3 sample games may now train the network we deploy (until now they were used
to check and choose settings only, Jose's decision of 2026-10-30). They ship with no licence file; Metrica publishes
them and runs the challenge.

## Why these three

- Our match is Metrica's tracker under Metrica's answer convention; 7 of the bench's 10 matches are DFL, so the B
  bar's "7 of 10" is mostly a DFL vote. On Metrica's 3, scored as our match will be (each by a network that never saw
  it), `netmet5` gains +0.040, 3/3 ([night/network.md](night/network.md)).
- The night's one lever that moved the bench was **what B's network trains on**; the gain came from Metrica-like data.
- Round 9's largest remaining value is the **missing players near the ball** (+0.022 ceiling,
  [cleaning.md](cleaning.md)); the tactical tracking can't supply them, and the broadcast camera follows the ball.

## The tracks

| track | what | judged against | page | deadline (machine clock) |
|---|---|---|---|---|
| **1. v16** | v15 with branch B's network replaced by `netmet5` (netmet's recipe, seeds 0-3 averaged; trial 1015's settings); built, ported and checked like v15 | the lab's `netmet5` (the port must reproduce it) | [../models/v16-netmet.md](../models/v16-netmet.md) | built and checked by **11-15 23:00** |
| **2. netweight** | B's network leaning harder on Metrica's games: weighted 2× / 4×, more damage draws | v15's B (Metrica's rule) **and** `netmet5` | [push/netweight.md](push/netweight.md) | screen 11-15 22:00; finalist 11-16 09:00; stop **11-16 12:00** |
| **3. broadcast** | broadcast-tracked players near the ball merged into the tactical tracking; simulated on the bench from documented statistics only | v15's B (the B bar) plus robustness | [push/broadcast.md](push/broadcast.md) | screen 11-15 evening; judge 11-16 09:00; stop **11-16 12:00** |

**Then:** 11-16 afternoon, the winner goes into v16 on Jose's go-ahead; the final run, the Linux byte-for-byte check
and the validator (pass/fail only). 11-17: the technical document; submission by 23:59 CET (17:59 machine clock).

## Rules for every track (the night's, unchanged)

- **Our match (BAR-ATH) is never read:** not its tracking (the broadcast set included), its labels, the public truth,
  nor any statistic of our output. The viewer page is never opened. On our match only scripts whose output is paths,
  SHA-256, row counts or lane names; no `--show-branch`; the switch is never overridden.
- **Training and tuning on external data only** (the organisers' rules).
- **Every bar is written on the track's page before its first score.** Seed noise on B's bench is 0.0035: a pass by a
  single network counts only if seed 1 passes too.
- **The machine** (8 GB RAM, 8 cores) is shared: heavy jobs only through the scratchpad's `slot.py` (4 slots), at most
  2 slots per track at a time.
- **Disk:** about 13 GB free at the start. `b16_c0` is read-only. Each track deletes its own bulky caches when done
  and nothing else. Under 3 GB free: stop and report.
- **No existing file edited** beyond what each brief allows; nothing committed or pushed.
- **Nothing adopted:** the upload changes only on Jose's go-ahead on the finished page.

## Log

- 10:35 the three tracks launched.
- ~10:45 Jose freed disk: about 28 GB free. Round 9's cache (data/external/lab_clean) kept; track 3 may hold two
  working copies at once.
- 11:33 tracks 2 and 3 held all 4 slots and track 1's checks waited: both told to hold at most 1 slot until track 1
  is done.
- 11:52 **track 1 done**: v16 built and checked ([v16](../models/v16-netmet.md)): the four nets retrain bit-identical,
  the port gives 0 differing frames (float32 and float64), SkillCorner 0.4090 reproduced per match to 5e-6; on our
  match three runs give SHA-256 `90599088…0523c`; v15's file unchanged (`209a2081…7b5e`). Not the upload. Tracks 2 and
  3 back to 2 slots.
- 12:10 **v16 adopted as the upload** (Jose: "Yes, adopt v16"); track 2 may still replace its network. A clean copy
  (fresh clone + the working tree's changes + v16's files, only the organisers' data) gives v16's SHA-256; the
  validator passes.
- 12:25 track 3 told to judge its merge on top of v16's B (no retrain: off_v16, central_v16, pessimistic_v16) right
  after its seed-1 noise check; the retrain on v15's recipe drops in priority.
- 17:05 track 2: W4's screen in. Against `ens2` (2 seeds each): bench +0.0033, Metrica's 3 +0.0192 (3/3), DFL's 7
  −0.0034. W2's (14:26): +0.0045, +0.0105, +0.0019. The page's rule (fixed before any score) makes the best on
  Metrica's 3 the finalist, so W4 goes ahead unless the combination beats it; bar (b) allows DFL's 7 no lower than
  `netmet5` by more than 0.003, so W4 may fail there. The rule is kept as written (changing it now would choose after
  seeing). D3 training.
- 19:15 **track 3 done** ([push/broadcast.md](push/broadcast.md)); every headline number checked against
  `bench/results/lab_broadcast/report.json` (19 settings), both reproductions (4.8e-6) and `port.json` (M and M2
  identical to the bench through the files, both fallbacks unchanged). On v16's B, no retrain: **M** (registered)
  central +0.0081 (8/10, every variant and both sources up), pessimistic −0.0018 and −0.0010 on a second draw: passes,
  robustness by 0.0002; **M2** (M + at most 11 a team, post hoc) +0.0076 (8/10), pessimistic −0.0005 / +0.0007;
  optimistic +0.0087 / +0.0091. The retrain on merged data fails (+0.0026, 5/10) and makes B depend on the merge (−0.0098
  with no broadcast). `possession/broadcast_merge.py` written, never run on our match; nothing wired into v16. The
  page's suggestion that Jose check the broadcast video for tracked referees is **not** taken up: choosing a rule from
  the provided video comes close to tuning on it, and M2 is safe on both pessimistic draws without that look.
- 19:30 **Jose: "Yes, with M2"**: v17 = v16 + the broadcast merge with rule M2 for branch B only
  ([v17](../models/v17-broadcast.md)); being built and checked.
- 19:45 **a slip, recorded:** recalling how v16's clean copy was checked, the coordinator printed its old validator
  log through a filter that let two summary lines through (v16's X share and possession count on our match). Nothing
  was or will be decided on them; from now on only the validator's pass/fail line is shown.
- 20:16 track 2: D3's screen in (against `ens2`: bench +0.0029, Metrica's 3 +0.0149, DFL's 7 −0.0023); the
  combination's rule fires (W4 and D3 both beat `ens2` on Metrica's 3): **W4D3** trains, seeds 0 and 1.
- 21:30 **v17 built and checked; the upload** ([v17](../models/v17-broadcast.md)): `possession/v17.py` (the switch on
  the delivered tracking; if B, the broadcast merged with M2, then v12.inputs and v16's B; any failure gives v16's file).
  `bench/check_v17.py` passes on J03WR9 normal and stressed: v16's letters without a broadcast or with a file missing,
  the tracking B reads equal to the bench's M2 merge at the shifts found (0.0 m). On the stressed ball the alignment
  lands 1 frame off the true shift, inside the ±2-frame residual the central simulation scored; nothing changed for it.
  Our match: `d7b159c5…ec04` three times and in a clean copy (tactical + broadcast files, no video); the validator
  passes. If track 2's finalist replaces v16's network, the merge's no-retrain runs (`off`, `central_M2`,
  `pessimistic_M2`) are repeated on it before the swap.
- 21:57-01:18 the machine stalled (macOS swap 9.8 of 10 GB, mostly desktop apps); track 2's folds slowed about 70×.
  01:20 track 2 dropped W4D3 by the clock, unscored; Jose closed the largest app at 01:21 and speed returned.
- 03:40 **track 2 done** ([push/netweight.md](push/netweight.md)); its numbers checked against
  `bench/results/lab_netweight/W4_ens4/` (`bar.json`, `compare_netmet5.json`, `judge.json`, `recipe_check.json`) and
  `W4_s23/compare_netmet_s23.json`. The finalist **W4** (Metrica's training matches at loss weight 4; seeds 0-3
  averaged) passes the bar written at 10:28 on both parts: (a) against v15's B, Metrica's 3 +0.0525 (3/3), DFL's 7
  +0.0010, worst variant +0.0152, SkillCorner +0.0028; (b) against `netmet5`, Metrica's 3 +0.0125, DFL's 7 −0.0017
  (limit −0.003), B's bench +0.0026. Out of selection (seeds 2+3 against netmet's 2+3): Metrica's 3 +0.0109 (3/3),
  DFL's 7 −0.0038, B's bench +0.0006. Nothing adopted.
- 03:44 the merge's no-retrain runs repeated on W4 before Jose is asked: `off_w4`, `central_M2_w4`,
  `pessimistic_M2_w4` (`bench/lab_broadcast.py` given a `_w4` tag: W4's four fold nets per match and its trial-1015
  tables, otherwise as `_v16`; `off_w4` must reproduce W4's `confirm_1015.csv` to 1e-5).
- 04:15 **the merge on W4** (`bench/results/lab_broadcast/report.json`, `w4_vs_v16.json`; the report's earlier rows
  unchanged): `off_w4` reproduces W4's B (30 units, max 4.97e-6). Judged by the bar track 3 wrote for v16 + merge
  (its parts against `off_w4`): `central_M2_w4` **+0.0051, 8/10**, every variant and both sources up (DFL +0.0024,
  Metrica +0.0113): parts 1-3 pass; **`pessimistic_M2_w4` −0.0032, 4/10 (DFL −0.0079, Metrica +0.0078): part 4 fails**
  (floor −0.002; v16's B: −0.0005). Head to head with the merge on, W4 against v16's network: **B's bench level**
  (central −0.0000, pessimistic −0.0002), Metrica's 3 +0.0119 / +0.0183 (3/3 both), DFL's 7 −0.0051 / −0.0081 (bar
  (b)'s DFL limit is 0.003). W4's gain and the merge's overlap: on the bench they don't add. By the bars written before
  these scores, v17 stays as it is; the choice goes to Jose.
- 08:45 **Jose: "Keep v17"** (asked "Keep v17 as the upload, without W4?"): v17 stays the upload, with `netmet5` as
  B's network; no v18 is built. W4's nets and results stay where track 2 left them, for the record. The push's model
  work ends here: v17's final run, clean copy and validator are done (21:30); left are the Linux byte-for-byte check
  and the technical document (11-17).
