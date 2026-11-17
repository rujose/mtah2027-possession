# Model v17: v16 with the broadcast's players merged into what branch B reads (rule M2)

| | |
|---|---|
| **status** | **built and checked 2026-11-15 21:30; the upload** (Jose, 19:30: "Yes, with M2", asked as "Add the broadcast merge (M2) to the upload as v17?"). The glue check passes on a bench match on both a normal and a stressed variant; on our match three runs give one file and a clean copy gives it too; the format check passes. Track 2's finalist W4 was run under the merge on 2026-11-16 (04:15): central +0.0051, pessimistic −0.0032 (track 3's floor is −0.002); with the merge on, W4 and v16's network are level on B's bench (Metrica's 3 +0.012, DFL's 7 −0.005); Jose kept v17 as it is (08:45: "Keep v17"; [push.md](../lab/push.md)) |
| **what** | v16 ([v16](v16-netmet.md)) with one change in branch B's input: when the switch takes B, the broadcast set (files 6-8) is lined up with the tactical tracking by the ball and **its players that the tactical tracking lacks are added** (rule M2: [push/broadcast.md](../lab/push/broadcast.md)); v8's repair and every input B reads are computed from the merged tracking; v16's B then runs on it, **its models unchanged**. The switch, branch A, the kick-off rule and every model file are v16's. If anything about the broadcast fails, B reads the delivered tracking and the file is v16's |
| **why** | Round 9's largest remaining value was the players missing near the ball (+0.022 ceiling, [cleaning.md](../lab/cleaning.md)); the tactical tracking can't supply them and the broadcast camera follows the ball. Track 3 of [the push](../lab/push.md) simulated our match's broadcast on the bench from docs/data.md's statistics only and judged the merge on v16's B |
| **bench** (B's bench, v16's B as trained, nothing retrained; 25 %, 50 %, stressed) | realistic simulation (`central`): **0.7544 against v16's 0.7468 (+0.0076), better on 8/10**, every variant and both sources up. Pessimistic simulation: −0.0005 (6/10) and +0.0007 on a second draw. Optimistic: +0.0091 (9/10) |
| **SkillCorner** | no broadcast there: v17 is v16 (0.4090) |
| **our match** | SHA-256 `d7b159c5bc392abd0240f5d2d16a893c26694b5d7ab3cb8bb5dafa4dec69ec04` (three runs agree: 2 threads twice, 1 once; clean copy the same; format check passes). It differs from v16's `90599088…0523c`, so the broadcast was merged (a fallback writes v16's file exactly) |
| **new code** | `possession/v17.py`, `possession/broadcast_merge.py` (track 3's module), `bench/check_v17.py`; the lab: `bench/lab_broadcast.py`. No new model file |
| **run** | `python -m bench.check_v17` (~15 min, a bench match only); `python -m bench.lab_broadcast port` (the module against the bench merge); then `python -m possession.v17` on our match: path, SHA-256 and format check only (`--show-branch` for Jose: the branch and whether the broadcast was merged) |

## What changes, and what doesn't

| | v16 | v17 |
|---|---|---|
| the switch | reads the delivered tracking | the same (it never sees the broadcast, so it takes v16's branch) |
| branch A | v14's | unchanged |
| branch B's input | the delivered tracking: v8's repair, v4/v6's P, v7, PathCRF, v9, v11, robust's features, the GRUs' features | **the delivered tracking + the broadcast's extra players (M2)**, then the same inputs computed from it |
| branch B's models | `v16_b_all.joblib` | the same file (no retraining: a B retrained on merged data was worse, and worse still without a broadcast) |
| when the broadcast can't be used | | a file missing, the alignment not trusted (rms ball distance above 12 m, or the next-best shift not 3× worse) or any error: **v16's file, byte for byte** |

## The merge (possession/broadcast_merge.py, track 3's; checked against the bench's merge)

1. **Line up** each graded half by the ball (tools/align_broadcast.py's method: one shift per half, where the two
   balls sit on top of each other). docs/data.md recorded this for our match on 2026-10-30: shifts 4,014 and 26,919
   frames, 5.63 / 7.39 m against 36.9 / 37.8 m for the next best.
2. **Team order** by position: broadcast players within 1.5 m of a tactical one vote for the same or the other label.
3. **Register** each valid broadcast frame on the tactical players (median vector to each broadcast player's nearest
   tactical player within 5 m, if at least 3), and drop the frame if the median distance left is above 1.5 m.
4. **Add** a broadcast player with no tactical player of his team within 2.5 m and none of the other team within
   1.5 m; **M2: a team never has more than 11** (the extra ones nearest a tactical teammate are dropped first).
5. **Pieces:** each run of added frames goes into a free slot of its team (the one free longest), a new slot only if
   none is free.

Rule M (steps 1-4 without the cap) was fixed before any score; the cap was added after the pessimistic run showed
false people and duplicates, so M2 carries that caveat. Its only change is a limit a real match can't break.

## The bench (track 3, [push/broadcast.md](../lab/push/broadcast.md); v16's B as trained, against `off_v16` 0.7468)

| simulated broadcast | M | **M2 (v17)** |
|---|---|---|
| central (data.md's numbers) | +0.0081, 8/10 | **+0.0076, 8/10** (p25 / p50 / stressed +0.0061 / +0.0080 / +0.0088; DFL / Metrica +0.0058 / +0.0119) |
| pessimistic | −0.0018, 4/10 | **−0.0005, 6/10** |
| pessimistic, second draw | −0.0010, 3/10 | **+0.0007, 4/10** |
| optimistic | +0.0087, 9/10 | **+0.0091, 9/10** |

- **The gain lands near the ball:** frame accuracy within 15 m of an added player +1.3 points (central).
- **The risk is people who aren't players:** a referee-like track given a team cost 0.0053 in the pessimistic view;
  M2's cap halves it. The real broadcast's behaviour here is unknown (data.md lists 51 slots, all with a team).
- **The simulation uses docs/data.md's statistics only** (players per frame, tracklet length, valid share, ball
  distance at the alignment); nothing of our match's broadcast tracking was read for it.

## Checks

**The merge module against the bench** (`python -m bench.lab_broadcast port`, track 3): `broadcast_merge.merge()` is
the bench's merge bit for bit for M and M2; through files in data.md's formats, both shifts are found exactly on the
normal ball and 0 slot-frames differ from the bench's merge (positions within 1.5e-5 m, the text round trip), with
either team order; a missing file and an unrelated ball fall back unchanged.

**v17's glue** (`python -m bench.check_v17`, `bench/results/check_v17/glue_J03WR9_{normal,stressed}.json`): J03WR9
from b16_c0's lab cache with the central simulated broadcast written as files 6-8 on its own timeline (shifted 500
and 2,500 frames). Both variants **pass**:

| | normal ball (switch: A) | fully stressed ball (switch: B) |
|---|---|---|
| no broadcast / a missing file: letters against v16's | 0 / 0 frames differ (`not-used`) | 0 / 0 frames differ (`off` / `fallback-missing`) |
| B's own path with a missing file | `fallback-missing`, 0 frames differ from B on the delivered match | the same |
| the alignment (true shift −500 / −2,500) | −500 / −2,500 (rms 0.81 / 0.85 m) | **−501 / −2,501** (rms 4.90 / 6.94 m; next best 34.7 / 33.4 m) |
| the tracking B reads against the bench's M2 merge at the shifts found, repaired | identical (0 slot-frames, 0.0 m), both team orders | identical (0 slot-frames, 0.0 m), both team orders |
| predict() with the files | branch A as v16, letters v16's | branch B as v16, status `merged`, letters B's on the merged match |
| the delivered tracking afterwards | unchanged | unchanged |
| B's letters, merged against delivered (information; the deployed models saw every bench match) | 0.91 % of frames | 0.77 % of frames |

**One finding, and why it changes nothing.** On the stressed ball (the ball moving through stoppages, as on our
match: data.md's rms at the shift is 5.63 / 7.39 m, like this run's 4.90 / 6.94 m and unlike the normal ball's 0.8 m)
the alignment lands **one frame (40 ms) off** the true shift in each half. Against the simulated arrays merged with no
alignment error, that reshuffles which free slot each piece takes (74,928 slot-frames differ, the same pieces in
other slots). The central simulation that gave +0.0076 already drew a **±2-frame** alignment residual in every
2-minute window (the pessimistic one also 40-frame windows), so a one-frame error is inside what the bench scored. The
module is unchanged; the check's pass rule allows the found shift within 2 frames of the true one and compares the
glue at the shifts found. (The first version of the check compared against the arrays with no alignment error and
expected a B-path status on a match that took A: both were the check's mistakes; its first outputs are kept as
`glue_J03WR9_*_first.json`.)

**Our match:** `python -m possession.v17` three times (2 threads twice, 1 once): `d7b159c5…ec04` each time. A clean
copy (fresh clone of 3cce485 + the working tree's changes + `possession/v16.py`, `v17.py`, `broadcast_merge.py`,
`models/v16_b_all.joblib`; in `data/` only the organisers' tactical and broadcast files without video, and PathCRF's
weights) gives the same; `validate_submission.py` passes (pass/fail only). Nothing else of our match was looked at:
not the merge's status, not the shifts found, not the branch.

## What it can't tell us

- **How good our broadcast tracking is.** The bench's broadcast is simulated from data.md's numbers; whether the real
  one tracks referees with a team, how far its players sit from the tactical ones and how often its team labels err
  near the ball are unknown. The pessimistic runs say that with a much worse broadcast M2 is about even.
- **Whether the merge helps on our match.** By the rules it can't be scored; the bench says +0.0076 on B's bench, small
  next to the match-to-match spread (±0.03).
