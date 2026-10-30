# Every data file, what it says, and whether we use it

Every number on this page comes from `python tools/explore_data.py` (about 15 s). Re-run it if
anything here looks wrong. "Graded" means the 158,857 frames that are scored (see
[challenge.md](challenge.md)).

Where the files live (unpacked from the organisers' Drive, never committed):

```
data/
├─ tactical/     the reference camera: row i of our submission is frame i of this video
│  ├─ BAR-ATH_tactical_atd.xml               1. tracking metadata
│  ├─ BAR-ATH_tactical_atd.txt               2. tracking
│  ├─ BAR-ATH_tactical_homography.csv        3. pitch-to-picture mapping
│  ├─ BAR-ATH_tactical_smart_tagging.xml     4. automatic phase tags
│  └─ BAR-ATH_tactical.mp4                   5. video
└─ broadcast/    the TV feed, NOT aligned in time with the tactical video
   ├─ BAR-ATH_broadcast_atd.xml              6.
   ├─ BAR-ATH_broadcast_atd.txt              7.
   ├─ BAR-ATH_broadcast_homography.csv       8.
   └─ BAR-ATH_broadcast.mp4                  9.
evaluation/gt/   (in the repo) the real answers for the first 5 minutes   10. + 11.
```

## Summary

| # | file | type | use in the first model? | why |
|---|---|---|---|---|
| 1 | tactical tracking metadata | XML | **yes** | the only source of which team each track slot belongs to and which way each team attacks |
| 2 | tactical tracking | text | **yes** | the main input: where every player and the ball are, every frame |
| 3 | tactical homography | CSV | no | only needed if we work with the video pixels |
| 4 | Smart Tagging | XML | **yes, as hints** | who is attacking, roughly where restarts are |
| 5 | tactical video | MP4 | for debugging | watching; later maybe our own ball detection or whistle detection |
| 6–9 | broadcast set | XML/text/CSV/MP4 | not yet | sees the ball more often, but has to be lined up in time first |
| 10–11 | public ground truth | CSV/JSON | checking only | rules forbid tuning on it |

---

## 1. `BAR-ATH_tactical_atd.xml`: the legend for the tracking

**Type:** XML in FIFA's EPTS format, 58 KB.

**What it says:**
- 25 frames per second, pitch 105 × 68 m (the repo measured 105.3 × 68.0 from corrected tracking).
- Two teams in this order: Barcelona (home), then "Atletic Club" (away, misspelled in the file).
- Barcelona attacks **left to right** in the first half, Athletic right to left.
- 46 "players", named `Track 1` … `Track 46`, with `ShirtNumber` = the slot number. These are
  **track slots, not people**: slots 1–24 are Barcelona, 25–46 Athletic.
- Each slot has three channels: x, y (0–1, "normalized") and speed. The ball has the same three.
- The format specification: each line is `frame:players:ball`, groups separated by `;`, values by `,`.

**What it does not have:** names, shirt numbers, positions, who has the ball, or whether the ball is
in play.

**Watch out:** it says speed is in km/h (`kph`), but the values are metres per second (file 2).

**How we use it:** to map every slot to its team, and to know which goal each team attacks (needed
to tell "pushing forward" from "falling back").

## 2. `BAR-ATH_tactical_atd.txt`: where everything is, every frame

**Type:** plain text, 187 MB, 199,319 lines, one per video frame.

**Format:** `frame:x,y,s;x,y,s;…(46 groups)…:x,y,s`, where the last group is the ball. Missing values
are `NaN`. Line *N* (frame id *N*) is video frame *N − 1*. The frame ids run 1…199,319 with no gaps.

**What is in it:**

| what | measured |
|---|---|
| players visible per graded frame | mean 20.5 (Barcelona 10.2, Athletic 10.3), min 1, max 25 |
| frames with fewer than 8 visible from a team | 6.8 % Barcelona, 5.7 % Athletic |
| frames with more than 11 from a team (someone counted twice) | 8.1 % Barcelona, 4.7 % Athletic |
| player speed values | median 1.34, p99 6.38, max 11.29: **metres per second** (matches speed computed from positions: median ratio 0.99, correlation 0.90) |
| mean speed of all visible players, per graded frame | percentiles 5/25/50/75/95 = 0.55 / 1.02 / 1.52 / 2.32 / 3.68 m/s: one bump, no natural split between "stopped" and "playing" |
| player speed empty while the position exists | 9.8 % |
| player positions | slightly outside 0–1 sometimes (players off the pitch near the lines) |
| track fragmentation | 2,218 tracklets, mean 68 s each; each slot is reused ~52 times (max 88) |
| ball visible | 56.9 % of all frames, **59.3 % of graded frames** |
| ball gaps in graded frames | **150 gaps, median 10.7 s**, 90th percentile 33.8 s, longest 221.5 s (3.7 min); gaps over 2 s hold 99.9 % of the missing frames |
| ball speed column | filled on only 8.7 % of ball frames, values median 1.2, max 7.3 m/s: **wrong, not usable** |
| ball speed computed from positions | median 5.5, p90 17.2, p99 50.3, max 134 m/s (the top values are glitches when a lost ball is re-found) |
| ball smoothness | frame-to-frame wobble median 1.1 cm: the track has been smoothed |
| ball outside the pitch lines | 4.4 % of ball frames |

**What that means:**
- **Slots are not people.** A slot is a tracklet: an unbroken stretch where the software followed
  someone. Anything we build must be about "a Barcelona player" or "the team", never "player 7".
- **The ball disappears in long blocks, not in flickers.** Joining the dots across a gap fixes
  almost nothing. The gaps are the main difficulty of the challenge.
- **We compute ball speed ourselves** from consecutive positions, and throw away impossible jumps.
- **The ball outside the lines is a free "dead ball" signal.**

**How we use it:** it is the main input. Distance from the ball to the nearest player of each team,
ball speed and direction, ball outside the lines, how fast everyone is moving, and each team's shape
and direction of movement.

## 3. `BAR-ATH_tactical_homography.csv`: pitch-to-picture mapping

**Type:** CSV, 22 MB, one row per video frame. Columns: `frame` (0-based, already a submission
row), `h00 … h22` (a 3×3 matrix, row by row), `reprojection_error` (pixels; lower is better),
`valid` (1 if a matrix exists).

**What it says:** for each frame, where a pitch point (x, y in 0–1, same as the tracking) appears in
the 1920×1080 picture. Valid on 100 % of frames, typical error 0.24 px (99th percentile 4.8, max
5.5). At the first graded frame the far corners land at pixels (44, 256) and (1903, 262), the near
corners at (−987, 898) and (2987, 940), and the centre spot at (963, 469). So the tactical camera is a
wide shot of the whole pitch, and the two nearest corners fall just outside the picture.

**How we could use it:** only if we work with the video. For example, detect the ball ourselves in
pixels and convert it to pitch position, or draw the tracking on top of the video to debug.

## 4. `BAR-ATH_tactical_smart_tagging.xml`: automatic phase tags

**Type:** XML in SportsCode format, 255 KB. Two sections: `ALL_INSTANCES` (the tags) and `ROWS`
(only display colours, ignore).

**Format of one tag:** `start` and `end` in seconds from the first video frame (× 25 = video frame),
a `code`, and labels. The labels are `Team` (Barcelona / Atletic Club / N/A), `Half`, and depending
on the code `Type`, `Side`, `Direction of ball entry` or `Max Players in the box` (a string, `7+`
possible).

**638 tags:**

| code | n | length (median, range) |
|---|---|---|
| PLAYERS IN THE BOX | 97 | 27.7 s (17.1–83.7) |
| BALL IN FINAL THIRD | 93 | 27.0 s (20.6–70.8) |
| BALL IN THE BOX | 72 | 17.4 s (15.2–32.5) |
| PROGRESSION | 66 | 17.6 s (7.7–37.1) |
| DEFENDING IN MIDDLE THIRD | 66 | 17.6 s (7.7–37.1) |
| SET PIECES | 59 | 35.0 s (5.0–35.0) |
| CREATING CHANCES | 30 | always 30.0 s |
| DEFENDING IN DEFENSIVE THIRD | 30 | always 30.0 s |
| LONG BALL | 29 | 12.4 s (8.1–22.2) |
| ATTACKING TRANSITION | 24 | 15.4 s (9.0–22.0) |
| DEFENSIVE TRANSITION | 24 | 15.4 s (9.0–22.0) |
| BUILD UP | 22 | 20.5 s (11.7–28.6) |
| DEFENDING IN ATTACKING THIRD | 22 | 20.5 s (11.7–28.6) |
| GOALS | 4 | 169.3 s (70.9–250.7) |

Team labels: Barcelona 314, Athletic 288, N/A 36. Set-piece types: throw-in 26, corner 12, goal kick
10, free kick 5, kick-off after a goal 4, kick-off at the start of a half 2.

**Three things DATA.md does not say:**

1. **The tags come in mirrored pairs.** Each "with the ball" tag has a twin with exactly the same
   start and end and the **opposite team**:
   - PROGRESSION ↔ DEFENDING IN MIDDLE THIRD (66 pairs)
   - CREATING CHANCES ↔ DEFENDING IN DEFENSIVE THIRD (28 pairs, plus 2 pairs that carry the *same*
     team, which must be an error)
   - ATTACKING TRANSITION ↔ DEFENSIVE TRANSITION (24 pairs)
   - BUILD UP ↔ DEFENDING IN ATTACKING THIRD (22 pairs)

   So "Barcelona, Defending in middle third" means **Athletic** has the ball. That is also why 42.2 %
   of graded frames sit under a Barcelona tag and an Athletic tag at the same time. (Any tag at all
   covers 82.3 %; a Barcelona tag 61.6 %; an Athletic tag 58.3 %.)
2. **The windows are fixed lengths.** Set pieces are exactly 5 s (kick-offs) or exactly 35 s
   (everything else), and Creating Chances is always 30 s. The start is roughly meaningful; the end
   is a template.
3. **Set-piece tags are imprecise and incomplete.** In the public 5 minutes there are 5 restarts
   (first live frame after a dead ball, plus the kick-off):

   | restart frame | nearest set-piece tag (within 15 s) |
   |---|---|
   | 11185 | Kick Off (Start), team N/A, starts **3.5 s before** |
   | 12989 | **none** |
   | 16400 | **none** |
   | 17838 | Free Kick, team Athletic, starts **12.2 s before** |
   | 18389 | Goal Kick, team Barcelona, starts **2.8 s before** |

   So a set-piece tag starts somewhere 3–12 s before its restart, and 2 of 5 restarts have no tag.
   Every throw-in has team `N/A`.

**How we use it:** as hints, never as answers. Attacking tags point to their team and defending tags
to the *opposite* team, which helps most while the ball is missing. A set-piece tag says "a restart
is coming within roughly the next 15 s", and its type and team say who restarts.

## 5. `BAR-ATH_tactical.mp4`: the tactical video

**Type:** H.264 video, 1920×1080, 25 fps, 199,319 frames, 7,973 s (2.2 h), 5.2 GB, **with an audio
track** (AAC).

**What it shows:** a fixed wide view of the whole pitch from the main stand, with the match clock
burned in at the top left and a "Media Coach" logo at the top right. Athletic plays in white,
Barcelona in blue and red.

**The audio is quiet but clean, and the referee's whistle is in it.** Loudness at five points across
the match: mean −56 to −63 dB, peaks around −44 dB (normal audio sits around −15 to −30 dB). So it
sounds almost silent unless turned up, but a whistle stands out as a narrow 2.5–4.5 kHz tone.
`python tools/find_whistles.py 11000 7800` on the first 5 minutes finds:

| tone at frame | what the public truth shows there |
|---|---|
| 11149 (3.9 kHz) | 1.4 s before the opening kick-off (11185) |
| 15835, 15881, 15887 | the truth turns dead at **15889**: a stoppage (1st half 03:06–03:08) |
| 16081 | inside that dead stretch |
| 17113, 17450 | 17113 is in open play: probably the crowd whistling. 17450 is inside a dead stretch |
| none | at 12851 and 18027 the ball went out. Nobody whistles for that, so no tone is expected |

So the whistle marks fouls, kick-offs and similar stops, not balls going out, and crowd whistles
cause false alarms. The thresholds in the script are eyeballed, not fitted.

**How we could use it:**
- Watching clips to understand what the model got wrong (see [viewer.md](viewer.md)).
- **A whistle clue for dead balls**, straight on the tactical timeline with no alignment needed.
- Later, maybe: our own ball detector for the long gaps.

## 6–9. The broadcast set

**Same formats as files 1, 2, 3 and 5.** Differences, measured:

| what | broadcast | tactical, for comparison |
|---|---|---|
| frames | 164,488 tracking lines; the video has 164,487 frames (an off-by-one in the source; the file is complete) | 199,319 |
| video | 1080p, 25 fps, 6,579 s, 14.0 GB, with audio; TV graphics, scoreboard clock top left | — |
| slots | 51 (Barcelona 26, Athletic 25) | 46 |
| players visible per frame | mean 11.5 | 20.5 |
| tracklets | 4,641, mean 16 s | 2,218, mean 68 s |
| ball visible | 50.5 % of all frames, **76.4 % of frames where the camera shows the pitch** | 59.3 % of graded |
| ball gaps | 194, median 9.2 s, longest 160.2 s | 150, median 10.7 s |
| homography valid | 65.8 % (the rest are replays, close-ups, crowd shots), error up to 303.8 | 100 % |

**What that means:** the TV camera zooms in on the action, so when it shows the pitch it sees the
ball more often than the tactical camera, but fewer players. A third of its frames are replays or
close-ups with almost nothing tracked.

**Lining it up with the tactical video turned out to be easy** (`python tools/align_broadcast.py`).
Both trackings give the ball in the same pitch coordinates, so the script slides one ball track
along the other and keeps the shift where the two balls sit on top of each other:

| | 1st half | 2nd half |
|---|---|---|
| broadcast frame = | tactical frame − **4,014** | tactical frame − **26,919** |
| next-best shift is worse by | 5.63 m vs 36.9 m rms ball distance | 7.39 m vs 37.8 m |
| shift holds in every 2-minute window | within ±2 frames in 100 % of 24 windows | in 96 % of 26 (one window off by 40 frames) |
| ball distance at the shift (median) | 0.80 m | 0.74 m |

The two halves need different shifts because the broadcast **cuts out half-time**: its first half
ends 7 s before its second-half kick-off. Live play has no other jumps. Replays are cut in *on top
of* live play, so the shift stays the same before and after them.

**But it adds little for the ball.** When the tactical tracking has lost the ball (41 % of graded
frames), the broadcast sees it on only **12–13 %** of those frames, and that's before removing
replays. Both cameras lose the ball at the same moments. Its audio is normal TV loudness (mean −24 dB,
probably commentary and crowd; not listened to), and the tactical audio already has the whistles.

**How we could use it:** now cheap to use, so it's an option for the ball gaps or a second view of
the players. It's a small gain, not the main lever.

## 10–11. The public ground truth (in the repo)

`evaluation/gt/BAR-ATH_possession_gt_public.csv` has 199,319 rows like a submission. Only frames
11,185–18,684 (the first 5 minutes of play) carry real labels: H 3,454, A 2,577, D 1,469. That is 21
segments and 17 possessions, mean 14.2 s, shortest **0.8 s**, longest 42.4 s. `…_meta.json` repeats
these numbers.

How the ball relates to the true labels in that window: it is visible on 99.9 % of Barcelona-possession
frames, 85.0 % of Athletic, and 66.4 % of dead-ball frames: **88.2 %** of the window overall, against
59.3 % over the whole match. These 5 minutes are much easier than the rest, so they say little about
the hard part (the long ball gaps).

**Use:** checking only, that frames are aligned, teams are the right way round, and the file is
well-formed. Never to choose a setting (rules, see [challenge.md](challenge.md)).
