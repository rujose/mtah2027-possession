# Lab round 7, `skillcorner`: v12 on 20 matches it has never seen

| | |
|---|---|
| **status** | **a test, not a candidate** (2026-11-09). Nothing in v12 changes; nothing was tuned on SkillCorner |
| **what** | SkillCorner's open data: 20 A-League 2024/25 matches of **broadcast** tracking (10 fps), loaded like our tracking and labelled from SkillCorner's own fields. v12 run on each exactly as on our match, `v12.predict(tracking, tags=None, halves, pitch)` with the deployment models; v10 and v11 beside it |
| **why** | v12 was built and judged on 10 bench matches whose damage is simulated. This is real messy tracking from a source that no model, setting or design choice has seen |
| **answer** | **v12 scores 0.308 (0.254–0.383) on SkillCorner, against 0.704 out of fold on the bench, and below v11 (0.366) on 19 of 20 matches.** Two causes, both measured: (1) **stoppages where the ball is still tracked**, moving (carried, thrown in): robust calls 81 % of those frames live; with the truth's stoppages v12 would score 0.473. (2) **The team, and when it changes**: 83 % right on live frames against 92 % on the bench, with changes spread ±1.2 s around the truth's against ±0.1 s; with the truth's team v12 would score 0.720, the bench's level. The cut-aways and the possession count (SkillCorner's labels change hands at Metrica's rate) don't explain it |
| **code** | `bench/skillcorner.py` (download, loader, labels, label checks), `bench/lab_skillcorner.py` (runs v12 / v10 / v11, scores, diagnoses) |
| **run** | `python -m bench.skillcorner download` (1.83 GB, each tracking file SHA-256-checked against its Git LFS pointer) → `python -m bench.skillcorner parse` (~5 s a match) → `python -m bench.lab_skillcorner run` (2–5 min a match, ~0.7 GB) → `python -m bench.lab_skillcorner report` (~3 min) |
| **results** | `bench/results/lab_skillcorner.json` (summary), `lab_skillcorner.csv` (every match × every version, oracle and baseline), `lab_skillcorner_labels.json` (label checks per match), `lab_skillcorner_bench_labels.json` (the same checks on the bench's truth); letters and robust's P in `data/external/skillcorner/predictions/` (gitignored) |
| **data** | **Tracking and labels: SkillCorner**, [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT licence. Thanks to SkillCorner for publishing it |

## The data

- **20 matches** (`matches.json`; the repo's README still says 10). For each we took only the match file
  (pitch, players, periods), the phases of play and the tracking: **1.83 GB**. Dynamic events, aggregates
  and body pose weren't needed.
- **What broadcast tracking looks like** (all 20, frames inside the periods):
  - **26 % of frames (19–34 %) have no tracking at all**: no player, no ball. These are the broadcast's
    cut-aways (replays, close-ups); 96 % of them fall in dead time.
  - In the tracked frames the ball **always** has a position: SkillCorner fills it in when it isn't
    detected (detected on 60 % of all frames, ~80 % of tracked ones). So "ball missing" here means "no
    tracking at all": 26 % of frames, against 41 % missing in the bench's simulated damage (and in ours).
  - Players are on screen 57 % of the time; the rest are extrapolated by SkillCorner.
  - **Positions are coarse at the ball:** the player SkillCorner itself names in possession is the nearest
    to its ball 97.5 % of the time, but within 1 m of it only 47 % (median 1.04 m; one match read).

## The loader (`bench/skillcorner.py`), fixed before anything was scored

| choice | what | why |
|---|---|---|
| origin, orientation | centre origin → our corner origin (+ half the pitch on each axis), no flip; pitch size from the match file | what `bench/idsse.py` does with DFL's centre origin |
| 25 fps | each 25 fps frame placed at its time on the 10 fps clock; positions linear between the two 10 fps frames around it, only where both have one | our code counts in 25 fps frames |
| the ball | **missing only where SkillCorner has none** (the cut-aways); kept where SkillCorner extrapolated it | the brief: it's SkillCorner's tracking |
| players in a cut-away | linear between the tracked frames around it (held at a period's ends); a player absent from a tracked frame (not yet on, or off) stays absent | our tracking always has the players; a quarter of frames with nobody on the pitch would test something our match never shows |
| speed | from positions over 0.16 s | as `bench/metrica.py` |
| halves | the two periods end to end, one empty frame between | as the bench's caches |
| **H / A** | each frame's `possession.group`, **read raw** ("home team", "away team", null; kloppy would read null as dead) | the brief |
| **D** | wherever **no phase of play** covers the frame (phases "are only defined when the ball is in play") | the brief |
| a null team in live time | carries the last team of that live stretch; a stretch that starts with null takes its first team; a stretch with no team at all takes its phase's team | "a loose ball stays with the last team". Nulls are 3 % of live frames, median 0.8 s; back-filled 0.6 %; from the phase 0.1 % |
| labels at 25 fps | the nearest 10 fps frame | |
| a secondary score, fixed with the rest | the truth with the cut-away frames made X | to see whether the cut-aways carry the gap |

## Are the labels sane?

The same checks on SkillCorner's labels (its tracking) and on the bench's truth (clean tracking):

| | SkillCorner (20) | IDSSE (7) | Metrica (3) |
|---|---|---|---|
| possessions per graded minute | **3.26** (2.82–3.94) | 4.58 (3.78–5.30) | 3.67 (3.38–3.98) |
| dead share | **0.464** (0.369–0.549) | 0.449 (0.407–0.517) | 0.352 (0.293–0.407) |
| median possession | 6.0 s (4.7–7.2) | 3.8 s | 5.6 s |
| possessions under 1 s | 8.5 % | 12.3 % | 6.9 % |
| reading check: the nearest player within 1 m of a slow, detected ball is of the labelled team | **0.894** (0.815–0.941) | 0.962 (0.948–0.977) | 0.985 (0.969–0.996) |
| clear control: the same with nobody of the other team within 3 m | 0.971 (0.947–0.988) | 0.988 (0.973–0.994) | 0.988 (0.966–1.0) |
| **at a true team change: the new team's nearest player to the ball** | **2.1 m** (1.5 m half a second later) | 0.6 m | 0.0 m |
| … within 1.5 m at the change | **33 %** | 97 % | 77 % |

- **The rate and the dead share are sane.** SkillCorner's possession changes hands at about Metrica's rate
  (and the organisers' public hint, 3.4 a minute), more slowly than IDSSE; its dead share sits inside the
  bench's.
- **Where one player clearly has the ball, the labels agree** (0.97, near the bench).
- **But its changes of team don't sit on a touch in its own tracking.** On the bench the new team is at the
  ball when the label changes (DFL: 0.6 m, 97 %); on SkillCorner it's 2.1 m away, and closest half a second
  later. Either SkillCorner's possession changes on its own timing (not the first touch), or its positions
  are ~1–2 m off at the ball (its own possessor is 1.04 m from its ball, median), or both. The video would
  tell; we don't have it. For a model that finds changes at touches, the truth's change moments are blurred.

## v12 on SkillCorner

| | SkillCorner (20) | bench (10, out of fold) |
|---|---|---|
| **spellF1** | **0.308** (0.254–0.383) | **0.704** |
| F1 at IoU 0.3 / 0.7 | 0.458 / 0.159 | 0.790 / 0.598 |
| precision / recall (mean over IoUs) | 0.286 / 0.335 | 0.768 / 0.652 |
| possessions predicted / true | 378 / 321 | 344 / 411 |
| macroF1 | 0.719 | 0.918 |
| frames right | 72.5 % | 92.6 % |
| **team, where both say live** | **0.831** (ball detected 0.837, not detected 0.742) | 0.925 |
| alive / dead F1 | 0.849 | |
| share called D | **0.294** (truth 0.464) | 0.434 (truth 0.420) |
| truth with the cut-aways ungraded (fixed in advance) | spellF1 0.320 | |

| match | teams | spellF1 | macroF1 | precision | recall | predicted | true | team, live | alive/dead F1 | v11 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2017461 | Melbourne V FC - Auckland FC | 0.254 | 0.682 | 0.214 | 0.314 | 452 | 308 | 0.761 | 0.842 | 0.319 |
| 1996436 | Auckland FC - Macarthur FC | 0.257 | 0.689 | 0.224 | 0.303 | 415 | 306 | 0.799 | 0.813 | 0.339 |
| 2010085 | Melbourne City - CC Mariners | 0.283 | 0.714 | 0.257 | 0.314 | 348 | 285 | 0.839 | 0.851 | 0.357 |
| 2007448 | Perth Glory - Brisbane FC | 0.285 | 0.707 | 0.253 | 0.327 | 371 | 287 | 0.847 | 0.822 | 0.367 |
| 2015213 | Western United - Auckland FC | 0.286 | 0.646 | 0.251 | 0.332 | 450 | 340 | 0.817 | 0.795 | 0.381 |
| 1953632 | CC Mariners - Melbourne City | 0.291 | 0.739 | 0.282 | 0.300 | 321 | 302 | 0.860 | 0.865 | 0.328 |
| 2011166 | Wellington P FC - Melbourne V FC | 0.293 | 0.719 | 0.259 | 0.338 | 408 | 313 | 0.795 | 0.845 | 0.333 |
| 1959846 | Melbourne V FC - Western United | 0.296 | 0.720 | 0.295 | 0.297 | 364 | 362 | 0.834 | 0.840 | 0.337 |
| 2006229 | Melbourne City - Macarthur FC | 0.299 | 0.730 | 0.304 | 0.294 | 318 | 329 | 0.856 | 0.854 | 0.386 |
| 2013725 | Western United - Sydney FC | 0.304 | 0.670 | 0.284 | 0.327 | 429 | 373 | 0.795 | 0.830 | 0.358 |
| 1886347 | Auckland FC - Newcastle | 0.310 | 0.737 | 0.293 | 0.330 | 361 | 321 | 0.837 | 0.862 | 0.365 |
| 1996435 | Sydney FC - Adelaide United | 0.311 | 0.728 | 0.306 | 0.317 | 368 | 355 | 0.839 | 0.859 | 0.377 |
| 1986691 | Western Sydney - Auckland FC | 0.316 | 0.729 | 0.280 | 0.361 | 379 | 294 | 0.840 | 0.848 | 0.400 |
| 1899585 | Auckland FC - Wellington P FC | 0.319 | 0.731 | 0.288 | 0.356 | 378 | 306 | 0.816 | 0.851 | 0.357 |
| 1925299 | Brisbane FC - Perth Glory | 0.320 | 0.715 | 0.289 | 0.358 | 405 | 327 | 0.854 | 0.855 | 0.383 |
| 2006363 | Melbourne V FC - CC Mariners | 0.324 | 0.738 | 0.312 | 0.336 | 367 | 340 | 0.846 | 0.862 | 0.372 |
| 2007721 | Auckland FC - CC Mariners | 0.336 | 0.727 | 0.311 | 0.367 | 368 | 312 | 0.854 | 0.860 | 0.390 |
| 2016236 | Western United - Adelaide United | 0.342 | 0.718 | 0.316 | 0.372 | 406 | 345 | 0.799 | 0.864 | 0.338 |
| 1874553 | Brisbane FC - Adelaide United | 0.342 | 0.734 | 0.332 | 0.353 | 349 | 328 | 0.857 | 0.856 | 0.386 |
| 1927964 | Western Sydney - Wellington P FC | 0.383 | 0.804 | 0.370 | 0.396 | 297 | 278 | 0.874 | 0.914 | 0.439 |

By length of the true possession (all 20 pooled; the bench's in brackets): found at IoU 0.3, under 1 s 2 %
(19 %), 1–2 s 10 % (51 %), 2–5 s 33 % (72 %), 5–10 s 60 % (91 %), 10–20 s 77 % (97 %), over 20 s 91 %
(97 %); at IoU 0.7, over 20 s only 46 % (86 %). Even the long possessions are found with loose edges.

### Stage by stage

| after | SkillCorner | gain | better on | bench | bench gain |
|---|---|---|---|---|---|
| robust, frame by frame | 0.125 | | | 0.385 | |
| + decoder | 0.294 | **+0.169** | 20/20 | 0.672 | +0.287 |
| + snap and move | 0.293 | −0.001 | 4/20 | 0.675 | +0.003 |
| + short | 0.291 | −0.001 | 3/20 | 0.689 | +0.015 |
| + segments | 0.308 | **+0.017** | 20/20 | 0.699 | +0.009 |
| + timing (v12) | 0.308 | −0.001 | 8/20 | 0.704 | +0.005 |

**The decoder and segments still pay; short, snap / move and timing don't.** Those three act at touches, and
SkillCorner's touches don't line up with its changes (above). Short finds candidates (89 on one match) but
only 12 % of them are right by the truth there, and its regression is rarely 50 % sure, so it mostly does
nothing; where it acts it loses (3 better, 12 worse, 5 unchanged). Segments, which judges whole possessions,
gains on every match, more than on the bench: it merges spurious possessions, of which SkillCorner has many.

### Against v10, v11 and the organisers' baseline

| | SkillCorner | v12 better on | bench (all 10) |
|---|---|---|---|
| **v12** | **0.308** | | 0.704 out of fold |
| v11 | 0.366 | 1/20 | 0.573 |
| v10 | 0.367 | 1/20 | |
| baseline, nearest player, no smoothing | 0.074 | 20/20 | 0.090 at 41 % damage (0.100 clean) |
| baseline smoothed ±25 frames | 0.224 | 20/20 | 0.241 (0.321) |
| baseline smoothed ±125 frames | 0.212 | 20/20 | 0.212 (0.248) |

The organisers' baseline scores about what it scores on the bench at 41 % damage, so the matches aren't
uniformly harder for a method that has learned nothing. **What v12 and v11 learned is what fails, v12's more**:
v11 loses 0.21, v12 0.40.

### The label-free numbers (`bench/transfer_check.py` `stats`)

| | bench range (v12 out of fold) | v12 on SkillCorner | PASS / NEAR / WARN (as `--ours` would say) | SkillCorner's truth |
|---|---|---|---|---|
| dead share | 0.314–0.533 | 0.294 (0.237–0.386) | 5 / 11 / 4 | 0.464 |
| possessions per minute | 3.17–4.12 | 3.83 (3.02–4.64) | 14 / 3 / 3 | 3.26 |
| clear control | 0.991–0.998 | 0.986 (0.975–0.995) | 4 / 3 / 13 | |

Every match gets at least one NEAR, 15 of 20 a WARN; none passes all three. **But the failure that costs the
most, the stoppages, mostly shows as a NEAR** (v12 calls 29 % D where the truth has 46 %, and the bench's
range starts at 31 %). Clear control's WARNs say less than they look: its bench range is 0.007 wide, and
v11 gets 6 WARNs on it too from the coarse broadcast positions alone.

## What explains the gap

Two diagnostic oracles (truth used to explain, not to choose), on v12's letters:

| | spellF1 | macroF1 |
|---|---|---|
| v12 | 0.308 | 0.719 |
| **the truth's stoppages**, v12's teams (where the truth is live and v12 said D, v12's nearest team) | **0.473** | 0.884 |
| **the truth's team** wherever both say live, v12's stoppages | **0.720** | 0.816 |
| v11 with the truth's stoppages | 0.439 | 0.876 |

### 1. Stoppages with the ball still tracked: the tracking, and v12's known weak spot

- In SkillCorner's dead time the ball is tracked on **47 % of frames**, and then it's **moving 88 % of the
  time** (carried to a throw-in, a corner, a free kick) and beyond a line 31 % of the time. Live, the ball
  is beyond a line 0.6 % of the time.
- Robust gives P(dead) **0.19** on those frames (0.97 where there's no ball). **v12 calls 19 % of them D;
  v11 calls 82 %** (v4's dead-ball model reads the players, not only the ball). With no ball, both get 99 %.
- This is **failure mode 1** on v12's page and case B in [robust.md](robust.md): the bench can't show it,
  because IDSSE freezes its dead ball (robust blanks it) and Metrica blanks it, so robust learned "dead ⇔
  no ball". SkillCorner's labels aren't the cause: its dead share matches the bench's.
- It costs **0.17** (the first oracle) and it's why v12 falls below v11: with the truth's stoppages, v12's
  teams beat v11's on all 20 (0.473 against 0.439).

### 2. The team, and when it changes: SkillCorner's labels and tracking together

- Team right on 83 % of frames both call live (bench 92.5 %); errors come in short stretches (median 1.3 s),
  ~70 % of their frames within 3 s of a true change (two matches read).
- Only **56 % of true changes of team** have a v12 change to the same team within 5 s (bench 77 %), and
  those are spread **−1.56 s to +0.92 s** (quartiles) around the truth's, against −0.08 to +0.12 s on the
  bench. The median is ~0 (−0.16 s): blur, not a lag. v11's spread is the same.
- That matches the label check above: SkillCorner's changes don't sit on a touch in its own tracking (the
  new team 2.1 m from the ball against 0.6 m on IDSSE). Across matches, v12's spellF1 goes with the labels'
  reading check (r = +0.54) and its team accuracy (+0.56).
- It costs up to **0.41** (the second oracle). Not all of it is recoverable from any tracking: some is
  SkillCorner's definition of when possession changes, some its coarse positions; telling them apart needs
  the video.

### 3. Not the cut-aways, not the count

- **Cut-aways:** v12 calls 99 % of their dead frames D; only 7 % of true possessions touch one (22 of 321);
  with them ungraded the score is 0.320 against 0.308.
- **The count convention:** SkillCorner's truth changes hands at Metrica's rate, where v12 does best on the
  bench (0.787). Across SkillCorner's matches the score doesn't follow the count (r = −0.01; on the bench
  −0.82). v12 over-predicts (378 against 321) because the stoppages it calls live become possessions
  (spurious ones: 218 a match, median 4.2 s).

## What it means for our match

1. **SkillCorner isn't our feed.** Ours is tactical tracking at 25 fps; its ball is missing 83 % of the time
   players stand still ([../bench.md](../bench.md)), so most of our stoppages should look like "no ball",
   and v12 called 99 % of SkillCorner's ball-less dead frames D. This test doesn't predict our score.
2. **But it confirms v12's main risk on real data.** Wherever our ball is tracked moving through a stoppage,
   v12 will call it live, and v11 wouldn't. The label-free dead share is the tell, and SkillCorner shows how
   weak a tell it is: a 0.17 loss mostly reads NEAR. **A dead share at or below the bench range's low end on
   our match should be read as this failure, not as noise.**
3. **v12's team side holds up better than v11's** (with the stoppages fixed, 20/20). A post hoc mix, v11's
   stoppages with v12's teams (no setting), scores **0.389, above v11 on 20/20 (+0.023)**. That's a lead, not
   a candidate: it was found by looking at this test, and the bench can't judge it (no moving dead ball
   there). Fixes worth judging the usual way: a "ball beyond a line" feature for robust, robust's P(dead)
   blended with v4's, training robust on stoppages with a tracked ball.
4. **The touch-level stages are the fragile ones** (snap / move, short, timing: about −0.003 together here,
   +0.023 on the bench). On a feed whose touches line up with the truth, as Metrica's do, they pay; on one
   where they don't, they cost a little. Segments and the decoder are robust.

## SkillCorner as training data later?

- **For the stoppages: yes, it's the only source we have with a tracked dead ball**, its dead share is sane,
  and the licence is clear (MIT, credit SkillCorner). It would teach robust what case B looks like.
- **For the team: with care.** The rate of change matches Metrica's and the organisers' hint, but its change
  moments sit ~2 m / ~0.5 s off a touch; training the touch-level stages (short, timing, segments' flips) on
  it would teach that blur.
- **The tracking is unlike ours** (broadcast, 10 fps, a quarter of frames cut away, extrapolated players and
  ball). A robust trained on it would need the bench's checks on both sources, as Metrica's matches did.
- **Keep a test.** Anything built after reading this page has seen all 20 matches. If SkillCorner is used for
  training, split it first (say 10 to train, 10 never opened) and say so.

## Caveats

- **The loader's choices shape the numbers** (players interpolated through cut-aways, SkillCorner's
  extrapolated ball kept, labels at the nearest 10 fps frame). They were fixed before scoring and not varied.
- **What SkillCorner's `group` means isn't documented**; whether people or an algorithm made it is unknown.
  Phases of play define D; their edges are SkillCorner's own timing.
- **Not like for like with the bench:** the bench's 0.704 is out of fold on simulated damage; SkillCorner is
  real broadcast tracking in another league. The comparison says how v12 fails, not by how much it will on
  our match.
- **What was looked at before the full run:** on one match (1874553), the raw fields, the labels, v12's
  label-free numbers and a few minutes of its letters against the truth. Nothing in the loader or the models
  changed afterwards. The oracles, the timing and geometry measures and the post hoc mix were added after
  seeing scores: they explain, they don't choose.
- **v10 and v11 run with Smart Tagging off** (`tags=[]`), like v12; the second-half kick-off clue does
  nothing without tags.
- **Credit:** tracking, possession and phases of play by SkillCorner (open data, MIT).
