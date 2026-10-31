# The test bench: measuring our models on other matches

| | |
|---|---|
| **status** | **built 2026-10-30**, as approved, with the changes listed under "What changed from the plan" |
| **run** | `python -m bench.idsse download` (once, 2.6 GB) → `python -m bench.idsse parse` (~1 min) → `python -m bench.measure` (~3 s) → `python -m bench.run` (~40 s for every model and level). Checks: `python -m bench.labels`, `python -m bench.degrade` |
| **what it is** | 7 open Bundesliga matches, labelled H/A/D per frame by their own data, with their tracking damaged to look like ours, scored with the organisers' scorer |
| **why** | No leaderboard, and the public 5 minutes may not choose between versions ([limitations.md](limitations.md)). This is where versions get compared and settings get chosen |
| **changes our submission?** | No. It changes how we decide what goes into it |
| **new dependencies** | none (Python's own XML reader, numpy, pandas) |
| **effort** | 1.5–2 days |

## What it can and can't tell us

**Can:** whether a change to the **tracking-based** part of a model helps. That covers the decoder,
on-the-ball rules, dead-ball detection from players, restart shapes and PathCRF. It tells us across
7 matches, at several levels of damage.

**Can't:**
- **The whistle (v2), Smart Tagging and the broadcast.** IDSSE has no audio, no tags and no video,
  so on the bench v2 equals v1, and v1 runs without its tag clues. Those clues keep reasoned
  weights.
- **Our final score.** The bench ranks ideas; it doesn't forecast. DFL's labels and our tracker's
  faults are only approximations of Metrica's convention and tracker.

## The data: IDSSE

- **What:** "An integrated dataset of spatiotemporal and event data in elite soccer" (Bassek, Rein,
  Weber, Memmert, *Scientific Data* 12:195, 2025). Figshare article 28196177, licence **CC BY 4.0**.
  Already approved by the organisers for a similar use (#16).
- **Matches:** 7 from 2022-23: 2 Bundesliga, 5 2. Bundesliga, all at 25 fps. That's about 1.0
  million frames, roughly 11 hours. None is ours (it's a different league and season), and the
  loader checks it anyway.
- **Files per match:** match information (team names, home/away, pitch size), events, and positions.
  The positions file is 349–419 MB; 2.63 GB in total. Checked on 2026-10-30 through the Figshare
  API.
- **Layout of a positions file**, checked by reading its first and last bytes:
  - one block per person per half (`FrameSet GameSection="firstHalf" TeamId=… PersonId=…`);
  - one line per frame: `N` (frame number), `X`, `Y` in metres with the centre spot at (0, 0),
    `S` speed in km/h;
  - the ball's block also carries `Z` (height), **`BallPossession`** (1 = home, 2 = away) and
    **`BallStatus`** (0 = dead, 1 = alive).
- **Where it goes:** `data/external/idsse/`, which the repo's `.gitignore` already excludes (`data/*`).
  Downloaded once with Figshare's published checksums verified.
- **Attribution:** CC BY requires citing it. The citation goes in our README and the technical
  document.

## How it works

### 1. Read (`bench/idsse.py`)

- **Parser:** our own, a streaming read with Python's built-in XML parser, one match at a time.
  - Memory stays small on the 8 GB machine.
  - We read `BallPossession` and `BallStatus` exactly as written. kloppy, the usual library, turns
    a missing possession value into "away" (found in the datasets research).
- **Cache:** each match becomes arrays in the same shape our loader gives for BAR-ATH: players
  `[frames, slots, 2]` in metres, speed, ball `[frames, 2]`, team per slot. Saved as one compressed
  file per match under `data/external/idsse/cache/`.
- **Coordinates:** IDSSE puts (0, 0) at the centre spot, while ours start at a corner, so we add
  half the pitch length and width. The pitch size comes from each match's metadata (105 × 68 m).
- **Speed:** km/h ÷ 3.6 → m/s, the unit our file uses ([data.md](data.md)).
- **Sanity checks, printed:**
  - total frames against the paper's 1,002,644;
  - 0.04 s between frames;
  - about 22 players on the pitch;
  - the share of dead frames;
  - no frame with alive status but no possession value. If any exist, they're counted and given
    the last team, the "stays with the last team" rule.

### 2. Label (`bench/labels.py`)

- `BallStatus` 0 → **D**; otherwise `BallPossession` 1 → **H**, 2 → **A**. Every frame of both
  halves is "graded".
- **Labels are used raw.** We don't reshape them to look like the organisers' truth, because any
  reshaping rule would need a setting, and we'd have nothing fair to choose it on.
- **Three checks, reported but not acted on**, because they say how far DFL's convention sits from
  Metrica's:
  1. how many possessions last under 0.5 s and under 1 s (are touches counted?);
  2. whether D starts when the ball crosses a line or later (the ball's position at D edges);
  3. whether the first team after each dead stretch is the team the event file says takes the
     restart.
- **A sensitivity run:** every comparison is repeated with possessions shorter than 0.5 s merged
  into their neighbours. If a ranking of two versions flips between the raw and merged labels, that
  comparison is too close to trust.

### 3. Measure our own tracking's faults (`bench/measure.py`)

What the damage imitates is measured on **BAR-ATH's input tracking only**. This is input data, so
no labels and no human-made data are involved. The output is a small file of summary statistics,
`bench/damage_stats.json`, holding distributions and rates, never frames.

| fault | what is measured on our tracking | measured (graded frames) |
|---|---|---|
| ball missing | share of frames; every gap's length | **40.7 %**; 149 gaps, median 10.7 s, mean 17.4 s |
| ball while players are stopped | ball-missing share inside v1's "everyone stopped" stretches vs elsewhere: a rough, label-free stand-in for dead time | **83.0 % stopped vs 29.5 % moving** (stopped = 21.0 % of frames) |
| losing the ball | chance a seen ball is missing on the next frame, by crowd (players within 3 m) and by whether players stop within the next 5 s | 0.0012–0.0038 per frame in play, up to 0.06 in a crowd as play stops |
| losing it near a line | the same chance by where the ball is: beyond a line, within 5 m of one, further in | 0.0058 / 0.0027 / 0.0011: **5× likelier beyond a line** |
| finding the ball | chance a missing ball is seen again on the next frame, by "stopped" | 0.0029 moving, 0.0015 stopped (a lost ball stays lost through a stoppage) |
| ball position noise | spread of the ball around its 5-frame mean, on slow stretches | 0.008 m: negligible, the track is already smooth |
| player track fragments | length of each unbroken run of a slot; slots per team | 1,219 pieces, median 35.5 s, mean 106.8 s; 24 + 22 slots |
| players missing | players visible per frame | 20.49 of 22 |
| speed missing | player speed empty while the position exists | 8.0 % |

These differ slightly from [data.md](data.md) (2,218 tracklets, mean 68 s; 9.8 % speed missing),
which counted over the whole video rather than only the graded halves.

### 4. Damage IDSSE to look like ours (`bench/degrade.py`)

Applied to each match with a fixed random seed per match and level, so every run sees the same
damage.

1. **Anonymise:** drop player IDs, keep the team.
2. **Fragment:** cut each player's track into pieces whose lengths follow ours, and put each piece
   into a free slot of its team (46 slots in total, as ours). Leave short holes between pieces, so
   that on average 20.5 of 22 players are visible.
3. **Blank 8.0 % of player speeds**, as ours.
4. **Lose the ball**, as a chain that switches between "seen" and "missing" every frame:
   - **Seen → missing:** ours, by crowd size (damaged players within 3 m of the true ball) and by
     whether players stop within the next 5 s (v1's own "stopped" clue on the damaged tracking),
     times how much likelier ours is to lose it beyond or near a line.
   - **Missing → seen:** ours, by whether players are stopped.
   - **Overall rate:** three levels, **30 %, 41 % and 55 %** missing, made by scaling the losing
     chance (scale found by bisection, pooled over the 7 matches). 41 % is ours; the other two
     show whether a method only works at one level.
   - Because the chances depend on "stopped" and on the lines, "no ball" carries roughly the
     dead-ball hint it carries on ours. See "Results" for how close it gets.
5. **Jitter the ball** by the noise measured on ours, and drop its height (we have none).
6. **Plus a level 0** with no damage at all, as a reference.

**Not imitated yet (known gaps):**
- false ball detections (the ball jumping to a player);
- identity swaps inside one piece;
- the same player appearing in two slots at once.

If the bench later disagrees with what we see on BAR-ATH, these are the first things to add.

### 5. Run and score (`bench/run.py`)

- **Command:** `python -m bench.run --model v1 --level 41` runs a model on all 7 matches.
- **Scoring:** every match is scored with the organisers' own `evaluate()` from
  `evaluation/possession_metrics.py`, the same function that will rank us.
- **Reported per match, then as the mean and spread across matches:**
  - spellF1 and macroF1;
  - dead-ball F1;
  - possessions found vs true, and mean possession length.
- **First things run on it:**
  - the organisers' baseline idea (nearest player, then a majority vote over ±25, ±50 and ±125
    frames), rewritten to take arrays instead of our files;
  - v1 without its tag clues.
- **A small refactor of v1:** it currently reads BAR-ATH's halves and pitch size by itself; it will
  take them as inputs instead. That must leave `evaluation/results/v1.csv` and `v2.csv`
  byte-identical (SHA-256 checked before and after).
- **Later steps** use it with **leave-one-match-out**: choose a setting on 6 matches, score on the
  7th, rotate through all 7, report all 7.

### 6. Checks the bench must pass before we trust it

1. **Reading check:** with no damage, where the ball is slow (under 2 m/s) and a player is within
   1 m of it, that player's team should almost always be the labelled team. If it isn't, the labels,
   the teams or the coordinates are mis-read.
2. **Scores fall as damage rises** (0 → 30 → 41 → 55 %). If they don't, the damage isn't biting.
3. **Smoothing helps the baseline's spellF1 at 41 %,** as it does in the organisers' table. If it
   doesn't, the bench doesn't punish flicker the way our match does.

## Rules we hold to here

- **Nothing about BAR-ATH's labels is used**, including the public 5 minutes. Only its input
  tracking feeds the damage statistics.
- **We don't adjust the damage to make the bench reproduce the organisers' published baseline
  scores** (0.2433 and the others). Those numbers come from the private ground truth, so fitting to
  them would be fitting to it indirectly. Check 3 above uses only the *direction* the organisers
  state, smoothing beats raw, which is a property of the metric, not of our match.
- IDSSE's licence attribution goes in the README and the technical document.
- No IDSSE data, cache or results with frames go in git. Summary tables of bench scores may.

## Decisions for Jose before building

| | choice | recommended | why |
|---|---|---|---|
| 1 | Our own parser, or kloppy | **own parser** | No new dependency, low memory, and we read possession exactly as written |
| 2 | Raw DFL labels, or reshaped toward the organisers' convention | **raw, plus the sensitivity run** | Any reshaping needs a setting we couldn't choose fairly |
| 3 | Add Metrica's 3 sample games now | **later** | They have no possession field; labels must be built from their events, which is a separate job |
| 4 | Commit `bench/damage_stats.json` (summary statistics of our match's input tracking) | **yes** | Like the numbers already in data.md: rates and distributions, nothing that rebuilds the tracking |

## Files

```
bench/
  idsse.py      download (checksums verified), parse, cache
  labels.py     BallStatus/BallPossession → H/A/D; the three convention checks
  measure.py    our tracking's faults → damage_stats.json
  degrade.py    damage IDSSE to look like ours, 4 levels, seeded
  run.py        run a model on the 7 matches, score with the organisers' evaluate()
data/external/idsse/      raw XML and cache (gitignored)
bench/damage_stats.json   our tracking's faults: rates and distributions (committed, decision 4)
bench/results/scores.csv  one row per model, level, label set and match: scores only (committed)
```

## What changed from the plan, and why

Measuring changed four things. None of them used a label from BAR-ATH.

1. **How the ball is lost.** The plan drew gap lengths at random. Measuring showed that on ours the
   ball is missing 83 % of the time while players stand still, against 29.5 % while they move: a
   lost ball stays lost through stoppages. Random gap lengths can't reproduce that, so the ball
   became a seen ↔ missing chain whose "find" chance depends on "stopped".
2. **Two more contexts for losing it,** added after comparing the damaged ball with ours (step-by-step
   numbers below):
   - our tracker is 5× likelier to lose the ball beyond a line, and 2.5× within 5 m of one;
   - it tends to lose it just **before** players slow down (it goes out, the whistle goes), so the
     losing chance depends on "players stop within the next 5 s" rather than "players are stopped".
3. **Self-check 1 was badly designed.** "The nearest-player baseline should be strong with no
   damage" fails for a good reason: even with perfect tracking, the nearest player's team matches
   possession on only 77 % of live frames, because during a pass the nearest player is often an
   opponent. Replaced by the reading check (a player within 1 m of a slow ball).
4. **Referees are tracked in IDSSE** (6 FrameSets per match); they're dropped, since our tracking
   has no referee slots. Measured fault values also differ slightly from data.md's (see the table
   in step 3).

## Results (2026-10-30)

### The data reads correctly

- **Frames:** 1,002,644 across the 7 matches' halves, exactly the paper's count. Every frame has a
  status; no alive frame lacks a possession value, so nothing had to be filled.
- **Reading check:** where a player is within 1 m of a slow ball, his team is the labelled team on
  **95.3–98.2 %** of frames (7,663–12,788 frames per match).
- **Dead ball:** 40.7–51.7 % of frames per match.

### How DFL's labels behave (the three checks)

| check | result (range over the 7 matches) | what it means for us |
|---|---|---|
| 1. short possessions | 370–511 possessions per match, median 3.0–4.7 s. Under 0.5 s: 14–32 per match; under 1 s: 37–86 | Touches are mostly **not** counted as possessions. Merging those under 0.5 s changes only 74–175 frames per match, so the raw/merged sensitivity run barely moves anything |
| 2. when D starts | 92–112 stoppages per match. In 40–81 the ball is outside the pitch when D starts, **3–4 frames (0.12–0.16 s)** after it crossed the line (90 % within 8 frames). In 22–59 the ball is inside: whistle stoppages | DFL marks D almost the moment the ball crosses. Metrica's timing is unknown; this is the "D edge" caveat |
| 3. who restarts | after **656 of 670** stoppages that have a restart event within 5 s (97.9 %), the first team with the ball is the team that restarts | Restarts are labelled consistently: the bench can score "who restarts" ideas (step 3 of the plan) |

### How close the damage gets to ours (level 41)

| | bench | ours |
|---|---|---|
| ball missing overall | 41.0 % (calibrated; losing chance × 0.704) | 40.7 % |
| … while players are stopped | 65.8 % | 83.0 % |
| … while players move | 34.3 % | 29.5 % |
| ball gaps per 90 minutes | 144 | 127 |
| gap length median / mean / p90 | 10.8 / 15.4 / 34.8 s | 10.7 / 17.4 / 34.2 s |
| player speed missing; players visible | 8.0 %; ~20.5 of 22 (set from ours by construction) | 8.0 %; 20.49 |

Each addition moved the "while stopped" share closer: 50.3 % (stopped only) → 58.5 % (+ lines) →
65.8 % (+ stopping soon). **Known gap:** on the bench "no ball" is a weaker dead-ball hint than on
ours, and the ball goes missing in live play a little more. So the bench is a bit harsher than ours
on team calls in play, and undersells ideas that read "no ball" as "dead".

### Scores (raw labels; merged labels give the same picture)

Mean over the 7 matches, with the spread (standard deviation) of spellF1 across matches:

| level | model | spellF1 | ± | macroF1 | D F1 | team right (both alive) | possessions found / true |
|---|---|---|---|---|---|---|---|
| 0 | baseline_smooth50 | 0.308 | 0.029 | 0.392 | 0.000 | 0.835 | 378 / 437 |
| 0 | v1 | 0.292 | 0.042 | 0.721 | 0.735 | 0.846 | 185 / 437 |
| 30 | baseline_smooth50 | 0.240 | 0.020 | 0.506 | 0.415 | 0.799 | 428 / 437 |
| 30 | v1 | 0.224 | 0.037 | 0.683 | 0.727 | 0.788 | 154 / 437 |
| **41** | baseline_raw | 0.084 | 0.012 | 0.530 | 0.579 | 0.746 | 1,487 / 437 |
| **41** | baseline_smooth25 | 0.207 | 0.008 | 0.535 | 0.540 | 0.781 | 583 / 437 |
| **41** | **baseline_smooth50** | **0.217** | 0.018 | 0.528 | 0.503 | 0.789 | 448 / 437 |
| **41** | baseline_smooth125 | 0.176 | 0.019 | 0.491 | 0.398 | 0.777 | 315 / 437 |
| **41** | **v1** | **0.224** | 0.034 | **0.683** | **0.732** | 0.784 | **148 / 437** |
| 55 | baseline_smooth50 | 0.183 | 0.019 | 0.525 | 0.558 | 0.785 | 445 / 437 |
| 55 | v1 | 0.178 | 0.030 | 0.664 | 0.725 | 0.760 | 128 / 437 |

v1 minus baseline_smooth50, match by match: at 41 % v1 is ahead on 6 of 7 (mean +0.007); at 30 %
on 1 of 7 (−0.016); at 55 % on 3 of 7 (−0.005).

**Self-checks:** reading check passed (above). Scores fall as damage rises, for every model.
Smoothing lifts the baseline's spellF1 from 0.084 to 0.217 at 41 %.

### What this says

- **On the bench, v1 ties the organisers' floor on the ranking metric.** It is far better on
  macroF1 (0.68 vs 0.53) and on dead ball (D F1 0.73 vs 0.50), but those don't decide the ranking.
- **The reason is plain: v1 merges.** It reports ~148 possessions per match where there are ~437,
  with a mean length of 25–35 s against a true 5.5–9.2 s. Its switch penalties (15 between teams,
  25 into and out of dead) need about a second of one-sided evidence to switch, so most real
  possessions are swallowed. The floor, whose only dead-ball sense is "the ball is missing", at least gets
  the count right.
- **Caveat:** DFL counts possession by "the player in control"; the organisers' truth follows the
  team that Metrica's corrected tracking names as owning the ball, which is the same kind of
  definition. If Metrica's possessions run longer, v1's merging costs less on our match than here.
  The bench can't settle that; it does say which direction to move.
- **Next:** step 1 of the plan (a decoder that knows how long possessions last, with its settings
  chosen here by leave-one-match-out) targets exactly this.

## Second set: Metrica's sample games (added 2026-10-30)

Added by the [convention check](convention-check.md), piece 4.
- **What:** 3 anonymised Metrica matches, labels built from Metrica's own events (recipe in
  `bench/metrica.py`). Their player speed is computed from positions, since the files have none.
- **How to use:** every bench command takes `--set idsse` (the default, unchanged), `--set metrica`
  or `--set all`. Each set is calibrated to the damage level on its own, so adding one never moves
  the other's numbers.
- **One difference:** Metrica blanks the ball in dead time (37.4 % of frames), so on these games
  "no ball" is a much stronger dead-ball hint than on ours, and the 0 % and 30 % levels both sit at
  about 37 %.

At 41 % on Metrica's labels: floor 0.364, v1 0.384, **v3 0.512** (344 possessions found where the
events say 351).
