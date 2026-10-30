# Model v2: v1 + the referee's whistle

| | |
|---|---|
| **status** | built 2026-10-30, as proposed |
| **change from v1** | one new clue. Everything else stays exactly as in [v1](v1-clues-and-decider.md), so any difference in the output comes from the whistle |
| **new input** | the tactical video's audio track (see [../data.md](../data.md), file 5) |
| **code** | `possession/audio.py` (whistle detector), `possession/v2.py` (v1 + whistle clue) |
| **run** | `python -m possession.v2` → `evaluation/results/v2.csv` in ~20 s (decoding the audio is most of it) |
| **output** | SHA-256 `48eabce61e177ca92cf6d4d09928d22cd97d17c3b0e3cd9586040bd1f0d80262`; byte-identical on re-run |
| **dependencies** | `requirements-team.txt`: numpy 2.5.4, pandas 3.1.0, av 19.0.2, on Python 3.14.7 |

## Why

v1 is blind to stoppages where the ball stays on the pitch and players keep moving. Example (found
by Jose in the viewer): at 1st half 03:06 the referee whistles, the truth is dead for 20.4 s, and
v1 keeps Barcelona for 15 s, getting 329 of the 511 dead frames wrong. None of its dead-ball clues
fire there. The whistle is the one clear signal, and it is on the tactical timeline already, so no
alignment is needed.

Football reason: during live play, the referee whistles to stop it (foul, offside, goal, end of a
half). So a whistle during live play is strong evidence the ball has just gone dead.

## What we measured first (audio and tracking only, no labels)

Over all 158,857 graded frames (`tools/find_whistles.py` approach, run on the whole match):

- **141 whistle-like tones, merged into 86 whistle events** (tones less than 2 s apart joined:
  referees often give two or three short blasts). A match has roughly 25 fouls, a few offsides,
  5 kick-offs and 2 half-ends, so 86 is the right order of magnitude for referee whistles plus some
  crowd and noise.
- **45 of the 86 happen where v1 already says dead.** The whistle changes nothing there.
- **41 happen during what v1 calls a possession.** These are the ones that matter. In 16 of them
  v1's own dead-ball clues fire within the next 15 s. In 25 there's no dead-ball sign at all. Some
  of those 25 are real stoppages v1 can't see (03:06 is one), and some are false alarms (crowd
  whistles, noise). Without labels we can't tell which.
- **Do players slow down after the whistle?** Comparing the mean speed of all visible players in
  the 3 s before with 2–6 s after: for 66 of the 86 events the change is within ±0.5 m/s, which is
  just noise. **11 show a clear slowdown** (≥ 0.5 m/s), including 03:06 (3.01 → 1.66 m/s). Kick-off
  whistles show the opposite: at the 1st-half kick-off, speed jumps from 0.38 to 4.11 m/s.

## How it works

### Piece 1: hear the whistles (new file, e.g. `possession/audio.py`)

1. **Decode the audio** to 16,000 samples per second, mono, with **PyAV pinned to one version**
   (`av==19.0.2`). It ships its own ffmpeg, so the organisers decode exactly the same numbers on
   their machine. With a system ffmpeg, a different version could decode slightly differently and
   break "byte-for-byte reproducible".
2. **One spectrum per video frame:** 640 samples = 40 ms = exactly one frame at 25 fps, with a Hann
   window (standard; it stops the frame edges creating fake frequencies).
3. **Whistle score per frame**, looking only between **2.5 and 4.5 kHz** (where referee whistles
   sound; the tones found were 2.6–3.9 kHz):
   - *prominence:* how far the loudest frequency stands above the typical level of that band.
     A whistle is one sharp tone, so it stands far above. Crowd noise and talking are spread out.
   - *level:* how loud that peak is compared with the match's typical peak, so faint far-off tones
     don't count.
4. **A tone** = at least **3 consecutive frames** (0.12 s) with prominence > **25 dB** and level
   > **15 dB**. **An event** = tones less than **2 s** apart, joined.

### Piece 2: check the players agree (new)

A crowd whistle doesn't stop the players; the referee's does. For each event, compare the mean
speed of all visible players in the **3 s before** with **2–6 s after**. The event **counts only if
speed drops by at least 0.5 m/s**. This also drops kick-off whistles automatically, since players
speed up after those.

### Piece 3: the clue (new)

For each counted event: **`2.0` per frame against H and A, for the 3 s after the whistle starts.**

### Piece 4: everything else

Unchanged from v1: same clues, same weights, same switch penalties, same decider.

### Worked through on 03:06

1. Tones at 15835, 15881, 15887 → one event starting at 15835.
2. Players' mean speed 3.01 → 1.66 m/s, a drop of 1.35 → counts.
3. Frames 15835–15909 get +2.0 against H and A. Over 75 frames that's 150, far more than the 25 it
   costs to switch to D (and the 25 to switch back later), so the decider switches to D at 15835.
4. After the 3 s, what keeps it in D is the same as in v1. The D → H/A penalty of 25 has to be
   beaten by "live play" evidence. In this stretch that evidence is weak (scattered "someone on the
   ball" frames, 0.5 each), and from 16218 "everyone stopped" pushes for D again. So it should stay
   dead until the free kick at ~16400.
5. Expected: D from 15835 instead of 16221 (the truth says 15889).

## Every setting and where its value comes from

| setting | value | source |
|---|---|---|
| decoder | PyAV `av==19.0.2`, 16 kHz mono | reproducibility; 16 kHz covers the whistle band with room |
| analysis window | 640 samples (40 ms), Hann | one window per video frame |
| whistle band | 2.5–4.5 kHz | where referee whistles sound; tones found at 2.6–3.9 kHz |
| prominence threshold | 25 dB | the far tail of the audio's own distribution (median 15.6, 99th percentile 28.9 dB); no labels |
| level threshold | 15 dB above the typical peak | same: 99th percentile is 17.9 dB |
| minimum tone | 3 frames (0.12 s) | a short blast is still ≥ 0.1 s; single-frame clicks are noise |
| merge gap | 2 s | referees give 2–3 short blasts for one stoppage |
| slowdown test | mean speed 3 s before vs 2–6 s after, drop ≥ 0.5 m/s | 66 of 86 events change by less than ±0.5, which is the noise band |
| clue weight | 2.0 per frame | same trust as "ball out": unambiguous when true |
| clue length | 3 s | long enough to beat both switch penalties (150 vs 50); short enough not to run over a quick restart |

## What it will change (predicted from unlabelled data)

- **10 whistles** during v1 possessions pass the slowdown test, and those are where v2 differs. In
  4 of them v1's own dead-ball clues fire within the next 15 s, so v2 mostly starts the dead ball
  earlier. In the other **6 there's no dead-ball sign within 15 s**: stoppages v1 misses or finds very
  late (03:06 is one: v1 goes dead 15.4 s after the whistle).
- **31 whistles** during v1 possessions fail the slowdown test and are ignored. That's the safe side:
  some are false alarms, but some are real stoppages we'll keep missing.

## What it won't fix, and what it could break

1. **Stoppages without a whistle** (ball out for a throw-in, corner or goal kick). Unchanged.
2. **Real whistles without a visible slowdown** (players were already slow, or kept jogging). Ignored.
3. **A false whistle that passes the slowdown test** splits a possession in two: possession, short
   dead ball, possession. That costs about as much as the stoppage it was meant to find. The slowdown
   test is there to keep this rare.
4. **The thresholds were set by looking at the audio's distribution, not fitted.** That's honest
   but not optimal.

## Alternative considered: trust every whistle

Skip the slowdown test and act on all 41 whistles during v1 possessions. It would catch more real
stoppages, but any crowd whistle during a possession would split it. Since we can't measure
precision without labels, the slowdown test is the safer bet. It can be revisited with data from
other matches.

## How we'll check it (without tuning on the public 5 minutes)

- Unlabelled: number of possessions, share of D, and the list of the 10 changed moments, each
  watched in the viewer to understand it (not to adjust frames).
- A "whistle" lane in the viewer, so every event shows next to the labels.
- The public 5 minutes: only to confirm nothing broke.

## Results

### The detector

The pinned PyAV decoder gives exactly the same 86 events (11 counted) as the ffmpeg command-line
exploration it replaced.

### What changed against v1 (no labels involved)

- **2,516 frames** differ from v1 (1.58 % of graded), in **10 stretches**, one after each counted
  whistle that fell during a v1 possession. The 11th counted whistle (2nd half 91:42) was already
  dead in v1.
- Possessions 173 → **177**; mean 23.9 → **22.9 s**; dead ball 35.0 → **36.3 %** of graded frames;
  Barcelona's share of ball-alive frames 69.2 → 70.1 %.

| whistle frame | match time | players' mean speed (m/s) | v1 at the whistle | v2 stays dead for | frames changed in the next 30 s |
|---|---|---|---|---|---|
| 15835 | 1st half 03:06 | 3.01 → 1.66 | H | 23.1 s | 383 |
| 17113 | 1st half 03:57 | 3.84 → 1.60 | A | 29.0 s | 420 |
| 23697 | 1st half 08:20 | 3.22 → 2.12 | A | **3.0 s** | 438 |
| 31173 | 1st half 13:19 | 2.64 → 1.32 | A | 4.0 s | 99 |
| 54740 | 1st half 29:02 | 4.09 → 2.37 | H | **3.0 s** | 75 |
| 74237 | 1st half 42:02 | 2.50 → 1.62 | H | 56.0 s | 222 |
| 88695 | 1st half 51:40 | 2.34 → 1.25 | A | 61.1 s | 275 |
| 118563 | 2nd half 48:12 | 2.79 → 1.17 | A | 37.3 s | 139 |
| 148019 | 2nd half 67:51 | 3.29 → 2.27 | A | **3.0 s** | 204 |
| 172217 | 2nd half 83:59 | 2.67 → 1.19 | H | 29.7 s | 198 |
| 183807 | 2nd half 91:42 | 1.78 → 0.88 | D | 22.2 s | 0 |

**Finding: three dead stretches last exactly 3.0 s** (08:20, 29:02, 67:51), the length of the push.
Probable cause: nothing else says "dead" there and the players keep moving, so the decider leaves D
the moment the push ends. A real stoppage almost never lasts 3 s (a free kick takes longer to set
up), so these are probably false alarms that passed the slowdown test. Each one splits a possession.
Not acted on yet; worth watching in the viewer.

### On the public 5 minutes (a check, not a ranking)

| | v1 | v2 |
|---|---|---|
| macroF1 | 0.7197 | **0.7888** |
| D F1 | 0.5620 | **0.7273** |
| spellF1 | 0.4276 | 0.4267 |
| possessions (truth 17) | 12 | 13 |
| frames right | 5,706 / 7,500 | 6,017 / 7,500 |

Two counted whistles fall in this window, one good and one bad:

- **03:06 (15835) helped.** v2 goes dead at the whistle, 2.2 s before the truth (15889), instead of
  v1's 13 s after. The truth is dead for 511 frames; v2's dead stretch covers all of them and ends
  12 frames after the real restart (16412 vs 16400).
- **03:57 (17113) hurt.** It passed the slowdown test (3.84 → 1.60 m/s), but the truth keeps Athletic
  in possession for another 7 s, then gives Barcelona 3.5 s, then goes dead at 17380, 10.7 s after
  the whistle. v2 calls all of it dead and loses that short Barcelona possession. A false alarm that
  got through: the risk this plan named.

spellF1 doesn't move (one possession found, one lost), and the window is far too small to say more.
Per the decision log, neither case is used to change a setting.
