# Model v1: per-frame clues + a decider that charges for switching

| | |
|---|---|
| **status** | built 2026-10-30 |
| **inputs** | tactical tracking (files 1–2), Smart Tagging (file 4); see [../data.md](../data.md) |
| **code** | `possession/data.py` (loading), `possession/decoder.py` (Viterbi), `possession/v1.py` (clues, costs, output) |
| **run** | `python -m possession.v1` → `evaluation/results/v1.csv` in ~4 s |
| **output** | SHA-256 `9dbcea2b755ab082b740d93462382dbfd17ca4b9bd057cde35fec4b0e071ee67`; byte-identical on re-run |
| **full-match score** | unknown until results day; the public window cannot rank models (see Results) |

## Why this model

Two facts from the data and the scoring decide the shape:

1. **The scoring counts whole possessions** ([../challenge.md](../challenge.md)). Flickering between
   teams invents fake possessions and destroys the score. The repo's baseline goes from 131
   possessions to 26 in the first 5 minutes just by smoothing (the truth has 17). So the core of
   the model must be something that **decides when to change label**, not just what each frame
   looks like.
2. **The nearest player to the ball is a bad clue on its own.** It credits whoever happens to be
   nearest, including a player a pass is flying past, and says nothing while the ball is missing
   (40.7 % of graded frames, in long blocks).

So: a few clues per frame, each with a reason to exist, and a decider that balances them against
the cost of changing label.

## How it works

### Units: "frames of evidence"

Every clue adds a cost, per frame, to the labels it argues against. A clue of weight 1.0 that stays
active for 25 frames adds 25. Switch penalties use the same unit, so a penalty of 25 means
"changing label needs about one second of clear evidence against the current one". This makes every
number comparable with every other.

### Piece 1: read and convert (`possession/data.py`)

- Tracking line *N* → video frame *N − 1*, so every array is on the submission timeline.
- Positions × (105.3, 68.0) → metres, so distances along and across the pitch are comparable.
- Slot → team from the metadata (first team in the file = home = Barcelona).
- The ball's own speed column is dropped (empty on 91 % of ball frames, wrong when filled).

### Piece 2: clean the ball

- **Ball speed** = distance between consecutive frames × 25, in m/s.
- **Jump filter:** a step faster than **40 m/s** is a re-acquisition glitch (the hardest shots are
  ~35–40 m/s; the data has fake steps up to 134 m/s). That frame's speed becomes "unknown".
- **Ball out:** visible and more than **1 m** beyond a line. The metre absorbs tracking error.

### Piece 3: clues

**Clue A, "who is on the ball"** (argues against the *other* team).
- On frames where the ball is visible and its speed is known and **below 8 m/s** (players dribble at
  up to ~7–8 m/s; passes travel at 10–30 m/s, so a faster ball is in flight and belongs to nobody).
- The team whose nearest player is **within 2 m** (a stride plus tracking error) **and** closer than
  the other team's nearest player.
- Strength 0–1 = (opponent's distance − own distance) / 2 m, capped at 1. If no opponent is
  visible, strength 1. A tight duel gives a weak clue; a free player gives a full one.
- Cost: `1.0 × strength` added to the other team's label.
- It also argues against D (`0.5`) when the ball is inside the pitch: someone playing a slow ball on
  the pitch is evidence that play is live.

**Clue B, "is the ball dead or alive"** (two-sided).
- Against H and A (= for D):
  - **ball out** → `2.0` per frame. Strongest, because it is unambiguous when true.
  - **everyone stopped** → `1.0` per frame. The 1-second mean speed of all visible players stays
    **under 1.0 m/s for at least 3 s**. The whole stretch is marked, including its first 3 s.
  - **set-piece tag started in the last 15 s** → `0.3` per frame. Tags start 2.8–12.2 s before the
    restart they describe, and 2 of 5 restarts in the public window had none, hence weak.
  - **before the 2nd-half kick-off tag** → `1000` per frame. The scored second half starts ~2 min
    before its kick-off, so this is certain. Found from the `Kick Off (Start)` tag, not hard-coded.
- Against D (= for live play):
  - **players active**: the same 1-second mean speed **above 2.0 m/s** → `0.5` per frame.
  - **someone on the ball inside the pitch** (from clue A) → `0.5` per frame.

**Clue C, "who is attacking"** (argues against the other team).
- BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING TRANSITION → their own team.
- DEFENDING IN ATTACKING / MIDDLE / DEFENSIVE THIRD and DEFENSIVE TRANSITION → the **opposite**
  team (they are the mirror twins, see [../data.md](../data.md)).
- Where tags support both teams at once, they cancel.
- Cost: `0.2` per frame. Weak on purpose: fixed-length windows, missed events. It matters most
  while the ball is missing, when nothing else speaks.

### Piece 4: the decider (`possession/decoder.py`)

Viterbi: among all label sequences for a half, the one with the lowest total of per-frame costs plus
switch penalties. Exact, deterministic (ties go to the lower label index: H, then A, then D), run
separately on each graded half.

| switch | penalty | meaning |
|---|---|---|
| H ↔ A | 15 | ~0.6 s of one-sided control by the other team |
| H/A → D | 25 | ~1 s of dead evidence |
| D → H/A | 25 | ~1 s of live evidence |

Worked examples (cost 1 per disagreeing frame, switch 3), verified on the real decoder:
- clue `H H A H H H` → `H H H H H H`: one wrong frame (1) is cheaper than two switches (6).
- clue `H H H H A A A A A A` → `H H H H A A A A A A`: the switch lands where the evidence changes.
- clue `H H A A A A A A` → `A A A A A A A A`: two wrong frames (2) are cheaper than one switch
  (3). **The penalty also swallows short stretches at the edges**, which is the same trade-off that
  will erase some very short real possessions.

When the ball is missing, clue A is silent, so the cheapest choice is usually to keep the current
label: football's own rule, "possession stays with the last team that had it."

### Piece 5: output

H, A or D on the 158,857 graded frames, `X` everywhere else (free there). Checked with
`evaluation/validate_submission.py`.

## What changed from the plan, and why

The plan was written before measuring. Measuring changed four things:

1. **"Everyone stopped" threshold: 1.5 → 1.0 m/s.** The plan reasoned "walking is 1.4 m/s". But the
   mean speed of all visible players has a single bump (median 1.52, quartiles 1.02 and 2.32 m/s)
   with no natural split (`tools/explore_data.py`). Players cover ~10–11 km a match, ~1.8 m/s on
   average, so slow open play easily sits near 1.5. At 1.5 m/s the clue would cover 45.7 % of graded
   frames; at 1.0 (strolling pace for the whole team) it covers 21.0 %. Measured on unlabelled
   data only.
2. **Clue B became two-sided.** As planned, nothing argued *against* D, so once the decider entered
   a dead ball it could never leave. Added "players active" (> 2.0 m/s; about two thirds of graded
   frames have a 1-second team mean below that) and "someone on the ball inside the pitch", each
   `0.5` against D.
3. **The worked example was wrong.** The plan said clue `H H A A A A A A` would switch at the third
   frame. The real decoder labels it all `A` (see above). The behaviour is correct; the example
   wasn't.
4. **The guessed settings got values**: switch penalties 15 / 25 / 25, weights as above. They are
   reasoned in "frames of evidence", not fitted to anything.

## Every setting, and where its value comes from

| setting | value | source |
|---|---|---|
| pitch size | 105.3 × 68.0 m | `games.py` (measured by the organisers) |
| jump filter | 40 m/s | physics: hardest shots ~35–40 m/s |
| ball-out margin | 1 m | tracking error at the line (judgement) |
| on-the-ball distance | 2 m | physics: one stride + tracking error |
| ball-in-flight speed | 8 m/s | physics: max dribbling speed vs pass speed |
| "stopped" | team mean < 1.0 m/s for ≥ 3 s | strolling pace; measured spread of team speed (unlabelled) |
| "active" | team mean > 2.0 m/s | clearly above a stroll; ~two thirds of frames are below it (unlabelled) |
| set-piece window | 15 s after tag start | measured tag offsets 2.8–12.2 s (3 examples) |
| weights | on-ball 1.0, out 2.0, stopped 1.0, set piece 0.3, before kick-off 1000, active 0.5, on-ball-alive 0.5, tag team 0.2 | **judgement**: relative trust in each clue |
| switch penalties | team 15, to dead 25, from dead 25 | **judgement**, in frames of evidence |

The weights and penalties are the weakest part: they are reasoned, not measured. Nothing was tuned on
the public 5 minutes.

## Results

### On the full match (no ground truth involved: what the model says)

| | v1 | baseline_smooth50 (the floor) |
|---|---|---|
| possessions | **173** | 594 |
| mean / median possession | **23.9 s / 16.7 s** | 7.2 s / 4.6 s |
| possessions shorter than 2 s | 3 | — |
| labels on graded frames | H 44.9 %, A 20.0 %, D 35.0 % | — |
| Barcelona share of ball-alive frames | 69.2 % | — |

Clue coverage over graded frames: someone on the ball 27.0 %, ball out 1.6 %, stopped 21.0 %, active
33.4 %, set-piece window 13.7 %, before 2nd-half kick-off 1.8 %, a tag supports Barcelona 27.0 %, a
tag supports Athletic 11.6 %.

### On the bench (7 other matches, tracking damaged to look like ours; added 2026-10-30)

Full numbers in [../bench.md](../bench.md). There are no Smart Tagging clues on the bench, so v1
runs without clue C, the set-piece window and the kick-off rule.

| damage (ball missing) | v1 spellF1 | floor (`baseline_smooth50`) | v1 possessions found / true | v1 D F1 |
|---|---|---|---|---|
| 30 % | 0.224 ± 0.037 | 0.240 ± 0.020 | 154 / 437 | 0.73 |
| **41 % (ours)** | **0.224 ± 0.034** | **0.217 ± 0.018** | **148 / 437** | 0.73 |
| 55 % | 0.178 ± 0.030 | 0.183 ± 0.019 | 128 / 437 | 0.73 |

A tie with the floor on the ranking metric, despite a much better dead ball. v1 merges: its
possessions average 25–35 s where the true ones average 5.5–9.2 s. This confirms weakness 2 below
("too few possessions, probably") with evidence that doesn't come from the public window.

### On the public 5 minutes (a check that nothing is broken, not a ranking)

| submission | macroF1 | spellF1 | possessions (truth: 17) | mean duration (truth: 14.2 s) |
|---|---|---|---|---|
| **v1** | **0.7197** | **0.4276** | 12 | 22.2 s |
| baseline_smooth125 | 0.5812 | 0.4242 | 16 | 18.5 s |
| baseline_smooth50 | 0.6207 | 0.3256 | 26 | 10.8 s |
| baseline_smooth25 | 0.6200 | 0.2868 | 36 | 7.6 s |
| baseline_raw | 0.5972 | 0.0378 | 131 | 2.0 s |

- **What this proves:** the pipeline is wired correctly. Frames line up and teams are the right way
  round (H↔H and A↔A dominate the confusion matrix).
- **What it does not prove:** that v1 is better on the full match. With 17 possessions, the scorer's
  bootstrap puts v1's spellF1 anywhere from 0.23 to 0.62 (5–95 %), and gaps below ~0.39 are noise.
  The organisers' own example: on this window smooth125 beats smooth50, and on the full match it
  loses.
- **Observed, not acted on:** in this window v1 labels 670 of the 1,469 dead-ball frames as H (D recall
  0.44), and it reports 12 possessions against 17. Changing settings because of that would be tuning
  on the public window, which the rules forbid.

## Known weaknesses

1. **Changes of possession inside ball gaps** (median 10.7 s). v1 keeps the last label through a gap,
   so a turnover while the ball is hidden is found late, or never if the gap ends in a stoppage.
2. **Too few possessions, probably.** Mean 23.9 s and 173 possessions over the match. The switch
   penalties lean towards merging. Short real possessions (the truth has one of 0.8 s) are lost.
3. **Dead-ball detection is coarse.** "Stopped" is a blunt signal (team speed has no clean split), and
   set-piece tags are late, early or missing. Example, found by Jose in the viewer: at 1st half
   03:06 (frame 15835) the referee whistles, and the truth is dead from 15889 to 16399 (20.4 s).
   v1 keeps Barcelona until 16221, so 329 of those 511 dead frames come out as H. None of its
   dead-ball clues fire: the ball stays visible and inside the pitch, players keep moving above
   1.0 m/s for 13 s, and this restart has no set-piece tag. The whistle is in the audio
   (`tools/find_whistles.py`).
4. **Throw-ins**: team `N/A` in the tags, so clue C says nothing about who restarts.
5. **The weights and penalties are judgement.** We have no legal data from this match to calibrate
   them.

## Next steps (not built; each needs its own explanation first)

- **Whistle clue:** a referee's whistle is strong evidence the ball just went dead. It's on the
  tactical timeline already (see [../data.md](../data.md), file 5).
- **Calibrate the weights and penalties on other matches** that publish per-frame possession and
  ball-in-play (candidate: IDSSE Bundesliga, to verify). This is the only legal way to replace
  judgement with measurement.
- **Team-shape clue for ball gaps:** which team is pushing towards the opponent's goal, which team
  is compact and retreating.
- **More tags as clues:** BALL IN FINAL THIRD, BALL IN THE BOX, LONG BALL and PLAYERS IN THE BOX also
  name the attacking team and are unused in v1.
