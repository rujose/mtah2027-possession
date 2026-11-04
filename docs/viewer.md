# The match viewer

A local web app that plays the tactical video with the tracking drawn on top, a 2D pitch map, and
timelines of labels and clues, all locked to the same frame.

> **Since 2026-11-03, our match shows no model output.** The organisers' clarification forbids
> checking our output against the video ([limitations.md](limitations.md), section 0). So for our
> match the viewer shows only what we may inspect to understand what counts as possession:
> - the video;
> - the tracking **as delivered**, not repaired;
> - Smart Tagging;
> - the public truth (first 5 minutes);
> - whether the ball is tracked.
>
> **Notes are off** there: they would be data created by hand on the provided game. The page hides
> the form, and the server drops any note for our match.
>
> **The bench matches keep everything below** (models, clues, notes): they are external data, with
> no video. Everything below about model lanes on our match describes the viewer before that date.

## Run it

```bash
.venv/bin/python -m viewer.server              # then open http://localhost:8765
.venv/bin/python -m viewer.server --rebuild    # after a model writes a new CSV to evaluation/results/
```

The first start builds `viewer/cache/` (~4 s, ~47 MB). The cache is the tracking and homography
re-encoded, so it is derived from the gated data: it is gitignored and must never be published.

## What's on screen

| part | what it shows |
|---|---|
| **video** | the tactical video. Dots on each tracked player (blue Barcelona, red Athletic) and the ball (yellow). A player with a yellow ring is their team's nearest player to the ball, within 2 m (v1's "on the ball" distance). "ball not tracked" appears when the tracking has no ball. |
| **label strip (under the video)** | truth, our models (newest first) and the baselines around the current frame, same window as the detail timeline (`[` `]` to zoom). The letter on the right of each lane is its label at the playhead. Empty lane = no label there (the truth only covers the first 5 minutes). Click to jump. |
| **pitch map** | the same frame from above, in metres. Ball trail over the last second. When the ball is lost, a dashed circle marks where it was last seen and how long ago. Bottom-left says which way Barcelona attacks in this half. |
| **labels at this frame** | the true label (first 5 minutes only), v1, and the four baselines. |
| **v1 clues** | which of v1's clues are on at this frame, the ball's speed, and the distance from the ball to each team's nearest player. |
| **Smart Tagging** | every tag whose window contains this frame. |
| **whole match** | every lane over the full video. Grey bands are not graded (before kick-off, half-time, after the whistle). The white line is the current frame, the light box is the detail window. Click to jump. |
| **detail** | the same lanes around the current frame (±30 s by default, zoom with `[` `]`). Click to jump. |

Lane colours: blue = Barcelona / H, red = Athletic / A, grey = dead ball / D, orange = a clue is on,
green = ball tracked.

**The current model's learned signals** (added 2026-10-30), at the bottom of the clue lanes and in
the side panel:

| lane | what it shows | colours |
|---|---|---|
| **v4: P(dead)** | the dead-ball model's probability that play is stopped, frame by frame. The panel shows the number | nothing below 50 %; light grey 50–80 % ("looks dead"); grey above 80 % |
| **v6: restart clue** | which team the restart model expects to take the next restart, and how sure it is. It's only on after 2 s of dead-looking play and for 3 s after, the same as in v6 | blue Barcelona, red Athletic; strong colour at 80 % or more, faded below; empty where the clue is silent |
| **v7: PathCRF** | PathCRF's estimate of which team has the ball, where its clue applies (every frame v6 calls live) | as the restart clue |
| **v8: keeper repaired** | where the keeper repair moved a piece of tracking to the other team, and to which (added with v8) | blue: given to Barcelona; red: given to Athletic |
| **v9: seen pass** | the passes v9 can snap a restart to: a player next to the ball, the ball leaving him fast. Drawn over the 0.4 s it takes to see it (added with v9) | blue: Barcelona's pass; red: Athletic's |

They're computed by running v9 when the cache is built (`python -m viewer.build_cache`, ~1 min), so
they show what the model sees, not the labels it ends up with.

**The dots are the repaired tracking** (since v8): a keeper tracked as the other team is drawn in
its own team's colour, as a new slot after the existing ones (slot 46 on our match: Barça's keeper,
frames 184,439–190,198). Everything else is the tracking as delivered.

## The bench matches (added 2026-10-30)

The header's title is a **match picker**: our match, plus the 10 bench matches (7 IDSSE, 3 Metrica
samples). A bench match opens at `http://localhost:8765/?match=<id>`.

- **No video:** those matches have none. The page runs on its own clock, and play, pause, jump,
  slow-down before a change and notes all work as before.
- **Their true labels**, in the lane "truth (bench)". On other matches we may look at the answers
  freely (RULES.md: the ban is on our match). So this is where to watch our models fail against the
  truth.
- **Models:** v1, v3 and the floor computed on the damaged tracking; v4 (= v5 there, no audio), v6
  and v7 **out of sample**, each match predicted by models trained without it.
- **Signals:** "ball visible", v1's clues, v4's P(dead), v6's restart clue, v7's PathCRF estimate.
  No v8 lane: the keeper repair never fires on the bench, so v8 is v7 there.
  There's no Smart Tagging or whistle on those matches.
- **"clean tracking" toggle:** switches the dots between the damaged tracking the models see (41 %
  ball missing, anonymous slots) and the tracking as published.
- **Team names** are "home" and "away", and each match keeps its own notes.

**Build:** `python -m bench.eval_pathcrf` (saves the out-of-sample predictions,
`data/external/bench_predictions/`), then `python -m viewer.build_bench_cache` (~20 s, ~450 MB in
`viewer/cache/bench/`, gitignored).

## Watching changes closely

The bar under the header (**Watch changes in**) picks which label rows to watch: truth and our models
by default, baselines optional. Then choose what happens as one of them is about to change label:

- **slow down**: playback drops to 0.1× (or 0.25× / 0.5×), starting 2 s before the change (1–5 s
  selectable), and stays slow until 1 s after it.
- **slow down and pause**: the same, and it also stops 6 frames (0.24 s) before the change. Press
  play to go through it in slow motion. It pauses once per change, and again if you jump back.
- **do nothing**: plays at your chosen speed.

Two more options in the same bar:

- **flash on the change frame** (on by default): on the exact frame where a watched row changes
  label, the video gets a yellow and white ring and a tag `change at this frame (15835): v2 A→D`.
  The ring lasts that one frame only, so stepping with `←` `→` shows exactly where the change sits.
  The tag stays for 1 s afterwards (`changed 3 frames ago …`), so a change isn't missed at normal speed.
- **frame colour from**: paints a ring around the video in the label colour of the chosen row at
  the current frame (blue Barcelona, red Athletic, grey dead ball). Off by default.

On the right, the bar shows the next watched change and what it is, for example
`next change in 1.4 s · v2 A→D`. `N` / `P` jump to 2 s (the lead time) before the next or previous
watched change, so pressing play runs straight into it.

## Review mode and notes

`R` (or **Review mode**) shows just the video, as large as fits, with the label strip and the note
box underneath. `F` is fullscreen. Both work together.

**Notes** are for writing down what you see: pick a kind (observation, tracking problem,
whistle / stop, restart, model question), type, and **Add note here** pins it to the current frame.
`M` jumps straight to the note box. Notes appear as white marks on the strip and the whole-match
timeline, and in a list (click to jump, delete to remove).

**The line we don't cross:** notes are for understanding. RULES.md bans hand labels on this match
in any amount and for any purpose, including comparing models against them. So a note is never
"H/A/D at this frame", and no model code reads the notes. They're saved in
`viewer/notes/notes.json`, which is gitignored.

## Keys

`Space` play/pause · `←` `→` one frame · with `Shift` one second · with `Alt` ten seconds ·
`[` `]` zoom · `N` / `P` next / previous watched change · `R` review mode · `F` fullscreen ·
`M` write a note · `J` jump · `Esc` leave a text box. The jump box takes a frame number, or a match
time like `1h 12:30` or `2h 67:05`.

## How the tracking is drawn on the video

The homography file has one 3×3 matrix per frame that maps a pitch position (the same 0–1
numbers as the tracking) to a pixel in the 1920×1080 picture. For each player:

1. take their position (x, y) and multiply the vector (x, y, 1) by that frame's matrix
2. divide the first two results by the third: that is the pixel
3. scale to the size the video is shown at and draw a dot there

Same maths as the organisers' `starter/load_homography.py`. The dot is redrawn on every frame the
video actually shows (`requestVideoFrameCallback`), so it can't drift from the picture.

## How it's built

| file | role |
|---|---|
| `viewer/build_cache.py` | turns tracking, homography, labels and v1's clues into compact binary files |
| `viewer/server.py` | standard-library web server; adds the byte-range support a browser needs to seek in the 5 GB video |
| `viewer/static/` | the page: plain HTML, CSS and JavaScript, no build step |

New models appear automatically: every CSV in `evaluation/results/` becomes a label lane after
`--rebuild`. Clue lanes are v1's, plus **v2: whistle**: orange is a whistle v2 counts (its 3-second
push towards dead), grey is a whistle it ignores because the players didn't slow down. Decoding the
audio makes the rebuild take ~20 s.

## Checked on 2026-10-30

- Playback stays in sync: 2 s of play advanced 49 frames, and the data frame matched the video frame.
- Dots land on the players at the kick-off and in open play.
- `N` from the kick-off jumps to frame 11223, where v1 changes from D to A.
