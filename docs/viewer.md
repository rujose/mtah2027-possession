# The match viewer

A local web app that plays the tactical video with the tracking drawn on top, a 2D pitch map, and
timelines of every model's labels and v1's clues, all locked to the same frame. It's for seeing
*why* a model got a moment wrong, not for scoring.

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
| **pitch map** | the same frame from above, in metres. Ball trail over the last second. When the ball is lost, a dashed circle marks where it was last seen and how long ago. Bottom-left says which way Barcelona attacks in this half. |
| **labels at this frame** | the true label (first 5 minutes only), v1, and the four baselines. |
| **v1 clues** | which of v1's clues are on at this frame, the ball's speed, and the distance from the ball to each team's nearest player. |
| **Smart Tagging** | every tag whose window contains this frame. |
| **whole match** | every lane over the full video. Grey bands are not graded (before kick-off, half-time, after the whistle). The white line is the current frame, the light box is the detail window. Click to jump. |
| **detail** | the same lanes around the current frame (±30 s by default, zoom with `[` `]`). Click to jump. |

Lane colours: blue = Barcelona / H, red = Athletic / A, grey = dead ball / D, orange = a clue is on,
green = ball tracked.

## Keys

`Space` play/pause · `←` `→` one frame · with `Shift` one second · with `Alt` ten seconds ·
`[` `]` zoom the detail · `N` / `P` next / previous change in v1 · `J` jump. The jump box takes a
frame number, or a match time like `1h 12:30` or `2h 67:05`.

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
`--rebuild`. Clue lanes are v1's for now.

## Checked on 2026-10-30

- Playback stays in sync: 2 s of play advanced 49 frames, and the data frame matched the video frame.
- Dots land on the players at the kick-off and in open play.
- `N` from the kick-off jumps to frame 11223, where v1 changes from D to A.
