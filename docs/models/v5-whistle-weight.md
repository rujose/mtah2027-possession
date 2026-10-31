# Model v5: v4 with the referee's whistle given back its weight

| | |
|---|---|
| **status** | built 2026-10-30, as option A of the proposal on the [v4 page](v4-dead-ball.md#proposal-2026-10-30-not-built-give-the-whistle-back-its-weight), approved by Jose |
| **change from v4** | one number: the counted-whistle weight, from 2.0 to **2.0 + 0.5 × −log(0.01) = 4.30** per frame for 3 s, written as that formula |
| **code** | `possession/v5.py`; `possession/v4.py` gained a hook so v5 reuses its pipeline (v4's output is unchanged) |
| **run** | `python -m possession.v5` → `evaluation/results/v5.csv` (~1 min) |
| **output** | SHA-256 `2a7201b1f3b182789dd98394fda5b4d7876d8d8542a72f17febdd09a56e34100`; byte-identical on re-run |

## Why

Jose watched the two counted whistles that v4 no longer followed with a stoppage (frames 23697 and
54740): both were real stoppages.

The cause, from v4's own output: there the dead-ball model sits at its floor (P = 0.01: confident
play goes on). At that point a frame called dead costs 0.5 × −log(0.01) = 2.30, while the whistle
only adds 2.0 against live. So the whistle lost on every frame of its window.

## The change, and where its value comes from

**The whistle keeps v2's weight on top of the strongest "live" evidence the model can give:**

    W_WHISTLE = v2's 2.0 + W_DEAD_BALL × −log(lower clip of P) = 2.0 + 0.5 × 4.605 = 4.303

- **What it guarantees:** within its 3 s, a counted whistle beats the model and both switch prices,
  as in v2. Even at the model's floor, (4.30 − 2.30) × 75 frames = 150, more than "into + out of D"
  (2 × 50 = 100).
- **Reasoning, not fitting.** The value follows from the costs, applies the same way to all 11
  counted whistles, and was set before looking at what it does.
- **Not the reason for accepting it:** "it fixes 23697 and 54740". That would turn Jose's two
  observations into hand labels used for validation, which RULES.md and #10 rule out.
- **No bench number exists**, because the bench has no audio. As with v2, the change rests on its
  reasoning, and the counts below need no labels.

## Results on our match (no ground truth involved)

| | v4 | v5 |
|---|---|---|
| counted whistles followed by dead ball within 3 s | 8 / 11 | **11 / 11** |
| any whistle followed by dead ball within 3 s | 71 / 86 | 74 / 86 |
| stoppages | 74 (median 26.5 s) | 77 (median 25.4 s) |
| dead ball, share of graded frames | 46.2 % | 46.3 % |
| possessions | 428 | 426 |
| frames that differ | — | 488 (0.31 % of graded) |

Dead frames in the 30 s after each counted whistle:

| whistle | 15835 | 17113 | 23697 | 31173 | 54740 | 74237 | 88695 | 118563 | 148019 | 172217 | 183807 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v4 | 581 | 726 | **0** | 200 | **0** | 750 | 750 | 750 | 137 | 726 | 499 |
| v5 | 581 | 734 | **75** | 200 | **75** | 750 | 750 | 750 | 212 | 726 | 499 |

**As expected, the two lost stoppages are now exactly 3 s long.** Once the whistle's window ends,
the model's "live" evidence ends the stoppage. That splits the possessions at the right moment but
restarts play too early: the end of stoppages is step 3's job (restart shapes).

**Public 5 minutes (a pipeline check, not a ranking):** spellF1 0.442, as v4. macroF1 0.818 against
0.819, from 8 more dead frames after the whistle at 17113. Not used to choose anything.
