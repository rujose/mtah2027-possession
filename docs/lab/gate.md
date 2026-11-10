# Lab candidate `gate`: a per-match switch between v13 and `augment`

| | |
|---|---|
| **status** | **built 2026-11-09, lab round 7 (a switch). The switch clears all four parts of the bar on both sets.** **Part 1** (normal bench): A on 17 of 17 matches (`all` 0.7162, `idsse` 0.6760: ±0). **Part 2** (stressed bench): B on 10/10 and 7/7, **+0.1075** (0.5982 → 0.7057) and **+0.0573** (0.6117 → 0.6690). **Part 3** (10 / 25 / 50 % stressed): within 0.005 of the better branch at every level, **narrowly**: `all` −0.0042 / −0.0046 / ±0, `idsse` ±0 / −0.0036 / ±0. **Part 4** (SkillCorner): B on 20/20, **0.412 against A's 0.319, +0.093, 20/20 better**. `signal` reads 0.02–0.10 on the normal bench, 0.36–0.63 stressed, 0.25–0.49 on SkillCorner; `THRESHOLD = 0.1445`. Nothing here reads our match: which branch it would get is the open question |
| **what** | a switch (round 7, [../lab.md](../lab.md) "A switch"): per match, the program picks **branch A** (v13: the temporal `average` start, then v12's post stages) or **branch B** (`augment`'s chain, [carried.md](carried.md)) from **one number it computes on the tracking and v12's inputs alone**. B when the number is above `THRESHOLD`, else A |
| **why** | A is the best chain on the bench (0.7162 / 0.6760, [temporal.md](temporal.md)) but shares robust's blind spot: a dead ball that stays tracked and moving (SkillCorner: v12 0.308; the stressed bench: v12 −0.127). B reads that ball as dead (stressed bench 0.706, SkillCorner 0.412) but costs ~0.01 to ~0.02 on the normal bench. Neither wins in both worlds; a switch could, if the tracking can say which world a match is in |
| **stage** | before everything: one number per match picks which chain runs |
| **code** | `bench/lab_gate.py` (new; reads the lab's, lab_temporal's and lab_carried's caches; changes nothing). `signal(track, pred, halves, pitch)` and `THRESHOLD` for `possession/v13.py` |
| **run** | `python -m bench.lab_gate --prepare --set all` / `idsse` (~3 min each) → `OMP_NUM_THREADS=4 … --fit --set …` (~20 min each) → `… --gru --set …` (~5 min each) → `--signals` (~2 min) → `--chains normal|p10|p25|p50|stressed --set …` (~3 min each, both branches) → `--sc-signals` → `--skillcorner` (~1 min a match, plus ~5 min once for A's retrained post stages) → `--report` |
| **results** | `bench/results/lab_gate.json` (every part of the bar), `lab_gate_signals.csv` (the signal on every bench match, variant and set), `lab_gate_switch.csv` (every match × variant × set: signal, left-out threshold, pick, A, B, the switch), `lab_gate_chains_<variant>[_idsse].json` (both branches, every stage, per match), `lab_gate_skillcorner.csv` (branch A on SkillCorner, every stage), `lab_gate_skillcorner_signals.csv`, `lab_gate_skillcorner_switch.csv`; caches in `data/external/lab_gate[_idsse]/` (gitignored) |
| **data** | the stoppages transplanted and the 20-match test: **SkillCorner** open data, [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT licence. Thanks to SkillCorner |

## Why

Round 7 left two chains that each win in one world and lose in the other ([../lab.md](../lab.md)):

| | normal bench, `all` / `idsse` | stressed bench, `all` / `idsse` | SkillCorner |
|---|---|---|---|
| **A** (v13: `average` → short → segments → timing / → recall → dead) | **0.7162 / 0.6760** | 0.5982 / 0.6117 | 0.319 |
| **B** (`augment` → the same post stages) | 0.6938 / 0.6603 | **0.7057 / 0.6690** | **0.412** |

(A's stressed and SkillCorner numbers are this page's; the rest are [temporal.md](temporal.md)'s and [carried.md](carried.md)'s, reproduced here.)

Jose approved v13 plus an automatic switch: per match the program computes one number on the tracking alone and
runs A or B. The bar ([../lab.md](../lab.md), round 7, "A switch", fixed before this ran), against branch A:

1. **the normal bench, both sets**: it picks A on every match, or else loses no more than 0.002;
2. **the stressed bench, both sets**: it picks B on at least 9 of 10 matches (read here as 90 %: 7 of 7 on `idsse`),
   a gain of at least +0.02;
3. **partly stressed benches** (10 %, 25 % and 50 % of the true stoppages stressed): at every level, within 0.005 of
   the better branch's mean, on both sets;
4. **SkillCorner's 20 matches**: a gain of at least +0.02, 15 of 20 better.

## The bench variants

Five versions of each bench match, both sets (`all`, 10 matches; `idsse`, 7). The players, the truth and every
players-only input (v4's P(dead), v6's restart P, PathCRF) are the same in all five; only the ball inside chosen true
stoppages changes, and with it v6's / v7's costs, robust's blanking and its 97 features (recomputed, as
`lab_carried.stressed` does).

| variant | the ball in the true stoppages |
|---|---|
| `normal` | the lab's bench (IDSSE's frozen dead ball, Metrica's missing one) |
| `p10`, `p25`, `p50` | 10 %, 25 %, 50 % of each match's stressed stoppages (by count; 90–111 a match, 98.7 on average) take `lab_carried`'s test-draw ball (`stressed_ball(m, TEST_SEED)`: SkillCorner's odd-indexed matches' stoppages, seed 0); the others keep the normal ball |
| `stressed` | all of them (`lab_carried.stressed`) |

The partial draw: one random order of each match's stressed stretches (`numpy` generator seeded by 2027 and the
match's name); `p10` takes the first 10 %, `p25` the first 25 %, `p50` the first 50 %, so each level contains the one
below it. Only the chosen stretches' frames change: a partial ball is the normal ball with the full stress's frames
copied in on the chosen stretches, so the recipe and the templates are `lab_carried`'s, untouched. Robust's features
change on more frames than the ball does (J03WMX at 10 %: 4,664 ball frames, 13,290 feature rows), since its features
look up to a few seconds around each frame: they were recomputed, not spliced.

## The two branches, held out, on every variant

Every number is out of fold, as the lab's: each match's start comes from models trained without it, and each post
stage is refitted on the other matches' letters **of the same branch on the same variant**, with v12's fold setting
for that match (`lab_mix.add_stage`, `lab_mix.fold_settings`, as `lab_temporal.chains` and `lab_carried.chains`).

- **A**: `average` = the mean of the GRU's P and robust's held-out P, clipped and rounded; decoded with `average`'s own
  fold pick (`lab_temporal_average[_idsse].json`), v9's snap, v11's move; then v12's post stages. On a stressed or
  partly stressed variant the GRU is the lab's own fold network (`data/external/lab_temporal[_idsse]/net_<match>.pt`;
  Metrica's 3 on `all` share `net_metrica1.pt`, trained on all 7 IDSSE matches, as in `lab_temporal`) with its fold's
  normaliser (`lab_temporal.normaliser(used, which)`), run on that variant's robust features; robust's P is its fold
  model's on the same features.
- **B**: `augment`'s fold model's P on the variant's features, decoded with robust's fold pick, then v12's post stages
  (`lab_carried.chains`).
- **Robust's and augment's fold models were refitted** (`lab_carried.fit_robust`, same rows, trees and seed:
  `lab_carried` saved only their P) to get their P on the partly stressed features. The refit reproduces the saved P
  on the normal and the stressed bench on every match, both kinds, both sets (largest difference 3·10⁻⁸, the float32
  file's rounding; `refit_check.csv`).
- **The GRU reproduces `lab_temporal`'s saved P** on the normal bench on every match (largest difference 3·10⁻⁸, the
  float32 file's rounding of a 0.001 grid; `gru_check.csv`).
- **The chains reproduce the lab's**: A on the normal bench gives `lab_temporal_chains`' `average` on every match (to
  5·10⁻⁶, that file's rounding: 0.7162 / 0.6760); B gives `lab_carried_chains`' `augment` exactly on the normal and the
  stressed bench (0.6938 / 0.6603 and 0.7057 / 0.6690).

## The signal (fixed before any branch was scored, and before it was computed on SkillCorner)

**Where the players say the ball is dead, how often is the ball seen and moving, against how often it is where they
say it's live.** For one match, from the repaired tracking and v12's inputs only (no truth, no labels):

    dead frames  = frames where v4's players-only P(dead) >= 0.5, the last 2 s of each such run left out
    live frames  = frames where v4's P(dead) < 0.2
    seen-moving  = the ball tracked, not blanked by robust's still rule (still >= 1 s), and moving >= 0.3 m/s over 0.2 s
    signal       = share of dead frames with a seen-moving ball / share of live frames with a seen-moving ball

(inside the halves, on frames with at least one player tracked; 0.0 if either part is empty). Near 0 when a dead
ball is missing or frozen: robust's blanking removes the frozen one, so in the normal bench's stoppages the ball is
almost never "seen and moving", and what's left comes from live frames v4 calls dead. Toward 0.5–0.6 when the dead
ball is tracked and moving through stoppages as on SkillCorner. Dividing by the live share takes out the feed's own
ball coverage (IDSSE's live ball is seen-moving 63–72 % of the time, Metrica's 90–92 %).

| setting | value | fixed | why |
|---|---|---|---|
| the witness | v4's players-only P(dead) (`pred["p_dead"]`) | before looking | the only dead-ball evidence v13 has that never reads the ball: the question is what the ball does where the players say dead |
| dead at | P(dead) ≥ 0.5 | ~19:25, from the bench variants' distributions | "more likely dead than not"; 31–52 % of in-half frames on the bench |
| trim | the last 2 s of each dead run | ~19:25, from the bench variants' distributions | the restart kick and the first touches fall inside v4's dead run while it still reads a set piece: a live, moving ball the players' model calls dead (the reasoning; not checked frame by frame) |
| live below | P(dead) < 0.2 | ~19:25 | the match's own baseline for how often its ball is seen and moving in play |
| seen and moving | tracked, not blanked by `lab_robust.still_frames` (still ≥ 1 s), ≥ `lab_carried.MOVING` (0.3 m/s) over `LOOK` (0.2 s) | before looking | the ball robust can actually read as moving: the same definitions `lab_carried` and `lab_skillcorner` used for SkillCorner's dead-time ball |
| players tracked | frames with at least one player | ~19:30 (an amendment, still before any branch score and before SkillCorner's signal) | a frame with no player gives v4 nothing to read (a broadcast's cut-aways: 26 % of SkillCorner's frames, [skillcorner.md](skillcorner.md)). Every in-half bench frame has players: the bench's signals didn't move (largest change 0.0) |

**How it was chosen.** On the five bench variants' signals only (no branch score, no SkillCorner): the brief's
candidate (P(dead) ≥ 0.5, the whole run) and variants of it, 125 in all (P(dead) at 0.5–0.9, runs of at least 0–10 s,
0–5 s trimmed at the end), each read as how well it orders adjacent stress levels across matches (the pooled AUC of
normal against 10 %, 10 % against 25 %, 25 % against 50 %). All ordered the levels well (mean AUC 0.904–0.988). The
brief's candidate scored 0.960: its Metrica normal matches read 0.10–0.145 and overlapped IDSSE's 10 % level.
Trimming the last 2 s of each dead run took a third to a half off Metrica's normal readings (to 0.05–0.10) and gave
0.983; the best of the 125 (0.988) adds P(dead) ≥ 0.6 and a 5 s minimum run, not worth two more settings for 0.005.
Every one of the 125 rises with the stress level on every match.

## The threshold rule (fixed with the signal, before any branch was scored)

**B where the match's signal is above the signal level at which B starts beating A.** Per set, leave one match out:
on the other matches, the mean B − A at each stress level (0, 10, 25, 50, 100 %) and the mean signal at each level;
the first level where the mean B − A turns positive, and the one before it; the threshold is the mean signal
interpolated linearly between those two levels to where B − A crosses 0. The held-out match takes B if its signal is
above it. **`THRESHOLD`** (what v13 uses) is the same rule on all 10 matches of `all`, v13's bench set.

Why this rule and not the brief's other example (the midpoint between the normal bench's largest signal and the
stressed bench's smallest): the gap is wide (0.100 to 0.360 on `all`), but its midpoint (0.23) sits near the 50 %
level. What was known before this ran (A ahead by 0.022 / 0.016 on the normal bench; B ahead of v12 by 0.13 / 0.07
on the stressed one, [temporal.md](temporal.md), [carried.md](carried.md)) put the crossing nearer 15–20 %, so the
midpoint would likely have handed part 3 to A at 25 %. The crossing rule puts the switch where it costs least on
either side. It reads branch scores, but only the other matches': the held-out match's own score never moves its
threshold (but see "What was approximated", 5).

## The signal's distributions

On every bench match, variant and set (`lab_gate_signals.csv`), then on SkillCorner's 20 (computed once, after
`THRESHOLD` was written):

| | IDSSE (7), `all` | Metrica (3), `all` | IDSSE set (7) |
|---|---|---|---|
| **normal** | 0.024–0.045 (median 0.032) | **0.052–0.100** (0.070) | 0.023–0.048 (0.031) |
| 10 % stressed | 0.057–0.121 (0.089) | 0.104–0.137 (0.107) | 0.062–0.127 (0.086) |
| 25 % | 0.130–0.212 (0.146) | 0.137–0.177 (0.138) | 0.131–0.219 (0.145) |
| 50 % | 0.240–0.356 (0.282) | 0.224–0.242 (0.234) | 0.242–0.346 (0.288) |
| **stressed** | 0.486–0.625 (0.545) | **0.360–0.492** (0.421) | 0.493–0.623 (0.550) |
| `THRESHOLD` | 0.1445 | | (the rule on `idsse`: 0.172) |

| SkillCorner (20) | min | 25 % | median | 75 % | max |
|---|---|---|---|---|---|
| signal | **0.254** | 0.334 | 0.340 | 0.384 | 0.488 |

- **The signal rises with the stress on every match** (strictly, all five levels, both sets) and its gap is wide at
  the ends: the normal bench's largest is 0.100 (metrica1), the stressed bench's smallest 0.360 (metrica2).
- **Metrica reads higher on the normal bench** (0.05–0.10 against IDSSE's 0.02–0.05): v4 calls more of Metrica's live,
  moving-ball frames dead. And it rises more slowly with the stress (stressed 0.36–0.49 against 0.49–0.63), so the two
  sources overlap from 10 % on.
- **SkillCorner reads 0.25–0.49**, between the bench's 50 % and fully stressed levels, as its dead ball (tracked on
  47 % of dead frames, moving on 88 % of those) says it should; every one of the 20 is above `THRESHOLD` by at least
  0.11.

## Results (2026-11-09)

### The threshold rule's output

| | `all` | `idsse` |
|---|---|---|
| mean B − A at 0 / 10 / 25 / 50 / 100 % (all matches) | −0.0223 / −0.0133 / **+0.0045** / +0.0410 / +0.1075 | −0.0157 / −0.0093 / −0.0013 / **+0.0194** / +0.0573 |
| mean signal at those levels | 0.045 / 0.098 / 0.160 / 0.275 / 0.515 | 0.033 / 0.089 / 0.164 / 0.294 / 0.556 |
| the crossing (all matches) | between 10 % and 25 %, at ~21 %: **0.1445** (`THRESHOLD`) | between 25 % and 50 %, at ~26 %: 0.172 |
| left-out thresholds (one per match) | 0.132–0.164 (median 0.143) | 0.135–0.195 (median 0.178) |

### The bar, every part

| part | `all` | `idsse` | verdict |
|---|---|---|---|
| **1. normal bench**: A everywhere, or loses ≤ 0.002 | **A on 10/10** (signals 0.024–0.100 against left-out thresholds 0.132–0.164): 0.7162, ±0 | **A on 7/7** (0.023–0.048 against 0.135–0.195): 0.6760, ±0 | **passes** |
| **2. stressed bench**: B on ≥ 90 %, gain ≥ +0.02 | **B on 10/10**: 0.7057 against A's 0.5982, **+0.1075** (10/10 better; IDSSE +0.114, Metrica +0.093, unseen +0.094) | **B on 7/7**: 0.6690 against 0.6117, **+0.0573** (7/7) | **passes** |
| **3. partly stressed**: within 0.005 of the better branch at each level | 10 %: **−0.0042** (A better; B picked on 1, metrica1); 25 %: **−0.0046** (B better by 0.0045; B on 4); 50 %: ±0 (B on 10/10) | 10 %: ±0 (A on 7/7); 25 %: **−0.0036** (A better by 0.0013; B on 3); 50 %: ±0 (B on 7/7) | **passes, narrowly** (the worst: −0.0046 against −0.005) |
| **4. SkillCorner**: gain ≥ +0.02, 15 of 20 better | **B on 20/20**: 0.412 against A's 0.319, **+0.093, 20/20 better** (with post stages retrained on each branch's own letters: 0.408 against 0.312, +0.095, 20/20) | | **passes** |

### Every variant, both branches and the switch

`all` (10 matches; each match's pick with its own left-out threshold):

| variant | A | B | B − A | switch | switch − A | switch − the better | B picked | better / worse than A |
|---|---|---|---|---|---|---|---|---|
| normal | **0.7162** | 0.6938 | −0.0223 | 0.7162 | ±0 | ±0 | 0/10 | 0 / 0 |
| 10 % | **0.7054** | 0.6921 | −0.0133 | 0.7012 | −0.0042 | −0.0042 | 1/10 | 0 / 1 |
| 25 % | 0.6913 | **0.6958** | +0.0045 | 0.6912 | −0.0001 | −0.0046 | 4/10 | 2 / 2 |
| 50 % | 0.6573 | **0.6983** | +0.0410 | 0.6983 | +0.0410 | ±0 | 10/10 | 10 / 0 |
| stressed | 0.5982 | **0.7057** | +0.1075 | 0.7057 | +0.1075 | ±0 | 10/10 | 10 / 0 |

`idsse` (7 matches):

| variant | A | B | B − A | switch | switch − A | switch − the better | B picked | better / worse than A |
|---|---|---|---|---|---|---|---|---|
| normal | **0.6760** | 0.6603 | −0.0157 | 0.6760 | ±0 | ±0 | 0/7 | 0 / 0 |
| 10 % | **0.6702** | 0.6608 | −0.0093 | 0.6702 | ±0 | ±0 | 0/7 | 0 / 0 |
| 25 % | **0.6646** | 0.6633 | −0.0013 | 0.6610 | −0.0036 | −0.0036 | 3/7 | 1 / 2 |
| 50 % | 0.6458 | **0.6652** | +0.0194 | 0.6652 | +0.0194 | ±0 | 7/7 | 7 / 0 |
| stressed | 0.6117 | **0.6690** | +0.0573 | 0.6690 | +0.0573 | ±0 | 7/7 | 7 / 0 |

A on the stressed bench (0.5982 / 0.6117) sits a little above v12's (0.5761 / 0.6023, [carried.md](carried.md)): the
GRU doesn't fix robust's blind spot, as [temporal.md](temporal.md) expected, it only carries its own gain into it.

Per match, `all`: the signal, then the pick (and the switch against A where it picked B); `t` is the match's left-out
threshold:

| match (`t`) | normal | 10 % | 25 % | 50 % | stressed |
|---|---|---|---|---|---|
| J03WMX (0.136) | 0.041 A | 0.121 A | 0.212 **B** (-0.006) | 0.306 **B** (+0.036) | 0.545 **B** (+0.101) |
| J03WN1 (0.147) | 0.032 A | 0.073 A | 0.130 A | 0.255 **B** (+0.046) | 0.576 **B** (+0.136) |
| J03WOH (0.164) | 0.034 A | 0.089 A | 0.146 A | 0.282 **B** (+0.087) | 0.518 **B** (+0.147) |
| J03WOY (0.147) | 0.028 A | 0.094 A | 0.190 **B** (+0.019) | 0.345 **B** (+0.058) | 0.616 **B** (+0.135) |
| J03WPY (0.154) | 0.025 A | 0.079 A | 0.144 A | 0.261 **B** (+0.046) | 0.507 **B** (+0.106) |
| J03WQQ (0.142) | 0.045 A | 0.118 A | 0.188 **B** (+0.001) | 0.356 **B** (+0.026) | 0.625 **B** (+0.076) |
| J03WR9 (0.142) | 0.024 A | 0.057 A | 0.139 A | 0.240 **B** (+0.041) | 0.486 **B** (+0.097) |
| metrica1 (0.132) | 0.100 A | 0.137 **B** (-0.042) | 0.177 **B** (-0.016) | 0.242 **B** (+0.022) | 0.421 **B** (+0.114) |
| metrica2 (0.144) | 0.070 A | 0.107 A | 0.137 A | 0.224 **B** (+0.035) | 0.360 **B** (+0.086) |
| metrica3 (0.139) | 0.052 A | 0.104 A | 0.138 A | 0.234 **B** (+0.012) | 0.492 **B** (+0.079) |

Per match, `idsse`:

| match (`t`) | normal | 10 % | 25 % | 50 % | stressed |
|---|---|---|---|---|---|
| J03WMX (0.151) | 0.048 A | 0.127 A | 0.219 **B** (-0.012) | 0.315 **B** (+0.014) | 0.550 **B** (+0.066) |
| J03WN1 (0.195) | 0.033 A | 0.075 A | 0.131 A | 0.256 **B** (+0.045) | 0.580 **B** (+0.073) |
| J03WOH (0.172) | 0.031 A | 0.086 A | 0.145 A | 0.288 **B** (+0.013) | 0.520 **B** (+0.031) |
| J03WOY (0.183) | 0.028 A | 0.097 A | 0.193 **B** (+0.018) | 0.346 **B** (+0.019) | 0.623 **B** (+0.072) |
| J03WPY (0.184) | 0.023 A | 0.075 A | 0.141 A | 0.266 **B** (+0.025) | 0.512 **B** (+0.055) |
| J03WQQ (0.135) | 0.038 A | 0.102 A | 0.177 **B** (-0.031) | 0.346 **B** (+0.001) | 0.613 **B** (+0.050) |
| J03WR9 (0.178) | 0.030 A | 0.062 A | 0.143 A | 0.242 **B** (+0.019) | 0.493 **B** (+0.053) |

Both branches per match on every variant are in `lab_gate_switch.csv` and `lab_gate_chains_<variant>[_idsse].json`.

### Where the two sources cross (information, read after the judging)

| B − A, `all` | normal | 10 % | 25 % | 50 % | stressed |
|---|---|---|---|---|---|
| IDSSE's 7 | −0.0167 | −0.0071 | **+0.0110** | +0.0486 | +0.1139 |
| Metrica's 3 | **−0.0356** | −0.0278 | −0.0108 | **+0.0231** | +0.0927 |

**On Metrica, B costs twice as much on the normal bench and starts winning later** (between 25 % and 50 %, at a signal
of about 0.18; IDSSE's crossing is at about 0.12). The pooled threshold sits between the two. Both of the switch's
losses on `all` are Metrica's: metrica1 at 10 % (signal 0.137 against its left-out 0.132: B, −0.042) and at 25 %
(−0.016). Our match is Metrica-tracked. This was not part of the bar and nothing was changed for it.

### SkillCorner, match by match

Branch A as v13 would run on a new match (below, "What was approximated"); B is `lab_carried`'s `augment` row; the
switch picks B on all 20, so it scores B's.

| match | signal | pick | v12 | A | B (the switch) | switch − A | A retrained | B retrained |
|---|---|---|---|---|---|---|---|---|
| 2017461 | 0.339 | B | 0.254 | 0.262 | 0.344 | +0.082 | 0.263 | 0.336 |
| 1996436 | 0.377 | B | 0.257 | 0.267 | 0.384 | +0.117 | 0.264 | 0.385 |
| 2010085 | 0.336 | B | 0.283 | 0.297 | 0.394 | +0.096 | 0.291 | 0.397 |
| 2007448 | 0.403 | B | 0.285 | 0.290 | 0.402 | +0.113 | 0.288 | 0.395 |
| 2015213 | 0.488 | B | 0.286 | 0.299 | 0.420 | +0.121 | 0.293 | 0.424 |
| 1953632 | 0.341 | B | 0.291 | 0.304 | 0.392 | +0.088 | 0.298 | 0.391 |
| 2011166 | 0.296 | B | 0.293 | 0.309 | 0.384 | +0.076 | 0.298 | 0.379 |
| 1959846 | 0.334 | B | 0.296 | 0.301 | 0.384 | +0.084 | 0.294 | 0.389 |
| 2006229 | 0.337 | B | 0.299 | 0.305 | 0.412 | +0.107 | 0.296 | 0.404 |
| 2013725 | 0.438 | B | 0.304 | 0.315 | 0.411 | +0.096 | 0.311 | 0.407 |
| 1886347 | 0.318 | B | 0.310 | 0.316 | 0.440 | +0.124 | 0.307 | 0.428 |
| 1996435 | 0.366 | B | 0.311 | 0.312 | 0.395 | +0.083 | 0.308 | 0.393 |
| 1986691 | 0.376 | B | 0.316 | 0.337 | 0.419 | +0.082 | 0.329 | 0.413 |
| 1899585 | 0.254 | B | 0.319 | 0.343 | 0.414 | +0.072 | 0.332 | 0.402 |
| 1925299 | 0.420 | B | 0.320 | 0.344 | 0.405 | +0.062 | 0.331 | 0.401 |
| 2006363 | 0.369 | B | 0.324 | 0.335 | 0.426 | +0.091 | 0.329 | 0.424 |
| 2007721 | 0.428 | B | 0.336 | 0.346 | 0.460 | +0.114 | 0.343 | 0.451 |
| 2016236 | 0.335 | B | 0.342 | 0.339 | 0.417 | +0.078 | 0.331 | 0.411 |
| 1874553 | 0.332 | B | 0.342 | 0.349 | 0.456 | +0.107 | 0.347 | 0.455 |
| 1927964 | 0.329 | B | 0.383 | 0.411 | 0.473 | +0.062 | 0.394 | 0.469 |
| **mean** | 0.361 | B 20/20 | 0.308 | **0.319** | **0.412** | **+0.093** (20/20) | 0.312 | 0.408 |

### How narrow part 3 is (information, read after the judging)

The same picks with one fixed threshold for every match instead of the left-out ones:

| fixed threshold | normal (`all` / `idsse`) | 10 % | 25 % | 50 % | passes everywhere |
|---|---|---|---|---|---|
| 0.10 | **−0.0039** (metrica1 to B) / ±0 | **−0.0105** / **−0.0057** | ±0 / −0.0013 | ±0 / ±0 | no |
| 0.12 | ±0 / ±0 | **−0.0062** / −0.0018 | ±0 / −0.0013 | ±0 / ±0 | no |
| **0.1445** (`THRESHOLD`) | ±0 / ±0 | ±0 / ±0 | +0.0004 / −0.0042 | ±0 / ±0 | yes |
| 0.17 | ±0 / ±0 | ±0 / ±0 | −0.0046 / −0.0036 | ±0 / ±0 | yes |
| 0.20 | ±0 / ±0 | ±0 / ±0 | −0.0050 / −0.0017 | ±0 / ±0 | yes (at the edge) |
| the midpoint rule (0.230 / 0.270) | ±0 / ±0 | ±0 / ±0 | −0.0045 / ±0 | −0.0035 / **−0.0127** | no |

Figures are the switch against the better branch's mean. A window from about 0.12 to 0.20 passes on both sets;
`THRESHOLD` is near its lower third. Below it the switch starts taking Metrica's 10 % matches to B; above it, B's 25 %
gains slip away. Parts 1 and 2 hold for any threshold from just above 0.100 (metrica1's normal signal) to 0.36
(metrica2's stressed one).

## Verdict

**The switch clears all four parts of the bar on both sets.** It picks A on every normal match (17 of 17; the largest
normal signal is 0.100 against thresholds of 0.132 and up), B on every stressed match (17 of 17, +0.108 / +0.057), lands
within 0.005 of the better branch at 10, 25 and 50 % (worst −0.0046), and on SkillCorner picks B on all 20: 0.412
against v13 alone's 0.319 (+0.093, 20/20).

What to weigh before using it:

1. **Part 3 is narrow**, as it must be near a crossing: −0.0042 and −0.0046 on `all` against a −0.005 floor. A
   switch can't beat the better branch where the two are close; it can only avoid losing much. Its two losses on
   `all` are one Metrica match picked B at 10 % (−0.042) and a 25 % level where the two branches differ by 0.0045.
2. **The two worlds are far apart, so the switch rarely has to be precise.** Normal matches read 0.02–0.10;
   SkillCorner and the stressed bench read 0.25–0.63. The hard cases are partial worlds (10–25 % of stoppages with a
   moving ball), where the two branches' means are within 0.013 of each other (single matches up to 0.05).
3. **Metrica's own crossing is later than the pooled one** (above): a Metrica-like match whose dead ball moves through a
   quarter of its stoppages would be switched to B and lose about 0.01; one with no moving dead ball reads 0.05–0.10
   and stays on A. Our match is Metrica-tracked.
4. **Part 4 is not independent of the design.** The stress transplants SkillCorner's own dead-time ball, and the
   signal was designed to read the stress. Nothing was tuned on SkillCorner's signal or scores (it was computed once,
   after `THRESHOLD`), but a test whose recipe came from the same 20 matches is a kind test.

## Using it

```python
from bench.lab_gate import signal, THRESHOLD, picks_b
fixed, pred = v12.inputs(tracking, halves, pitch, "all")      # v13 has these at run time
s = signal(fixed, pred, halves, pitch)                         # pure: no cache, no ids, no truth
branch = "B" if picks_b(s) else "A"                            # B when s > THRESHOLD (0.1445)
```

- `signal(track, pred, halves, pitch=None) -> float` reads `track["ball"]`, `track["players"]` and `pred["p_dead"]`
  only; `signal_parts(track, pred, halves)` returns its three frame masks (players say dead, players say live, ball
  seen and moving) for a look inside.
- `THRESHOLD = 0.1445`: the rule's output on all 10 matches of `all` (0.14448, rounded), v13's bench set. On the
  `idsse` set the same rule gives 0.172 (information: v13 runs on `all`).
- Branch A on a new match is v13 itself; branch B is `augment`'s deployment model
  (`lab_carried.deployment_augment()`) with `lab_carried.deploy_chain` (v12's deployment post stages, or
  `lab_carried.deploy_post("augment")`'s retrained ones).
- Nothing here has been run on our match, and nothing here reads its signal.

## What was fixed when (2026-11-09)

| time | what |
|---|---|
| ~19:15 | the partial benches: levels 10 / 25 / 50 %, by count of stressed stretches, one nested random order per match (seed 2027), `lab_carried`'s test-draw ball; before any variant was built |
| ~19:25 | the signal (P(dead) ≥ 0.5, the last 2 s trimmed, live below 0.2, seen and moving as `lab_carried`) and the threshold rule (left-out crossing), from the signal's distributions on the bench variants only: no branch score, no SkillCorner |
| ~19:30 | the amendment: frames with at least one player tracked (the bench's signals unchanged to 0.0) |
| 19:27–19:52 | both branches on the five variants of `all`; ~19:40 branch A on SkillCorner (its scores printed then; nothing was chosen after) |
| ~19:53 | `THRESHOLD = 0.1445` written into `bench/lab_gate.py` from the rule on `all` |
| ~19:54 | the signal on SkillCorner's 20, for the first time |
| 19:52–20:08 | both branches on the five variants of `idsse`; the report |

Nothing was changed after any score was seen.

## What was approximated

1. **Branch A on SkillCorner is v13 as the lab can build it, not v13's port.** The GRU is `lab_temporal`'s
   `net_metrica1.pt` (trained on all 7 IDSSE matches, float32, 2 threads) with the normaliser of those 7; robust is
   v12's deployment model (its P reproduces `lab_skillcorner`'s cached P to 3·10⁻⁸ on all 20); `average` is decoded with
   its setting chosen on all 10 (team 2, dead 25, minimum 10), then v9's snap and v11's move; the post stages are
   v12's deployment models (the main rows) or post stages retrained on `average`'s out-of-fold letters, as
   `lab_carried.deploy_post` does for `augment` ("retrained": out of fold with the chosen settings 0.7195 on the
   bench; on SkillCorner 0.312 against 0.319). `possession/v13.py` and `bench/train_v13.py` (another builder) may differ
   (deployment networks retrained, float64); the switch's verdict on SkillCorner has 0.06 to spare on the worst match.
2. **Branch B on SkillCorner is `lab_carried`'s rows** (`lab_carried_skillcorner.csv`, `augment` and `augment
   retrained`), not recomputed: the same deployment pipeline (`lab_carried.deploy_chain`), v12 reproduced exactly there.
3. **The signal on SkillCorner** reads `lab_carried`'s cache of v12's inputs on SkillCorner's tracking
   (`data/external/lab_carried/skillcorner/`: `v12.inputs`, PathCRF included), exactly what v13 would compute.
4. **PathCRF's P on the stressed and partly stressed benches is the normal bench's** (as in `lab_carried`): it reads
   the players only, but answers only on v6's live stretches, which the new ball can move. It touches both branches.
5. **The left-out threshold is not fully nested.** A held-out match's threshold uses the other matches' chain scores,
   and their post stages were trained with the held-out match among their training matches. Its truth reaches its own
   threshold only through nine other matches' post-stage models; a fully nested run would retrain every chain per
   match (10× the time).

## Caveats

1. **The partial benches are `lab_carried`'s recipe at partial doses.** The signal's scale (0.03 normal, ~0.15 at the
   crossing, ~0.5 stressed) is the recipe's: SkillCorner's dead ball laid on the bench's restart spots. A real feed
   whose dead ball is tracked less often, or moves more slowly, would read lower for the same harm to robust.
2. **The signal is only as good as v4's players-only P(dead).** Its baseline on the normal bench is mostly live,
   moving-ball frames v4 calls dead (more on Metrica: 0.05–0.10 against 0.02–0.05). On a feed where v4 is worse, the
   normal reading rises toward the threshold.
3. **It reads one thing: a seen, moving ball where the players stand.** A failure robust has for another reason (a
   different damage pattern, a feed whose ball is never blanked because it jitters) is not what it measures.
4. **Both branches keep their own caveats**: `augment` trains on the recipe the stressed bench tests
   ([carried.md](carried.md), caveat 3); A's margin on `idsse` is thin ([temporal.md](temporal.md)). And every number
   here is at the bench's 41 % damage, as every lab candidate's.
5. **Many tries, one bench.** 125 versions of the signal were looked at (distributions only) before one was fixed;
   the threshold rule was fixed with it. Part 3's margins (−0.0042 / −0.0046 / −0.0036 against −0.005) are the kind a
   different draw of the partial stoppages could move.

## The open question: how our match's stoppage ball behaves

Nobody will read the signal on our match, so this page can't say which branch it would pick there. What the bench
says about the two possibilities:

- **If our dead ball is mostly missing or still** (Metrica's normal bench; our ball is missing 83 % of the time the
  players stand still, [../bench.md](../bench.md)), the signal should read like Metrica's normal matches, 0.05–0.10,
  below `THRESHOLD` (0.1445) by 0.04 to 0.09: A, v13 as it is.
- **If our seen dead ball moves through stoppages** (carried, thrown back, interpolated through them), it reads like
  the partly or fully stressed bench and the switch hands it to B, which is worth +0.02 to +0.11 on the bench
  at 50 % or more, and +0.09 on SkillCorner.
- **The uncomfortable middle**: a Metrica-like match with a moving dead ball in about a quarter of its stoppages
  reads 0.14–0.18, where the pooled threshold switches it to B while Metrica's own crossing (~0.18) says A was
  still slightly better (about −0.01 at 25 %). The bench can't say how likely that middle is for our match.

`python -m bench.transfer_check --ours` (Jose's to run) reads our match's dead share label-free; this page adds
nothing that reads our match.
