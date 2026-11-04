# Lab candidate `dead`: missed stoppages found by a second dead-ball classifier

| | |
|---|---|
| **status** | **built 2026-11-04, lab round 2 (on the mix); clears the bar on `all`**: held out spellF1 0.6278 against the mix's 0.6156 (**+0.0121**), better on **10/10**. **On `--set idsse` it fails one check**: +0.0122, 6/7, but its one PathCRF-unseen match (J03WR9) loses 0.0017 |
| **what** | idea 10f. A per-frame classifier, trained on the other bench matches, says which frames *inside* the predicted possessions are dead. It reads v4's P(dead) with context plus what v4 lacks: deceleration, the players' accelerations and how coherently each team moves, and the ball (still or moving, seen outside and how long ago, how long unseen, players closing on it or not, time since a touch, where it lies). Every stretch it is confident about, long enough, becomes D |
| **stage** | `post`: on any input letters (judged on the mix's out-of-fold letters, `lab_mix.kept_letters`) |
| **code** | `bench/lab_dead.py` (new file only; nothing else changed) |
| **run** | `python -m bench.lab_dead` (~10 min for both sets, ~0.7 GB; features cached in `data/external/lab_dead/`); `--explore` (~1 min, what the missed stoppages look like and the ceiling); `--nested [--set idsse]` (~9 / ~4 min, the stricter check) |
| **results** | `bench/results/lab_dead[_idsse].json` (the judge's report, held-out AUCs, what each fold wrote), `lab_dead_grid[_idsse].csv` (every match × setting), `lab_dead_nested[_idsse].json` |
| **mix with** | `model = lab_dead.fit(others, letters_of, features="ball")`, then `lab_dead.apply(match, letters, model, thr=0.4, min_len=2.0)` (see [For the mix](#for-the-mix)) |
| **new dependencies** | none (scikit-learn's `HistGradientBoostingClassifier`, already installed) |

## Results (summary)

| | **all 10, held out** | IDSSE only (`--set idsse`, 7) |
|---|---|---|
| **spellF1** | **0.6278 against the mix's 0.6156 (+0.0121), better on 10/10** | 0.6123 against 0.6001 (+0.0122), better on 6/7 |
| IDSSE | 0.6090 against 0.5967 (+0.0123), 7/7 | (as above) |
| Metrica | 0.6717 against 0.6599 (+0.0118), 3/3 | — |
| **PathCRF-unseen** (J03WR9, metrica1–3) | 0.6597 against 0.6499 (+0.0098), 4/4 | **J03WR9: 0.6206 against 0.6223 (−0.0017)** |
| macroF1 | **0.8919 against 0.8775 (+0.0144)** | 0.8971 against 0.8816 |
| D's F1 | 0.9328 against 0.9131 | 0.9514 against 0.9311 |
| **chosen on all the matches** | **features `ball`, thr 0.4, min_len 2 s** | features `ball`, thr 0.3, min_len 4 s |
| **the bar** | mean ✓, share 10/10 ✓, each source ✓, unseen ✓: **passes** | mean ✓, share 6/7 ✓, unseen ✗: **fails** |
| nested check | **+0.0120, 10/10** (IDSSE +0.0120, Metrica +0.0118, unseen +0.0104): passes | +0.0111, 6/7, J03WR9 −0.0017: fails the same way |

- **Every fold picked the `ball` features**, at thr 0.4 (8 folds) or 0.5 (2), min_len 2 s.
- **What it writes** (held out, all 10): 183 stretches, **161 (88 %) mostly dead in the truth**;
  28k frames changed, 88 % of them to the truth's label. Of the 158 stoppages the mix missed,
  **102 get a D**, 27 of the 45 that had merged two possessions of the same team.
- **The players-only features pass too, but small**: +0.0029, 7/10 (judged over that half of the
  grid alone). The ball carries most of it.

## Why

**The error budget on the mix** ([../lab.md](../lab.md), "What's left"): missed stoppages are worth
+0.022 if every true D run with no predicted D were written in; false stoppages only +0.004. Round
1's `stoppage` page found that v4's P(dead) can't find them: inside them its median is 0.21–0.45.

### What the missed stoppages look like (`--explore`, on the kept letters; truth read, nothing picked)

- **158 over the 10 matches (11–27 per match, 16 a match)**, of ~100 true stoppages each.
- **Not short:** length quantiles 10 / 25 / 50 / 75 / 90 % = 1.9 / 3.4 / 5.1 / 8.0 / 11.6 s.
- **72 % lie inside one predicted possession** (one team throughout); 28 % contain a predicted team
  change. **45 (28 %) are between two true possessions of the same team**: the mix merges those two.
- **Far from any predicted D:** essentially all their frames are a team's and at least 1 s from a D
  in the letters (what this stage calls *eligible*).
- **v4's P(dead) inside them:** median 0.32 (IDSSE) and 0.33 (Metrica).
- **The ball:** on IDSSE it's **seen on 98 %** of their frames (median), inside the pitch; on Metrica
  on 1 % (blanked). The ball is seen outside somewhere inside 48 % (IDSSE) and 55 % (Metrica) of them.

So on IDSSE these are stoppages where the ball is visible and players keep moving, which v4 (no
ball by design) can't see.

### How each source's damaged ball behaves (all frames)

| source | ball seen, dead | seen, live | seen outside the pitch when seen, dead | live |
|---|---|---|---|---|
| IDSSE (7) | 44.6 % | 70.6 % | 51.5 % | 0.4 % |
| Metrica (3) | **2.9 %** | 89.3 % | 13.3 % | 1.2 % |

- **Metrica's samples blank the ball while it's dead**, so on them "no ball" is almost the truth
  itself. Our match is Metrica-tracked but not blanked like that (our ball is missing on 83 % of
  frames where players stand still and 29.5 % where they move, [../ideas.md](../ideas.md) #2).
- **IDSSE's damaged ball is the honest version**: the damage (`bench/degrade.py`) loses the ball by
  crowd size, players about to stop and closeness to a line, as measured on our tracking, **never by
  the truth's labels**; and the clean IDSSE ball is tracked through dead time.
- **The ball lying still is physical, not a placeholder**: steps under 1 cm between frames on 32 %
  of seen dead frames against 0.5–1 % of live ones (J03WMX, J03WOH; metrica1 27 % against 0.9 %),
  never exactly 0.
- **Hence the `ball` features are trained on IDSSE matches only**, as idea #2 asks ("teach the
  dead-ball model how much 'no ball' means on IDSSE damaged like ours, not on Metrica's blanked
  samples"). A Metrica match held out is predicted by a model that never saw Metrica.

### The ceiling (`--explore`: the missed stoppages written D with the truth's own boundaries)

| written | gain | better |
|---|---|---|
| every missed stoppage | +0.0224 | 10/10 |
| only those between two possessions of the same team | +0.0090 | 10/10 |
| only those between different teams | +0.0133 | 10/10 |
| only those of 2 s or more | +0.0217 | 10/10 |
| only those of 4 s or more | +0.0168 | 10/10 |
| each 0.5 s short at both ends | +0.0188 | 10/10 |
| each 1 s short at both ends | +0.0155 | 10/10 |
| each 2 s short at both ends | +0.0088 | 10/10 |

Fuzzy boundaries keep most of the value (1 s short each end: 70 % of it), so a per-frame classifier
with a smoothed threshold is a fair tool; and the short ones (under 2 s) are worth little.

## What it does

### 1. Eligible frames

A frame is **eligible** when the input letters give it to a team and it is at least **CLEAR = 25
frames (1 s)** from any D in the letters (same half). The classifier is trained and applied only
there: the inside of predicted possessions, where the missed stoppages are. Edges of stoppages the
letters already have are the `stoppage` stage's job.

### 2. Features (prediction-time inputs only)

From the match's damaged tracking and v4's out-of-sample P(dead). All order-free and symmetric
(no slot identity, home and away swappable), like v4's.

**Set `players`** (38 features; trained on IDSSE and Metrica):

| feature | what | why |
|---|---|---|
| `p_dead` | v4's P(dead) | the existing dead-ball evidence (posture, set-piece layouts) |
| `speed_mean`, `share_slow`, `share_fast` | players' mean speed (positions over 0.16 s, as v4); share under 1 and over 4 m/s | as v4, to combine with the new ones |
| `accel_median` | median over players of the velocity change over 0.4 s (m/s²) | live play is reacting (turns, starts, stops); in a stoppage players cruise or walk |
| `team_speed_max`, `team_speed_min` | speed of each team's mean velocity (the block shifting) | Jose: in live play "opponents keep shifting" (idea #6) |
| `coherence_max`, `coherence_min` | per team, length of the mean heading of players moving over 1 m/s (≥ 3 of them) | a team shifting together against players wandering |
| context | means over the past and the next 2 s and 5 s of `p_dead`, `speed_mean`, `share_slow`, `share_fast`, `accel_median`, `team_speed_max`, `coherence_max` | a stoppage is a stretch, not a frame |
| `speed_drop` | mean speed over the past 5 s minus the next 5 s | **deceleration** (idea #5): "they keep jogging", but slower |

**Set `ball`** (81 features: the above plus 43; trained on IDSSE only):

| feature | what | why |
|---|---|---|
| `ball_seen`, `since_seen`, `until_seen` | seen now; s since / until seen (capped 30 s) | "no ball" as IDSSE's damage teaches it (idea #2) |
| `ball_speed`, `since_moving`, `until_moving` | the ball's speed; s since / until it was last seen moving over 1 m/s (capped 10 s) | a ball lying still for seconds is a free kick being set, not play |
| `ball_out`, `since_out`, `until_out` | seen outside the pitch; s since / until (capped 30 s) | the ball went out (a quick throw-in the letters ran through) |
| `line_dist`, `corner_dist`, `in_six`, `spot_dist` | from the last seen position: to the nearest line (negative outside), to a corner, inside a 6-yard box, to the centre spot | where restarts are taken |
| `near_dist`, `near_count` | nearest player to the ball; players within 5 m | live play crowds the ball |
| `closing` | mean closing speed on the ball of its 3 nearest players (relative velocity toward it) | Jose's idea #1: in play the nearest players close in; dead, they don't |
| `follow` | cosine between the players' mean velocity and the ball's (ball moving) | idea #1/#6: the team moves with the ball |
| `since_touch`, `until_touch` | s since / until a seen touch by either team (v11's detector, ΔV 3 m/s; capped 30 s) | no touches for a while |
| context | past / next 2 s and 5 s of `ball_seen`, `ball_speed`, `ball_out`, `near_dist`, `closing`, `follow` | as above |

**An illustration** (one match, J03WMX, eligible frames, medians dead against live): ball speed 0.09
against 6.0 m/s; since the ball moved 8.2 against 0 s; since a touch 9.2 against 1.0 s; closing
−0.13 against +0.54 m/s; players' acceleration 0.63 against 0.99 m/s²; P(dead) 0.26 against 0.01.

### 3. The classifier

- **Target:** the truth says D, on the eligible frames of the other matches' input letters (the
  mix's out-of-fold letters, so it learns what the mix misses), every 5th frame (v4's subsample):
  ~12–20 k rows a match, **3–8 % dead**.
- `HistGradientBoostingClassifier`: 200 rounds, rate 0.1, 15 leaves, at least 100 rows per leaf,
  L2 1.0, no early stopping, seed 0. No class weights, so P stays a probability. Not tuned.
- **Held-out AUC on the eligible frames** (each match, trained without it):

| match | v4's P(dead) | `players` | `ball` |
|---|---|---|---|
| J03WMX | 0.916 | 0.957 | 0.998 |
| J03WN1 | 0.835 | 0.941 | 0.991 |
| J03WOH | 0.923 | 0.964 | 0.994 |
| J03WOY | 0.899 | 0.955 | 0.993 |
| J03WPY | 0.928 | 0.978 | 0.997 |
| J03WQQ | 0.902 | 0.953 | 0.991 |
| J03WR9 | 0.925 | 0.958 | 0.998 |
| metrica1 | 0.824 | 0.908 | 0.974 |
| metrica2 | 0.894 | 0.933 | 0.982 |
| metrica3 | 0.874 | 0.932 | 0.980 |
| **mean** | **0.892** | **0.948** | **0.990** |

The `ball` model never saw a Metrica match and still reaches 0.97–0.98 there (helped by the
blanking, see failure 1).

### 4. The action

1. P on every eligible frame, smoothed by a centred mean over **SMOOTH = 25 frames** (1 s; eligible
   frames only).
2. **A stretch** = a maximal run of eligible frames with smoothed P > `thr`, at least `min_len` s.
   It becomes D. Nothing else changes: the team on either side stays, no D is removed.
3. **Slivers:** a possession that the cut leaves shorter than **SLIVER = 20 frames** (and was longer
   before) goes to D too: the decoder's minimum possession, so the stage never leaves a possession
   the decoder couldn't have made. Possessions that were already short (the `short` stage's) are
   left alone.

### Constants and the grid

| setting | value | why |
|---|---|---|
| velocities | over 0.16 s | v4's speed span, so speeds mean the same as in its features |
| acceleration span | 0.4 s | long enough to be above position jitter, short enough for a turn |
| moving | 1 m/s | below it a player's heading (and the ball's course) is jitter |
| coherence needs | 3 moving players | fewer and the mean heading is noise |
| nearest players | 3 | the ones contesting the ball |
| close | 5 m | "near the ball", the challenge distance (`learned`'s CLOSE) |
| corner / centre spot | 3 m / 2 m | v4's corner distance; a ball placed on the spot |
| touch ΔV | 3 m/s | v11's, both sets |
| caps | 10 s (ball moved), 30 s (seen, outside, touch) | past them nothing more is said; keeps the features bounded |
| context | 2 s and 5 s | stoppages are 5 s at the median; v4 found ±4 s mattered most |
| CLEAR | 1 s | stoppage starts and ends after round 1 sit mostly within ±1 s of the truth; beyond that a dead frame belongs to a stoppage the letters missed |
| STEP | every 5th frame | v4's; neighbours are near-copies |
| SMOOTH | 1 s | so a stretch doesn't flicker into pieces |
| SLIVER | 20 frames | v3's minimum possession |
| **features** | **`players`, `ball`** | players-only is safe on any ball; `ball` adds the ball, trained on IDSSE only (above) |
| **thr** | **0.7, 0.6, 0.5, 0.4, 0.3** | the metric's arithmetic: a right split gains about 1.4/(P+G), a wrong one costs 0.6–2/(P+G), so break-even is near 0.3–0.5; up to 0.7 for caution |
| **min_len** | **4, 2, 1 s** | missed stoppages: 10 % under 1.9 s, median 5 s; the ceiling says the short ones are worth little |

31 settings: the no-op first (ties keep the mix), then `players` before `ball`, each from the most
cautious (highest threshold, longest stretch) down. **All fixed before any held-out run; nothing
was changed after.**

## Results

### All 10 matches (`python -m bench.lab_dead`)

| match | source | picked (features, thr, min_len) | the mix | held out | Δ | stretches (mostly dead) | missed stoppages (found; same-team) | D's F1 |
|---|---|---|---|---|---|---|---|---|
| J03WMX | IDSSE | ball, 0.4, 2 | 0.5025 | 0.5172 | +0.0147 | 19 (17) | 12 (11; 1) | 0.924 → 0.943 |
| J03WN1 | IDSSE | ball, 0.4, 2 | 0.5770 | 0.5817 | +0.0048 | 14 (11) | 11 (8; 1) | 0.946 → 0.952 |
| J03WOH | IDSSE | ball, 0.4, 2 | 0.5992 | 0.6119 | +0.0127 | 23 (21) | 17 (10; 7) | 0.924 → 0.955 |
| J03WOY | IDSSE | ball, 0.4, 2 | 0.6277 | 0.6413 | +0.0136 | 23 (23) | 16 (12; 4) | 0.923 → 0.945 |
| J03WPY | IDSSE | ball, 0.5, 2 | 0.6640 | 0.6804 | +0.0164 | 15 (13) | 15 (11; 4) | 0.947 → 0.963 |
| J03WQQ | IDSSE | ball, 0.4, 2 | 0.5863 | 0.6063 | +0.0199 | 23 (21) | 14 (12; 6) | 0.919 → 0.947 |
| J03WR9 | IDSSE, unseen | ball, 0.4, 2 | 0.6202 | 0.6239 | +0.0037 | 22 (17) | 11 (11; 3) | 0.933 → 0.958 |
| metrica1 | Metrica, unseen | ball, 0.4, 2 | 0.6662 | 0.6715 | +0.0052 | 11 (9) | 15 (6; 1) | 0.893 → 0.903 |
| metrica2 | Metrica, unseen | ball, 0.5, 2 | 0.6671 | 0.6840 | +0.0168 | 13 (12) | 20 (9; 0) | 0.860 → 0.872 |
| metrica3 | Metrica, unseen | ball, 0.4, 2 | 0.6463 | 0.6596 | +0.0133 | 20 (17) | 27 (12; 0) | 0.862 → 0.892 |
| **mean** | | | **0.6156** | **0.6278** | **+0.0121, 10/10** | **183 (161: 88 %)** | **158 (102; 27 of 45)** | **0.913 → 0.933** |

- **Bar:** mean ✓, 10/10 ✓, IDSSE +0.0123 ✓ and Metrica +0.0118 ✓, unseen +0.0098 ✓. **Passes.**
- **macroF1 0.8775 → 0.8919 (+0.0144)**, the tie-break metric, from D's F1.
- **Against the ceiling:** +0.0121 of the +0.0224 (54 %).
- **Possessions** 302 → 315 a match (true 411): the splits add some back.
- Of the 27,918 frames changed, 24,705 (88 %) are D in the truth.

**Means over the 10 matches, by setting** (not held out; condensed):

| features | thr 0.7 (4 / 2 / 1 s) | 0.6 | 0.5 | 0.4 | 0.3 |
|---|---|---|---|---|---|
| no-op | 0.6156 | | | | |
| `players` | 0.6180 / 0.6173 / 0.6173 | 0.6177 / **0.6192** / 0.6154 | 0.6188 / 0.6176 / 0.6113 | 0.6178 / 0.6142 / 0.6052 | 0.6168 / 0.6101 / 0.5995 |
| `ball` | 0.6251 / 0.6260 / 0.6251 | 0.6258 / 0.6265 / 0.6256 | 0.6259 / 0.6281 / 0.6259 | 0.6264 / **0.6282** / 0.6256 | 0.6270 / 0.6272 / 0.6244 |

**`ball` is flat**: every one of its 15 settings is between +0.009 and +0.013, so the gain doesn't
hang on the threshold. `players` peaks at +0.0036 and drops fast when it writes more (its stretches
are mostly dead 48–80 % of the time, by setting, against 81–97 % for `ball`).

**Each feature set alone** (the judge over the no-op plus only that set's settings; diagnostics,
the verdict is the whole grid's):

| | held out | better | IDSSE / Metrica / unseen | chosen on all | bar |
|---|---|---|---|---|---|
| `players` alone | 0.6186 (+0.0029) | 7/10 | +0.0022 / +0.0047 / +0.0008 | players, 0.6, 2 s | passes (narrowly) |
| `ball` alone (= the candidate) | 0.6278 (+0.0121) | 10/10 | +0.0123 / +0.0118 / +0.0098 | ball, 0.4, 2 s | passes |

### IDSSE only (`--set idsse`)

Trained on the other 6 IDSSE matches; the mix is the IDSSE-only chain (`lab_mix_base_idsse.json`).

| match | picked | the mix | held out | Δ | stretches (mostly dead) | missed (found; same-team) |
|---|---|---|---|---|---|---|
| J03WMX | ball, 0.3, 4 | 0.5148 | 0.5276 | +0.0128 | 11 (11) | 11 (8; 1) |
| J03WN1 | ball, 0.3, 4 | 0.5757 | 0.5797 | +0.0040 | 13 (10) | 13 (8; 2) |
| J03WOH | ball, 0.3, 4 | 0.5953 | 0.6107 | +0.0154 | 15 (15) | 17 (10; 8) |
| J03WOY | ball, 0.3, 4 | 0.6534 | 0.6675 | +0.0141 | 17 (17) | 16 (8; 2) |
| J03WPY | ball, 0.3, 4 | 0.6686 | 0.6827 | +0.0141 | 14 (13) | 12 (9; 3) |
| J03WQQ | ball, 0.3, 4 | 0.5704 | 0.5974 | +0.0269 | 20 (18) | 15 (11; 5) |
| J03WR9 (unseen) | ball, 0.3, 4 | 0.6223 | 0.6206 | **−0.0017** | 11 (9) | 8 (6; 3) |
| **mean** | | **0.6001** | **0.6123** | **+0.0122, 6/7** | **101 (93: 92 %)** | **92 (60; 24)** |

- **Bar:** mean ✓, 6/7 ✓, **unseen ✗** (J03WR9, the set's only PathCRF-unseen match, −0.0017).
  **Fails.**
- macroF1 0.8816 → 0.8971; D's F1 0.9311 → 0.9514.
- On J03WR9 the `ball` settings range from −0.014 (thr 0.3, 1 s) to +0.004 (thr 0.7, 2 s); the
  picked one (0.3, 4 s) loses 0.0017. The IDSSE-only mix already finds more of its
  stoppages (8 missed, against 11 in the `all` mix), and 2 of its 11 stretches are wrong.

### The stricter check (`--nested`)

`lab.judge` picks match m's setting from the other matches' scores, and each of their classifiers
was trained with m among its training matches. `--nested` removes that: for each m and each other
match j, the classifiers are trained without m and j, j is scored for every setting, and m gets the
setting best over those scores; m's own score is the main run's (trained without m). Same grid.

| | nested | judge (above) |
|---|---|---|
| **all 10** | **0.6276 against the mix's 0.6156 (+0.0120), better on 10/10** | +0.0121, 10/10 |
| IDSSE | 0.6087 against 0.5967 (+0.0120), 7/7 | +0.0123, 7/7 |
| Metrica | 0.6717 against 0.6599 (+0.0118), 3/3 | +0.0118, 3/3 |
| PathCRF-unseen | 0.6604 against 0.6499 (+0.0104), 4/4 | +0.0098, 4/4 |
| picks | `ball` on every fold; thr 0.3 (4 folds), 0.4 (3), 0.5 (3); min_len 2 s (7), 4 s (3) | `ball`, 0.4–0.5, 2 s |

**The same as the judge**: it would pass the bar too. The folds spread over more thresholds than in
the judge, which is the flat in-sample table again: any `ball` setting does about as well.

With `--set idsse` (a diagnostic; the verdict there is the judge's): **0.6111 against 0.6001
(+0.0111), better on 6/7; J03WR9 again −0.0017** (it picks thr 0.3, 4 s, as in the judge), so the
unseen check fails the same way. Picks: `ball` on every fold, thr 0.3–0.7, min_len 2 or 4 s.

## Failure modes

1. **The ball on our match.** The `ball` model learned from IDSSE's damaged ball, which the bench
   built to look like ours (missing 30 % in live play, more when players stop), and it transfers to
   Metrica's matches (+0.012, 3/3) without seeing one. But Metrica's samples blank the ball in dead
   time, so there "unseen for long" is a stronger dead sign than it will be on our match; their gain
   may flatter. If our tracker reports the ball through stoppages, `since_moving`, `ball_out` and
   `closing` carry it (as on IDSSE); if it loses it, the model falls back on `since_seen` and the
   players. The players-only set (+0.0029, 7/10) is the floor if the ball turns out to mean
   something else on ours.
2. **J03WR9, the IDSSE match PathCRF never saw, gains least** (+0.0037 on `all`, −0.0017 on `idsse`).
   It has the fewest missed stoppages (8–11) and its wrong stretches (5 of 22, 2 of 11) cost about
   what the right ones gain. Where the letters already miss little, the stage has little to gain
   and the same rate of wrong stretches.
3. **12 % of stretches are wrong** (22 of 183; less than half of their frames dead in the truth).
   Not inspected one by one; slow live play with the ball still (a keeper holding it, a player
   shielding it) is the likely kind. Each one splits a possession into two.
4. **It only adds D.** False stoppages (+0.004 in the budget) are untouched; removing D the model
   rejects wasn't built.
5. **Still missed:** 56 of 158 stoppages get no D, proportionally more on Metrica, where the ball
   is blank in dead time (27 of 62 found, against 75 of 96 on IDSSE).
6. **A stretch doesn't snap to the ball.** Its edges come from the smoothed probability (±1 s).
   Running round 1's `stoppage` stage again after this one would snap the new team → D changes to
   seen exits; not tried here (one change at a time).
7. **Second-order leaks, as in every round-2 candidate:** the training matches' kept letters and
   their v4 P(dead) were made by models that had the held-out match among their training matches.
   The nested check covers the choice of setting, not these.

## Verdict

**Keep for the `all` chain:** it clears every check of the bar on the 10 matches (+0.0121, 10/10,
both sources and the unseen matches up), every fold picks the `ball` features, the gain is flat
across that set's thresholds, and macroF1 rises by +0.014. **It doesn't clear the bar on `--set
idsse`**: the mean gain is the same (+0.0122, 6/7) but its single unseen match loses 0.0017, so for
an IDSSE-only chain it should stay out (or be judged again there with the cautious settings, as a
new round).

## For the mix

```python
from bench import lab_dead
model = lab_dead.fit(others, letters_of=lambda o: letters_before[o.id], features="ball")  # IDSSE matches only are used
new = lab_dead.apply(match, letters_before[match.id], model, thr=0.4, min_len=2.0)
```

- `fit(matches, letters_of=None, features="players"|"ball", **_) -> model`: `matches` are lab match
  objects; `letters_of(match)` their input letters out of fold (default: the mix's kept letters of
  the `all` set; pass it explicitly for `idsse`).
  With `features="ball"` only the IDSSE matches among them are used.
- `apply(match, letters, model, thr=1.0, min_len=4.0, **_) -> letters`; thr 1.0 is the no-op.
- **Where:** after the mix (stoppage → short → segments → **dead**), which is where it was judged.
  Placed earlier, it must be re-fitted on that stage's letters, and `segments` after it would see
  new D neighbours.
- **For our match:** train on the 7 IDSSE bench matches with their kept letters; thr 0.4, min_len 2 s
  (the setting chosen on all 10).
