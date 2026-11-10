# Lab candidate `carried`: dead-ball calls that survive a ball carried through a stoppage

| | |
|---|---|
| **status** | **built 2026-11-09, lab round 7 (a robustness candidate). Neither candidate clears the robustness bar; v12 stays.** Both **fail part 1** (the normal bench): `players` −0.0158 (1/10), IDSSE set −0.0153 (0/7); `augment` −0.0095 (1/10), IDSSE set −0.0095 (1/7); every source and the unseen matches down by more than 0.005. Both **pass parts 2 and 3 by far**: on the stressed bench `augment` +0.1297 (10/10) / +0.0668 (7/7), `players` +0.0715 (10/10) / +0.0324 (7/7); on SkillCorner `augment` **0.412** (+0.104, 20/20; above v11's 0.366 on all 20), `players` 0.391 (+0.084, 20/20). **The stressed bench costs v12 −0.127** (0.7033 → 0.5761; IDSSE set −0.068, 0.6699 → 0.6023), 0/10 |
| **what** | a robustness candidate (round 7). **A stressed bench** (a test): each bench match with SkillCorner's own dead-time ball transplanted into its true stoppages. **Two candidates**, each as the start of v12's chain: **`players`**, robust's P(dead) floored by v4's players-only P(dead); **`augment`**, robust retrained with the stress (another draw) put into its training matches |
| **why** | [skillcorner.md](skillcorner.md), "What explains the gap" §1: in SkillCorner's stoppages the ball is tracked on 47 % of dead frames, moving 88 % of that time; robust gives P(dead) 0.19 there and v12 calls 19 % of those frames D (v11 82 %); with the truth's stoppages v12 would score 0.473 instead of 0.308. The bench can't show it (IDSSE freezes its dead ball and robust blanks it; Metrica blanks it), so robust learned "dead ⇔ no ball" ([robust.md](robust.md), case B; [v12](../models/v12-lab-chain.md), failure mode 1) |
| **stage** | `decode`: a new start (robust's P changed before the decoder), then v12's post stages refitted on its output |
| **code** | `bench/lab_carried.py` (new; reads the lab's caches, SkillCorner's cache and its cached v12 run; changes nothing) |
| **run** | `python -m bench.lab_carried --measure-sc` (~10 s) → `--stress --set all` / `--set idsse` (~1 min each) → `OMP_NUM_THREADS=4 … --fit --set …` (~20 min each, < 1 GB) → `--grid --set …` (~10 min) → `--chains normal|stressed --set … [--starts …]` (~1.5 min a start) → `--sc-prepare` (~2.5 min a SkillCorner match: v12's inputs, PathCRF included) → `--deploy-post players|augment` (~5 min each) → `--skillcorner` → `--report` |
| **results** | `bench/results/lab_carried.json` (the bar, every part), `lab_carried_skillcorner_ball.json` (SkillCorner's dead-time ball, measured), `lab_carried_ball[_idsse].csv` (the stressed ball, checked), `lab_carried_players_grid[_idsse].csv` and `lab_carried_players[_idsse].json` (the floor's grid and picks), `lab_carried_chains_{normal,stressed}[_idsse].json` (every chain, every step, per match; `players_w0.5` is the post hoc row), `lab_carried_skillcorner.csv` (every SkillCorner match × start × stage); caches in `data/external/lab_carried[_idsse]/` (gitignored) |
| **data** | the stoppages transplanted and the 20-match test: **SkillCorner** open data, [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata), MIT licence. Thanks to SkillCorner |

## Why

v12 starts from `robust`, a per-frame P(H / A / D) whose dead-ball evidence is, above all, the missing ball
([robust.md](robust.md), "The cost of robustness"): blanked IDSSE dead time is 99.7 % "ball missing" against 30 % of
live time. On SkillCorner's broadcast tracking the ball is tracked through half of each stoppage and keeps moving
(fetched, carried, thrown back), and robust reads it as live. Our match is Metrica-tracked and its ball is missing 83 %
of the time players stand still ([../bench.md](../bench.md)), so "no ball" should mostly hold on it, but nobody can
look. This page asks for insurance: dead-ball calls that survive a tracked, moving ball **without losing on the
normal bench**. The bar is round 7's robustness bar ([../lab.md](../lab.md)), fixed before this candidate ran:

1. **the normal bench, both sets: no loss beyond noise**: the mean within 0.002 of v12's chain, neither source nor
   the unseen matches down by more than 0.005;
2. **the stressed bench, both sets: a gain of at least +0.02, 70 % of matches better**;
3. **SkillCorner's 20 matches: a gain of at least +0.02, 15 of 20 better.**

## The stressed bench (a test, fixed before anything was scored on it)

### SkillCorner's dead-time ball, measured (`--measure-sc`)

All 20 matches, every true stoppage (2,079; 104 a match), the ball as the SkillCorner loader gives it
(`bench/results/lab_carried_skillcorner_ball.json`). "Moving": ≥ 0.3 m/s over 0.2 s; "beyond a line": more than 0.5 m
outside; both as `lab_skillcorner.dead_diagnosis`, which these numbers reproduce (47 %, 88 %, 31 %).

| | SkillCorner, dead | SkillCorner, live |
|---|---|---|
| ball tracked (not a cut-away) | **46.5 %** | 98.3 % |
| … moving, of the tracked frames | **88.0 %** | 99.2 % |
| … beyond a line | **31.1 %** (by 0.7–3.3 m, 10–90 %) | 0.6 % |
| … in a still run ≥ 1 s (what robust's blanking removes) | **8.7 %** | 0.2 % |
| ball speed, 10 / 25 / 50 / 75 / 90 % | 0.27 / 0.59 / 1.28 / 2.81 / 5.74 m/s | median 5.5 m/s |
| nearest player to the ball, 10 / 25 / 50 / 75 / 90 % | 0.7 / 1.4 / 2.9 / 5.1 / 8.1 m | median 1.6 m |

**The shape of a stoppage** (what the recipe has to reproduce):

| | first 1 s | 1–2 s | 2–4 s | 4–8 s | 8–16 s | > 16 s |
|---|---|---|---|---|---|---|
| tracked, by time since the stoppage began | 98.5 % | 98.5 % | 98.0 % | 76 % | 36 % | 28 % |
| median speed (tracked) | 4.8 m/s | 2.9 | 1.7 | 1.2 | 1.1 | 0.7 |
| beyond a line (tracked) | 24 % | 38 % | 46 % | 42 % | 36 % | 13 % |

| s before the restart | 0.2 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| ball to the restart spot, median | 0.8 m | 1.4 m | 2.0 m | 2.9 m | 3.9 m | 4.9 m | 6.0 m |

| stoppage length | < 2 s | 2–5 s | 5–10 s | 10–20 s | 20–40 s | ≥ 40 s |
|---|---|---|---|---|---|---|
| stoppages | 69 | 101 | 301 | 565 | 677 | 366 |
| ball tracked | 93 % | 96 % | 98 % | 78 % | 48 % | 29 % |

| restart (ball at the first live frame) | touchline | field | goal area | corner | centre | not seen (cut-away) |
|---|---|---|---|---|---|---|
| stoppages | 540 | 910 | 92 | 113 | 63 | 361 |
| ball tracked | 90 % | 70 % | 53 % | 45 % | 33 % | 34 % |
| beyond a line (tracked) | 58 % | 18 % | 22 % | 34 % | 20 % | 30 % |

What it says: the ball is tracked through the first ~4 s of almost every stoppage (still rolling out: 4.8 m/s in the
first second), then the broadcast cuts away in the long ones (replays); it comes back before the restart and walks in
to the spot (2 m away 1 s before, 6 m 16 s before). It is **never still before the restart** (the still run that ends
a stoppage is 0 s at every quantile: SkillCorner's interpolated ball and a ball held for a throw-in both move), and it
is not at a player's feet (2.9 m from the nearest, median). So the brief's "carried, then still at the spot" is not
SkillCorner's shape; the recipe transplants SkillCorner's own stoppages instead of drawing one by hand.

### The recipe (`stress_ball`)

For each bench match (the lab cache: the damaged tracking at 41 %, as the models see it) and each **true** dead
stretch inside a half that ends in a restart:

1. **The restart spot**: the clean ball at the first live frame (within 1 s); the stretch's **kind** from it: corner
   (within 3 m of one), touchline (within 1.5 m of it or beyond), centre (within 2 m of the spot), goal area (within
   6.5 m of a goal line and 10 m of the line between the goals), field (the rest). The same rule on both sources.
2. **A SkillCorner template**: among SkillCorner stoppages of the same kind with a seen restart spot, the 5 nearest in
   length (log ratio), one drawn at random (`numpy` generator seeded by the draw and the match's name).
3. **Transplanted**: the template's ball relative to its own restart spot, laid on the bench's restart spot, mirrored
   on each axis where the two restarts are on opposite halves; time-stretched to the bench stretch's length (median
   ratio 1.00, 10–90 % 0.96–1.08); its untracked frames (cut-aways) stay missing.
4. **No jump at the start**: over the first 2 s (at most half the stretch) the path is blended from the clean ball's
   last live position, as a ball still rolling on would be.
5. **Live frames untouched**, the players untouched. 19 of the 1,006 true stretches (1.8 % of dead frames) end their
   half with no restart and keep the bench's own ball.
6. **Then as on any feed**: v1's ball clues and v6's / v7's costs recomputed on the new ball (`ball_costs`, which
   reproduces the bench's cached costs exactly on the unstressed ball); robust's still-ball blanking and its 97
   features recomputed on it (`stressed_features`). v4's P(dead), v6's restart P and PathCRF read the players only and
   are kept.

**Two draws**: the test (seed 0, templates from SkillCorner's odd-indexed 10 matches) and `augment`'s training (seed
1, the even-indexed 10), so the test bench never shows `augment` a stoppage it trained on.

### What the stressed ball looks like (`--stress`, before any score; truth only to find the stoppages)

| true dead frames | seen | moving (of seen) | beyond a line (of seen) | still ≥ 1 s (of seen) | missing after robust's blanking |
|---|---|---|---|---|---|
| **SkillCorner** | **46.5 %** | **88.0 %** | **31.1 %** | **8.7 %** | 57.6 % |
| IDSSE, normal bench | 44.6 % | 0.8 % | 31.4 % | 99.2 % | 99.7 % |
| Metrica, normal bench | 2.9 % | 37 % | 11 % | 42 % | 99.1 % |
| **IDSSE, stressed** | **50.4 %** | **87.2 %** | **37.2 %** | **8.7 %** | **54.0 %** |
| **Metrica, stressed** | **49.8 %** | **90.0 %** | **43.5 %** | **6.6 %** | **53.4 %** |
| IDSSE, training draw | 51.7 % | 85.0 % | 36.0 % | 11.1 % | 54.0 % |

Live frames: unchanged (seen 70.7 % IDSSE, 89.2 % Metrica; the moving share moves by 0.2 points where a 0.2 s window
crosses into a stoppage). **The stressed dead ball matches SkillCorner's** on what it was built from; it lies beyond a
line somewhat more often (37–43 % against 31 %) because the bench's restarts are more often throw-ins (41 % of its
stretches against SkillCorner's 31 %), where SkillCorner's ball is beyond the line 58 % of the time. That makes the
stress slightly easier than SkillCorner (a ball outside is a dead cue for v1's clue B and for robust). The kept recipe
was not changed after this check.

## The candidates (each fixed before it was judged)

### (a) `players`: dead evidence that doesn't trust the ball

robust's P with its P(dead) floored by v4's players-only P(dead) (`match.p_dead`: posture, speeds, set-piece layouts;
no ball by design), before the decoder:

  P'(D) = max(P(D), w · p_dead),  P'(H), P'(A) = P(H), P(A) · (1 − P'(D)) / (1 − P(D)),  clipped and rounded as robust's.

It only ever raises P(dead), keeps the team split, and composes with any start's P (below, on `average`). One setting,
**w ∈ {0, 0.5, 0.75, 1, 1.25, 1.5, 2}** (0 is v12's start; ties go to the smaller w). **Chosen leave one match out on
the start's spellF1 (decoded with robust's own decoder pick for the match, then v9's snap and v11's move) averaged
over the normal and the stressed bench**: each match counts once on each bench, so the floor is chosen as if our ball
were equally likely to behave like the bench's or like SkillCorner's. The decoder's prices stay robust's (its fold
pick), so the floor is the only thing chosen. SkillCorner plays no part in the choice.

### (b) `augment`: robust taught that a dead ball can be seen and moving

robust's fold models retrained on each training match **as it is plus the same match under the training stress**
(seed 1, templates from SkillCorner's other 10 matches): every 10th graded frame of each copy, teams exchanged, so
twice robust's rows; the same trees, seed, clip and rounding; IDSSE matches only, as robust. Nothing to choose: the
decoder is robust's fold pick. The normal copy stays in so "no ball" keeps its weight where it holds; the stressed
copy says "a seen, moving ball can be dead too".

### (c) SkillCorner's labels as training: not built (time; see the caveats).

## Every setting, and when it was fixed

| setting | value | fixed | why |
|---|---|---|---|
| moving / beyond a line | 0.3 m/s over 0.2 s / 0.5 m | before measuring | `lab_skillcorner.dead_diagnosis`'s, so the brief's 47 / 88 / 31 % are reproduced |
| template kinds | corner 3 m, touchline 1.5 m, centre 2 m, goal area 6.5 m × ±10 m, field | before the stress was built | where restarts are taken; corner and centre radii as `lab_dead`'s (3 m, 2 m) |
| nearest templates | 5, by log length ratio | before the stress was built | lengths within a few % (measured after: 0.96–1.08), and still a draw |
| ramp | 2 s, at most half the stretch | before the stress was built | SkillCorner's ball rolls on for ~2 s (4.8 then 2.9 m/s); without it the ball would jump at the whistle, a spurious touch |
| restart spot | first clean ball within 1 s of the restart | before the stress was built | the clean bench ball is there on live frames; the damaged one may not be |
| pools and seeds | test: odd-indexed SkillCorner matches, seed 0; augment's training: even-indexed, seed 1 | before the stress was built | the test never shows `augment` a stoppage it trained on |
| PathCRF on the stressed bench | the normal bench's | before the stress was built | it reads the players only; rerunning it was out of budget (caveats) |
| `players`' floors | 0, 0.5, 0.75, 1, 1.25, 1.5, 2 | before the grid ran | from "keep robust" to "v4 decides wherever it's more than half sure" |
| `players`' choice | leave one match out, normal + stressed together | before the grid ran | the insurance question; choosing on the normal bench alone would pick w = 0 by construction (it can't show the failure) |
| `augment`'s rows | both copies, every 10th frame each | before any fit | keep the normal convention's weight, add the stressed one |
| judging | v12's chain from each start, every stage refitted on its output with v12's fold setting for that match (`lab_mix.add_stage`, as `lab_motion --chains`), on each bench and set | before any chain ran | round 7's protocol |
| SkillCorner | deployment models; `players` at the w chosen on all 10 matches; `augment` trained on all 7 IDSSE matches (plus their training draw) | before any SkillCorner score | as v12 was run there |

## Results (2026-11-09)

### v12 on the stressed bench: the failure, reproduced

| | `all` (10) | `idsse` (7) |
|---|---|---|
| v12's chain, normal bench (recomputed here; lab_temporal's to 5·10⁻⁶) | 0.7033 | 0.6699 |
| **v12's chain, stressed bench** | **0.5761 (−0.1272, 0/10 better)** | **0.6023 (−0.0676, 0/7)** |
| … IDSSE / Metrica / unseen | −0.124 / −0.136 / −0.129 | — / — / J03WR9 −0.072 |
| robust alone (start), normal → stressed | 0.6729 → 0.4980 (−0.175) | 0.6365 → 0.4704 (−0.166) |
| macroF1, normal → stressed | 0.918 → 0.818 | 0.916 → 0.887 |
| robust's P(dead) on true dead frames with the ball seen, stressed | **0.21** (argmax D on 21 %) | 0.20 |

**The stress reproduces SkillCorner's failure**: robust's P(dead) where the dead ball is seen is 0.21 on the stressed bench
against 0.19 on SkillCorner, and it calls 21 % of those frames D (v12 19 % there). Where the ball is missing it still
gives 0.97 on both. v12 loses 0.13 of its 0.70 on `all`; on `idsse` less (−0.068), because that chain ends with the
`dead` stage, whose `ball` features (ball beyond a line, time since it moved, players closing on it) are refitted on
the stressed letters and win back +0.045 there (0.557 → 0.602): the chain's only stage that reads stoppages from more
than "no ball".

Stage by stage (means): `all`, stressed: robust 0.498 → short 0.511 → segments 0.573 → timing 0.576. `idsse`,
stressed: 0.470 → 0.485 → 0.545 → recall 0.557 → dead 0.602. Segments merges a share of the spurious possessions the
stoppages become (+0.06, as on SkillCorner, +0.017 there).

### The bar, part 1: the normal bench (v12's chain from each start, against v12's chain)

| | `players`: `all` | `players`: `idsse` | `augment`: `all` | `augment`: `idsse` |
|---|---|---|---|---|
| **mean** | 0.6875 (**−0.0158**) | 0.6546 (**−0.0153**) | 0.6938 (**−0.0095**) | 0.6603 (**−0.0095**) |
| better | 1/10 | 0/7 | 1/10 | 1/7 |
| IDSSE | −0.0143 | −0.0153 | −0.0076 | −0.0095 |
| Metrica | −0.0193 | — | −0.0137 | — |
| unseen | −0.0141 | −0.0063 (J03WR9) | −0.0112 | −0.0109 (J03WR9) |
| macroF1 (v12 0.918 / 0.916) | 0.908 | 0.908 | 0.911 | 0.913 |
| possessions a match (v12 340 / 374) | 324 | 358 | 340 | 370 |
| **part 1** (mean within 0.002, no source or unseen down > 0.005) | **fails** | **fails** | **fails** | **fails** |

Both cost about 0.01–0.016 on the normal bench, on every source and on the unseen matches: far outside the noise
band, on both sets. Where it comes from (mean P(dead), held out): on live frames `players` raises it from 0.053 to
0.099 and `augment` to 0.069 (`all`; argmax D on live frames 4.2 % → 7.5 % and 5.5 %): false stoppages inside live
play. Where the ball is missing in a true stoppage they were already sure (P(dead) 0.95 for all three), so there is
nothing to win back on the normal bench to pay for them.

### The bar, part 2: the stressed bench

| | `players`: `all` | `players`: `idsse` | `augment`: `all` | `augment`: `idsse` |
|---|---|---|---|---|
| **mean** (v12 0.5761 / 0.6023) | 0.6476 (**+0.0715**) | 0.6347 (**+0.0324**) | 0.7057 (**+0.1297**) | 0.6690 (**+0.0668**) |
| better | **10/10** | **7/7** | **10/10** | **7/7** |
| IDSSE / Metrica / unseen | +0.070 / +0.075 / +0.070 | — / — / +0.031 | +0.129 / +0.130 / +0.125 | — / — / +0.057 |
| start alone (against robust on the stressed bench) | +0.124 | +0.119 | +0.178 | +0.174 |
| macroF1 (v12 0.818 / 0.887) | 0.885 | 0.898 | 0.920 | 0.922 |
| P(dead), dead frames with the ball seen (robust 0.21 / 0.20) | 0.64 | 0.66 | **0.94** | **0.95** |
| **part 2** (≥ +0.02, 70 % better) | **passes** | **passes** | **passes** | **passes** |

**`augment` on the stressed bench (0.7057) scores what v12 scores on the normal one (0.7033)**: trained on one draw of
transplanted stoppages, it reads another draw's (templates from SkillCorner's other 10 matches) as dead 95 % of the
time. `players` gets half of that: v4's players-only P(dead) is a weaker witness (median ~0.3 inside the stoppages the
lab missed, [dead.md](dead.md)), so a floor at 0.75 × it lifts P(dead) to 0.64 where robust said 0.21.

Per match (`all`; each match's chain against v12's on the same bench):

| match | v12 normal | players | augment | v12 stressed | players | augment |
|---|---|---|---|---|---|---|
| J03WMX | 0.6000 | 0.5734 (-0.0266) | 0.5849 (-0.0151) | 0.4684 | 0.5397 (+0.0713) | 0.5960 (+0.1276) |
| J03WN1 | 0.6180 | 0.6018 (-0.0162) | 0.6132 (-0.0048) | 0.5048 | 0.5627 (+0.0579) | 0.6466 (+0.1418) |
| J03WOH | 0.7014 | 0.6804 (-0.0210) | 0.7075 (+0.0062) | 0.5684 | 0.6439 (+0.0756) | 0.7198 (+0.1514) |
| J03WOY | 0.6957 | 0.6820 (-0.0137) | 0.6768 (-0.0189) | 0.5261 | 0.6446 (+0.1184) | 0.6848 (+0.1587) |
| J03WPY | 0.7109 | 0.6999 (-0.0111) | 0.7030 (-0.0080) | 0.6040 | 0.6685 (+0.0644) | 0.7155 (+0.1115) |
| J03WQQ | 0.6603 | 0.6474 (-0.0129) | 0.6510 (-0.0093) | 0.5571 | 0.6038 (+0.0467) | 0.6651 (+0.1080) |
| J03WR9 | 0.6857 | 0.6870 (+0.0013) | 0.6821 (-0.0036) | 0.5789 | 0.6337 (+0.0548) | 0.6857 (+0.1068) |
| metrica1 | 0.7876 | 0.7736 (-0.0139) | 0.7760 (-0.0115) | 0.6504 | 0.7335 (+0.0831) | 0.7959 (+0.1455) |
| metrica2 | 0.7866 | 0.7748 (-0.0118) | 0.7818 (-0.0047) | 0.6627 | 0.7254 (+0.0627) | 0.7801 (+0.1174) |
| metrica3 | 0.7868 | 0.7548 (-0.0320) | 0.7619 (-0.0249) | 0.6397 | 0.7200 (+0.0803) | 0.7679 (+0.1282) |
| mean | 0.7033 | 0.6875 (-0.0158) | 0.6938 (-0.0095) | 0.5761 | 0.6476 (+0.0715) | 0.7057 (+0.1297) |

`idsse`:

| match | v12 normal | players | augment | v12 stressed | players | augment |
|---|---|---|---|---|---|---|
| J03WMX | 0.5849 | 0.5622 (-0.0226) | 0.5950 (+0.0101) | 0.5199 | 0.5528 (+0.0329) | 0.6129 (+0.0930) |
| J03WN1 | 0.6306 | 0.6037 (-0.0268) | 0.6181 (-0.0125) | 0.5516 | 0.5993 (+0.0477) | 0.6283 (+0.0766) |
| J03WOH | 0.7144 | 0.7044 (-0.0099) | 0.7063 (-0.0081) | 0.6400 | 0.6601 (+0.0201) | 0.6960 (+0.0560) |
| J03WOY | 0.7020 | 0.6818 (-0.0202) | 0.6879 (-0.0140) | 0.6291 | 0.6705 (+0.0415) | 0.6976 (+0.0685) |
| J03WPY | 0.7246 | 0.7068 (-0.0179) | 0.7157 (-0.0090) | 0.6628 | 0.6855 (+0.0227) | 0.7300 (+0.0672) |
| J03WQQ | 0.6483 | 0.6449 (-0.0034) | 0.6259 (-0.0223) | 0.5997 | 0.6305 (+0.0308) | 0.6484 (+0.0486) |
| J03WR9 | 0.6843 | 0.6780 (-0.0063) | 0.6734 (-0.0109) | 0.6129 | 0.6441 (+0.0312) | 0.6702 (+0.0573) |
| mean | 0.6699 | 0.6546 (-0.0153) | 0.6603 (-0.0095) | 0.6023 | 0.6347 (+0.0324) | 0.6690 (+0.0668) |

### The bar, part 3: SkillCorner's 20 matches

Each candidate run as v12 runs on our match (`deploy_chain`): v12's inputs on SkillCorner's tracking
(`v12.inputs`, PathCRF included, as [skillcorner.md](skillcorner.md) ran it), robust's features, then the start's P
decoded with v12's deployment setting, v9's snap, v11's move, short → segments → timing. `players` uses v12's
deployment robust and the w chosen on all 10 bench matches (0.75); `augment` a deployment model trained on the 7
IDSSE matches of `all`, each as it is and under the training draw. **v12 is reproduced exactly** first (every one of
the 20 matches: 0 frames differ from its cached letters; robust's P to 3·10⁻⁸). The post stages are v12's deployment
models (trained on robust's out-of-fold letters); the "post retrained" columns retrain them on the candidate's own
out-of-fold letters on the normal bench, as `bench/train_v12.py` does for v12 (`--deploy-post`; out of fold with
the chosen settings: `players` 0.6905, `augment` 0.6895, v12 0.7036). Nothing was chosen on SkillCorner.

| | spellF1 | gain | better | macroF1 | dead called (truth 0.464) | dead recall, ball seen | … ball missing | live called D | P(dead), dead and ball seen |
|---|---|---|---|---|---|---|---|---|---|
| v12 | 0.308 | | | 0.719 | 0.294 | 0.19 | 0.99 | 0.015 | 0.19 |
| **`players`** | **0.391** | **+0.084** | **20/20** | 0.822 | 0.443 | 0.79 | 0.99 | 0.049 | 0.61 |
| **`augment`** | **0.412** | **+0.104** | **20/20** | 0.839 | 0.465 | 0.88 | 1.00 | 0.052 | 0.86 |
| `players`, post retrained | 0.389 | +0.081 | 20/20 | | | | | | |
| `augment`, post retrained | 0.408 | +0.100 | 20/20 | | | | | | |
| v11 ([skillcorner.md](skillcorner.md)) | 0.366 | | | | | 0.82 | | | |
| the truth's stoppages with v12's teams (an oracle) | 0.473 | | | | | | | | |

**Part 3 passes for both, by five times its margin**, and both beat v11 on every match (`augment` +0.046 over
v11, `players` +0.026): with the stoppages read, v12's better teams show (skillcorner.md's lead 3, now from a
candidate fixed on the bench). `augment` closes 63 % of the way from v12 to the stoppage oracle. **`augment` on the 10
matches whose stoppages it never trained on: +0.109 (10/10); on the 10 whose dead-ball paths it trained on: +0.100
(10/10)**: no sign that it learned SkillCorner's templates rather than the idea. Retraining the post stages changes
little (−0.003, −0.004). Stage by stage, the start carries it all (v12 0.293, `players` 0.382, `augment` 0.404 before
the post stages).

| match | templates | v12 | `players` | `augment` | `players`, post retrained | `augment`, post retrained | v11 |
|---|---|---|---|---|---|---|---|
| 2017461 | test | 0.254 | 0.323 (+0.069) | 0.344 (+0.090) | 0.320 | 0.336 | 0.319 |
| 1996436 | test | 0.257 | 0.360 (+0.102) | 0.384 (+0.127) | 0.355 | 0.385 | 0.339 |
| 2010085 | augment's | 0.283 | 0.370 (+0.087) | 0.394 (+0.111) | 0.371 | 0.397 | 0.357 |
| 2007448 | augment's | 0.285 | 0.393 (+0.108) | 0.402 (+0.117) | 0.384 | 0.395 | 0.367 |
| 2015213 | test | 0.286 | 0.384 (+0.098) | 0.420 (+0.134) | 0.388 | 0.424 | 0.381 |
| 1953632 | test | 0.291 | 0.358 (+0.067) | 0.392 (+0.101) | 0.354 | 0.391 | 0.328 |
| 2011166 | test | 0.293 | 0.356 (+0.063) | 0.384 (+0.091) | 0.350 | 0.379 | 0.333 |
| 1959846 | augment's | 0.296 | 0.369 (+0.073) | 0.384 (+0.089) | 0.364 | 0.389 | 0.337 |
| 2006229 | augment's | 0.299 | 0.395 (+0.096) | 0.412 (+0.113) | 0.393 | 0.404 | 0.386 |
| 2013725 | augment's | 0.304 | 0.384 (+0.080) | 0.411 (+0.107) | 0.377 | 0.407 | 0.358 |
| 1886347 | test | 0.310 | 0.406 (+0.096) | 0.440 (+0.129) | 0.408 | 0.428 | 0.365 |
| 1996435 | augment's | 0.311 | 0.395 (+0.084) | 0.395 (+0.084) | 0.392 | 0.393 | 0.377 |
| 1986691 | test | 0.316 | 0.404 (+0.088) | 0.419 (+0.103) | 0.399 | 0.413 | 0.400 |
| 1899585 | augment's | 0.319 | 0.399 (+0.080) | 0.414 (+0.096) | 0.396 | 0.402 | 0.357 |
| 1925299 | test | 0.320 | 0.394 (+0.074) | 0.405 (+0.085) | 0.394 | 0.401 | 0.383 |
| 2006363 | test | 0.324 | 0.407 (+0.083) | 0.426 (+0.102) | 0.403 | 0.424 | 0.372 |
| 2007721 | test | 0.336 | 0.431 (+0.095) | 0.460 (+0.124) | 0.434 | 0.451 | 0.390 |
| 2016236 | augment's | 0.342 | 0.424 (+0.082) | 0.417 (+0.075) | 0.419 | 0.411 | 0.338 |
| 1874553 | augment's | 0.342 | 0.420 (+0.078) | 0.456 (+0.114) | 0.418 | 0.455 | 0.386 |
| 1927964 | augment's | 0.383 | 0.456 (+0.073) | 0.473 (+0.091) | 0.452 | 0.469 | 0.439 |
| **mean** | | **0.308** | **0.391 (+0.084)** | **0.412 (+0.104)** | 0.389 | 0.408 | 0.366 |

### `players` on the `average` start (normal bench, the chain; information for the mix)

| | `all` | `idsse` |
|---|---|---|
| `average`'s chain ([temporal.md](temporal.md)) | 0.7162 | 0.6760 |
| `average` floored by `players` (each match's fold w, 0.75 everywhere) | 0.6995 (**−0.0166** against `average`, 0/10; −0.0038 against v12) | 0.6621 (**−0.0139**, 1/7; −0.0077 against v12) |

The floor costs `average` what it costs robust (−0.014 to −0.017), on every source: composed, it would undo most of
`average`'s gain over v12. It composes mechanically (one line on any P), but not for free.

### The floor's grid (start level, mean of the matches; information: the folds all picked 0.75)

| w | 0 (v12) | 0.5 | **0.75** | 1 | 1.25 | 1.5 | 2 |
|---|---|---|---|---|---|---|---|
| `all`, normal | 0.6729 | 0.6704 | **0.6624** | 0.6526 | 0.6417 | 0.6312 | 0.6157 |
| `all`, stressed | 0.4980 | 0.5776 | **0.6222** | 0.6222 | 0.6181 | 0.6112 | 0.6001 |
| `idsse`, normal | 0.6365 | 0.6342 | **0.6267** | 0.6164 | 0.6073 | 0.5995 | 0.5869 |
| `idsse`, stressed | 0.4704 | 0.5523 | **0.5893** | 0.5910 | 0.5892 | 0.5849 | 0.5764 |

Every fold of both sets picked w = 0.75 (the best mean of the two benches). There is no free floor: the normal bench
loses from the first step, the stressed bench gains most of what it can by 0.75.

### Post hoc, information only: a gentler floor (w = 0.5)

Run **after** seeing the grid and the chains above, so it is not a candidate and is not judged; it answers "is there a
floor that buys the insurance for free?" (`--chains … --starts players_w0.5`, every match at w = 0.5):

| | normal: mean | IDSSE / Metrica / unseen | stressed: mean | better |
|---|---|---|---|---|
| `all` | 0.7014 (−0.0019, 3/10) | −0.0043 / +0.0037 / +0.0025 | 0.6156 (+0.0396) | 10/10 |
| `idsse` | 0.6682 (−0.0017, 4/7) | −0.0017 / — / −0.0016 | 0.6190 (**+0.0168**) | 7/7 |

At w = 0.5 the normal bench would sit inside part 1's band on both sets, but the stressed `idsse` gain (+0.017) would
miss part 2's +0.02 (on SkillCorner it would give 0.351, +0.044, 20/20). So no single floor clears both parts on both sets, even chosen with hindsight: v4's players-only
P(dead) can't separate a stoppage from live play well enough to be leaned on harder without false stoppages.

## Verdict

**Neither candidate clears the robustness bar; v12 stays as it is.** Both fail part 1 and pass parts 2 and 3 by far:

| | part 1, normal bench (`all` / `idsse`) | part 2, stressed bench (`all` / `idsse`) | part 3, SkillCorner | bar |
|---|---|---|---|---|
| `players` | −0.0158 (1/10) / −0.0153 (0/7); IDSSE −0.014, Metrica −0.019, unseen −0.014 / −0.006: **fails** | +0.0715 (10/10) / +0.0324 (7/7): passes | 0.391, +0.084, 20/20: passes | **fails** |
| `augment` | −0.0095 (1/10) / −0.0095 (1/7); IDSSE −0.008, Metrica −0.014, unseen −0.011 / −0.011: **fails** | +0.1297 (10/10) / +0.0668 (7/7): passes | 0.412, +0.104, 20/20: passes | **fails** |

What it means, plainly:

1. **The failure is real and big where it applies.** A SkillCorner-like dead ball costs v12 0.127 of its 0.703 on the
   bench (`idsse` chain 0.068), and 0.17 on SkillCorner itself (skillcorner.md's oracle); robust's P(dead) where the dead ball is seen is 0.21
   on the stressed bench and 0.19 on SkillCorner.
2. **It can be fixed, at a price.** `augment` reads a tracked, moving dead ball as dead (P 0.94 on the stressed
   bench, 0.86 on SkillCorner), scores on the stressed bench what v12 scores on the normal one (0.706 against
   0.703), and lifts SkillCorner from 0.308 to 0.412, above v11 on all 20. But it costs ~0.01 on the normal bench,
   on every source, on both sets: more live frames called dead (P(dead) on live frames 0.053 → 0.069). `players` is
   weaker on both counts (v4's players-only P(dead) is a blunt witness), and a gentler floor (post hoc) trades one
   part for the other without clearing both.
3. **So it's a bet on our ball, which the bar can't settle.** At these numbers `augment` pays off if the chance that
   our seen ball moves through stoppages like SkillCorner's is above about 7–9 % (0.0095 lost × the rest against
   0.10–0.13 gained). Our match is Metrica-tracked with its ball missing 83 % of the time players stand still
   ([../bench.md](../bench.md)), which argues for the normal bench; nobody has looked at whether its seen ball moves in
   stoppages. The label-free check that would say ([learned.md](learned.md): the still share of the seen ball while
   players stand; and v12's dead share against the bench's range, [skillcorner.md](skillcorner.md) "What it means" 2)
   is Jose's to run. If it says "SkillCorner-like", `augment` is the hedge, built and measured here.
4. **The stressed bench stays for later candidates**: `stressed(m, which)` gives any bench match with the test draw,
   costs recomputed; a robustness fix can now be judged on it before anything touches SkillCorner.

## For the mix (not kept; if Jose overrides the bar)

```python
from bench import lab_carried as C
# bench, held out: robust's / augment's P on either bench, decoded with robust's fold pick
p = C.held_out_p("augment", "normal" | "stressed", match_id, which)        # or ("robust", "stressed", ...)
letters = C.decoded(match, p, lab_mix.fold_settings(which)["robust"][match_id], which)
p_players = C.floor(p_any_start, match.pred["p_dead"], w=0.75)               # composes with any start's P
match = C.stressed(match_id, which)                                           # the stressed bench (test draw)
# deployment, as v12 runs (SkillCorner or our match): augment's model and its retrained post stages
aug = C.deployment_augment()                                                  # data/external/lab_carried/augment_all.joblib
steps = C.deploy_chain(match, C.proba(aug, {"X": X, "X_swapped": Xs}), v12.load_models("all"), "all",
                       post=C.deploy_post("augment"))
```

`augment` for our match would need the training draw's features for all 7 IDSSE matches (`stressed_features(m,
"all", TRAIN_SEED)`, cached) and `possession/v12.py` pointed at the new robust model; nothing here touches it.

## Caveats

1. **The stressed bench is SkillCorner's dead ball on the bench's players.** The transplanted ball moves as
   SkillCorner's does relative to the restart spot, but the bench's players didn't fetch it: mid-stoppage it can be
   metres from anyone (on SkillCorner too: 2.9 m from the nearest, median). It lies beyond a line a little more often
   than on SkillCorner (37–43 % against 31 %, the bench's restarts being more often throw-ins), which makes it a little
   easier. Its live frames are the bench's.
2. **PathCRF's P on the stressed bench is the normal bench's.** PathCRF reads the players only, but it answers only on
   v6's live stretches, and v6's costs move with the new ball (v1's clue A: a slow ball near a player argues against
   D). Where v6 now calls live a frame it called dead, PathCRF stays silent instead of answering. Rerunning
   `bench.eval_pathcrf` on the stressed tracking was out of this budget. It touches every start equally.
3. **`augment` trains on the same recipe the stressed bench tests.** Different draws and different SkillCorner
   matches' stoppages, but the same transplant: its +0.13 on the stressed bench is partly "it learned this recipe".
   SkillCorner (part 3) is the honest test of it; there its training templates come from the even-indexed 10 matches,
   so those 10 are not unseen by it (their dead-time ball paths, not their labels or players). Both halves are
   reported.
4. **SkillCorner was looked at before.** [skillcorner.md](skillcorner.md) diagnosed this failure on all 20 matches,
   and this page measured their dead-time ball to build the stress. Nothing here was tuned on SkillCorner's scores
   (`players`' floor is chosen on the bench; `augment` has no setting), but the problem itself was found there.
5. **`players`' floor is chosen on a bench half made of the stress.** Choosing on the normal bench alone would pick
   w = 0 (it can't show the failure), so the choice weighs both equally, by design; a different weighting (our match
   more likely Metrica-like than SkillCorner-like) would pick a smaller w and pay less on the normal bench, as the
   post hoc w = 0.5 shows. That weighting was not chosen beforehand and isn't judged here.
6. **On SkillCorner, the post stages of the main rows are v12's deployment models** (trained on robust's
   out-of-fold letters); the "retrained" rows train them, as `bench/train_v12.py` does, on the candidate's own
   out-of-fold letters on the normal bench (`--deploy-post`). Both are reported.
7. **Only 41 % damage**, as every lab candidate.
8. **What was looked at before the full runs, and what changed.** SkillCorner's dead-time ball was measured first
   (`--measure-sc`; a first pass counted "beyond a line" at 0 m instead of 0.5 m and was rerun before the recipe was
   written). The stressed ball was checked (the table above) before any score; nothing changed after. The SkillCorner
   run's code was checked on one match (1874553: v12 reproduced, `players` 0.420) after two bugs were fixed in it (a
   variable shadowing the scorer's X code, and an undefined name), with w already fixed; nothing changed after. The
   post hoc w = 0.5 rows were added after every candidate's bench and stressed scores were in.
9. **(c), SkillCorner's labels as training, was not built.** It would need SkillCorner split for a test (it is
   now this page's part 3), and its labels' blurred team changes ([skillcorner.md](skillcorner.md)) would come with
   the dead ones.
