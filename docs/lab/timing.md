# Lab candidate `timing`: each change of team moved to the frame a learned ranker picks

| | |
|---|---|
| **status** | **built 2026-11-04, lab round 2, on the mix; clears the bar** on both bench sets: held out spellF1 **0.6282 against the mix's 0.6156 (+0.0126), better on 10/10**; IDSSE only (`--set idsse`) +0.0039, better on 5/7 |
| **what** | for each predicted H ↔ A change, candidate frames up to 3 s either side (touches by either team, the new team reaching the ball or becoming nearest, and the change itself); a classifier trained on the other bench matches gives each P(the true change is within 0.2 s of it); the change moves to the most likely one when it beats staying and reaches P_MIN. v11's guard: never crosses another change, never swallows a possession |
| **stage** | `post`: works on any letters (judged on the mix's out-of-fold letters, `lab_mix.kept_letters`) |
| **code** | `bench/lab_timing.py` (new file only; nothing else changed) |
| **run** | `python -m bench.lab_timing [--set all\|idsse]` (~1 min on `all`; 24 s and 0.54 GB peak on `idsse`, measured; a second run gives byte-identical results); `--nested` (~4 min on `all`, ~1 min on `idsse`: a stricter check of the same grid) |
| **results** | `bench/results/lab_timing[_idsse].json` (the judge's report, plus timing, per-match moves, ranker quality, ceilings), `lab_timing_grid[_idsse].csv` (every match × setting), `lab_timing_nested[_idsse].json` |
| **mix with** | `bench.lab_timing.fit(others, letters_of, rule="learned", features="all")` then `bench.lab_timing.apply(match, letters, model, rule="learned", features="all", window=3.0, p_min=0.35)` |
| **new dependencies** | none (sklearn's `HistGradientBoostingClassifier`, as `segments`) |

## Results (summary)

| | **all 10, held out** | IDSSE only (`--set idsse`, 7) |
|---|---|---|
| **spellF1** | **0.6282 against the mix's 0.6156 (+0.0126), better on 10/10** | **0.6040 against 0.6001 (+0.0039), 5/7** |
| IDSSE | 0.6018 against 0.5967 (+0.0051), 7/7 | (as above) |
| Metrica | 0.6899 against 0.6599 (**+0.0300**), 3/3 | — |
| **PathCRF-unseen** (J03WR9, metrica1–3) | 0.6740 against 0.6499 (**+0.0241**), 4/4 | J03WR9: 0.6304 against 0.6223 (+0.0081) |
| macroF1 | 0.8817 against 0.8775 | 0.8830 against 0.8816 |
| **chosen on all the matches** | **features `all`, window 3 s, P_MIN 0.35** (every fold: 3 s, 0.35; 5 folds `all`, 5 `no_pathcrf`) | features `no_pathcrf`, window 2 s, P_MIN 0.35 |
| **nested check** | **0.6292 (+0.0135), 10/10**; IDSSE +0.0058, Metrica +0.0316, unseen +0.0253 | 0.6050 (+0.0049), 6/7 |
| **timing** (changes matching a true one, held-out picks) | within 0.2 s **58 % → 69 %**; median 0.12 → 0.08 s; 3rd quartile 0.84 → 0.48 s; over 1 s 21 % → 14 % | 57 % → 61 %; 0.12 → 0.08 s; 0.80 → 0.68 s; 20 % → 17 % |
| possessions | unchanged (every setting, every match) | unchanged |
| **the bar** | mean ✓, share 10/10 ✓, each source ✓, unseen ✓: **passes** | mean ✓, 5/7 ✓ (the minimum), unseen ✓: **passes** |

Against v11 (0.5727), the mix with `timing` after it is at **0.6282 (+0.0555)**.

## Why

**The error budget on the mix** ([../lab.md](../lab.md), "What's left"): team-change timing is worth
**+0.048** if every predicted H ↔ A change within 2 s of a true one of the same kind sat on it.

**Measured first, on the mix's out-of-fold letters** (all 10 matches, 2,278 predicted changes; 78 %
within 2 s of a true change of the same kind):

| | |
|---|---|
| distance to the true change | median **0.12 s**; quartiles 0.04 / **0.84 s**; within 0.2 s **58 %**; on the frame 41 % |
| on the truth's frame or nearly (≤ 0.2 s) | 1,041 changes: worth +0.002 if made exact |
| 0.2–1 s off | 372 changes: worth **+0.013** |
| **more than 1 s off** | 369 changes (37 a match; 175 early, 194 late): worth **+0.034** |

So the budget is not in fine-tuning the changes v11 already put on a touch: it's in the fifth of
changes that sit on **the wrong touch**, 1–2 s from the right one. Of those, **46–64 % already sit on
a new-team touch** (v11 took the touch nearest the change, within 1 s), and between the predicted
and the true change there are on average 0.7–0.8 more new-team touches and 0.4–0.55 old-team ones:
a duel, a first touch and a second, a pass between two new-team players. A new-team touch within
0.2 s of the true change exists in 65 % of them (the ball is seen at the true change in 81 %).
Picking *which* touch is a ranking problem, which a nearest-touch rule can't solve: v11's move
again with a 2 s window gains only +0.001 here (below).

**The ceilings** (truth read; exploration, nothing picked from them): moving every matched change
onto its true one, with v11's guard, **+0.046**; onto the right candidate nearest it (what a perfect
ranker over these candidates could do), **+0.029**; 83 % of matched changes have a right candidate
(IDSSE 80 %, Metrica 92 %).

## What it does

### Candidates

For each change from team O to team N at frame *g*, inside a half:
- **the frames it may move to:** from O's run's second frame to N's run's last frame (O keeps at
  least its first frame, N its last), and at most MAX_WINDOW (3 s) from *g*;
- **candidates among them:** a **loose touch** by either team (the ball's velocity changes by ≥ 1 m/s
  over 0.2 s either side, a player of that team within 3 m; the peak of each run of such frames, as
  `touches.py`); the first frame where **N's nearest player comes within 2 m**; the first frame where
  **N's nearest becomes nearer than O's**; and **g itself** (staying).

About 8–10 candidates per change (1,644–2,368 a match).

### Features (prediction-time inputs only)

From the match's damaged tracking, the out-of-sample outputs in `match.pred`, and the input letters.
*k* is the candidate.

| feature | what |
|---|---|
| `dk`, `adk`, `is_g` | (k − g) in s, its size, k is g |
| `tl_n`, `tl_o`, `tv_n`, `tv_o` | k is a loose touch by N (O); a v11 touch (2 m, ΔV 3 m/s) by N (O) |
| `rs_n`, `ns_n` | k is where N's nearest comes within 2 m; where N becomes nearest |
| `dv_k`, `sp_b`, `sp_a`, `cos_k` | the ball's velocity change at k, its speed over the 0.2 s before and after, the cosine between the two directions |
| `dn_k`, `do_k`, `margin_k` | N's (O's) nearest distance at k, capped at 30 m; O's minus N's |
| `dn_soon`, `do_recent` | N's closest over the 0.5 s after k; O's closest over the 0.5 s before |
| `seen_k`, `gap_b`, `gap_a` | the ball seen at k; frames since it was last seen, until it's next seen (capped at 2 s) |
| `nt_before`, `ot_before`, `nt_after`, `ot_after` | loose touches by N (O) in the 2 s before (after) k |
| `last_touch`, `next_touch` | the team of the last (next) loose touch within 2 s: +1 N, −1 O, 0 none or both |
| `closer_n`, `between_n`, `between_o` | N's touches closer to g than k; N's (O's) touches between g and k |
| `pdead_k` | mean P(dead) over 1 s either side |
| `run_o`, `run_n`, `g_on_touch`, `g_dv` | log length of O's and N's runs; g is on a v11 touch by N; the ball's velocity change at g |
| `pn_k`, `pn_before`, `pn_after`, `pn_jump` | PathCRF's P(N) at k, its mean over 1 s before and after, the jump (`all` only) |
| `m7_before`, `m7_after`, `m7_jump` | v7 decoder's cost of O minus N, mean over 1 s before and after, the jump (`all` only) |
| `m6_*` | the same from v6's costs, which have no PathCRF clue (`no_pathcrf` only) |

Missing values (ball unseen, PathCRF silent) are left missing: the classifier handles them.

### Target (truth of the training matches only)

A candidate is **right** when a true change of the same kind (O → N) is within **TOL = 5 frames
(0.2 s)** of it. A change with no true one near has no right candidate, so the classifier also learns
when nothing should move.

### The classifier and the move

One `HistGradientBoostingClassifier` per feature set, with `segments`' fixed settings (200 rounds,
rate 0.05, 8 leaves, at least 40 rows per leaf, L2 1.0, no early stopping, seed 0); about 18,000
candidates a fold. Not tuned.

Each change, in time order, moves to its **most likely candidate within the window** (ties: the one
nearest *g*), when that candidate isn't *g*, its P is above *g*'s and at least **P_MIN**. The move is
v11's, on the letters as they are by then: to an earlier *k* only if O's frames from *k* − 1 to *g*
are all O's; to a later *k* only if N's frames from *g* to *k* are all N's. So a move never crosses
another change and never swallows a possession: the teams in order and the number of possessions
never change (checked: identical counts for every setting on every match).

### Every constant, and why

| setting | value | why |
|---|---|---|
| loose touch: reach | 3 m | v11's 2 m plus a metre: the damaged players and ball jitter, and the ranker reads the distance anyway. A candidate missed is lost for good; a spurious one costs only a row |
| loose touch: ΔV | 1 m/s | well under v11's 3 m/s for the same reason (the damaged ball's jitter makes ~2 m/s); the size of the change is a feature |
| look either side | 0.2 s (5 frames) | the touch detector's (`touches.LOOK`) |
| MAX_WINDOW | 3 s | the widest window in the grid: the far-off changes are 1–2 s away, and the budget's "same change" is 2 s |
| TOL | 0.2 s | the page's own timing measure (as v11's); the truth's change sits on the new team's touch, which the detector finds to a frame or two |
| touch counts | 2 s either side | a possession's touches before a change and the other team's after, at the scale of the far-off errors |
| PathCRF and costs | means over 1 s either side | long enough to average PathCRF's flicker, short enough to see the jump at a change |
| soon / recent | 0.5 s | does N reach the ball just after *k*, was O at it just before |
| caps | 30 m, 2 s gaps, 30 s runs | bounded features; past these a value says nothing more |
| classifier | as `segments` | small and regularised; not tuned |
| `nearest` rule | v11's move (ΔV 3, nearest N touch) at ±1 s and ±2 s | the simplest alternative: does a wider nearest-touch window already do it? |
| **window** | **1, 2, 3 s** | chosen leave one match out |
| **P_MIN** | **0.5, 0.35, 0.65** | 0.5 first; lower because "beats g" is already required and a candidate is one of ~8, so a calibrated 0.35 can be the best bet; higher for caution |
| **features** | **`all`, `no_pathcrf`** | PathCRF was pretrained on 6 of the 7 IDSSE matches, so on them it's too good; `no_pathcrf` reads neither PathCRF nor v7's costs |

The grid (21 settings, simplest first: the no-op, `nearest` at 1 and 2 s, then the ranker for each
feature set × window × P_MIN) was fixed before any held-out run and not changed after it.

## How it's chosen and judged

- **Input:** the mix's out-of-fold letters on the 10 bench matches (`lab_mix.kept_letters`; each
  match's trained stages fitted without it).
- **Leave one match out:** for each held-out match the classifier is trained on the other 9 matches'
  candidates (their kept letters and their truth), and `lab.judge` scores it with the setting best on
  the other 9.
- **The bar** ([../lab.md](../lab.md)), against the mix (`lab_mix.kept_base`): better held-out mean,
  ≥ 70 % of matches better, no drop on IDSSE nor on Metrica, none on the 4 PathCRF-unseen matches.
  Then the same on IDSSE alone (`--set idsse`: that set's mix, trained on 6 matches).
- **Nested** (`--nested`): as `segments`, the settings picked with no model trained on the held-out
  match (for each other match j, models trained without the held-out one and j).

## Results

### All 10 matches

**Means over the 10 matches, every setting** (not held out):

| setting | spellF1 | macroF1 | moves a match |
|---|---|---|---|
| no-op (the mix) | 0.6156 | 0.8775 | 0 |
| `nearest` ±1 s / ±2 s | 0.6157 / 0.6166 | 0.8775 / 0.8774 | |
| ranker, `all`, 1 s (P_MIN 0.5 / 0.35 / 0.65) | 0.6176 / 0.6180 / 0.6167 | 0.878 | 38 / 45 / 28 |
| ranker, `all`, 2 s | 0.6246 / 0.6262 / 0.6231 | 0.880–0.881 | 48 / 60 / 35 |
| **ranker, `all`, 3 s** | 0.6275 / **0.6295** / 0.6255 | 0.881–0.882 | 51 / **64** / 38 |
| ranker, `no_pathcrf`, 1 s | 0.6169 / 0.6171 / 0.6170 | 0.878 | 38 / 43 / 28 |
| ranker, `no_pathcrf`, 2 s | 0.6252 / 0.6266 / 0.6230 | 0.880–0.881 | 49 / 58 / 36 |
| ranker, `no_pathcrf`, 3 s | 0.6277 / 0.6295 / 0.6255 | 0.881–0.882 | 51 / 61 / 38 |

The two feature sets tie (0.62951 against 0.62946 at 3 s, 0.35): PathCRF adds nothing here, so the
gain doesn't lean on it. **Every fold picked 3 s and P_MIN 0.35** (5 folds `all`, 5 `no_pathcrf`).

**Held out:**

| match | source | the mix | held out | gain | moves | within 0.2 s before → after |
|---|---|---|---|---|---|---|
| J03WMX | IDSSE | 0.5025 | 0.5045 | +0.0020 | 45 | 50 % → 55 % |
| J03WN1 | IDSSE | 0.5770 | 0.5808 | +0.0038 | 57 | 58 % → 64 % |
| J03WOH | IDSSE | 0.5992 | 0.6082 | +0.0090 | 44 | 60 % → 63 % |
| J03WOY | IDSSE | 0.6277 | 0.6304 | +0.0026 | 30 | 65 % → 70 % |
| J03WPY | IDSSE | 0.6640 | 0.6645 | +0.0005 | 46 | 60 % → 65 % |
| J03WQQ | IDSSE | 0.5863 | 0.5980 | +0.0117 | 55 | 49 % → 56 % |
| J03WR9 | IDSSE, unseen | 0.6202 | 0.6264 | +0.0062 | 30 | 62 % → 72 % |
| metrica1 | Metrica, unseen | 0.6662 | 0.6983 | +0.0320 | 105 | 64 % → 84 % |
| metrica2 | Metrica, unseen | 0.6671 | 0.6961 | +0.0289 | 109 | 57 % → 82 % |
| metrica3 | Metrica, unseen | 0.6463 | 0.6754 | +0.0291 | 100 | 60 % → 79 % |
| **mean** | | **0.6156** | **0.6282** | **+0.0126, 10/10** | | **58 % → 69 %** |

| | the mix | held out | better |
|---|---|---|---|
| IDSSE (7) | 0.5967 | 0.6018 (+0.0051) | 7/7 |
| Metrica (3) | 0.6599 | 0.6899 (+0.0300) | 3/3 |
| PathCRF-unseen (4) | 0.6499 | 0.6740 (+0.0241) | 4/4 |
| macroF1 | 0.8775 | 0.8817 | |

**All four checks pass.**

**The timing** (every predicted change within 2 s of a true one of the same kind, held-out picks):

| | before (the mix) | after |
|---|---|---|
| within 0.2 s | 58.4 % | **68.6 %** |
| on the frame (≤ 0.04 s) | 41.5 % | 49.5 % |
| median distance | 0.12 s | **0.08 s** |
| quartiles | 0.04 / 0.84 s | 0.00 / **0.48 s** |
| over 1 s off | 20.7 % | 13.8 % |

**The moves** (the chosen setting, `all`, 3 s, 0.35, on every held-out match; distance to the nearest
true change of the same kind):

| | moves | earlier / later | fixed (> 0.2 s → ≤ 0.2 s) | broken (≤ 0.2 s → > 0.2 s) | > 0.5 s closer / farther | land within 0.2 s |
|---|---|---|---|---|---|---|
| all 10 | 640 (64 a match) | 416 / 224 | **224** | 22 | 206 / 34 | 71 % |
| IDSSE (7) | 326 | 222 / 104 | 92 | 18 | 82 / 25 | 65 % |
| Metrica (3) | 314 | 194 / 120 | 132 | 4 | 124 / 9 | 76 % |
| `--set idsse` (its setting: `no_pathcrf`, 2 s) | 300 | 176 / 124 | 69 | 9 | 62 / 17 | 56 % |

Median shift 0.20 s (0.16 s on IDSSE); most of the rest are small corrections among frames already
within 0.2 s. A fifth of the moves are at changes with no true one within 2 s (they matter little).
On IDSSE about 5 fixes per break and 3 big gains per big loss; on Metrica 33 fixes per break and
14 big gains per big loss.

**The ranker** (held out, each fold's own model): AUC 0.92 (0.86–0.95 a match; the same for both
feature sets); its top candidate is right for 56 % of changes, and for **81.5 % of the changes that
have a right candidate**. On Metrica 0.94 AUC and 88–90 %; on IDSSE 0.86–0.94 and 71–84 %.

**Against the ceilings:** +0.0126 of the +0.029 a perfect ranker over these candidates reaches, of the
+0.046 for every matched change on its true frame. On IDSSE the ranker reaches about a sixth of the
candidate ceiling (+0.005 of +0.029 on average; +0.013 to +0.041 a match); on Metrica all of it
(+0.029–0.032 against +0.026–0.032): a few of its moves fix changes 2–3 s off, which the 2 s ceilings leave out.

### IDSSE only (`--set idsse`)

Means: no-op 0.6001, `nearest` 0.6001 / 0.6019, ranker at 3 s, 0.35: 0.6052 (`all`) / 0.6051
(`no_pathcrf`); best 0.6056 (`no_pathcrf`, 2 s, 0.35). Every fold picked P_MIN 0.35; 5 of 7 the 2 s
window.

| match | the mix (IDSSE chain) | held out | gain | picked |
|---|---|---|---|---|
| J03WMX | 0.5148 | 0.5211 | +0.0062 | `all`, 3 s |
| J03WN1 | 0.5757 | 0.5780 | +0.0023 | `no_pathcrf`, 2 s |
| J03WOH | 0.5953 | 0.5926 | **−0.0028** | `no_pathcrf`, 2 s |
| J03WOY | 0.6534 | 0.6599 | +0.0065 | `no_pathcrf`, 2 s |
| J03WPY | 0.6686 | 0.6675 | **−0.0011** | `no_pathcrf`, 3 s |
| J03WQQ | 0.5704 | 0.5784 | +0.0080 | `no_pathcrf`, 2 s |
| J03WR9 (unseen) | 0.6223 | 0.6304 | +0.0081 | `no_pathcrf`, 2 s |
| **mean** | **0.6001** | **0.6040** | **+0.0039, 5/7** | |

macroF1 0.8816 → 0.8830. Timing: within 0.2 s 57 % → 61 %; median 0.12 → 0.08 s; over 1 s 20 % →
17 %. Ranker AUC 0.91, top candidate right for 78 % of the changes that have one. **Passes**, with
exactly the 5 of 7 needed: trained on 6 IDSSE matches it's weaker than with Metrica's 3 in.

### The stricter check (`--nested`)

`lab.judge` picks match m's setting from the other matches' scores, but their models were trained
with m among their 9 training matches. `--nested` removes that: for each m and each other match j,
the classifier is trained without m and j, j is scored on every setting, and m's setting is the one
best on those scores (ties: the first in grid order). m is then scored as in the main run.

| | nested | judge (above) |
|---|---|---|
| **all 10** | **0.6292 against 0.6156 (+0.0135), 10/10** | +0.0126, 10/10 |
| IDSSE (7) | 0.6025 (+0.0058), 7/7 | +0.0051, 7/7 |
| Metrica (3) | 0.6915 (+0.0316), 3/3 | +0.0300, 3/3 |
| PathCRF-unseen (4) | 0.6752 (+0.0253), 4/4 | +0.0241, 4/4 |
| **`--set idsse` (7)** | **0.6050 against 0.6001 (+0.0049), 6/7** (J03WOH −0.0006) | +0.0039, 5/7 |

Every nested fold of `all` picked 3 s and 0.35 again (6 `no_pathcrf`, 4 `all`); on `idsse`, 0.35
always, 3 s in 4 folds and 2 s in 3. It holds, slightly higher: the choice of setting isn't what
carries the gain.

## Failure modes

1. **IDSSE gains little.** +0.005 on `all`'s IDSSE matches, +0.004 and 5/7 on `--set idsse`, against
   +0.03 on Metrica. On IDSSE the touches line up with the truth less often (80 % of matched changes
   have a right candidate, against 92 %) and the ranker's top pick is right less often (71–84 %
   against 88–90 %). Our match is Metrica-tracked, but its tracking is its own: if its touches line
   up like IDSSE's, expect the IDSSE gain, not Metrica's.
2. **Metrica's gain may flatter it.** Metrica's three matches gain +0.029–0.032 each, as much as a
   perfect ranker over these candidates would within 2 s. The ranker trains on the other two Metrica
   matches in each fold, so it learns Metrica's ball and its labelling; our match's labels follow
   the organisers' convention, which may differ.
3. **The grid's edge.** Every fold of `all` picked the widest window (3 s) and the lowest P_MIN
   (0.35), and the score was still rising (2 → 3 s: +0.003; 0.5 → 0.35: +0.002). Not extended: the
   grid was fixed before the run. A wider window risks moving a change onto another change's touch;
   a lower P_MIN, coin-flip moves.
4. **Moves off a right frame.** A change already on its true frame can be moved off it when another
   candidate looks better (see the moves above). The "beats g" rule bounds it.
5. **A possession shrunk to a frame.** The guard keeps at least one frame of each possession, as
   v11's. Two moves toward each other can leave a possession of a frame or two; the count of
   possessions doesn't change, but such a remnant can't match a true possession.
6. **Unseen ball at the true change** (19 % of matched changes): no touch to rank; the change stays
   or moves to the wrong touch.
7. **Trained on the mix's letters.** It learns the mix's errors; put before or after a different
   stage, it must be refitted on that stage's out-of-fold letters (`fit(..., letters_of)`).

## Verdict

**Keep.** It clears every check of the bar on both bench sets against the mix (+0.0126, 10/10;
IDSSE only +0.0039, 5/7), and the nested check holds (+0.0135, 10/10; IDSSE only +0.0049, 6/7). It
puts 69 % of the matched changes within 0.2 s of the truth (from 58 %) and cuts the ones over 1 s off
from 21 % to 14 %, without changing the teams or the number of possessions. The gain is solid on
Metrica (+0.030) and small on IDSSE (+0.005), so on our match expect somewhere between the two.

For the mix, after `segments` (the last stage it was judged on):

```python
from bench import lab_timing
model = lab_timing.fit(others, letters_of, rule="learned", features="all")      # others: the other bench matches
letters = lab_timing.apply(match, letters, model, rule="learned", features="all", window=3.0, p_min=0.35)
```

`letters_of(o)` must give each training match's out-of-fold letters from the stages before it (as
`lab_mix.run_chain` does). `features="no_pathcrf"` ties on the bench (0.62946 against 0.62951 in
sample) and is the one `--set idsse` picks (with `window=2.0`); it's the safer choice if PathCRF on our
match turns out weaker than on the bench.
