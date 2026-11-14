# Model v14: v13 with two settings from a search on Metrica's convention

| | |
|---|---|
| **status** | **built 2026-11-12, Jose's go-ahead.** The port gives the lab's letters on every frame of two bench matches (float32 and float64); the stoppage switch passes its bar with v14's branch A |
| **what** | v13 ([v13](v13-temporal.md)) with branch A changed on bench set `all`: the start weighs the GRU **0.8** and robust 0.2 (v13: 0.5 each), and **v9's snap and v11's move are off**; v12's post stages refitted on this start's letters. Branch B, the switch, its threshold and the IDSSE-only set are v13's |
| **why** | The answer key follows Metrica's owner convention ([owner-rule.md](../owner-rule.md)); a nested search of 2,000 settings on Metrica's 3 games found these two changes helping every game ([search-metrica.md](../search-metrica.md)) |
| **bench** (`all`, every stage refitted out of fold) | **0.7241 held out against v13's 0.7162 (+0.0080), better on 8/10**; Metrica 0.8245 (+0.0156, 3/3); DFL 0.6812 (+0.0047, 5/7); PathCRF-unseen 0.7946 (+0.0142, 4/4). With the settings chosen on all matches (what v14 runs): 0.7258 against v13's 0.7195 |
| **new code** | `possession/v14.py`, `possession/models/v14_all.joblib` (1.8 MB), `bench/train_v14.py`, `bench/check_v14.py` (the port), `bench/check_v14_switch.py` (the switch's bar with v14's A); the search: `bench/search_metrica.py` |
| **run** | `OMP_NUM_THREADS=2 python -m bench.train_v14` (~1 min), `python -m bench.check_v14` (~15 min), `python -m bench.check_v14_switch chain --variant V` (5 variants) then `report`, `python -m bench.check_v14_switch skillcorner`; then `python -m possession.v14` on our match: path, SHA-256 and format check only (`--show-branch` for Jose) |

## What changes, and what doesn't

| | v13 | v14 |
|---|---|---|
| start (branch A) | (P_GRU + P_robust) / 2 | **0.8 P_GRU + 0.2 P_robust** (both through float32, then clipped and rounded, as v13) |
| v9's snap, v11's move (branch A) | on | **off** |
| v12's post stages (short, segments, timing) | fitted on v13's start's letters | **refitted on v14's** (same settings) |
| decoder setting, GRU, robust, inputs, kick-off rule | | unchanged |
| stoppage switch, threshold 0.1445, branch B (`augment`) | | unchanged |
| bench set `idsse` | | v14 is v13 (the change isn't a gain there: −0.0008, 3/7) |

## Built: what the checks say

| check | result |
|---|---|
| the lab's chain from v14's start, recomputed by `check_v14` | 0.7241, as `search_metrica confirm` recorded |
| **the port with fold models: letters differing from the lab's chain** | **0 of 146,811 (J03WR9), 0 of 145,007 (metrica1)**, in float32 and in float64; the start's P equal to the lab's |
| `v14.predict` end to end on J03WR9's damaged tracking, twice | the same letters; branch A |
| `train_v14` out of fold with the chosen settings | 0.7258 (v13 the same way: 0.7195) |

## The stoppage switch with v14's branch A (`check_v14_switch`)

| bench | v13's A | v14's A | B | the switch with v14 |
|---|---|---|---|---|
| normal | 0.7162 | **0.7241** | 0.6938 | 0.7241 (A on 10/10): part 1 passes |
| 10 % of stoppages stressed | 0.7054 | 0.7124 | 0.6921 | 0.7124 (A on 10/10) |
| 25 % | 0.6913 | 0.6979 | 0.6958 | 0.6993 (B on 5): above both branches |
| 50 % | 0.6573 | 0.6699 | 0.6983 | 0.6983 (B on 10) |
| stressed | 0.5982 | 0.6107 | 0.7057 | 0.7057 (B on 10, +0.095): part 2 passes |

- **Part 3 passes at every level** (the switch is within 0.005 of the better branch, or above both).
- **v14's A is better than v13's on every variant**, the stressed ones included (+0.007 to +0.013).
- **The threshold rule, recomputed with v14's A, would put the cut at 0.168** (left out: 0.155–0.190) against the
  deployed 0.1445: a stronger A means B starts winning a little later. The deployed threshold passes every part, so
  it stays; retuning it on these same benches would be a choice made after seeing the results.
- **Part 4 (SkillCorner, `check_v14_switch skillcorner`):** v14's A scores 0.3196 there (v13's 0.3123); the
  switch picks B on 20 of 20 (0.4077, unchanged from v13): **+0.088 over v14's A, better on 20 of 20** (the bar
  asks +0.02 on 15). So v14 on SkillCorner is v13: 0.408.

## Our match

Run 2026-11-12 by Claude under Jose's rule (only the path, the SHA-256 and the format check's pass line printed; the
file and its counts not looked at): three runs (default threads twice, `OMP_NUM_THREADS=1` once) give the same file,
SHA-256 `3efbef53e4394c318815fa33d8284af9ab6f675667c09008dfdba2e51a54a7c4`, and the format check passes (199,319 rows).

**That is v13's SHA-256.** v14 changes only branch A, so an identical file means **the stoppage switch sends our match
to branch B** (`augment`'s chain), where v14 and v13 are the same. Nobody ran `--show-branch`; the equal hashes
told it. So **v14's gain doesn't reach our file**: our submission rests on branch B (bench 0.694, stressed bench
0.706, SkillCorner 0.408). The switch's choice stands as the algorithm made it; nothing is overridden by hand.

## Failure modes

1. **The two settings were spotted on Metrica's 3 games**, so their gain there (+0.016) is partly in sample; the
   search's held-out estimate is +0.008. The DFL and PathCRF-unseen matches took no part in the choice and don't
   lose (+0.005, +0.014).
2. **Snap and move were v9's and v11's contributions** (restarts on the seen pass, changes on the new team's touch).
   With the timing stage after them they no longer pay on the bench; on a match whose touches are seen better than
   the bench's damaged ones, they might have.
3. **More weight on the GRU** means more weight on a model trained on 7 matches; robust's trees are the more
   conservative half of the start.
4. **v13's failure modes apply** ([v13](v13-temporal.md)), the switch's in-between zone included.
