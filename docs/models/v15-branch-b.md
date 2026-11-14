# Model v15: v14 with a new branch B, the chain our match runs

| | |
|---|---|
| **status** | **built 2026-11-14, Jose's go-ahead ("Yes, adopt v15").** The network is bit-identical to the lab's; the port gives the lab's letters on 3 bench units (0 frames in float32, 1 in float64); the built file reproduces the search's SkillCorner scores on all 20 matches |
| **what** | v14 ([v14](v14-search.md)) with branch B changed on bench set `all`: B's start is **0.92 `gru_b` + 0.08 `augment`** (v13/v14: `augment` alone), **snap and move off**, the decoder's and post stages' settings from a nested search, and **the post stages trained on stressed copies** of the bench. Branch A, the stoppage switch, its threshold (0.1445) and the IDSSE-only set are v14's |
| **why** | v14's file on our match equals v13's: the switch sends our match to branch B, so B sets our score. B was built as a robustness fix and never tuned: no temporal network, v12's settings (chosen for branch A's bench), post stages trained on the **normal** bench though B only runs where the ball looks stressed. Lab round 8 ([../lab/branch-b.md](../lab/branch-b.md)) |
| **bench** (B's bench: 25 %, 50 % and fully stressed; each match's stage models trained without it) | **0.7329 against v13's B 0.6975 (+0.0353), better on 10/10**; nested, held out: **+0.0334, 10/10**; every variant +0.034 or more; Metrica's 3 +0.023 to +0.073. Normal bench (where A runs): 0.7233 against v13's B 0.6894 and v14's A 0.7241 |
| **SkillCorner** (20 real matches, deployment models) | 0.4085 against 0.4077 (+0.0008, 11/20): about nothing |
| **new code** | `possession/v15.py`, `possession/models/v15_b_all.joblib` (1.9 MB), `bench/train_v15.py`, `bench/check_v15.py` (the port and SkillCorner), `bench/check_v15_switch.py` (the switch's bar with v15's B); the lab: `bench/lab_branch_b.py`, `bench/search_b.py` |
| **run** | `OMP_NUM_THREADS=2 python -m bench.train_v15` (~15 min), `python -m bench.check_v15 port` (~5 min), `python -m bench.check_v15 skillcorner` (~10 min), `python -m bench.check_v15_switch --trial 1015` (after `search_b confirm` and `skillcorner`); then `python -m possession.v15` on our match: path, SHA-256 and format check only (`--show-branch` for Jose) |

## What changes, and what doesn't

| | v13 / v14's branch B | v15's branch B (search trial 1015) |
|---|---|---|
| start | `augment` (robust's trees retrained with SkillCorner-like stoppage balls) | **0.915 P_`gru_b` + 0.085 P_`augment`** (v14's mix: both through float32, clipped, rounded); `gru_b` is lab_temporal's GRU trained on the 7 DFL matches as they are **and** under `augment`'s training stress |
| P(dead) | as is | × 0.909, rows renormalised |
| decoder | v12's (switch prices 4.0 / 25, minimum possession 10) | switch prices 2.82 / 48.7, minimum possession 7 |
| v9's snap, v11's move | on | **off** |
| short, segments, timing | on, v12's thresholds | on; short p_min 0.340, segments flip 0.910 / merge 0.661, timing p_min 0.450 |
| what the post stages are trained on | `augment`'s letters on the normal bench | **the trial's own chain's letters on the bench under `augment`'s training stress** (stoppage balls from SkillCorner's even-indexed matches; the test variants use the odd-indexed ones) |
| branch A, the switch, threshold 0.1445, kick-off rule, bench set `idsse` | | unchanged (v14's) |

The search's top is flat (the best 10 of 1,500 trials within 0.002), and it agrees on two things: weight the
network (rank correlation with the score +0.85) and turn snap and move off (79 of the top 100). **Most of the gain
is the network** (+0.028 to +0.030 with B's old settings); the searched settings add about +0.006.

## Built: what the checks say

| check | result |
|---|---|
| `gru_b` retrained on the 7 DFL matches (`train_v15`) | **bit-identical** to the lab's fold `net_metrica1.pt` (the file isn't written otherwise) |
| **the port with fold models** (`check_v15 port`: J03WR9 stressed, metrica1 50 %, J03WOH 25 %) | float32: the lab's P and **0 frames** differing on all 3; float64 (as deployed): P one rounding step off, **1 frame** differing on J03WR9, the same scores |
| the built file on SkillCorner's 20 (`check_v15 skillcorner`) | **0.4085**, `search_b`'s per-match scores reproduced to 5e-6; the switch picks B on 20/20 |

## The stoppage switch with v15's B (`check_v15_switch --trial 1015`)

| bench | v14's A | v13's B | **v15's B** | the switch picks B on | switch | switch − the better branch |
|---|---|---|---|---|---|---|
| normal | 0.7241 | 0.6894 | **0.7233** | 0 | 0.7241 | ±0 |
| 10 % of stoppages stressed | 0.7124 | 0.6921 | 0.7255 | 0 | 0.7124 | −0.013 |
| 25 % | 0.6979 | 0.6949 | 0.7289 | 5 | 0.7115 | −0.017 |
| 50 % | 0.6699 | 0.6958 | 0.7319 | 10 | 0.7319 | ±0 |
| stressed | 0.6107 | 0.7019 | 0.7379 | 10 | 0.7379 | ±0 |
| SkillCorner (20) | 0.3196 | 0.4077 | 0.4085 | 20 | 0.4085 | ±0 |

- **Parts 1, 2 and 4 pass; part 3 fails at 10 % and 25 %**, not because B lost anywhere: B got better, and the
  threshold, placed where the old B overtook A, now keeps A on some matches where the new B is the better branch.
- **The threshold stays at 0.1445**, as fixed before scoring. Lowering it (round 7's rule rerun: 0.0485) or B
  everywhere would fix part 3, but both send Metrica's normal games to B, which loses on all 3 of them; and
  **neither changes our file**: the switch already takes B on our match, and would at any lower threshold.

## Our match

Run 2026-11-14 by Claude under Jose's rule (only the path, the SHA-256 and the format check's pass line printed; the
file and its counts not looked at; no `--show-branch`): three runs (default threads twice, `OMP_NUM_THREADS=1` once)
give the same file, SHA-256 `209a2081beae40b1a7971fa958f87e7ca495536eee673af10a37df87069b7f5e`, and the format check
passes (199,319 rows).

**It differs from v14's file** (`3efbef53…a7c4`, which equals v13's), as it should: v15 changes only branch B, and
v14's equal hash had already shown that the switch takes B on our match. So **round 8's gain is the first since v13
that reaches our file**. The switch's choice stands as the algorithm made it; nothing is overridden by hand.

## Failure modes

1. **The gain is measured on a synthetic ball.** B's bench is the 10 bench matches with SkillCorner's stoppage
   balls pasted into their true stoppages. On SkillCorner's own 20 broadcast matches (real tracking of the kind B
   runs on, another labelling convention: every version scores about 0.41 there) the new B gains +0.0008, 11/20.
   What carries the case is the bench, mostly Metrica's 3 games (our labels' convention): +0.023 to +0.073.
2. **The honest size is the nested +0.033**; trial 1015's +0.035 is in sample, and the nested picks leak a little
   (the other matches' stage models saw the held-out match, as in search_metrica).
3. **B now leans on a network trained on 7 matches** (weight 0.92); `augment`'s trees are the more conservative
   part. On the normal bench this B is close to A (0.7233 against 0.7241), so if the switch sent our match to B for
   a reason other than a moving stoppage ball, B costs little there; v13's B cost 0.035.
4. **The post stages learned from stressed copies.** If our seen ball behaves in stoppages unlike SkillCorner's,
   they were trained on the wrong kind of match; the search found the training data a weak choice (top 100:
   stressed 42, both 34, normal 24).
5. **The switch's in-between zone** (10–25 % of stoppages stressed) now keeps A where B would be better; on our
   match the switch takes B, so this concerns other matches.
6. **v13's and v14's failure modes apply** ([v13](v13-temporal.md), [v14](v14-search.md)).
