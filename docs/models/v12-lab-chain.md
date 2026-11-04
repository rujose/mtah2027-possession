# Model v12: the lab's chain, ported to our match

| | |
|---|---|
| **status** | **built 2026-11-04** with all five recommended decisions; the port reproduces the lab on every frame of two bench matches. Supersedes the v12 and v13 proposed on the [v11 page](v11-touch-line.md): the lab built and judged both ideas, and much more |
| **what** | a learned per-frame model in place of v1's hand weights, the decoder, v9's snap and v11's move, then the lab's post stages: **robust → short → segments → timing** (bench set `all`), or **robust → short → segments → recall → dead** (bench set `idsse`) |
| **why** | [docs/lab.md](../lab.md): four rounds, 10 builders, every stage kept only if it beat the chain without it on the bar |
| **bench** | **0.7033 held out against v11's 0.5727 (+0.1306), better on 10/10**; IDSSE set 0.6699 against 0.5625 (+0.1074), 7/7 |
| **new code** | `possession/v12.py`, `possession/models/v12_all.joblib` and `v12_idsse.joblib` (1.3 and 1.5 MB), `bench/train_v12.py` (trains them), `bench/check_v12.py` (the port reproduces the lab); the lab's stage code in `bench/lab_*.py`, `bench/lab.py`, `possession/lab_learned_features.py` |
| **run** | `python -m bench.train_v12 [--set idsse]` (~15 min each), `python -m bench.check_v12` (~20 min), then `python -m possession.v12` on our match: it prints only the path, the SHA-256 and the format check |

## What it does, stage by stage

Every stage was built by a lab builder; its page has the why of each constant.

| stage | what it does | settings (`all`; `idsse` where different) | page |
|---|---|---|---|
| **inputs** | v10's pieces on our tracking, as the bench builds them: keeper repair, v1's clues, the dead-ball model (v4), the restart model (v6), PathCRF (v7); out come v6's and v7's costs, P(dead), P(home restarts), the restart clue, PathCRF's P(home) | v10's `SETTINGS[BENCH_SET]` | [v10](v10-clean-base.md) |
| **robust** | a classifier gives P(home / away / dead) for every frame, from ~70 features (each team's distance to the ball, the ball's speed and course, touches, passes, the inputs above). Any seen ball standing still (< 0.3 m/s) for at least 1 s counts as missing, in training and here. Its −log P is the decoder's cost | trained on the bench's IDSSE matches; team switch 4, dead switch 25, minimum possession 10 frames | [robust](../lab/robust.md), [learned](../lab/learned.md) |
| decoder, snap, move | the min-length decoder, then v9's restart snap and v11's touch move | v9's and v11's | [v9](v9-restart-pass.md), [v11](v11-touch-line.md) |
| **short** | inside a possession, a stretch of 0.5–5 s where the other team clearly touches the ball (or touches it last before a stoppage) becomes theirs, if a logistic regression on 10 pieces of touch evidence is at least 50 % sure | p_min 0.5 | [short](../lab/short.md) |
| **segments** | a classifier checks each possession: merged into the stoppage if it's likely spurious, team flipped if it's likely wrong | flip 0.6, merge 0.7 (`idsse`: no PathCRF features, flip 0.8) | [segments](../lab/segments.md) |
| **timing** (`all`) | each change of team moved to the candidate frame within 3 s a ranker trusts most (touches, the new team reaching the ball) | p_min 0.35 | [timing](../lab/timing.md) |
| **recall** (`idsse`) | short stretches for the other team inside a possession (its player within 1 m and nearest, touches, PathCRF), written in if a classifier is at least 40 % sure | p_min 0.4 | [recall](../lab/recall.md) |
| **dead** (`idsse`) | stretches a dead-ball classifier (ball and players) is sure about become D | threshold 0.3, at least 4 s | [dead](../lab/dead.md) |

## The bench (leave one match out; each stage against the chain without it)

| | `all`: held out | vs v11 | matches better | `idsse` | vs v11 |
|---|---|---|---|---|---|
| robust alone | 0.6729 | +0.1001 | 10/10 | 0.6365 | +0.0740, 7/7 |
| + short | 0.6900 | +0.0171 | 10/10 | 0.6598 | +0.0232, 6/7 |
| + segments | 0.6989 | +0.0089 | 10/10 | 0.6643 | +0.0045, 6/7 |
| + timing / + recall, dead | **0.7033** | +0.0044 | 8/10 | **0.6699** | +0.0046, +0.0009 |

By source (`all`): IDSSE 0.6674 (+0.100, 7/7), Metrica 0.7870 (+0.201, 3/3), the 4 matches PathCRF
never saw 0.7617 (+0.178, 4/4). macroF1 0.9182 against 0.8705.

## Porting it to our match

1. **The inputs exactly as the bench made them.** The models learned from the bench's inputs, so
   v12 builds ours the same way: v10's cost path with **Smart Tagging's team hints off** (no bench
   match has tags, so no model has seen them) and the whistle and set-piece window off (as in v10).
   The 2nd-half kick-off clue and the keeper repair stay (both are in the bench's path; the repair
   never fires there).
2. **Deployment models, trained once on every bench match of the set** (`bench/train_v12.py`),
   saved in `possession/models/` and committed (trained on external data only, as v4's): the robust
   classifier (IDSSE matches), then each post stage trained on the other stages' output on the
   bench matches **out of fold**, as in the lab, so what it learns from looks like what it will see.
3. **`possession/v12.py`**: `predict(tracking, tags, halves, pitch, bench_set)` builds the inputs,
   wraps them like a lab match and runs the chain with the saved models. `main` writes
   `evaluation/results/v12.csv` and prints only the path, the SHA-256 and the format check. It
   reuses the lab's stage code (`bench/lab_*.py`) so the port can't drift from what was judged.
4. **`bench/check_v12.py`**: on bench matches (J03WR9 and metrica1), v12's code path with fold models
   (trained without the match) must reproduce the lab's chain letters **frame for frame**; then the
   deployment path runs end to end on a bench match, twice, with the same output.
5. **Our match:** `python -m possession.v12` twice (the SHA-256 must repeat), then the format check.

## Built: what the checks say

- **The models** (`bench/train_v12.py`): trained on every bench match of the set, each post stage
  on the out-of-fold letters of the stage before it, with each stage's setting chosen on all the
  set's matches. Run out of fold with those settings, the chain scores **0.7036** on `all` (the
  lab's 0.7033, which used each match's own fold setting) and **0.6688** on `idsse` (lab 0.6699).
- **The port** (`bench/check_v12.py`): on J03WR9 and metrica1, v12's own code (robust's features
  computed directly, the match wrapped as ours is), with models trained without the match as the
  lab's were, gives the lab's letters on **every frame** (0 of 146,811 and 0 of 145,007 differ);
  robust's P agrees with the lab's to 3·10⁻⁸ (the lab saved it as float32).
- **The deployment models** give the same letters twice on a bench match.
- **Our match:** `python -m possession.v12` run twice gives the same file (SHA-256
  `bfd2548d…a88e24`), and the format check passes (199,319 rows). Its output was not looked at.
- **The inputs** (`v12.inputs`) run on a bench match's damaged tracking; v7 decoded from them
  agrees with the lab's v7 on 96.5 % of graded frames. Not 100 %: v12's inputs come from the
  dead-ball and restart models trained on every bench match, the lab's from models trained without
  that match.

## Failure modes

1. **A ball that moves in our stoppages.** Robust blanks a ball that stands still, so IDSSE's frozen
   dead-time ball can't teach it a convention our feed may not share. Our ball is missing 83 % of
   the time players stand still ([bench.md](../bench.md)), so "no ball" transfers. But if our *seen*
   ball is carried around during stoppages, robust may call some of them live, as learned would.
   Bounded: dead-ball errors were +0.022 of the budget before the lab. A label-free check exists
   ([learned.md](../lab/learned.md)); nobody has run it.
2. **The bench flatters it.** Several builders explored their designs on all 10 matches' truth
   (short, recall, timing), and many candidates were tried on the same matches. Leave-one-match-out
   protects settings and weights, not those choices. Expect less than +0.13 on our match.
3. **PathCRF in sample.** PathCRF was trained on 6 of the 7 IDSSE matches; robust and segments read
   it. The 4 matches it never saw hold (+0.178), and our match is one it never saw.
4. **Our inputs come from models trained on all bench matches**, not out of fold, so they're a bit
   sharper than what the classifiers learned from. Small, the same for every version since v4.
5. **Smart Tagging off.** v10 used its team hints on our match; v12 doesn't. The bench can't say
   what they were worth.
6. **Metrica's samples** (audit item D): the `all` chain trains on them; `BENCH_SET = "idsse"`
   switches to the IDSSE-only chain and models.
7. **Only 41 % damage was tested** in the lab (the earlier versions were also checked at 30 % and
   55 %).

## Decisions for Jose

| # | question | recommended | why |
|---|---|---|---|
| 1 | Start from robust, learned or learned's team half | **robust** | best on both sets, and it can't use IDSSE's frozen ball |
| 2 | The full greedy chain per set, or only robust → short → segments (passes on both sets) | **the full chain per set** | +0.004–0.006 more; each tail stage passed the bar on its set; the port reuses the same code |
| 3 | Smart Tagging hints off in v12 | **off** | no model has seen them; keeping them would feed the classifiers inputs unlike any they learned from |
| 4 | Commit the deployment models (external data only) | **yes** | the public repo has to regenerate the CSV byte for byte |
| 5 | The port must reproduce the lab frame for frame on two bench matches before our match runs | **yes** | the lab's numbers only mean something for code that is the lab's |
