# Model v7: v6 + PathCRF, who has the ball while the ball is missing

| | |
|---|---|
| **status** | **built 2026-10-30**, as proposed, with the six recommended choices |
| **run** | `python -m possession.v7` → `evaluation/results/v7.csv` (~2 min; downloads the 19 MB weights once, hash-checked). `python -m bench.eval_pathcrf` re-does the bench evaluation and search (~15 min) |
| **output** | SHA-256 `17d8d8d29b0740e4d2895320870ebafd7f28f0f1c3866f632c02127d7aaffb08` |
| **added while building** | **PyTorch Geometric 2.8.1**, because PathCRF's unchanged model code imports it at the top of `models/utils.py`. Pinned with everything it pulls in, in `requirements-team.txt` |
| **change from v6** | one new clue: a team-possession estimate from PathCRF, a published model that reads only the players' movement. Its weight, and whether it applies only while the ball is missing, are chosen on matches PathCRF never saw |
| **new code** | `third_party/pathcrf/` (PathCRF's inference code, MPL-2.0, unchanged), `possession/pathcrf_adapter.py` (our tracking → its input; its output → a per-frame team probability), `bench/eval_pathcrf.py`, `possession/v7.py` |
| **new dependency** | PyTorch 2.15.0, pinned (installs on our Python 3.14; it pulls in Jinja2, networkx, sympy, filelock, fsspec and others) |
| **effort** | ~2 days, most of it the adapter |

## Why

**Measured on the bench first** (revised v6, out of sample, 41 % damage, all 10 matches):

| | ball seen | ball missing |
|---|---|---|
| share of live frames | 77 % | 23 % |
| v6 names the wrong team | 10.7 % | **26.4 %** |
| true changes of possession (3,112 in all) happening here | 74 % | 26 % |
| … that v6 catches, to the right team, within 2 s | 59 % | **6 %** |

- **Inside ball gaps v6 is close to blind.** It keeps the last team (football's "a loose ball stays
  with the last team"), so a change of possession while the ball is missing is found late, or only
  at the next stoppage.
- **That's 42 % of all its wrong-team frames**, from 23 % of the live time.
- **It's also the open point (c) of the v6 diagnosis:** when the restart team loses the ball within
  seconds, the next possession is missed.

**PathCRF** (Kim et al., KDD 2026; [../research/models-literature.md](../research/models-literature.md) §1.8)
is the only published, pretrained model that infers possession from players alone.
- **Input:** the 22 players' movement, with no ball.
- **Output:** at every step (5 per second), either "player u has the ball" or "the ball is travelling
  from u to v". A CRF, a sequence model like our decoder, forbids impossible sequences.
- **Reported:** 92.4 % frame-level team accuracy in play, on one held-out match of clean tracking.

## What we know about it (read from its repository, 2026-10-30)

- **Repository:** github.com/hyunsungkim-ds/pathcrf, MPL-2.0.
  - Five trained models; the best is `saved/140` ("dynamic sparse CRF", 3.43 M parameters,
    `state_dict_best.pt`, 19.3 MB).
  - Inference is in `inference.py`, with a walkthrough in `tutorial.ipynb`.
- **Training:**
  - on the same 7 DFL (IDSSE) matches as our bench;
  - events re-synchronised with ELASTIC; 5 fps; 10 s windows;
  - pitch flipping as augmentation; 11 players per team;
  - leave-one-match-out, with `saved/140` being fold 6 (`"lomocv": true, "cv_fold_idx": 6`).
- **Fold 6** should be J03WR9, the match in the README's animation. **To verify** from its split code
  before relying on it.
- **Dependencies:** its `requirements.txt` lists numpy, pandas, tqdm, scipy, scikit-learn, matplotlib
  and pytest. The code imports **torch**, which isn't listed.

**The catch for testing:** it was trained on 6 of our 7 IDSSE matches. Scoring it on those would
flatter it. **Only 4 bench matches are unseen:** J03WR9 (if fold 6 is confirmed) and Metrica's 3
games.

## How it works

### Piece 1: run PathCRF unchanged (`third_party/pathcrf/`)

- **Vendored:** its model and inference code goes in our repository, unchanged, with its licence
  file. MPL-2.0 allows this as long as those files stay under MPL-2.0.
- **Weights:** downloaded from its repository at a pinned commit, SHA-256 checked, the way IDSSE
  is (decision 3).
- **Reviewed before running:** we read the vendored code before it ever runs. It's third-party
  code.
- **Inference:** on the CPU, one thread, fixed seed, so it's deterministic.

### Piece 2: the adapter (`possession/pathcrf_adapter.py`)

PathCRF expects what DFL's clean tracking gives: 11 continuous tracks per team, a known goalkeeper,
no gaps. We have 46 anonymous slots in pieces, players missing, identity swaps. The adapter:

1. **Follows PathCRF's own preprocessing exactly** (to be read from `datatools/`): its coordinates,
   its 8 numbers per player (two flags plus x, y, vx, vy, speed, acceleration), and its 4 "outside"
   nodes.
2. **Stitches slot pieces into 11 tracks per team.**
   - When one piece ends and another of the same team starts within ~2 s and a few metres, they're
     the same player.
   - Over each 10 s window, keep the 11 most present tracks and fill short holes by straight-line
     interpolation.
   - A window with fewer than 11 trackable players per team gets no answer (the clue stays silent).
3. **Finds each team's goalkeeper:** the track deepest toward its own goal over the window.
4. **Feeds only live stretches** (v6's own dead-ball decision says where), as PathCRF was trained.
5. **Turns its output into P(home has the ball)** at 25 fps.
   - "Player u has it" counts for u's team.
   - "The ball travels u → v" counts for u's team, because a pass in flight stays with the passer, as
     in our convention.
   - Where it gives probabilities rather than one path, we use them; otherwise the path, softened.
     Upsampled 5 → 25 fps.

### Piece 3: check that it works before using it (`bench/eval_pathcrf.py`)

1. **Reproduce:** on **clean** J03WR9 (the unseen IDSSE match), our adapter plus PathCRF should get
   close to the paper's 92 % team accuracy in play. If it doesn't, the adapter is wrong; stop and fix
   that first.
2. **Under our damage:** on J03WR9 and Metrica's 3 games damaged like ours, its team accuracy
   **inside ball gaps**, against v6's 26.4 % wrong there.
3. **Changes caught:** the share of true changes of possession inside gaps that it places within
   2 s, against v6's 6 %.

### Piece 4: into the costs (`possession/v7.py`)

The same form as revised v6:
- **only the unlikely team pays, by the odds:** weight × log(P / (1 − P)), on A when home is likelier
  and on H when away is;
- **no cost on D,** so it can't make stoppages cheaper or dearer.

### Piece 5: settings, chosen on the 4 unseen matches

| setting | candidates | why |
|---|---|---|
| weight | 0.25, 0.5, 1, 2 | as before |
| where it applies | only while the ball is missing; everywhere in live play | the measured gap is inside ball gaps; with the ball seen, v6 is already decent (10.7 % wrong) |

- **Procedure:** leave one match out over the 4 unseen matches (v3's tie rule).
- **Reported but not used to choose:**
  - the 6 IDSSE matches PathCRF trained on, labelled as flattering;
  - the two-sided accounting from v6;
  - changes of possession caught inside gaps;
  - 30 % and 55 % damage.

Everything else stays v6's.

## What to expect

- **On the bench:** fewer wrong-team frames inside gaps, and more changes of possession caught there.
  The spellF1 gain depends on how much accuracy survives the move from clean tracking to ours, which
  is the main unknown.
- **On our match** (no labels): how often v7 changes team inside gaps against v6, and the possession
  count.

## Expected failures and risks

1. **Domain shift**, the biggest risk. 92.4 % was measured on clean optical tracking. Ours has
   swaps, missing players and duplicates, and the stitching will make mistakes. The bench measures
   how much is lost, but only on 4 matches.
2. **Only 4 unseen matches** to choose 2 settings and judge the gain. Results will be noisy; the
   6 seen matches can't stand in for them.
3. **It needs 11 tracks per team.** Where our tracking has fewer (on average 20.5 of 22 are visible),
   windows go silent.
4. **New heavy dependency** (PyTorch) and third-party code. It's pinned and vendored, but a
   different machine could still differ in the last digits. Rounding P to 3 decimals guards the
   labels, as in v4, and committing its small output is the fallback (#14; decision 2).
5. **It's trained on the Bundesliga's DFL convention,** whose possessions run ~20 % shorter than
   Metrica's (the convention check). The bench's Metrica games show how much that matters.

## Decisions for Jose before building

| | choice | recommended | why |
|---|---|---|---|
| 1 | Use PathCRF's published weights, no retraining | **yes** | Retraining is ~50 epochs × ~10 min per fold on a high-end GPU; days on the M2 |
| 2 | Run PathCRF at reproduction time (CPU, one thread), with committing its small per-step output as the fallback if it isn't byte-stable | **yes** | Keeps the pipeline honest end to end; #14 allows the fallback |
| 3 | Vendor its inference code (MPL-2.0) and download the weights at a pinned commit with a hash check | **yes** | The organisers can access everything; no 19 MB file in our git history |
| 4 | Choose settings only on the 4 matches it never saw; report the other 6 separately | **yes** | Scoring it on its own training matches would flatter it |
| 5 | Add PyTorch 2.15.0, pinned | **yes** | PathCRF needs it |
| 6 | Name it v7 | **yes** | One change per version |

## Results (2026-10-30)

### Facts confirmed while building

- **Fold 6 is J03WR9.** PathCRF's `train.py` sorts the match files and `cv_fold_idx` picks the test
  file; validation is the file before it, J03WQQ, which chose the checkpoint. So J03WR9 and
  Metrica's 3 games are unseen ([../../third_party/pathcrf/VENDORED.md](../../third_party/pathcrf/VENDORED.md)).
- **The vendored code** has no network calls, shell calls, `eval`, `exec` or `pickle.load`. The
  weights are loaded weights-only.

### Piece 3: does it work?

**1. Reproduction.** On the **clean** J03WR9, our adapter plus PathCRF names the right team on
**91.0 %** of live frames, against the paper's 92.4 %. The adapter is right. (35 s on the M2's CPU.)

**2. Under our damage** (41 %, out of sample, over the stretches v6 calls live):

| | frames | PathCRF right | v6 right |
|---|---|---|---|
| unseen matches, **ball missing** | 38,604 | **85.7 %** | 69.5 % |
| unseen matches, ball seen | 293,388 | 91.0 % | 90.6 % |
| seen matches (flattering), ball missing | 126,066 | 88.8 % | 69.2 % |
| seen matches (flattering), ball seen | 319,347 | 90.0 % | 87.1 % |

**3. Changes of possession while the ball is missing** (unseen matches): v6 catches 12 of 152
(8 %) within 2 s; **v7 catches 84 (55 %).**

### Piece 5: the settings

All 4 folds (leave one unseen match out) picked **weight 0.5, applied on every live frame it
answers** (not only while the ball is missing):

| held-out match | v7 | v6 |
|---|---|---|
| J03WR9 | 0.569 | 0.452 |
| metrica1 | 0.571 | 0.560 |
| metrica2 | 0.553 | 0.529 |
| metrica3 | 0.567 | 0.461 |
| **mean** | **0.565** | **0.501** |

**Better on 4 of 4, by +0.064**, the largest gain since v3. On the 6 matches PathCRF trained or
validated on (flattering): 0.549 against 0.407.

**Two-sided accounting** (unseen matches, scores 0–5 per real possession):
- after restarts 986 → 1,097;
- all others 2,442 → **2,794**;
- matched at IoU 0.5: 688 → 796 of 1,423;
- possessions reported: 1,322 → 1,330.

The gain is spread across all possessions, not just restarts.

**Reproducibility:** a second run of the whole evaluation gave byte-identical results.

### On our match (no ground truth involved)

| | v6 | v7 |
|---|---|---|
| possessions | 439 | **347** |
| median / mean length | 4.0 s / 7.7 s | 4.6 s / 9.7 s |
| dead ball | 46.5 % | 46.9 % |
| frames that differ | — | 13,558 (8.5 % of graded) |

- **PathCRF answers on 53.5 % of graded frames:** v6's live stretches. Only 11.8 % of those have the
  ball missing, since on our match the ball goes missing mostly while play is stopped.
- **Fewer possessions (−21 %).** On the unseen bench matches the count barely moved (1,322 →
  1,330). On ours it drops. v7 now reports ~295 possessions per 90 minutes, a little below
  Metrica's sample games (304–358 by their events).

**Domain-shift check** (label-free: how PathCRF agrees with our own evidence):

| PathCRF agrees with… | bench, unseen | bench, seen | our match |
|---|---|---|---|
| v6's team, on live frames | 84.4 % | 81.2 % | 82.3 % |
| the team of the player right on the ball (v1's clue A) | **90.9 %** | 90.2 % | **86.0 %** |

- **Somewhat less consistent on our match:** about 5 points against the direct evidence, but it
  isn't broken.
- **No team bias:** Barcelona's share of live frames is 62.0 % by PathCRF against v6's 63.2 %.
- **Duplicated slots don't confuse it:** where more than 11 slots of a team are in use, agreement is
  86 %.

**What it means:** v7's gain on the bench is large and consistent. On our match, part of it may be
lost to the tracking difference, and the drop in possessions is the thing to watch. It's visible in
the viewer (the "v7: PathCRF" lane).

**Public 5 minutes** (pipeline check only, not a ranking): macroF1 0.857, spellF1 0.606, 18 possessions (truth 17); v6 0.818 / 0.442 / 21. It points the same way as the bench, but 17 possessions can't rank versions, and nothing was chosen on it.

