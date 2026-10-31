# Model v4: v3 + a dead-ball model learned from the players, trained on the bench

| | |
|---|---|
| **status** | **built 2026-10-30**, as proposed, with the five recommended choices. The rule picked the trees (by 0.013), so scikit-learn is needed at run time |
| **run** | `python -m possession.v4` → `evaluation/results/v4.csv` (~1 min). `python -m bench.train_dead_ball` re-trains and re-searches (~10 min); the submission doesn't need it, because the trained model is committed (`possession/models/dead_ball_trees.joblib`, 0.5 MB) |
| **output** | SHA-256 `2ac7c203687f19bf8355c447c8cf8ef7712152a2e1e806b9ce1e9a4348cc5ca5`; byte-identical on re-run |
| **change from v3** | v1's two hand rules for "players stopped / players active" are replaced by a probability that the ball is dead, from a model trained on the 10 bench matches (both sets). Its weight and the dead-ball decoder settings are chosen on the bench. Everything else stays |
| **new code** | `possession/features.py` (per-frame features from the players), `bench/train_dead_ball.py` (training, checks, search), `possession/v4.py` |
| **new files committed** | the trained model (small, trained on other matches only) and its settings |
| **new dependencies** | scikit-learn 1.9.1 and what it pulls in (scipy, joblib, threadpoolctl, narwhals, cloudpickle), pinned (decision 2) |
| **effort** | ~1–1.5 days |

## Why

**The bench says v3 misses about half the stoppages.** Measured at our damage level with v3 as it
is (exploration, other matches only):

| | IDSSE (7) | Metrica (3) |
|---|---|---|
| dead-ball precision (when v3 says D, it is D) | 0.91 | 0.82 |
| dead-ball recall (share of dead frames v3 calls D) | **0.64** | **0.59** |
| real stoppages | 710 | 288 |
| … found (v3 says D on at least half of it) | 369 (52 %) | 111 (39 %) |
| … missed, median length | 341, **13.4 s** | 177, **9.2 s** |
| … missed entirely between two possessions of the same team, so two possessions merge | 65 | 25 |
| v3 dead stretches that are mostly live (false stoppages) | 9 | 17 |

So v3 rarely invents a stoppage but misses many real ones, and they aren't short.

**Why each missed stoppage costs us on spellF1:**
- **Between two possessions of the same team:** the two merge into one prediction, which can match at
  most one of them.
- **Between different teams:** the hand-over lands at the wrong time, which cuts the IoU of both
  neighbours.

**Today's evidence for "dead" is blunt:**
- ball out of the pitch, which needs the ball, and it is missing 41 % of the time;
- "everyone stopped": team mean speed under 1.0 m/s for 3 s, a threshold with no natural split
  ([../data.md](../data.md));
- set-piece tags and the whistle, which only our match has.

A stoppage where players keep jogging, like 03:06 in the public window, slips through all of them.

**Published evidence that players alone can do much better:** Lang et al. 2022 (Scientific Reports)
reach 92 % frame accuracy on in-play vs stopped from player positions only, finding ~80 % of
stoppages with start and end within ±2 s ([../research/models-literature.md](../research/models-literature.md)).
They had 102 hand-labelled matches; we have 10 with labels from their own data.

## How it works

### Piece 1: per-frame features from the players only (`possession/features.py`)

Computed the same way on every bench match (after damage) and on our match. **No ball features**
(decision 3): Metrica's sample games blank the ball in dead time, so a model that could see the
ball would learn "no ball = dead", which is false on our match (our ball is missing on 29.5 % of
frames while players move). Ball-out stays as its own clue, as in v1.

Player speed is computed from positions (distance over 0.16 s) on every match, ours included, so
the feature means the same thing everywhere. Our file's own speed agrees with that: median ratio
0.99 ([../data.md](../data.md)).

| group | features (per frame) | why |
|---|---|---|
| movement | mean, median and 90th-percentile speed of all visible players; share under 1 m/s; share over 4 m/s; speed gap between the two teams | stoppages are slow; restarts have one team still and the other repositioning |
| shape | spread along and across the pitch; distance between the team centres; each team's compactness (the larger and the smaller of the two) | set pieces bunch players; open play stretches them |
| set-piece layouts | players within 2 m of a touchline (busier side); within 2 m of a goal line; in a 6-yard box; in a penalty area; within 3 m of a corner; within 9.15 m of the centre spot; outside the pitch; share of each team on its own side of halfway (kick-off) | the Laws of the Game force these shapes at restarts |
| time context | for the movement shares and the layouts: the mean over the past and the next 1 s, 4 s and 20 s | a stoppage is a stretch, not a frame. Lang et al. found the ±4 s context mattered most. The future is allowed (#15) |

That's about 85 features. They're **order-free**: they never ask which slot is which player, so
identity swaps and anonymous slots don't matter. They're also **symmetric**: home and away can be
swapped without changing them, since being dead doesn't depend on which team is which.

Left out on purpose: **how many players are visible.** Our tracker may lose players at stoppages in
ways the bench's damage doesn't imitate, so this feature could mean something different on our
match.

### Piece 2: the model (`bench/train_dead_ball.py`)

- **Target:** dead (D) or live (H or A) on every frame of the 10 bench matches, at 41 % damage.
  Every 5th frame is used for training (0.2 s apart; neighbouring frames are near-copies), and all
  frames for prediction.
- **Two candidates** (decision 1), compared end to end:
  - **logistic regression:** a weighted sum of the features, squashed to a probability. Simple and
    deterministic, and inference is a dot product we can run in plain numpy;
  - **gradient-boosted trees** (scikit-learn's HistGradientBoosting): a sum of small decision trees,
    the same family as Lang et al.'s best (AdaBoost). Captures interactions, such as "slow **and**
    someone on the touchline".
- **Settings of the model itself:** library defaults with a fixed random seed; no tuning beyond the
  candidates above.
- **Out-of-sample probabilities:** for each bench match, the model is trained on the other 9 and
  predicts that one. Every probability used to choose anything below comes from a model that never
  saw that match.

### Piece 3: into the costs (`possession/v4.py`)

With P = the probability the ball is dead (kept between 0.01 and 0.99):
- against H and A: **w × −log(1 − P)** per frame;
- against D: **w × −log(P)** per frame.

These are the usual "how surprised would we be" costs: at P = 0.5 they are equal and say nothing; at
P = 0.9 being live costs 2.3 × w and being dead costs 0.1 × w.

**Replaced:** v1's "stopped" (1.0 against H and A) and "active" (0.5 against D), which were hand
rules for the same question.

**Kept:** ball out, someone on the ball, set-piece window, before the 2nd-half kick-off, the whistle,
and all of v3's decoder.

### Piece 4: settings chosen on the bench

| setting | candidates | why |
|---|---|---|
| model weight `w` | 0.25, 0.5, 1, 2 | from "a nudge" to "as strong as ball-out" (2.0 per frame) |
| dead-ball switch price | 5, 10, 25, 50 | re-chosen, since dead-ball evidence is what it balances |
| minimum stoppage | 1, 25, 50 frames | a model that sees stoppages better may now benefit from one; ~1 % of real stoppages are under 2 s |
| model family | logistic, trees | decision 1 |

The team price (0.5) and minimum possession (20 frames) stay v3's: one change at a time.

**The procedure is v3's:**
- leave one match out over all 10;
- best mean spellF1, and the fewest possessions among those within 0.01 of the best;
- **the family:** logistic regression unless the trees beat it by more than 0.01.

**Reported but not used to choose:**
- dead-ball precision and recall and stoppages found, against v3's table above;
- the 30 % and 55 % damage levels, using the model trained at 41 %;
- merged labels;
- per set, IDSSE and Metrica.

**For the submission:** the model is trained on all 10 matches and the settings are chosen on all
10. The trained model is committed, so the submission doesn't need the bench.

### Piece 5: byte-identical on any machine

- Inference runs on one CPU thread.
- P is rounded to 3 decimals before it becomes a cost, so a last-digit floating-point difference on
  the organisers' machine can't flip a label.
- scikit-learn and its dependencies are pinned in `requirements-team.txt`.
- If logistic regression wins, its weights are stored as plain numbers and inference needs no
  scikit-learn at all.

## Every setting, and where its value comes from

| setting | value | source |
|---|---|---|
| features | as listed above | reasoning: the Laws of the Game, Lang et al.; nothing fitted |
| speed for features | distance over 0.16 s | same on every match; matches our file's own speed |
| training frames | every 5th | neighbouring frames are near-copies |
| clip of P | 0.01–0.99 | so one frame can't cost more than ~4.6 × w |
| rounding of P | 3 decimals | reproducibility |
| family, `w`, dead price, minimum stoppage | chosen | bench, leave-one-match-out, tie rules |
| everything else | as in v3 | [v3-decoder.md](v3-decoder.md) |

## What to expect on our match (no labels involved)

- **More dead ball** than v3's 39.9 %, and stoppages that start and end closer to the play.
- **Possessions that v3 merged across a stoppage become two**, so the count may rise somewhat from
  429. Reported next to v3's.
- **A label-free check:** of the 86 whistle events, how many v4 marks dead within 3 s, against v3.
  This is evidence of stoppages we don't need labels to see.
- **The public 5 minutes:** run as a pipeline check only.

## Expected failures and risks

1. **Slow open play looks like a stoppage**: a goalkeeper holding the ball, a slow build-up. Lang
   et al.'s stoppage precision was 81 %. On the bench we'll see the false stoppages count against
   v3's 9 and 17.
2. **The bench's damage is imitated, not real.** Players-only features dodge the ball-missing mismatch
   (the bench's known gap), but our tracker's identity swaps and player losses may still move
   features like spread and compactness in ways the bench didn't see.
3. **Two definitions of "dead".** DFL marks D 0.12–0.16 s after the ball crosses a line; Metrica's
   events mark the end of the BALL OUT event. The organisers' truth follows Metrica's corrected
   tracking. Small edge differences, mostly.
4. **Ten matches.** Enough for ~85 features with a simple model; trees with defaults may overfit,
   which the held-out scores will show.
5. **New dependencies** (decision 2), and a pickled model tied to the pinned version if the trees win.
6. **Tags and whistle can't be benched**, as before. Their weights stay as in v3.

## Decisions for Jose before building

| | choice | recommended | why |
|---|---|---|---|
| 1 | Model family | **try both, keep logistic regression unless trees win by more than 0.01** | Simpler, deterministic and dependency-free at run time; trees only if they clearly earn it |
| 2 | Add scikit-learn 1.9.1 (pinned) | **yes** | Needed to train either candidate fairly; at run time only if the trees win |
| 3 | Players-only features, no ball | **yes** | Metrica's blanked ball would teach "no ball = dead"; ball-out stays a separate clue |
| 4 | Replace "stopped" and "active" with the model | **yes** | They are hand rules for the same question; keeping both would count the same evidence twice |
| 5 | Name it v4 (v3 + learned dead ball) | **yes** | One change per version, as before |

## Results (2026-10-30)

### The model on its own (out of sample, all 10 bench matches, 41 % damage)

| | dead-ball precision | recall | F1 (frames, P > 0.5) |
|---|---|---|---|
| logistic regression | 0.880 | 0.873 | 0.877 |
| gradient-boosted trees | 0.898 | 0.912 | **0.905** |

79 features, about 29,000 training frames per match. That's close to Lang et al.'s 92 %, on
damaged tracking and with labels from two providers.

### End to end, leave one match out (spellF1)

| | held-out mean ± sd | better than v3 on |
|---|---|---|
| v3 | 0.412 | — |
| v4 with logistic regression | 0.420 ± 0.061 | 7 / 10 |
| **v4 with trees** | **0.433 ± 0.065** | **8 / 10** |

- **The family rule picked the trees:** they beat logistic regression by 0.013, just past the 0.01
  margin fixed beforehand.
- **Every fold picked the same setting:** trees, weight 0.5, dead-ball price 50, no minimum stoppage.
  The plain best was weight 0.5, price 10, minimum 50 frames (0.437, inside the tie band).

| match | v4 | v3 | possessions found / true |
|---|---|---|---|
| J03WMX | 0.371 | 0.365 | 278 / 511 |
| J03WN1 | 0.360 | 0.318 | 294 / 500 |
| J03WOH | 0.418 | 0.375 | 281 / 402 |
| J03WOY | 0.411 | 0.383 | 275 / 456 |
| J03WPY | 0.390 | 0.369 | 288 / 421 |
| J03WQQ | 0.384 | 0.370 | 259 / 401 |
| J03WR9 | 0.449 | 0.405 | 245 / 370 |
| metrica1 | 0.550 | 0.559 | 312 / 354 |
| metrica2 | 0.526 | 0.494 | 316 / 318 |
| metrica3 | 0.470 | 0.482 | 398 / 381 |

**Better on all 7 IDSSE matches** (mean 0.398 against 0.369), and **level on Metrica's** (0.516
against 0.512: one better, two slightly worse).

### Dead ball, where the change was aimed

| | IDSSE v3 | IDSSE v4 | Metrica v3 | Metrica v4 |
|---|---|---|---|---|
| precision | 0.91 | **0.94** | 0.82 | 0.82 |
| recall | 0.64 | **0.90** | 0.59 | **0.90** |
| stoppages found (≥ half called dead) | 369 / 710 | **594 / 710** | 111 / 288 | **219 / 288** |
| missed between same-team possessions (merges) | 65 | **34** | 25 | **13** |
| false stoppages | 9 | 8 | 17 | 23 |

The stoppages v3 missed are mostly found now, with almost no new false ones (Metrica +6).

**Checks:** with merged labels 0.444 against v3's 0.423; at 30 % damage 0.440 against 0.428; at
55 % damage 0.377 against 0.345. The gain holds everywhere.

**What it says:** dead ball improved far more than the score did (recall 0.64 → 0.90, spellF1
+0.021). Once stoppages are found, the next limit is who has the ball between them: steps 3–5 of
the plan (restart shapes, PathCRF for the ball-less frames, a touch rule).

### On our match (no ground truth involved)

| | v3 | v4 |
|---|---|---|
| possessions | 429 | 428 |
| median / mean length | 4.6 s / 8.9 s | 4.1 s / 8.0 s |
| dead ball, share of graded frames | 39.9 % | 46.2 % |
| stoppages | 66, median 23.8 s | 74, median 26.5 s |
| any whistle followed by dead ball within 3 s | 58 / 86 | 71 / 86 |
| whistles v2 counts (whistle + players slow down) followed by dead ball within 3 s | 10 / 11 | **8 / 11** |

- The model says the ball is dead on 45.6 % of graded frames. That is in line with the bench
  (DFL 40–52 % dead).
- **Two counted whistles lost** (frames 23697 and 54740): v3 marked a stoppage after them, v4 marks
  none in the next 30 s.
  - **Why:** the whistle's weight (2.0 per frame for 3 s) was set to beat "into + out of D" in v3.
    Now the model's "live" cost also argues against D, and where the model is confident play goes
    on, it wins.
  - **What it means:** either those stoppages aren't real (the model is right), or the model is
    wrong on a stoppage where players keep moving, which is the 03:06 kind (that one is still found).
    The bench can't tell, since it has no audio. Watching those two moments in the viewer would show
    which. That is allowed for understanding, but not for setting anything frame-specific.
- **Fewer, longer stoppages than the bench suggests:** 74 over 106 minutes, against ~95 per 90 on the
  bench. Short live spells between two stoppages are probably being swallowed by the dead-ball price
  of 50. Recorded, not acted on.

### On the public 5 minutes (a pipeline check, not a ranking)

v4: macroF1 0.819, spellF1 0.442, 21 possessions (truth 17). The pipeline is wired correctly. Not
used to choose anything.

## Open after building

1. **The whistle against the model.** Should a counted whistle keep beating the model's "live"
   evidence, as it beat v1's rules? That would mean re-reasoning its weight, since the bench can't
   test it. Looking at frames 23697 and 54740 in the viewer first would show which side is right.
2. **Stoppage count on our match** (74, long). A check worth making before step 3, since restart
   shapes would add evidence exactly at the ends of stoppages.


## Proposal (2026-10-30): give the whistle back its weight

**Built as option A: [v5](v5-whistle-weight.md).** Counted whistles followed by a stoppage went from 8/11 to 11/11.

**What Jose saw.** Watching frames 23697 and 54740 in the viewer, both whistles were real
stoppages, so v4 is wrong there.

### Why v4 loses them (measured on v4's own output; no labels)

Around the 11 whistles v2 counts (whistle + players slow down by ≥ 0.5 m/s):

| whistle (frame) | players' speed before → after | P(dead), 3 s after | P(dead), 3–30 s after | v4 dead frames in the next 30 s |
|---|---|---|---|---|
| 15835 | 3.01 → 1.66 | 0.15 | 0.69 | 581 |
| 17113 | 3.84 → 1.60 | 0.22 | 0.81 | 726 |
| **23697** | 3.22 → **2.12** | **0.01** | 0.42 | **0** |
| 31173 | 2.64 → 1.32 | 0.63 | 0.22 | 200 |
| **54740** | 4.09 → **2.37** | **0.01** | 0.02 | **0** |
| 74237 | 2.50 → 1.62 | 0.77 | 0.95 | 750 |
| 88695 | 2.34 → 1.25 | 0.64 | 0.98 | 750 |
| 118563 | 2.79 → 1.17 | 0.73 | 0.92 | 750 |
| 148019 | 3.29 → 2.27 | 0.09 | 0.24 | 137 |
| 172217 | 2.67 → 1.19 | 0.55 | 0.94 | 726 |
| 183807 | 1.78 → 0.88 | 0.96 | 0.56 | 499 |

- **The two lost whistles are the "players keep jogging" kind.** Players still averaged over 2 m/s
  afterwards, and the model sits at its floor, P = 0.01: fully confident play goes on.
- **The arithmetic.** At P = 0.01, a frame called dead costs 0.5 × −log(0.01) = **2.30**. The
  whistle adds only **2.0** against live, so for every frame of its 3 s window, dead is the dearer
  answer even before the two switch prices (2 × 50 = 100). In v3 nothing argued that strongly for
  live, so the whistle won.
- **So the whistle's weight lost its meaning.** In v2 it was set to beat the switch prices with room
  to spare (150 against 50). v4 added a clue that can cancel it.

### The rule this stays within

Jose's viewing supplies the **idea** ("a counted whistle should not be overruled by the model"), as
watching 03:06 supplied the whistle itself in v2. Turning observations into general rules is allowed
(#14).
- **What it may not supply** is a constant tied to these frames, or a check that "it fixes 23697
  and 54740" as the reason to accept it. That would make Jose's two observations into hand labels
  used for validation (#10).
- **So:** the new weight comes from reasoning about costs, it applies the same way to all 11
  counted whistles, and it is set before looking at what it does.
- **What we report afterwards** are label-free counts: dead ball after counted whistles, total
  stoppages, possessions.

### Options

**A (recommended). The whistle keeps v2's weight on top of whatever the model can say against it.**
- **New weight** = v2's 2.0 + the model's strongest live evidence, W_DEAD_BALL × −log(0.01)
  = 2.0 + 2.30 = **4.30** per frame, for the same 3 s window.
- **Written as that formula in the code,** so it keeps its meaning if the model's weight or clip
  ever change.
- **What it guarantees:** within its 3 s, a counted whistle always beats the model and both switch
  prices, as it did in v2. Even at the model's floor, (4.30 − 2.30) × 75 = 150 > 100.
- **What it doesn't:** after those 3 s the model decides when play restarts. Where it keeps saying
  "live" (54740: P = 0.02 for 30 s), the stoppage will last only about 3 s. That still splits the
  possessions at the right moment, but the restart lands too early.
- **What it touches:** only the 11 counted whistles. Nothing else in the match changes.

**B. Hold the stoppage until play visibly restarts.** After a counted whistle, the model's "live"
evidence is ignored until the players' mean speed is back to where it was before the whistle (that
event's measured "before" speed), with a 60 s cap: 90 % of bench stoppages last under 49 s.
- **Better:** it gets the length of the stoppage right, not just its start.
- **Worse:** it's a new rule with two more constants, and finding when a stoppage ends is exactly
  what step 3 (restart shapes) is for. Building it now would duplicate that work.

**C. Leave v4 as it is.** 2 stoppages out of ~100 in the match. The bench says v4 is better overall,
and this may cost a possession or two.

### Recommendation

- **A now,** as **v5** (one change on top of v4).
- **The end of stoppages** (B's job) goes to step 3, where restart shapes give the moment play
  resumes and who restarts.
- **No bench number** is possible: the bench has no audio. As with v2, the change rests on its
  reasoning, and we report the label-free counts on our match.
