# Model v6: v5 + who restarts, learned from the players' layout at stoppages

| | |
|---|---|
| **status** | **built and revised 2026-10-30.** The first version gained right after restarts but lost almost as much elsewhere (see "Diagnosis"). The **revision** (only the unlikely team pays, by the odds; the clue waits for 2 s of dead-looking play) keeps the gain and removes most of the loss: held-out spellF1 **0.443** against v4/v5's 0.433, better on 8/10 |
| **run** | `python -m possession.v6` → `evaluation/results/v6.csv` (~1.5 min). `python -m bench.train_restart` re-trains and re-searches (~15 min); the model is committed as `possession/models/restart_logistic.json` |
| **output** | SHA-256 `bbe22b21cbfe6d45c4e89d34a3a23b3b2d2a35ba0d27f7e809ce94bac81620a3` (revised); byte-identical on re-run |
| **change from v5** | one new clue: during a stoppage and just after it, the probability that each team takes the restart, from a model trained on the 10 bench matches. Its weight and how long it lasts after a stoppage are chosen on the bench. Everything else stays |
| **new code** | `possession/restart.py` (team-specific layout features, the model), `bench/train_restart.py` (training, checks, search), `possession/v6.py` |
| **new files committed** | the trained model and its settings |
| **new dependencies** | none (scikit-learn is already pinned for v4) |
| **effort** | ~1–1.5 days |

## Why

**Measured on the bench first** (v4, which equals v5 there since the bench has no audio; out of
sample, 41 % damage, other matches only):

| | IDSSE (7) | Metrica (3) |
|---|---|---|
| restarts (stoppages followed by play) | 696 | 286 |
| … whose stoppage v4 finds | 584 | 217 |
| stoppage end, predicted minus true | median 0.0 s; 60 % within 1 s, 81 % within 3 s | median +0.6 s; 47 % within 1 s, 76 % within 3 s |
| … more than 3 s late / early | 15 % / 4 % | 22 % / 2 % |
| **first possession after a found stoppage has the right team** | **62 %** | **72 %** |

- **Timing is mostly fine.** v4's dead-ball model already ends most stoppages within a few seconds.
- **The team is not.** With two teams, guessing gives 50 %. About a third of the time, play restarts
  with the wrong team.
- **Each wrong restart costs on spellF1:** the first possession after the stoppage is the wrong team,
  so it can't match. If the decoder switches to the right team a second later, it has also
  invented a short fake possession.

**Why v4 gets it wrong:** after a stoppage, the decoder picks H or A from "who is on the ball", and
at restarts the ball is usually missing. On ours, it is missing on 83 % of frames while players are
stopped. Nothing else says who takes the restart: Smart Tagging leaves every throw-in as team "N/A",
and the bench has no tags at all.

**But the layout says it.** The Laws of the Game force a shape at every restart, and the shape tells
whose restart it is, without the ball:

| restart | the restarting team | the other team |
|---|---|---|
| throw-in | one player on or beyond the touchline | — |
| corner | one player at a corner; several in the opponents' area | its goalkeeper in its own 6-yard box, defenders in the area |
| goal kick | goalkeeper in its own 6-yard box | **all** outside the penalty area (Law 16) |
| kick-off | players in the centre circle, all in their own half | all in their own half, **none** in the centre circle |
| free kick | the taker at the spot; attackers forward | a wall ~9.15 m away |

## How it works

### Piece 1: features for each team (`possession/restart.py`)

Computed per frame from the damaged tracking, players only, for one team ("this team") against the
other. No ball: it's missing at most restarts, and Metrica's samples blank it in dead time.

| group | for this team and for the other team, separately |
|---|---|
| touchline and goal line | players beyond a touchline; within 1 m inside one; beyond each goal line |
| corners | players within 3 m of the corners at each end |
| boxes | players in the 6-yard box and in the penalty area, at each end |
| centre | players in the centre circle; share of the team in each half |
| movement and shape | mean speed; spread; the team's centre along the pitch |
| time context | the touchline, corner, box and centre counts averaged over the past and next 2 s |

About 50 features. They describe "this team" and "the other team", not home and away, and the ends
of the pitch, not attack directions.

**Mirroring.** Every training frame is used four times:
- as it is;
- with the teams swapped (and the answer flipped);
- with the pitch flipped end to end;
- with both.

This is the mirroring that the SoccerNet 2025 team-action winners credited
([../research/practitioner-advice.md](../research/practitioner-advice.md)). The model can't learn
"home usually restarts" or "this end is special", only the shapes.

### Piece 2: the model (`bench/train_restart.py`)

- **Target:** on every dead frame of the 10 bench matches, at 41 % damage, which team restarts
  after that stoppage: the team of the first live frame after it.
  - On IDSSE that's the team DFL's events name on 98 % of restarts ([../bench.md](../bench.md),
    check 3).
  - On Metrica it comes from the set-piece events.
- **Training rows:** every 5th dead frame, mirrored four ways.
- **Prediction:** P(this team restarts) for each team on every frame, made symmetric by averaging
  the two orientations.
- **Two candidates, as in v4:** logistic regression and gradient-boosted trees. Logistic regression
  unless the trees win end to end by more than 0.01.
- **Out of sample:** each bench match is predicted by a model trained on the other 9.

### Piece 3: into the costs (`possession/v6.py`)

With R = P(home restarts), smoothed over the stoppage:
- **During a stoppage:** w × −log(R) on H and w × −log(1 − R) on A, the cost of saying each team.
  (Corrected while building: the proposed page had H and A swapped.) This doesn't touch D,
  so it can't create or end stoppages; it only says which team is the likelier one to come out of it.
- **For a short while after:** the same costs carry on for `carry` seconds after the frames the
  dead-ball model calls dead (P(dead) > 0.5). That's where the decoder actually chooses the first
  team.
- **The smoothing:** R is the mean over the last 2 s of dead-looking frames, so one odd frame of
  layout doesn't decide the restart.

### Piece 4: settings chosen on the bench

| setting | candidates | why |
|---|---|---|
| weight `w` | 0.25, 0.5, 1, 2 | from a nudge to as strong as ball-out |
| `carry` after a stoppage | 1, 3, 5 s | half of real possessions last under 3.7 s, so this should stay short |
| family | logistic, trees | as v4 |

Everything else stays v5's (and so v4's and v3's). The procedure is v3's and v4's:
- leave one match out over all 10;
- best mean spellF1, and the fewest possessions within 0.01 of the best;
- the final model trained on all 10.

**Reported but not used to choose:**
- restart-team accuracy (the 62 % / 72 % above);
- stoppage-end timing;
- the score per set;
- 30 % and 55 % damage;
- merged labels.

## What to expect

- **On the bench:** restart-team accuracy well above 62 % / 72 %. The spellF1 gain will be smaller,
  since restarts are one of several places possessions go wrong.
- **On our match** (no labels): how often the first team after a stoppage changes from v5. For
  stoppages that have a Smart Tagging set-piece tag with a team, how often the model agrees with the
  tag's team. That second one is a label-free cross-check, because the tags are automatic (FAQ: "not
  ground truth").

## Expected failures and risks

1. **Quick restarts** (a throw-in taken at once, a short free kick): the layout barely forms before
   play goes on. The model sees little.
2. **Free kicks** have no line or box to anchor them; the wall is the clue. They'll be the weakest
   type.
3. **Our tracker's player losses near the touchlines** (the camera's edge) could hide a throw-in
   taker. The bench's damage doesn't imitate where players are lost, only how often.
4. **The carry can overreach:** if a possession changes within `carry` seconds of the restart, the
   clue argues for the wrong team. That's why `carry` is chosen short on the bench.
5. **Not used:** Smart Tagging's set-piece team, which only our match has. It stays a cross-check,
   not a clue, unless we decide to weigh it by reasoning later.

## Decisions for Jose before building

| | choice | recommended | why |
|---|---|---|---|
| 1 | Aim step 3 at **who restarts**, and leave the stoppage-end timing for later | **yes** | The bench shows timing is mostly right (≥ 76 % within 3 s) and the team is the gap (62–72 %) |
| 2 | A learned model on layout features with mirroring, rather than five hand-built detectors per restart type | **yes** | One model covers all five shapes and their mixtures; hand detectors need ~15 thresholds and fail on the in-between cases. The shapes from the Laws are in the features, so it stays explainable |
| 3 | Smart Tagging's set-piece team as a cross-check only, not a clue | **yes** | The bench can't weigh it; keeping it out keeps v6 measurable |
| 4 | Name it v6 | **yes** | One change per version |

## Results (2026-10-30)

### The restart model on its own (out of sample, stoppage frames of all 10 bench matches)

| | the restarting team, right on |
|---|---|
| logistic regression | 74.5 % of stoppage frames |
| gradient-boosted trees | 86.9 % |

### End to end, leave one match out (spellF1)

| | held-out mean ± sd | better than v4/v5 on |
|---|---|---|
| v4 / v5 (equal on the bench) | 0.433 | — |
| v6, logistic | 0.435 ± 0.062 | 6 / 10 |
| v6, trees | 0.439 ± 0.054 | 7 / 10 |

- **The family rule kept logistic regression:** the trees were ahead by 0.004, inside the 0.01
  margin.
- **The chosen setting:** weight 1, carry 3 s. The folds didn't all agree: six picked weight 1,
  four picked weight 2, and two picked a 1 s carry. The plain best was weight 0.25, carry 3 s
  (0.443, inside the tie band).
- **Per set:** IDSSE 0.409 against 0.398; Metrica 0.498 against 0.516 (worse on two of three).
- **Checks:** 30 % damage 0.445 against 0.440; 55 % damage 0.391 against 0.377; merged labels
  0.446 against 0.444.

### Restarts, where the change was aimed (chosen setting)

| | IDSSE v4/v5 | IDSSE v6 | Metrica v4/v5 | Metrica v6 |
|---|---|---|---|---|
| stoppages found | 584 / 696 | 611 / 696 | 217 / 286 | 231 / 286 |
| **first possession after a found stoppage has the right team** | 62 % | **87 %** | 72 % | **84 %** |
| stoppage end within 1 s / 3 s | 60 % / 81 % | 58 % / 78 % | 47 % / 76 % | 45 % / 71 % |

### On our match (no ground truth involved)

| | v5 | v6 |
|---|---|---|
| possessions | 426 | 429 |
| dead ball, share of graded frames | 46.3 % | 48.3 % |
| stoppages | 77 | 83 |
| frames that differ | — | 6,502 (4.1 % of graded) |
| counted whistles followed by dead ball | 11 / 11 | 11 / 11 |

**Cross-check with Smart Tagging's set-piece tags** that name a team (automatic tags, not ground
truth). On the 25 such tags with a dead-looking stretch in the 15 s after them:

| | agrees with the tag's team |
|---|---|
| the restart model itself | **23 / 25** |
| v5's first team after the stoppage | 10 / 25 |
| v6's first team after the stoppage | 14 / 25 |

The "first team after" measure is rough. It takes the first live label after the last dead frame in
the window, which can be a later possession if the ball is lost quickly.

**Public 5 minutes (pipeline check, not a ranking):** spellF1 0.444 (v5 0.442), macroF1 0.809
(0.818), 19 possessions (truth 17).

### A correction to piece 3

The page said the clue "doesn't touch D, so it can't create or end stoppages". **That isn't quite
true.** It adds cost to one team only. So where the decoder would otherwise come out of a stoppage
with the unlikely team, staying dead becomes relatively cheaper. That is why dead ball rose (46.3 %
→ 48.3 % on our match, more stoppages found on the bench) and stoppage ends got slightly worse
(within 3 s: 81 % → 78 %, 76 % → 71 %).

### What this says

- **v6 does what it was built for.** The team after a stoppage is right far more often on the bench
  (62 % → 87 %, 72 % → 84 %), and on our match the restart model agrees with Smart Tagging on 23 of
  25 tagged restarts.
- **It barely moves the ranking metric:** +0.002 to +0.006 held out, inside the noise, and worse on
  Metrica's games.
- **Two gaps explain part of it:**
  - the decoded first team agrees with the tags far less often (14/25) than the model does (23/25),
    so the decoder is overriding the clue at many restarts;
  - fixing the first team may not fix the possession if its boundaries are wrong, or if the ball is
    lost within seconds anyway.
- **Not yet known:** which of those two dominates. That decides whether to strengthen the clue,
  change how it enters the decoder, or leave restarts alone.


## Diagnosis (2026-10-30): why a much better restart team barely moves spellF1

Bench only, out of sample, 41 % damage, all 10 matches. The exploration scripts are not committed;
the numbers are below. Each real possession gets a score from 0 to 5: how many of spellF1's five
thresholds (IoU 0.3–0.7) it is matched at.

### 1. The decoder does not override the clue

Of the 842 restarts v6 finds, the restart model points to the right team on 725. On those, **v6's
first team is right 99 % of the time**: only 9 are overridden. So the clue gets through.

(On our match, Smart Tagging's cross-check gave 23/25 for the model against 14/25 for v6. That
measure looks for "the first live label after the last dead frame within 15 s of the tag", which can
land on a later possession; the bench measure is the reliable one.)

### 2. Gains right after restarts, losses elsewhere

| real possessions | score summed, v4/v5 | v6 | change |
|---|---|---|---|
| the first possession after a restart | 1,871 | 2,265 | **+394** |
| all other possessions | 5,722 | 5,479 | **−243** |

v6 reports 103 more possessions (2,946 → 3,049); precision 0.517 → 0.508, recall 0.370 → 0.377 at
IoU 0.5. **The gain is real, and most of it is given back elsewhere.**

### 3. Where the losses come from

**a. The clue makes staying dead cheaper.** It adds cost to both teams: −log R to one, −log(1 − R) to
the other, and both are positive. So relative to D, both teams get dearer near every stoppage.
- Restarts come out later: within 3 s 81 % → 78 % (IDSSE), 76 % → 71 % (Metrica).
- On our match, dead ball rose from 46.3 % to 48.3 %.
- Even where v4 already had the right team, v6 does slightly worse on both the first and the next
  possession (395 such restarts: −95 score points in total).

**b. The clue fires in live play.** The dead-ball model says P(dead) > 0.5 on 7.5 % of truly live
frames. So the restart clue, which switches on wherever play "looks dead", is active on 17.4 % of
live frames, and on 8.8 % of live frames more than 5 s from any stoppage. That's a team push based
on a layout that means nothing in open play.
- Half of all frames where v6 differs from v4 are in live play far from stoppages; the clue is active
  on 27 % of those.
- The rest are knock-on changes of the decoder.

**c. Knock-on to the next possession.** When the restart team loses the ball within 3 s, v6 now
holds the right restart team, but holds it too long:
- the first possession's score goes 0 → 1.09;
- the next one (the other team's) drops 1.65 → 1.12.

v4 had "accidentally" matched that next possession by starting with the wrong team. Overall the
fixed restarts still gain +319 net. Of the fixed restarts still unmatched, the real first
possession is short (median 2.5 s) while v6's runs 8.8 s, starting 2.6 s late.

### 4. What it means

The restart model is good: right on 87 % / 84 % after a stoppage. **How the clue enters the costs**
loses most of the gain: (a) and (b) are side effects of the formula, (c) is the next possession's
evidence being too weak when the ball is missing.

## Proposal (2026-10-30): a revised v6

**Built; results in the next section.**

Two changes to how the clue enters. Both are reasoned; the weight and the carry are then re-chosen on
the bench with the same procedure.

**A. Cost only the unlikely team, by the odds.** Instead of −log R on H and −log(1 − R) on A, add
**w × log(R / (1 − R)) to the unlikely team only** (the more likely team gets nothing).
- **Why:** this keeps exactly the same preference between the two teams, but the likely team is no
  longer dearer than before. So staying dead isn't made relatively cheaper, which fixes (a).
- **Cost:** no new constant.

**B. Only after a real-looking stoppage.** Switch the clue on only after **at least 2 s** of
dead-looking play (P(dead) > 0.5), not after a single frame.
- **Why:** on the bench, only 1 % of real stoppages are shorter than 2 s, while blips in open play
  are mostly shorter. That targets (b).
- **Cost:** one new constant, 2 s, from the bench's stoppage lengths ([../bench.md](../bench.md)),
  not tuned.

**Not addressed here: (c).** Seeing the next change of possession when the ball is missing is
exactly step 4 (PathCRF). Shortening the carry would trade restart gains for it; the bench already
chose 3 s over 1 s.

**How we'd judge it:**
- the same leave-one-match-out search;
- the same two-sided accounting (first possessions after restarts, and all others);
- the restart timing and dead share.

**If the "other" losses don't shrink, v6 isn't worth keeping and v5 stays the main line.**

## Results of the revision (2026-10-30)

Built as proposed: revision A in `restart.costs`, revision B in `restart.restart_belief`
(`MIN_STOPPAGE` = 2 s). The search re-ran with the same procedure. The restart model itself is
unchanged: same data, same features, a byte-identical model file. The two-sided accounting and the
follow-through check are now part of `bench/train_restart.py`, so they're reproducible.

### Two-sided accounting (bench, out of sample, chosen setting)

| | v4 / v5 | first v6 | **revised v6** |
|---|---|---|---|
| first possession after a restart (score summed, 0–5 each) | 1,871 | 2,265 (+394) | **2,300 (+429)** |
| all other possessions | 5,722 | 5,479 (−243) | **5,660 (−62)** |
| net | — | +151 | **+367** |
| possessions matched at IoU 0.5 (of 4,114 real) | 1,524 | 1,550 | **1,595** |
| possessions reported | 2,946 | 3,049 | 3,107 |

### Restarts

| | IDSSE v4/v5 | IDSSE revised v6 | Metrica v4/v5 | Metrica revised v6 |
|---|---|---|---|---|
| stoppages found | 584 / 696 | **584** (first v6: 611) | 217 / 286 | **217** (first v6: 231) |
| right team after a found stoppage | 62 % | **87 %** | 72 % | **85 %** |
| stoppage end within 1 s / 3 s | 60 % / 81 % | 60 % / 80 % | 47 % / 76 % | 47 % / 76 % |

- **Side effect (a) is gone:** the same stoppages are found, and they end at the same time as v4/v5.
- **The decoder follows the clue:** 690 restarts where the clue was right and v6 started with the
  right team, against 5 overridden.

### End to end (spellF1, leave one match out)

| | held-out mean ± sd | better than v4/v5 on |
|---|---|---|
| v4 / v5 | 0.433 | — |
| revised v6, logistic | **0.443 ± 0.060** | **8 / 10** |
| revised v6, trees | 0.438 ± 0.055 | 6 / 10 |

- **Logistic regression won outright this time.**
- **The chosen setting is the same as before:** weight 1, carry 3 s. Six of ten folds picked it; the others picked weight 2 (three) or weight 0.5 with a 5 s carry (one).
- **Per set:** IDSSE 0.413 against 0.398; Metrica 0.517 against 0.516, no longer worse.
- **Checks:** 30 % damage 0.449 against 0.440; 55 % 0.394 against 0.377; merged labels 0.456
  against 0.444.

### On our match (no ground truth involved)

| | v5 | first v6 | **revised v6** |
|---|---|---|---|
| possessions | 426 | 429 | 439 |
| dead ball | 46.3 % | 48.3 % | **46.5 %** |
| stoppages | 77 | 83 | 78 |
| frames that differ from v5 | — | 6,502 | 2,720 (1.7 %) |
| Smart Tagging set-piece team agrees with the first team after the stoppage (25 tags; model alone: 23) | 10 | 14 | **16** |

- **Stoppage ends:** 70 fall on the same frame in v5 and the revised v6, and the team after them
  differs on 15.
- **Public 5 minutes** (pipeline check): identical to v5 (spellF1 0.442, macroF1 0.818). The first
  5 minutes have no restart where the two differ.

### What this says

- **The diagnosis was right:** the loss came from how the clue entered the costs, not from the
  model or the decoder.
- **The revised v6 is a small but consistent gain:** +0.010 held out, better on 8 of 10 matches,
  holding at every damage level and with merged labels. That's below the full match's noise band,
  so on its own it can't prove the real score rises. But it is paired and consistent on the bench,
  and it has no side effect.
- **Still open: (c), the next possession after a quickly lost restart.** That needs better evidence
  of who has the ball while it's missing (step 4).

