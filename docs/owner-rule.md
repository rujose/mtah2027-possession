# What "owning the ball" means in Metrica's corrected tracking (2026-11-12)

| | |
|---|---|
| **question** | The organisers' truth is "derived mechanically from corrected tracking": D where it marks the ball dead, otherwise "the team the tracking names as owning it" (CHALLENGE.md). v13 learned mostly from DFL's owner field. Does Metrica's notion of owning differ, and does that cost v13? |
| **data** | Metrica's 3 public sample games, which are corrected Metrica tracking, with their events. External data only: nothing of our match was read |
| **code** | `bench/owner_rule.py` (~10 min) → `bench/results/owner_rule.json` |
| **answer** | Metrica's corrected ball **rides with its owner**. Its owner convention counts **any deliberate play, one touch included, but not accidental contact**. **v13 already sits close to that convention** on Metrica's games, so the hoped-for hidden gain is small; what's left is how the bench weighs the sources, and telling deliberate touches from deflections |

## 1. The corrected ball rides with its owner

While a player keeps the ball between receiving a pass and making his next one (2 s or more):

| | game 1 (174 holds) | game 2 (184 holds) |
|---|---|---|
| ball to holder, median over a hold | **0.11 m** (90 % of holds under 0.21 m) | **0.12 m** (under 0.20 m) |
| how far the path bends from a straight line: ball / holder (median) | 1.04 m / 0.87 m | 0.86 m / 0.83 m |
| holds where the holder bends 1 m more than the ball | 0 % | 0 % |

The ball isn't drawn in straight lines from event to event: it bends with the holder, about 0.1 m from him. Between
players it moves freely and smoothly. So the corrected tracking does encode control.

On live frames with the ball seen, it's within 0.3 m of a player 53 % (game 1), 51 % (game 2) and 32 % (game 3) of
the time. Game 3 (the EPTS export, like ours) attaches less tightly.

## 2. Which touches count: deliberate play, not contact

Rules applied to the corrected tracking, scored against each game's event labels (`bench/metrica.py`'s recipe;
the events' dead ball kept, so only the owner convention differs):

| rule | possessions (games 1 / 2 / 3) | spellF1 against the event labels |
|---|---|---|
| the event labels | 354 / 318 / 381 | |
| **any touch** (a player within 0.2 m for one frame) owns it; a loose ball stays | 381 / 382 / 422 | **0.81 / 0.75 / 0.77** (frames agree 95 / 95 / 93 %) |
| a touch must last 3+ frames | 242 / 225 / 233 | 0.64 / 0.62 / 0.55 |
| a touch must last 25+ frames | 176 / 160 / 145 | 0.49 / 0.54 / 0.39 |
| a touch counts if that team holds it 5+ frames or touches it next too (r = 0.5 m) | 290 / 261 / 310 | 0.74 / 0.74 / 0.72 |

- **One touch is enough** when it's a deliberate play: a first-time pass, a shot, a failed pass that's then
  intercepted all start a possession in Metrica's labels. Demanding control for a few frames, or a confirming
  second touch, loses most of them.
- **What the touch rule over-counts** (381–422 against 318–381) is accidental contact: deflections, blocks,
  challenges. That's the organisers' written rule too: "a touch without control, a deflection or an aerial duel
  won without control doesn't count" ([convention-audit.md](convention-audit.md)).

## 3. Where v13 stands (branch A, held out on the bench)

| match | v13 spellF1 | possessions true / v13 | under 2 s true / v13 | v13 against the touch rule |
|---|---|---|---|---|
| metrica1 | 0.815 | 354 / 328 | 68 / 60 | 0.712 |
| metrica2 | 0.795 | 318 / 303 | 69 / 58 | 0.651 |
| metrica3 | 0.816 | 381 / 354 | 87 / 64 | 0.692 |
| DFL's 7 | 0.627–0.710 | 370–511 / 313–395 | 94–191 / 47–89 | |

- **On Metrica's convention v13 is close:** 5–8 % fewer possessions than the truth, and most of the short ones.
  On DFL's it predicts about 20 % fewer, and misses most of DFL's many short possessions.
- **So v13 did not learn DFL's busier convention wholesale.** Its decoder, its minimum length and its stages
  produce something much closer to Metrica's. The 2026-11-12 hypothesis that this was a hidden loss is mostly
  refuted.
- **The exposure that remains:** if the organisers' owner field counted every contact (the touch rule), v13 would
  score about 0.1 lower (0.65–0.71). Their written rule says it doesn't, which is the reading v13 matches.

## What follows

1. **The bench's weighting.** Every lab verdict was decided mostly by DFL's 7 matches, whose convention is the
   one further from the answer key. Candidates that helped Metrica's 3 and hurt DFL's 7 were discarded (rounds
   5–6: no-snap, switch, motion). A bench whose labels all follow Metrica's convention would judge them on the
   right definition. Routes: DFL's events through our Metrica recipe (done once in the convention audit, but event
   times are ±1.8 s off), or ELASTIC's frame-accurate events for 3 of the DFL matches (CC BY 4.0,
   [research](research/README.md)).
2. **Deliberate touch or deflection.** This is the one thing that separates the touch rule from the answer key's
   convention, and Metrica's games label it (a contact followed by the team's event, or not). A model that tells
   them apart from the tracking would sharpen the stages that add possessions at touches.
3. **The organisers' email** (question 6) asks about exactly this, so an answer would settle it.
