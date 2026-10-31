# Convention check: do Metrica's possessions run longer than DFL's?

| | |
|---|---|
| **status** | **built 2026-10-30**, as proposed, with the three recommended choices |
| **run** | `python -m bench.metrica download` (once, ~190 MB) → `python -m bench.metrica parse` (~10 s) → `python -m bench.convention` (pieces 1–3, ~2 s) → `python -m bench.run --set metrica` and `python -m bench.tune_decoder --set metrica` / `--set all` (piece 4, ~5 min each) |
| **answer** | **Metrica's events change hands about 20 % less often than DFL's, but v3's settings are still the best choice on Metrica's labels: the search picks exactly v3's setting on Metrica's games alone and on all 10 matches** |
| **question** | v3's decoder settings were chosen on DFL's possession labels. Does Metrica, whose convention the organisers' truth follows, count possession the same way, or does it change hands less often? |
| **changes our submission?** | Not directly. It decides whether v3's settings stand or get re-chosen with Metrica-style labels in the bench |
| **new dependencies** | none |
| **effort** | ~1 day for pieces 1–3; ~0.5–1 day more for piece 4 |

## Why

- v3 chose a low team switch price (0.5) and a 0.8 s minimum on the [bench](bench.md), whose
  labels are DFL's: 437 possessions per match, median 3.7 s.
- If the organisers' truth changes hands less often, v3 over-splits our match.
- Two hints point that way, and neither can settle it
  ([models/v3-decoder.md](models/v3-decoder.md)):
  - on the 5 public minutes v3 reports 20 possessions against a true 17, but that window must not
    choose between versions;
  - on our match v3's count rose ×2.4 over v2, against ×1.9 on the bench.
- **The one allowed source of Metrica's own judgement is Metrica's public sample data:** 3 matches
  with tracking and events tagged by Metrica. Human-made data about **other** matches is allowed
  (RULES.md, #8, #10).

**What this can and can't show.**
- **It is not the truth's exact recipe.** The organisers' truth is built mechanically from Metrica's
  **corrected tracking** (the team that tracking names as owning the ball). The sample games give
  Metrica's **events**, which are human tags following Metrica's written definitions. Turning
  events into frames needs a recipe of our own.
- **So the comparison is made like for like.** The same recipe is also applied to **DFL's events**
  on the 7 bench matches. Then:
  - DFL events (recipe) vs DFL's per-frame labels shows what the recipe itself does;
  - Metrica events (recipe) vs DFL events (recipe) shows the difference between the two conventions
    and the matches.
- **Three anonymous matches are few.** Their league and level are unknown, so style differences mix
  with convention differences. The result is evidence, not proof.

## The data: Metrica's sample games

- **Source:** github.com/metrica-sports/sample-data. The README says it exemplifies Metrica's
  "Elite data package" and that tracking and events are synchronised. Pitch 105 × 68 m, coordinates
  0–1 with (0, 0) at the top left, 25 fps.
- **Events** (checked 2026-10-30):

  | game | events | format |
  |---|---|---|
  | 1 | 1,745 | CSV: team, type, subtype, period, start/end frame, from/to player, start/end x/y |
  | 2 | 1,935 | same |
  | 3 | 3,620, of which 1,395 are carries (dribbles) | JSON, same fields; subtypes can be lists, e.g. `GROUND, FAULT, WON` |

  The types are PASS, CARRY (game 3 only), RECOVERY, BALL LOST, CHALLENGE, BALL OUT, SET PIECE,
  SHOT, FAULT RECEIVED and CARD. Subtypes include INTERCEPTION, THEFT, HEAD, GOAL KICK, CLEARANCE,
  OFFSIDE, and the challenge outcomes WON/LOST/FAULT/ADVANTAGE.
- **Tracking** (needed only for piece 1's checks and piece 4):
  - games 1–2: two CSVs per match, home and away, ~33 MB each;
  - game 3: EPTS like ours (53 MB) plus an XML.
  - Players have identities and there's no speed column; the ball is blanked during dead time on
    most frames (the datasets research measured 94 % of event-dead frames on a sample).
- **Licence:** none stated. The README asks users to be responsible and to acknowledge the source.
  It's the organisers' company's own public repository, so they can access it. It is listed as grey
  zone G8 in [limitations.md](limitations.md) (decision 1 below).
- **Where:** `data/external/metrica/` (gitignored). Downloaded from GitHub with the files' git
  checksums verified.

## Piece 1: H/A/D from Metrica's events (`bench/metrica.py`)

A recipe that follows Metrica's written definitions and the organisers' rules (CHALLENGE.md):
possession changes only when the other team gains control; a loose ball stays with the last team.

1. **Owner.** From each **controlling event**, the event's team owns the ball from its start frame
   until the next controlling event.
   - Controlling events: PASS (any subtype, goal kicks included), CARRY, SHOT, RECOVERY, SET PIECE.
   - A pass in flight stays with the passer's team.
   - BALL LOST doesn't flip the owner: the flip happens at the other team's next controlling event.
   - **CHALLENGE doesn't flip the owner on its own** (decision 2). A won challenge that leads to
     control is followed by a RECOVERY, which flips it. This is also how Metrica's definitions mark
     "significant interference".
2. **Dead ball starts** at:
   - the end frame of a BALL OUT;
   - the start of a foul (a challenge with FAULT in its subtypes, or FAULT RECEIVED);
   - an OFFSIDE;
   - the end of a SHOT whose result is a GOAL or OUT.

   A challenge marked ADVANTAGE doesn't stop play.
3. **Dead ball ends** at the next restart: a SET PIECE start (kick-off, throw-in, free kick, corner,
   penalty), or a PASS with subtype GOAL KICK. Everything before each period's kick-off is D.
4. **Periods:** each half is scored separately, with an X frame between them, as on the bench.

**Checks, printed:**
- the share of D frames, and stoppages per 90 minutes;
- how D agrees with "ball missing" in their tracking: high agreement is expected, since the samples
  blank the ball in dead time;
- the **reading check** from the bench: where a player is within 1 m of a slow ball, his team should
  be the labelled team. That tests that the events line up with the tracking.

## Piece 2: the same recipe on DFL's events (`bench/convention.py`)

Applied to the 7 bench matches, mapped one to one:

| recipe role | Metrica event | DFL event |
|---|---|---|
| controlling event | PASS, CARRY, SHOT, RECOVERY, SET PIECE | Play, ShotAtGoal, OtherBallAction, BallClaiming, any restart |
| a challenge that flips possession | (none on its own; see RECOVERY) | TacklingGame with `PossessionChange="true"`, to the winner's team |
| restart | SET PIECE, PASS · GOAL KICK | KickOff, ThrowIn, FreeKick, CornerKick, GoalKick, Penalty |

- **Dead ball on DFL: we use DFL's own per-frame status, not events.** DFL's events have no "ball
  out" event, and the question here is who owns the ball, not when play stops.
- **Placing DFL events on frames:** each event goes to the frame with the nearest tracking time, as
  in the bench's restart check. The paper reports event times off by −0.37 ± 1.82 s on average. That
  blurs the edges of possessions, but barely changes how many there are.

## Piece 3: the comparison (`bench/convention.py`)

Measured per match, and compared **only during live play**, so that differences in dead-ball
detection don't leak into the possession comparison:

| measure | what it captures |
|---|---|
| **team changes per 10 minutes of live play** (H↔A, directly or across a stoppage) | the core question: how often possession changes hands |
| median and mean length of a team's unbroken stretch in live play | the same, as durations |
| possessions per 90 minutes, as spellF1 counts them (stoppages split them) | what the metric sees |
| share of possessions under 1 s and under 2 s | whether brief touches count |
| stoppages per 90 minutes; median stoppage | context only |

Three columns:
- **(a)** DFL's per-frame labels;
- **(b)** DFL's events through the recipe;
- **(c)** Metrica's events through the recipe.

**Decision rule, fixed before looking:**
- **(b) vs (a): what the recipe does.** If (b) is much stickier than (a), part of any Metrica
  difference is the recipe, and (c) is compared with (b), never with (a).
- **(c) vs (b): the convention.**
  - **No evidence of a gap:** Metrica's 3 matches fall **inside the range** of DFL's 7 on team
    changes per live 10 minutes. v3's settings stand.
  - **A gap:** all 3 fall **outside** that range on the same side. Metrica changes hands less (or
    more) often, so go to piece 4 and re-choose v3's settings with Metrica's games on the bench too.
  - **In between** (some inside, some outside): report it, and do piece 4 if decision 3 says so.

## Piece 4: Metrica's games as a second bench (decision 3)

The comparison above is indirect. The direct test: **do v3's settings still beat v1's when the
labels are Metrica's?**

1. **Load their tracking** into the same arrays as the bench:
   - players in metres;
   - speed computed from positions, since there's no speed column;
   - teams per slot.
2. **Damage it** with the same `bench/degrade.py`. One known difference: their ball is already blanked
   in dead time, so the damaged ball goes missing in stoppages more than ours. On ours it's 83 % while
   players stand still, which is in fact close to that.
3. **Run v1, v3's setting and the floor**, and re-run the decoder grid with the 3 Metrica games added.
   Report:
   - whether v3's setting still wins on Metrica's games;
   - what the grid would choose on Metrica's games alone;
   - what it would choose on all 10 matches.
4. **If the choice on all 10 differs from v3's,** that is a v4 with its own page. Nothing changes
   silently.

## Rules we hold to here

- **Only other matches' data.** Metrica's sample games are human-tagged data about other matches:
  allowed in any amount, as long as the organisers can access it (it's Metrica's public repository)
  and the technical document says what, who tagged it (Metrica) and how much (3 matches).
- **No BAR-ATH labels, and not the public 5 minutes**, for anything here.
- **Attribution:** the README asks users to acknowledge the source. The citation goes in
  [README.md](README.md)'s "data from other matches" and the technical document.
- No Metrica data in git; summary tables of results may go in.

## Expected failures and risks

1. **Event recipe ≠ tracking ownership.** The truth comes from corrected tracking, not events. Both
   providers' events could share a bias that tracking ownership doesn't have. Piece 2 measures that
   bias on DFL (where we have both); for Metrica we can't.
2. **Three matches of unknown level.** A gap could be style (fewer, longer possessions in a slower
   league) rather than convention. Piece 4 sidesteps that: it asks what works on those matches, not
   why.
3. **Game 1–2 have no carries.** Ownership during dribbles is inferred from the surrounding passes.
   That's fine for "who", but the start of a dribble after a recovery may be placed late.
4. **DFL event timing** (±1.8 s): it blurs durations in (b), and much less the counts and changes per
   minute.

## Decisions for Jose before building

| | choice | recommended | why |
|---|---|---|---|
| 1 | Use Metrica's sample data before the organisers answer G8 (no licence file) | **yes** | It's their company's own public repo, used to check and choose settings, never as a submission input. Add it to the questions if we post an issue |
| 2 | Let CHALLENGE · WON flip possession in the recipe | **no**, only RECOVERY (Metrica) and `PossessionChange` (DFL) | Each provider's own "possession changed" marker, so the recipe treats both alike |
| 3 | Do piece 4 (Metrica's games as a second bench) whatever pieces 1–3 find | **yes** | It's the direct test of "does v3 over-split under Metrica's labels", and it makes the bench 10 matches with two conventions for every later step |

## Files

```
bench/metrica.py            download (git checksums), events -> H/A/D, tracking -> arrays (piece 4)
bench/convention.py         the recipe on DFL events, the comparison table, the decision rule
data/external/metrica/      raw files and cache (gitignored)
bench/results/convention.csv  one row per match and label source: the measures above (committed)
```

## Results (2026-10-30)

### Piece 1: Metrica's labels are sound

| game | dead ball, share of frames | stoppages per 90 min | dead frames with their ball blanked | live frames with their ball blanked | reading check (frames) | stoppages ended by play, not a tagged restart |
|---|---|---|---|---|---|---|
| 1 | 40.7 % | 91.2 | 92.3 % | 2.6 % | 97.0 % (17,310) | 26 |
| 2 | 35.4 % | 90.9 | 97.8 % | 9.9 % | 99.4 % (14,169) | 27 |
| 3 | 29.3 % | 89.2 | 99.4 % | 4.3 % | 99.1 % (15,700) | 18 |

- Our event recipe's dead ball lines up with the stretches where Metrica blanked the ball
  (92–99 %).
- Where a player is within 1 m of a slow ball, his team is the labelled team on 97–99 % of frames,
  so events and tracking line up.
- Some stoppages (18–27 per game) ended when play visibly went on without a tagged restart, which
  the recipe allows for.

### Piece 2: what the recipe does on DFL

- On live frames, DFL's events through the recipe name the same team as DFL's per-frame labels on
  **70–88 %** of frames (median 84 %). The rest is event timing (±1.8 s) and the recipe.
- **The recipe is stickier than DFL's frames:** 67.1 changes of possession per 10 live minutes
  against 77.7. That's why Metrica is compared with (b), not (a).

### Piece 3: the comparison (mean per source; per-match values in `bench/results/convention.csv`)

| measure | (a) DFL frames, the bench's labels | (b) DFL events, recipe | (c) Metrica events, recipe |
|---|---|---|---|
| **changes of possession per 10 live minutes** | 77.7 | **67.1** (range 51.3–82.9) | **52.8** (56.9, 48.0, 53.6) |
| median unbroken team stretch in live play | 4.1 s | 5.1 s | 6.0 s |
| possessions per 90 min | 412 | 382 | 330 |
| median possession | 3.8 s | 4.4 s | 5.6 s |
| possessions under 1 s | 12.3 % | 10.7 % | 6.9 % |
| stoppages per 90 min | 95.7 | 95.7 | 90.4 |
| median stoppage | 20.1 s | 20.1 s | 15.0 s |

**Decision rule: "in between".** Two Metrica games fall inside DFL's range and one below it.
Like for like, Metrica's events change hands about **21 % less often** than DFL's, and their
possessions are longer (median 5.6 s against 4.4 s). That fits the hint that motivated this page,
but it's not large or consistent enough to call a gap by the rule fixed beforehand. Decision 3 said
to do piece 4 regardless.

### Piece 4: Metrica's games on the bench

**Their damage:** their ball is already blanked on 37.4 % of frames (almost all of the dead time),
so the 30 % level can't go below that and the 0 % level isn't clean. At 41 % the losing chance
needed only a scale of 0.136 on top of the blanking.

**The models on Metrica's labels** (41 % damage, raw labels):

| game | floor | v1 | **v3** | v3 possessions / true |
|---|---|---|---|---|
| 1 | 0.395 | 0.382 | **0.559** | 309 / 354 |
| 2 | 0.358 | 0.377 | **0.494** | 316 / 318 |
| 3 | 0.338 | 0.392 | **0.482** | 406 / 381 |
| mean | 0.364 | 0.384 | **0.512** | 344 / 351 |

**v3 doesn't over-split on Metrica's labels:** it finds 344 possessions where the events say 351.
It beats v1 and the floor on every game, and at every damage level (30 %: 0.529; 55 %: 0.427).

**The decoder search** (same grid and rules as v3's):

| chosen on | setting picked for the submission | held-out spellF1 | v1 | folds picking v3's setting |
|---|---|---|---|---|
| 7 IDSSE matches (v3's own search) | team 0.5, dead 50, min possession 20, min dead 1 | 0.369 ± 0.026 | 0.224 | 7 of 7 |
| 3 Metrica games | **the same** | 0.498 ± 0.016 | 0.384 | 1 of 3 (the others: team 2 or dead 25, both within the tie band) |
| all 10 matches | **the same** | 0.412 ± 0.075 | 0.272 | **10 of 10** |

**So there is no v4.** The setting chosen on all 10 matches is v3's, as the page required before
anything would change.

### What this says

- **The convention difference is real but modest.** Metrica's events change hands about a fifth less
  often than DFL's.
- **It doesn't change the answer.** On Metrica's own labels, v3's low switch price and 0.8 s minimum
  still win clearly, the search picks them, and the possession count comes out right.
- **A label-free cross-check on our match:** v3 reports 429 possessions over 105.9 graded minutes,
  365 per 90. Metrica's sample games have 304–358 per 90 by their events. Our count sits just above
  that range, so it is not an obvious over-split.
- **What remains unknown:** the truth comes from Metrica's corrected tracking, not events. Nothing
  public measures that convention directly; this is the closest allowed evidence.

