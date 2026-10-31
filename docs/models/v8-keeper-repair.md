# Model v8: v7 + a repair for keepers tracked as the other team

| | |
|---|---|
| **status** | **built 2026-10-30** (Jose approved the page with both recommended choices) |
| **change from v7** | one step before everything else: tracking pieces that are clearly a goalkeeper but sit in the other team's slots are moved to the right team. All of v7's clues, models and settings stay |
| **code** | `possession/repair.py` (the rule), `possession/v8.py`, `bench/keeper_repair.py` (the false-repair check) |
| **run** | `python -m possession.v8` (~2 min, writes `evaluation/results/v8.csv`); `python -m possession.repair` (what the rule finds, ~3 s); `python -m bench.keeper_repair` (~3 min, writes `bench/results/keeper_repair.json`) |
| **output** | SHA-256 `cfc62b21…` (`cfc62b213c6b6227d1e09e7ff8beaf20654f5d006e1938d9858044325b1ad74b`) |
| **new dependencies** | none |

## Why

**Jose saw it in the video:** "sometimes a keeper is tracked as the other team, or a player who doesn't
exist appears near them" ([../ideas.md](../ideas.md), idea 4).

**Confirmed on our tracking, without labels:**
- In the 2nd half, **Athletic slot 45 stands in Barça's 6-yard box** while the ball is far away.
- It's one unbroken piece: **frame 184,439 to the end of the video.** It's never more than 8 m from
  Barça's goal (median 4.2 m) and inside Barça's penalty area 100 % of the time. That's Barça's
  goalkeeper in an Athletic slot.
- **230 s of that piece are graded:** frames 184,439–190,198, the end of the 2nd half. The rest is
  after the final whistle, where every label is X anyway.
- For 8 s it's the nearest player to the ball inside Barça's area. There v6 says Athletic 96 % of
  the time; v7 already says Barça, because PathCRF reads it from movement.

**What a wrong-team keeper still breaks in v7:**
- **v1's "who is on the ball":** the keeper with the ball counts for Athletic.
- **The restart model:** a goal kick is exactly "this team's keeper in its own 6-yard box, the other
  team outside the area". With the keeper in the wrong team, the shape points to the wrong team.
- **PathCRF's input:** a "goalkeeper" in the wrong team, and one team a player short.

## The rule (`possession/repair.py`)

For each half and each goal:

1. **The defending team** of a goal is the team whose players stand nearer to it on average over the
   half.
2. **A suspect stretch:** a slot of the **other** team is inside that goal's 6-yard box, **no**
   defending-team slot is in the box, and the ball is **seen** more than 35 m from the goal, **for at
   least 5 s in a row.** A missing ball doesn't count as far.
3. **The piece:** the slot's unbroken run of presence, within the half, that contains the stretch.
   It's repaired only if it stays **inside that penalty area on at least 90 % of its frames**: a
   keeper does; an attacker who wandered in doesn't.
4. **The repair:** the piece moves, whole, to a **new slot** with the defending team, appended after
   the existing ones (slot 46 on our match). Nothing is deleted or invented; positions and speeds are
   untouched; existing slot numbers keep their meaning. Every later step (clues, models, PathCRF,
   the viewer) sees it as the defending team's player.

| setting | value | why |
|---|---|---|
| 6-yard box | 5.5 m deep, 18.32 m wide | Laws of the Game |
| penalty area | 16.5 m deep, 40.32 m wide | Laws of the Game |
| ball far | seen more than 35 m from the goal | about a third of the pitch: no attacker has a reason to stand in the 6-yard box then |
| lasting | at least 5 s (125 frames) | a keeper stands there; an attacker passing through doesn't linger with the ball that far away |
| piece inside the area | at least 90 % of its frames | keepers rarely leave the area |

All of them come from reasoning and the Laws, not from our match's labels. The rule applies to
every frame of every match alike, so it is an algorithmic tracking improvement, which RULES.md
allows. It is not a hand correction: nobody picks the frames.

**v8 itself** is one line: `v7.predict(repaired tracking)`. The whistles come from the audio and
are the same.

## Results

### The rule

| | pieces looked at | repaired |
|---|---|---|
| **our match** | 1: slot 45, half 2, frames 184,439–190,198 (230 s); the rule fired on 110 s of it; inside Barça's area 100 % | **1**, as slot 46 (Barça) |
| **bench, 10 matches, published tracking** | 0 | 0 |
| **bench, 10 matches, damaged at 30 / 41 / 55 %** | 0 / 0 / 0 | 0 / 0 / 0 |

- **Zero false repairs on the bench.** The bench never swaps teams, so any repair there would be a
  mistake, and the rule never fires, not even on a suspect stretch.
- **For the same reason, the bench can't score v8:** it equals v7 there by construction.

### Our match, against v7 (label-free)

- **446 frames differ (0.28 % of graded):** all inside the repaired piece, except 3 frames just
  before it (the decoder looks at the whole half).
- **Possessions** 347 → 351; dead share 46.9 % → 46.8 %; the whistle check is unchanged (74/86;
  counted 11/11).
- **Inside the piece:**

| | v7 | v8 |
|---|---|---|
| "who is on the ball" | — | changes team on 82 frames (3.3 s) |
| restart model, mean P(Barça restarts) | 0.449 | **0.689** (moved by more than 0.1 on 5,195 of 5,759 frames) |
| PathCRF, mean P(Barça has the ball) | 0.543 | 0.516 (moved by more than 0.1 on 857 of 3,815 frames) |
| labels H / A / D (frames) | 2,037 / 1,817 / 1,905 | 2,012 / 1,949 / 1,798 |

**Read:**
- The restart model moves the most, as expected. With the keeper back in Barça, stoppages near
  Barça's goal look like Barça's restarts (goal kicks).
- The labels move both ways, mostly as boundary shifts under 2 s. Without labels we can't say which
  is right. The repair is kept because it fixes the input, not because of a measured gain.
- **The one change worth watching:** frames 189,127–189,272 (5.8 s), where v7 said dead or Barça and
  v8 says Athletic. Smaller ones: 184,819–184,901 (H → A), 186,345–186,390 (A → H) and 186,290–186,325
  (A → D).

**Public 5 minutes (pipeline check only):** identical to v7 (spellF1 0.606, macroF1 0.857), as it
must be. The repair is in the last minutes of the 2nd half.

## Not covered

The "player who doesn't exist" near a keeper (a phantom track). It never puts a player on the wrong
team, and it's harder to tell from a real defender standing close. If the viewer shows it matters,
it's a separate rule.

## Risks

1. **A real attacker** who stands in the 6-yard box for 5 s with the ball 35 m away, and stays in the
   area for the whole piece. Unlikely in open play, and the 90 % check guards against it. It never
   happened on the bench.
2. **One match, one keeper, one occurrence.** The effect on the score is small: 230 s near one goal.
   It fixes a known defect rather than chasing points.

## Decisions (Jose, 2026-10-30)

| | choice | taken | why |
|---|---|---|---|
| 1 | Repair the whole piece, not only the 110 s where the rule fires | **yes** | The piece is one tracked person, always inside the area; repairing only the fired seconds would leave the same keeper in the wrong team the rest of the time |
| 2 | Name it v8 (v7 + keeper repair) | **yes** | One change per version |
