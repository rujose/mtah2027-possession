# Convention audit: does v12 answer the organisers' question? (2026-11-09)

v12 learned what possession is from the bench's labels, mostly IDSSE's. The organisers rank on their
own truth, with their own definition. If the two disagree systematically, a strong bench score could
hide a weaker real one. This audit compares them **from documents and the bench only**: nothing of our
match's truth, public window included, was read.

```bash
.venv/bin/python -m bench.convention_audit      # ~10 min; bench/results/convention_audit.json
```

## The organisers' definition, point by point

From CHALLENGE.md. The bench's sources: IDSSE's labels (DFL's tracking feed: `BallStatus`, `BallPossession`
per frame; `bench/labels.py`), and Metrica's 3 matches (Metrica's events through a recipe written to
follow the organisers' rules; `bench/metrica.py`).

| organisers' rule | IDSSE's labels | Metrica's labels | agree? |
|---|---|---|---|
| Truth derived **mechanically from corrected tracking**: D where it marks the ball dead, else the team it names as owning the ball | the same kind: DFL's tracking feed names the owner and the ball status per frame | built from human-tagged events | IDSSE closer in kind |
| Three states, H, A, D | yes | yes | yes |
| A loose ball stays with the last team | yes (a gap carries the last team) | yes (a pass in flight, a lost ball stay with the last team) | yes |
| After a stoppage there's always a clear possessor | the first team is the restart's team after 97.9 % of stoppages | the restart's team | yes |
| D starts when the referee stops play | 3–4 frames after the ball crosses a line, or at the whistle | the end of a BALL OUT event, a foul, an offside | close; their exact timing is unknown |
| **A change needs controlled possession**: a touch without control, a deflection or an aerial duel won without control doesn't count | DFL's own rules for its owner field: not documented here | follows the rule: only controlling events (pass, carry, shot, recovery) change it | **the open point** |

## How often possession changes, by source

| | possessions per 90 min | changes per 10 live min | under 1 s |
|---|---|---|---|
| IDSSE's frame labels (what robust learned from) | 412 | 77.7 | 12.3 % |
| DFL's events through the organisers' rules | 382 | 67.1 | 10.7 % |
| Metrica's events through the organisers' rules | 330 | 52.8 | 6.9 % |
| organisers' public window, from their text ("17 possessions" in 5 minutes) | ~306 if it held all match | | |

Source: [convention-check.md](convention-check.md), and CHALLENGE.md / EVALUATION.md for the 17. The public
window is 5 minutes of one match: a hint, not a measure, and never used to choose anything.

**So the organisers' truth looks stickier than what robust learned from**, closer to Metrica's
rules-following labels.

## Does that hurt v12? Measured on the bench

v12 after each stage, and v12 without `short` (the stage that adds short possessions at touches), each
out of fold, scored on IDSSE's 7 matches against three versions of the truth:

| | IDSSE's labels | DFL's events, the organisers' rules | (control proxy, 0.2 s) |
|---|---|---|---|
| robust | 0.644 | 0.502 | 0.449 |
| + short | 0.657 | 0.499 | 0.430 |
| + segments | 0.664 | 0.508 | 0.447 |
| v12 | 0.664 | 0.508 | 0.447 |
| v12 without short | 0.650 | **0.512** | 0.468 |
| `short`'s gain (better on) | +0.014 (6/7) | **−0.003 (3/7)** | −0.020 (0/7) |

- **Against DFL's events under the organisers' rules, `short` is about neutral** (−0.003, 3 of 7): no sign
  that it fights the definition. The whole column is lower because event times are off by about ±1.8 s,
  which blurs every version's edges alike.
- **On Metrica's rules-following labels, `short` helps:** +0.017 to +0.023 on all 3, in every chain it was
  tested in ([lab.md](lab.md)). And v12 doesn't over-split there: about 328 possessions against 351 true.
- **The control proxy failed as a stand-in.** "A player within 1 m of a ball slower than 2 m/s" leaves only
  125–168 possessions per 90 minutes, far below any source: one-touch play rarely slows the ball that much.
  Against a truth that merges that much, anything that adds possessions loses, so that column says nothing
  about `short`. It's kept as measured, not used.

## Verdict

- **No evidence that v12 answers a different question.** Where the bench's labels follow the organisers'
  written rules (Metrica's; DFL's events), v12 and its riskiest stage hold up: helpful on one, neutral on
  the other.
- **What stays open is the organisers' own owner field:** how their corrected tracking decides who "owns" the
  ball, and whether a brief controlled touch by the other team starts a possession. Only they can say. It's
  the question for the email.
- **What would change the plan:** an answer that their truth ignores possessions shorter than some length,
  or requires a completed pass. Then `short` and `recall` would be removed and the chain judged again on
  Metrica's labels. Until then, v12 stays.
