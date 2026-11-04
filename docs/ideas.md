# Ideas queue

Ideas waiting their turn, with where each came from. When one is picked up it gets a model page
written before the code, and its settings come from the bench, never from our match. Observations
from watching become **general rules**, never constants tied to specific frames (RULES.md, #14).

| # | idea | from | why it might help | how it would be tested | status |
|---|---|---|---|---|---|
| 1 | **Players' movement relative to the ball says whether play is live.** In play, the nearest players close in on the ball and the team shifts with it. When it's dead, they stop caring: the ball goes one way and nobody follows, and players walk or turn away | Jose, 2026-10-30 | Catches stoppages where players keep jogging (v4's weak spot: the whistles at 23697 and 54740) and slow live play that looks stopped. Only where the ball is seen, so mostly the first seconds of a stoppage | Ball-relative features (nearest players moving toward the ball; team centre moving with the ball; ball moving with nobody reacting) added to v4's dead-ball model, trained only on frames where the ball is seen, on the bench | queued after step 4 |
| 2 | **"No ball" is a real dead-ball sign on our data too**, because our tracking is Metrica's as well: missing on 83 % of frames where players stand still against 29.5 % where they move. Weaker than Metrica's cleaned samples (92–99 % against 3–10 %) | Jose, 2026-10-30 | v4 excludes the ball, to avoid learning the samples' too-strong version, and so throws away a real signal | First close the bench's known gap (its damaged ball is missing 66 % while stopped, against our 83 %). Then teach the dead-ball model how much "no ball" means, on IDSSE damaged like ours, not on Metrica's blanked samples | queued after step 4 |
| 3 | Jose's answers to the questions: see below | Jose, 2026-10-30 | Football knowledge the data can't give | See ideas 4–9 | answered |

## Jose's observations from watching (2026-10-30)

Asked in three rounds: how stoppages look, who has the ball when the tracking loses it, and tracking
quality. The answers become **general rules or features**, applied to every frame alike and
measured on the bench where it can see them. None of them is a constant tied to a moment of our
match.

| question | Jose's answer |
|---|---|
| What do players do in the first seconds after the referee stops play? | **They keep jogging a while** |
| In slow live play, what shows it's still live? | **The ball keeps moving, opponents keep shifting, teams stay spread out** |
| Quick restarts in this match? | Doesn't remember |
| In crowded moments, what tells you which team came out with the ball? | **Where players start running** |
| How did Athletic play the ball forward? | Mixed, long and short |
| How did Athletic defend Barça's build-up? | Varied |
| Where does the tracking go wrong most? | **When a player has the ball close to another player** |
| What does the ball dot do when the ball is in the air? | **It stays on the ball's shadow on the ground** (no height) |
| Are the goalkeepers well tracked? | **Sometimes a keeper is tracked as the other team, or a player who doesn't exist appears near them** |

### What each answer becomes

| # | idea | from | how it would be tested | status |
|---|---|---|---|---|
| 4 | **Repair keepers tracked as the other team.** A slot that stays inside a goal's 6-yard box for a long stretch while the ball is far away is that goal's keeper: give it the defending team. A general rule over all frames, run as code, not a hand correction (RULES.md allows algorithmic improvements of the tracking) | keepers answer | **Confirmed on our match, label-free:** in the 2nd half, Athletic slot 45 sits in Barça's 6-yard box for 225 s with the ball over 35 m away; slots 40 and 44 for 15 s and 13 s. Slot 45 is the nearest player to the ball inside Barça's area for 8 s, where v6 says Athletic 96 % of the time and v7 says Barça 100 %. It also distorts the restart model (goal-kick shape) and PathCRF's inputs. The bench's damage doesn't swap teams, so the bench can't score it; judged by the label-free counts of how many frames change | **built as [v8](models/v8-keeper-repair.md)** (2026-10-30): 1 piece repaired on our match (slot 45, 230 s graded), 0 on the bench |
| 5 | **Deceleration, not just slowness, marks a stoppage.** Players keep jogging after the whistle, so "team speed now against 2–4 s ago" (falling from running to jogging) is a better early sign than "everyone under 1 m/s" | stopping answer | Add speed-change features to v4's dead-ball model; same out-of-sample bench procedure as v4 | queued, with ideas 1 and 2 (one dead-ball revision) |
| 6 | **Live play looks alive around the ball:** the ball keeps moving, the opponents shift with it, teams stay spread | slow-live answer | Strengthens idea 1 (players' movement relative to the ball): add "the defending team's centre moves with the ball" where the ball is seen. Spread is already in v4's features | merged into idea 1 |
| 7 | **Who won the ball shows in where players start running:** the team that gets it moves toward the other goal, the other turns back | crowded-moment answer | A team clue from each team's velocity along its attack direction, in the seconds after a crowded moment or inside ball gaps. Partly what PathCRF reads already; as an explicit clue it can be measured on the bench and checked against PathCRF's 86 % agreement on our match | queued |
| 8 | **The ball's shadow and close contacts mislead "nearest player".** A high ball's dot sits on the grass under it, so it passes "near" players it flies over. With two players close together, the nearest is a coin toss | tracking and high-ball answers | Exactly what step 5 (count a gain only on evidence of a touch) addresses: require the ball to change course, keep "in flight" until a touch. v1's 8 m/s flight cut already guards part of it | folded into step 5 |
| 9 | No team-specific rules from the style answers (Athletic mixed, varied). They would be constants tied to this match's teams, and the answers show no clear pattern anyway | style answers | — | not pursued |

## Possessions start at the restart, not at the reception (Jose, 2026-10-30)

**Jose's observation, watching the viewer:** after a stoppage, our labels start the new possession
when the ball reaches the receiver. It should start **at the moment of the pass**, whatever the
restart: a throw-in, a free kick, a kick-off.

**What the bench says** (exploration run, v7's out-of-sample bench predictions; no labels of our
match used):

| | the ball at the start of a possession after a stoppage |
|---|---|
| **truth, IDSSE, clean ball** (698 starts) | **leaving 100 %, arriving 0 %**; above 8 m/s in the 2 s before: **0 %** |
| truth, all 10, damaged ball (317 with the ball seen) | leaving 98 %, arriving 2 %; above 8 m/s before: 3 % |
| **v7, bench, damaged ball** (335 seen) | leaving 20 %, arriving 25 %; above 8 m/s before: **56 %** |
| **v7, our match** (53 seen) | leaving 15 %, arriving 36 %; above 8 m/s before: **62 %** |

- **The truth agrees with Jose, exactly:** a possession after a stoppage starts on the frame the ball
  is played, on every IDSSE restart. Our models often start it once the ball is already travelling,
  on our match even more than on the bench.
- **Timing against the truth on the bench:** of 990 restarts, 670 have a v7 start of the same team
  within 10 s. The median lag is 0, but the spread is wide: 30 % are more than 0.5 s late and 31 %
  more than 0.5 s early (quartiles −0.64 / +0.64 s). Another 28 % have no v7 start nearby (the
  stoppage was missed or merged: the dead-ball model's job), and 4 % have the wrong team.
- **Ceiling:** moving each of the 670 starts exactly onto the true one lifts the bench's mean spellF1
  from 0.556 to **0.574 (+0.018), better on 10/10**. A real rule won't place every start perfectly,
  so the gain will be smaller.

| # | idea | from | how it would be tested | status |
|---|---|---|---|---|
| 10 | **Snap the end of each stoppage to the restart pass.** Near each predicted dead → team change, find the moment the ball goes from still to played and start the possession there. Where the ball isn't seen, the taker's movement would have to stand in for it. A general rule, the same for every restart | Jose, restart answer | On the bench, same procedure as every model: the window around each change and the "ball played" threshold chosen by leave-one-out, measured against the +0.018 ceiling; label-free on our match, how often the ball is "leaving" at our starts (15 % now) | **built as [v9](models/v9-restart-pass.md)** (2026-10-30): held-out +0.0056, better on 10/10; on our match 34 of 40 moved starts went earlier |

**Proposed order** (updated 2026-10-30):
1. ~~idea 4 (keeper repair)~~: built as v8;
2. ~~idea 10 (start at the restart pass)~~: built as v9;
3. one dead-ball revision combining ideas 1, 2, 5 and 6, on the bench (it also covers the 28 % of
   restarts with no predicted stoppage nearby);
4. step 5 with idea 8;
5. idea 7 if the bench shows room beyond PathCRF.

## v10 candidates (2026-11-03)

### Where v9's remaining spellF1 is (bench only)

An exploration run over the 10 bench matches, using external data only. Each "oracle" fixes **one**
kind of error with the true labels and nothing else, so its gain is that error's share of what's
left. v9 is 0.561 (window chosen on these same matches); it predicts 349 possessions per match
against 411 true.

| error kind | spellF1 if fixed | gain |
|---|---|---|
| **missed possessions** (a true possession with no predicted one of its team at IoU ≥ 0.3) | 0.812 | **+0.250** |
| **team-change timing** (H ↔ A changes moved onto the true ones) | 0.658 | **+0.096** |
| **team of each possession** (the majority true team) | 0.616 | +0.054 |
| spurious possessions (predicted, no true match: merged away) | 0.604 | +0.043 |
| missed stoppages | 0.584 | +0.022 |
| stoppage start timing (team → D) | 0.576 | +0.015 |
| restart timing (D → team), left after v9 | 0.575 | +0.013 |
| false stoppages | 0.564 | +0.003 |

**The missed possessions:**
- **How many:** 1,443 of 4,114 (35 %).
- **Short:** median 1.6 s (quartiles 0.9 / 3.0 s); 60 % under 2 s, 91 % under 5 s.
- **The ball is seen:** median 100 % of their frames.
- **A player of that team** is within 2 m of the ball at some point in 66 % of them.
- **Where:** 52 % sit between two possessions of the other team (a brief interception or a loose
  touch); 48 % next to a stoppage.

**Team changes:** of 2,180 matched to a true one, the lag is centred (median 0) but wide
(quartiles −0.52 / +0.64 s): 29 % more than 0.5 s late, 26 % more than 0.5 s early.

**So:** most of what's left is **short possessions and the exact moment the ball changes team**.
Both are about **touches**, which we don't detect yet: who touched the ball, and when.

### The variants

Every one is trained and tuned on external data only, and none needs Claude to see our match
(Jose runs our match). Each is one change, scored with the same leave-one-out procedure, and can
be run on IDSSE alone if Metrica's samples turn out not to be usable.

| | variant | attacks | ceiling | how | effort |
|---|---|---|---|---|---|
| **10a** | **Built: [v11](models/v11-touch-line.md), adopted (held out +0.010, 10/10; IDSSE only +0.011, 7/7).** **Touch detector + team changes at the touch.** Find touches from the ball's path: the ball changes course or speed with a player within ~2 m. Move each H ↔ A change to the first seen touch by the new team, as v9 does for restarts | team-change timing | part of +0.096 | extend `possession/passes.py` (passes are touches that send the ball away; add receptions, interceptions, deflections); window chosen on the bench | ~4 h |
| **10b** | **Built in the lab: [short](lab/short.md), kept in v12.** **Short possessions from touches.** Where the other team touches the ball between two possessions of one team (A–B–A), insert the short possession, if the touch is clear | missed possessions (the 52 % between other-team possessions) | part of +0.250 | on 10a's touches; a minimum touch clarity chosen on the bench; checked against spurious possessions | ~3 h after 10a |
| **10c** | **Built in the lab: [learned](lab/learned.md) and [robust](lab/robust.md); robust is v12's start.** **A learned per-frame model in place of the hand weights.** One classifier, P(H / A / D) per frame, trained on the bench from every signal we have (nearest players to the ball, ball speed and course, ball seen, touches, PathCRF, the dead-ball and restart models). Its −log P becomes the decoder's cost; the decoder's settings are chosen again | team of each possession, spurious and missed possessions | part of +0.054 / +0.043 / +0.250 | the bench's usual out-of-sample training (as v4 and v6); replaces v1's judgement weights, the weakest part left | ~1 day |
| **10d** | **Built in the lab: [decoder](lab/decoder.md), discarded** (a shorter minimum never helps). **The decoder re-chosen for short possessions.** With touches as evidence, a shorter minimum possession (now 0.8 s) and a cheaper team switch may now pay | missed possessions | part of +0.250 | the v3 search again, with 10a/10b's evidence in | ~1 h, after 10a |
| **10e** | **Built in the lab: [stoppage](lab/stoppage.md); kept on v11, out on top of the learned model.** **Stoppage start at the ball going out.** v9's mirror: move each team → D change to the moment the ball is seen crossing a line | stoppage start timing | +0.015 | check the truth's convention first (as with v9); snap window chosen on the bench | ~2 h |
| **10f** | **Built in the lab: [dead](lab/dead.md); in v12's `idsse` chain.** **Dead-ball revision** (ideas 1, 2, 5, 6): ball-relative movement, "no ball" learned on damaged IDSSE, deceleration | missed stoppages | +0.022 | retrain v4's model with the new features, same procedure | ~4 h |
| **10g** | **Built as [v10](models/v10-clean-base.md) (2026-11-03).** **Clean base for the rules** (audit items B–D): whistle and set-piece settings set again without our match, or those clues dropped; v5's weight reverted; IDSSE-only retraining if needed | compliance, not score | — | the bench can't score the whistle or tags (no audio, no tags): judged by the rules, not by points | ~3 h |
| 10h | (Measured in the lab's [stoppage](lab/stoppage.md): too rare to build.) Restart timing when the pass isn't seen (ball reappearing, the taker's movement) | restart timing left | +0.013 | after 10a | ~2 h |
| 10i | `X` where we're unsure | macroF1 only (the tie-breaker) | small | X is forward-filled for spellF1, so it can only change macroF1 | ~1 h |

**Proposed order:**
1. **10g** first, because the submission has to be clean whatever else we build.
2. **10a → 10b → 10d**: the touch line attacks the two biggest errors, and each step is measured
   before the next.
3. **10c** if time allows: the largest change, and the one that retires the hand weights.
4. **10e** and **10f** as quick extras.

**Deadline check:** submission 2026-11-17; team registration by 2026-11-10. 10g plus the touch
line fit in about 3 days.

