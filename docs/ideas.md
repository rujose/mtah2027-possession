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

