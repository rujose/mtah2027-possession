# The whole process, explained in depth

Written 2026-11-16, the day before the deadline, for Jose: everything from zero to v17, and how we got there.

- **How it relates to the other pages:**
  - [explained.md](explained.md) is the short version, written on 2026-10-30 and stopping at v12;
  - [technical-document.md](technical-document.md) is the few-page summary for the organisers;
  - this page sits between them and the detailed pages. It explains each idea as if new, tells the story in order,
    and links to the page that holds the details.
- **Where its facts come from:** every number here is taken from those pages and from the code in
  [../possession/](../possession/). Where two pages disagree, it follows the later page and the code.
- **What it leaves out:** nothing about what our program outputs on the provided match (counts, shares, scores). We
  don't read those (Part 2 says why).

**Contents**
1. [The task from zero](#part-1-the-task-from-zero)
2. [The rules, and the line we walk](#part-2-the-rules-and-the-line-we-walk)
3. [Other matches, and the test bench](#part-3-other-matches-and-the-test-bench)
4. [The story, version by version](#part-4-the-story-version-by-version)
5. [The lab, round by round](#part-5-the-lab-round-by-round)
6. [v17 end to end](#part-6-v17-end-to-end)
7. [Every decision, and who made it](#part-7-every-decision-and-who-made-it)
8. [What's solid, what's weak, what's next](#part-8-whats-solid-whats-weak-whats-next)
9. [Glossary](#part-9-glossary)

---

## Part 1: the task from zero

### 1.1 One match, one letter per frame

- **The match:** FC Barcelona (home) against Athletic Club (away), LaLiga 2024-25, matchday 2. The files call it
  `BAR-ATH`.
- **What we hand in:** a CSV with a header `frame,label` and exactly **199,319 rows**, one per frame of the
  *tactical* video (the wide fixed camera). That's 25 frames a second, about 2 h 13 min including the time before
  kick-off and half-time.
- **The letters:**

| letter | meaning |
|---|---|
| `H` | Barcelona has the ball, ball in play |
| `A` | Athletic has the ball, ball in play |
| `D` | dead ball: out of play, a foul, a goal, waiting for a restart |
| `X` | no answer |

- **Only the two halves are graded:** frames 11,185–90,651 and 110,808–190,197, which is **158,857 frames**, about
  106 minutes. Everything outside is ignored.
- **One trap:** the second half is graded from frame 110,808 but kicks off at 113,742, so its first **2,934 graded
  frames are all `D`** (the FAQ says so).

([challenge.md](challenge.md) is our summary of the organisers' documents.)

### 1.2 What counts as possession

From [../CHALLENGE.md](../CHALLENGE.md):
- While the ball is in play, exactly one team has possession.
- Possession changes only when the other team gains **controlled** possession. A touch without control, a deflection,
  or an aerial duel won without keeping the ball does **not** change it.
- While the ball is loose or contested, possession stays with the last team that had it.
- After every stoppage there is a clear team that restarts.
- **The answer key** was made mechanically from Metrica's *corrected* tracking: `D` where it marks the ball dead,
  otherwise the team it names as owning the ball.

That last point matters a lot later. "Owning the ball" in Metrica's corrected tracking is a convention: when exactly
does a touch become ownership? We measured it on Metrica's public games (section 3.7).

### 1.3 How we are scored

**A possession** is one unbroken run of the same team's letter while the ball is alive. `H H D D H H` is two
possessions, not one: a stoppage splits them.

**spellF1, the ranking metric:**
1. Our possessions and the true ones are paired one to one, same team only, the best overlaps first.
2. Overlap is measured as **IoU** ("intersection over union"): the time both cover, divided by the time either covers.
   1 means identical stretches, 0 means no overlap.
3. A pair counts as found if its IoU passes a threshold. From the pairs come precision (how many of ours are real) and
   recall (how many real ones we found), and their F1 (one number from both: 1 perfect, 0 useless).
4. The F1 is averaged over five thresholds: 0.3, 0.4, 0.5, 0.6 and 0.7.

**A small example.** The truth has Athletic in possession for frames 100-150 (50 frames).
- We say Athletic for 110-150: overlap 40, union 50, IoU 0.8. Found at all five thresholds.
- We say Athletic for 120-150: overlap 30, union 50, IoU 0.6. Found at 0.3-0.6, missed at 0.7.
- We also invent a 5-frame Barcelona possession in the middle: the true Athletic possession is now split in two, each
  piece overlaps it less, and the extra Barcelona possession matches nothing. Precision and recall both drop.

So spellF1 punishes two things: **flicker** (inventing many short possessions) and **merging** (gluing two real
possessions into one). Being one frame late is almost free; one second late costs about a third of a possession's
value.

**macroF1** breaks ties: the share of frames with the right letter, scored per letter (H, A, D) and averaged. A
program that flickers can have a good macroF1 and a terrible spellF1. The organisers' own baselines show it:

| baseline | macroF1 | spellF1 |
|---|---|---|
| nearest player to the ball, raw | 0.6532 | 0.0486 |
| the same, majority vote over ±2 s (**the floor to beat**) | 0.6724 | **0.2433** |

**How `X` is scored:** spellF1 replaces each `X` with our previous answer in the same half. So an `X` inside a
possession is free, but an `X` across a change drags the old team forward. We only write `X` outside the halves.

**Noise:** on one full match, differences under about 0.04 spellF1 are luck. There is **no leaderboard** and no
feedback before results day.

### 1.4 What we were given

| file | what it is | do we use it? |
|---|---|---|
| tactical tracking (`_atd.xml` + `_atd.txt`) | every frame, where each tracked player and the ball are on the pitch (x, y in metres, no height), plus a speed | **yes, the main input** |
| tactical homography | per frame, a 3×3 table that turns pitch positions into video pixels | no (only needed to draw on the video) |
| Smart Tagging | Metrica's automatic phase tags ("Barcelona build-up", "corner") | **only the 2nd half's kick-off tag** |
| tactical video (with audio) | the wide camera, with the referee's whistle in its sound | watched to understand the data; the program reads nothing from it since v10 |
| broadcast set (tracking, homography, video) | the TV feed: zooms, replays, cut-aways; tracked like the tactical video but not in sync with it | **yes, since v17** (its players, and its homography's per-frame validity flag) |
| the public 5 minutes of ground truth | the real answers for the first 5 minutes | only to check the pipeline isn't broken |

([data.md](data.md) measures every file. Every number on that page comes from `tools/explore_data.py`.)

### 1.5 What's wrong with the tracking

The tactical tracking is produced automatically by Metrica's software and was **not corrected**. Measured on the
graded halves ([data.md](data.md), [bench.md](bench.md)):

| fault | measured | what it means |
|---|---|---|
| **ball missing** | **41 %** of graded frames (40.7 %), in 150 gaps, median 10.7 s, longest 3.7 min | the ball disappears in long blocks, not flickers: joining the dots fixes almost nothing |
| ball missing while players stand still | **83 %**, against 30 % while they move | a missing ball is a strong hint of a stoppage, which helps |
| **players cut into pieces** | about **1,200 pieces** over the halves, spread over **46 slots** (24 Barcelona, 22 Athletic) | a slot is not a person: when the software loses someone and finds them again, they usually land in another slot. Only the slot's team is fixed |
| players visible | **20.5 of 22** on average | at any moment one or two players are missing, sometimes near the ball |
| more than 11 of a team | 8.1 % of frames for Barcelona, 4.7 % for Athletic | someone counted twice |
| player speed missing | 8 % (9.8 % over the whole video) | |
| the ball's speed column | filled on 8.7 % of ball frames, with wrong values | we compute the ball's speed ourselves from positions |
| ball outside the lines | 4.4 % of ball frames | a free dead-ball hint |

Two consequences shape everything after:
- **Everything has to be about "a Barcelona player" or "the team"**, never "player 7".
- **The ball is the main difficulty.** For 41 % of the match we must guess who has it from the players alone.

The public 5 minutes are unrepresentative here: the ball is visible on 88.2 % of them, against 59.3 % over the
graded match. That's one more reason they can't be used to choose anything.

---

## Part 2: the rules, and the line we walk

### 2.1 What's forbidden, and what's allowed

The organisers clarified the rules for everyone on **2026-11-03**, overriding the documentation and earlier answers
to individual questions ([limitations.md](limitations.md), section 0):

| not allowed | allowed |
|---|---|
| checking our output against the video | automatic checks using only the provided data |
| hand labels, human-produced data, any data made by hand on the provided game | inspecting the data and the video to understand what counts as possession |
| uploading the provided data to AI services | the video, tracking, Smart Tagging or other generated data **as inputs**, if nothing was trained on the ground truth or the video |
| training, fitting or calibrating on the ground truth or the provided video | files of intermediate steps |
| | any dataset we have the rights to use |

**Training and tuning: only on external data.** Every parameter that needs tuning must be tuned on other matches.

Other rules from the start ([limitations.md](limitations.md), section 1):
- no human-made data about this match from anywhere (Opta, StatsBomb, match reports);
- no rules or constants tied to specific frames ("between 03:06 and 03:26 the ball is dead" in code);
- the public 5 minutes check the pipeline only: choosing a version because it scores better there is tuning;
- the file must be regenerated **byte for byte** by our public repository;
- the data must not be published: `data/` never enters git.

### 2.2 How we read the rules before 2026-11-03, and what changed

**Before the clarification**, under our reading of 2026-10-30 ([decisions.md](decisions.md)), we inspected the data
normally, including with Claude (the AI coding assistant), and no LLM ever labelled frames. The clarification
overrode that reading. Our audit found five items ([limitations.md](limitations.md), section 0):

| item | what had happened | what we did |
|---|---|---|
| **A** | provided data reached an AI service: Claude saw single-frame screenshots of the viewer (with the video), taken while we built and tested it, printed tracking values, our labels and statistics of the match | from then on Claude runs nothing on the match whose output it reads, beyond file paths, SHA-256 hashes, row counts and pass/fail format checks. Disclosed in the technical document |
| **B** | our output was checked against the video: v2's whistle changes watched, v5's whistle weight came from watching two frames, v9's idea came from seeing late restarts | the whistle (v2, v5) removed in v10. v9's rule rests on the bench's truth and its window was chosen on the bench only (and its "snap" is switched off in the branch our match runs since v15) |
| **C** | two settings had been calibrated on this match: the whistle detector (on its audio) and a set-piece window (on three tags and the public window) | **both removed in v10** |
| **D** | Metrica's sample data has no licence file | used as its README asks, credited. From 2026-11-15 it also trains the network our match runs (Jose's decision) |
| **E** | grey: statistics of the provided tracking shaped external data | the bench's damage (section 3.3) and later v17's simulated broadcast follow statistics of our tracking (never labels or video). The parameters themselves are still fitted on external data |

**The line we walk, in one sentence:** looking at the provided data to *understand* it is allowed; turning what we see
on specific frames into labels, corrections, settings or choices is not.

Examples of the line from our log:
- **Allowed:** the keeper repair (v8). Jose saw a keeper tracked as the other team. The fix is a general rule with
  constants from the Laws of the Game and reasoning, and it would work on any match.
- **Allowed:** an *idea* can come from watching an error (v2's whistle idea came from Jose watching a v1 error in the
  public window). Its *settings* must come from elsewhere.
- **Not allowed:** a whistle threshold set from our audio, or a version chosen because it looks better on the video.

### 2.3 Our own working rules, added on top

- **A page before a build.** Every model version has a page in [models/](models/) written *before* it is built: why,
  how, every setting and where its value comes from, expected failures. Results are added after.
- **A bar before a score.** Every candidate is judged against a bar written down before its first score
  ([lab.md](lab.md)). Moving the bar after seeing scores would be choosing after the fact.
- **Claude never reads our match since 2026-11-03:** not its tracking (broadcast included), its labels, the public
  truth, nor any statistic of our output. On our match it runs only scripts whose output is paths, SHA-256, row counts
  or lane names. The validator's label counts stay hidden (only its pass/fail line is shown).
- **The switch is never overridden** (section 6.5). `--show-branch` prints which branch our match took; it's Jose's
  to run, never Claude's, and nothing is ever changed by hand from it.
- **The viewer's `--models` flag** (since 2026-11-04) can show our versions' labels over the video so Jose can see a
  version run. Claude may build it, never open it, and no version or setting has been chosen from it.
- **Subagents follow the same rules.**

**One slip, recorded** ([lab/push.md](lab/push.md), 2026-11-15 19:45): while recalling how v16's clean copy was
checked, an old validator log was printed through a filter that let two summary lines through. Nothing was decided on
them, and since then only the validator's pass line is shown.

### 2.4 The grey zones we live with

These are disclosed in the technical document:
- **What happened before 2026-11-03** (items A-C above).
- **Statistics of our tracking shape the bench** (item E): how often and how long the ball goes missing, the piece
  lengths, the missing players and speeds; and, for v17, the broadcast's players per frame, piece lengths, valid share
  and alignment distance. We list it because it sits near the line: it makes the bench resemble our match, without
  labels or video.
- **Metrica's sample data has no licence file** (item D). Metrica publishes it and runs the challenge, but there is
  no written permission.
- **The viewer flag** (section 2.3).

---

## Part 3: other matches, and the test bench

### 3.1 Why we need other matches

Without labels for our match, no leaderboard, and a public window we may not use to choose, there is only one way to
tell a better version from a worse one: **run both on other matches whose answers we have**, made to look like ours.
That is the bench ([bench.md](bench.md)). Every setting and every version choice in this project was made there.

### 3.2 The data from other matches

| dataset | what it is | who labelled it | how we use it |
|---|---|---|---|
| **IDSSE** (Bassek, Rein, Weber, Memmert, *Scientific Data* 2025; CC BY 4.0) | 7 matches of 2022-23 (2 Bundesliga, 5 2. Bundesliga), clean tracking at 25 fps, 1,002,644 frames (~11 h) | DFL's official feed: per-frame `BallPossession` and `BallStatus`, read exactly as written | training and the bench |
| **Metrica's sample games** (github.com/metrica-sports/sample-data; no licence file) | 3 anonymised matches, corrected tracking and tagged events | Metrica tagged the events; we turn them into per-frame H/A/D with a fixed recipe (`bench/metrica.py`) | training and the bench; they carry the answer key's convention |
| **SkillCorner open data** (MIT) | 20 A-League 2024-25 matches of **broadcast** tracking at 10 fps | SkillCorner's own ball-status and possession fields | a test on real messy tracking; its stoppage balls are also transplanted into the bench (section 3.5) |
| **PathCRF's pretrained weights** (MPL-2.0) | a published model's trained weights | trained by its authors on DFL data (6 of IDSSE's 7 matches) | one input to our model (section 6.3) |

**"Unseen" matches.** PathCRF was trained on 6 of the 7 IDSSE matches, so on those its output is better than it will
be on ours. The 4 matches it never saw (IDSSE's J03WR9 and Metrica's 3) are the only ones where every input is truly
out of sample. The lab's bar checks them separately.

Two data sources we couldn't use: **PFF's World Cup 2022** (its terms forbid benchmarking use) and **SoccerTrack v2**
(it needs an account we didn't make).

### 3.3 The damage: making other matches look like ours

IDSSE's and Metrica's tracking is clean. Ours is not. So each bench match is damaged to imitate our faults
(`bench/degrade.py`), with a fixed random seed so every run sees the same damage:

1. **Anonymise:** drop player IDs, keep the team.
2. **Fragment:** cut each player's track into pieces whose lengths follow ours, and put each piece into a free slot of
   its team (46 slots, as ours). Leave short holes so that on average 20.5 of 22 players are visible.
3. **Blank 8 % of player speeds**, as ours.
4. **Lose the ball** as a chain that, every frame, may switch between "seen" and "missing":
   - the chance to lose a seen ball follows ours, by how crowded it is (players within 3 m) and by whether players stop
     within the next 5 s, times how much likelier ours is to lose it near or beyond a line (5× beyond a line);
   - the chance to find a missing ball follows ours, by whether players are stopped;
   - the overall rate is scaled to three levels: **30 %, 41 % (ours) and 55 %** missing. Several levels catch a method
     that only works at one.
5. **Drop the ball's height** (we have none) and add our tiny position noise.

**How close it gets** (level 41):

| | bench | ours |
|---|---|---|
| ball missing overall | 41.0 % | 40.7 % |
| … while players are stopped | 65.8 % | 83.0 % |
| … while players move | 34.3 % | 29.5 % |
| gap length median / mean | 10.8 / 15.4 s | 10.7 / 17.4 s |

**Known gap:** on the bench a missing ball is a weaker hint of a stoppage than on ours. So the bench undersells ideas
that read "no ball" as "dead", and is a little harsher on team calls in play.

**Not imitated:** false ball detections, identity swaps inside a piece, the same player in two slots, and wrong
teams (that last one was measured later: section 5.10).

### 3.4 Scoring on the bench honestly

- **Held out:** a match is scored only by models and settings that never saw it.
- **Leave one match out:** to choose a setting, try each candidate on 9 matches, pick the best, score it on the 10th;
  repeat for each match. The average of the 10 scores is the **held-out score**, the honest estimate.
- **Nested:** when a search itself picks among many settings, the pick is also made without the held-out match.
- **Out of fold:** a later stage is trained on the earlier stages' outputs for matches those earlier stages never
  saw, so what it learns from looks like what it will meet.
- **Noise:** between random seeds of the same recipe, the bench moves by about **0.0035**; from match to match by
  **±0.03**; and in round 5 one comparison moved by 0.002 from settings alone. So a gain under ~0.005 is weak evidence.
- **The winner's curse:** trying many ideas on the same 10 matches flatters the best one. The bar's per-match and
  per-source checks are there to catch that, and we expect less on our match than the bench says.

**The bar** ([lab.md](lab.md), fixed before any candidate ran): a candidate is kept only if
1. its held-out mean beats the version without it;
2. at least 7 of 10 matches improve (5 of 7 on the IDSSE-only set);
3. neither source drops (IDSSE's matches, Metrica's);
4. the 4 matches PathCRF never saw don't drop.

From round 7 on, a gain had to be at least **+0.005**.

**Two bench sets.** `all` is the 10 matches. `idsse` is IDSSE's 7 alone, kept as a fallback if Metrica's unlicensed
data had to go. Each set gets the stages that pass on it, so for a while the two chains differed. Our file uses `all`.

### 3.5 Stressed benches, and B's bench

In round 7 (2026-11-09) a test on SkillCorner's real tracking found a failure the bench couldn't show (section 3.6):
**in some matches the ball stays tracked and moving through stoppages** (carried to a throw-in, placed for a free
kick). On IDSSE the dead ball freezes, and on Metrica's and ours it's mostly missing, so our models had learned
"dead ⇔ no ball".

So the lab built **stressed variants** of the bench ([lab/carried.md](lab/carried.md)):
- each match's true stoppages get **SkillCorner's own dead-time ball**, transplanted: the template's ball relative
  to its restart spot, laid onto the bench's restart spot;
- **25 % and 50 % stressed**: that share of each match's stoppages gets such a ball; 10 % was also used;
- **fully stressed**: every stoppage gets one;
- **the split:** the test variants take their templates from SkillCorner's **odd-indexed** 10 matches; training
  copies (for `augment` and the branch-B networks) take them from the **even-indexed** 10. So no test shows a model a
  stoppage it trained on.

**B's bench** is the mean of the 25 %, 50 % and fully stressed variants. It's where branch B (the branch our match
runs) is judged from round 8 on. The numbers 0.698 → 0.733 → 0.747 → 0.754 in Part 4 are on B's bench, and can't be
compared with the normal bench's 0.724.

### 3.6 SkillCorner: the first test on real tracking

On 2026-11-09 v12 was run on SkillCorner's 20 matches, a source no model, setting or design had seen
([lab/skillcorner.md](lab/skillcorner.md)). It scored **0.308**, against 0.704 on the bench, and **below v11** (0.366)
on 19 of 20 matches. The diagnosis found two causes:
1. **Stoppages with the ball still tracked and moving:** v12 called 81 % of those frames live. With the true
   stoppages it would have scored 0.473.
2. **The team, and when it changes:** right on 83 % of live frames (92 % on the bench), with changes spread ±1.2 s
   around the truth (±0.1 s on the bench). Partly SkillCorner's labels and broadcast positions: its new team is 2.1 m
   from the ball at a change, 0.6 m on IDSSE.

The first led to the stressed bench, `augment`, and the switch (v13). Since then every version scores **about 0.41**
on SkillCorner. That's the plainest reminder that **the bench's 0.75 is not a forecast of our score**: it ranks ideas.

### 3.7 Conventions: whose answer key?

Each data provider has its own rules for when possession changes.
- **DFL's labels change hands about 25 % more often than Metrica's:** 412 possessions per 90 minutes against 330
  ([owner-rule.md](owner-rule.md)).
- **Metrica's convention**, measured on its 3 games on 2026-11-12 ([owner-rule.md](owner-rule.md)): the corrected
  ball rides with its owner (about 0.1 m from him), and **any deliberate play counts**, a one-touch pass or a shot
  included, but **accidental contact doesn't** (deflections, blocks).
- **Why it matters:** the answer key follows Metrica's convention, and our match is Metrica's tracker. But 7 of the
  bench's 10 matches are DFL. So a bar that counts "7 of 10 matches better" is mostly a DFL vote. From v14 on, the
  later versions lean toward Metrica's games, with written rules for when that's allowed (Part 4, v14 and v16).
- **v13 already sat close** to Metrica's convention: 5-8 % fewer possessions than Metrica's labels on its 3 games.

### 3.8 What the bench can't tell us

- **Our final score.** Damage and convention only approximate ours.
- **Inputs other matches don't have:** audio (the whistle), Smart Tagging, and the broadcast. IDSSE has none of them.
  For the broadcast, v17's test had to *simulate* one (section 6.7).

---

## Part 4: the story, version by version

Each version changes one thing, so each gain or loss has a cause. All scores below are spellF1, held out, at 41 %
damage. They're comparable within a column, not always across rows: the bench grew from 7 matches to 10, and from
v15 on branch B is judged on B's bench.

| version | date | the change | bench |
|---|---|---|---|
| floor | | the organisers' nearest player, ±2 s vote | 0.217 (IDSSE) / 0.364 (Metrica) |
| v1 | 10-30 | hand-weighted clues + a Viterbi decoder | 0.224 (IDSSE) |
| v2 | 10-30 | + the referee's whistle from the audio | can't be benched |
| v3 | 10-30 | decoder settings chosen on the bench, with minimum lengths | 0.369 (IDSSE); 0.412 (10) |
| v4 | 10-30 | dead-ball probability from gradient-boosted trees | 0.433 (10) |
| v5 | 10-30 | the whistle's weight restored | = v4 on the bench |
| v6 | 10-30 | which team restarts (logistic regression) | 0.443 (10) |
| v7 | 10-30 | + PathCRF | 0.565 on the 4 unseen (v6 0.501) |
| v8 | 10-30 | + the keeper repair | = v7 (never fires there) |
| v9 | 10-30 | possessions start at the seen restart pass | 0.561 (v7 0.556), 10/10 |
| v10 | 11-03 | no whistle, no set-piece window (the clarification) | = v9 |
| v11 | 11-03 | each change of team moved onto the new team's touch | 0.571, 10/10 |
| **v12** | 11-04 | the lab's learned chain | **0.703**, 10/10 |
| v13 | 11-09 | a temporal network (GRU) + the stoppage switch | 0.716; SkillCorner 0.408 |
| v14 | 11-12 | branch A re-weighted on Metrica's convention | 0.724 |
| v15 | 11-14 | branch B rebuilt | B's bench 0.698 → 0.733 |
| v16 | 11-15 | B's network trained on Metrica's games too | B's bench 0.747 |
| **v17** | 11-15 | the broadcast's players merged into what B reads | B's bench **0.754** (simulated broadcast) |

### 4.1 v1: clues and a decoder (2026-10-30)

[models/v1-clues-and-decider.md](models/v1-clues-and-decider.md)

**The idea:** compute a few **clues** per frame, turn them into **costs** (how surprised we'd be by each letter),
and let a **decoder** pick the cheapest sequence of H/A/D (section 6.9 explains the decoder).
- **Three clues:** who's on the ball (the nearest player of each team to a seen, slow ball); is play dead (everyone
  stopped, the ball out); and Smart Tagging's hints.
- **Switch prices:** 15 for changing team, 25 for entering or leaving D. Reasoned, not fitted.

**What the bench said** (built the same day): v1 **tied the floor** (0.224 against 0.217), though far better on dead
ball and per-frame accuracy. The reason was plain: **v1 merged**. It found about 148 possessions a match where there
were about 437. Its switch prices needed about a second of one-sided evidence to change team, so most real
possessions were swallowed.

### 4.2 v2: the whistle (2026-10-30)

[models/v2-whistle.md](models/v2-whistle.md)

A referee's whistle means the ball is dead. v2 detected whistles in the tactical video's audio and counted one only if
the players slowed down after it. The idea came from Jose watching a v1 error. The bench has no audio, so it could
never be measured. Its detector's thresholds were set on our audio, which the clarification later forbade, so it
left in v10.

### 4.3 The bench, and v3: the decoder that knows how long things last (2026-10-30)

[bench.md](bench.md), [models/v3-decoder.md](models/v3-decoder.md)

Once the bench existed, the first question was v1's merging. v3 kept v2's clues and changed only the decoder:
- **a much lower team switch price** (0.5, was 15);
- a dead-ball price of 50;
- **a minimum possession of 20 frames (0.8 s)**: once a team's possession starts, it lasts at least that long.

The minimum stops flicker without needing a high switch price, so real short possessions can get through. Settings
were chosen on the bench, leaving each match out, and all 7 folds picked the same setting. **+0.14** (0.224 → 0.369),
the largest single jump until v12.

**The convention check** ([convention-check.md](convention-check.md)) came next: Metrica's events change hands about
20 % less often than DFL's. On Metrica's own labels v3's settings still won, so nothing changed.

### 4.4 v4: a learned dead ball (2026-10-30)

[models/v4-dead-ball.md](models/v4-dead-ball.md)

v1's two hand rules for "players stopped / active" became **a probability that the ball is dead**, from
**gradient-boosted trees** (section 6.3) on 79 features of the players (their speeds, how many stand on a line, the
team shapes), trained on the 10 bench matches. A rule fixed in advance chose trees over logistic regression (they won
by 0.013). Stoppages found rose from 48 % to 81 %; the bench from 0.412 to **0.433**, better on 8 of 10.

### 4.5 v5: the whistle's weight (2026-10-30)

Jose watched two whistles v4 missed, and v5 gave the counted whistle a stronger weight (4.30). It changes nothing on
the bench (no audio). This was output checked against the video (item B), so it left with the whistle in v10.

### 4.6 v6: who restarts (2026-10-30)

[models/v6-restart-team.md](models/v6-restart-team.md)

After every stoppage one team restarts. During and just after a stoppage, v6 estimates **which team will restart**
from the players' layout, with a **logistic regression** (section 6.3) trained on the bench. The first version gained
after restarts but lost almost as much elsewhere. A diagnosis found why, and the revision fixed it: only the unlikely
team pays a cost, by the odds, and the clue waits for 2 s of dead-looking play. The right restarting team rose from
62 % to 87 % (IDSSE); the bench rose to **0.443**, better on 8 of 10.

### 4.7 v7: PathCRF, who has the ball without the ball (2026-10-30)

[models/v7-pathcrf.md](models/v7-pathcrf.md)

The biggest weakness so far: 41 % of the time there's no ball, and the clues then had little to say about the team.
**PathCRF** (Kim et al., KDD 2026) is a published model that infers possession from the **players' movement alone**
(section 6.3). We run its authors' trained weights, unchanged, through our own adapter. Its weight (0.5 on every live
frame) was chosen on the **4 matches it never saw**. There, v7 scored **0.565 against v6's 0.501**, better on 4 of 4,
and the changes of team caught inside ball gaps rose from 8 % to 55 %.

### 4.8 v8: the keeper repair (2026-10-30)

[models/v8-keeper-repair.md](models/v8-keeper-repair.md)

Jose noticed a keeper tracked in the other team's slots. v8 gives a keeper back to his own team, with a general rule
whose constants come from the Laws of the Game and reasoning (section 6.2). It never
fires on the bench (0 false repairs at every damage level), so the bench can't score it. It's kept because it fixes
the input.

### 4.9 v9: possessions start at the restart pass (2026-10-30)

[models/v9-restart-pass.md](models/v9-restart-pass.md)

Jose saw that our possessions after restarts started late. v9 "snaps" the start of each possession after a stoppage
to the restart pass, where a pass is seen. On the bench's truth the rule holds every time (698 of 698), and its
search window was chosen on the bench, leaving each match out. Moved starts within 0.2 s of the truth rose from 20 %
to 83 %; the bench rose to 0.561 against v7's 0.556, **better on 10 of 10**. The idea came from watching our output
over the video (item B). The snap is switched off in both branches since v14/v15, because the searches found the
chain better without it.

### 4.10 The clarification, and v10 (2026-11-03)

[models/v10-clean-base.md](models/v10-clean-base.md)

The organisers' clarification arrived. v10 is v9 with everything set on our match removed:
- the whistle (v2, v5);
- the set-piece window (set on three tags and the public window).

It also prepared IDSSE-only twins of every stage behind one switch, in case Metrica's data had to go, and removed our
output from the match viewer. Jose took all eight recommended options. On the bench v10 **equals v9** (0 frames
differ: the bench has no audio or tags).

### 4.11 v11: each change of team on the new team's touch (2026-11-03)

[models/v11-touch-line.md](models/v11-touch-line.md)

On the bench's truth, every change of team happens at a touch by the new team (a new-team player within 2 m, at all
2,349 IDSSE changes). Ours were ±0.6 s off. v11 moves each change to the nearest seen touch by the new team: a player
of it within 2 m while the ball's velocity changes by at least 3 m/s, within ±1 s. Moves never cross another change or
swallow a possession. Changes within 0.2 s of the truth rose from 31 % to 70 %; the bench rose to **0.571**, better on
10 of 10.

### 4.12 The lab, and v12: from hand weights to learned models (2026-11-04)

[lab.md](lab.md), [models/v12-lab-chain.md](models/v12-lab-chain.md)

Jose asked for versions built side by side by subagents, the winners kept and mixed. Four rounds that day (Part 5)
produced a chain:
- **robust:** a gradient-boosted classifier gives P(home / away / dead) for every frame from 97 features (the ball,
  the players around it, the teams, touches and passes, and the earlier models' outputs as inputs). A seen ball
  standing still for 1 s or more is treated as missing, in training and in use, so the model can't lean on IDSSE's
  frozen dead ball. Its −log P is the decoder's cost;
- the decoder;
- **short:** short possessions of the other team where it clearly touches the ball;
- **segments:** each possession checked as one object, merged into a stoppage if spurious, its team flipped if wrong;
- **timing:** each change of team moved to the frame a ranker trusts most.

**0.703 against v11's 0.573: +0.131, better on 10 of 10**, the largest gain of the project. (v11 scores 0.573 in the
lab's harness, where each match gets the setting best on the others; 0.571 on its own page.) Smart Tagging's team
hints were dropped (no bench match has tags, so no learned model could see them).

### 4.13 v13: a temporal network, and the stoppage switch (2026-11-09)

[models/v13-temporal.md](models/v13-temporal.md), [lab/temporal.md](lab/temporal.md), [lab/gate.md](lab/gate.md)

Two things from round 7:
- **The temporal start ("average"):** a small **bidirectional GRU** (section 6.8) reads robust's 97 features over a
  minute of context; its P is averaged with robust's. Its frames are only slightly better, but its P **doesn't
  flicker**, so it does part of the post stages' job. The first start since robust whose gain survived the chain:
  **0.716 against 0.703**, better on 8 of 10.
- **The stoppage switch:** SkillCorner showed v12's blind spot (section 3.6). `augment` (robust retrained with
  SkillCorner-like stoppage balls) fixed it there (0.412) but cost about 0.01 on the normal bench. Neither chain won
  in both worlds, so the switch picks one per match from **one number computed on the tracking alone** (section 6.5):
  **branch A** (the normal chain) or **branch B** (built for a ball that moves through stoppages). It passed all four
  parts of its bar, and on SkillCorner v13 scores **0.408** against v12's 0.308.

### 4.14 v14: Metrica's convention (2026-11-12)

[models/v14-search.md](models/v14-search.md), [owner-rule.md](owner-rule.md), [search-metrica.md](search-metrica.md)

With five days left, the question was the answer key's convention (section 3.7). A nested random search of 2,000
settings, judged mainly on Metrica's 3 games (DFL's 7 as a guard), agreed on two changes for branch A: weigh the GRU
0.8 (was 0.5) and turn v9's snap and v11's move off. **0.724 against 0.716**, better on 8 of 10; Metrica +0.016 (3 of
3).

**The surprise:** v14's file on our match had the **same SHA-256 as v13's**. Branch A had changed, so our match must run
**branch B**, which had never been tuned. That was learned from the hashes alone, without reading the branch.

### 4.15 v15: branch B rebuilt (2026-11-14)

[models/v15-branch-b.md](models/v15-branch-b.md), [lab/branch-b.md](lab/branch-b.md)

Round 8 rebuilt branch B:
- **a GRU for B (`gru_b`):** the temporal network trained the way `augment` was, on each match as it is and with
  stoppage balls transplanted;
- **B's start:** 0.915 × the GRU + 0.085 × `augment`, P(dead) × 0.909;
- **a nested search of 1,500 settings** of B's whole chain on the stressed benches. The winner, **trial 1015**: switch
  prices 2.82 (team) and 48.7 (dead), minimum possession 7 frames, snap and move off, post stages trained on stressed
  copies, short 0.340, segments 0.910 / 0.661, timing 0.450.

**B's bench: 0.733 against 0.698, +0.035, better on 10 of 10** (nested +0.033). The network is most of it (+0.028 to
+0.030 alone). On SkillCorner it's level (+0.0008). v15's file differed from v14's: **the first gain since v13 to reach
our file**.

### 4.16 v16: B's network trained on Metrica's games too (2026-11-15)

[models/v16-netmet.md](models/v16-netmet.md), [lab/night/network.md](lab/night/network.md)

The night of 11-14 tried nine ideas; one lever moved the bench: **what B's network trains on**. `netmet5` is four
networks of `gru_b`'s recipe (seeds 0-3), trained on **all 10 bench matches** (Metrica's 3 included), their P averaged.
- **B's bench: 0.747 against 0.733, +0.014.** Metrica's 3 **+0.040, 3 of 3** (each scored by networks that never saw
  it); DFL's 7 +0.003 (within the seed noise).
- **It fails the B bar's "7 of 10"** (6 of 10) but passes "Metrica's rule" (Metrica up on all 3, DFL not down, no
  variant down, SkillCorner level): the precedent v14 set.
- **Decided with it:** Metrica's unlicensed sample games now train the network our match runs.

### 4.17 v17: the broadcast's players near the ball (2026-11-15)

[models/v17-broadcast.md](models/v17-broadcast.md), [lab/push/broadcast.md](lab/push/broadcast.md)

Round 9 had found the largest error left: **players missing near the ball**. Putting every missing player within 15 m
of the ball back at his true position would gain +0.022, and nothing built from the tactical tracking recovers any of
it. The broadcast camera follows the ball. So track 3 of the push merged the broadcast tracking's players that the
tactical tracking lacks into what branch B reads (rule **M2**, section 6.7). Judged on a broadcast **simulated** from
data.md's statistics, with v16's B unchanged: **+0.0076, better on 8 of 10**; a pessimistic simulation −0.0005 and
+0.0007. If anything about the broadcast fails, v17 writes v16's file byte for byte.

**v17 is the upload.** Our match's file: SHA-256 `d7b159c5bc392abd0240f5d2d16a893c26694b5d7ab3cb8bb5dafa4dec69ec04`.
It differs from v16's, so the broadcast was merged (Part 6 explains how we know without looking).

---

## Part 5: the lab, round by round

[lab.md](lab.md) is the protocol and the full log. The pattern of every round: a question, candidates built in
parallel by subagents, a bar written before any score, and every result kept, including what failed.

### 5.1 Round 1 (2026-11-04): five builders on v11

| candidate | what | held out vs v11 (0.5727) | verdict |
|---|---|---|---|
| short | short possessions at clear touches | +0.018, 10/10 | kept |
| segments | each possession checked: merged or flipped | +0.015, 7/10 | kept (narrowly) |
| decoder | minimum possession and switch price chosen again | +0.007, 6/10 | discarded |
| stoppage | each change into D moved to just after the ball leaves the pitch | +0.005, 10/10 | kept |
| **learned** | a per-frame classifier P(H/A/D) in place of v1's hand weights | **+0.088, 10/10** | kept |

**The error budget** (oracles on round 1's mix): fixing every **missed possession** would gain +0.252. They are short
(median 1.8 s, 57 % under 2 s), half between two possessions of the other team, half next to a stoppage. Timing of
changes +0.048, wrong teams +0.027, spurious possessions +0.026, missed stoppages +0.022. **Lesson:** most of the room
is in short possessions we don't see.

### 5.2 Round 2: building on the mix

recall (more short possessions written in by a classifier), timing (changes moved to the best candidate frame) and
dead (a dead-ball classifier writing D inside possessions) all passed on `all`; recall and dead failed the "unseen"
check on `idsse`. **A rule made explicit:** each bench set gets the stages that pass on it.

### 5.3 Rounds 3 and 4: the learned model as the start

- **Round 3:** start from the learned model's P, decoded, then add each post stage only if it clears the bar against
  the chain so far: **0.7019**, 10 of 10.
- **The open risk:** IDSSE freezes its ball in dead time, so the learned model could learn "a still ball means dead".
- **Round 4, `robust`:** any seen ball standing still (under 0.3 m/s) for 1 s or more counts as missing, on every
  match, in training and in use. It blanks under 1 % of live frames and 44 % of IDSSE's dead frames, so IDSSE's dead
  time looks like ours (missing). The chain from robust: **0.7033 (+0.131 vs v11)**, 10 of 10. That became v12.
- **Lesson:** a learned per-frame model beat every hand-weighted clue, and the post stages that pay depend on the start.

### 5.4 Round 5 (2026-11-06): two leads, both discarded

Robust trained possession by possession, and v12 without snap and move. Both failed. **Lesson:** changes that favour
one source hurt the other, and gains near 0.002 are noise.

### 5.5 Round 6 (2026-11-07 to 11-09): ideas from other teams

A survey of the challenge repo's forks ([field.md](field.md)):
- **kept:** the 2nd half's kick-off rule (the FAQ states the answer; v12 had dropped it);
- **built:** label-free checks on our match (`bench/transfer_check.py`), for Jose to run;
- **discarded:** ball-free team movement (`motion`), and a team switch price set frame by frame;
- **not tried:** weighting external frames by how much they look like our match (it needs our match's tracking in
  training, against the rules).

**A pattern:** dropping snap and move, the frame-by-frame price and motion each helped Metrica's 3 and not IDSSE's 7.
Noted, but 3 matches chosen after the fact can't carry it. It was taken up properly in v14, with a rule written first.

### 5.6 Round 7 (from 2026-11-09): new directions

| candidate | result | verdict |
|---|---|---|
| hsmm: a decoder that prices how long possessions last | +0.0004 | discarded: **the frames limit recall, not the decoder** (43 % of the short possessions v12 misses have P favouring the other team) |
| ballfill: the missing ball filled in from the players | −0.005 | discarded: 87 % of live missing-ball frames sit in gaps of 10 s or more, where any fill is 13-17 m off |
| ensemble: robust, learned and motion averaged | +0.004 | discarded: **the chain absorbs a better start** |
| **temporal (`average`)** | **+0.013, 8/10** | **kept: v13** |
| skillcorner (a test) | v12 0.308 | the blind spot found (section 3.6) |
| carried: `players` and `augment` | fail the normal bench, fix the stressed one and SkillCorner (0.412) | not alone, but `augment` became branch B |
| **gate (the switch)** | passes all four parts | **kept: v13** |

**Lesson:** "insurance has a price on the bench". A start that reads a moving dead ball as dead costs a little where
the ball isn't stressed. The switch pays the price only where the tracking says the risk is real.

### 5.7 Round 8 (2026-11-12 to 11-14): branch B

The question came from v14's equal hash: our match runs B, and B had never been tuned. Fixed before any score: B's
bench (25 %, 50 %, fully stressed) and the **B bar** (+0.005, 7 of 10 better, no variant down more than 0.002,
SkillCorner no loss beyond 0.005). Result: trial 1015, v15 (section 4.15).

**What the search agreed on:** weight the network (rank correlation with the score +0.85), drop snap and move (79 of
the top 100 trials). The top was flat: the best 10 of 1,500 within 0.002.

### 5.8 Round 9 (2026-11-14): cleaning the tracking

[lab/cleaning.md](lab/cleaning.md)

Jose: try many kinds and amounts of cleaning (joining pieces, filling gaps, smoothing) and record what doesn't help.

| | gain |
|---|---|
| the ceiling: every missing player at his true position | +0.024, 10/10 |
| only the missing players within 15 m of the true ball | **+0.022**, 10/10 |
| perfect joins with straight-line fills | −0.004 |
| 24 real join and fill settings | −0.003 to −0.008 |

**Nothing kept.** Nine tenths of the ceiling is *where the missing players near the ball really were*, and no fill
built from the tracking recovers any of it. That pointed to the broadcast (v17).

### 5.9 The night of 2026-11-14 to 11-15

[lab/night.md](lab/night.md)

Jose, at 23:05: have subagents think of new ways to improve, commit to the most promising, and keep going until 10 am.
Nine candidates, every bar fixed before the first score. **Nothing passed.** What it established:
- **The only lever that moved the bench is what B's network trains on:** adding Metrica's 3 games (`netmet5`, later
  v16). Every seed tried gained on Metrica's games.
- **Seed noise is 0.0035 on B's bench**, and single matches move ±0.02 with the seed alone.
- **Averaging seeds doesn't raise the score** (`netens`): the chain absorbs a better P.
- **SkillCorner's real stoppage balls in training** help on SkillCorner and cost on the bench (`netsc`, `netsc2`).
- **Ghost features** (pointers to where missing players were last seen) reach about a tenth of the near-ball value.

### 5.10 Tracking errors (2026-11-15)

[lab/tracking-errors.md](lab/tracking-errors.md)

Jose saw players drawn in the wrong team's colour (the keeper too) and players placed absurdly. The bench never does
that, so errors were injected into it:

| errors | B's bench drop |
|---|---|
| 1 % of outfield frames in the wrong team | −0.018 |
| 3 % | −0.041 |
| each keeper wrong ~10 % of the time (v8's repair on) | −0.013 |
| 3 % of frames moved 10-20 m | −0.014 |

Every setting loses on all 10 matches, mostly when the wrong player is near the ball. Jose's estimate from watching:
"rare, a few seconds". At about −0.0005 per half-minute of one player, that's under 0.005, so nothing was built. It
stays a next-phase item.

### 5.11 The two-day push (2026-11-15 to 11-16)

[lab/push.md](lab/push.md)

Jose: "if there is a moment to make a difference i think is now". Three tracks:
1. **v16:** `netmet5` built, ported and checked (section 4.16).
2. **netweight:** B's network leaning harder on Metrica's games. Its finalist **W4** (Metrica's matches at loss weight
   4) passed its own bar against `netmet5`. **But under v17's merge** it was level with v16's network on B's bench and
   **failed the merge's pessimistic floor** (−0.0032 against −0.002). W4's gain and the merge's overlap. Not adopted.
3. **broadcast:** the merge (section 4.17), which became v17.

**Lesson:** two gains that each pass alone don't always add up. Check them together before combining.

---

## Part 6: v17 end to end

This part follows the program `python -m possession.v17` from the files to the CSV, and explains each idea as if new.

```
tactical tracking ──► v8 keeper repair ──► inputs (v4, v6, PathCRF, touches, passes, clues) ──► the switch's signal
                                                                                                       │
                                     ┌─────────────── below 0.1445: branch A (v14) ◄───────────────────┤
                                     │                                                                 │
                                     │       above: branch B ──► broadcast merge M2 ──► inputs again   │
                                     │                               │ (any failure: no merge)         │
                                     ▼                               ▼                                 │
                              A's start, decoder,          B's start (4 GRUs + augment) ──► decoder ──►
                              short, segments, timing      short ──► segments ──► timing
                                     │                               │
                                     └──────────────┬────────────────┘
                                                    ▼
                                      kick-off rule ──► X outside the halves ──► CSV
```

### 6.1 Reading the files

- `possession/data.py` reads the tactical tracking: per frame, 46 slots of (x, y) in metres, each slot's team, the
  ball's (x, y) or "missing", and the speeds.
- The graded halves come from the organisers' `games.py`. **Each half is processed on its own** everywhere below.
- Smart Tagging is read for one tag only: the 2nd half's kick-off.
- The broadcast files are read only if branch B runs (section 6.7).

### 6.2 The keeper repair (v8)

Before anything else, a keeper tracked in the other team's slots is given back to his own team
(`possession/repair.py`, [models/v8-keeper-repair.md](models/v8-keeper-repair.md)):
- per half and per goal, the defending team is the one whose players stand nearer that goal on average;
- a suspect: a slot of the *other* team inside that goal's 6-yard box, with no defender's slot in it, while the ball
  is more than 35 m away, for 5 s in a row;
- its piece is repaired if it spends at least 90 % of its frames inside the penalty area. The piece moves whole to a
  new slot of the defending team; positions are never changed.

The boxes are the Laws of the Game; the rest is reasoning, not fitted on our match.

### 6.3 The earlier models, now inputs

The early versions' pieces survive as **inputs** to the learned model. Each is computed on our tracking exactly as on
the bench (`v12.inputs`).

**v4's dead-ball probability, from gradient-boosted trees.**
- *What a decision tree is:* a chain of yes/no questions on the features ("is the mean player speed below 1 m/s?",
  then "is someone within 2 m of a touchline?"), ending in a leaf with an answer.
- *Gradient boosting:* many small trees added together, each one trained to fix the mistakes of the sum so far. It
  can learn combinations ("slow **and** someone on the touchline") that a weighted sum can't.
- v4's trees read 79 features of the players and give P(dead) per frame. They were trained on the bench. This is the
  **players-only** P(dead): it never sees the ball. The switch reads it (section 6.5).

**v6's restart model, a logistic regression.**
- *What it is:* the simplest learned model. A weighted sum of the features, squashed into 0-1 by an S-shaped curve.
  Its weights are learned from labelled examples.
- v6's version gives, during and just after a stoppage, the probability that each team restarts, from the players'
  layout. Trained on the bench, with every example also shown with the teams swapped and the pitch mirrored, so it
  learns shapes, not "home usually restarts".

**PathCRF.**
- *What it is:* a published model (Kim et al., KDD 2026) that reads only the 22 players' movement, five times a
  second, and says at each step either "player u has the ball" or "the ball is travelling from u to v".
- *The CRF part:* a conditional random field is a sequence model, like our decoder, that forbids impossible
  sequences (the ball can't jump from one player to another without travelling).
- *Our use:* we run its authors' trained weights unchanged (`third_party/pathcrf/`), through our adapter
  (`possession/pathcrf_adapter.py`). The adapter turns our anonymous pieces into the 11-per-team tracks it expects and
  turns its output into a per-frame P(home has the ball). It matters most while our ball is missing.
- Run in **float64** (see section 6.11).

**The touch and pass detectors (v11, v9).** A touch: a player within 2 m while the ball's velocity changes by at least
3 m/s. A pass: a player of a team plays the seen ball away, fast ([../possession/passes.py](../possession/passes.py)).
Both feed the features and the post stages.

**v6's and v7's costs.** The old hand-weighted costs per letter survive as features too.

### 6.4 The 97 features, and the still ball

`possession/lab_learned_features.py` computes **97 numbers per frame**, in groups:

| group | examples |
|---|---|
| the ball | seen or not; seconds since and until it was seen; its speed before and after, acceleration, change of velocity, turn; where it is (from home's goal, from the touchline, beyond a line) |
| players around the ball | each team's nearest player to the ball, the difference between them; how many of each team within 2 m and 5 m; who's "on the ball"; the same around where the ball was last seen and will next be seen |
| windows | means of several of those over the past and the next 1 and 3 seconds |
| the teams | each team's mean speed, spread, how far up the pitch |
| touches and passes | per team, seconds since and until its last and next touch and pass; counts in the past and next 3 s |
| earlier models | P(dead), P(home restarts), the restart clue, PathCRF's P(home) and its windows, v6's and v7's costs |

**Teams exchanged:** the features are also computed with the teams swapped (home's columns hold away's), and the
models use both, so they can't learn "home is special". The GRUs, for example, average a reading of the match as it is
and one with the teams swapped (section 6.8).

**The still-ball rule (from robust, round 4):** any run of a seen ball standing still (under 0.3 m/s) for 1 s or more
is treated as **missing**, before the features are computed. That stops the models from learning IDSSE's frozen dead
ball, which our tracking doesn't have.

### 6.5 The stoppage switch: one number per match

**The problem it solves:** branch A is the best chain where a dead ball is missing or frozen (the normal bench).
Branch B is built for a dead ball that stays tracked and moving (the stressed bench, SkillCorner). Each loses in the
other's world. So the program decides per match which world it's in, **from the tracking alone**.

**The signal** (`bench/lab_gate.signal`, used by `possession/v13.branch`):
1. Take v4's **players-only** P(dead) on every frame inside the halves with a player tracked.
2. **"The players say dead":** P(dead) ≥ 0.5, with the last 2 s of each such run left out (the restart kick).
3. **"The players say live":** P(dead) < 0.2.
4. **"Ball seen and moving":** tracked, not blanked by the still-ball rule, and moving at 0.3 m/s or more over the next
   0.2 s.
5. **Signal** = the share of "seen and moving" among the dead frames ÷ the same share among the live frames.

In words: *where the players say play is stopped, how often is the ball seen moving, compared with when they say
play is live?* Near 0 when the dead ball is missing or frozen; near 0.5 when it's carried around in stoppages.

| where | signal |
|---|---|
| the normal bench | 0.02-0.10 |
| the fully stressed bench | 0.36-0.63 |
| SkillCorner's 20 | 0.25-0.49 |

**The threshold, 0.1445.** Fixed on 2026-11-09 by a rule written before any branch was scored: on the bench, find
the stress level where branch B starts to beat branch A, and take the mean signal there. Set on all 10 matches,
before SkillCorner's signal was computed. Above it, **branch B**; otherwise **branch A**.

**Its bar** (four parts): A on every normal bench match; B on at least 9 of 10 stressed matches with +0.02;
within 0.005 of the better branch at 10, 25 and 50 % stress; and +0.02 on SkillCorner. v13's switch passed all four.
After v15 and v16 improved B, part 3 failed at 10 % and 25 %, not because B lost anywhere but because B got better and
the threshold now keeps A on some matches where B would win. **The threshold stays at 0.1445**, as fixed. Lowering
it wouldn't change our file: our match takes B already, and would at any lower threshold.

**Our match runs branch B.** We know it without reading the branch: v14 changed only branch A, and its file was
identical to v13's. Nobody overrides the switch by hand.

### 6.6 Branch A, for completeness

What a match with a normal ball gets (v14, unchanged in v17): start = 0.8 × the GRU (one network, trained on IDSSE's 7)
+ 0.2 × robust; the decoder (team switch 2, dead switch 25, minimum possession 10); snap and move off; then short
(0.5), segments (flip 0.6 / merge 0.7) and timing (0.35), refitted on that start's letters. It's the version scored
0.724 on the normal bench.

### 6.7 The broadcast merge, rule M2 (branch B only)

`possession/broadcast_merge.py`. The idea: the TV camera follows the ball, so it sees players near the ball that the
tactical tracking has lost. Add them.

**The steps:**
1. **Line up** each half by the ball. Both trackings give the ball in pitch coordinates, so the broadcast's ball is
   slid along the tactical one, and the shift where the two balls sit on top of each other wins. One shift per half
   (the broadcast cuts out half-time).
2. **Trust check:** the alignment is used only if the ball distance at the best shift is under 12 m (rms) and the
   next-best shift is at least 3× worse.
3. **Team order:** broadcast players within 1.5 m of a tactical one vote whether the broadcast's team labels are the
   same as ours or swapped.
4. **Register** each valid broadcast frame on the tactical players: the median offset from each broadcast player to
   his nearest tactical player within 5 m (if at least 3 such pairs) is subtracted. A frame whose remaining median
   distance is above 1.5 m is dropped. (Only frames the broadcast homography marks valid are used: replays and
   close-ups are skipped.)
5. **Add** a broadcast player with **no tactical teammate within 2.5 m** and **no opponent within 1.5 m**. That
   means a player the tactical tracking has lost, not a duplicate.
6. **M2's cap:** a team never has more than 11. If it would, the added players nearest a tactical teammate go first.
7. **Pieces:** each run of added frames goes into a free slot of its team (the one free longest), a new slot only
   if none is free.

Then the keeper repair and **every input B reads are computed again from the merged tracking**, and branch B runs on
it with v16's models, unchanged.

**The fallback:** if a file is missing, the alignment isn't trusted, or anything raises an error, branch B reads the
tactical tracking as delivered, and the file is **v16's, byte for byte**. v17's file differs from v16's, which tells
us the merge ran, without looking at its status.

**How it was judged** ([lab/push/broadcast.md](lab/push/broadcast.md)). The bench has no broadcast, so track 3
**simulated** one on each bench match from its clean tracking, using only data.md's statistics of our broadcast set:
- about 11.5 players visible per frame;
- pieces with a mean length of 16 s;
- 65.8 % of frames valid, the invalid ones mostly in blocks around stoppages (replays);
- an alignment residual of a couple of frames.

Three versions were simulated: realistic (`central`), pessimistic (worse in every way, plus tracked people who aren't
players), and optimistic.

| simulated broadcast | M (fixed before any score) | M2 (v17) |
|---|---|---|
| central | +0.0081, 8/10 | **+0.0076, 8/10** |
| pessimistic | −0.0018 | **−0.0005** |
| pessimistic, second draw | −0.0010 | **+0.0007** |
| optimistic | +0.0087 | +0.0091 |

- **The gain lands near the ball:** frame accuracy within 15 m of an added player +1.3 points.
- **The risk is people who aren't players:** a referee-like track given a team cost 0.0053 in the pessimistic view.
  The cap of 11 halves that. **The cap was added after that run**, so M2 is the one part of the rule chosen after
  seeing a score. Its only change is a limit a real match can't break.
- **Retraining B on merged tracking** was worse (+0.0026, 5 of 10) and would make B depend on the broadcast (−0.0098
  without one), so B's models stay v16's.
- **Not done:** looking at the broadcast video for tracked referees to choose between M and M2. Choosing a rule from
  the provided video comes close to tuning on it.

### 6.8 Branch B's start: P(home / away / dead) for every frame

B's start mixes two models.

**1. The temporal network: four bidirectional GRUs (`netmet5`).**
- *A neural network* is a large function with many adjustable numbers (weights), learned from examples by nudging the
  weights to reduce the error, many passes over the data.
- *A recurrent network* reads a sequence one step at a time and carries a memory from step to step. A **GRU** ("gated
  recurrent unit") is a common kind whose gates decide what to keep and what to forget.
- *Bidirectional:* one GRU reads forwards in time and another backwards, so each moment sees both what came before and
  what comes after. Our program labels the whole match at once, so the future is fair to use.
- *Ours:* 2 layers, 64 units each way. It reads the 97 features, standardised (minus the training matches' mean,
  divided by their spread, clipped at ±5), with missing values set to 0 plus 44 "missing" flags.
- *How it reads a match:* every 5th frame (5 per second), each of the 5 offsets as its own sequence, so every frame
  gets an answer. The half is cut into 60-second windows starting every 30 s; each frame's P is a weighted mean of
  the windows covering it, weighting a window's centre more (triangular weights).
- *Teams exchanged:* the match is also read with the teams swapped, and the two readings averaged.
- *Training:* `gru_b`'s recipe. Each training match is shown as it is **and** with SkillCorner's stoppage balls
  transplanted (the even-indexed templates). For `netmet5`, **all 10 bench matches** (Metrica's 3 included), four
  networks with random seeds 0-3, their P averaged.
- Each network's P is clipped to 0.01-0.99 and rounded to 3 decimals, then the four are averaged, clipped and rounded
  again.

**2. `augment`: gradient-boosted trees.** Robust's trees, retrained on the 7 DFL matches with stoppage balls
transplanted into their stoppages.

**The mix:** P = **0.915 × P_GRU + 0.085 × P_augment**; then **P(dead) × 0.909**, and each frame's three numbers
rescaled to sum to 1; then clipped to 0.01-0.99 and rounded to 3 decimals.

The weights (0.9152, 0.9095) come from round 8's search (trial 1015), on the stressed benches.

### 6.9 The decoder: from probabilities to letters

`possession/decoder.py`, `viterbi_min_length`. This is the heart of the method, unchanged in spirit since v3.

**The problem.** Per frame we have P(H), P(A), P(D). Taking the most likely letter on each frame would flicker
between teams dozens of times a minute, and spellF1 punishes every invented possession. We want a sequence that
follows the evidence but changes rarely.

**The costs.**
- **Per frame:** the cost of letter k is **−log P(k)**. A likely letter is cheap (−log 0.9 = 0.11), an unlikely one is
  expensive (−log 0.3 = 1.20; −log 0.01 = 4.6).
- **Per change:** a **switch price** each time the letter changes. Branch B: **2.82** between H and A; **48.7** to
  enter or leave D (so a stoppage that starts and ends costs 97.4).
- **A minimum length:** once H or A starts, it lasts at least **7 frames** (0.28 s). D has no minimum.

**Viterbi** finds the sequence of letters with the **lowest total cost**, exactly. Trying all sequences is impossible
(3 letters over 80,000 frames), but the cheapest way to be in letter j at frame t only depends on the cheapest ways to
be in each letter at frame t−1. So one pass forward, keeping for each letter the best total so far and where it came
from, and one pass back, recover the answer. The minimum length is handled by treating a letter that must last k
frames as a chain of k steps, which still takes one pass.

**A worked example** (branch B's prices, numbers rounded):

1. **A blip.** Barcelona has the ball: P(H) = 0.9. For 2 frames the evidence leans to Athletic: P(A) = 0.7,
   P(H) = 0.3.
   - Stay H through the blip: those 2 frames cost 2 × 1.20 instead of 2 × 0.36, **0.85 extra each, 1.69 in all**.
   - Follow it: 2 changes, **2 × 2.82 = 5.64**, and a 2-frame possession breaks the 7-frame minimum anyway.
   - The decoder stays H. A two-frame flicker is gone.
2. **A real change.** From some frame on, P(A) = 0.7 and stays there.
   - Staying H costs 0.85 extra per frame; changing costs 2.82 once (no change back).
   - After about 3-4 frames staying H is dearer than one change, so the decoder changes, at the frame that makes
     the total cheapest: where the evidence turns.
3. **A short pause.** Play goes quiet: P(D) = 0.8, P(H) = 0.15.
   - Calling it D saves 1.90 − 0.22 = **1.67 per frame**, but entering and leaving D costs **97.4**.
   - So a pause this clear becomes D only if it lasts about **58 frames (2.3 s)** or more. Shorter pauses stay live,
     which is what the answer key does too.

The prices and the minimum were chosen by the search (round 8), not by hand. A higher team price merges possessions; a
lower one lets real short ones through but risks flicker.

### 6.10 The post stages

Three stages edit the decoded letters. Each was trained on bench matches **out of fold**: on the letters the stages
before it produced for matches those stages never saw. For branch B, they were trained on trial 1015's own chain under
the training stress, from `netmet5`'s out-of-fold P.

| stage | what it does | model | threshold |
|---|---|---|---|
| **short** | inside a possession of team O, a 0.5-5 s stretch where team N clearly touches the ball and O touches it again later (or N's touch is the last before the ball goes dead) becomes N's | logistic regression on 10 pieces of touch evidence | **0.340** |
| **segments** | each possession is judged as one object: merged into its neighbours if likely spurious, its team flipped if likely wrong | two gradient-boosted classifiers | merge **0.661**, flip **0.910** |
| **timing** | each change of team moves to the candidate frame within 3 s (touches, the new team reaching the ball or becoming nearest) most likely to be the true change; never across another change, never swallowing a possession | gradient-boosted classifier (a ranker) | **0.450** |

Short attacks the largest error in the budget, missed short possessions. Segments cleans spurious and wrong-team
possessions. Timing moves changes onto the touch, where the truth puts them.

### 6.11 The kick-off rule, X, and the file

- **Kick-off:** the 2nd half's graded frames before its kick-off are D. The kick-off is found from Smart Tagging's
  tag (the FAQ: "those 2,934 frames are graded and all of them are D").
- **X:** every frame outside the two halves. Inside them, we never abstain.
- **Writing:** `frame,label`, 199,319 rows; the program prints the path and the SHA-256.
- **Byte for byte:**
  - every probability is clipped to 0.01-0.99 and rounded to 3 decimals, so a tiny arithmetic difference can't flip
    a label;
  - PathCRF and the GRUs run in **float64** (more digits), so another CPU's rounding shouldn't move a letter;
  - every library version is pinned (`requirements-team.txt`);
  - every trained model is committed in `possession/models/`;
  - PathCRF's weights are downloaded once from a pinned commit and refused if their SHA-256 differs.
- **Checked:** three runs (1 and 2 threads) and a fresh clone with only the organisers' data give the same file. Tested
  on macOS arm64 only so far ([reproduce.md](reproduce.md)).

### 6.12 What each piece learned from

| piece | trained or chosen on |
|---|---|
| keeper repair | the Laws of the Game and reasoning |
| v4's dead-ball trees, v6's restart model | the 10 bench matches |
| PathCRF | its authors, on DFL data (6 of IDSSE's 7) |
| the switch's signal and threshold | the bench's distributions and branch scores; checked on SkillCorner |
| the merge rule M | fixed before any score; M2's cap after the pessimistic run |
| B's four GRUs | all 10 bench matches, as they are and stressed |
| `augment` | the 7 DFL matches, stressed |
| the mix, decoder and post-stage settings | round 8's nested search on the stressed benches |
| B's post stages | the 10 bench matches under the training stress, out of fold |
| the kick-off rule | the FAQ |

Nothing in it was trained, fitted or chosen on our match's labels, its public window or its video.

---

## Part 7: every decision, and who made it

Jose made every choice of what goes into the upload. The project's habit: a page, then a question, then Jose's
answer, quoted in [decisions.md](decisions.md).

**The start (2026-10-30).**
- Work inside a clone of the organisers' repository; data stays out of git.
- Never tune on the public 5 minutes. Everything explained and documented before it's built (Jose's requirement).
- Thresholds may be placed from unlabelled distributions (looking at how a quantity is spread, never at how it
  relates to labels).
- Ideas may come from watching errors; settings may not.
- A viewer for diagnosis, with free-text notes but never a row of hand labels.
- v1 to v9 built the same day, each with Jose taking the recommended options on its page.

**The clarification (2026-11-03).** v10 with all eight recommended options; v11 with all seven. Claude stops
reading anything of our match. Jose emailed the organisers about items A and E, Metrica's samples, and v9's origin.

**The viewer flag (2026-11-04).** Jose's call, to see the versions run: `--models` brings our labels back over the
video, behind a flag. His reading: looking is not checking, as long as nothing seen there changes a model. Claude
flagged item B; Jose decided.

**The lab (2026-11-04).** Jose: "build different versions with subagents, analyse them, keep what's valuable and
discard the rest, mix the winners into a better version, and keep improving". v12 built with five recommended
decisions (robust as the start, the full chain per set, Smart Tagging off, the models committed, the port checked
frame for frame).

**v13 (2026-11-09).** "Yes, both": the temporal start, plus the switch, decided per match from the tracking alone.

**v14 (2026-11-12).** Jose took the five days left for "hard, ambitious work", asked to "brute force" the settings
on Metrica's games, and said "Yes, build v14".

**v15 (2026-11-14).** "Yes, improve B", then "Yes, adopt v15".

**Round 9 and the night (2026-11-14).** Try many kinds of cleaning, recording what doesn't help. At 23:05: "have
subagents think in other ways to improve … try that until 10 am tomorrow".

**Tracking errors (2026-11-15).** "Rare, a few seconds": nothing built.

**The push (2026-11-15).** "Yes, all three", which also decided that Metrica's 3 sample games may train the deployed
network. Then "Yes, adopt v16".

**v17 (2026-11-15, 19:30).** "Yes, with M2". Track 3 had suggested Jose check the broadcast video for tracked
referees; that was not taken up (section 6.7).

**W4 (2026-11-16, 08:45).** "Keep v17". Choosing W4 would have meant deciding after the scores that Metrica's 3
outweigh both written bars.

**The public repository (2026-11-16).** "Yes, go ahead": a new GitHub repo with the full history (the dated commits
and the decision log are the evidence that every choice came from the bench), private first so a teammate can test
the reproduction on Linux, public before the form is sent.

**What Claude does and doesn't do.** It builds, runs the bench, writes the pages and proposes. It doesn't adopt a
model, commit, push or send anything without Jose's go-ahead, and since 2026-11-03 it doesn't read our match.

---

## Part 8: what's solid, what's weak, what's next

### 8.1 Solid

- **Every rule respected since the clarification**, and what came before it disclosed.
- **Every setting chosen on other matches** by a procedure written down first, with held-out scores.
- **One change per version**, so each gain or loss has a cause; every decision dated, with its reason.
- **Reproducible:** the file regenerates byte for byte from a fresh clone; the ports reproduce the lab's letters frame
  for frame; the networks retrain bit-identical.
- **The direction is consistent across tests:** each adopted step was better on most matches, on both sources, and
  (where measured) on the unseen matches.

### 8.2 Weak

1. **The bench is not our match.** 10 matches, 7 of them in a busier convention than the answer key's. Many ideas
   were tried on the same 10, which flatters the best (the winner's curse). On SkillCorner's real tracking every
   version scores about 0.41 against 0.75 on the bench.
2. **The switch is one number.** A match near the threshold could take the less suited branch. Ours sits on the B
   side, but how far we don't know (we don't read it).
3. **Branch B's gains rest on a synthetic ball.** On SkillCorner's real stoppage balls v15's B was level with the old
   B. Much of v15's and v16's case is Metrica's 3 games.
4. **Missing players near the ball** are the largest error found (+0.022 ceiling). The merge is our answer, judged on a
   simulated broadcast. How good the real one is (referees with a team, positions, team labels near the ball), we
   can't measure without labels.
5. **Tracking errors we don't repair:** wrong teams near the ball cost about 0.018 per 1 % of outfield frames. Only
   keepers are repaired.
6. **No ball height.** A high ball sits on the grass in the data, so it passes "near" players it flies over and can
   look like a touch.
7. **Deliberate touch or deflection** is exactly the line in the answer key's convention, and we don't model it
   directly.
8. **Metrica's sample data has no licence file**, and it now trains the network our match runs.

### 8.3 What we'd do next, with more time

- automatic tracking of other matches, from the same kind of tracker, with labels;
- a model that tells deliberate touches from deflections, trained on Metrica's games;
- a repair for wrong teams near the ball (the harness `bench/lab_trackerr.py` is the test it must pass);
- branch B trained together with the broadcast merge, on real broadcast tracking of other matches;
- a network leaning toward Metrica's games (W4), checked on more Metrica-tracked matches.

### 8.4 Where we stand among other teams

[field.md](field.md), as of 2026-11-14. About five serious public entries, all with a learned per-frame model and a
Viterbi decoder. Most of their numbers can't be compared with ours, because **the label rule moves a score more than
the model does**: merging short possessions or using a "controlled for N seconds" rule makes the test easier. Ours
scores against the providers' own flags with nothing merged.

| fair comparison | ours | the closest public entries |
|---|---|---|
| Metrica's 3, held out, ~41 % ball missing, nothing merged | v14/v15 0.825 (settings found on these 3; held-out estimate ≈0.817) | TEKAB2 0.805 |
| IDSSE's 7, provider labels | 0.676 (IDSSE set) | 0.664 |
| SkillCorner's 20 | 0.408-0.409, never trained on its labels | 0.68, trained on SkillCorner with other labels |

**At or near the top of what's public, not clearly ahead.** We'll know our real score on results day (before
November 26).

---

## Part 9: glossary

| word | meaning |
|---|---|
| **A / H / D / X** | away has the ball / home has it / dead ball / no answer |
| **`all`, `idsse`** | the two bench sets: all 10 matches, or IDSSE's 7 alone. Our file uses `all` |
| **augment** | robust's trees retrained with SkillCorner-like stoppage balls; branch B's smaller part since v15 |
| **B's bench** | the mean of the 25 %, 50 % and fully stressed variants: where branch B is judged |
| **bar** | the test a candidate must pass, written before its first score |
| **bench** | other matches with labels, damaged to look like ours: where every choice is made |
| **bidirectional GRU** | a recurrent network read forwards and backwards in time (section 6.8) |
| **branch A / B** | the two chains the switch chooses between; ours runs B |
| **broadcast set** | the TV feed's tracking, homography and video (files 6-9) |
| **clip and round** | probabilities kept within 0.01-0.99 and rounded to 3 decimals, so tiny arithmetic differences can't move a label |
| **convention** | a provider's rule for when possession changes; the answer key follows Metrica's |
| **cost** | how much a frame's evidence argues against a letter: −log P since v12 |
| **damage** | what `bench/degrade.py` does to clean tracking to make it look like ours |
| **decoder (Viterbi)** | finds the cheapest sequence of letters given per-frame costs, switch prices and minimum lengths |
| **DFL** | the German league's data company; IDSSE's labels are DFL's |
| **feature** | a number describing one frame (a distance, a speed, a count) |
| **float32 / float64** | numbers with about 7 / 16 significant digits; float64 makes results steadier across machines |
| **fold** | one round of leave-one-match-out: trained on 9, tested on the 10th |
| **gradient-boosted trees** | many small decision trees added together, each fixing the sum's mistakes |
| **graded frames** | the 158,857 frames of the two halves that are scored |
| **held out** | scored on a match the model or setting never saw |
| **IDSSE** | 7 open DFL matches with per-frame possession and ball status (CC BY 4.0) |
| **IoU** | overlap of two time stretches ÷ the time either covers |
| **leave one match out** | choose on all bench matches but one, score on that one, repeat for each |
| **logistic regression** | a weighted sum of features squashed into 0-1 |
| **M, M2** | the broadcast merge rule, fixed before any score; M2 adds "a team never has more than 11" |
| **macroF1** | per-frame accuracy per letter, averaged; the tie-break |
| **minimum possession** | the shortest stretch of H or A the decoder allows (7 frames in branch B) |
| **mirroring** | showing a model each example also with the teams swapped and the pitch flipped |
| **nested search** | a search whose pick for each held-out match is made without that match |
| **netmet5** | branch B's network since v16: four GRUs trained on all 10 bench matches, averaged |
| **noise** | how much a score moves by chance: 0.0035 between seeds on B's bench, ±0.03 between matches |
| **out of fold** | a later stage trained on earlier stages' outputs for matches those stages never saw |
| **PathCRF** | a published model that infers possession from the players' movement alone |
| **piece (tracklet)** | an unbroken run of one slot following one person |
| **port** | the deployed program (`possession/vN.py`); "the port gives the lab's letters" means it reproduces the bench run frame for frame |
| **post stage** | a stage that edits decoded letters: short, segments, timing |
| **public window** | the first 5 minutes of the answer key; pipeline checks only |
| **robust** | v12's per-frame classifier, with any still ball blanked |
| **SHA-256** | a fingerprint of a file; two files with the same SHA-256 are identical |
| **signal** | the switch's number: how often the ball is seen moving where the players say dead, against where they say live |
| **SkillCorner** | 20 open A-League matches of broadcast tracking: a real test, and the source of stoppage balls |
| **slot** | one of 46 numbered columns in the tracking; not a person |
| **snap, move** | v9's (restart pass) and v11's (touch) adjustments; off in both branches since v14/v15 |
| **spellF1** | the ranking metric: possessions matched by overlap, F1 averaged over IoU 0.3-0.7 |
| **stressed** | a bench variant whose stoppages carry SkillCorner's moving dead ball |
| **switch price** | the decoder's cost for changing letter (2.82 between teams, 48.7 into or out of D in branch B) |
| **trial 1015** | round 8's winning setting for branch B |
| **unseen matches** | the 4 bench matches PathCRF never trained on (J03WR9 and Metrica's 3) |
| **W4** | track 2's network with Metrica's matches at loss weight 4; not adopted |
