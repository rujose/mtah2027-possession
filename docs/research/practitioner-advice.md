# Research: what practitioners recommend

Date: 2026-10-30

**Summary**

1. Metrica's own written convention is the closest thing to the ground truth's rules. Possession changes only when the other team **gains control or significantly interferes**. Small touches that don't change the outcome don't count. Saves and significant blocks count as a *recovery* (a gain) for the defending team.
2. Providers disagree on three points that matter to us: last *touch* versus last *control*, a third "in contest" state (FIFA), and whether a restart starts a new possession (StatsBomb yes, Opta no). Our metric splits possessions at every D, which behaves like StatsBomb's rule.
3. People who build possession from tracking data all use the same recipe. They use a small "possession zone" around each player (0.5 to 1.5 m). A gain also needs evidence of a touch, such as a change in ball direction or speed, or a speed cap. A loss happens only when someone else takes control.
4. Competition winners won with temporal context and post-processing, not with exotic models. They used smoothing plus peak or boundary picking, mirrored training data with swapped team labels, and fixes that keep team labels consistent over time.
5. Two things reliably improve segment F1. The first is boundary detection followed by a majority vote inside each segment (+9 to +11 F1 points in ASRF). The second is a decoder with explicit durations (semi-Markov or a minimum-length chain) in place of a plain switch penalty. Frame accuracy can look fine while segments are fragmented.

---

## Jargon used below

- **Possession zone (PZ):** a circle of radius R around a player. If the ball is inside, that player *may* have it.
- **Duel zone:** a circle around the ball. If players from both teams are inside, the situation counts as a duel.
- **Dead ball / in play:** the ball is out of play when it has crossed a line or the referee has stopped play (IFAB Law 9).
- **Segmental F1 / F1@k:** match predicted segments to true segments by temporal IoU (overlap divided by union), then compute F1. *k* is the IoU threshold. Our spellF1 averages this over k = 0.3 to 0.7.
- **Over-segmentation:** predicting one true segment as several pieces (our "flicker").
- **HMM / HSMM:** hidden Markov model / hidden semi-Markov model. An HSMM models how long each state lasts explicitly.
- **Viterbi:** the dynamic program that finds the single best label sequence under an HMM.

---

## 1. How providers define possession

### 1.1 Metrica Sports (most important: they built the ground truth)

Source: Metrica's public event-definitions PDF in their sample-data repo. This document defines *event* data, not the tracking-derived ground truth. Treat it as the best available statement of Metrica's thinking, and check it against the public sample.

- **Start.** A player has possession once they touch the ball. A player also has it once they consciously choose to let the ball run so they can collect it somewhere else.
- **End.** Possession ends when one of four things happens:
  - another player takes over;
  - a shot is deflected off the woodwork or another player;
  - the referee significantly deflects the ball;
  - the ball goes out of play.
- **Interference that doesn't count.** Possession is *not* lost when an opponent gets a touch but the play still reaches its target, or when the player keeps control.
- **Out of play.** The ball has crossed the outer lines, *or* the referee has stopped play.
- **RECOVERY event.** A player "significantly interferes" with the opponent's possession. A player who brings the ball back into play from a dead ball that isn't a set piece also gets a RECOVERY.
  - Its subtypes include INTERCEPTION (this covers winning a challenge as the opponent tries to receive), THEFT, BLOCKED and SAVED.
  - BLOCKED: a defender who isn't the last line of defence significantly deflects a shot. They may or may not keep the ball.
  - SAVED: the goalkeeper or last defender stops an on-target shot.
- **CHALLENGE (ground or aerial).** It is WON when the player clearly gets the touch, or when the challenge ends with their team in possession. This fits the public sample counting a 1.3 s won header as a possession: a won header that significantly redirects the ball looks like a RECOVERY under these rules.
- **Advantage.** The FORCED intervention covers a referee who plays advantage and gives the free kick later. So the dead ball starts when play is actually stopped, not at the foul.
- **BALL OUT.** The event goes to the last player who touched the ball before it left the field.
- **Smart Tagging** (Metrica's automatic tags) lists:
  - attack types: Build Ups, Counter Attacks, Direct Attacks, Organized Attacks;
  - transitions;
  - set pieces: Goals, Throw-Ins, Goal Kicks, Corner Kicks;
  - ball in final third or in the box;
  - sprints.

  Smart Tagging runs on top of Metrica's automatic tracking. No definitions or accuracy figures are published. **Unverified:** how each tag maps to frames, and how accurate it is.
- **Not found / unverified:** a Metrica help-centre page that defines possession for tracking data. The help-centre collections (Nexus, PlayBase, FAQ) don't expose one. kloppy's Metrica EPTS parser fills **no** possession or ball-state field. databallpy also says Metrica sample tracking has no possession information.

### 1.2 Other providers

| Provider | Unit | What changes possession | Loose / contested ball | Dead ball | Status |
|---|---|---|---|---|---|
| **Opta / Stats Perform** | Possession = one or more sequences by one team | Opponent **gains control** | Defensive actions don't start a sequence unless a controlled action follows. A GK save doesn't start one. | A stoppage ends a *sequence* but **not** the possession. A corner after a save stays in the same possession. | Verified (The Analyst). Older claims say Opta's possession % is pass-share, not time (third-party, 2012). |
| **StatsBomb (Hudl)** | Possession ID on every event | A new possession starts once a team **shows it has established control** | 50/50 outcomes are coded as Won / Lost / Success To Team / Success To Opposition | **A restart can start a new possession for the same team.** Example: a blocked pass goes out for a throw-in to the same team. | Verified (Open Data Events v4.0.0 spec) |
| **Wyscout (Hudl)** | Possession = a sequence of events by the same team. Possession % is pure time. | Lost duels, and missed balls or rebounds recovered quickly, **do not** break possession | (see left) | Time out of play counts for neither team | Verified (Wyscout data glossary) |
| **FIFA (EFI)** | Three states: Team A, **In contest**, Team B | Controlled possession | **In contest** = neither team has control: loose balls after duels, clearances, GK rebounds, blocks, deflections, misplaced passes | Not stated | Verified (FIFA Training Centre). How it is computed from tracking: unverified. |
| **FIFA EPTS data standard** | A transfer format (metadata + channel spec), co-developed with Barça Innovation Hub | No possession field found | — | One vendor's EPTS ball channels include an "alive" flag | Possession definition: **not found** |
| **DFL / Sportec** | Per-frame `BallPossession` and `BallStatus` attributes | kloppy reads BallPossession = 1 as home, else away, and BallStatus = 1 as alive | — | BallStatus | Field semantics come from kloppy's code. The official DFL definitions catalogue is reportedly not public. **Unverified.** |
| **Tracab** | Per-frame team possession + alive/dead | Provider flag | — | Alive/dead flag | databallpy warns the possession flag is sometimes **not switched when it should be** |
| **SkillCorner** | Per-frame `possession` {player, group} | Not documented | Not documented | No phase of play when the ball is out. kloppy marks frames with no possession group as dead. | Field verified. Rules: **unverified.** |
| **Second Spectrum** | Per-frame `lastTouch` (home/away) + `live` | kloppy maps **last touch** to the ball-owning team | Last-touch logic means every deflection flips ownership | `live` flag | kloppy code verified. Second Spectrum's own definition: **unverified.** |

### 1.3 Where they disagree and what that means for us

1. **Last touch versus last control.** Second Spectrum (via kloppy) uses last touch. Metrica, Opta, StatsBomb and Wyscout all need control or a significant interference.
   - *Implication:* a "nearest player" or "last touch" feature will flicker on deflections. We need a gain test, not a proximity test.
2. **Third "in contest" state.** FIFA has one. Metrica's ground truth doesn't: loose balls stay with the last team.
   - *Implication:* if our model has a "contested" state internally, it must output the previous owner, never X or D.
3. **Restarts.** StatsBomb starts a new possession at a restart, even for the same team. Opta doesn't. In our metric a D run always splits H or A runs, so we behave like StatsBomb automatically.
   - *Implication:* a short pause after the ball goes out costs us nothing only if we label it D. If we keep H through a throw-in, the predicted segment swallows two true ones.
4. **Saves and blocks.** Metrica's event rules make SAVED and BLOCKED a RECOVERY for the defending team. Opta says a save does *not* start a sequence.
   - *Implication (needs checking on the public sample):* after a save or a significant block, the ground truth may show a short possession for the defending team before D or the next attack. Check this before we suppress such blips.
5. **Dead-ball start.** Metrica's out-of-play rule includes "referee stopped play". With advantage, that is the later whistle.
   - *Implication:* anchor D starts at the whistle or at the line crossing, not at contact.

---

## 2. Practitioner advice: computing possession from tracking

### 2.1 Control radius and duels

- **Vidal-Codina et al. 2022** (with FIFA co-authors; used on FIFA Club World Cup and Bundesliga data). This is the most complete rule set.
  - PZ radius: **0.5 m** for one provider and **1 m** for two others. Duel-zone radius: **1 m**.
  - If both the possession and the duel conditions hold, the duel wins.
  - They advise tuning radii per provider by cross-validation on a few matches.
  - Lower-quality data (one tactical camera, broadcast) needs a **larger radius** and retuned settings.
- **databallpy** implements the same method. Defaults: `pz_radius` 1.5 m, ball speed-change threshold 5 m/s, angle-change threshold 10°.
  - The docs warn that the method fails when the ball is tracked badly.
  - The example in the docs uses different values from the defaults.
- **Morra et al. 2020** searched distance thresholds between 0.1 and 2.0 m.
  - Their possession also needs the possessor to be the closest player.
  - The ball's speed must stay under a cap **to avoid false possession during crosses**.
- **Patents on team-sport tracking** commonly use 1 m. (Secondary, low weight.)

### 2.2 Evidence of a touch (gains)

- **Vidal-Codina.**
  - A **gain** needs the ball's direction *or* speed to change while it is inside one player's PZ. If neither changes, those frames are false positives: the ball just passed by.
  - A **loss** needs two things: the ball leaves the PZ with real displacement, *and* someone else is next in control. If the same player gets the ball back with nobody else touching it, that is not a loss.
- **Link & Hoernig 2017** (Bundesliga, PLOS ONE).
  - "Kick detection" needs the player within a distance threshold *and* a local peak in smoothed ball acceleration of **at least 4 m/s²**. This filters out balls passing overhead, since there is no ball height.
  - A ball action ends when a 1 s forward prediction of the ball keeps it beyond a threshold.
  - A pass belongs to the passer's team until the receiver gains it.
  - When the ball is out of play, **neither team** has possession.
  - Results: F = 0.88 for ball actions and F = 0.83 for individual possession. Thresholds were grid-searched on 5 matches.
- **Kim et al. 2023 (KDD)** used 3 Metrica sample matches plus K League GPS data.
  - Their "touch" rule is: predicted possession probability divided by distance, above 0.5. A local maximum between 0.2 and 0.5 also counts, which catches one-touch passes.

### 2.3 Smoothing positions

- **Vidal-Codina:** Savitzky-Golay filter (a polynomial smoother) on the ball, order 2, window 7 frames (about ±0.12 s at 25 fps), before computing speeds and angles.
- **Friends of Tracking (Laurie Shaw, Metrica sample code):** velocities use Savitzky-Golay with window 7 and order 1. Raw speeds above **12 m/s** are dropped as position errors before smoothing.
- **Link & Hoernig:** a Rauch-Tung-Striebel (RTS, a two-pass Kalman) smoother on trajectories. Ball segments shorter than 1 s after a jump are discarded.

### 2.4 When the ball is missing

- **Do not treat "missing ball" as "dead ball" in our data.** Some providers (Track160, Hawk-Eye) mark dead balls by leaving out the ball (Vidal-Codina). Our tracking loses the ball on 41% of graded frames, mostly during live play, so that convention does not apply.
- **Players alone carry most of the team signal.** Kim et al. inferred the ball from player trajectories only (Set Transformer + hierarchical LSTM).
  - Team-level possession accuracy: **~86%**. Player-level: **~65%**.
  - Their post-processing pins the ball to the inferred possessor and interpolates between possessors.
- **Pitch control as a cue** (Spearman 2018, MIT Sloan; implemented in Friends of Tracking). It models who can reach a point first:
  - time-to-intercept with reaction time 0.7 s and max speed 5 m/s;
  - uncertainty σ = 0.45 s;
  - average ball speed 15 m/s.

  This can score which team is likely to collect a ball in flight from its last known position and velocity. (Our idea, built on their published parameters.)

### 2.5 Dead-ball detection

- **Set-piece triggers from player layout** (Vidal-Codina): kick-off (everyone in own half, one player near the centre), penalty, goal kick (a player in own goal area), corner (a player near a corner mark), throw-in (a player beyond the touchline).
  - Their listed future work includes a referee-whistle audio feed for more exact in/out intervals. Our audio plan matches what they wanted.
- **Whistles are only part of the story** (IFAB guidelines).
  - Referees *don't* need to whistle to stop play for a clear goal kick, corner, throw-in or goal.
  - They *do* whistle for free kicks, penalties, cards, injuries and substitutions.
  - The whistle also *starts* play at kick-offs and penalties.
  - A free kick is in play once it is kicked and clearly moves. No whistle is needed unless the referee asks for one.
- **Whistle-detector design** (Raventós et al., UPC, 5 full matches):
  - Energy in the 3.5 to 4.5 kHz band alone gave precision under 0.6 at recall 0.9. Cheering, horns and music cause false alarms.
  - Adding a tonality (spectral entropy) test plus a check for **2 or 3 spectral peaks at least 150 Hz apart** raised results to **recall 93%, precision 88%**.
  - The peak test exists because spectator whistling is a single tone, while a pea-less referee whistle has three tones. This matters in LaLiga, where crowds whistle often (our note).

### 2.6 Other useful notes

- **One opponent touch shouldn't end a possession** (Soccermatics, Sumpter's course, on Wyscout events).
  - A chain ends only after 2 "points". An opponent's touch is worth 1 point. A proper interception, a shot, a foul, an offside or the ball going out is worth 2.
  - The author calls it an approximation. It is the same idea as Metrica's "significant interference".
- **Possession lengths have a known shape** (J-League, 95 matches, arXiv 2308.05460).
  - Team possessions average about **7.2 s**, with an exponential tail (relaxation time about 10 s).
  - A single player's possession averages 2.5 s and follows a gamma distribution.
  - Use this as a sanity prior for durations. Their definition differs from Metrica's: a goalkeeper touch also ends their possession.
- **Contested time counts as "not yet recovered"** in the DFL/AWS Ball Recovery Time. Unclear moments stay on the clock until a team clearly regains the ball. This is close to "loose balls stay with the last owner".
- **From video only:** Sarkar et al. (CVPR workshop 2019) model ball-player "split" and "merge" events as a min-cost flow network to count passes and possession. This is an option if the ball can be re-detected in the broadcast video.
- **Barça's own analysts** (Fernández, Bornn, Cervone 2021) build possession value on tracking plus synchronised event data, using pitch-control features. They publish no rule for detecting possession.

### 2.7 Learned models

- **Borghesi et al. 2023** (Expert Systems with Applications; Serie A tracking).
  - A temporal convolutional network (TCN) classifies each frame as **inactive / home / away**, which is almost our label set.
  - Reported **86.2% accuracy**, about 7 points above rule-based methods.
  - Full text not read (paywall blocked), so other details are unverified.
- **Kim et al. 2023:** see 2.4.

---

## 3. Competition winners

| Competition | Task | Winner and score | What the winners say mattered |
|---|---|---|---|
| **SoccerNet Ball Action Spotting 2023** | Spot passes and drives in broadcast video (mAP@1 s) | Ruslan Baikulov (lRomul), 86.47 mAP@1 | • "Slow fusion": 2D convolutions early, then 3D. This alone gave about +15 points (65 → 80). <br>• Inputs: 15 grayscale frames at 12.5 fps. <br>• Pretraining on the larger action-spotting set: about +2. <br>• Balanced sampling of empty and event frames. <br>• Post-processing: **Gaussian filter (σ = 3 frames) + peak detection, height 0.2, minimum gap 15 frames.** <br>• The author credits the top DFL Shootout solutions as inspiration. |
| **SoccerNet BAS 2024** | 12 ball-action classes | T-DEED (Xarles et al.), 73.39 mAP@1 | • Multitask training with the big SoccerNet set (two heads). <br>• 100-frame clips. <br>• Ensemble of two clip-sampling strategies. <br>• Runner-up: a transformer with 61 s of context, 71.35 mAP@1. |
| **SoccerNet Team BAS 2025** | 12 classes + team (left/right) | "dudek" (K. Czarnogórski), 60.03 Team-mAP@1 | • **One joint head for team × action** in place of separate heads. <br>• **Horizontal mirroring with team labels swapped.** <br>• Pretraining on 500+ matches. <br>• Bayesian tuning of soft-NMS thresholds. <br>• The report says **team handling was the main differentiator**. Some entries predicted the team in possession per frame or enforced temporal consistency. <br>• 2nd place fixed team labels for SHOT and GOAL by looking back 4 s at related actions. |
| **SoccerNet Game State Reconstruction 2024 / 2025** | Player positions, roles, jersey numbers, teams on a minimap | Constructor.tech (2024), KIST-GSR (2025) | • Team assignment by jersey-colour clustering, then cluster mean x-position to get left/right. <br>• The ball is **not** part of the task. |
| **Kaggle DFL Bundesliga Data Shootout 2022** | Spot play / throw-in / challenge events in video | 1st-place write-up **not found (unverified)** | • 6th place (Fixstars): stacked neighbouring grayscale frames ("2.5D"), higher resolution, one binary loss per class, an augmentation fix (random crops were cutting out the ball), temporal NMS. Validation AP went from 0.24 to 0.75. |
| **More Than A Hack** | 2026 edition ("Football GPT Bootcamp", the *first* edition): generative-AI tactical insights on 20 Barça matches with tracking, Smart Tagging XML and video. 2027 is the second edition. | The winning team isn't named on Metrica's recap | • Metrica's recap says the winning project combined tracking data, **ball-trajectory analysis, contextual audio** and conversational AI. <br>• **No 2024 or 2025 edition was found.** |

**What transfers to us:**

- Temporal context beats per-frame cleverness.
- Post-processing (smoothing + peak or boundary picking) is part of every winning recipe.
- Mirror augmentation with swapped team labels is cheap. For us it means: check that clues are symmetric when home and away are swapped.
- Enforcing team consistency over time is where team-level scores are won.
- FOOTPASS (2025, a play-by-play spotting dataset) argues for the same thing: use tactical knowledge as a *prior* over long time spans to support noisy computer-vision outputs. That is what our clue-plus-decoder design already does.

---

## 4. Segment-level metrics: what reliably helps

### 4.1 How the metric behaves

- **Definition.** Segmental F1@k comes from Lea et al. 2017. The MS-TCN reference code works like this:
  - it ignores background segments (for us, D and X are like background);
  - it matches each predicted segment to its highest-IoU true segment of the same label;
  - each true segment can be matched only once (a "hits" array), greedily.
- **Every segment counts the same.** A missed 0.8 s possession costs as much as a missed 40 s one.
- **Frame accuracy can mislead.** It can be high while segments are fragmented (Ding et al. 2023 survey).
- **IoU arithmetic for our case** (our calculation). Shift a segment of L frames by s frames and IoU = (L − s)/(L + s).
  - A **0.8 s (20-frame)** possession survives IoU 0.7 only if the boundary error is ≤ 3.5 frames (0.14 s). At 0.5 the limit is 6.7 frames; at 0.3 it is 10.8 frames.
  - A 10 s possession tolerates about 44 frames (1.8 s) at IoU 0.7.
  - So boundary precision matters mostly for **short** possessions. Long ones mostly need to be **unbroken**.
- **Splitting is expensive** (our calculation). Split one true possession into two equal halves:
  - each half has IoU 0.5, so one half matches at k ≤ 0.5 and nothing matches at 0.6 or 0.7;
  - you also pay one or two false positives.

  A 3-frame wrong-team blip inside a long possession does this *and* adds a false segment for the other team.

### 4.2 Techniques, with reported effects

1. **Boundary detection + majority vote inside each segment (ASRF, Ishikawa et al. 2021).**
   - Method: one branch predicts per-frame boundary probability. Boundaries are local maxima above 0.5 (results barely changed from 0.1 to 0.9). Each segment takes the majority class of its frames.
   - On 50Salads: **F1@{10,25,50} rose 8.8 / 9.0 / 10.6 points** and Edit rose 10.8. Frame accuracy rose only 1.9.
   - It works on top of any segmentation output.
2. **Smoothing loss** (MS-TCN's truncated MSE on log-probabilities between neighbouring frames). It penalises over-segmentation. ASRF's Gaussian-weighted version added **+1.2 to +3.6 F1**. This is a smaller effect than refinement.
3. **Gaussian smoothing of per-frame probabilities.** The Ding survey calls it "highly effective" for segmentation metrics. The 2023 BAS winner used σ = 3 frames.
4. **Explicit duration models.**
   - In a plain HMM, how long a state lasts is implicitly **geometric** (Yu 2010). The single most likely length is 1 frame. A switch penalty alone therefore still allows short blips.
   - A hidden semi-Markov model sets durations explicitly, at cost O((M² + MD + MD²)T), where M = states and D = max duration.
   - A cheaper equivalent: expand each state into a left-to-right chain of N phases (a "Coxian" duration). This gives a minimum duration and a peaked length distribution, and plain Viterbi still runs at O(MNT).
   - Weakly supervised segmentation commonly uses Poisson length models with a minimum length (NN-Viterbi and successors, per the Ding survey).
   - A duration-constrained Viterbi (forbidding exits before a minimum dwell) is the other standard variant (speech-recognition patents and papers).
5. **Soft labels near boundaries** (linear or sigmoid label smoothing) help learned models. This matters only if we train something.

---

## 5. Pitfalls people report

1. **Proximity ≠ possession.**
   - Balls passing near players (crosses, long balls) trigger false gains (Morra; Vidal-Codina).
   - The nearest player may not be in control, for example when facing away or shielding (Link & Hoernig).
   - A player can be in possession without touching the ball, which makes gains look late (Link & Hoernig; Metrica's "conscious decision" clause).
2. **No ball height.**
   - Headers, balls flying overhead and aerial duels are ambiguous in 2D (Link & Hoernig; Morra; Vidal-Codina).
   - In our data the missing-ball frames probably cluster in exactly these moments. (Our inference: unverified.)
3. **Blocked and deflected shots get misclassified.** 70% of Vidal-Codina's shot misses were blocked shots labelled as passes. Morra's deflection F-score stayed under 0.4.
4. **Provider flags can be wrong.** Tracab-style possession flags sometimes aren't switched (databallpy). Smart Tagging is automatic and not ground-truthed, so its timing may be off.
5. **Definitions silently differ** between event-based and tracking-based possession. databallpy shows a case where they disagree for 150+ frames.
6. **Thresholds don't transfer between providers.** Vidal-Codina had to use different PZ radii per provider and advises cross-validating per provider.
7. **Whistles cut both ways.**
   - They start play as well as stop it.
   - Throw-ins, corners and goal kicks often have no whistle.
   - Crowd whistles and cheering cause false alarms (IFAB; Raventós).
8. **Optimising frame accuracy hurts segment F1.** Smoothing too hard has the opposite failure: it erases real short possessions (0.8 s is only 20 frames).
9. **Left/right versus home/away.** Team-side labels flip at half-time. SoccerNet scores team relative to the camera, which is why mirror augmentation helps there. For us: check the homography orientation per half.
10. **Abstaining mid-possession.** If the scorer treats X as breaking a run (check the challenge scorer), an X inside an H run splits it into two segments. Avoid X inside runs. (Our inference: unverified.)

---

## Top 10 recommendations for us

1. Change team only on a **gain by the other team**, meaning control or significant interference. Keep loose and contested balls with the previous owner. *(Metrica event definitions; Opta; FIFA EFI shows the contrast)*
2. Test a gain as: ball inside a ~1 to 1.5 m zone **and** a change in ball direction or speed (or a kick-acceleration peak of at least 4 m/s²). Proximity alone is not enough. *(Vidal-Codina 2022; Link & Hoernig 2017; databallpy)*
3. Cap ball speed for gains so that crosses and long balls passing players don't flip possession. *(Morra et al. 2020)*
4. Replace the geometric switch penalty with a **minimum-duration chain** (Coxian / HSMM) per team state, tuned on other matches. *(Yu 2010 HSMM review; Ding et al. 2023)*
5. Add **boundary detection then majority vote per segment** after Viterbi, as a second pass. *(ASRF, Ishikawa et al. 2021: +9 to +11 F1)*
6. Gaussian-smooth the per-frame clue scores (σ about 3 frames) before decoding. *(Ding survey; SoccerNet BAS 2023 winner)*
7. When the ball is missing, hold the last owner and lean on **player-only cues** such as team shape and pitch control at the ball's projected landing spot. Players alone reach about 86% team-level accuracy. *(Kim et al. 2023; Spearman 2018 / Friends of Tracking)*
8. Make D boundaries from **line crossing via homography + set-piece player layouts + whistles**. Use whistles mainly for foul, card and kick-off stoppages. Require the 3.5 to 4.5 kHz multi-peak signature to reject crowd whistles. *(Vidal-Codina; IFAB guidelines; Raventós et al.)*
9. Check the public sample for **saves, blocks and won headers**. Metrica's rules make them short possessions for the defending team, so don't smooth those away if the ground truth keeps them. *(Metrica event definitions; public-sample header example)*
10. Tune every threshold on **spellF1 across several other matches**, never on frame accuracy. Keep clues symmetric when home and away are swapped. *(Lea 2017 / Ding survey; Vidal-Codina cross-validation; SoccerNet 2025 mirroring)*

---

## Not found or unverified (searched, no reliable source)

- Metrica's definition of possession for tracking data or Smart Tagging. Smart Tagging accuracy.
- DFL/Sportec official `BallPossession` / `BallStatus` definitions. The catalogue reportedly isn't public.
- Second Spectrum's and SkillCorner's own written possession rules. Only parser behaviour (kloppy) was checked.
- FIFA EPTS: no possession field found in the format.
- The DFL Shootout 1st-place write-up.
- More Than A Hack 2026 winner names, and any 2024 or 2025 edition.
- StatsBomb Conference or Barça Sports Analytics Summit talks specifically on possession from tracking.
- Hudl StatsBomb's phases-of-play post doesn't define where a possession starts or ends.
- Borghesi 2023 full method (only the abstract-level result was seen).

---

## Sources

**Provider definitions**
- Metrica event definitions (PDF): https://github.com/metrica-sports/sample-data/blob/master/documentation/events-definitions.pdf
- Metrica sample data README: https://github.com/metrica-sports/sample-data
- Metrica Smart Tagging blog: https://www.metrica-sports.com/blog/smart-tagging-football-analysis
- Metrica Smart Tagging 3.0: https://www.metrica-sports.com/blog/smart-tagging-3-0-football-analysis
- Metrica help centre: https://www.metrica-sports.com/help-center/help
- Metrica Nexus 2.0 walkthrough: https://www.metrica-sports.com/blog/metrica-nexus-2.0-tutorial-the-complete-walkthrough
- Metrica / MyGamePlan tagging: https://www.metrica-sports.com/blog/automate-your-tagging-workflow-with-mygameplan-and-metrica-nexus
- Opta possessions and sequences: https://theanalyst.com/articles/possessions-and-sequences-in-football
- Opta definitions: https://theanalyst.com/articles/opta-football-stats-definitions
- Opta possession as pass share (NBC, 2012): https://www.nbcsports.com/soccer/news/opta-possession-stat-not-based-on-time-on-number-of-passes
- StatsBomb Open Data Events v4.0.0 (PDF): https://github.com/statsbomb/open-data/blob/master/doc/Open%20Data%20Events%20v4.0.0.pdf
- Wyscout glossary, ball possession: https://dataglossary.wyscout.com/ball_possession/
- FIFA EFI possession control: https://www.fifatrainingcentre.com/en/fwc2022/efi-metrics/efi-metric--possession-control.php
- FIFA "In contest": https://www.fifatrainingcentre.com/en/resources-tools/football-language/in-contest/in-contest.php
- FIFA EPTS: https://inside.fifa.com/innovation/standards/epts/epts-1
- FIFA EPTS standard data format: https://inside.fifa.com/innovation/standards/epts/research-development-epts-standard-data-format
- SkillCorner open data: https://github.com/SkillCorner/opendata
- SkillCorner player possession article: https://www.skillcorner.com/articles/understanding-player-possession-with-skillcorner-game-intelligence
- kloppy Second Spectrum parser: https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/secondspectrum.py
- kloppy SkillCorner parser: https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/skillcorner.py
- kloppy Sportec parser: https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/sportec/deserializer.py
- kloppy Metrica EPTS parser: https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/metrica_epts/deserializer.py
- DFL dataset paper (Bassek et al. 2025): https://www.nature.com/articles/s41597-025-04505-y
- DFL definitions catalogue (forum, not public): https://www.bigsoccer.com/threads/does-anyone-have-the-definitionskatalog-offizielle-spieldaten.2080544/
- AWS Bundesliga Ball Recovery Time: https://aws.amazon.com/blogs/machine-learning/bundesliga-match-fact-ball-recovery-time-quantifying-teams-success-in-pressing-opponents-on-aws
- Hudl StatsBomb phases of play: https://www.hudl.com/blog/phases-of-play-hudl-statsbomb
- IFAB Law 9: https://www.theifab.com/laws/latest/the-ball-in-and-out-of-play/
- IFAB Law 13 free kicks: https://www.theifab.com/laws/latest/free-kicks/
- IFAB guidelines on the whistle: https://www.theifab.com/laws/latest/guidelines/body-language-communication-and-whistle/

**Possession from tracking**
- Vidal-Codina et al. 2022: https://arxiv.org/abs/2202.00804
- Link & Hoernig 2017: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953
- databallpy possession docs: https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html
- databallpy v0.5.0 docs: https://databallpy.readthedocs.io/en/v0.5.0/features/team_and_player_possession.html
- Morra et al. 2020: https://arxiv.org/abs/2004.04147
- Kim et al. 2023 (KDD): https://arxiv.org/abs/2306.08206
- Borghesi et al. 2023 (TCN possession): https://doi.org/10.1016/j.eswa.2023.119780
- Borghesi thesis record: https://webthesis.biblio.polito.it/18073
- Friends of Tracking code (Laurie Shaw): https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
- FoT velocities: https://raw.githubusercontent.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/master/Metrica_Velocities.py
- FoT pitch control: https://raw.githubusercontent.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/master/Metrica_PitchControl.py
- Spearman pitch control explainer: https://archive.trainingground.guru/articles/william-spearman-how-liverpool-create-pitch-control
- Fernández, Bornn, Cervone 2021 (Barça EPV): https://arxiv.org/abs/2011.09426
- Possession statistics (J-League, arXiv 2308.05460): https://arxiv.org/abs/2308.05460
- Soccermatics possession chains: https://soccermatics.readthedocs.io/en/latest/gallery/lesson4/plot_PossesionChain.html
- Sarkar et al. 2019 (possession from video): https://openaccess.thecvf.com/content_CVPRW_2019/html/CVSports/Sarkar_Generation_of_Ball_Possession_Statistics_in_Soccer_Using_Minimum-Cost_Flow_CVPRW_2019_paper.html
- Whistle detection (Raventós et al., UPC): https://gtav.upc.edu/en/publications/papers/2014/the-importance-of-audio-descriptors-in-automatic-soccer-highlights-generation
- Whistle detection (Amrita): https://www.amrita.edu/publication/automated-referee-whistle-sound-detection-for-extraction-of-highlights-from-sports-video/

**Competitions**
- SoccerNet 2025 results: https://arxiv.org/abs/2508.19182
- SoccerNet 2024 results: https://arxiv.org/abs/2409.10587
- SoccerNet 2023 results: https://arxiv.org/abs/2309.06006
- BAS 2023 winner code: https://github.com/lRomul/ball-action-spotting
- T-DEED: https://github.com/arturxe2/T-DEED
- DFL Shootout 6th place (Fixstars slides): https://speakerdeck.com/player/37475620112b41ceb78d5de04503e229
- DFL Shootout announcement: https://www.dfl.de/en/news/dfl-launches-bundesliga-data-shootout-to-unearth-new-solutions-for-automated-data-recording/
- More Than A Hack 2027 post: https://www.metrica-sports.com/blog/more-than-a-hack-2027-build-something-real-with-fc-barcelona-match-data
- More Than A Hack 2026 recap: https://www.metrica-sports.com/blog/inside-the-bar%C3%A7a-innovation-hack-ai-football-data-and-tactical-intelligence-at-talent-arena-2026
- More Than A Hack 2026 announcement: https://www.metrica-sports.com/blog/turning-football-data-into-tactical-intelligence
- Talent Arena challenge page: https://talentarena.tech/more-than-a-hack-barca-innovation-hub/
- FOOTPASS dataset: https://arxiv.org/abs/2511.16183

**Segment metrics**
- Lea et al. 2017 (segmental F1): https://arxiv.org/abs/1611.05267
- MS-TCN: https://arxiv.org/abs/1903.01945
- MS-TCN eval code: https://raw.githubusercontent.com/yabufarha/ms-tcn/master/eval.py
- ASRF (Ishikawa et al. 2021): https://arxiv.org/abs/2007.06866
- Ding, Sener, Yao survey (TPAMI 2023): https://arxiv.org/abs/2210.10352
- Yu 2010 HSMM review: https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf
- Duration-constrained Viterbi in speech (ICSLP 1994): https://www.isca-archive.org/icslp_1994/katoh94_icslp.html
