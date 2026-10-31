# The whole thing, explained from zero

Written 2026-10-30 for someone seeing this for the first time. Part 1 explains the words; part 2
explains what has been done, with the numbers needed to judge it.

---

## Part 1: the words

### The task

- **Frame:** one still picture of the video. There are 25 per second, so 199,319 for the whole video.
- **Possession:** which team has the ball. A **possession** (as the metric counts it) is one
  unbroken stretch of the same team having the ball.
- **Dead ball (D):** play is stopped: the ball is out, a foul was whistled, a goal, waiting for a
  restart.
- **H / A / D / X:** the four answers per frame:
  - **H:** Barcelona (home) has the ball;
  - **A:** Athletic (away) has it;
  - **D:** dead ball;
  - **X:** "no answer".
- **Graded frames:** the frames that count, the two halves (~106 minutes). Before kick-off,
  half-time and after the final whistle are ignored.
- **Stoppage:** a stretch of dead ball.
- **Restart:** the moment play resumes after a stoppage (throw-in, corner, goal kick, free kick,
  kick-off), and the team that takes it.

### The data

- **Tracking:** for every frame, where each player and the ball are on the pitch (x and y; no height).
  Produced automatically by Metrica's software and not corrected, so it has errors.
- **Slot:** one of 46 numbered columns in the tracking. A slot is not a person. The software fills a
  slot while it follows someone, and when it loses them and finds them again they usually land in
  another slot. Only the slot's team is fixed (1–24 Barcelona, 25–46 Athletic).
- **Ball missing:** on 41 % of graded frames the tracking has no ball. Mostly in crowds, and
  mostly (83 %) while play is stopped.
- **Smart Tagging:** Metrica's automatic tags of what's happening ("Barcelona build-up", "corner").
  Not checked by people, so it's a hint, not an answer. Every throw-in has team "N/A".
- **Homography:** for each frame, a small table of numbers (3×3) that converts a position on the
  pitch into a position in the video picture. It's how the viewer draws the dots on the video.
- **Tactical video / broadcast video:** the wide fixed camera that shows the whole pitch, which is
  the one we label; and the TV feed (zooms, replays), which is out of sync with it.
- **Ground truth:** the correct answers. The organisers have it for the whole match and keep it
  secret.
- **Public window:** the first 5 minutes of ground truth, which they gave us. **Only** for checking
  our pipeline isn't broken, never for choosing anything.

### The score

- **Precision:** of the possessions we report, how many are real.
- **Recall:** of the real possessions, how many we found.
- **F1:** one number combining the two (their harmonic mean). 1 is perfect, 0 is useless.
- **IoU ("intersection over union"):** how well two time stretches overlap: the shared time
  divided by the total time either covers. 1 means identical, 0 means no overlap.
- **spellF1, the ranking metric:**
  - every real possession is paired with at most one of ours (same team, best overlap first);
  - a pair counts as found if its IoU passes a threshold;
  - the F1 of that pairing is averaged over thresholds 0.3, 0.4, 0.5, 0.6 and 0.7.

  Inventing extra short possessions ("flicker") or merging two real ones both cost points.
- **macroF1:** the frame-by-frame score (right letter on each frame), averaged over H, A and D. It's
  only used to break ties.
- **Floor:** the organisers' simple baseline (nearest player to the ball, then smoothing).
  spellF1 0.2433 on the full match. The thing to beat.
- **Noise band:** differences smaller than this are luck, not skill. About 0.04 spellF1 on the
  full match.

### How our models work

- **Clue:** a piece of evidence computed from the data, for example "a Barcelona player is within
  2 m of a slow ball", "everyone is standing still", "a referee whistle".
- **Cost:** how much a clue argues against an answer on one frame. Our costs mean "how surprised we
  would be" if that answer were true.
- **Decoder (Viterbi):** the part that turns per-frame costs into one answer per frame. It finds the
  sequence of H/A/D with the lowest total cost, exactly, by trying all possibilities efficiently.
- **Switch price:** a fixed cost charged every time the answer changes (H to A, or into or out of
  D). A high price merges possessions; a low one lets real short ones through, but risks flicker.
- **Minimum length:** a rule that once an answer starts it lasts at least N frames. It stops flicker
  without needing a high switch price.
- **Probability P(dead):** a number from 0 to 1 that a model gives each frame, "how likely play is
  stopped right now".
- **Feature:** a number describing the situation on one frame, such as mean player speed, or how
  many players are on a touchline. Models take features in and give probabilities out.
- **Logistic regression:** a simple model. A weighted sum of the features, squashed into 0–1.
- **Gradient-boosted trees:** a stronger model. Many small yes/no decision trees added together; it
  can learn combinations ("slow **and** someone on the touchline").
- **Training:** letting a model learn its internal numbers from labelled examples.
- **Weight:** how much a clue counts against the others; also a setting we choose.
- **Mirroring:** showing the model every example also with the teams swapped and the pitch flipped,
  so it can't learn "home usually restarts", only the shapes.
- **PathCRF:** a published model (KDD 2026) that guesses who has the ball from the players'
  movement alone, without the ball. We run its authors' trained version, which learned on other
  matches.
- **Clip and rounding:** probabilities are kept between 0.01 and 0.99 and rounded to 3 decimals,
  so one frame can't dominate, and tiny computer differences can't change a label.

### How we test without labels for our match

- **Bench:** our test bed. Other matches whose frame-by-frame answers we have, made to look like ours.
- **IDSSE:** 7 open Bundesliga matches (German league). Each frame says which team has the ball and
  whether it's dead. Licence CC BY 4.0. The organisers approved its use for another team.
- **Metrica sample games:** 3 anonymous matches Metrica published, with events tagged by Metrica.
  We built H/A/D from those events.
- **Damage (degrading):** we break the other matches' clean tracking the way ours is broken:
  - drop the ball like ours;
  - cut players into anonymous pieces over 46 slots;
  - blank some speeds.

  Every fault is measured on **our** tracking, never on its answers.
- **Damage level:** the share of frames without a ball: 30 %, 41 % (ours), 55 %. Several levels
  catch a method that only works at one.
- **Calibrate:** scale the "lose the ball" chance until a level's share is reached.
- **Out of sample / held out:** scored on a match the model or setting never saw.
- **Leave-one-match-out:** choose settings on all bench matches but one, score on that one, repeat
  for each. The average is the **held-out score**, the honest estimate.
- **Overfitting:** a model or setting that fits its examples too closely and does worse on new
  ones. Held-out scores expose it.
- **Tie rule:** when several settings score within 0.01 of the best, we take the one that reports
  the fewest possessions. Inside the noise, the cautious choice.
- **Family margin:** we keep the simpler model (logistic regression) unless the trees win by more
  than 0.01.
- **Convention:** each data provider's rules for when possession changes (a touch, or control?).
  The organisers' answers follow Metrica's; the bench's main set follows DFL's.
- **Raw / merged labels:** the bench's answers as given, and with possessions under 0.5 s merged
  away. If a comparison flips between the two, it's too close to trust.

### Rules words

- **Hand labels:** a person writing down answers. Forbidden on our match for any use, even to check
  ourselves. Allowed, in any amount, on other matches.
- **Human-made data about the match:** event data, match reports, published stats. Forbidden. Only
  automatic data and our algorithms may touch our match.
- **Byte-identical:** re-running our code must produce exactly the same file, down to the last
  character.
- **Grey zone:** something the rules don't clearly allow or forbid. We ask the organisers before
  building on it.

---

## Part 2: what has been done

### The approach in one paragraph

The program computes **clues** from the tracking, audio and tags, turns them into **costs** per
frame, and lets a **decoder** pick the cheapest sequence of H/A/D. Every setting is either reasoned
from football (and written down with its reason) or **chosen on the bench**, on other matches. Our
match's answers are never seen, and the public 5 minutes are only a check that nothing is broken.
Each change is a new numbered version with its page written **before** coding, and each version
changes one thing.

### The versions

| version | what it changed | how its settings were chosen |
|---|---|---|
| v1 | first model: 3 clues (who's on the ball; is play dead; Smart Tagging) + decoder with a high switch price | reasoning |
| v2 | + the referee's whistle from the audio, counted only if players slow down after it | reasoning (the bench has no audio) |
| **bench** | 7 IDSSE matches, later + 3 Metrica games, damaged to look like ours | measured on our tracking's faults |
| v3 | decoder: low switch price (0.5, was 15) + minimum possession of 0.8 s | bench, leave-one-match-out |
| convention check | do Metrica's possessions change hands as often as DFL's? | — |
| v4 | dead-ball probability from a trees model on 79 player features, replacing two hand rules | trained and chosen on the bench |
| v5 | the whistle's weight restored (4.30), after Jose watched two whistles v4 missed | reasoning (formula), not tuned on those two |
| v6 | which team restarts, from a model of the players' layout at stoppages | trained and chosen on the bench |
| v7 | + PathCRF: who has the ball, from the players' movement, especially while the ball is missing | published model; its weight chosen on the 4 bench matches it never saw |
| v8 | + keeper repair: a keeper tracked as the other team is given back to its own (Jose spotted it) | a general rule from the Laws and reasoning; never fires on the bench, so v8 = v7 there |
| v9 | + each possession after a stoppage starts at the restart pass, when a pass is seen (Jose spotted ours were late) | the search window chosen on the bench, leaving each match out |

### The numbers to judge it by

**On the bench** (spellF1, held out, at our damage level; higher is better):

| | IDSSE (7) | Metrica (3) | all 10 |
|---|---|---|---|
| floor (organisers' baseline) | 0.217 | 0.364 | — |
| v1 | 0.224 | 0.384 | 0.272 |
| v3 | 0.369 | 0.512 | 0.412 |
| v4 = v5 on the bench | 0.398 | 0.516 | 0.433 |
| v6 (revised) | 0.413 | 0.517 | 0.443 |
| v7 (on the 4 matches PathCRF never saw) | J03WR9 0.569 | 0.564 | 0.565 vs v6's 0.501 there |

- **The big steps:**
  - **v3 (the decoder):** +0.14. v1 merged possessions: it found ~150 per match where there are
    ~440.
  - **v4 (dead ball):** +0.02. It finds 81 % of stoppages, where v3 found 48 %.
- **v6 (revised): +0.010, better on 8 of 10 matches.** It gets the restarting team right far more
  often (62 % → 87 % on IDSSE). Its first version gained after restarts but lost almost as much
  elsewhere; a diagnosis found why, and the revision fixed it ([models/v6-restart-team.md](models/v6-restart-team.md)).
- **Bench scores are not a forecast of our real score.** They rank ideas. The bench's damage and its
  possession convention only approximate ours.

**On our match** (no answers; what the models report):

| | v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|---|
| possessions | 173 | 177 | 429 | 428 | 426 | 439 |
| (v7) | | | | | | 347 |
| (v8) | | | | | | 351 |
| (v9) | | | | | | 351 |
| dead ball, share of time | 35.0 % | 36.3 % | 39.9 % | 46.2 % | 46.3 % | 46.5 % |

For scale: Metrica's sample games, by their own events, have 304–358 possessions per 90 minutes.
v5's 426 over 106 minutes is ~360 per 90.

**Public 5 minutes** (pipeline check only, 17 real possessions, too few to rank): every version
from v4 on scores 0.44 there; v1 and v2 0.43; v3 0.34. It only proves the files line up and the
pipeline works.

### Checks that the work is sound

- **Reading the data:**
  - the IDSSE frame count matches its paper exactly (1,002,644);
  - where a player is within 1 m of a slow ball, his team is the labelled team 95–99 % of the time,
    on both bench sets.
- **The decoder:** the new minimum-length decoder was checked against trying every possible answer
  on 400 small problems. With no minimum it reproduces the old decoder byte for byte.
- **Reproducibility:** every version's output file is byte-identical on re-run, and so are the bench
  results.
- **Damage faithfulness:** how often and how long the ball goes missing matches ours, with one known
  gap. On the bench a missing ball is a weaker sign of a stoppage than on ours (66 % against 83 %
  missing while players stand still).
- **Convention:**
  - Metrica's events change hands ~21 % less often than DFL's;
  - on Metrica's own labels, v3's settings still win and are picked again;
  - so it doesn't change the answer.

### What's solid, and what's weak

**Solid:**
- every rule respected and written down;
- settings chosen on other matches by an honest procedure;
- every number reproducible;
- one change per version, so each gain or loss has a cause.

**Weak:**
- **10 matches is a small base**, especially for the learned models. The 64 PFF World Cup matches
  would help, but their licence must be cleared with the organisers first.
- **The bench imitates our tracking but isn't it.** Two conventions (DFL, Metrica) approximate the
  truth's.
- **The whistle and Smart Tagging can't be tested on the bench** (no audio, no tags elsewhere).
  They rest on reasoning.
- **We will not know our real score until results day.**

### Where it stands and what's next

- **Best-supported version: v7**, with the largest gain since v3 on the matches PathCRF never saw
  (+0.064, 4 of 4). To watch: it reports fewer possessions on our match (347), and PathCRF is a
  little less consistent with our tracking than with the bench's.
- **v8** = v7 on tracking where Barça's keeper, tracked as Athletic for the last 230 s, is given
  back to Barça. The bench can't score it (the rule never fires there); it's kept because it fixes
  the input.
- **Current version: v9** = v8 with each possession after a stoppage starting at the restart pass,
  where a pass is seen. Held-out on the bench 0.561 against v7's 0.556, better on 10 of 10.
- **Next** (ideas.md):
  - a dead-ball revision from Jose's answers;
  - step 5, counting a gain of possession only when the ball is actually touched.
- **Outside the code:**
  - register 3–5 team members by **Nov 10**;
  - decide where our public repository lives;
  - write the technical document (these pages are its raw material);
  - submit by **Nov 17, 23:59 CET**.
