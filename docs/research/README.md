# Research: what exists, and what we take from it

Written 2026-10-30 from four research reports, each made by a separate research agent that read the
sources on the web. The reports carry the detail and every source URL; this page is the synthesis
and the proposed order of work. **Nothing on this page has been built yet.**

| report | question it answers |
|---|---|
| [models-literature.md](models-literature.md) | Which published methods exist for possession, dead ball, the missing ball and segment decoding? |
| [datasets-and-code.md](datasets-and-code.md) | Which open data from *other* matches and which libraries can we build on, under which licences? |
| [practitioner-advice.md](practitioner-advice.md) | How do providers define possession, what do practitioners and competition winners recommend? |
| [extensions.md](extensions.md) | Which extra signals beyond the provided tracking are worth it on an M2 with 8 GB, and are they allowed? |

## In short

1. **We can have a real measuring stick.** IDSSE (7 Bundesliga matches, 25 fps, CC BY 4.0, already
   approved by the organisers in #16) marks every frame with the team in possession and whether the
   ball is alive. If we make its tracking look like ours and score it with the organisers' own
   scorer, we can compare versions and set weights fairly.
2. **Our decider can't be improved by searching harder, only by changing what it optimises.**
   v1 already finds the exact best answer to "evidence against the labels + a penalty per switch".
   What helps segment scores, according to the action-segmentation field (which uses the very metric
   we're ranked on): telling the decoder how long possessions and stoppages last, and refining
   boundaries.
3. **Dead ball is the biggest known gap, and there's a published fix.** Lang et al. 2022 detect
   "in play / stopped" from player positions alone at 92 % frame accuracy. We can train the same
   kind of model on IDSSE.
4. **For the 41 % of frames without a ball there's a pretrained model.** PathCRF (KDD 2026) infers
   who has the ball from player movement alone. Its weights are public, trained on IDSSE. It needs an
   adapter for our anonymous, swapping tracks.
5. **The broadcast doesn't rescue the missing ball.** Its tracking has a ball on 50.5 % of its
   frames, but we had already measured ([../data.md](../data.md)) that it covers only 12–13 % of
   the tactical gaps: both cameras lose the ball at the same moments. Its remaining use is as a
   dead-ball hint (replays and close-ups are shown mostly while play is stopped).

## What we learned, by topic

### A measuring stick on other matches

- **IDSSE** gives H/A/D almost directly: `BallStatus` 0 → D, else `BallPossession` 1 → home, 2 → away.
  About 11 hours of play.
  - Caveat: DFL's possession may follow "player in control" more tightly than the organisers'
    convention on loose balls. Treat bench numbers as a *ranking* of ideas, not a forecast of our
    score.
  - Trap: kloppy turns a missing `BallPossession` into "away"; read it raw or cross-check with
    floodlight.
- **Metrica's 3 sample games** are the same provider and format family (Game 3 is EPTS like ours),
  but have no possession field; labels must be built from their events. Their ball disappears on 94 %
  of dead frames and 0.1 % of live ones, so a model trained there learns "no ball = dead", which is
  false on ours. They have no licence file.
- **Making their tracking look like ours** ("degrading"): drop the ball on ~41 % of frames, mostly
  in crowded moments, break players into anonymous fragments spread over 46 slots, add identity
  swaps. The pattern to imitate is measured on **our own input tracking**, which is input data, not
  labels, so it's allowed.
- **Scoring:** the organisers' `evaluation/possession_metrics.py` works on any truth array, so we
  import it rather than write our own.
- **Protocol:** choose settings on 6 IDSSE matches, report on the 7th, rotate. Freeze everything
  before the final run on our match.
- **Limit:** the open datasets have no audio and no video. Clues only our match has (whistle,
  Smart Tagging, the broadcast) can't be scored there; see [../limitations.md](../limitations.md) §2.

### Decoding for the metric

- spellF1 is "segmental F1 at an IoU threshold" (Lea et al. 2017), chosen in that field precisely
  because it punishes over-segmentation.
- A plain switch penalty (our v1) makes 1-frame segments the most likely length, which is what
  produces flicker. Two fixes are reliable:
  - **Duration-aware decoding.** The cheap version gives each label a chain of k sub-states, so a
    possession must last at least k frames. The full version gives each label a duration
    distribution (a hidden semi-Markov model). The durations are measured on IDSSE.
  - **Boundary detection, then a majority vote inside each segment.** This is worth +9 to +11 F1
    points in action segmentation (ASRF); our data's gain is unknown.
- Gaussian-smoothing per-frame clue scores (σ ≈ 3 frames) before decoding is in every winning
  SoccerNet recipe.

### Dead ball and restarts

- Lang et al. 2022: a per-frame classifier (AdaBoost) on player positions, plus the same features
  shifted by ±0.4 to ±80 s, then a 3.6 s median filter. It reaches 92 % frame accuracy, and ~80 % of
  stoppages are found with their start and end within ±2 s. No code is public. We'd use order-free
  team features (mean speed, spread, players near lines, corners or spots) because we have no
  identities.
- **Restart shapes** come from the Laws of the Game:
  - kick-off: both teams in their own half;
  - goal kick: goalkeeper in the 6-yard box;
  - corner: a player at the corner arc;
  - free kick: a wall 9.15 m from a still point;
  - throw-in: a still player on the touchline.

  A shape gives the end of the stoppage and **which team restarts**. That fills the hole that Smart
  Tagging leaves every throw-in's team as "N/A".
- Whistles cut both ways: they start play as well as stop it, throw-ins, corners and goal kicks
  often have none, and crowds whistle. A 3.5–4.5 kHz multi-peak check rejects crowd whistling.

### The missing ball

- **PathCRF:** a graph model over the 22 players, run at 5 fps on 10-second windows of play. At
  each step it picks "player u has the ball" or "the ball is travelling from u to v".
  - Results: 92.4 % frame-level team accuracy, but on one clean test match.
  - Weights: 19 MB, MPL-2.0.
  - Our adapter must pack our 46 slots into 11 continuous tracks per team, find the goalkeepers,
    smooth over identity swaps, and feed only in-play stretches. Expect less than 92 % on our data.
- **PathCRF was trained on the DFL/IDSSE matches.** Testing it on IDSSE would flatter it; it has to
  be checked on matches it never saw (its own held-out test match, Metrica's samples, SkillCorner).
- **Broadcast tracking:** already measured in [../data.md](../data.md). It sees the ball on only
  12–13 % of the frames where the tactical tracking has lost it, before removing replays. It's a
  small gain, not a lever.
- **Bridging a gap from both ends:** we know where the ball was before and after a gap. If the
  nearest team differs at the two ends, place the switch where the ball could physically have
  arrived. Our decider partly does this already.

### What possession means (the convention)

- Metrica's own written event definitions are the closest public text to the ground truth's rules.
  - Possession changes only when the other team **gains control or "significantly interferes"**.
  - A touch that doesn't change the outcome doesn't count.
  - Saves, significant blocks and won challenges are **recoveries** for the defending team. That
    fits the 1.32 s won header the public truth shows (#17).
- From tracking, everyone uses the same recipe for a gain:
  - the ball inside a 0.5–1.5 m zone around the player;
  - **and** evidence of a touch: a change in the ball's direction or speed, or a kick-acceleration
    peak of ≥ 4 m/s².

  Proximity alone credits players the ball merely flies past (crosses, long balls), so a speed cap
  helps.

### Low value for us, and why

- **Ball detection in the tactical video:**
  - the ball is ~3–4 px wide in a 4.9 Mbit/s stream;
  - most gaps are crowds where it's hidden anyway;
  - it takes 5–20 h of compute on the M2;
  - the best pretrained weights come from datasets with non-commercial or unclear terms.

  Classical (no-training) detection is clean rule-wise but weak.
- **Pretrained audio models** add little over our tonal whistle detector. **Simulators** aren't
  realistic enough. **SoccerNet video models** need an NVIDIA GPU and broadcast-style training data
  under an NDA.

## Where the reports are wrong or need care

- practitioner-advice.md, pitfall 10, says an `X` inside a run splits it. **It doesn't** in our
  scorer: `X` is forward-filled before possessions are matched (`fill_abstentions`).
- datasets-and-code.md, step 5, plans our own scorer. Not needed: we import the organisers'.
- practitioner-advice.md, recommendation 9, says to check the public sample for saves and headers.
  Reading the public truth to understand **what the labels mean** is its stated purpose. Choosing a
  number (a minimum duration, a radius) so that it matches the public truth is tuning, and not
  allowed. Numbers come from IDSSE.
- extensions.md ranks "the broadcast ball fills tactical gaps" first. The agent couldn't read our
  data notes: [../data.md](../data.md) had already measured that it covers only 12–13 % of the gaps.
  It drops to a side item.
- Not yet verified in any report: the PFF World Cup data's licence, the Metrica sample data's
  licence, how DFL possession treats loose balls, whether kloppy reads our anonymous 46-slot EPTS.

## Proposed order of work (waiting for a go-ahead)

| step | what | why | effort | rule status |
|---|---|---|---|---|
| 0 | **Test bench:** IDSSE → H/A/D, degraded to look like our tracking, scored with the organisers' scorer, leave-one-match-out | Every later step needs it to be measured | 1.5–2 days | allowed (#10, #16) |
| 1 | **Duration-aware decoder** (minimum-duration chain first) | Separates "how long a possession lasts" from "how much evidence a switch needs" | 1–2 days | allowed |
| 2 | **Dead-ball classifier** from players, trained on IDSSE | Biggest known gap; each missed stoppage merges two possessions | 2–3 days | allowed |
| 3 | **Restart shapes**: end of stoppage and team that restarts | Possession starts drive IoU; fixes N/A throw-ins | 1–2 days | allowed |
| 4 | **PathCRF adapter** for ball-missing frames | The only public ball-free possession model | 3–5 days | allowed (MPL-2.0, trained on other matches) |
| 5 | **Touch-based gain rule** for visible-ball frames | Stops passes and crosses flipping possession | ~1 day | allowed |
| 6 | **Learned per-frame scores** replacing hand-set weights | Only once the bench shows the hand weights are the bottleneck | 2–3 days | allowed |
| side | **Broadcast "not on the pitch"** (replay, close-up, crowd) as a weak dead-ball hint | Cheap; judged by reasoning and label-free checks, since the bench has no broadcast | 3–4 h | allowed (provided asset) |

That's 12–19 working days for one person, against 18 days left. Steps 0–2 are the core; steps 3–6
depend on how many people work on it.

## Questions for the organisers

The rules ask us to "ask before you build on it". One public issue could cover:

1. Self-training on this match with algorithmic pseudo-labels (grey zone G1 in
   [../limitations.md](../limitations.md)).
2. Transcribing the broadcast commentary (G2).
3. Weights trained on ISSIA-CNR, whose terms say "non-commercial research only".
4. Whether a per-frame ball track from our own detector may be committed, or must be regenerated
   (it looks like a "frame-level tracking export").
5. Whether the PFF World Cup 2022 data and Metrica's sample data (no licence file) are acceptable.

Only 1, 2 and 5 matter for the plan above; 3 and 4 only if we do video ball detection.

## Resource check (2026-10-30, after v5)

Jose asked whether we use the other-match sources, and to check them. Two forked sessions answered
from their own work and the repo (no new web searches), plus one GitHub API read of SkillCorner's
repository.

### What is in use, and what isn't

| source | status | how / why not |
|---|---|---|
| **IDSSE** (7 DFL matches, CC BY 4.0) | **in use** | the bench; v3's decoder settings; v4's training |
| **Metrica sample games** (3) | **in use** | the convention check; the bench's second set; v4's training |
| **SkillCorner open data** (MIT) | available, not used | Licence clear (GitHub: MIT; repo "SkillCorner Open Data with 10 matches of broadcast tracking data", last pushed 2026-09-14, folders `aggregates`, `bodypose`, `matches`, `matches.json`). Broadcast tracking at 10 fps: settings counted in frames (v3's minimum lengths, v4's time shifts) must become seconds; off-screen players are extrapolated; dead ball is only implied. **Sound for H against A on live frames, not for D.** Best concrete use: an unseen-match check for PathCRF (step 4), since PathCRF was trained on IDSSE |
| **PFF World Cup 2022** (64 matches) | blocked | The volume play: 64 matches against our 10, which is what a learned model like v4's would want. Terms not found, so **ask the organisers first (grey zone G8)**. Its ball goes missing for camera-framing reasons rather than crowding; players carry jerseys (still to be fragmented); 29.97 fps to resample; fouls only in `events.json`; its possession convention unknown, so the convention check would be rerun on it |
| DFL Bundesliga Data Shootout (Kaggle 2022) | not bench material | video with event times; no per-frame H/A/D; licence unverified |
| SoccerTrack v1, ISSIA-CNR, SoccerNet-Tracking | not bench material | ball-detector training data only; licences unverified, non-commercial (G6), or under an NDA |
| Link & Hoernig's and Vidal-Codina's tracking sets | not obtainable | private |

**So the bench candidates stay four:** IDSSE and Metrica (in use), SkillCorner (usable now, with the
limits above), PFF (after the organisers answer).

### Points that apply to any source (from the forked sessions)

- **The whistle, Smart Tagging and the broadcast can't be tuned on any tracking source:** none has
  audio, tags or video. Their weights stay reasoned (v2, v3, v5 pages). Video sets with sound
  (SoccerNet, SoccerTrack v2) are unchecked for audio.
- **Never trust a loader's speed field:** our own file says km/h but holds m/s. Our code computes
  speed from positions (v4's features) or converts explicitly (IDSSE km/h ÷ 3.6).
- **Hand labels on other matches are allowed in any amount** ([../limitations.md](../limitations.md)
  §3), as long as the technical document says what was labelled, how, by whom and how much. That's a
  fallback for a source with good tracking but no labels.

### The reports disagree here: settle before quoting a number

- **SkillCorner's match count:** [datasets-and-code.md](datasets-and-code.md) §1.5 counted 20 match
  folders; its README and repository description say 10.
- **SoccerTrack v2's licence:** CC BY 4.0 in datasets-and-code.md, unverified in
  [extensions.md](extensions.md).
- **PFF's frame rate** (29.97 fps, read from a kloppy test fixture), its total size, and the names of
  its foul fields: unverified.
- **SkillCorner's possession labels:** whether they're made by people is unknown.

## Data search, 2026-11-10: player-only and possession-only sources

Jose's idea: with players' positions from other matches, possession could be labelled by hand (allowed on other
games in any amount, RULES.md, if the technical document says what, how, by whom and how much). Sources not
covered before, or covered without checking these points:

| source | what it has | licence / access | for us |
|---|---|---|---|
| **SoccerTrack v2** (10 university matches, ~900 min) | fixed 4K panoramic video (BePro cameras, like a tactical camera); **every player of both teams, every frame**, from BePro's provider tracking (`raw/<match>_tracker_box_data.xml` → pitch metres); **ball actions by hand with the acting team** (pass, drive, header, out, throw-in, free kick, shot, tackle, goal…); a ball track **reconstructed between those events, not measured** | **CC BY 4.0**, code MIT; Hugging Face, gated (accept conditions with an account); 145 GB in all, mostly video; the annotations are a small part | **the one that fits.** Possession can be derived from the team-tagged actions (last team to act; "Out" to the restart is dead), then checked by hand against the video. A test first (as SkillCorner), training later. The ball is reconstructed, so its "seen" pattern is not a tracker's |
| **Signality / Hammarby** (3 Allsvenskan games) | automatic optical tracking + video (not synced); the format has a per-frame `state` and ball `team`, i.e. possession per frame | no licence ("please be responsible"); hosted on an Uppsala course page; Signality now belongs to Spiideo | promising (an automatic tracker like Metrica's, possession per frame), but rights unclear: ask before using |
| **SoccerNet-GAR** (64 World Cup 2022 matches) | PFF's broadcast tracking (players + 3D ball, 30 fps, refined by hand), group-activity labels | SoccerNet NDA; licence "other"; the data is PFF's | PFF's terms forbid benchmarking use (limitations G8), so the same block |
| **Alfheim / Tromsø IL** (3 home matches, 2013) | panoramic video + ZXY sensor positions of **Tromsø's players only**, no ball | free for research, citation required | one team only: our models need both |
| **SoccerMon** | GPS of two women's teams' players, in their own sessions and matches | CC BY 4.0 | one team at a time, no ball: no |
| **TeamTrack** | short full-pitch clips with bounding boxes (soccer, basketball, handball) | unverified | clips, no possession: no |
| **Event-only sets** (StatsBomb, Wyscout CC BY 4.0, Impect 306 Bundesliga 2023/24 matches, FOOTPASS CC BY-NC) | who has the ball, event by event | as listed | no tracking, so they can neither train nor test a tracking model: priors only (possession and stoppage lengths) |
| **StatsBomb open data, now [hudl/open-data](https://github.com/hudl/open-data)** (rechecked 2026-11-12) | events with a possession id and team, lineups, and 360 freeze frames (the visible players' positions at each event moment only) for 12 of 80 competition-seasons; La Liga up to 2020/21, mostly Barcelona's matches | StatsBomb's terms: credit StatsBomb and use its logo; the user agreement in `LICENSE.pdf` (no commercial use, per the 2026-10-30 check) | no continuous tracking: snapshots at events can't feed a model that reads seconds of movement. And any Barcelona–Athletic match in it must never be used: data about our match is forbidden |
| **SkillCorner open data** (rechecked 2026-11-12) | the same 20 A-League matches we use; nothing new since 2026-09-14 | MIT | already in use: a judge, and stoppage balls for `augment` |

Searches: [SoccerTrack v2](https://atomscott.github.io/SoccerTrack-v2/) and its
[dataset card](https://huggingface.co/datasets/atomscott/soccertrack-v2); [SoccerNet-GAR](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR);
[open-football map](https://github.com/withqwerty/open-football) (Alfheim, SoccerMon, Impect); Metrica's sample-data
README (Signality / Hammarby).

### Paid data (checked 2026-11-12)

- **The rules allow it in principle** ("a commercial API" is on RULES.md's list), but "the organisers can access and
  inspect it" and "if your approach depends on something you cannot share, ask before you build on it". Paid
  tracking licences usually forbid redistribution, and our public repo must regenerate the CSV from committed
  models trained on it: ask the organisers before buying anything.
- **The professional trackers** (Second Spectrum, now Genius Sports; TRACAB; Hawk-Eye; Stats Perform; SkillCorner,
  now Sportradar) sell to clubs, leagues and media under contracts with no public per-match price; academic access is
  by request. Gradient Sports (formerly PFF) gives its 2022 World Cup data for free, but its terms forbid
  benchmarking (G8).
- **Metrica's own plans** (PlayBase Advanced €80, Nexus Pro €150 a month, free trial) list "Automatic Tracking", but no
  tracking export. Metrica's tracker on other matches would carry the same artefacts as ours, which no other source
  does; Metrica runs the challenge, so that's a question for them, asked publicly.
- **Not for Phase 1:** quotes and contracts take longer than the days left before 2026-11-17.
