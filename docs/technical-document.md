# <team name>: ball possession from automatic tracking

More than a Hack 2027, Phase 1. Technical document for our submission `<team name>.csv`.

- **The file** comes from model **v17**. The repository regenerates it byte for byte from the organisers' data
  (SHA-256 `d7b159c5bc392abd0240f5d2d16a893c26694b5d7ab3cb8bb5dafa4dec69ec04`, section 7).
- **Repository:** https://github.com/rujose/mtah2027-possession. Every version, setting, experiment and decision has its own page under
  `docs/`; this document summarises them. The long explanation is `docs/explained-in-depth.md`.

## 1. The approach

**The problem.** The provided tracking is automatic and uncorrected:
- **The ball is missing on 41 % of graded frames:** 83 % while the players stand still, 30 % in open play.
- **Players are cut into about 1,200 anonymous pieces** over 46 slots; on average 20.5 of the 22 are visible.

There are no labels for the match, and the public 5 minutes may only check the pipeline. So every model and every
setting is fitted and chosen on **other matches**.

**The test bench** (`docs/bench.md`):
- **Matches:** 10 with per-frame H/A/D labels, 7 from DFL (IDSSE) and Metrica's 3 sample games.
- **Damage:** each match is damaged to look like ours: players anonymised and cut into pieces of our lengths, the
  ball lost at our rates (by crowding, nearness to a line and whether play is stopping), 8 % of speeds blanked.
  The rates were measured on our tracking alone, without labels or video.
- **Score:** the organisers' scorer, each match **held out** from everything trained or chosen for it.
- **Harder variants:** 25 %, 50 % or all of each match's stoppages get a tracked, moving ball taken from
  SkillCorner's real tracking. SkillCorner's 20 open matches are a test on real tracking.

### The pipeline (v17)

| step | what it does | trained or chosen on |
|---|---|---|
| **inputs** | • a repair for keepers tracked as the other team (v8); <br>• a dead-ball probability from the players (gradient-boosted trees, v4); <br>• the restarting team, from the players' layout (logistic regression, v6); <br>• **PathCRF** (KDD 2026, its authors' weights), which infers who has the ball from player movement alone (v7). <br>With touches, passes and distances to the ball: 97 features per frame | the bench (v4, v6); PathCRF's authors (DFL data) |
| **stoppage switch** | One number per match: on frames the players-only dead-ball model calls dead, the share where the ball is seen and moving, divided by that share on frames it calls live. Above 0.1445 the match runs **branch B**, built for a ball that moves through stoppages; otherwise branch A. **Our match runs B** | threshold fixed on the bench, checked on SkillCorner |
| **broadcast merge** (B only) | The broadcast tracking is lined up with the tactical tracking by the ball (one shift per half) and registered on the tactical players frame by frame. A broadcast player with no tactical teammate within 2.5 m and no opponent within 1.5 m is added in a free slot of his team, at most 11 a team. If a file is missing, the alignment isn't trusted or anything fails, the file is v16's, byte for byte | rule fixed before any score, except the cap of 11 (section 5); judged on a simulated broadcast |
| **start** | P(home / away / dead) per frame: **0.915 × the mean of four bidirectional GRUs** (2 layers, 64 units each way, every 5th frame, 60 s windows, averaged with the teams exchanged) **+ 0.085 × gradient-boosted trees**; P(dead) × 0.909 | GRUs: all 10 bench matches, plain and stressed (v16); trees: the 7 DFL matches, stressed |
| **decoder** | Viterbi over H/A/D with −log P as the cost, a price of 2.82 for each change of team and 48.7 for entering or leaving dead ball, and a minimum possession of 7 frames (0.28 s) | a nested search of 1,500 settings on the stressed bench (v15) |
| **post stages** | • **short:** a 0.5-5 s possession for the other team where a logistic regression on touch evidence is at least 34 % sure; <br>• **segments:** a possession is merged into the stoppage if likely spurious (≥ 0.661) or its team flipped if likely wrong (≥ 0.910); <br>• **timing:** each change of team moves to the frame within 3 s that a ranker trusts most, if at least 45 % sure | the bench, out of fold |
| **kick-off** | The second half's graded frames before its kick-off are D (the FAQ says so); the kick-off comes from Smart Tagging's tag | the FAQ |

Every graded frame gets H, A or D; only frames outside the two halves are X.

### Missing and noisy tracking

- **The missing ball:** the per-frame model reads the players as well as the ball, PathCRF infers the ball's path
  from player movement alone, and the decoder keeps the last team until the other team's evidence pays the price
  of a change.
- **A ball that moves through stoppages:** the switch sends our match to branch B, trained with real moving
  stoppage balls.
- **Players missing near the ball:** the broadcast merge.
- **Keepers tracked as the other team:** the v8 repair.
- **The rest** (pieces, swaps, blanked speeds): the learned models are trained, and every setting chosen, on the
  bench damaged to our rates. Cleaning the tracking first was tried and lost (section 2).

### Why this, rather than something else

The pipeline grew one change at a time over 17 versions, each kept only if it passed a bar written before its
first score.

| step | bench (held out) | why it was kept |
|---|---|---|
| organisers' baseline (nearest player, smoothed) | 0.217 (DFL) / 0.364 (Metrica) | the floor |
| v1-v11: hand-weighted clues and a decoder, with a learned dead ball (v4), the restart team (v6) and PathCRF (v7) | 0.573 | each step better on the bench; then it levelled off |
| **v12: a learned per-frame model and post stages** | **0.703** (10/10 matches better) | the largest single gain |
| v13-v14: a temporal start (GRU), the stoppage switch, branch A tuned to Metrica's convention | 0.716 → 0.724 | on SkillCorner, v12 called 81 % of real stoppages live when the ball stays tracked: 0.308, against 0.408 with the switch |
| **v15-v17: branch B rebuilt** (our match's branch) | B's bench (25 %, 50 %, fully stressed): 0.698 → 0.733 → 0.747 → **0.754** | v15: a GRU trained under stress; v16: also trained on Metrica's games (+0.040 on them); v17: the broadcast's players near the ball |

For scale: the bench moves about 0.0035 between seeds and ±0.03 from match to match. On SkillCorner's real tracking
the same models score about 0.41, so the bench ranks ideas; it doesn't forecast our score.

## 2. Assets used and dropped

| asset | used? | how, or why not |
|---|---|---|
| tactical tracking | **yes** | the main input |
| broadcast tracking | **yes, since v17** | merged into what branch B reads; from its homography file, only the per-frame validity flag |
| Smart Tagging | **only the 2nd half's kick-off tag** | Its team hints fed v1-v10 and were **dropped in v12**: no bench match has tags, so their worth can't be measured |
| audio | **dropped (v10)** | A whistle detector fed v2-v9, but its thresholds had been set on this match's audio; removed once the organisers clarified that tuning must use external data |
| tactical and broadcast video | **not read by the program** | watched by team members to understand the data (section 6) |
| tactical homography | **not read** | |
| public 5 minutes of ground truth | **pipeline check only** | never used to choose a version or a setting |
| IDSSE, Metrica, SkillCorner, PathCRF | **yes** | section 3 |

**Tried and dropped**, each judged on the bench against a bar written before its first score (`docs/lab.md`,
`docs/lab/night.md`, `docs/lab/push.md`):
- **Decoding and starts:** a duration-aware decoder (HSMM); the missing ball filled in from the players; models
  averaged or stacked; B's network averaged with A's.
- **Cleaning the tracking first:** 24 join, fill and smoothing settings, all worse (−0.003 to −0.008).
- **Training changes:** features pointing to missing players; SkillCorner's matches in training; a "clean owner"
  head; more seeds; B's post stages fed B's own probabilities; B retrained on merged tracking.
- **Metrica's games weighted 4× (`W4`):** better than v16's network alone, but nothing on top of the broadcast
  merge, and it failed the merge's robustness floor.
- **PFF's World Cup 2022 data:** its terms forbid benchmarking.

## 3. Training data

No data of any kind from the provided game was used to train, fit or choose anything, and we made no labels of our
own, on any game.

| dataset | source | who labelled it | how much | used for |
|---|---|---|---|---|
| **IDSSE** | Bassek, Rein, Weber, Memmert (2025), *Scientific Data* 12:195, doi:10.1038/s41597-025-04505-y | DFL's official feed: per-frame `BallPossession` and `BallStatus`, read as written | 7 matches (2022-23, Bundesliga and 2. Bundesliga), 1,002,644 frames at 25 fps (~11 h) | every learned model; the bench |
| **Metrica Sports sample data** | github.com/metrica-sports/sample-data | Metrica Sports tagged the events; we turned them into per-frame H/A/D with a fixed recipe that follows the organisers' rules (`bench/metrica.py`) | 3 anonymised matches | the dead-ball and restart models, the post stages, branch B's GRUs; the bench |
| **SkillCorner open data** | github.com/SkillCorner/opendata | SkillCorner's ball-status and possession fields | 20 A-League 2024-25 matches, broadcast tracking at 10 fps | testing; and the moving ball inside the stoppages of the 10 even-indexed matches, put into training copies of the bench (the odd 10 make the test variants) |
| **PathCRF's weights** | github.com/hyunsungkim-ds/pathcrf | DFL's data (6 of IDSSE's 7 matches), used by PathCRF's authors | as published | one input; its value judged on the 4 bench matches it never saw |

## 4. The possession rules we implemented

The models learn the definition from the labels above. What we decided is **which labels to learn from** and **how
the output is shaped**.

**The convention.** The answer key follows Metrica's tracking. On Metrica's 3 public games (`docs/owner-rule.md`),
any deliberate play starts a possession, a one-touch pass or a shot included; accidental contact (deflections,
blocks) does not. DFL's labels change hands about 25 % more often (412 against 330 possessions per 90 minutes), so
the later versions lean toward Metrica's games (v14's search, v16's training).

| case | what our pipeline does |
|---|---|
| **ball in play** | every graded frame is H, A or D; we never abstain inside the halves |
| **loose ball, pass in flight, lost ball** | stays with the last team: a missing or loose ball is no evidence for a change, and a change has a price in the decoder |
| **a touch or deflection without control** | a change must last 7 frames and pay its price; `short` adds a 0.5-5 s possession only where touch evidence is strong; `segments` removes possessions that look spurious |
| **a won challenge or aerial duel** | learned from the labels: a won duel followed by that team's play changes possession, one without control does not. We asked the organisers how a header won without control is labelled (issue #17); with no answer, this follows Metrica's and DFL's labels |
| **when the change happens** | `timing` moves it to the new team's most likely touch within 3 s |
| **dead ball** | D when the learned P(dead) wins in the decoder; entering or leaving D is expensive (48.7), so short pauses stay live. Branch B is trained for a ball carried or placed during stoppages |
| **restarts** | after a stoppage there is always a possessor: the restarting team is predicted from the players' layout (v6) and enters the per-frame model; the possession starts when play does |
| **the 2nd half's graded frames before kick-off** | D (the FAQ: "those 2,934 frames are graded and all of them are D") |
| **dead-ball ownership** | not produced: a dead ball is D |

## 5. Limitations

1. **The bench is not our match.** 10 matches is a small base, and many ideas were tried on the same ones: holding
   each match out protects settings and weights, not those choices. On SkillCorner's real tracking our models score
   0.41 against 0.75 on the bench; when v12 was diagnosed there, the team was right on 83 % of live frames (92 % on
   the bench), and changes of team landed within about ±1.2 s of the truth (±0.1 s on the bench).
2. **The switch is one number.** A match near its threshold could take the less suited branch.
3. **Missing players near the ball** are the largest error we found: putting back every player missing within
   15 m of the ball would gain about +0.022 on the bench. No cleaning of the tactical tracking recovers it; the
   broadcast merge is our answer.
4. **The broadcast merge was judged on a simulated broadcast**, built only from the statistics of our broadcast set
   in `docs/data.md`. Realistic: +0.0076; pessimistic: −0.0005 and +0.0007. Its main risk is people who aren't
   players, such as a referee tracked with a team. The cap of 11 a team was added after the first pessimistic run
   showed that problem, so it is the one part of the rule chosen after seeing a score. The real broadcast's quality
   can't be measured without labels.
5. **Tracking errors we don't repair:** injected into the bench, they cost branch B about 0.018 for every 1 % of
   outfield frames given the wrong team. Only keepers are repaired.
6. **No ball height:** a high ball sits on the grass, so it passes "near" players it flies over.
7. **Deliberate touch or deflection** is not modelled directly.

**With more time:** automatic tracking of other matches, from the same kind of tracker, with labels; a model that
tells deliberate touches from deflections, trained on Metrica's games; a repair for wrong teams near the ball;
branch B trained together with the broadcast merge, on real broadcast tracking.

## 6. What touched the provided game, and grey zones

The full audit is at the top of `docs/limitations.md`.

**Since the organisers' clarification of 2026-11-03:**
- every parameter is fitted or chosen on external data;
- the program reads the provided tracking, the broadcast tracking and one Smart Tagging tag, as inputs only;
- the public 5 minutes only check the pipeline.

**Statistics of the provided tracking** (never labels, never video) set two things:
- the bench's damage: how often and how long the ball goes missing, piece lengths, missing players and speeds;
- the simulated broadcast that judged v17's merge.

We list it because it sits near the line. The rules allow inspecting the provided data, and nothing was fitted
on its labels or video; it is what makes the bench resemble our match.

**Before the clarification of 2026-11-03,** under our earlier reading of the rules:
- **An AI service saw provided data.** We developed with an AI coding assistant (Claude). In the first days it
  saw screenshots of our viewer with the video, printed tracking values, our labels and statistics of the match.
  Since the clarification it runs nothing on the match whose output it reads, beyond file paths, SHA-256 hashes,
  row counts and pass/fail format checks. No LLM produces or reads labels in the pipeline.
- **Two settings had been set on this match:** the whistle detector, on its audio, and a set-piece window, on
  three tags and the public window. **Both were removed in v10.**
- **Our output was watched over the video:**
  - v2's and v5's whistle changes were checked that way, and v5's whistle weight came from it. Both left with the
    whistle in v10;
  - **the restart rule (v9)** came from seeing that our possessions after restarts started late. Its window was
    chosen on the bench only, and its snap is switched off in the branch our match runs.
- **Since 2026-11-04** our viewer can again show our versions' labels over the video, behind a flag, so a team
  member can see a version run. No version or setting has been chosen from it.
- **The keeper repair (v8)** came from seeing a keeper tracked as the other team. That is watching the data, which
  the rules allow; it is a general rule whose constants come from the Laws of the Game and reasoning.

**Metrica's sample data has no licence file.** We use it as its README asks, credited.

## 7. Reproduction

Python 3.14.7, about 4 minutes on a laptop:

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements-team.txt          # exact versions
# the organisers' files, with their delivered names:
#   data/tactical/   BAR-ATH_tactical_atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml
#   data/broadcast/  BAR-ATH_broadcast_atd.xml, _atd.txt, _homography.csv
.venv/bin/python -m possession.v17                      # writes evaluation/results/v17.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/v17.csv
```

- **Expected SHA-256:** `d7b159c5bc392abd0240f5d2d16a893c26694b5d7ab3cb8bb5dafa4dec69ec04`. Three runs (1 and 2
  threads) and a fresh clone with only the organisers' data give that file, and the format check passes.
- **Determinism:** every model we trained is in `possession/models/`. PathCRF and the GRUs run in float64, so
  another CPU should not move a label. Tested on macOS arm64 only.
- **Network:** only to download PathCRF's pinned weights once; they are refused if their SHA-256 differs.
- **If the broadcast folder is missing** or can't be lined up, the program writes v16's file
  (`90599088419d812cbb341168b27ff0ff767549b75c8dfa12a51de290ccc0523c`).
- **Retraining** every model from the external data is described in `docs/reproduce.md`; it is not needed to
  regenerate the file.

## Credits

- **IDSSE:** Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and event
  data in elite soccer. *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y. CC BY 4.0.
- **Metrica Sports sample data:** github.com/metrica-sports/sample-data.
- **SkillCorner open data:** github.com/SkillCorner/opendata, data by SkillCorner, MIT.
- **PathCRF:** Kim H, Lee K, Seo S, Ko S-K, Yoon J, Park C. *PathCRF: Ball-Free Soccer Event Detection via
  Possession Path Inference from Player Trajectories*. KDD 2026. doi:10.1145/3770855.3818152. MPL-2.0.
- **The organisers** (Metrica Sports, FC Barcelona, Mobile World Capital Barcelona), for the challenge, its data
  and its scorer.
