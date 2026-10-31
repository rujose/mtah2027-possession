# Research: open data and code we can start from

Date: 2026-10-30. Scope: open football datasets and code we could use to train models, choose settings and build our own scorer on matches other than ours (FC Barcelona vs Athletic Club, LaLiga 2024-25, matchday 2).

## Summary

1. **Best stand-in: IDSSE.** It has 7 Bundesliga matches at 25 fps, licensed CC BY 4.0, and the organisers already approved it. Every ball frame carries official `BallPossession` (1 = home, 2 = away) and `BallStatus` (0 = dead, 1 = alive) fields, so H/A/D labels come out almost directly.
2. **Format twin: Metrica sample-data.** Sample Game 3 is Metrica EPTS at 25 fps, the same format family as ours. Games 1–2 are Metrica CSV. None has a possession field. Metrica's own events give H/A/D, and Metrica's event definitions of possession match the organisers' rules.
3. **Volume and realistic ball gaps: PFF/Gradient World Cup 2022.** It covers 64 matches of broadcast tracking at 29.97 fps, with per-frame game events (`home_ball`, `OUT`). It is free behind a form, but the licence is unverified, so ask the organisers before using it. SkillCorner (MIT, 20 A-League matches, 10 fps, per-frame possession team) is the fallback.
4. **Code: kloppy 3.19.1** reads Metrica EPTS, IDSSE, PFF, SkillCorner and SportsCode XML. floodlight and databallpy expose IDSSE possession and ball status. **PathCRF** (MPL-2.0) infers team possession without the ball on IDSSE, with 92.4% frame accuracy, which bears directly on our 41% missing ball.
5. **Traps.** kloppy turns a missing `BallPossession` into "away". databallpy marks every Metrica frame without a ball as dead. In Metrica Sample Game 3 the ball is missing on 94% of dead frames but only 0.1% of live ones, so a model trained there learns "no ball = dead". StatsBomb data and FOOTPASS annotations are non-commercial.

## Words used in this note

- **Tracking data**: x/y positions of every player and the ball, many times per second.
- **Event data**: a human-made log of actions (pass, shot, foul, throw-in) with a time and a team.
- **fps**: frames per second. Ours is 25.
- **EPTS**: FIFA's "Electronic Performance and Tracking Systems" exchange format. It is an XML metadata file that describes a text file of positions. Metrica uses it.
- **Ball status (alive/dead)**: whether play is running (alive) or stopped for an out, foul, goal, etc. (dead).
- **Possession / spell**: which team owns the ball. A spell is an unbroken run of the same team's label while the ball is alive.
- **H/A/D**: our labels. H = home (Barcelona) owns the ball, A = away owns it, D = ball dead.
- **Synchronisation**: shifting event timestamps so each event lines up with the right tracking frame.
- **Broadcast tracking**: tracking computed from the TV feed. Players and the ball off screen are missing or estimated.
- **CC BY 4.0**: open licence. Any use, commercial included, is allowed if you credit the source.
- **Unverified**: I could not confirm it from a primary page.

## 1. Datasets

### 1.1 Comparison table

"Per-frame H/A/D" means the dataset itself gives a team owner and a ball-alive flag for every tracking frame. "Derive" means we must build them from events.

| Dataset | Contents | Matches | fps | H/A/D source | Licence / access | Size | Load effort | Fit for us |
|---|---|---|---|---|---|---|---|---|
| **IDSSE** (DFL / Sportec, Bassek et al. 2025) | Optical tracking (players + ball, with z, speed), official events, match info | 7 (Bundesliga 1 and 2, 2022/23) | 25 | **Per-frame**: `BallPossession` 1/2, `BallStatus` 0/1 on the ball's frames | CC BY 4.0, open download (Figshare, Hugging Face mirror) | 2.63 GB (positions XML 349–419 MB per match) | Easy (one kloppy / floodlight / databallpy call), slow XML parse | **Best**: official labels, same fps, approved |
| **Metrica sample-data** | Game 1–2: tracking CSV + event CSV. Game 3: tracking EPTS (XML + txt) + event JSON | 3 (anonymised) | 25 | **Derive** from Metrica events (BALL OUT, SET PIECE, RECOVERY, PASS, CARRY in Game 3) | No formal licence. README asks to "acknowledge the source". Open on GitHub | Tracking 30–53 MB per file; events 0.1–1.7 MB | Easy (kloppy `metrica.load_open_data`) | **Format twin**. Metrica's possession definitions. Small |
| **PFF FC / Gradient Sports, World Cup 2022** | Broadcast tracking (players by jersey, ball, visibility flags), per-frame game events, event file, rosters, grades | 64 | 29.97 | **Per-frame-ish**: `game_event` with `home_ball` flag; `OUT` → dead until next on-ball event | Free via web form. Terms of use not found (**unverified**) | **Unverified** (one `.jsonl.bz2` per match) | Medium (form, kloppy `pff.load_tracking`, resample 29.97→25) | **Volume** and realistic ball gaps. Licence must be cleared |
| **SkillCorner open data** | Broadcast tracking (extrapolated off-screen players), per-frame `possession` {player, group}, dynamic events, phases of play | 20 folders in repo (README says 10), A-League 2024/25 | 10 | **Per-frame** team in possession. Ball alive only implied (phases are "only defined when the ball is in play") | MIT, credit SkillCorner. Open (Git LFS) | ~89 MB tracking per match | Easy (kloppy `skillcorner.load_open_data`) | Fallback. 10 fps, broadcast, dead ball not explicit |
| **StatsBomb open data (+360)** | Events with `possession` / `possession_team`, lineups. 360 = freeze frames at event moments only. No tracking | 80 competition-seasons (incl. 18 La Liga seasons: 1973/74 and 2004/05–2020/21; mostly Barcelona matches, **unverified**) | n/a | Derive per event only, not per frame | User agreement: no redistribution, **no commercial use**, StatsBomb logo on publications | Large repo | Easy | Priors only (spell and stoppage durations). Licence risk |
| **Wyscout public events** (Pappalardo et al. 2019) | Events (one per ball touch) incl. "out of play" style events. No tracking | One season of 7 competitions incl. La Liga (exact count **unverified**) | n/a | Derive per event (seconds) | CC BY 4.0 (Figshare record) | events.zip 77 MB | Medium (socceraction / kloppy) | Priors only |
| **SoccerNet-v2 action spotting** | 500 broadcast games. 17 classes incl. Ball out, Throw-in, Foul, Kick-off, Corner, free kicks, Goal; each tagged home/away and visible/not | 500 (+50) | Video 25 (per data page) | Derive D intervals (out → restart) on broadcast video; no tracking | Labels open (no password). Videos need SoccerNet NDA via Hugging Face | Large | Medium–hard | Optional: train a broadcast-video dead-ball detector |
| **SoccerNet Ball Action Spotting / Team BAS** | 12 ball actions (Pass, Drive, Out, Throw In, Free Kick, Goal…), Team BAS adds left/right team | 7 public + 2 hidden (English leagues) | Video | Derive possession changes from team-tagged actions | NDA for video (labels' access **unverified**) | Medium | Medium | Small; optional |
| **SoccerNet GSR / Tracking** | Short broadcast clips with pitch positions, team (left/right), role, jersey | 200 clips (length **unverified**) | **Unverified** | None (ball annotation **unverified**) | Download via SoccerNet pip; terms **unverified** | Medium | Medium | Low |
| **FOOTPASS** (Ochin et al. 2025) | 54 full matches 2023/24 (incl. La Liga, UCL), play-by-play on-ball events with team + jersey, tracklets, positions | 54 | 25 | Derive possession from team-tagged on-ball events; ball out not mentioned | Annotations **CC BY-NC 4.0** (Hugging Face `SoccerNet/SN-PCBAS-2026`). Videos under SoccerNet NDA | Large | Medium–hard | Possible extra source. Non-commercial |
| **SoccerTrack v2** (Scott et al. 2025) | Fixed-camera panoramic 4K video (like a tactical camera), per-frame pitch positions (GSR), 12 ball-action classes (BAS) | 10 university matches (~900 min) | **Unverified** | Derive from BAS ("Out", "Throw In", "Free Kick"…); no ball track | CC BY 4.0 (data), MIT (code). Open download | Large | Medium | Optional: closest open "tactical video" |
| **Last Row** | Hand-made tracking of goal build-ups | 19 goal sequences (Liverpool 2019) | 20 | None (only open play before goals) | No formal licence: "credit the source" | Small | Easy | Not useful |
| **Signality / Hammarby** | Tracking + video (not synced), no events. Signality format has per-frame `state` ("running") and ball `team` | 3 | **Unverified** (25 per course text) | Per-frame in format (kloppy reader) | Course material; access and licence **unverified** | ? | ? | Only if obtainable |
| **EIGD-S** (Biermann et al. 2021) | Soccer video + event labels; positions only for handball | small | n/a | n/a | Handball part CC BY-NC-SA | n/a | n/a | Not useful (no soccer tracking) |

I found no other open full-match tracking with possession labels. Kloppy has readers for Second Spectrum, StatsPerform, TRACAB, Hawk-Eye and Signality, but only test fixtures are public.

None of these datasets contains our match (LaLiga 2024-25). StatsBomb La Liga stops at 2020/21. FOOTPASS is 2023/24. Wyscout is one older season. SoccerNet-v2 is older. Keep a guard anyway, see section 3.

### 1.2 IDSSE (DFL / Sportec open data)

**What it is.** Seven complete 2022/23 matches: 2 Bundesliga and 5 2. Bundesliga, the latter all with Fortuna Düsseldorf. Each match has three XML files: match information, events, positions. The positions come from TRACAB optical tracking (16–20 cameras). Frame timestamps step by 0.04 s, which I checked in a 60 KB range read, so it is 25 fps. The paper counts 1,002,644 position frames and 11,137 events in total.

**Possession and ball status.** The paper says ball frames carry `BallPossession` (1 = home, 2 = away) and `BallStatus` (0 = inactive, 1 = active). It defines possession as "defined by the player in control of the ball". The match state is inactive "when the match is interrupted by the referee". kloppy, floodlight and databallpy all parse these two fields.

**Turning it into H/A/D.**
- `BallStatus == 0` → D.
- Otherwise `BallPossession == 1` → H, `== 2` → A, where "home" is the home team in the match info.

**What to check before trusting it (unverified):**
- **Loose balls.** DFL defines possession by "control". Does `BallPossession` keep the last team while the ball is loose, as our convention requires, or does it flip on any touch?
- **Flicker.** In kloppy's DFL test file (a different match, `DFL-MAT-003BN1`) `BallPossession` goes 2 → 1 → 2 over three consecutive frames at kick-off. Count how often one-team spells shorter than ~0.5 s occur in IDSSE.
- **Timing of D.** Does `BallStatus` turn 0 when the ball crosses the line, or at the whistle? The organisers' D comes from Metrica's corrected tracking and may use a different moment. Shift the D edges by ±0.5 s and see how much our score moves.
- **Restart team.** After each D run, is the first alive team the team that takes the restart? Check against the events (ThrowIn, FreeKick, CornerKick, GoalKick, KickOff).

**Events available** (element names seen in kloppy's Sportec test file): Play, ShotAtGoal, TacklingGame, Foul, ThrowIn, FreeKick, CornerKick, GoalKick, KickOff, FinalWhistle, BallClaiming, OtherBallAction, Offside, and others. The paper used databallpy to synchronise passes and shots. The mean shift was −0.37 ± 1.82 s, maximum 27 s. The PathCRF repo already ships ELASTIC-synchronised IDSSE events (`data/sportec_synced`).

**Gotchas.**
- The Hugging Face README lists match IDs `J03WPF` and `J03WQF`, which do not exist. The real files are `J03WQQ` and `J03WR9`. kloppy and floodlight use the right IDs.
- kloppy maps any `BallPossession` other than 1 to the away team, including a missing value.

**Licence.** CC BY 4.0 on Figshare (DOI 10.6084/m9.figshare.28196177). Cite Bassek, Rein, Weber, Memmert (2025), *Scientific Data* 12:195.

### 1.3 Metrica Sports sample-data

**What it is.** Three anonymised matches. Games 1–2: tracking split into home and away CSVs plus an event CSV. Game 3: EPTS (FIFA XML metadata + `:`/`;`/`,`-separated txt) plus event JSON. All run at 25 fps: Game 1 time steps are 0.04 s, and the Game 3 XML says `<FrameRate>25</FrameRate>`. Coordinates are normalised 0–1 on a 105 × 68 m pitch, as in ours.

**Differences from our input.** Players have identities, not anonymous team-only slots. Game 3's EPTS declares only position channels (no speed). There is no possession or ball-status field. I did not open our own EPTS under `data/`, so I cannot say whether kloppy loads our anonymous slots.

**The ball is blanked in dead time.** I read 12 small slices of Game 3 tracking (4,826 frames) and compared them with dead periods derived from events (below). The ball is `NaN` on **94% of event-dead frames** and **0.1% of event-alive frames**. In this sample, a missing ball almost means "dead". In our automatic tracking, missing ball (41%) comes from detection failures in crowds. Do not let a model learn this shortcut on Metrica data.

**Turning events into H/A/D.** Metrica's event definitions PDF defines possession much as the organisers do. A player is in possession from the first touch. Possession "is not considered lost when an opposing player's interference does not have a significant impact". RECOVERY also covers bringing the ball back into play from a dead ball that is not a SET PIECE. Recipe:
1. **Owner.** The team of the latest controlling event that started at or before the frame. Controlling events: SET PIECE, PASS (including subtype GOAL KICK), CARRY (Game 3 only), SHOT, RECOVERY. A pass in flight stays with the passer's team. BALL LOST and CHALLENGE do not flip the owner on their own. The flip happens at the other team's next RECOVERY, PASS or CARRY.
2. **Dead from:**
   - the BALL OUT end frame;
   - the start frame of a foul (CHALLENGE with subtype FAULT, or FAULT RECEIVED);
   - any OFFSIDE subtype;
   - the end of a SHOT with result GOAL or OUT.
3. **Dead until** the next restart: a SET PIECE start (KICK OFF, THROW IN, FREE KICK, CORNER KICK, PENALTY) or a PASS with subtype GOAL KICK. Note that goal kicks are PASS subtypes, not SET PIECEs. Everything before the first kick-off is D.
4. **Rough check on Game 3.** This gives 126 stoppages (median 13 s) and 32.7% D. Stoppages with no event (injuries, substitutions) are probably missed. Use long ball-`NaN` runs as a cross-check.

**Event counts (Game 3).** 1,395 CARRY, 1,121 PASS, 335 CHALLENGE, 294 RECOVERY, 288 BALL LOST, 90 SET PIECE, 64 BALL OUT, 20 SHOT, 12 FAULT RECEIVED.

**Licence.** None stated. The README says to be responsible and acknowledge the source. The organisers can reach the data: it is Metrica's own public repo.

### 1.4 PFF FC / Gradient Sports, FIFA World Cup 2022

**What it is.** All 64 World Cup 2022 matches. Files: broadcast tracking per match (`{game_id}.jsonl.bz2`), events (`events.json`), and `metadata.csv`, `players.csv`, `rosters.csv`, `competitions.csv`, plus PDF specs. Tracking comes from PFF's partnership with Sportlogiq. PFF FC is now Gradient Sports.

**Frame contents** (seen in kloppy's PFF test file). `homePlayers` and `awayPlayers` hold jersey, x, y, speed, `confidence` and `visibility` (VISIBLE / ESTIMATED), plus smoothed versions. `balls` hold visibility, x, y, z. Each frame also has `game_event` and `possession_event`.
- `game_event_type` takes FIRSTKICKOFF, SECONDKICKOFF, OTB (on the ball), OUT, END, SUB, G (hits post and stays in play), VID (video missing), and others.
- `home_ball` is true or false.
- Metadata says `"fps": 29.97`.

**Turning it into H/A/D.** This follows kloppy's logic.
- Ball dead after OUT or END, until the next kick-off or OTB.
- The owner is the `home_ball` value of the latest game event.
- Fouls are not marked OUT in kloppy's logic. Stoppages for fouls have to come from the event file (field names **unverified**).
- Resample to 25 fps by timestamp.

**Why it is useful.** It is many matches. Its ball goes missing for real reasons (off camera): in kloppy's 100-frame test file, 50 frames had no ball entry. The missingness mechanism (camera framing) differs from ours (crowding).

**Licence and access.** A web form gives "instant access". The page asks for social posts to tag @PFF_FC. I could not find the terms of use. SoccerNet-GAR, which reuses this data, sits behind an NDA. Status: **unverified, ask the organisers** whether they accept it and can access it.

### 1.5 SkillCorner open data

A-League 2024/25 broadcast tracking at 10 fps. Players out of shot are extrapolated; `is_detected` flags real detections. Each frame has `possession` = {player_id, group (home/away)}. The README says 10 matches, but the repo holds 20 match folders (checked 2026-10-30). Tracking files are Git LFS (~89 MB each). Dynamic events have `game_interruption_before/after` fields. Phases of play cover only in-play time: in match 1886347 they cover 54% of frames.

**H/A/D recipe.** Frames inside a phase are alive. Frames outside all phases are D (**unverified**: gaps may also be unclassified play). Inside phases, a null possession group keeps the last team. Then upsample to 25 fps. kloppy maps a null possession straight to "dead", which is wrong for loose balls.

**Licence.** MIT; please credit SkillCorner.

### 1.6 Event-only and video-only sources

- **StatsBomb open data.** Events carry a curated possession ID and team, but there is no tracking (360 data are freeze frames at event moments). The user agreement forbids redistribution and commercial exploitation "of the data or any analysis derived". A sponsor-run hackathon may count as commercial. Use only for priors such as spell length and stoppage length, and only if the organisers agree.
- **Wyscout (Pappalardo et al.).** Events, CC BY 4.0. Safe for priors. Timestamps are in seconds.
- **SoccerNet-v2.** The 17 classes include Ball out, Throw-in, Foul, Kick-off, Corner, Indirect and Direct free-kick, Goal, Penalty, Offside. Each is tagged home/away and shown/unshown. The labels download without a password, but the videos need the NDA. This is the only large source to train a **broadcast-video** dead-ball detector. Our broadcast video is a permitted input.
- **SoccerTrack v2.** Panoramic fixed-camera video, which looks like a tactical camera. The data is CC BY 4.0, with per-frame pitch positions and 12 ball-action classes. It is the open analogue of our tactical video if we try a video model.
- **FOOTPASS.** Annotations are CC BY-NC 4.0 and the videos sit under the NDA. It has 54 full 2023/24 matches with team-tagged on-ball events at 25 fps. It is a good extra source of possession changes, but non-commercial.

## 2. Code libraries

| Library | Version (date) | Licence | What it gives us | Caveats |
|---|---|---|---|---|
| **kloppy** | 3.19.1 (PyPI, 2026-10-03) | BSD-3-Clause | One API for many providers. `metrica.load_tracking_epts(meta_data, raw_data)` reads Metrica EPTS (position, speed, distance sensors). `metrica.load_open_data(match_id=1/2/3)`. `sportec.load_open_tracking_data(match_id)` / `load_open_event_data` read IDSSE from Hugging Face with `ball_state` and `ball_owning_team` per frame. Also `pff.load_tracking`, `skillcorner.load_open_data`, `sportscode.load` (our Smart Tagging XML), StatsBomb, Wyscout | EPTS frames get `ball_state=None` and `ball_owning_team=None`. Missing `BallPossession` → away. SkillCorner null possession → dead. Untested on our anonymous 46-slot EPTS |
| **floodlight** | 1.2.0 (2026-05-04) | MIT | `io.dfl.read_position_data_xml` returns xy, **possession** and **ballstatus** `Code` objects per half. `io.datasets.IDSSEDataset` downloads and loads all 7 IDSSE matches. Also TRACAB, Second Spectrum, Kinexon, StatsBomb, Opta readers | No Metrica reader listed |
| **databallpy** | 0.8.1 (2026-08-11) | MIT | Loads IDSSE (`ball_status`, `team_possession` columns from DFL), Metrica (incl. the EPTS txt), TRACAB, Inmotio. Event–tracking sync with Needleman-Wunsch. `add_team_possession` (from synced events). `add_individual_player_possession` (tracking-only rules after Vidal-Codina et al. 2022: 1.5 m zone, 5 m/s speed change, 10° direction change) | Metrica parser sets `ball_status = "dead"` wherever the ball is missing: wrong for us. In its docs example, event-based and provider possession agreed on ~90% of alive frames |
| **PathCRF** (hyunsungkim-ds/pathcrf) | repo, last push 2026-06-08 (KDD 2026) | MPL-2.0 | **Ball-free** possession: infers the possession path from player trajectories only. Uses IDSSE (5 train / 1 val / 1 test), downsampled to 5 fps. Frame-level team possession accuracy 92.40% on the test match. Pretrained models in `saved/`. Its labels come from ELASTIC-synced events plus ball-touch detection, not from `BallPossession` | Trains on in-play segments only (no dead-ball model). Labels use a different definition |
| **ELASTIC** (hyunsungkim-ds/elastic) | repo, last push 2026-09-07 (CIKM 2026) | MPL-2.0 | Event–tracking sync that uses only tracking features (no event locations). Tutorial on IDSSE. Beats ETSY in the paper | Only needed if we build event-derived labels |
| **ballradar** (hyunsungkim-ds/ballradar) | repo (KDD 2023) | **No licence file** (all rights reserved by default) | Infers ball trajectory and ball possessor from player trajectories. Public pipeline runs on the 3 Metrica sample games | Reading is fine. Copying code is not, without permission |
| **socceraction** | 1.5.3 (2024-08-15) | MIT | Converts StatsBomb / Wyscout / Opta events to SPADL (a common action format). Useful for event-based priors | No tracking |
| **unravelsports** | 1.2.1 (PyPI, 2026-01-13) | MPL-2.0 | kloppy → Polars conversion. Graph neural network tooling. Pressing intensity. EFPI formation detection (team shape) | No possession or ball-in-play model |
| **mplsoccer** | 1.8.1 (2026-09-14) | MIT | Pitch plots for visual checks of our labels | Plotting only |
| **LaurieOnTracking** (Friends of Tracking) | no releases, last push 2022-11-03 | MIT | Metrica CSV reader, velocity smoothing, pitch control, EPV tutorials | CSV format only. Old |
| **ETSY** (ML-KULeuven) | no PyPI release, last push 2024-05-13 | Apache-2.0 per README (GitHub API says "NOASSERTION") | Rule-based event–tracking sync, needs SPADL events | Superseded by ELASTIC in benchmarks |
| **sync.soccer** | last push 2023-08-30 | AGPL-3.0 | Needleman-Wunsch sync for Opta F24 + TRACAB | Haskell. AGPL would bind our public repo |

**Anything that already computes possession or ball-in-play from tracking:**
- Possession needing the ball: databallpy's individual possession rules.
- Possession without the ball: PathCRF, ballradar's possessor classifier.
- Vidal-Codina et al. 2022 (possession plus event detection from tracking): a paper with no public code found.
- **I found no open tool that detects ball alive/dead from tracking alone.** Every dead-ball signal in these datasets comes from provider fields or events. We will have to build that part ourselves.

## 3. Recommendation

### Pick these three

1. **IDSSE: main validation and training set.** It has official per-frame owner and status, 25 fps, ~11 hours of play, CC BY 4.0, and it is already approved.
2. **Metrica sample-data (3 games): convention and format check.** It is the same provider and format family, and Metrica's event definitions are the closest public text to the organisers' rules. Use it to test that our pipeline runs on Metrica-shaped input. Also use it to see whether IDSSE-tuned settings transfer to Metrica-style labels.
3. **PFF World Cup 2022 (64 matches): scale and real ball gaps**, if the organisers clear the licence. If not, use **SkillCorner (20 matches)** for robustness checks only, because of its 10 fps and implicit dead ball.

### Plan: from raw files to a validation set in the H/A/D convention

**Step 0. Paperwork and guard.**
- Write `docs/external-data.md` listing each dataset, URL, licence and citation. CC BY needs attribution.
- Send the organisers the list (IDSSE is already approved; add Metrica sample, PFF, SkillCorner).
- Add a hard check that no dataset row is FC Barcelona vs Athletic Club 2024-25.

**Step 1. Load and cache.**
- kloppy for all three. floodlight as a second IDSSE reader to cross-check possession and ball-status parsing.
- Save each match once as Parquet, at 25 fps, in our own column layout.

**Step 2. Build labels.**
- **IDSSE**: D if `BallStatus = 0`, else H/A from `BallPossession`. Then apply our convention.
  - *Short-spell rule.* A spell of team B shorter than *k* frames, between two spells of team A, becomes A. This handles "touch without control" and "loose ball stays with last team".
  - *Choosing k.* Pick *k* by agreement with an independent event-based owner: ELASTIC-synced IDSSE passes and receptions from the PathCRF repo, or databallpy `add_team_possession`.
  - *Report.* Log how many frames the rule changes.
- **Metrica**: the event recipe in 1.3.
  - Check it on the frames where Games 1–3 overlap in meaning: restarts and turnovers.
- **PFF**: game-event recipe in 1.4, plus foul stoppages from `events.json`.

**Step 3. Make the inputs look like ours.**
- Normalise to 0–1.
- Drop player identities and keep only team.
- Break each player track into fragments, spread over up to 46 anonymous slots. Match the number of fragments per match to what our own input shows; counting slots in our input uses no labels.
- Recompute speed from positions as our EPTS does.
- Resample PFF (29.97 fps) and SkillCorner (10 fps) to 25 fps.

**Step 4. Degrade the ball to mimic ours** (41% missing, concentrated in crowded moments).
1. *Measure on our own input tracking.* This is input data, not labels or human data, so it is allowed; confirm with the organisers if unsure. Per frame, compute:
   - whether the ball is missing;
   - crowding: players within 2/3/5 m of the last seen ball position, and the gap to the second-nearest opponent;
   - distance of the ball from the touchlines;
   - ball speed just before the gap.
   Then measure the length of missing runs.
2. *Fit a two-state model* (ball seen / ball missing). The chance of entering a missing run depends on crowding, through a logistic fit. Run lengths are drawn from our measured run-length histogram.
3. *Apply it to IDSSE, PFF and Metrica alive frames,* using their true ball and players to compute crowding. Calibrate so the overall missing rate on graded-style frames is 41% and the crowding curve matches ours.
4. *Add the other faults* seen in our ball track: position jitter (σ from our smooth segments), occasional false detections (jumps to a player), and no z coordinate.
5. *Treat dead frames separately.* Metrica's sample already blanks the ball in dead time, and IDSSE never does. Set the dead-time missing rate to what our input shows during long low-motion stretches, so "no ball" is not a free dead-ball signal.
6. *Make three severities* (about 30%, 41%, 55% missing). A method that only works at one level is fragile.

**Step 5. Our own scorer.**
- Implement spellF1 as the organisers describe it (see `docs/challenge.md`): spells as objects, one-to-one IoU matching, F1 averaged over thresholds 0.3–0.7, and `X` replaced by our previous answer. Add per-frame macro-F1.
- Sanity-check on IDSSE with the organisers' baselines (nearest player; majority vote over ±1, ±2, ±5 s). The ranking should look like the repo's table (±2 s best spellF1 among them). If it does not, our labels or scorer are off.

**Step 6. Protocol.**
- Leave one match out across IDSSE: choose settings on 6 matches, report on the 7th, rotate.
- Report Metrica and PFF separately as transfer tests.
- Freeze settings before the final run on our match. Never tune on our match, not even on the 5 public minutes.

### Open questions to settle early

- How does DFL `BallPossession` behave on loose balls and deflections?
- Will the organisers accept PFF data, and StatsBomb or FOOTPASS for priors?
- Does kloppy load our anonymous 46-slot EPTS? Test it on our file; that is a quick check.

## Sources

Datasets
- Metrica sample-data repo: https://github.com/metrica-sports/sample-data
- Metrica README: https://raw.githubusercontent.com/metrica-sports/sample-data/master/README.md
- Metrica event definitions: https://github.com/metrica-sports/sample-data/blob/master/documentation/events-definitions.pdf
- Metrica files read (small, or range-read slices only): https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_metadata.xml, .../Sample_Game_3/Sample_Game_3_events.json, .../Sample_Game_3/Sample_Game_3_tracking.txt, .../Sample_Game_1/Sample_Game_1_RawEventsData.csv, .../Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv
- IDSSE Hugging Face mirror: https://huggingface.co/datasets/pysport/idsse-data and https://huggingface.co/datasets/pysport/idsse-data/tree/main
- IDSSE Figshare: https://doi.org/10.6084/m9.figshare.28196177 (record via https://api.figshare.com/v2/articles/28196177)
- IDSSE paper: https://doi.org/10.1038/s41597-025-04505-y (read at https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- PFF release post: https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data
- Gradient enhanced dataset post: https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset
- SoccerNet-GAR (PFF reuse, NDA): https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR
- fast-forward Gradient loader docs: https://fast-forward.readthedocs.io/en/latest/providers/gradientsports/
- SkillCorner open data: https://github.com/SkillCorner/opendata (README, phases and dynamic-events CSVs for match 1886347)
- StatsBomb open data: https://github.com/statsbomb/open-data (now https://github.com/hudl/open-data), LICENSE.pdf and data/competitions.json
- Wyscout events (Pappalardo et al.): https://figshare.com/collections/Soccer_match_event_dataset/4415000 (records https://api.figshare.com/v2/articles/7770599 and /7770422), https://kdd.isti.cnr.it/node/1266, paper https://www.nature.com/articles/s41597-019-0247-7 (not readable here)
- SoccerNet data page: https://www.soccer-net.org/data
- SoccerNet spotting: https://github.com/SoccerNet/sn-spotting ; SoccerNet-v2 paper: https://ar5iv.labs.arxiv.org/html/2011.13367
- SoccerNet 2025 challenge results (Team BAS): https://arxiv.org/abs/2508.19182
- SoccerNet Game State Reconstruction: https://github.com/SoccerNet/sn-gamestate
- FOOTPASS: https://arxiv.org/abs/2511.16183 , https://arxiv.org/html/2511.16183 , https://github.com/JeremieOchin/FOOTPASS , https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026
- SoccerTrack v2: https://atomscott.github.io/SoccerTrack-v2/ , https://arxiv.org/abs/2508.01802
- Last Row: https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row
- Signality / Hammarby (course, not opened): https://uppsala.instructure.com/courses/44338
- EIGD: https://arxiv.org/abs/2108.11149 , https://github.com/MM4SPA/eigd

Code
- kloppy: https://github.com/PySport/kloppy , https://pypi.org/project/kloppy/
  - Sportec reader: https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/tracking/sportec/deserializer.py
  - Sportec / IDSSE provider: https://github.com/PySport/kloppy/blob/master/kloppy/_providers/sportec.py
  - Metrica provider and EPTS reader: https://github.com/PySport/kloppy/blob/master/kloppy/_providers/metrica.py , https://github.com/PySport/kloppy/tree/master/kloppy/infra/serializers/tracking/metrica_epts
  - PFF, SkillCorner, Signality readers: https://github.com/PySport/kloppy/tree/master/kloppy/infra/serializers/tracking
  - SportsCode provider: https://github.com/PySport/kloppy/blob/master/kloppy/_providers/sportscode.py
  - Test fixtures used: https://github.com/PySport/kloppy/tree/master/kloppy/tests/files (sportec_positional.xml, sportec_events.xml, pff_3812.jsonl, pff_metadata_3812.json)
- floodlight: https://github.com/floodlight-sports/floodlight , https://floodlight.readthedocs.io/en/latest/modules/io/dfl.html , https://floodlight.readthedocs.io/en/latest/modules/io/datasets.html , https://pypi.org/project/floodlight/
- databallpy: https://github.com/Alek050/databallpy , https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html , https://databallpy.readthedocs.io/en/v0.5.0/getting_started/loading_in_a_match_page.html , source files `metrica_tracking_data_parser.py` and `tracab_parser.py` in the repo, https://pypi.org/project/databallpy/
- PathCRF: https://github.com/hyunsungkim-ds/pathcrf , https://arxiv.org/abs/2602.12080 , https://arxiv.org/html/2602.12080
- ELASTIC: https://github.com/hyunsungkim-ds/elastic , https://arxiv.org/abs/2508.09238
- ballradar: https://github.com/hyunsungkim-ds/ballradar , https://arxiv.org/abs/2306.08206
- Vidal-Codina et al. 2022: https://arxiv.org/abs/2202.00804
- socceraction: https://github.com/ML-KULeuven/socceraction , https://pypi.org/project/socceraction/
- unravelsports: https://github.com/UnravelSports/unravelsports , https://pypi.org/project/unravelsports/
- mplsoccer: https://github.com/andrewRowlinson/mplsoccer , https://pypi.org/project/mplsoccer/
- LaurieOnTracking: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
- ETSY: https://github.com/ML-KULeuven/ETSY
- sync.soccer: https://github.com/huffyhenry/sync.soccer

Method note: versions, licences and last-push dates come from the PyPI JSON API and the GitHub API on 2026-10-30. The Metrica checks used files under 2 MB plus 12 range reads of ~150 KB from the 53 MB Game 3 tracking file. IDSSE frame timing came from a 60 KB range read. Nothing was downloaded into the repo, and no match data of ours was read or sent anywhere.
