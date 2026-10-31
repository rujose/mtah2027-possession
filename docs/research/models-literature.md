# Research: models that exist for this problem

**Date:** 2026-10-30. Every claim below comes from a page fetched on this date, unless it is marked **unverified**.

## Summary

1. The closest published method is **PathCRF** (KDD 2026). It infers possession from player positions alone, with no ball. Its weights are public, and it was trained on 7 open Bundesliga matches. It reports 92.4 % frame-level team-possession accuracy on one clean test match.
2. **Dead-ball detection from players only** works: Lang et al. 2022 report 92 % frame accuracy. Open DFL data (7 matches, CC BY 4.0) has per-frame "ball alive/dead" and "team in possession" flags we can train and tune on.
3. Our metric is the action-segmentation field's **segmental F1 at an IoU threshold** (Lea et al. 2017). That field fixes flicker with duration-aware decoding (hidden semi-Markov models), smoothing losses and boundary refinement.
4. Video methods (SoccerNet ball-action spotting, T-DEED) need an NVIDIA GPU and broadcast video, so they fit us poorly. Ball re-detection from our tactical video (WASB) is possible but slow on an M2.
5. Ranked plan: (1) duration-aware decoder, (2) learned dead-ball detector, (3) PathCRF as the clue for ball-missing frames, (4) touch-based ball-control rule, (5) learned per-frame scores. Tune all of them on other matches, using a test bench that degrades clean data to look like ours.

## How to read this

- **Tracking data:** x/y positions of players (and the ball) at a fixed rate, here 25 frames per second.
- **IoU (intersection over union):** the overlap of two time spans divided by their combined span. 1 means identical, 0 means no overlap.
- **F1:** the balance of precision ("how many of my answers are right") and recall ("how many of the real ones did I find").
- **HMM (hidden Markov model):** a model with hidden states (here H, A, D) that switch over time and produce noisy clues. **Viterbi** is the algorithm that finds the single best state sequence. Our v1 decider is one.
- **"Fit"** below means fit to *our* setting. Our setting is: ball missing on 41 % of graded frames; 46 anonymous slots that know their team but not the player; identity swaps; settings chosen only on other matches or by reasoning; an M2 with 8 GB and no NVIDIA GPU; about 2.5 weeks left.
- **Effort** is for one person who already knows our code.

---

## 0. Labelled data from other matches (what makes most methods usable)

The rules allow labelled data from other matches. Almost every method below needs such data, either to train or to choose settings. These are the open sources.

| Dataset | What it has | Labels useful to us | Licence | Size / notes |
|---|---|---|---|---|
| **Sportec Open DFL dataset** (Bassek et al., *Scientific Data* 2025) | 7 Bundesliga 1/2 matches, 2022-23. Optical tracking (TRACAB Gen5) of all players and the ball, plus official events | Per-frame **`BallPossession`** (1 = home, 2 = away; "the player in control of the ball") and **`BallStatus`** (0 = inactive, 1 = active). Effectively H/A/D for every frame | CC BY 4.0 | Position XML is 349–419 MB per match, about 2.6 GB in total. Load with kloppy (`sportec.load_open_tracking_data`), which maps these fields to `ball_owning_team` and `ball_state`. Parse one match at a time on 8 GB |
| **Metrica Sports sample data** | 3 anonymised matches. Tracking at 25 fps with ball, plus synchronised events | Events include `RECOVERY`, `BALL LOST`, `BALL OUT`, `SET PIECE`, `FAULT RECEIVED` and `CHALLENGE` (counted in Game 1). From these we can build H/A/D. Same company as our ground truth, so the conventions are likely the closest (**unverified**) | No licence file. The README asks users to "be responsible" and acknowledge the source | Game 1 and 2 tracking is about 30–33 MB per team file. Game 3 tracking is 53.4 MB |
| **SkillCorner open data** | 10 A-League 2024-25 matches. Broadcast tracking at 10 fps, so players and the ball drop out off screen | Per-frame `possession` (player and group). `phases_of_play` are "only defined when the ball is in play". kloppy marks a frame "dead" whenever possession is empty, so "dead" may also mean "unknown" | MIT | Closest to our degraded input (missing data). We could not verify whether the possession labels are human-made |
| **PFF FC / Gradient World Cup 2022** | 64 matches. Broadcast tracking plus graded events | Not checked | Free form sign-up. Licence **unverified** | Frame rate and ball coverage **unverified** |

Useful tools: **kloppy** (BSD-3, loads all of the above), **databallpy** (MIT, has a Vidal-Codina possession function), **floodlight** (MIT).

---

## 1. Possession from tracking data

### 1.1 Link & Hoernig 2017, "Individual ball possession in soccer"
- **Citation:** Link D, Hoernig M. *PLOS ONE* 12(7): e0179953, 2017. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/>
- **Inputs:** x/y of players and ball at 25 Hz, plus the provider's "running" flag (ball in or out of play). Ball height is not used.
- **How it works:** A player *gains* the ball when he is the nearest player within a distance threshold **and** the smoothed ball acceleration has a local peak of at least 4 m/s² (a kick). He *loses* it when another player gains it, play stops, or he is too far from where the ball will be in 1 s. A Bayesian network then decides "controlled or not". It uses the duration, ball speed and acceleration statistics, and the number of opponents within 0.5–5 m. Team possession is the union of the team's player possessions.
- **Code:** none found (**unverified**).
- **Results:** 60 Bundesliga matches (TRACAB), with 5 matches labelled by hand. The possession-span detector reached F-score 0.88. The control classifier was 96.7 % correct overall, but found only 50 % of uncontrolled spans (kappa 0.67, against 0.72 between two humans).
- **Fit:** It needs the ball. That is fine on our 59 % visible frames and useless on the rest. It needs no player identity. Its rule ("a touch is a ball-acceleration peak near a player") matches our ground-truth convention: a deflection is not a change of control. Runs anywhere.
- **Effort:** 1 day to add the kick-detection test to our "who is on the ball" clue.

### 1.2 Vidal-Codina et al. 2022, "Automatic event detection in football using tracking data"
- **Citation:** Vidal-Codina F, Evans N, El Fakir B, Billingham J. *Sports Engineering* 25:18, 2022. CC BY 4.0. <https://arxiv.org/abs/2202.00804>
- **Inputs:** x/y of players and ball at 25 Hz, plus a dead-ball flag (or a missing ball when play is dead).
- **How it works:** Two deterministic rule stages. (1) Possession: each player has a "possession zone" (radius 0.5 m or 1 m depending on the provider). A *gain* needs the ball to change direction or speed inside the zone. A ball passing by unchanged is not a gain. A *loss* needs the ball to leave the zone and another player to touch it next. Opponents both inside 1 m is a "duel". (2) Events: possession changes plus the Laws of the Game give passes, shots and so on. **Set-piece triggers** read player layouts during dead time. Examples: all players in their own half plus one near the centre spot means a kick-off; a player beyond the touchline means a throw-in; a player near a corner flag means a corner.
- **Code:** not released by the authors. **databallpy** (MIT) reimplements the player-possession step as `add_individual_player_possession(pz_radius=1.5, bv_threshold=5.0, ba_threshold=10.0, min_frames_pz=0)`.
- **Results:** 31 matches from three providers (Track160, TRACAB, Hawk-Eye). Passes had above 90 % precision and recall. Shots had 78 % precision and 53 % recall. Corners, throw-ins and goal kicks were above 90 %. It took 3 minutes per match on 1 CPU with 4 GB RAM. The authors name as limits that the ball must be present and the dead-ball flag must exist. They suggest "a synchronized audio feed that provides timestamped referee whistles" as future work, which is what our v2 does.
- **Fit:** Same as Link & Hoernig for the ball. The **set-piece triggers need only players**, so they work when the ball is missing. They tell us both "the ball is dead" and **who restarts**.
- **Effort:** possession rule 1 day (start from databallpy). Set-piece triggers 1–2 days.

### 1.3 Other rule-based event recognisers (lower priority)
- Khaustov & Mozgovoy 2020, "Recognizing events in spatiotemporal soccer data", *Applied Sciences* 10(22):8046. A rule-based baseline for possession, passes and shots. <https://www.mdpi.com/2076-3417/10/22/8046>. Detailed rules and numbers not read.
- Morra et al. 2020, "Slicing and dicing soccer", ICIAR 2020. Atomic and complex events via interval temporal logic, tested on *synthetic* data. Code on GitLab. <https://arxiv.org/abs/2004.04147>
- **Fit:** both need the ball and add nothing beyond 1.1–1.2.

### 1.4 Wei et al. 2015, "Predicting ball ownership in basketball from a monocular view using only player trajectories"
- **Citation:** Wei X, Sha L, Lucey P, Carr P, Sridharan S, Matthews I. ICCV Workshops 2015. <https://openaccess.thecvf.com/content_iccv_2015_workshops/w21/html/Wei_Predicting_Ball_Ownership_ICCV_2015_paper.html>
- **How it works:** It sorts players by *role* rather than identity ("permuting" the data). This gives a compact feature that predicts the ball owner from player positions alone. A ball detector can be fused in when available.
- **Code / results:** no code found. Accuracy numbers **unverified** (the page returned 403).
- **Fit:** It is the original "ball owner without the ball" idea. Role-ordering avoids the need for identities, which suits our anonymous slots. Basketball only.

### 1.5 Kim et al. 2023, "Ball trajectory inference from multi-agent sports contexts using Set Transformer and hierarchical Bi-LSTM" (Ball Radar)
- **Citation:** Kim H, Choi H-J, Kim CJ, Yoon J, Ko S-K. KDD 2023. <https://arxiv.org/abs/2306.08206>
- **Inputs:** player trajectories only, at 10 Hz.
- **How it works:** A **Set Transformer** is an attention network whose output does not depend on the order of the players ("permutation-invariant"), so identities are not needed. It first classifies **which player has the ball**, then regresses the ball path. A **Bi-LSTM** (a recurrent network reading the sequence forwards and backwards) models time. A rule-based post-processor keeps the result physically possible.
- **Code:** <https://github.com/hyunsungkim-ds/ballradar> (moved from `pientist/ballradar`). It has **no licence file**, so we may read it for ideas but should not copy it without permission. No pretrained weights are listed. Python 3.10, PyTorch 1.13.
- **Results:** player-level possessor accuracy 64.7 %. **Team-level possession accuracy 85.85 %**. Ball position error 3.66 m. Trained on 15 GPS matches (private) plus the 3 Metrica sample games.
- **Fit:** It works with no ball and no identities, but it needs a fixed set of 22 players per frame. It runs on an M2. Superseded by PathCRF (1.8), which is by the same first author, includes Ball Radar as a baseline, and has an open licence and weights.
- **Effort:** not recommended separately.

### 1.6 Capellera et al. 2024, "TranSPORTmer: a holistic approach to trajectory understanding in multi-agent sports"
- **Citation:** ACCV 2024. <https://arxiv.org/abs/2410.17785>, <https://openaccess.thecvf.com/content/ACCV2024/html/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.html>
- **How it works:** A set-attention transformer over players and time. An extra "CLS agent" labels **every frame** as *pass / possession / uncontrolled / out-of-play*. The same model also infers the ball when it is hidden.
- **Data:** 283 LaLiga 2022-23 matches (provider not named), at 6.25 Hz.
- **Results:** frame-state accuracy 87–90 % in its forecasting and imputation settings. With the ball fully hidden, ball error was 2.71 m against Ball Radar's 3.89 m. Trained on an RTX A6000.
- **Code:** no official code found (**unverified**). An unofficial reimplementation is in the PathCRF repo (`models/transportmer.py`).
- **Fit:** The idea is right for us: per-frame states that include out-of-play, from players. But the data is private and there is no code, so it is not usable as is.

### 1.7 Peral et al. 2025, "Multi-modal soccer scene analysis with masked pre-training"
- **Citation:** Peral M, Capellera G, Ferraz L, Rubio A, Agudo A. arXiv 2512.19528, December 2025. <https://arxiv.org/abs/2512.19528>
- **How it works:** Same family as TranSPORTmer. It adds **image crops of each player cut from the tactical camera** and randomly hides whole crop sequences during pre-training.
- **Results:** ball-state accuracy 85.85 %, possessor accuracy 89.56 %, ball error 1.17 m. Crops alone raise possessor accuracy from 83.66 % to 86.38 %. It still struggles when a window starts or ends without a clear possessor.
- **Code:** none. The data is proprietary (283 matches).
- **Fit:** It shows our **tactical video** carries possession signal that tracking misses. Not reproducible in 2.5 weeks on an M2.

### 1.8 Kim et al. 2026, "PathCRF: ball-free soccer event detection via possession path inference from player trajectories" ★
- **Citation:** Kim H, Lee K, Seo S, Ko S-K, Yoon J, Park C. KDD 2026. <https://arxiv.org/abs/2602.12080>
- **Inputs:** player trajectories only, no ball, downsampled to 5 fps, in 10-second windows. Each player is a node with 8 numbers: 2 indicator flags plus x, y, vx, vy, speed and acceleration. Teams are 11 a side. The goalkeepers are used to order the teams left and right. There are 4 extra "outside" nodes, one per touchline or goal line.
- **How it works:** At each time step the model picks one *edge* of a graph. A self-loop on player *u* means "*u* controls the ball". An edge *u→v* means "the ball is travelling from *u* to *v*". An edge to an outside node means "the ball went out". A **CRF (conditional random field)** is a sequence model, like an HMM, that scores whole label sequences. Here it forbids impossible changes, for example a reception by someone the ball was not travelling to. Viterbi decoding returns the best possession path. The encoder is a set-attention network, so it does not depend on player order within a team.
- **Code and weights:** <https://github.com/hyunsungkim-ds/pathcrf>, **MPL-2.0**. Five trained models are in `saved/100…140` (about 19 MB each, 3.43 M parameters). The best is `saved/140`, "dynamic masked CRF". It was trained on the Sportec open DFL data, with events re-synchronised by ELASTIC. The repo also has loaders for Metrica and TRACAB and a script that adds tracking noise. `tutorial.ipynb` runs inference.
- **Results (1 test match, clean optical data):** **frame-level team possession accuracy 92.40 %**. Possession share 62.64 % predicted against 62.13 % true. Event F1 75.69 % (77.21 % at 25 fps). Pure Viterbi decoding on top of a model without a CRF gave event F1 73.14 %, against 65.34 % with no decoding. Inference took 22.9 s per 90-minute match on an RTX 4090. Training at 5 fps took 565 s per epoch.
- **Limits:** It runs only on **in-play episodes** of at least 10 s, so it does not detect stoppages itself. Episodes with any missing value are skipped. Results come from 1 test match of clean data.
- **Fit:**
  - Works with the ball missing: **yes**, that is its purpose.
  - Needs identities: **no**, but it needs **11 continuous tracks per team over each 10 s window** and a goalkeeper per team. We must pack our 46 slots into 22, fill gaps, and smooth over identity swaps (a swap makes a fake velocity spike).
  - Labels from other games: none needed for inference, because the weights are trained on Bundesliga data. We should still *check* it on degraded other-match data. Domain shift from clean TRACAB to our automatic tracking is the main risk.
  - M2 with 8 GB: inference should fit easily (3.4 M parameters, 5 fps). Retraining at about 10× slower than a 4090 is borderline (estimate, **unverified**).
- **Effort:** 3–5 days. Most of it is the adapter (slot packing, goalkeepers, in-play episodes) plus checking on other matches.

---

## 2. Event and game-state detection (dead ball, set pieces)

### 2.1 Lang et al. 2022, "Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data" ★
- **Citation:** Lang S, Wild R, Isenko A, Link D. *Scientific Reports* 12:16291, 2022. <https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/>
- **Inputs:** **player positions only** (22 players, x/y, velocity, acceleration) at 25 fps. Each frame also gets the same features shifted ±0.4, 0.8, 2, 4, 20, 40 and 80 s, which makes 1,320 inputs.
- **How it works:** A per-frame classifier for in play vs stopped. AdaBoost (a sum of many small decision trees) was best, ahead of random forest, a single decision tree and logistic regression. A 3.64 s median filter smooths the output. Stoppages under 2 s are ignored.
- **Data:** 102 Bundesliga 2017-18 matches labelled by hand (45 train, 10 validation, 47 test).
- **Results:** frame accuracy 92.0 % and F1 93.1 %. Stoppage-level (at least 2 s, matched at IoU ≥ 0.5) precision 81.1 % and recall 78.9 %. Start and end times were within ±2 s for about 80 % of stoppages (mean error 1.7–1.8 s). Features within ±4 s mattered most.
- **Code:** none. The data is not public.
- **Fit:** No ball needed. Its features use fixed player identities, but we can swap in **order-free team features**: mean and median speed, spread, centroid, players near each line, corner or spot. We can train on **DFL `BallStatus`** (7 matches), plus Metrica events and SkillCorner phases, all other matches. Runs on CPU in minutes. Our v1 found only 44 % of dead-ball frames in the public window, so this is the biggest known gap.
- **Effort:** 2–3 days.

Earlier work cited by Lang et al. (not checked by us): Wei et al. classified ball status in 4 s chunks with a decision tree. Gudmundsson & Wolle extracted ball-in and ball-out events from player and ball data.

### 2.2 Sanford et al. 2020, "Group activity detection from trajectory and video data in soccer"
- **Citation:** Sanford R, Gorji S, Hafemann LG, Pourbabaee B, Javan M. CVPR Workshops 2020. <https://arxiv.org/abs/2004.10299>
- **How it works:** Self-attention over the set of players, run on trajectories or on video. Non-maximum suppression handles instantaneous event labels.
- **Results:** most events can be found from either input to within 0.5 s.
- **Code:** none.
- **Fit:** It confirms that trajectories alone are enough for event timing. Not reproducible without their labelled data.

### 2.3 Synchronising events with tracking (only for building labels on other matches)
- **ETSY** (KU Leuven, Apache-2.0), a rule-based event-to-tracking synchroniser: <https://github.com/ML-KULeuven/ETSY>. The paper is **unverified**.
- **ELASTIC** (Kim et al., CIKM 2026, MPL-2.0), trajectory-based synchronisation; PathCRF used it: <https://github.com/hyunsungkim-ds/elastic>.
- **Fit:** useful only if we build H/A/D labels for other matches from events, for example SkillCorner or PFF. The DFL data already has per-frame flags, so we do not need these for it.

### 2.4 Referee whistle from audio
- The classical approach is a band-pass filter plus a threshold. One 2011 paper (Amrita) reports sensitivity 92.5 % and reliability 80.5 % across soccer, football and basketball audio (from a search snippet; **unverified**). <https://www.amrita.edu/publication/automated-referee-whistle-sound-detection-for-extraction-of-highlights-from-sports-video/>
- **PANNs** (Kong et al. 2020), AudioSet-pretrained audio taggers: <https://github.com/qiuqiangkong/audioset_tagging_cnn> (MIT). The label list includes **"Whistle"** (index 402), as well as "Cheering", "Applause" and "Crowd". AudioSet's own quality check says only 6 of 10 sampled "Whistle" clips were correct (<https://research.google.com/audioset/dataset/whistle.html>).
- **Fit:** runs on an M2 (CPU is fine for 2 h of audio). It is a ready-made alternative or cross-check for our v2 whistle detector. Its accuracy on referee whistles in a stadium is **unverified**.
- **Effort:** 0.5–1 day.

---

## 3. When the ball is missing

| Approach | Idea | Fit for us |
|---|---|---|
| **PathCRF** (1.8) | Skip the ball and infer the possession path directly | Best fit. Weights public |
| Ball Radar (1.5), TranSPORTmer (1.6), Peral et al. (1.7) | Infer the ball (or the possessor) from players | Superseded by PathCRF, or no code |
| **Maksai, Wang & Fua 2016**, "What players do with the ball: a physically constrained interaction modeling", CVPR 2016 <https://openaccess.thecvf.com/content_cvpr_2016/html/Maksai_What_Players_Do_CVPR_2016_paper.html> | Tracks the ball together with "who has it" as a mixed-integer program with physical constraints (the ball flies straight or is carried by a player). Tested on soccer, basketball and volleyball | The idea is useful for gap filling: across a gap, the ball is either carried by one player or flying between two. Code **unverified**. Our version would be a small dynamic program, 2–3 days |
| **WASB** (Tarashima et al., BMVC 2023) <https://arxiv.org/abs/2311.05237>, code <https://github.com/nttcom/WASB-SBDT> (MIT, soccer weights on Google Drive) | Re-detect the small ball in video with a high-resolution network and temporal consistency | Could cut the 41 % missing. The training "soccer" set and its similarity to our tactical camera are **unverified**. More than 30 FPS on a GPU. On an M2 it is perhaps many hours for 159k frames (**unverified**). 2–4 days with risk |
| **Smoothing and gap interpolation** (Link & Hoernig used a Rauch–Tung–Striebel smoother, a Kalman smoother that runs forwards and backwards) | Fill short gaps between the last and next ball sighting. Use the interpolated path for the "nearest player" clue | Cheap (half a day). Only good for short gaps. Long gaps are where the ball is hidden in crowds |

---

## 4. Decoding for a segment-level IoU metric

**Key point.** spellF1 is the **segmental F1@k** metric from action segmentation (Lea et al. 2017). A predicted segment is a true positive if its IoU with a true segment passes a threshold. Extra matches to the same true segment are false positives. Our version averages over thresholds 0.3 to 0.7. Lea et al. chose this metric *because* "it penalizes over-segmentation errors". So that field's tools for over-segmentation apply to us directly.

- Lea C, Flynn MD, Vidal R, Reiter A, Hager GD. "Temporal convolutional networks for action segmentation and detection", CVPR 2017. <https://arxiv.org/abs/1611.05267>

### 4.1 What our v1 decider already is
Viterbi with a fixed cost per switch finds the label sequence that minimises "evidence against the labels + penalty × number of switches". That is the same objective as **penalised change-point detection** with a linear penalty (PELT, Killick, Fearnhead & Eckley 2012; we saw this citation in the rupturesRcpp docs but did not read the paper). So v1 is already the exact optimum of a sensible objective. Improvements must change the *objective*: how long segments last, how sure each frame's clue is, and where boundaries sit. A better search will not help.

### 4.2 Hidden semi-Markov models (HSMM): duration-aware decoding ★
- **Citation:** Yu S-Z. "Hidden semi-Markov models", *Artificial Intelligence* 174(2):215–243, 2010. <https://doi.org/10.1016/j.artint.2009.11.011> (PDF: <https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf>)
- **How it works:** In an HMM, the time spent in a state is implicitly geometric. The most likely duration is always 1 frame, which is exactly what produces flicker. An HSMM gives each state an **explicit duration distribution**, for example "an H possession lasts 0.8 s to a few minutes, typically about 10 s". Viterbi then scores whole segments. The general cost is O((M² + MD + MD²)T) for M states, maximum duration D and T frames. A cheaper and common trick is the **expanded-state HMM**. Each label becomes a left-to-right chain of k sub-states (Yu discusses these "phase" chains). This enforces a *minimum* duration of k frames, and the cost stays tiny.
- **Libraries:** `pyhsmm` (MIT, Bayesian, old) <https://github.com/mattjj/pyhsmm>. `hsmmlearn` (GPL-3.0) <https://github.com/jvkersch/hsmmlearn>. `edhsmm` (MIT, small) <https://github.com/poypoyan/edhsmm>. `hmmlearn` (BSD-3) has no durations. Writing it ourselves in numpy is about 100 lines.
- **Fit:** no ball, identities or labels needed. Duration distributions for possessions and dead spans can be **measured on the DFL and Metrica matches** (other games). Fast on an M2: with M = 3 and a minimum-duration chain, T = 159k is trivial. A full segment Viterbi with D ≈ 1,500 frames is about 10⁹ operations, which needs numba or vectorisation.
- **Effort:** 1–2 days.

### 4.3 Semi-Markov CRF
- **Citation:** Sarawagi S, Cohen WW. "Semi-Markov conditional random fields for information extraction", NeurIPS 2004. <https://proceedings.neurips.cc/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html>
- **How it works:** Like an HSMM, but trained discriminatively. Features can describe a whole segment (its length, the mean clue over the segment, whether it starts with a set-piece trigger). It has a semi-Markov Viterbi.
- **Fit:** the right shape for us, but it needs labelled sequences, which we only have from other matches. Training is cheap on CPU. Worth it only after the bench (Section 7, step 0) exists.
- **Effort:** 3–4 days. PathCRF (1.8) is a modern CRF instance with constrained transitions.

### 4.4 Change-point detection
- **PELT** (exact penalised segmentation, about linear time). It is in **ruptures** (BSD-2; Truong, Oudre & Vayatis, *Signal Processing* 167:107299, 2020). <https://github.com/deepcharles/ruptures>, docs <https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/>. `min_size` sets the minimum distance between change points, and `pen` is the switch penalty.
- **BOCPD** (Bayesian online change-point detection, Adams & MacKay 2007). <https://arxiv.org/abs/0710.3742>. It tracks the posterior over "time since the last change".
- **Fit:** both are generic. For labelled states, the HSMM in 4.2 does the same job with our clues as emissions. Use ruptures only as a cross-check on a continuous signal, such as team mean speed for dead-ball boundaries.

### 4.5 Deep temporal action segmentation
| Method | What it adds | Code / licence | Reported (50Salads) |
|---|---|---|---|
| **MS-TCN** (Abu Farha & Gall, CVPR 2019) <https://arxiv.org/abs/1903.01945> | Stacked dilated temporal convolutions, each stage refining the last. A **smoothing loss**: a truncated mean squared error on frame-to-frame changes of the log-probabilities (τ = 4, λ = 0.15) | <https://github.com/yabufarha/ms-tcn>, MIT plus Commons Clause (no selling). MS-TCN++ <https://github.com/sj-li/MS-TCN2> is MIT | F1@50 = 64.5 against 21.5 for one stage. The smoothing loss alone adds about 5 F1 points |
| **ASRF** (Ishikawa et al., WACV 2021) <https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html> | A second head predicts **boundary probability**. At test time, cut only at boundary peaks above 0.5, then **majority-vote** the class inside each piece | <https://github.com/yiskw713/asrf>, MIT | F1@50 +10.6, Edit +8.3 over MS-TCN variants |
| **ASFormer** (Yi et al., BMVC 2021) <https://arxiv.org/abs/2110.08568> | Transformer backbone | <https://github.com/ChinaYi/ASFormer>, MIT | — |
| **DiffAct** (Liu et al., ICCV 2023) <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Diffusion_Action_Segmentation_ICCV_2023_paper.html> | Diffusion model that iteratively denoises the label sequence | <https://github.com/Finspire13/DiffAct> | F1@50 = 83.7 |
| Survey: Ding, Sener & Yao, "Temporal action segmentation: an analysis of modern techniques", TPAMI 2024 | Overview | <https://arxiv.org/abs/2210.10352> | — |

- **Fit:** These models are built for video features, but they work on any per-frame feature vector, including our clue values and team aggregates. They are small enough to train on CPU at a few fps. The catch is training data: 7–10 other matches is little. Two parts are cheap and transferable without deep learning:
  - (a) the **smoothing-loss idea**, which we already approximate with switch costs;
  - (b) **ASRF's refinement**: find boundary candidates from a separate "something changed here" score (whistle, speed drop, ball deflection near an opponent), then vote the label inside each piece. It is a 1-day post-processor.
- **Effort:** ASRF-style refinement 1 day. A full MS-TCN on bench features 3–5 days.

---

## 5. SoccerNet and other video challenges

| Task | Best result | Code / weights | Fit |
|---|---|---|---|
| **Ball Action Spotting 2023** (pass, drive; tolerance 1 s) | 1st place lRomul: 86.47 mAP@1 (mean average precision) on the challenge set | <https://github.com/lRomul/ball-action-spotting> MIT. Pipeline tuned for an RTX 3090 | Broadcast video. Needs an NVIDIA GPU to train |
| **Ball Action Spotting 2024** (12 classes incl. Out, Throw In, Free Kick, Goal; 7+2 EFL games) | T-DEED 73.39 mAP@1 (baseline 56.15) | <https://github.com/arturxe2/T-DEED> GPL-3.0. SoccerNetBall checkpoints on Google Drive, with `inference.py` (796×448 input) | Inference on our videos is possible. But our broadcast video is not synced, and the tactical wide view is a different domain from broadcast. Speed on an M2 is **unverified**. Events like "Out" and "Throw In" would only add dead-ball clues |
| **Team Ball Action Spotting 2025** (same 12 classes + left/right team) | dudek 60.03 Team-mAP@1. Baseline (T-DEED) 51.72 | Winner code **unverified** | Gives team-aware touches from video. Same domain and compute problems |
| **Game State Reconstruction** (players on a minimap with team, role, jersey) | 2025 winner KIST-GSR 63.90 GS-HOTA. Baseline 29.01 | <https://github.com/SoccerNet/sn-gamestate> GPL-3.0 | Solves tracking and identity, which we already have from Metrica. Not possession. Low fit |
| **FOOTPASS** (Ochin et al. 2025), play-by-play on-ball actions with tactical context | Dataset paper <https://arxiv.org/abs/2511.16183> | Annotations to Hugging Face. Videos under the SoccerNet NDA | Low fit |
| **DFL Bundesliga Data Shootout** (Kaggle 2022): play, challenge and throw-in in Bundesliga videos | Top solutions linked from the lRomul README (Team Hydrogen and others) | Data licence **unverified** | Closest *video* domain if those videos are wide tactical views (**unverified**). Too heavy for our timeline |

---

## 6. Fit at a glance

| Method | Works with ball missing? | Needs player identity? | Needs labels (other matches)? | M2 / 8 GB? | Public code | Effort |
|---|---|---|---|---|---|---|
| HSMM / min-duration Viterbi (4.2) | yes (decoder) | no | duration stats only | yes, trivial | write ourselves; libraries MIT/GPL | 1–2 d |
| Lang et al. dead-ball classifier (2.1) | yes | no (use team aggregates) | yes, DFL `BallStatus` | yes | no (sklearn) | 2–3 d |
| Vidal-Codina set-piece triggers (1.2) | yes | no | thresholds only | yes | databallpy (possession only) | 1–2 d |
| PathCRF (1.8) | **yes** | no, but needs 11 tracks a side + goalkeepers | pretrained (DFL) | inference yes | **yes, MPL-2.0, weights** | 3–5 d |
| Link & Hoernig / Vidal-Codina touch rule (1.1–1.2) | no (visible frames only) | no | thresholds only | yes | databallpy MIT | 1–2 d |
| ASRF-style boundary refinement (4.5) | yes | no | optional | yes | idea | 1 d |
| MS-TCN on bench features (4.5) | yes | no | yes | yes, small | MIT (+CC) | 3–5 d |
| Semi-Markov CRF (4.3) | yes | no | yes | yes | write ourselves | 3–4 d |
| Ball Radar (1.5) | yes | no | pretrained? no | yes | no licence | — |
| TranSPORTmer / Peral et al. (1.6–1.7) | yes | no | private data | — | no | — |
| WASB ball re-detection (3) | fills it | no | pretrained | slow | MIT, weights | 2–4 d, risky |
| PANNs whistle (2.4) | n/a | no | pretrained | yes | MIT, weights | 0.5–1 d |
| T-DEED / SoccerNet BAS (5) | n/a (video) | no | pretrained | slow, **unverified** | GPL-3.0, weights | 4+ d, risky |

---

## 7. Ranked shortlist for us

**Step 0, which everything below depends on: a test bench built from other matches (1.5–2 days).** We may not tune on this match. We need somewhere to *measure*.
1. Load the 7 Sportec DFL matches with kloppy, one at a time, and save them compactly (parquet).
2. Turn `BallPossession` and `BallStatus` into H/A/D per frame.
3. Write a **degrader** that makes clean data look like ours: drop the ball on about 41 % of frames, mostly when many players are near it or it is fast; scramble slot identities at random moments; add identity swaps; recompute speeds.
4. Score with the official spellF1 scorer.
5. Optionally add the 3 Metrica sample games (closest label convention) as a second bench.

Caveat: DFL's "player in control" possession and Metrica's corrected-tracking possession may differ on contested balls (**unverified**). Treat bench numbers as a *ranking* of ideas, not a score forecast.

1. **Duration-aware decoder (HSMM / minimum-duration Viterbi).** It targets the metric directly. It needs nothing new except duration statistics.
   *First step:* on the bench, measure the length distributions of true H/A possessions and D spans. Replace each label in our Viterbi with a chain of k sub-states, for example k = 10 frames (0.4 s, half the shortest real possession). Compare spellF1 with v1 on degraded bench matches.
2. **Learned dead-ball detector (Lang et al. 2022), plus Vidal-Codina set-piece triggers for "who restarts".** Each missed stoppage merges two possessions, and v1 caught only 44 % of D.
   *First step:* compute order-free team features per frame (team mean and median speed, spread, centroid, players beyond each line or near a corner, spot or centre circle). Shift them ±0.4…±80 s as in the paper. Train an AdaBoost or gradient-boosted classifier on 5 DFL matches and test on 2. Feed P(dead) to the decoder as a clue.
3. **PathCRF as the possession clue on ball-missing frames.** It is the only public, pretrained, ball-free possession model.
   *First step:* clone the repo and run `tutorial.ipynb` on one DFL match on the M2 to confirm it runs and how fast. Then write the adapter: pick 11 active slots per team per frame and hold the last position over gaps; find goalkeepers; smooth, then compute vx, vy, speed and accel; cut in-play episodes from our D decision. Run `saved/140`. Turn edges into teams (a pass in flight stays with the sender's team until reception). Upsample 5 → 25 fps. Check it on the degraded bench *before* trusting it on our match.
4. **Touch-based ball-control rule (Link & Hoernig, Vidal-Codina)** for visible-ball frames. "Nearest player" credits players the ball flies past. A touch needs the ball to change direction or speed, or to show an acceleration peak, inside the player's zone.
   *First step:* start from databallpy's `add_individual_player_possession` (MIT). Add the ≥ 4 m/s² kick-detection test. Choose radius and thresholds on the bench, then swap it in for clue A.
5. **Learned per-frame scores replacing hand weights**, optionally with MS-TCN-style smoothing or ASRF-style boundary refinement.
   *First step:* on the bench, train a gradient-boosted H/A/D classifier on our existing clues plus items 2–4. Feed its log-probabilities to the item-1 decoder in place of the hand-set costs. Keep the hand-set version as a fallback if the bench shows no gain.

Not shortlisted, but worth one evening each if time remains: PANNs "Whistle" as a second opinion for v2; WASB ball re-detection on a 5-minute *non-graded* stretch (before kickoff or at halftime) to judge speed and quality.

---

## Sources

Possession and events from tracking
- Link & Hoernig 2017: <https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/>
- Vidal-Codina et al. 2022: <https://arxiv.org/abs/2202.00804>, PDF <https://arxiv.org/pdf/2202.00804v3>, <https://dspace.mit.edu/handle/1721.1/145347>
- databallpy possession docs: <https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html>, repo <https://github.com/Alek050/databallpy>
- Khaustov & Mozgovoy 2020: <https://www.mdpi.com/2076-3417/10/22/8046>
- Morra et al. 2020: <https://arxiv.org/abs/2004.04147>
- Wei et al. 2015: <https://openaccess.thecvf.com/content_iccv_2015_workshops/w21/html/Wei_Predicting_Ball_Ownership_ICCV_2015_paper.html>
- Kim et al. 2023 (Ball Radar): <https://arxiv.org/abs/2306.08206>, <https://arxiv.org/html/2306.08206v1>, <https://github.com/hyunsungkim-ds/ballradar>
- Capellera et al. 2024 (TranSPORTmer): <https://arxiv.org/abs/2410.17785>, <https://arxiv.org/html/2410.17785v2>, <https://openaccess.thecvf.com/content/ACCV2024/html/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.html>
- Peral et al. 2025: <https://arxiv.org/abs/2512.19528>, <https://arxiv.org/html/2512.19528v1>
- Kim et al. 2026 (PathCRF): <https://arxiv.org/abs/2602.12080>, <https://arxiv.org/html/2602.12080v2>, <https://github.com/hyunsungkim-ds/pathcrf>
- Lang et al. 2022: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/>
- Sanford et al. 2020: <https://arxiv.org/abs/2004.10299>
- ETSY: <https://github.com/ML-KULeuven/ETSY>
- ELASTIC: <https://github.com/hyunsungkim-ds/elastic>
- Maksai, Wang & Fua 2016: <https://openaccess.thecvf.com/content_cvpr_2016/html/Maksai_What_Players_Do_CVPR_2016_paper.html>

Open data and tools
- Bassek et al. 2025 (Sportec Open DFL): <https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/>, figshare <https://doi.org/10.6084/m9.figshare.28196177>, mirror <https://huggingface.co/datasets/pysport/idsse-data>
- Metrica sample data: <https://github.com/metrica-sports/sample-data>
- SkillCorner open data: <https://github.com/SkillCorner/opendata>
- PFF FC / Gradient World Cup 2022: <https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset>
- kloppy: <https://github.com/PySport/kloppy> (Sportec loader: `kloppy/infra/serializers/tracking/sportec/deserializer.py`)
- floodlight: <https://github.com/floodlight-sports/floodlight>

Audio and ball detection
- Amrita whistle detection: <https://www.amrita.edu/publication/automated-referee-whistle-sound-detection-for-extraction-of-highlights-from-sports-video/>
- PANNs: <https://arxiv.org/abs/1912.10211>, <https://github.com/qiuqiangkong/audioset_tagging_cnn>
- AudioSet "Whistle": <https://research.google.com/audioset/dataset/whistle.html>
- WASB: <https://arxiv.org/abs/2311.05237>, <https://github.com/nttcom/WASB-SBDT>

Segmentation and decoding
- Lea et al. 2017: <https://arxiv.org/abs/1611.05267>, <https://openaccess.thecvf.com/content_cvpr_2017/html/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.html>
- Yu 2010 (HSMM review): <https://doi.org/10.1016/j.artint.2009.11.011>, <https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf>
- pyhsmm: <https://github.com/mattjj/pyhsmm>. hsmmlearn: <https://github.com/jvkersch/hsmmlearn>. edhsmm: <https://github.com/poypoyan/edhsmm>. hmmlearn: <https://github.com/hmmlearn/hmmlearn>
- Sarawagi & Cohen 2004: <https://proceedings.neurips.cc/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html>
- Truong, Oudre & Vayatis 2020 (ruptures): <https://arxiv.org/abs/1801.00718>, <https://github.com/deepcharles/ruptures>, <https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/>
- Adams & MacKay 2007 (BOCPD): <https://arxiv.org/abs/0710.3742>
- MS-TCN: <https://arxiv.org/abs/1903.01945>, <https://ar5iv.labs.arxiv.org/html/1903.01945>, <https://github.com/yabufarha/ms-tcn>, MS-TCN++ <https://github.com/sj-li/MS-TCN2>
- ASRF: <https://openaccess.thecvf.com/content/WACV2021/html/Ishikawa_Alleviating_Over-Segmentation_Errors_by_Detecting_Action_Boundaries_WACV_2021_paper.html>, <https://ar5iv.labs.arxiv.org/html/2007.06866>, <https://github.com/yiskw713/asrf>
- ASFormer: <https://arxiv.org/abs/2110.08568>, <https://github.com/ChinaYi/ASFormer>
- DiffAct: <https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Diffusion_Action_Segmentation_ICCV_2023_paper.html>, <https://github.com/Finspire13/DiffAct>
- Ding, Sener & Yao (survey): <https://arxiv.org/abs/2210.10352>

SoccerNet and video challenges
- SoccerNet 2024 results: <https://arxiv.org/abs/2409.10587>
- SoccerNet 2025 results: <https://arxiv.org/abs/2508.19182>, <https://arxiv.org/html/2508.19182v1>
- T-DEED: <https://arxiv.org/abs/2404.05392>, <https://github.com/arturxe2/T-DEED>
- 2023 BAS winner: <https://github.com/lRomul/ball-action-spotting>
- Game State Reconstruction: <https://arxiv.org/abs/2404.11335>, <https://github.com/SoccerNet/sn-gamestate>
- FOOTPASS: <https://arxiv.org/abs/2511.16183>
- DFL Bundesliga Data Shootout: <https://bundesliga.com/en/aws-bundesliga-data-shootout>
