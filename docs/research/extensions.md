# Research: extra signals beyond the provided tracking

2026-10-30 · scope: BAR-ATH possession labels (H / A / D / X), Apple M2 with 8 GB RAM, about 2.5 weeks left.

## Summary

1. **Cheapest big win: the broadcast recording's own tracking.** It has a ball on 50.5% of its frames, and its homography is missing exactly when the TV shows replays, close-ups and crowd. Our existing alignment tool can carry both onto the tactical timeline. First step: measure how many tactical ball gaps it fills.
2. **Best new model: PathCRF (2026).** It infers who has the ball from player movement alone, and its weights were trained on 7 public Bundesliga matches. It reached 92.4% frame-level team-possession accuracy in play on its test match. The weights are 19 MB, MPL-2.0 licensed, and run in minutes on a CPU.
3. **Best validation set: IDSSE (CC BY 4.0).** It gives per-frame `BallPossession` and `BallStatus` at 25 Hz for 7 matches. If we degrade it to look like our tracking and run the official scorer on it, we can set weights fairly instead of by guesswork.
4. **Ball detection in the tactical video is possible on the M2 but weak.** The ball is about 3–4 px wide in a 4.9 Mbit/s 1080p stream, and most gaps are occlusions. Expect modest gains for 5–20 h of compute. A per-frame ball track is also probably not committable.
5. **Low value:** pretrained audio models add little over our tonal whistle detector, and simulators are not worth the time. **Needs an organiser answer first:** self-training on this match and commentary transcription.

---

## How to read this

- **Rule status** uses three values. *Allowed* means the rules or an organiser answer cover it. *Grey* means ask first. *Out* means it has been ruled out.
- **Runtime** figures are my estimates for an M2 with 8 GB RAM, not measurements, unless a source is cited. Before committing to any stage, run a 10-minute benchmark on 1,000 frames.
- **Unverified** marks anything I could not confirm on a primary page.
- The sibling note [models-literature.md](models-literature.md) covers possession models from tracking in more depth (Link & Hoernig, Lang et al. on in-game status, HSMM decoding). Here I only cover them where they add a new signal.

### Words used

- **Homography:** a 3×3 matrix that maps pitch coordinates to image pixels for one frame.
- **Viterbi:** an algorithm that finds the single most likely label sequence, given per-frame evidence and a cost for each switch.
- **HMM (hidden Markov model):** a model of hidden states (here H/A/D) that emit observations (the clues).
- **HSMM (hidden semi-Markov model):** an HMM that also models how long each state lasts.
- **EM / Baum-Welch:** fitting an HMM's numbers by repeatedly guessing the hidden states, then re-estimating the numbers, without labels.
- **SED (sound event detection):** finding when a sound happens, not just whether a clip contains it.
- **mAP / F1 / PSDS:** standard accuracy scores. Higher is better.
- **Tile / slice:** cutting a big frame into smaller crops so a small object looks bigger to the detector.
- **Domain shift:** a model trained on one kind of footage or tracking doing worse on another.
- **MPS / CoreML / ONNX Runtime:** ways to run neural networks on a Mac's GPU or neural engine (MPS, CoreML) or a portable CPU runtime (ONNX).
- **Pseudo-label:** a label produced by an algorithm, not a person.

### Facts used for the estimates (from our repo's DATA.md, not from the data folder)

- Tactical video: 199,319 frames, 25 fps, 1920×1080, 4.9 GB, so about **4.9 Mbit/s**. That is a low bitrate for a whole-pitch view.
- Graded frames: 158,857. The ball is missing on 41% of them, so **about 64,700 graded frames have no ball**.
- If the pitch spans most of the 1920 px width, 105 m is about 18 px/m, so the 22 cm ball is about **4 px at best**, and smaller on the far side.
- Broadcast: 164,488 frames. Its tracking has the ball on **50.5%** of its frames. Its homography is valid on **65.8%**, and the gaps come in 122 stretches of replays, close-ups and crowd shots.

### A rule catch that affects several ideas

RULES.md forbids publishing "derived data that would reconstruct" the match data, and names frame-level tracking exports as an example. The organisers allow committing *small* outputs of slow stages (event times and scores, #14). **A per-frame ball track from our own detector looks like a frame-level tracking export**, so we should not commit it. Two options: make the detector stage run deterministically on CPU at regeneration time (runtime is unlimited), or commit only event-level summaries such as "ball seen near team X at frame N". If we want to commit a per-frame track, we should ask first.

---

## 1. Ball detection in the tactical video

**Signal.** Fill the 41% ball gaps directly, which feeds the nearest-player clue we already have. The realistic gain comes from balls **in flight** (long passes, clearances, crosses), because they move against a static background. The ball **inside a crowd** is hidden by players, and no detector will see it.

### Candidates

| tool | what it is | training data (public?) | licence | size | notes |
|---|---|---|---|---|---|
| **Classical: background subtraction + blob filter** (OpenCV MOG2 or frame differencing, plus a physics-checked tracker) | No learning. Moving pixels minus projected player boxes leaves ball candidates. | none | OpenCV Apache-2.0 | – | Works only if the tactical camera is fixed. Check whether the per-frame homographies are near-constant. No training data, so it raises no rule question (our limitations.md says the same). |
| [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (BMVC 2023) | Small HRNet-style heatmap net. 3 stacked frames in, 288×512 input, **~1.5M params**. | Soccer weights trained on the **ISSIA-CNR / F.I.G.C.-C.N.R. dataset**: 6 fixed Full-HD cameras, 25 fps, one match, ball re-annotated by NTT. Train on clips ID-1..4 (11,994 frames), test on ID-5..6 (5,999 frames). | Code **MIT**. The dataset's terms are "non-commercial research in video signal processing only". | weights on Google Drive (size not listed) | Soccer F1 88.3 (step 3) and AP 86.2 (step 1), against TrackNetV2's F1 86.6, at a 4 px tolerance. FPS of 55.7 / 23.6 measured on a V100 server. The same model zoo also ships TrackNetV2, ResTrackNetV2, MonoTrack, DeepBall and BallSeg soccer weights trained on the same data. |
| [FootAndBall](https://github.com/jac99/FootAndBall) (2020) | Light fully convolutional ball and player detector. Accepts any input resolution. | ISSIA-CNR cameras 1–4 + SoccerPlayerDetection_bmvc17 set 1 (public) | Code **MIT**. Same ISSIA terms apply. | weights `model_20201019_1416_final.pth` in the repo | Claims real time on high-resolution input (no hardware given). Param count not stated ("two orders of magnitude" fewer than SSD/YOLO). |
| [roboflow/sports](https://github.com/roboflow/sports) `football-ball-detection.pt` | YOLOv8x trained for 50 epochs. Reported mAP@0.5 0.925 ([blog](https://blog.roboflow.com/tracking-ball-sports-computer-vision/)). Uses slicing (`InferenceSlicer`). | Frames from the **DFL Bundesliga Data Shootout** (Kaggle 2022), annotated on Roboflow | Repo MIT. Ultralytics is **AGPL-3.0**. DFL data terms **unverified** (the Kaggle rules page did not render, and the Roboflow dataset page returned 403). | YOLOv8x is about 68M params (well-known figure, not re-checked) | Trained on broadcast-style clips (**unverified**: I did not inspect the frames), where the ball is bigger than in our whole-pitch view. |
| Ultralytics YOLO11 pretrained on COCO, class "sports ball" | Generic detector | COCO (public) | AGPL-3.0 | YOLO11n 2.6M … YOLO11x 56.9M params | Zero licence doubt, but COCO balls are large and close up. Useful only as a baseline. |
| SoccerNet-trained ball detectors | e.g. a fine-tuned YOLOv8l in a [2023 tracking challenge report](https://arxiv.org/pdf/2308.16651) | SoccerNet-Tracking: broadcast clips with ~215k ball boxes | Videos sit behind the **SoccerNet NDA** ([soccer-net.org/data](https://www.soccer-net.org/data)) | – | I found no released ball weights. Weights trained on NDA video are grey. |
| TrackNet V3/V4 | Ball heatmap nets | badminton / tennis | various | – | Wrong sport, and WASB already beats TrackNetV2 on soccer. Skip. |

Datasets with **ball labels in a fixed, wide view** (for fine-tuning ourselves):

- **SoccerTrack v1** (CVPRW 2022): 8K fisheye wide view at 30 fps plus 4K drone video, with ball bounding boxes for the cameras (per the paper's search summary). Code GPL-3.0. Dataset licence **unverified**. Kaggle `atomscott/soccertrack`.
- **SoccerTrack v2** (2025): 10 matches of 4K panoramic full-pitch video, but **no per-frame ball positions**. It has 12-class ball-action timestamps instead. Licence **unverified**.

I found **no public benchmark of tiny-ball detection in fixed full-pitch 1080p video**, which is our exact setting.

### Runtime on the M2 (estimates)

- **Classical:** MOG2 on 1080p runs at roughly 100+ fps on CPU, so the whole match takes about 1 hour. It is deterministic.
- **WASB:** the downscaled full frame is useless, because the ball becomes under 1 px. We need native-resolution 512×288 tiles: 16 per frame, or 2–4 tiles around the last and next known ball position. A V100 runs about 18 forward passes/s (55.7 fps over 3 frames). An M2 GPU has about a quarter of that compute, so about 4–5 passes/s on MPS and perhaps 1–2/s on CPU. For about 64,700 gap frames with step 3: **4 tiles take about 5 h on MPS and 12–24 h on CPU. All 16 tiles take about 20 h on MPS.**
- **YOLO11s with 8 slices of 640 px:** about 90 ms per slice on CPU with ONNX (Ultralytics' CPU figure for YOLO11s is 90.0 ms), so about **13 h on CPU**. CoreML might be 5–10× faster (**unverified**: the only Apple figure I found is YOLO11n at 256 px on an M1, 7–8 ms median, [nativeORT vignette](https://cran.r-project.org/web/packages/nativeORT/vignettes/BenchmarkPerformance.html)).
- **YOLOv8x (Roboflow) with slices:** likely several hundred ms per slice, so more than 1 day. Too slow to iterate on.

### Main risks

1. The ball is about 3–4 px and compressed at 4.9 Mbit/s. Fast balls smear, and the far touchline is worse.
2. Every pretrained model was trained on closer views (ISSIA half-pitch cameras, broadcast), so expect a large domain shift. WASB's test split is the *same match* as its training split, so its 88% F1 says nothing about a new stadium.
3. The gaps are mostly crowds, where the ball is invisible to any method.
4. False positives (heads, socks, pitch markings) can do more harm than missing balls. Gate candidates by physics (continuity, speed under about 35 m/s) and by agreement with the tracking.

### Rule check

- **Classical, no training:** allowed.
- **Pretrained WASB / FootAndBall:** allowed by the competition rules (trained on other matches, #2). But the *dataset licence* ("non-commercial research") is **grey** for a sponsor-run hackathon, so ask.
- **Roboflow / DFL weights:** grey (DFL data terms unverified). Ultralytics AGPL is fine for a public repo.
- **Training a detector on this match with the provided tracking as labels:** **out** (#2).
- **Committing the per-frame output:** grey (see the rule catch above).

**Verdict:** start classical and gated by physics. Add WASB tiles only if the classical method recovers in-flight balls well. Budget 16–30 h of work plus overnight runs.

---

## 2. Audio: pretrained sound-event models

**Signal.**

- A referee's whistle marks dead ball (stoppage) and often restarts.
- Crowd roars and boos follow shots, fouls and goals, so a dead ball tends to follow them.
- Ball kicks would give touch times, which mark possession boundaries.

### What the models can hear

- **AudioSet classes we care about** (YAMNet's 521-class map and the AudioSet-Strong 456-class list, both downloaded and checked):
  - `Whistle` (/m/0l156k, which sits under *Alarm*, with children *Kettle whistle* and *Steam whistle*)
  - `Whistling` (human), `Wolf-whistling` (Strong only), `Cheering`, `Crowd`, `Roar`, `Booing` (Strong), `Chant`, `Applause`, `Clapping`, `Thump, thud`, `Air horn`
- **There is no `Kick` or `Ball` class** except *Basketball bounce*. **There is no `Referee whistle` class.**
- **How good is `Whistle`?** AudioSet lists 663 annotations for it. The ontology page shows no quality score and does not mention sport or referee whistles. (*Steam whistle* is rated 78% accurate.) I could not find a per-class AP for `Whistle` for any model (**unverified**). Expect it to fire on clear whistles. It is unproven on a distant referee whistle under stadium noise.
- **A useful distinction:** `Whistle` (an instrument) versus `Whistling` / `Wolf-whistling` (human). Spanish crowds whistle in protest, and our tonal detector may confuse that with the referee. A pretrained model may separate the two (**unverified**).

### Models

| model | licence | params / size | time resolution | AudioSet score | notes |
|---|---|---|---|---|---|
| [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) | Apache-2.0 (tensorflow/models) | 3.7M, 69.2M multiplies per frame | 0.96 s window, 0.48 s hop | balanced mAP 0.306 | Needs Keras 2, not Keras 3. Coarse timing. |
| [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) CNN14 / **Cnn14_DecisionLevelMax** | Code MIT. Weights CC BY 4.0 ([Zenodo](https://zenodo.org/records/3987831)) | about 80M; checkpoint **327 MB** (MobileNetV2 is 20.8 MB) | DecisionLevelMax gives frame-wise output | mAP 0.431 (tagging), 0.385 (SED model) | The checkpoint exceeds 50 MB, so I did not download it. |
| [EfficientAT](https://github.com/fschmid56/EfficientAT) mn10_as / mn04_as | MIT | 4.88M / 0.98M | clip-level | mAP 47.1 / 43.2 | Fast on CPU. Clip-level only. |
| [PretrainedSED](https://github.com/fschmid56/PretrainedSED) (ICASSP 2025): BEATs, ATST-F, fPaSST, M2D, ASiT "Strong", **frame_mn10 / frame_mn06** | MIT | frame_mn10 3.83M, frame_mn06 1.62M; transformers about 90M | frame-level, trained on **AudioSet Strong (447 classes)**; 40 ms pseudo-label grid | PSDS1 about 0.45–0.47 (transformers) | The best fit for *timing* a whistle. AudioSet-Strong includes `Whistle`. |
| [AST](https://github.com/YuanGongND/ast) | BSD-3-Clause | about 87M | clip-level | (not re-checked) | Superseded by the above. |
| [BEATs](https://github.com/microsoft/unilm) | MIT | about 90M | clip-level (frame-level via PretrainedSED) | (not re-checked) | Use the PretrainedSED version. |
| [LAION-CLAP](https://github.com/LAION-AI/CLAP) / [MS-CLAP](https://github.com/microsoft/CLAP) | CC0-1.0 / MIT (MS-CLAP archived) | not stated | clip-level, zero-shot from text prompts | – | It is an audio–text embedding model, not a generative LLM. Text-prompted labelling sits close to our "no LLM/VLM labeller" decision in spirit, though. Low value anyway. |

### What the literature says about whistles

The classic detector is the one we already built:

- Band energy in **3.5–4.5 kHz** (Goertzel algorithm, 48 kHz audio, 4,800-sample frames).
- An entropy test for tonality.
- Keep frames with 2–3 spectral peaks.
- Result: **recall 93%, precision 88%** on annotated broadcast matches, with vuvuzelas, horns and supporters' whistles named as the false-alarm sources ([arXiv 1411.6496](https://arxiv.org/abs/1411.6496)).
- Audio *alone* is weak for soccer events, but helps when fused with video. One study gained 7.43 points of action-classification accuracy on SoccerNet by adding audio ([survey](https://arxiv.org/pdf/2309.12067), [Vanderplaetse & Dupont](https://arxiv.org/pdf/2011.04258)).
- Crowd loudness (RMS) rises around important events, and more so for home-team events ([MDPI 2021](https://www2.mdpi.com/2504-4990/3/4/51)). That is a weak team hint.

### Ball kicks

No pretrained class exists. A kick would need onset detection (a sudden rise in spectral energy) and a pitch-side mic. Broadcasters often place effects mics around the pitch, which a tactical camera mic probably lacks (**unverified** for this match). With commentary on top, the broadcast is also hard to use for this. Low value.

### Runtime (estimates)

The whole 2 h 13 min of audio:

- YAMNet: about 1 min on CPU.
- CNN14: a few minutes.
- frame_mn10: a few minutes.
- A BEATs-Strong transformer: about 10–30 min on CPU.

All are cheap. Decode with PyAV as v2 does, and commit the event times and scores (#14).

### Main risk

It mostly duplicates the v2 whistle clue, and correlated errors (both detectors fooled by fans whistling) give a false sense of confidence.

### Rule check

Allowed. Models trained on AudioSet are pretrained on other data, and committing their outputs is allowed (#14). Any threshold must be set from the audio's own distribution or from other matches, never from the public 5 minutes.

**Verdict:** a cheap second opinion. Use frame_mn10 or BEATs-Strong's `Whistle` vs `Whistling` posterior to veto v2 false alarms, and `Cheering`/`Roar`/`Booing` as a weak "stoppage likely soon" clue. 4–8 h.

---

## 3. The broadcast video as a dead-ball signal

**Why it correlates with stoppages.** The TV director shows the live wide camera while the ball is in play. Replays, close-ups of players and coaches, crowd shots and graphics (substitutions, VAR, stats) are cut in **when the ball is dead**, because nothing would be missed. The cut back to the wide camera usually comes just before a restart. So the broadcast's shot structure is a free, independent dead-ball detector. Its timing is good for the *end* of a stoppage and late by a second or two for the *start*. This is a hypothesis built from production practice; I found no paper measuring it.

### What we already have for free

The broadcast homography is missing on 34.2% of frames, in 122 stretches, which DATA.md calls "replays, close-ups and crowd shots". **That flag is already a shot-type classifier**, and needs no model. Map it through our alignment tool. The main thing a model adds is telling **replays** (always dead ball, and showing the past) from **live close-ups**.

### Tools

| tool | what it adds | licence | notes |
|---|---|---|---|
| [TransNetV2](https://github.com/soCzech/TransNetV2) | Shot-cut detection. Input frames resized to **48×27**. F1 77.9 (ClipShots), 96.2 (BBC), 93.9 (RAI). | MIT. Weights in git-lfs. | The original needs TF 2.1, and a PyTorch inference version exists in the repo. I found no speed figure in the paper. The tiny input makes it fast, so video decoding is the bottleneck: about 164k frames, roughly 10–20 min on an M2 (estimate). |
| Replay logo detection (classical template matching) | Replays open and close with a logo wipe. A Spanish/Catalan broadcast study found the logo frames by luminance peaks plus k-means templates, with replay **recall 99.12%, precision 100%** ([arXiv 1411.6496](https://arxiv.org/abs/1411.6496)). | – | We need one example of the LaLiga replay logo. Picking a template from the video is "watching to design a feature", not labelling. Still, keep the template selection algorithmic or documented. |
| Grass-ratio shot type (Ekin, Tekalp & Mehrotra 2003) | Share of dominant pitch colour → long shot / medium / close-up / out-of-field. | – | Classic and cheap ([record](https://research.hub.ku.edu.tr/entities/publication/0ebd0d09-be01-4093-b04e-e07e2f94f805)). Mostly redundant with the homography flag. |
| Graphics overlays | Pixel change in the fixed regions where banners appear | – | A weak cue, cheap. |
| Scoreboard clock OCR (Apple Vision `VNRecognizeTextRequest`, Tesseract) | Match time per broadcast frame | Vision is local on macOS, Tesseract is Apache-2.0 | **The football clock does not stop at stoppages**, so OCR does not detect dead balls. It only helps alignment, which we have already solved. Low value. |
| SoccerNet-v2 camera-shot models | 13 camera classes, replay grounding | NDA video | Grey, and heavy. Skip. |

### Main risks

1. Alignment errors across cuts and replays: replays break the monotonic time map.
2. The director cuts late. A stoppage often stays on the wide shot for 1–3 s.
3. Close-ups during live play (for example after a near miss) look like dead ball.

So use the cue to **confirm** a dead stretch and to time its **end**, not to start one.

### Rule check

Allowed. The broadcast is a provided asset, and the FAQ says to use it. Commentary speech-to-text is a separate question (grey G2 in limitations.md).

**Verdict:** high value per hour if built on the existing homography flag plus a replay detector. 6–10 h.

---

## 4. Ball and ball-carrier imputation from player movement

**Signal.** When the ball is missing, players still move *because of* the ball. Defenders converge on the carrier, and team-mates offer passing lanes. A model can learn which team, and roughly which player, has the ball.

| model | input | weights / data | licence | reported accuracy | fit |
|---|---|---|---|---|---|
| **[PathCRF](https://github.com/hyunsungkim-ds/pathcrf)** (Kim et al., [arXiv 2602.12080](https://arxiv.org/abs/2602.12080), 2026) | Player tracks only. 11 + 11 players. Downsampled to **5 fps**, 10 s windows. A Set Transformer feeds a CRF (conditional random field: a model that scores whole label sequences and forbids impossible jumps), decoded with **Viterbi**. Four "out" nodes for the ball leaving each side. | **Pretrained weights in the repo** (`saved/100…140`, **about 19 MB each**, 3.43M params), trained on the **Sportec Open DFL dataset = IDSSE** (7 Bundesliga 1/2 matches) with leave-one-match-out folds | **MPL-2.0** | Test match: frame-level **team-possession accuracy 92.40%**, event F1 75.69% (77.21% at 25 fps), sender 80.4% / receiver 82.3% | **Best fit.** It produces our label's "who" from what we have on every frame. The repo also ships Metrica and TRACAB loaders and a noise generator (`preprocess_noise.py`: occlusion merges, bias, outliers). |
| [Ball Radar](https://github.com/pientist/ballradar) (Kim et al., KDD 2023, [arXiv 2306.08206](https://arxiv.org/abs/2306.08206)) | Player tracks at 10 Hz, 10 s windows. Predicts the possessor, then the ball path. | **No pretrained weights**. Trained on Fitogether GPS data (private, 15 matches) + Metrica samples (3 matches, public). | **No licence file**, so all rights are reserved by default. | Ball error 3.66 m, **team-possession accuracy 85.85%**, player-level 64.70% | Superseded by PathCRF from the same group. The code is not licensed for reuse. |
| Simple convergence heuristic (ours) | For each team, the least-squares point where players' velocity lines meet, and how many opponents run toward a team's cluster | none | – | unknown; measure on IDSSE | A few hours. Explainable. A good cross-check feature. |

### Runtime

The paper reports 22.88 s for a whole match on an RTX 4090. On an M2 CPU, expect about 5–20 min (estimate). Memory is small.

### Main risks

1. **Domain shift.** The model expects 22 clean TRACAB tracks with stable identities. Ours are anonymous slots that appear, vanish and swap, with 20.5 objects on average. We must map tracks into 11 slots per team per window (Hungarian matching on position, padding missing players). Swaps inside a 10 s window will hurt.
2. It is trained **in play only**, so it says nothing about dead ball. Our D clues stay in charge of D.
3. 92.4% is on one DFL test match with clean data. Expect lower on ours. Validate on degraded IDSSE (idea 7) before trusting it.

### How to fuse it

Use its per-frame team probability as one more emission (evidence term) in our Viterbi decider. Weight it up on frames where the ball is missing and down where the ball is seen.

### Rule check

Allowed. It was trained on other matches (IDSSE, CC BY 4.0, already approved for our use in #16). Cite IDSSE and PathCRF in the README. Its output is per-frame team probabilities. Regenerating them on CPU is fast, so there is no need to commit them.

**Verdict:** the strongest new model for the "who has it during a ball gap" question. 12–24 h, mostly the track-to-slot mapping.

---

## 5. Team-shape priors (who has the ball, from formation)

**Signal.** Teams spread out when they have the ball and contract when they don't. Clemente et al. found significant differences in centroid, stretch index (mean distance from the team centre), surface area and effective area between possession and non-possession phases, all at p ≤ 0.001 (3 matches, [record](https://comum.rcaap.pt/entities/publication/7738dce7-e938-4248-902a-6f07f2321f04)).

### Features

Compute per team, per frame, in the attacking direction:

- centroid x and its velocity
- length and width
- stretch index
- convex-hull area
- defensive-line height
- the gap between the two teams' lines
- the number of players in the opponent's half
- mean forward speed

Use the *difference* between the two teams, so the model sees relative shape.

### Training data with possession labels (all other matches)

| dataset | what | possession label | licence |
|---|---|---|---|
| **IDSSE** ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/), figshare DOI 10.6084/m9.figshare.28196177) | 7 matches (2 Bundesliga, 5 2. Bundesliga), TRACAB Gen5, 25 Hz, events | **`BallPossession` (1 home / 2 away) and `BallStatus` (0 dead / 1 alive) on every frame.** No accuracy figure is given for these flags. | **CC BY 4.0** |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) | 10 A-League 2024/25 matches, **broadcast** tracking at 10 fps, ball included | per-frame `possession` with `group` (home/away team) | MIT (they also ask for credit) |
| [Metrica sample data](https://github.com/metrica-sports/sample-data) | 3 matches, 25 Hz (`Time [s]` steps of 0.04), **ball column present** (checked), events with frames | derived from events (passes, ball lost, set pieces) | No licence file. "If you use it for anything public, please acknowledge the source." |
| PFF FC World Cup 2022 ([announcement](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data)) | 64 matches, broadcast tracking + events | via events | **unverified** (access terms not found) |

Loaders: [kloppy](https://github.com/PySport/kloppy) (BSD-3; Metrica, Sportec, PFF and others) and [floodlight](https://github.com/floodlight-sports/floodlight) (MIT; reads IDSSE).

**Model.** Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`) or logistic regression predicting P(home has ball). Train on IDSSE and SkillCorner, test on a held-out match. Training takes minutes on CPU and is deterministic with a fixed seed. Commit the model file.

### Main risks

1. Shape **lags** a turnover by 1–3 s, because teams need time to reorganise. It is good for "who" in long gaps and poor for switch timing.
2. Largely overlaps PathCRF, which learns shape and movement together.
3. Anonymous slots and duplicates distort hull area. Prefer robust features (medians, trimmed spreads).

### Rule check

Allowed (#16 for IDSSE, other matches).

**Verdict:** worth it mainly as a cheap fallback if PathCRF's slot mapping fails, and as the first experiment on the IDSSE pipeline. 8–12 h.

---

## 6. Learning on this match without labels

### What counts as a "label" here

A label is the **answer** we are scored on (H/A/D per frame), or anything standing in for it as a training target. That includes the provided tracking's ball positions when used as the target for a ball detector, which is what #2 ruled out.

These are **not** labels:

- The match's own inputs (tracking, audio, video).
- Statistics of those inputs (for example, how speeds are spread).
- Likelihoods a model assigns to them.

RULES.md explicitly allows "learning without human labels: unsupervised or self-supervised methods, or pseudo-labels produced by an algorithm, provided no person decided any of them". Our decisions log already uses the weakest form: thresholds placed from unlabelled distributions.

### Options

- **EM / Baum-Welch on our HMM.** Keep the states fixed in meaning (H, A, D, possibly "H-in-flight" and so on). Initialise from the hand-set values. Let EM re-estimate:
  - **transition rates**, which are our switch penalties
  - **emission noise**, meaning how trustworthy each clue is on *this* match's tracking quality

  Tools: [hmmlearn](https://github.com/hmmlearn/hmmlearn) (BSD-3, limited maintenance) or about 100 lines of numpy forward-backward. The forward-backward pass also gives **per-frame posteriors**, which minimum-Bayes-risk decoding for spellF1 needs.
- **HSMM (duration-aware).** Possessions and stoppages have typical lengths, and an HSMM uses them. [hsmmlearn](https://hsmmlearn.readthedocs.io) is GPL-3.0 and archived, and [edhsmm](https://pypi.org/project/edhsmm/) is archived too, so write our own. Run at 5 fps to keep the cost O(T·K·D) small: T time steps, K states, D the longest duration considered. See models-literature.md §4.2.
- **Clustering / self-supervised features.** For example, cluster team shapes into formation states, or learn embeddings of 10 s tracking windows. Low value here, because we have labelled public data (IDSSE) for the same purpose.

### Runtime

Seconds to minutes. It is deterministic if the initialisation and iteration count are fixed.

### Main risks

1. EM maximises the likelihood of the *observations*, not spellF1. Unanchored states drift into whatever explains the data best, such as "ball visible vs not" instead of "home vs away".
2. Over-fitting to quirks of this match.

Mitigations: anchor state meanings with fixed emission structure, fit few free numbers, and cap the change from the hand-set values. Better still, **fit the same HMM with labels on degraded IDSSE first** (idea 7), and use EM on BAR-ATH only as a small adaptation step.

### Rule check

- **EM / unsupervised:** allowed by RULES.md text ("What remains fine"). Say so in the technical document.
- **Self-training a model on this match from algorithmic pseudo-labels** (our own confident frames, or the provided ball track): **grey G1**, because #2 ruled out the detector case. Ask before building.

**Verdict:** a medium-low gain on its own. The posteriors are worth having for decoding. 8–16 h.

---

## 7. Synthetic data

### Option A: degrade public tracking to look like ours (recommended)

Take IDSSE, which has true H/A/D at 25 Hz, and damage it the way our tracker is damaged:

- **Drop the ball** to a 41% overall rate. Bias the drops toward frames with many players within a few metres of the ball, high ball speed, or the ball in the air (IDSSE has ball height). DATA.md says these are our failure modes. Matching *rates and conditions* from BAR-ATH's own missingness pattern uses no labels.
- **Anonymise players:** split tracks into slots, swap identities between nearby team-mates, drop players to about 20.5 visible on average, and add occasional duplicates.
- **Add position noise.** Reuse PathCRF's `preprocess_noise.py` (occlusion merges, bias σ 0.5–2 m, outliers).
- **Fake Smart-Tagging-like set-piece tags** from IDSSE events, with some dropped or shifted, and team set to N/A on throw-ins as in ours.

Then run our full pipeline and **score it with the organisers' scorer**. That gives us a validation set for choosing versions, switch penalties and clue weights. The organisers suggested exactly this kind of tuning on other matches (#10), and IDSSE was approved (#16). It also lets us *learn* each clue's likelihood, for example how often "nearest player to ball" is right at a given distance, instead of hand-setting it.

- **Effort:** 12–20 h. **Runtime:** minutes per match on CPU.
- **Risk:** if the damage model is unfaithful, settings tuned on it will not transfer. Keep it simple and check it against BAR-ATH's *input* statistics: gap-length histogram, slot counts, swap rate.
- **Rule:** allowed.

### Option B: a football simulator

[Google Research Football](https://github.com/google-research/football) (Apache-2.0):

- It gives `ball_owned_team` (−1 / 0 / 1) and `game_mode` (normal, kick-off, goal kick, free kick, corner, throw-in, penalty) every step ([observation docs](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md)). That is perfect H/A/D ground truth.
- But the repo was **archived on 2026-08-19**.
- It needs a C++ build with SDL and Boost (Apple-silicon build status **unverified**).
- Its bot movement looks nothing like LaLiga.

Low value for 10–20 h. Skip.

**Verdict:** Option A is the highest-leverage item for the remaining time, because it replaces guesswork with measurement on fair data.

---

## 8. Other clever things within the rules

### 8a. The broadcast ATD's ball fills tactical gaps (top pick)

The TV camera zooms in, so its tracker may see the ball when the wide tactical camera loses it in a crowd. Our limitations.md already notes this.

1. Using the existing alignment tool, measure the share of the ~64,700 graded tactical gaps where the aligned broadcast frame is **live** (valid homography, not a replay) **and** has a ball.
2. Where both cameras have a ball, measure their distance in metres. That checks the alignment and replay handling, using no labels.
3. Fill gaps only where the broadcast frame is confirmed live, because a replay would inject *past* positions.

- **Effort:** 4–8 h. **Gain:** high if the overlap is large; I cannot predict it without the data.
- **Rule:** allowed (a provided asset).

### 8b. Where the TV camera points is a ball-location proxy

The camera operator keeps the ball near the image centre. Projecting the broadcast image centre through its homography gives a pitch point near the ball (hypothesis, **unverified**; error perhaps 5–15 m). On tactical gaps where the broadcast also lacks the ball but shows a wide live shot, the team whose players are nearest that point is a weak possession clue. The zoom level (homography scale) also tightens around duels.

- **Effort:** 3–6 h on top of 8a.
- **Rule:** allowed. Measure its error on frames where the tactical ball *is* present; that is evaluation, not training.

### 8c. Restart geometry from tracking

Restarts have fixed shapes in the Laws of the Game:

- **Kick-off:** both teams in their own halves.
- **Goal kick:** the goalkeeper in the 6-yard box, opponents out of the area.
- **Corner:** a player at the corner arc.
- **Free kick:** a defensive wall about 9.15 m from a stationary point.
- **Throw-in:** a stationary player on the touchline.

Detecting the shape gives the **end of a dead stretch** and **the team that restarts**. That fills the known hole that Smart Tagging tags **every throw-in as team N/A**. Thresholds come from the laws and from IDSSE events, not from our match.

- **Effort:** 8–16 h. **Gain:** medium-high, because restarts start possessions, and possession starts drive IoU.
- **Rule:** allowed.

### 8d. Two-sided gap bridging with ball physics

For each gap we know where the ball was before and after. If the same team is nearest at both ends, keep that team. If the teams differ, place the switch at the earliest time the ball could physically reach the receiver (speed under about 30–35 m/s, plus a typical pass speed). Using the future is allowed (#15). Our Viterbi decider partly does this already. Making it explicit is about 3–5 h.

### 8e. Ball-action spotting on the broadcast

SoccerNet Team Ball Action Spotting 2025 requires 12 classes (pass, drive, header, shot, throw-in and so on) **plus the acting team**. That is close to our task ([write-up](https://int8.io/team-ball-action-spotting-challenge-2025), [T-DEED](https://github.com/arturxe2/T-DEED) GPL-3.0).

But I found **no released weights**, the data sits behind the SoccerNet NDA (grey), and the training cost is far beyond an M2 in 2.5 weeks. SoccerTrack v2 has the same action labels on *panoramic* video, which is closer to ours, but its licence and download are unverified. Skip for this phase. Revisit for a later phase.

### 8f. Things to avoid or ask about

- **Commentary transcription** (Whisper, run locally) gives humans describing this match: grey G2.
- **CLAP / CLIP zero-shot frame or audio labelling** goes against our no-LLM/VLM-labeller decision in spirit. Allowed by the rules, but not needed.
- **Training any model on BAR-ATH using its tracking as targets:** out (#2) or grey (G1).

---

## Ranked table

The gain notes say which part of the metric each idea moves:

- **who:** the right team in long gaps
- **when:** switch timing
- **end:** where possessions stop, which is what D drives

| rank | idea | expected gain | effort (h) | main risk | rule status |
|---|---|---|---|---|---|
| 1 | **8a + 3: broadcast ATD ball + "no-homography / replay" stretches**, via the existing alignment | **High.** Who and end. It fills ball gaps from a zoomed camera and gives a near-free dead-ball and restart-time cue. The size depends on overlap, which is the first thing to measure. | 6–12 | Replays injecting past positions; the director cutting late | Clearly allowed |
| 2 | **7A: degraded IDSSE (+ SkillCorner) as a scored validation set; learn clue likelihoods there** | **High, indirect.** It makes every other change measurable and lets weights be fitted fairly instead of guessed. | 12–20 | Damage model unfaithful to our tracker | Clearly allowed (#10, #16) |
| 3 | **4: PathCRF possession path from players** | **High** on the ~64,700 no-ball frames (who). 92.4% in play on a clean DFL test match. Expect less on ours. | 12–24 | Anonymous slots and identity swaps; in play only | Clearly allowed (other matches, MPL-2.0, CC BY data) |
| 4 | **8c: restart geometry** (throw-in team, goal kick, corner, free kick, kick-off) | **Medium-high.** End and who at every restart; fixes Smart Tagging's N/A throw-ins. | 8–16 | False positives in slow open play | Clearly allowed |
| 5 | **8d: two-sided gap bridging with ball physics** | **Medium.** When, inside gaps. Cheap, and partly in v1 already. | 3–5 | Overlap with the existing decider | Clearly allowed |
| 6 | **5: team-shape prior** (IDSSE / SkillCorner) | **Medium-low.** Who in long gaps. Lags turnovers and overlaps PathCRF. | 8–12 | Lag; distorted by duplicate slots | Clearly allowed |
| 7 | **1: ball detection, classical first, WASB tiles second** | **Medium-low.** Mostly in-flight balls (when). Occluded balls stay invisible. 5–20 h of compute. | 16–30 | 3–4 px ball at 4.9 Mbit/s; false positives; per-frame output may not be committable | Classical: clearly allowed. ISSIA-trained weights: **grey** (dataset is "non-commercial research only"). Roboflow/DFL: **grey** (terms unverified). |
| 8 | **6: HMM / HSMM by EM on this match** | **Low-medium.** When and end, via calibrated penalties and posteriors for minimum-Bayes-risk decoding. Better done supervised on IDSSE first. | 8–16 | Likelihood ≠ spellF1; states drift | Allowed by RULES.md ("learning without human labels"). Note #2 in the technical document. |
| 9 | **8b: TV camera aim as ball proxy** | **Low-medium.** Who on gaps where both cameras lack the ball. | 3–6 (after rank 1) | The operator leads or lags the ball | Clearly allowed |
| 10 | **2: pretrained audio SED** (frame_mn10 / BEATs-Strong; Whistle vs Whistling; crowd roar/boo) | **Low-medium.** Fewer v2 whistle false alarms; weak stoppage hint. | 4–8 | Correlated errors with our tonal detector | Clearly allowed |
| 11 | **3-extra: TransNetV2 cuts, graphics, grass ratio** beyond the homography flag | **Low.** Mostly redundant with the provided flag. | 4–8 | – | Clearly allowed |
| 12 | Scoreboard OCR | **Low.** The clock never stops, so it only helps alignment, which is done. | 3–5 | – | Clearly allowed |
| 13 | Fine-tuning our own ball detector on public wide-view data (SoccerTrack v1, ISSIA) | Medium potential, unlikely to pay off in time | 30–50 | Licences unclear; M2 training time | **Grey** (dataset licences) |
| 14 | Google Research Football simulator | **Low.** Unrealistic dynamics; archived repo. | 10–20 | Build issues on Apple silicon | Clearly allowed |
| 15 | Broadcast ball-action spotting (T-DEED / Team BAS) | Medium potential, not feasible now | 40+ | No weights; NDA data; compute | **Grey** (SoccerNet NDA) |
| 16 | Commentary speech-to-text | Medium potential | 6–10 | – | **Grey G2**: ask first |
| 17 | Self-training a detector or model on BAR-ATH with algorithmic pseudo-labels | – | – | – | **Grey G1**, and **out** when the provided tracking is the target (#2) |

### Suggested order

1. Rank 1: measure the broadcast overlap first.
2. Start rank 2 in parallel.
3. Run rank 3 on the IDSSE pipeline from rank 2 before running it on BAR-ATH.
4. Ranks 4 and 5 are cheap and independent.
5. Leave ball detection (rank 7) for when the cheaper fills are measured. Its value is whatever gap remains after them.

### Questions worth one GitHub issue

1. Are weights trained on ISSIA-CNR (WASB, FootAndBall), whose terms say "non-commercial research only", acceptable?
2. May a per-frame ball track from our own detector be committed, or must it be regenerated?
3. G1 and G2 as already listed in limitations.md.

---

## Sources

Our repository (read, not under `data/`): `RULES.md`, `DATA.md`, `docs/limitations.md`, `docs/decisions.md`, `docs/research/models-literature.md`.

**Ball detection**
- https://github.com/nttcom/WASB-SBDT
- https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md
- https://github.com/nttcom/WASB-SBDT/blob/main/GET_STARTED.md
- https://raw.githubusercontent.com/nttcom/WASB-SBDT/main/src/configs/dataset/soccer.yaml
- https://arxiv.org/abs/2311.05237
- https://arxiv.org/html/2311.05237
- https://pspagnolo.jimdofree.com/download/ (ISSIA-CNR / F.I.G.C.-C.N.R. dataset terms)
- https://github.com/jac99/FootAndBall
- https://github.com/roboflow/sports
- https://github.com/roboflow/sports/tree/main/examples/soccer
- https://raw.githubusercontent.com/roboflow/sports/main/examples/soccer/setup.sh
- https://blog.roboflow.com/tracking-ball-sports-computer-vision/
- https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg (returned 403)
- https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/rules (did not render)
- https://bundesliga.com/en/aws-bundesliga-data-shootout
- https://docs.ultralytics.com/models/yolo11/
- https://github.com/ultralytics/ultralytics
- https://github.com/obss/sahi
- https://cran.r-project.org/web/packages/nativeORT/vignettes/BenchmarkPerformance.html
- https://www.soccer-net.org/data
- https://arxiv.org/pdf/2204.06918
- https://arxiv.org/pdf/2308.16651
- https://www.alphaxiv.org/benchmarks/kaust/soccernet-tracking
- https://github.com/AtomScott/SoccerTrack
- https://openaccess.thecvf.com/content/CVPR2022W/CVSports/html/Scott_SoccerTrack_A_Dataset_and_Tracking_Algorithm_for_Soccer_With_Fish-Eye_CVPRW_2022_paper.html
- https://arxiv.org/html/2508.01802v1
- https://atomscott.github.io/SoccerTrack-v2/
- https://arxiv.org/abs/2404.13868

**Audio**
- https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
- https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv
- https://research.google.com/audioset/ontology/whistle.html
- https://research.google.com/audioset/ontology/steam_whistle.html
- https://research.google.com/audioset/ontology/whistling.html
- https://research.google.com/audioset/download_strong.html
- http://storage.googleapis.com/us_audioset/youtube_corpus/strong/mid_to_display_name.tsv
- https://github.com/qiuqiangkong/audioset_tagging_cnn
- https://zenodo.org/records/3987831
- https://arxiv.org/abs/1912.10211
- https://github.com/fschmid56/EfficientAT
- https://github.com/fschmid56/PretrainedSED
- https://arxiv.org/abs/2409.09546
- https://github.com/YuanGongND/ast
- https://github.com/microsoft/unilm
- https://github.com/LAION-AI/CLAP
- https://github.com/microsoft/CLAP
- https://arxiv.org/abs/1411.6496 (whistle detector and replay-logo detector)
- https://arxiv.org/pdf/2309.12067
- https://arxiv.org/pdf/2011.04258
- https://www2.mdpi.com/2504-4990/3/4/51

**Broadcast**
- https://github.com/soCzech/TransNetV2
- https://github.com/soCzech/TransNetV2/tree/master/inference
- https://arxiv.org/abs/2008.04838
- https://arxiv.org/html/2008.04838
- https://research.hub.ku.edu.tr/entities/publication/0ebd0d09-be01-4093-b04e-e07e2f94f805
- https://arxiv.org/abs/2011.13367
- https://github.com/SilvioGiancola/SoccerNetv2-DevKit

**Imputation and possession from players**
- https://github.com/hyunsungkim-ds/pathcrf
- https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/main/saved/140/args.json
- https://arxiv.org/abs/2602.12080
- https://arxiv.org/html/2602.12080
- https://github.com/pientist/ballradar
- https://arxiv.org/abs/2306.08206
- https://arxiv.org/html/2306.08206

**Team shape and open tracking**
- https://comum.rcaap.pt/entities/publication/7738dce7-e938-4248-902a-6f07f2321f04
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/
- https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md
- https://github.com/spoho-datascience/idsse-data
- https://github.com/SkillCorner/opendata
- https://github.com/metrica-sports/sample-data
- https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv (first 1.5 kB only, to check the header)
- https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data
- https://github.com/PySport/kloppy
- https://github.com/floodlight-sports/floodlight

**HMM / HSMM**
- https://github.com/hmmlearn/hmmlearn
- https://hsmmlearn.readthedocs.io
- https://pypi.org/project/edhsmm/

**Simulator and action spotting**
- https://github.com/google-research/football
- https://github.com/google-research/football/blob/master/gfootball/doc/observation.md
- https://github.com/arturxe2/T-DEED
- https://int8.io/team-ball-action-spotting-challenge-2025
- https://arxiv.org/pdf/2409.10587
