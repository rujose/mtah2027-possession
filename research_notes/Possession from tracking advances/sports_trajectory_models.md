# Pretrained multi-agent sports trajectory models and possession inference from other sports (2024–2026)

Scope note. Researched 2026-11-12. The team survey (docs/research/models-literature.md, extensions.md, datasets-and-code.md) already covers PathCRF, Ball Radar, TranSPORTmer, Peral et al. 2025, Wei et al. 2015, Link & Hoernig, MS-TCN/ASRF and the Polito TCN possession paper; they are referenced here only for comparison. "Pipeline" below means the team's model: per-frame H/A/D for one Metrica match (25 fps, ~199k frames, ball missing on ~41% of frames, identity switches, anonymous track slots), bidirectional GRU + GBT on ~97 hand features, min-length Viterbi, post stages, trained leave-one-match-out on 10 external matches, Apple M2 (no CUDA), frozen for 2026-11-17.

Bottom line across all questions: **no public, licence-clean, pretrained "foundation model" for team-sport tracking data exists that could drop into the pipeline.** The pretrained models that exist either (a) ship weights trained on data with unclear or restrictive terms (UniTraj), (b) ship no code or weights (TacticAI, TacticGen, AdaSports-Traj, T-MAE, Stats Perform's CVIU 2026 model, Li & Link 2026), or (c) need a CUDA GPU or a 7B LLM (UniTraj's Mamba kernels, TacticExpert). PathCRF (already in the survey) remains the only pretrained, ball-free, open-licence soccer possession model. What other sports offer is mostly **architecture and training-recipe evidence** for the next phase.

## 1. Which pretrained multi-agent trajectory ("foundation") models exist, what were they trained on, are weights public, under what licence, and do they report downstream gains?

### Takeaway
The closest thing to an open pretrained sports-trajectory model is UniTraj (ICLR 2025): Apache-2.0 code with public checkpoints, but trained on Stats Perform NBA, NFL Next Gen Stats and SoccerTrack drone clips, with CUDA-only Mamba layers and no possession task. Everything else is either unreleased (TacticAI, TacticGen, Stats Perform's reconstruction model, AdaSports-Traj, T-MAE) or not trajectory-based (SoccerMaster is a video model). The only peer-reviewed evidence that self-supervised pretraining on soccer tracking helps a downstream classification task is Rudolph & Brefeld's masked autoencoder (T-MAE), and its numbers and code were not retrievable.

### Cited Findings

**UniTraj / "Sports-Traj" (Xu & Fu, ICLR 2025)**
- One model for "arbitrary trajectories as masked inputs" (prediction, imputation, and so on) for multi-agent sports. It uses a Ghost Spatial Masking module inside a Transformer encoder plus a Bidirectional Temporal Mamba (a state-space sequence layer) — [arXiv 2405.17680](https://arxiv.org/abs/2405.17680v2); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/30682).
- Benchmarks it introduced: Basketball-U (93,490 train / 11,543 test sequences, 1 ball + 10 players), Football-U (NFL; 10,762 / 2,624, ball + 22 players), Soccer-U (9,882 / 2,448, ball + 22 players) — [AdaSports-Traj paper, which reuses them](https://arxiv.org/html/2509.16095).
- Repo: [colorfulfuture/UniTraj-pytorch](https://github.com/colorfulfuture/UniTraj-pytorch), **Apache-2.0** (GitHub API; last push 2025-03-17, 28 stars). Checkpoints (`best_ade_epoch.pth`, `last_epoch.pth`) for basketball, football and soccer are on Google Drive alongside raw and pre-processed data. Requires `pytorch-cuda=11.8`, `mamba-ssm`, `causal-conv1d`. No separate inference command; evaluation runs `main.py --evaluate` — [repo README](https://github.com/colorfulfuture/UniTraj-pytorch).
- Training-data sources per the README: basketball from **Stats Perform** (via ezhan94's calibratable-style-consistency repo), American football from **NFL Next Gen Stats** (Big Data Bowl repo), soccer from **SoccerTrack top-view** via SportsLabKit. The README states no data licences — [repo README](https://github.com/colorfulfuture/UniTraj-pytorch). The two GitHub code repos carry MIT (calibratable-style-consistency) and GPL-3.0 (SoccerTrack, SportsLabKit) code licences (GitHub API), which say nothing about the data's terms.
- No possession, event-detection or ball-ownership downstream task is reported. The tasks are trajectory completion only (minADE/FDE) — [AdaSports-Traj tables](https://arxiv.org/html/2509.16095).

**AdaSports-Traj (Xu & Fu, arXiv 2509.16095, Sep 2025)**
- Built on UniTraj; adds a role- and domain-aware adapter and hierarchical contrastive learning. minADE20 single-dataset: Basketball-U 4.21 vs UniTraj 4.77; Football-U 3.04 vs 3.55; Soccer-U 91.52 vs 94.59. Trained on all three sports and tested on each: 8.74 / 9.04 / 104.66 vs UniTraj 11.12 / 11.89 / 124.56 — [arXiv HTML](https://arxiv.org/html/2509.16095).
- Requires pre-defined role and domain labels; no code or weights mentioned; paper under CC BY-NC-SA 4.0 — [arXiv HTML](https://arxiv.org/html/2509.16095).

**T-MAE, masked autoencoder for multi-agent trajectories (Rudolph & Brefeld)**
- Workshop version (MLSA 2023, Springer CCIS vol. 2035, 2024): "pretraining transformer models improves supervised classification of tracking data from elite soccer" — [Leuphana record](https://fis.leuphana.de/en/publications/masked-autoencoder-pretraining-forevent-classification-inelite-so/).
- Journal version: *Machine Learning* 114 (2025), DOI [10.1007/s10994-024-06647-3](https://doi.org/10.1007/s10994-024-06647-3). A factorized transformer (attention over time and over agents) with masking of **whole individual agent trajectories**; permutation-equivariant in agents, so downstream tasks can be permutation-invariant; needs no hand-crafted features; aimed at classification "from only a few labeled sequences"; also tested on insect pose data — [ML Anthology entry](https://mlanthology.org/mlj/2025/rudolph2025mlj-masked).
- No code repository is linked — [ML Anthology entry](https://mlanthology.org/mlj/2025/rudolph2025mlj-masked).

**TacticAI (Google DeepMind + Liverpool FC, Nature Communications 2024)**
- 9,693 Premier League corners (7,176 after filtering) from Liverpool FC. GATv2 graph network with D2 group convolutions (equivariance to horizontal and vertical pitch reflections). Receiver prediction top-3 accuracy 0.782 ± 0.039. The paper says modelling the reflection symmetry gives "a clear advantage"; the ablation numbers sit in Supplementary Table 2 (not retrieved) — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10951310/).
- Code: "We are unable to release our code for this work as it was developed in a proprietary context." Data: "protected and are not available due to data privacy laws and licensing restrictions" — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10951310/).

**TacticGen (Xu, Liu et al., arXiv 2604.18210, Apr 2026)**
- Multi-agent diffusion transformer (DiT-based), trained on event and tracking data from 1,432 top-tier matches (2018–2025): over 3.3M events and about 100M frames. Partners include Real Analytics and Birmingham City FC. Tasks: trajectory prediction, guided tactic generation, a basketball benchmark. No possession or event-detection task. No code or weight release or software licence stated; paper under CC BY-NC-SA 4.0 — [arXiv HTML](https://arxiv.org/html/2604.18210v1).

**Stats Perform's large-scale reconstruction model (Hughes, Lucey, Horton et al., CVIU 264:104626, 2026)**
- "Multimodal transformer–diffusion framework for large-scale reconstruction of soccer tracking data" — [researchr listing](https://researchr.org/publication/HughesLHGFS26); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1077314225003492) (returned HTTP 403, abstract not read).

**Other "foundation"-labelled or adjacent work**
- SoccerMaster (arXiv 2512.11016, Dec 2025) is a *vision* foundation model for soccer, not a tracking-data model — [arXiv PDF](https://arxiv.org/pdf/2512.11016) (search snippet only).
- TrajSV (arXiv 2508.11569) tokenizes field trajectories into 3 m cells and 1 s segments for sports-video representations; no released weights were found — [arXiv PDF](https://arxiv.org/pdf/2508.11569) (search snippet only).
- Trajectory "LLMs": TacticExpert feeds graph tokens from a spatio-temporal graph transformer into Vicuna-7B-v1.5 (see Q2). No 2025–2026 paper was found that tokenizes *soccer tracking* for an LLM. The one soccer "language" forecasting paper works on event data, not tracking — [arXiv 2402.06820](https://arxiv.org/pdf/2402.06820).
- baller2vec / baller2vec++ (Alcorn & Nguyen, 2021): multi-entity Transformers whose per-timestep input is an *unordered set* of NBA players plus the ball; MIT licence, last push Aug 2021. No weights are distributed; the README has training instructions only — [baller2vec repo](https://github.com/airalcorn2/baller2vec), [arXiv 2102.03291](https://arxiv.org/abs/2102.03291), [baller2vec++ repo](https://github.com/airalcorn2/baller2vecplusplus).
- Synthetic tracking: Radke & Tilbury (AAMAS 2025 demo) generate soccer tracking data from the Google Research Football (GRF) simulator in a real-tracking-like schema, with feature and event extraction — [arXiv 2503.19809](https://arxiv.org/abs/2503.19809). GRF's observation already contains `ball_owned_team` ({-1 not owned, 0 left, 1 right}), `ball_owned_player`, and `game_mode` (Normal, KickOff, GoalKick, FreeKick, Corner, ThrowIn, Penalty) — [GRF observation docs](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md). GRF is **Apache-2.0** and the repo is **archived** (GitHub API, last push 2025-06-17).
- 2026 Kaggle NFL Big Data Bowl (Prediction): the 3rd-place write-up is titled "PreTrain-FineTune: Spatio-Temporal Transformer with Multi-Aux-Loss for NFL Trajectory Forecasting." The details of its pretraining were not visible — [third-party README summarising top solutions](https://github.com/AceGiqMo/NFL-Big-Data-Bowl-2026).

### Inferences
- **For the frozen model (days): nothing here is usable.** UniTraj is the only one with public weights. But (i) its checkpoints were trained on Stats Perform, NGS and SoccerTrack data whose redistribution terms are unclear, which is a problem with commercial partners; (ii) its Mamba layers depend on CUDA kernels, so it will not run on an M2 as shipped; (iii) Soccer-U is short university drone clips, and its minADE of ~90–125 vs ~3–9 for the other sports suggests different units (likely pixels; unverified), so it is far from 25 fps professional optical tracking; (iv) it has no possession head.
- **Next phase (weeks): T-MAE-style self-supervised pretraining is the most transferable idea.** Masking whole agent trajectories inside a factorized time × agent transformer matches the pipeline's two pain points: anonymous slots (permutation-equivariant) and missing or broken tracks (masked agents are what the model learns to cope with). It can be pretrained on licence-clean unlabelled tracking (IDSSE 7 matches CC BY 4.0, Metrica 3 samples, SoccerTrack v2 10 matches CC BY 4.0 per the team's dataset notes), plus the test match itself if the organisers allow self-supervision on it (already an open organiser question in extensions.md). A model of a few million parameters trains on an M2 via PyTorch MPS (plausible, not tested).
- **GRF synthetic data is the only free source of per-frame possession and dead-ball labels at scale** (Apache-2.0). `ball_owned_team = -1` also covers loose balls in play, so H/A/D needs "last owner carried forward" for in-play, and `game_mode != Normal` for D. Expect a large sim-to-real gap (simulated movement, 2D-ish coordinates in [-1, 1] × [-0.42, 0.42]). Use it for pretraining or curriculum only, never as a stand-in for real validation. It is weeks of work, and the archived repo's build on Apple Silicon is unverified.
- TacticAI's D2 reflection result supports a cheap, label-preserving trick for the next phase: **y-axis mirroring (left↔right touchline) as training augmentation and test-time averaging.** It keeps H/A/D labels unchanged. An x-mirror also needs the attacking direction and team labels swapped. NFL BDB 2026's 1st place also credits flip and rotation augmentation for beating overfitting on limited data (Q3).

### Gaps
- T-MAE's quantitative gains (how much pretraining helps at 1%, 10% and 100% labels) and its exact soccer dataset (provider, number of matches, public or not) could not be read; Springer redirected to login.
- Whether UniTraj's basketball (Stats Perform) and NFL (Kaggle BDB) training data permit redistribution of derived weights for commercial-partner use is unverified. Kaggle BDB rules were not checked.
- Hughes et al. (CVIU 2026) abstract, method and code status not retrieved (403). Likely Stats Perform-internal data; no code found.
- TacticAI's no-symmetry ablation numbers (Supplementary Table 2) were not retrieved.
- No industry "tracking foundation model" (Stats Perform, Genius/Second Spectrum, SkillCorner, Hawk-Eye) with accessible weights was found. Searches turned up only detection and tracking models on Hugging Face (e.g. RT-DETR/YOLO/RF-DETR soccer detectors), not trajectory encoders — [search results incl. huggingface.co/smallTech/rtdetr-sportsmot](https://huggingface.co/smallTech/rtdetr-sportsmot).

## 2. Basketball: ball-handler / possession inference from player tracking without the ball (successors to Wei et al. 2015), NBA SportVU models, and what transfers

### Takeaway
I found no 2024–2026 basketball paper that infers the ball handler from player tracking alone as a dedicated, evaluated task. NBA tracking includes the ball, so the field has largely moved to ball-inclusive modelling. The only new "who holds the ball" result is TacticExpert (2025), a graph-transformer-plus-7B-LLM trained on ~1,780 images from a French-league image dataset, with no code found. Basketball's real contribution is as a benchmark in multi-sport models (UniTraj, MIDAS, TranSPORTmer) and as the origin of set-based multi-entity encoders (baller2vec).

### Cited Findings
- **TacticExpert** (Xu, Liu, Zhang; arXiv 2503.10722, Mar 2025): a graph transformer encoder (2 spatio-temporal layers, 256-dim, 5-expert mixture-of-experts, "group-equivariant attention") whose node embeddings are passed as graph tokens to **Vicuna-7B-v1.5**. Tasks: node classification "who is most likely to hold the ball" (**macro-F1 0.8333**), pass link prediction (AUC 0.7264), shot graph classification (macro-F1 0.6750) — [arXiv HTML](https://arxiv.org/html/2503.10722v1).
  - Data: Kaggle "DeepSport Basketball-Instants" (French pro league), **1,780 images from 38 games**, not SportVU. Its season description is internally inconsistent. Trained on 2× RTX 3090. Only internal ablations; no external baselines. The paper says code and data are open-sourced, but gives no URL — [arXiv HTML](https://arxiv.org/html/2503.10722v1).
- **baller2vec** (2021): unordered-set input of players and ball per timestep. By learning only to predict the *ball* trajectory, it picked up player attributes, and some attention heads appear to "anticipate passes". MIT licence; no released weights — [repo README](https://github.com/airalcorn2/baller2vec).
- **MIDAS** uses the first 100 SportVU NBA games as one of its three imputation benchmarks (Q5) — [arXiv HTML](https://arxiv.org/html/2408.10878).
- **UniTraj/AdaSports-Traj** use Stats Perform basketball (Basketball-U, 10 players + ball) — [UniTraj repo](https://github.com/colorfulfuture/UniTraj-pytorch); [AdaSports-Traj](https://arxiv.org/html/2509.16095).
- Older SportVU work that touches the ball handler: DeepHoops (2019) predicts "terminal actions" that end a possession, from player *and ball* trajectories — [arXiv 1902.08081](https://arxiv.org/pdf/1902.08081). Neural Relational Inference (2018) learns interaction graphs on SportVU pick-and-roll data — [arXiv 1802.04687](https://arxiv.org/pdf/1802.04687).
- Targeted 2025 searches for basketball possession-change or transition detection from tracking returned only these older papers plus soccer work (Polito TCN, PathCRF) — [search results incl. DeepHoops](https://arxiv.org/pdf/1902.08081), [Polito TCN, ESWA 2023](https://iris.polito.it/bitstream/11583/2976470/7/1-s2.0-S0957417423002816-main.pdf).

### Inferences
- **Days: nothing to adopt.** TacticExpert is not credible as a transferable model for the pipeline: tiny image-derived data, an LLM needing about 50 GB of GPU memory, no repo, no external baselines.
- **What transfers (next phase):** (a) set or role-ordered encodings that do not need player IDs. This is the line from Wei 2015 (role ordering) through baller2vec to PathCRF's set attention, and it suits Metrica's anonymous slots. (b) Training an auxiliary head to predict the ball (position, or nearest-player-to-ball) on frames where it is visible, then using the learned representation on ball-missing frames. baller2vec's "predict the ball only" objective is a basketball precedent. (c) Basketball's 10-agent scenes make its results optimistic for 22-agent soccer with 41% missing ball.
- Basketball has always had the ball in tracking, so "possession without ball" research was never pressing there. Its absence is structural, not a sign the problem is solved.

### Gaps
- No 2024–2026 dedicated successor to Wei et al. 2015 was found. MIT Sloan 2025–2026 papers were not searched exhaustively.
- TacticExpert's repo could not be located; its claimed open-sourcing is unverified.
- Data terms for the public 2015–16 SportVU dump that MIDAS and baller2vec use were not checked.

## 3. American football, ice hockey, handball, rugby, field hockey: methods for possession or play state from player positions

### Takeaway
None of these sports has a published, transferable "possession from players only" model from 2024–2026. In the NFL and the NHL the ball or puck is itself tracked (NHL possession comes from puck-and-player tracking plus a provider possession feed), so research uses provider possession rather than inferring it. What transfers is (a) NFL Kaggle-winner recipes for small datasets (pairwise attacker × defender relational encoders that ignore player order; flip and rotation augmentation; probabilistic and auxiliary losses; transformers over players) and (b) handball evidence that a graph network trained on ~10 labelled games can segment game phases at scale.

### Cited Findings

**American football (NFL Big Data Bowl)**
- 2026 Big Data Bowl analytics winner: Lucca Ferraz (Rice), "Ghostbusters", evaluating defender movement while the ball is in the air via hypothetical "ghost defender" distributions; not a trajectory deep model — [NFL Operations](https://operations.nfl.com/gameday/analytics/big-data-bowl); [Rice News](https://news.rice.edu/news/2026/classroom-pros-rice-sport-analytics-program-turning-data-opportunity).
- 2026 Kaggle **Prediction** track (predict player positions while the ball is in the air):
  - 1st place (Kaggle user ohkawa3, write-up dated 23 Jan 2026): trained only on the provided data; credits "powerful data augmentation, well-designed loss functions, and an effective model architecture". Inputs are the 20 pre-pass frames relative to the ball landing point and the targeted receiver. Ablations: rotation augmentation and early-frame prediction reduce overfitting, an auxiliary loss stabilises training, and Gaussian NLL improves accuracy — [Kaggle profile/search snippet](https://www.kaggle.com/chack3).
  - A third-party summary adds flip and x/y-shift augmentation and temporal frame shift to the 1st-place recipe — [AceGiqMo README](https://github.com/AceGiqMo/NFL-Big-Data-Bowl-2026).
  - 3rd place: "PreTrain-FineTune: Spatio-Temporal Transformer with Multi-Aux-Loss". 5th place: factorized spatio-temporal encoder ("SqueezeFormer over time, Transformer over players"), delta parameterisation, Huber loss, EMA weights — [AceGiqMo README](https://github.com/AceGiqMo/NFL-Big-Data-Bowl-2026) (secondhand; write-ups not read directly).
- 2020 Kaggle BDB winner "The Zoo" (Singer & Gordeev): a CNN over a defender × offense pairwise tensor of relative position, speed and acceleration. The model is order-invariant ("you can put the players in any order and the model will treat them the same"). The NFL later deployed it as Next Gen Stats Expected Rushing Yards — [NFL Operations](https://operations.nfl.com/gameday/analytics/stats-articles/what-the-top-predictions-looked-like-in-the-nfls-big-data-bowl); [AWS ML blog](https://aws.amazon.com/blogs/machine-learning/identifying-defense-coverage-schemes-in-nfls-next-gen-stats/); [NFL.com](https://www.nfl.com/news/next-gen-stats-intro-to-expected-rushing-yards).

**Ice hockey**
- NHL puck and player tracking: about 14 infrared cameras plus sensors in the puck and jerseys; puck at 60 Hz, players at 12 Hz. The public NHL EDGE data does not yet offer "puck on stick" possession — [ESPN](https://www.espn.com/nhl/story/_/id/38724195/nhl-brings-advanced-puck-tracking-stats-public); [search summary of Pitassi thesis](https://uwspace.uwaterloo.ca/items/20229aef-d769-468e-9f51-fe1ae8f094f6).
- Pitassi (Waterloo MSc, 2025): a pipeline for cleaning and filtering *individual possession data* from the NHL's tracking system (2023–24, 2024–25), and adapting NHL inference algorithms to recover undetected shot attempts. Possession comes from the NHL's "DISH" feed — [UWSpace](https://uwspace.uwaterloo.ca/items/20229aef-d769-468e-9f51-fe1ae8f094f6). Companion LINHAC 2025 paper on individual puck possessions (frequency, duration, distance) — [LINHAC](https://ecp.ep.liu.se/index.php/linhac/article/view/1208).
- Big Data Cup 2025 combined Stathletes event data with broadcast-derived player tracking (AHL). The 2027 cycle's data is goalie and puck tracking from a shooting session — [Stathletes Big Data Cup](https://stathletes.com/big-data-cup). Their NESSIS 2025 deck describes a YOLOv12/YOLOv5 + BoT-SORT broadcast pipeline, with no standalone possession model in the excerpt — [NESSIS deck](https://docs.google.com/presentation/d/1MXXVGP7CJ5BI1FRC7hxcZpyoZbn5IbRW/htmlpresent).
- Faceoff detection from overhead-video player tracklets (Nsiempba et al., Waterloo): first- and second-order positional features with kNN classification, with "whistle" events as negatives. This is the hockey analogue of restart detection from player formations, but the evaluation and code were not found in the extracted text — [JCVIS PDF](https://openjournals.uwaterloo.ca/index.php/vsl/article/download/6369/6237/27499).
- Puck localisation from broadcast video plus player-position heatmaps (Vats et al., 2021) is the closest hockey "infer the object from players" work, but it is video-based — [arXiv 2105.10563](https://arxiv.org/pdf/2105.10563).

**Handball**
- FAUPA-ML (Bassek, Raabe, Banning, Memmert, Rein): a graph-based deep network trained on expert video labels from **10 games**, then applied to **539** German Bundesliga matches (2019/20–2021/22), finding 62,648 positional attacks and 19,262 counter-attacks — [conference record](https://lida.sport-iat.de/ta/Record/4088670?lng=de); [J. Sports Sci. 2023 record](https://www.iat.uni-leipzig.de/datenbanken/iks/dbv-base-softball/Record/4085680).
- Lefèvre et al. (PLOS ONE 2023): rule-based game-phase recognition from LPS data (resampled to 10 Hz), thresholds set empirically, 25 games of a Spanish team — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0289752).

**Soccer analogue from the same "phase" line**
- Li & Link (arXiv 2606.09289, v4 Nov 2026): a temporal graph attention network for in-possession *phases*. It uses 7 Bundesliga TRACAB matches at 25 fps, leave-one-match-out. **Possession segmentation is given as input** (it is not inferred). See Q4 for numbers — [arXiv abstract](https://arxiv.org/abs/2606.09289); [arXiv HTML](https://arxiv.org/html/2606.09289).

**Rugby, field hockey**
- No positional-data possession or phase-detection papers were found (see Gaps).

### Inferences
- **Days:** nothing to adopt. Hockey and NFL "possession" is sensor-based. The handball and NFL methods have no released, licence-clean weights for soccer.
- **Next phase, worth copying:**
  1. *Pairwise team-vs-team relational block* (The Zoo): build a 11 × 11 tensor of home-slot × away-slot relative features, apply 1×1 convolutions or MLPs, then pool over both axes. This is order-invariant, which suits anonymous Metrica slots. It is natural for "who is pressing whom / which team is closer to the ball's likely location" and cheap on an M2.
  2. *Small-data recipe from the 2026 NFL winners:* mirror augmentation (label-preserving y-flip; x-flip with team swap), temporal shift, auxiliary losses (e.g. predicting ball position or velocity when visible), and probabilistic outputs. All of these directly target overfitting with 10 training matches.
  3. *Handball FAUPA-ML* is a precedent for the team's exact regime: label about 10 games, train a graph model, and apply it at scale. Treat it as encouraging, not as a benchmark (no accuracy numbers retrieved).
- Hockey restart detection from formations (faceoffs) parallels soccer set-piece restarts as a D-state cue. The team's survey already has player-only set-piece triggers (Link & Hoernig), so it adds little.

### Gaps
- Big Data Bowl 2024 (tackling) and 2025 (pre-snap) winners were not researched; they are unlikely to bear on possession.
- The 2026 Kaggle Prediction write-ups were read only through search snippets and a third-party README; the architectures are not confirmed from primary sources.
- FAUPA-ML's accuracy against expert labels and its exact graph architecture were not retrieved.
- No sources were found for rugby or field-hockey possession from positions. The NHL's possession-inference algorithm (how DISH possession is derived) is not public.

## 4. Set/graph architectures for permutation-invariant team modelling (DeepSets, Set Transformer, GNNs, equivariant models) newer than 2024 with evidence on sports tasks; comparison with GRU/TCN baselines

### Takeaway
Since 2024 the evidence in sports leans towards a set encoder over players combined with a strong temporal model. Set Transformer + Bi-LSTM (MIDAS) beats RNN and attention imputers. Temporal transformers matter more than the graph part (Li & Link: Transformer ≈ temporal-GNN at intention level, both ≫ pure GNN and random forest). Small position-based GNNs beat large video models on group activities (SoccerNet-GAR). Reflection equivariance helps (TacticAI). I found **no head-to-head of a set or graph encoder against a GRU or TCN on per-frame possession labelling**, so the gain over the team's BiGRU + GBT is unmeasured.

### Cited Findings
- **MIDAS** (Choi, Kim, Lee et al.; arXiv 2408.10878, v4 Jul 2025): a Set Transformer for permutation-equivariant player embeddings, then per-player Bi-LSTMs. It predicts position, velocity and acceleration, combines forward and backward "derivative-accumulating" estimates, and learns per-player, per-time mixing weights. Data at 10 Hz: Metrica samples (soccer, 3 matches), first 100 SportVU NBA games, NFL BDB 2021 (via NRTSI preprocessing) — [arXiv HTML](https://arxiv.org/html/2408.10878).
  - Position error (uniform missing): soccer **1.3205** vs ImputeFormer 2.0898; basketball 0.9727 vs 1.3622; NFL 0.1542 vs 0.3673. In the agent-wise basketball setting ImputeFormer (1.3345) slightly beats MIDAS (1.3862). With 3 basketball games of training data: 1.1438 vs 1.6741. Baselines also include linear and cubic interpolation, BRITS, NAOMI, NRTSI, CSDI and Graph Imputer. Imputes players only, not the ball — [arXiv HTML](https://arxiv.org/html/2408.10878).
  - Code [gkswns95/midas](https://github.com/gkswns95/midas): **no licence** (GitHub API: `license: null`; last push 2025-06-20), PyTorch 1.10.1, soccer script only, no weights mentioned — [repo](https://github.com/gkswns95/midas).
- **Li & Link 2026, temporal graph attention network (T-GAN)** for in-possession phases (soccer, 7 Bundesliga TRACAB matches, 25 fps, leave-one-match-out): frame-wise player graphs → GNN → attention pooling per team with positional roles → Transformer over frames → MLP head. Weighted focal loss. **A post-filter that reassigns short runs** (10–50 frame minimums) — [arXiv HTML](https://arxiv.org/html/2606.09289).
  - Phase-level temporal IoU after filtering (macro): T-GAN **0.61**, pure Transformer 0.58, pure GNN 0.51, random forest 0.46, rules 0.38. Intention-level: T-GAN **0.74** = Transformer 0.74 > GNN 0.71 > RF 0.62 > rules 0.59. Frame F1 for T-GAN 0.87 (intention), 0.76 / 0.79 (phases). The authors conclude that temporal modelling drives coherent segmentation and that the graph adds value for transitions such as counter-attacks — [arXiv HTML](https://arxiv.org/html/2606.09289); [abstract](https://arxiv.org/abs/2606.09289). No code or data release — [arXiv HTML](https://arxiv.org/html/2606.09289).
- **SoccerNet-GAR / "Pixels or Positions?"** (Karki, Ramazanova, Cioppa, Giancola, Ghanem; arXiv 2511.12606, v3 Apr 2026): a role-aware GNN on player positions reaches **77.8% balanced accuracy** vs 60.9% for the best video model, with 180K vs 86.3M parameters and about 7× fewer GPU hours. 64 World Cup 2022 matches, 87,939 activities in 10 classes — [arXiv abstract](https://arxiv.org/abs/2511.12606). The team's notes already flag the dataset as NDA / PFF-restricted (docs/research/README.md).
- **TacticAI**: GATv2 with D2 reflection-equivariant group convolutions; reflection symmetry reported as "a clear advantage"; shot-prediction F1 0.71 ± 0.01 — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10951310/).
- **T-MAE**: a factorized transformer that is permutation-equivariant in agents, used for few-label classification — [ML Anthology](https://mlanthology.org/mlj/2025/rudolph2025mlj-masked).
- **Handball FAUPA-ML**: a GNN on positional data, trained on about 10 labelled games — [J. Sports Sci. record](https://www.iat.uni-leipzig.de/datenbanken/iks/dbv-base-softball/Record/4085680).
- **GNN counter-attack analysis** in soccer (Bekkers & Sahasrabudhe, arXiv 2411.17450, 2024), the basis of the unravelsports GNN tooling already in the team's notes — [arXiv PDF](https://arxiv.org/pdf/2411.17450).
- **NFL 2026 Prediction top solutions** use transformers over players with a separate temporal encoder (factorized), not GNNs; a GRU + attention-pooling ensemble appears in non-top public repos — [AceGiqMo README](https://github.com/AceGiqMo/NFL-Big-Data-Bowl-2026); [FelixWan777 repo](https://github.com/FelixWan777/nfl-player-trajectory-prediction).
- Generic 2025 equivariance work (e.g. wreath-equivariant layers, ICLR 2025; permutation-equivariant neural graph CDEs, NeurIPS 2025) has **no sports evaluation** — [arXiv 2410.06665](https://arxiv.org/pdf/2410.06665); [arXiv 2506.20324](https://arxiv.org/abs/2506.20324v1).

### Inferences
- **Days:** no architecture change is advisable before the freeze. Li & Link independently use the same kind of minimum-run post-filter as the team's min-length Viterbi, and report it lifting segment IoU. That is mild external support for the current design, not a reason to change it.
- **Next phase:** the evidence supports replacing part of the hand-feature front end with a **set encoder over the 22 slots (Set Transformer or Zoo-style pairwise home × away block) feeding the existing BiGRU**, rather than a full GNN. Li & Link show the temporal model carries most of the signal and the GNN alone underperforms. A set encoder also handles a variable or fragmented number of tracks (identity switches → more, shorter tracks, padded with masks), which hand features and fixed-slot GRUs handle poorly.
- MIDAS's derivative-accumulating imputation is the best-documented recipe for filling short player-track gaps. With no licence, the team can only reimplement it from the paper. It addresses *player* gaps and ID-switch repair, not the 41% missing ball.
- Expected size of gains on spellF1 is unknown. The closest evidence (Li & Link's tIoU 0.58 → 0.61 from adding a graph to a transformer) is small, and it is on a different task with possession given.

### Gaps
- No published comparison of a set or graph encoder against a BiGRU or TCN on per-frame possession (H/A/D-like) labels with segment-F1 scoring.
- Li & Link's frame-level baseline table was empty in the retrieved text; window length, parameter count and whether their 7 matches are the public IDSSE set are unconfirmed (TRACAB, 25 fps, 7 Bundesliga matches is consistent with IDSSE but not stated).
- SoccerNet-GAR's per-class results and whether "possession"-like classes are included were not retrieved.

## 5. Multi-agent trajectory prediction and imputation benchmarks from sports, and the state of the art there

### Takeaway
There is no single accepted sports benchmark. The active ones are UniTraj's Basketball-U / Football-U / Soccer-U (best published: AdaSports-Traj) and the Metrica / SportVU / NFL imputation suite used by MIDAS (best published there: MIDAS). For ball inference with the ball hidden, TranSPORTmer and PathCRF (already in the survey) are the soccer reference points. All the public soccer splits are tiny (3 Metrica matches; SoccerTrack drone clips), which is the same scarcity the team faces.

### Cited Findings
- **UniTraj benchmarks** (minADE20, lower is better), per-sport training: AdaSports-Traj 4.21 / 3.04 / 91.52 vs UniTraj 4.77 / 3.55 / 94.59 on Basketball-U / Football-U / Soccer-U; one model trained on all sports: 8.74 / 9.04 / 104.66 vs 11.12 / 11.89 / 124.56. Multi-sport training is *worse* than per-sport training for both models — [AdaSports-Traj](https://arxiv.org/html/2509.16095).
- **MIDAS imputation suite** (Metrica 3 matches, 100 SportVU games, NFL BDB 2021; 10 Hz): MIDAS best in most settings vs ImputeFormer, Graph Imputer, NRTSI, NAOMI, CSDI, BRITS. Soccer uniform-mask PE 1.3205 vs 2.0898. Downstream: total-distance MAPE 1.58% vs 3.22% — [arXiv HTML](https://arxiv.org/html/2408.10878).
- **Ball inference without the ball**: TranSPORTmer 2.71 m vs Ball Radar 3.89 m ball error; PathCRF 92.4% frame team-possession accuracy on IDSSE — [team survey summary, models-literature.md; TranSPORTmer arXiv](https://arxiv.org/abs/2410.17785); [PathCRF arXiv](https://arxiv.org/abs/2602.12080).
- **Off-camera player recovery**: TacSIm (CVPR 2026) reports an RC-VAE beating Social-GAN and Trajectron++ at recovering hidden players in broadcast data — [CVPR 2026 supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wen_TacSIm_A_Dataset_CVPR_2026_supplemental.pdf) (search snippet only).
- **Large-scale soccer reconstruction** (Stats Perform, CVIU 2026): transformer–diffusion; results not retrieved — [researchr](https://researchr.org/publication/HughesLHGFS26).
- **NFL 2026 Prediction** (players while the ball is in the air, RMSE): a third-party reproduction of the 5th-place recipe got test RMSE 0.80 and of the 1st-place recipe 0.4538 CV-RMSE. The protocols differ and are not comparable. A constant-velocity baseline gives 1.70 vs 0.84 for that author's own model — [AceGiqMo README](https://github.com/AceGiqMo/NFL-Big-Data-Bowl-2026).
- Surveys: no dedicated 2025–2026 survey of multi-agent trajectory modelling in team sports was found. The nearest are a 2025 Frontiers review of neural networks for football tactical analysis — [Frontiers 2025](https://www.frontiersin.org/articles/10.3389/fspor.2025.1569155/full) — and a 2025 general (pedestrian) multi-agent trajectory prediction review — [arXiv 2506.14831](https://arxiv.org/html/2506.14831v2).

### Inferences
- These benchmarks score **coordinate error**, not possession segments. Results on them say little about spellF1. The useful signal is that set-based, bidirectional, multi-task models win on small sports datasets.
- Multi-sport pretraining *hurt* per-sport accuracy in UniTraj/AdaSports (U2S worse than S2S). That is a caution against assuming cross-sport (basketball/NFL → soccer) pretraining will help the pipeline. Same-sport, same-sensor self-supervision (T-MAE style on soccer tracking) is the better bet.
- MIDAS's 3-game basketball result (1.14 vs 1.67) is a data point that the set + Bi-LSTM structure degrades gracefully with very little data, relevant to 10 training matches.

### Gaps
- No public benchmark for per-frame possession or ball-in-play labelling from tracking exists across sports. PathCRF's IDSSE split is the closest soccer one.
- Units and frame rates of Soccer-U (likely pixel coordinates from drone video) were not confirmed.

## 6. Fit for the pipeline: usable within days vs next phase (weeks+)

### Takeaway
**Within days (before the 2026-11-17 freeze): nothing new from this inventory should go into the model.** No candidate is pretrained on licence-clean soccer data, runs as-is on an M2, and outputs possession. PathCRF, already in the survey, remains the only one that does. **Next phase:** the strongest leads are (1) T-MAE-style masked-trajectory pretraining of a small set or factorized encoder on licence-clean unlabelled soccer tracking, (2) a Zoo-style or Set-Transformer slot encoder feeding the BiGRU, trained with the NFL winners' augmentation and auxiliary-loss recipe, (3) GRF synthetic pretraining with free possession and game-mode labels, and (4) a MIDAS-style track-gap imputer reimplemented from the paper.

### Cited Findings
| Item | Trained on | Weights | Licence | Runs on M2? | Possession / H/A/D output? | Fit |
|---|---|---|---|---|---|---|
| UniTraj ([repo](https://github.com/colorfulfuture/UniTraj-pytorch)) | Stats Perform NBA, NFL NGS, SoccerTrack top-view | yes (Google Drive) | code Apache-2.0; data terms unstated | no as shipped (CUDA `mamba-ssm`) | no | Next phase at best; retrain on clean data |
| AdaSports-Traj ([arXiv](https://arxiv.org/html/2509.16095)) | same as UniTraj | none stated | paper CC BY-NC-SA | — | no | Ideas only |
| T-MAE ([ML Anthology](https://mlanthology.org/mlj/2025/rudolph2025mlj-masked)) | elite soccer tracking (provider unverified) | none | no code | yes if reimplemented (small) | via fine-tuning | **Next phase, top idea** |
| TacticAI ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10951310/)) | Liverpool FC corners | none | proprietary | — | no | Idea: reflection symmetry/augmentation |
| TacticGen ([arXiv](https://arxiv.org/html/2604.18210v1)) | 1,432 matches, ~100M frames | none stated | paper CC BY-NC-SA | — | no | No |
| TacticExpert ([arXiv](https://arxiv.org/html/2503.10722v1)) | 1,780 basketball images | repo not found | unstated | no (7B LLM) | ball holder (basketball) | No |
| MIDAS ([arXiv](https://arxiv.org/html/2408.10878), [repo](https://github.com/gkswns95/midas)) | Metrica 3, SportVU 100, NFL | none | **no licence** | likely (PyTorch 1.10) | no (player imputation) | Next phase, reimplement |
| baller2vec ([repo](https://github.com/airalcorn2/baller2vec)) | NBA SportVU | none | MIT | yes (old PyTorch) | no | Idea only |
| Li & Link T-GAN ([arXiv](https://arxiv.org/html/2606.09289)) | 7 Bundesliga TRACAB | none | no code | — | no (possession given) | Evidence only |
| GRF synthetic ([docs](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md), [demo paper](https://arxiv.org/abs/2503.19809)) | simulator | n/a | Apache-2.0 (archived) | build unverified | **yes**: `ball_owned_team`, `game_mode` | Next phase, pretraining only |
| NFL BDB recipes ([NFL](https://operations.nfl.com/gameday/analytics/stats-articles/what-the-top-predictions-looked-like-in-the-nfls-big-data-bowl), [README](https://github.com/AceGiqMo/NFL-Big-Data-Bowl-2026)) | NFL NGS | Kaggle notebooks | Kaggle terms (unchecked) | yes (recipes) | no | Next phase: augmentation and pairwise block |
| FAUPA-ML handball ([record](https://www.iat.uni-leipzig.de/datenbanken/iks/dbv-base-softball/Record/4085680)) | Bundesliga handball LPS | none | — | — | phases | Evidence only |

### Inferences
- **"Days" list (only if the freeze allows any post-hoc change; otherwise move to next phase):**
  1. *Y-mirror test-time averaging*: run the frozen GRU + GBT on the original and the touchline-mirrored match (labels unchanged) and average P(H/A/D) before the decoder. It costs one extra inference pass. It is supported by TacticAI's reflection-symmetry result and the NFL winners' flip augmentation. **Unverified on this pipeline; must be checked leave-one-match-out on the bench before use.** Hand features that are not mirror-symmetric (e.g. ones tied to a camera side) would make it neutral or harmful.
  2. Otherwise: no new model. Keep PathCRF, from the existing survey, as the only pretrained ball-free clue.
- **Next phase ranking (weeks):**
  1. *Self-supervised masked-trajectory pretraining* (T-MAE recipe: factorized time × agent transformer, mask whole agent tracks) on IDSSE (CC BY 4.0) + Metrica samples + SoccerTrack v2 (CC BY 4.0) + the target match if the organisers allow. Then fine-tune a small H/A/D head, or feed its embeddings to the GBT. About 2–4 weeks on an M2 (estimate).
  2. *Slot-set front end*: Zoo-style home × away pairwise block or Set Transformer pooling into the existing BiGRU, plus y-mirror / x-mirror-with-team-swap augmentation, temporal shift, and auxiliary ball-position or velocity heads on visible-ball frames (NFL 2026 recipe; baller2vec's ball-prediction objective). About 1–2 weeks.
  3. *GRF synthetic pretraining* with free possession and dead-ball labels; sim-to-real gap large. About 1–2 weeks to generate and convert, more to validate.
  4. *Track-gap and ID-switch repair* via a MIDAS-style Set Transformer + Bi-LSTM imputer, reimplemented (no licence on the original code).
  5. *Not recommended:* cross-sport pretrained weights (UniTraj/AdaSports: licence-unclear data, CUDA dependency, multi-sport training hurt per-sport accuracy), LLM-based models (TacticExpert), closed models (TacticAI, TacticGen, Stats Perform).
- **Licence posture:** the only items that are clean for commercial partners *and* organiser-accessible are reimplementations from papers, Apache-2.0 / MIT code with no third-party weights (UniTraj code, baller2vec, GRF), and training on CC BY data (IDSSE, SoccerTrack v2). Any third-party checkpoint trained on Stats Perform, NGS/Kaggle, PFF (SoccerNet-GAR) or Liverpool data should be treated as blocked until its terms are checked.

### Gaps
- None of the next-phase items has been tested on the team's bench; the effort estimates are judgement, not measurement.
- Whether a pure-PyTorch Mamba fallback would let UniTraj run on MPS was not checked.
- Whether competition rules allow self-supervised pretraining on the unlabelled target match is an open organiser question (also raised in extensions.md).
