# Soccer possession, ball-in-play and on-ball events from tracking: what is new since late 2024 (searched 2026-11-12)

Scope note: the team survey of 2026-10-30 (`docs/research/models-literature.md`) already covers PathCRF, BallRadar, TranSPORTmer, Peral et al. 2025, Lang et al. 2022, Link & Hoernig 2017, Vidal-Codina 2022, SoccerNet BAS 2023-24, T-DEED, WASB, ETSY, the ELASTIC repo, FOOTPASS (dataset paper) and the 2025 TBAS/GSR headline scores. Those appear below only where something new was found (new venue, new version, new numbers, new licence facts). Evidence-quality tags used: **[peer-reviewed]**, **[preprint]**, **[challenge report]**, **[snippet]** (only seen in a search-result summary, not in the source itself).

Pipeline under judgement: per-frame H/A/D for one 25 fps Metrica automatic match (ball missing ~41 %, ID switches), spellF1 over IoU 0.3–0.7; BiGRU + GBT on ~97 features incl. PathCRF P, min-length Viterbi, learned post stages; leave-one-match-out on 7 DFL/IDSSE + 3 Metrica samples (damaged) and 20 SkillCorner matches as test; M2, no NVIDIA GPU; training data must have clear commercial-compatible rights; model frozen for 17 Nov 2026.

## Q1. New methods for team possession, ball carrier, or ball-in-play/dead status from player trajectories (ball absent or unreliable)

### Takeaway
No published successor to PathCRF that does ball-free possession better has appeared; PathCRF (now formally KDD 2026) is still the newest ball-free possession model, and still only covers in-play episodes. The genuinely new items are adjacent: (a) a trajectory-only ball-action-spotting baseline on a new CC BY 4.0 dataset (SoccerTrack v2) that quantifies how much the ball matters (mAP@1 s 0.22 without ball vs 0.60 with), (b) a tracking-only group-activity benchmark (SoccerNet-GAR) where restart classes (Out, Free kick) are recognised at ~86 % from tracking with 32 % ball missing, and (c) phase-segmentation papers from the Link group and others that use possession as an *input*, not an output. No new paper on in-play/dead-ball classification from player positions alone (a Lang et al. 2022 successor) was found.

### Cited Findings
- **PathCRF is now published at KDD 2026** (DOI 10.1145/3770855.3818152) — [ACM DOI](https://doi.org/10.1145/3770855.3818152). Repo still MPL-2.0, last push 2026-06-08 (checked via GitHub API) — [GitHub](https://github.com/hyunsungkim-ds/pathcrf). **[peer-reviewed]**
- PathCRF v2 split on the Sportec Open DFL data: train 5 matches (364,003 frames), val 1 match, test 1 match (85,304 frames); frame-level team-possession accuracy 92.40 % on the single test match; possession share 62.64 % predicted vs 62.13 % true; inference restricted to in-play episodes ("no on-ball events occur during stoppages"); out-of-play handled only via 4 outside nodes with absorbing self-loops; assumes 22 players (no red cards). A robustness/generalisation check on "an additional non-public dataset" is in Appendix B (not read) — [arXiv 2602.12080v2](https://arxiv.org/html/2602.12080v2)
- PathCRF picks the correct possession edge among 26² = 676 candidates with 69.64 % accuracy; event F1 75.69 % — [arXiv 2602.12080](https://arxiv.org/html/2602.12080) **[snippet]**
- PathCRF's stated future work: partial human guidance for ambiguous events, broadcast video features, finer kick categories — no noise/ID-switch experiment visible in main text — [arXiv 2602.12080v2](https://arxiv.org/html/2602.12080v2)
- **Peral et al. (masked pre-training, ball state + possessor from players + crops) is now WACV 2026** — [CVF open access](https://openaccess.thecvf.com/content/WACV2026/html/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.html). Numbers unchanged from the survey (ball-state acc 85.85 %, possessor acc 89.56 %); no code. **[peer-reviewed]**
- **Capellera, Rubio, Ferraz, Agudo, "Heteroscedastic Diffusion for Multi-Agent Trajectory Modeling"**, arXiv 2605.10717 (11 May 2026), accepted to IEEE TPAMI; extension of their CVPR 2025 paper (arXiv 2503.18589). Tasks: trajectory completion and forecasting with uncertainty on NBA, Basketball-U, Football-U, Soccer-U. Abstract gives no ball-specific or possessor numbers and no code link — [arXiv 2605.10717](https://arxiv.org/abs/2605.10717) **[peer-reviewed, abstract only read]**
- **SoccerTrack v2** (Scott, Uchida, Kuroda, Kim, Fujii; arXiv 2508.01802, v2): 10 university-level matches, 932 min at 25 fps, fixed 4K panoramic video, per-frame pitch coordinates, jersey-based persistent IDs, roles, team side; 21,428 ball-action events in 12 classes, 99.5 % attributed to a player in the same ID space; a per-frame ball-status flag exists (excluded from the baseline's features); "Out" is synthesised from restart events — [arXiv 2508.01802v2](https://arxiv.org/html/2508.01802v2) **[preprint]**
- SoccerTrack v2 trajectory-only BAS baseline: offline MS-TCN-style TCN, 6 dilated residual blocks (receptive field ±25 s), 82 player-trajectory features at 5 Hz (+19 ball features in the "+ball" condition), Gaussian targets (SD 300 ms). 5-fold cross-match mAP@1 s: chance 0.008, **trajectory only 0.222 ± 0.021**, trajectory + ball 0.599 ± 0.064; mAP@5 s 0.524 vs 0.666. Per-class AP@1 s (traj → +ball): Pass 0.286 → 0.853, Drive 0.227 → 0.830, Throw-in 0.254 → 0.910, **Out 0.148 → 0.364**, Tackle 0.046 → 0.121 — [arXiv 2508.01802v2](https://arxiv.org/html/2508.01802v2)
- SoccerTrack v2 licences: dataset (video + GSR/BAS/MOT annotations, Hugging Face + Google Drive) **CC BY 4.0** incl. commercial use; code **MIT**; collected under university ethics approval, no player names — [LICENSE-DATA](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA), [README](https://github.com/AtomScott/SoccerTrack-v2). Hugging Face card reports `cc-by-4.0`, gated with automatic approval (checked via HF API) — [HF dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2). Repo also contains a rule-based `scripts/event_detection_tracking/event_detection.py` (seen via GitHub code search, not read).
- **SoccerNet-GAR** ("Pixels or Positions?", arXiv 2511.12606, Nov 2025, KAUST + Liège): 64 FIFA World Cup 2022 matches, 94,285 group activities (HF card: 87,939 filtered), 10 classes (Pass, Tackle, Out, Header, High pass, Throw-in, Cross, Free kick, Shot, Goal); tracking from PFF FC (broadcast-derived, manually refined) at 30 fps with 3D ball, **ball missing in 32.4 % of frames**; role-aware GNN (GIN + attention + positional edges, 197K params) gets 67.2 % balanced accuracy vs 58.1 % for the best video model; per-class: Out 85.7, Free kick 86.5, Throw-in 65.6, Pass 70.1, Tackle 50.7. No ablation without ball — [arXiv 2511.12606v1](https://arxiv.org/html/2511.12606v1) **[preprint]**
- SoccerNet-GAR licence: HF dataset is gated (manual approval) with licence field "other" (checked via HF API) — [HF](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR); a related card lists CC-BY-NC-4.0 — [HF card](https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-GAR/blob/main/README.md) **[snippet]**; code repo has no licence file (GitHub API: `license: null`) — [GitHub](https://github.com/drishyakarki/pixels_vs_positions)
- **Li & Link, "Intention Driven Identification of In-Possession Match Phases ... through Temporal Graph Learning"**, arXiv 2606.09289 (v4 6 Nov 2026): hierarchical segmentation into 3 intentions / 6 in-possession phases with a temporal graph attention network; 7 Bundesliga matches, leave-one-match-out, frame- and sequence-level evaluation; no numbers in abstract; no code — [arXiv 2606.09289](https://arxiv.org/abs/2606.09289) **[preprint]**
- **Renner, Koster, Bauer, Schienle, "Automated Detection of Match Phases in Football ... Using Graph Neural Networks"**, arXiv 2610.11571 (8 Oct 2026): 7-class match phases second by second on 203 matches; GNN-LSTM with Delaunay adjacency; possession and ball-status indicators are *inputs*; strongest baseline 4.6 % lower macro F1; no code — [arXiv 2610.11571](https://arxiv.org/abs/2610.11571) **[preprint]**
- No 2025–2026 hockey or basketball paper on ball/puck possession from player tracking only surfaced in search — [search result summary pointing only to soccer work, e.g. PathCRF](https://arxiv.org/html/2602.12080)

### Inferences
- Nothing published in 2025–2026 replaces PathCRF as the ball-free possession prior; the PathCRF feature already in the model remains the state of the art input. Its weak spots for us (in-play only, 22 continuous tracks, 1-test-match evidence, no ID-switch study) have not been addressed in the published v2.
- SoccerTrack v2 numbers give a useful calibration: without the ball, touch-level timing from trajectories is poor (mAP@1 s 0.22) but at ±5 s it is 0.52. This matches the pipeline's design choice of using players for *segment* evidence and the ball (when present) for *boundary* timing.
- SoccerNet-GAR suggests restart types (Out, Free kick) are well separated from tracking at the clip level even with ~1/3 ball missing — supporting dead-ball evidence from player configuration, but its data/code licences make it unusable for training under the competition rules.
- The Link group (Li & Link 2026) evaluates on 7 Bundesliga matches with LOMO — almost certainly the same IDSSE matches the team uses (inference from the count; not confirmed).

### Gaps
- No 2025–2026 paper found that classifies ball-in-play vs dead from player positions alone (successor to Lang et al. 2022); search terms "ball in play dead ball detection tracking 2025/2026" returned none.
- PathCRF Appendix B (non-public robustness dataset) was not read (page truncated); whether it covers noisy/automatic tracking is unknown.
- SoccerTrack v2's per-frame ball-status flag: present in the data per the paper, but its exact field name and quality were not verified (HF card gated to unauthenticated fetch).
- Capellera et al. TPAMI: whether the soccer experiments include the ball or possessor, and any code release, not verifiable from the abstract.

## Q2. What is new in ball-trajectory imputation / inference from player context (soccer), and how large are errors in long gaps (≥10 s)?

### Takeaway
Little that is both new and usable: the 2025–2026 work is diffusion-based multi-agent completion (Capellera et al., TPAMI 2026), event-conditioned reconstruction (Hughes/Lucey et al., CVIU 2026; AgentImputer, CIKM 2025) and the earlier MIDAS imputer — none with code we could find for the ball case, and **none reports ball error as a function of gap length, so no source gives ≥10 s gap errors**.

### Cited Findings
- **Hughes, Lucey, Horton, Gammulle, Fookes, Sridharan, "Multimodal transformer–diffusion framework for large-scale reconstruction of soccer tracking data"**, Computer Vision and Image Understanding 264:104626 (2026) — [researchr](https://researchr.org/publication/HughesLHGFS26), [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1077314225003492). Uses tracking + event annotations as long-term context, then a diffusion model to reconstruct player and ball trajectories; reports 98 % of reconstructed passes "photorealistic" vs 82 % without the diffusion step — **[snippet]** (ScienceDirect returned 403). Same first author's earlier AAAI work (Event2Tracking) fuses event data with tracking — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33289/35444) **[snippet]**
- **AgentImputer** (CIKM 2025), "Imputing Multi-Agent Trajectories from Event and Snapshot Data in Soccer": reconstructs player positions from event data + snapshots with LSTM/GNN modules, not from tracking — [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3746252.3760868) **[snippet]**
- **MIDAS**, "Trajectory Imputation in Multi-Agent Sports with Derivative-Accumulating Self-Ensemble" (arXiv 2408.10878, v3): multi-agent imputation under uniform and agent-wise missingness with physical plausibility; search summary did not show ball-specific results — [arXiv 2408.10878v3](https://www.arxiv.org/pdf/2408.10878v3) **[snippet]**
- Capellera et al. TPAMI 2026: completion and forecasting with heteroscedastic (per-agent, per-step) uncertainty on Soccer-U and Football-U; no ball-error numbers or code on the abstract page — [arXiv 2605.10717](https://arxiv.org/abs/2605.10717)
- Peral et al. (WACV 2026) ball error 1.17 m in its windows (already in survey) — [CVF](https://openaccess.thecvf.com/content/WACV2026/html/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.html)
- Cautionary data point: SoccerTrack v2's released ball track is itself an imputation — "straight lines at constant speed between anchors", with most anchors being annotated events; two matches have ball/players clamped to the pitch rectangle — [arXiv 2508.01802v2](https://arxiv.org/html/2508.01802v2)
- CVsports 2026 (CVPR 2026) includes "Physics-Based 3D Ball Trajectory Reconstruction from Monocular Soccer Video: A Multi-Model Benchmark" (Grad, Czajkowski, Varashylau) — [CVsports 2026 listing](https://openaccess.thecvf.com/CVPR2026_workshops/CVsports) **[snippet]** (page returned 403)
- Bhatt et al., Royal Society Open Science 12(10):251175 (Oct 2025), interpolate continuous full-pitch *player* positions from 1 Hz broadcast-like samples restricted to players within 30 m of the ball — [RSOS](https://royalsocietypublishing.org/rsos/article/12/10/251175/236076/Continuous-football-player-tracking-from-discrete) **[snippet]**

### Inferences
- For the pipeline's ~41 % missing ball, the practical options remain the ones in the survey (PathCRF path as a surrogate for the ball; physical gap-filling between sightings). Diffusion completion models would give uncertainty bands, which would be a natural per-frame confidence feature, but without code and with GPU-scale training they are next-phase at best.
- Event-conditioned reconstruction (Hughes et al., AgentImputer) needs event data for the target match; the competition match does not come with events, so these are not applicable except for building cleaner training data for *other* matches.
- Because no paper reports error vs gap length, the team's own bench (damaged IDSSE/Metrica with known truth) remains the only way to quantify long-gap behaviour.

### Gaps
- No source found reporting ball-position error or possessor accuracy specifically for gaps ≥10 s (in any 2025–2026 paper).
- Full text of Hughes et al. (CVIU 2026) not accessible (403); data source (likely Stats Perform, unverified), ball-specific errors and code status unknown.
- MIDAS authors/venue and whether it imputes the ball were not verified.

## Q3. SoccerNet 2025 and 2026 challenges relevant here (GSR, BAS/TBAS, PCBAS/FOOTPASS, GAR)

### Takeaway
SoccerNet 2026 dropped GSR and TBAS and introduced **Player-Centric Ball Action Spotting (PCBAS)** on FOOTPASS (who-did-what with team + jersey), won by a video-plus-game-state ensemble at 58.94 Macro-F1; the 2nd-place entry is the only one that explicitly uses tracking rows, ball detections and **possession heuristics** as features. All of it is broadcast-video based, data is gated under a non-standard licence, and no winner code was linked — so nothing here is usable for training under the competition rules; at most it offers method ideas.

### Cited Findings
- **SoccerNet 2025** tasks: Team Ball Action Spotting, Monocular Depth Estimation, Multi-View Foul Recognition, Game State Reconstruction — [arXiv 2508.19182](https://arxiv.org/abs/2508.19182) **[challenge report]**
- **TBAS 2025** (baseline 51.72 Team-mAP@1): 1st dudek (K. Czarnogórski) 60.03 — T-DEED re-implemented in the "dude.k" framework with a single head predicting joint team×action classes, mirroring with label swap, two-phase training, Bayesian tuning; 2nd Intellindust-AI-Lab 56.78 — mixes SoccerNet/SNB data, class-specific ensembles and "team alignment" post-processing that corrects team labels from preceding related actions; 3rd KIST-BAS 56.08 (no summary). None of the described TBAS methods uses tracking/game-state data; no code links in the report — [arXiv 2508.19182 (HTML)](https://arxiv.org/html/2508.19182)
- **GSR 2025** (baseline GS-HOTA 29.01): KIST-GSR 63.90 (YOLO-X, Deep-EIoU + OSNet ReID, multi-frame keypoint pitch mapping, LLaMA-3.2-Vision for role/jersey, tracklet split-and-merge, jersey-colour team clustering); Constructor.Tech 63.81; lianyou 62.76 — [arXiv 2508.19182 (HTML)](https://arxiv.org/html/2508.19182)
- **SoccerNet 2026** (report arXiv 2607.07320, accepted to ACCV 2026): five tasks — Ball Action Anticipation (BAA), Player-Centric Ball Action Spotting (PCBAS), Novel View Synthesis, Spiideo SoccerNet SynLoc, VQA; GSR not among them; 427 teams, 1,129 entries — [arXiv 2607.07320](https://arxiv.org/html/2607.07320v1) **[challenge report]**
- **PCBAS 2026**: built on FOOTPASS; 8 classes (Drive, Pass, Cross, Shot, Header, Throw-in, Tackle, Block); metric Macro F1@0.15 — true positive only if within ±12 frames and class, team and jersey number all match; baseline TAAD + DST 46.41; 6 teams, 124 submissions. 1st PAVE (Altawijri & Mathkour) 58.94 — temporal transformer on ROI-aligned X3D player features, per-player attention in DST, 4-variant ensemble with agreement filtering; 2nd PC-SSAS (Linkov et al.) 56.40 — Mamba state-space backbone over player tokens fused with frozen visual embeddings, using sparse tracking rows, ball detections, tactical player states and possession heuristics as features; 3rd TeamKIST 55.69; no code links given — [arXiv 2607.07320](https://arxiv.org/html/2607.07320v1)
- PCBAS team reports on arXiv: "Per-Player Attention with Agreement-Based Ensembling" (2606.28389) — [arXiv](https://arxiv.org/pdf/2606.28389); "Retraining and Post-Processing Extensions to the FOOTPASS Baselines" (2606.09679) — [arXiv](https://arxiv.org/pdf/2606.09679) **[preprint, not read]**
- **BAA 2026**: observe 30 s, predict 10 classes in next 5 s; 7 public EFL matches + 2 private; baseline 16.76 mAPavg; 1st FAANTRA-WS 24.08; one entry (VLM-TCF) extracts possession descriptors with a VLM — [arXiv 2607.07320](https://arxiv.org/html/2607.07320v1)
- **FOOTPASS** (Ochin et al., arXiv 2511.16183): 54 full broadcast matches, 102,992 human-annotated on-ball events, player identities, tracklets and game-state variables — [arXiv 2511.16183](https://arxiv.org/pdf/2511.16183). The 2026 PCBAS data on Hugging Face (`SoccerNet/SN-PCBAS-2026`) is gated with manual approval and licence "other" (HF API check) — [HF](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026)
- FOOTPASS precursor: Ochin, Devineau, Stanciulescu, Manitsaris, "Game State and Spatio-Temporal Action Detection in Soccer Using GNNs and 3D CNNs", ICPRAM 2025, pp. 636–646 — GNN over surrounding players' positions/velocities/teams fused end-to-end with a 3D CNN improves over video alone; paper licensed CC BY-NC-ND 4.0 — [SciTePress PDF](https://scitepress.org/Papers/2025/131611/131611.pdf)
- **ME-DST** (Wang, Yang, Wang; arXiv 2608.01696, Aug 2026): keeps per-role-slot sequences, temporal + spatial attention, role embeddings, tracking-derived tactical features fused with X3D-L/Swin3D-S predictions; Micro F1 0.778 on FOOTPASS, +10.3 points over TAAD + DST; no code link — [arXiv 2608.01696](https://arxiv.org/abs/2608.01696) **[preprint]**
- SoccerNet-GAR (Q1) is the SoccerNet group-activity dataset; it is not listed as a 2026 challenge task in the report — [arXiv 2511.12606](https://arxiv.org/html/2511.12606v1), [arXiv 2607.07320](https://arxiv.org/html/2607.07320v1)
- GSR baseline repo remains GPL-3.0 (last push 2026-05-02, GitHub API) — [GitHub](https://github.com/SoccerNet/sn-gamestate)

### Inferences
- The SoccerNet line has moved from "what happened" to "who did it" (PCBAS) and "what happens next" (BAA). Neither produces per-frame team possession or ball-in-play labels, and all inputs are broadcast video. Fit for the pipeline is low.
- One transferable idea: Intellindust's "team alignment" post-processing (fix the team of an action using the preceding related action) mirrors the pipeline's per-possession check; and PC-SSAS shows possession heuristics help a learned event model — consistent with feeding hand-made possession features into the GRU/GBT.
- Licences: SN-PCBAS-2026 and SoccerNet-GAR are gated "other"/NC; under the competition's commercial-partner rule, treat both as unusable for training.

### Gaps
- No code repositories were found for the TBAS 2025 winner (dude.k) or any PCBAS 2026 top-3 entry (GitHub search returned nothing for "dude.k soccernet").
- The exact licence text of SN-PCBAS-2026 / FOOTPASS was not read (gated).

## Q4. Event-tracking synchronisation and automatic event detection from tracking: anything that yields per-frame possession labels usable as training targets?

### Takeaway
ELASTIC is now a CIKM 2026 paper with MPL-2.0 code and, importantly, a **CC BY 4.0 benchmark of 3 Sportec/IDSSE matches with frame-accurate event frames and ball-receive frames** — the cleanest new touch-level ground truth for the pipeline's own bench. DataBallPy (MIT, JOSS 2026) is a second open synchroniser. New tracking-based event detection (Mills et al. 2026; Wang et al. 2025) is FIFA/MIT work on World Cup data, rule-based or skeletal, with no code; it *consumes* a provider live/dead flag rather than inferring it. SoccerTrack v2 is the one new open dataset from which per-frame H/A/D-like labels could be derived for 10 more matches under CC BY 4.0.

### Cited Findings
- **ELASTIC workshop version**: arXiv 2508.09238 (12 Aug 2025), ECML PKDD 2025 MLSA workshop (Porto); 2,134 events with ground-truth timestamps on 3 Eredivisie matches; also detects pass end-points and separates major/minor events — [arXiv 2508.09238](https://arxiv.org/abs/2508.09238v1) **[workshop]**
- **ELASTIC extended version**: "ELASTIC: Trajectory-Based Synchronization of Event and Tracking Data in Soccer", Kim, Choi, Lee, Seo, Boomstra, Yoon, Park, CIKM 2026 (DOI 10.1145/3799682.3841035), arXiv 2608.30227 — [arXiv 2608.30227](https://arxiv.org/abs/2608.30227) **[peer-reviewed]**
- ELASTIC method: infers event start/end from player **and ball** trajectories only (ignores annotated event locations); extracts candidate touch frames from ball acceleration, player-ball distance and kick distance; aligns events to candidates per in-play segment with Needleman-Wunsch; compared against ETSY, Biermann, DataBallPy and a greedy variant — [ELASTIC README](https://github.com/hyunsungkim-ds/elastic)
- ELASTIC benchmark: 3 Sportec Open DFL matches (J03WMX, J03WN1, J03WPY) re-annotated frame by frame by three annotators (median taken), false-positive events removed and missing ones inserted; `benchmark/gt/` gives `frame_id` (event moment) and `receive_frame_id` (ball reception); event data redistributed under **CC BY 4.0**; code **MPL-2.0** (repo last push 2026-09-07) — [ELASTIC README](https://github.com/hyunsungkim-ds/elastic)
- PathCRF's own training events were re-synchronised with ELASTIC (survey) — [PathCRF repo](https://github.com/hyunsungkim-ds/pathcrf)
- **DataBallPy** (Oonk, Grob, Kempe), JOSS 2026 (doi 10.21105/joss.10223): Needleman-Wunsch synchronisation with a sigmoid-scaled cost on time difference and ball-to-event distance; naive timestamp sync made combined tracking+event xG features near useless; v0.7.0 (4 Nov 2025) added a kloppy integration — [JOSS](https://joss.theoj.org/papers/10.21105/joss.10223), [docs](https://databallpy.readthedocs.io/en/latest/getting_started/synchronisation_page.html). Licence MIT, last push 2026-11-03 (GitHub API) — [GitHub](https://github.com/alek050/databallpy)
- ETSY is dated 2023 (Van Roy et al.) in the ELASTIC README — [README](https://github.com/hyunsungkim-ds/elastic); one search summary placed it in MLSA 2025 proceedings (pp. 41–54) — [researchr](https://researchr.org/conferenceseries/mlsa/publications) **[snippet, likely a mis-attribution]**. Repo licence field "NOASSERTION", last push 2024-05-13 — [GitHub](https://github.com/ML-KULeuven/ETSY)
- **Mills et al., "Automatic event detection in association football using broadcast-derived tracking data"**, Sports Engineering 29(1):15 (2026), FIFA + Sheffield Hallam (+ MIT Sports Lab co-authors), open access CC BY: one 90-min 2022 World Cup match; broadcast-derived auto-events, particularly from Camera 1, can approach multi-camera optical accuracy for certain events; the algorithm takes a frame-by-frame live/dead status as input and flags set pieces at dead→live switches; poor ball tracking in the penalty box can misattribute goals; missing ball height impaired goalkeeper-action detection; FIFA-validated tracking systems rose from <5 (2018) to >40 (2025) — [SHURA record](https://shura.shu.ac.uk/37410/), [Springer](https://link.springer.com/article/10.1007/s12283-026-00549-4) **[peer-reviewed; details partly from snippets, full text blocked]**
- **Wang, Mills, Billingham, Robertson, Hosoi, "Semi-automated last touch detection for out-of-bounds possession decisions in football"**, Sports Engineering 28(2):36 (2025): touch-probability model for the last-touch frame + rules/supervised player attribution; skeletal + ball optical tracking; World Cup 2022 duels; up to 82.5 % correct restart-possession decisions, >90 % for aerial duels — [LIDA record](https://lida.sport-iat.de/dfb/Record/4094620) **[peer-reviewed, abstract]**
- SoccerTrack v2 events carry player IDs (99.5 %) and team side, plus a per-frame ball-status flag, under CC BY 4.0 — [arXiv 2508.01802v2](https://arxiv.org/html/2508.01802v2), [LICENSE-DATA](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA)
- Assist identification from event + tracking data (Sports Engineering 2025) used SVM/k-NN/RF positional features for passes, receptions and touches — [Springer](https://link.springer.com/article/10.1007/s12283-025-00533-4) **[snippet]**

### Inferences
- ELASTIC's benchmark does not add new matches (the 3 are already in the IDSSE bench), but it gives **frame-accurate touch and reception frames**, which DFL's own possession flags and event timestamps do not guarantee. That is directly the target for the pipeline's "change timing" and "short possessions at touches" stages, and for checking whether IDSSE's per-frame possession flags themselves lag the true touch.
- ELASTIC and DataBallPy both need the ball trajectory, so they cannot run on the competition match (41 % ball missing); their role is to make *external* training labels sharper.
- Per-frame team-possession targets derivable from new open sources: (1) SoccerTrack v2 (team-attributed events + ball-status flag, 10 matches, CC BY 4.0 — but amateur, annotated rather than automatic tracking, and its ball track is interpolated between events); (2) ELASTIC-synced events on IDSSE (forward-fill possession from each reception to the next pass). Both are clear-rights.
- The FIFA/MIT line treats live/dead as a provider input — confirming there is still no off-the-shelf, validated dead-ball inferer from tracking; the pipeline's own learned D class remains necessary.

### Gaps
- ELASTIC's CIKM numbers (synchronisation accuracy per event type, gains over ETSY/DataBallPy) were not read — arXiv abstract gives none, full paper not fetched.
- Mills et al. 2026 full results (per-event precision/recall) not read (Springer and SHURA PDF blocked by bot protection).
- No 2025–2026 deep-learning event detector trained on tracking only (successor to Vidal-Codina 2022) with public code was found; search summaries describe the area as still mostly rule-based/classical ML.

## Q5. Automatic (uncorrected) optical tracking artefacts: vendor handling of the ball in dead time, data-quality studies, robustness to noise

### Takeaway
Public evidence remains thin and indirect. The useful new quantitative points are: PFF FC broadcast tracking (manually refined) still has the ball missing on 32.4 % of frames; one SkillCorner open-data match has 17.7 % of ball positions and 40.7 % of player records extrapolated, and SkillCorner exposes `is_detected` and a per-frame possession field but defines phases only while the ball is in play; Metrica's ball position has been observed to equal the possessor's position. No vendor documentation on how Metrica's automatic tracking represents the ball during dead time was found.

### Cited Findings
- SoccerNet-GAR's PFF FC tracking (automatically extracted from broadcast, manually refined) has the ball missing in 32.4 % of frames — [arXiv 2511.12606v1](https://arxiv.org/html/2511.12606v1)
- SkillCorner: the ball carries an `is_detected` flag; in open-data match 1886347, 40.7 % of player records and 17.7 % of ball positions are extrapolated — [kloppy issue #639](https://github.com/PySport/kloppy/issues/639). The open-data tracking file `{id}_tracking_extrapolated.jsonl` includes a per-frame `possession` field (player and team) — [SkillCorner opendata](https://github.com/SkillCorner/opendata). SkillCorner phases of play are defined only when the ball is in play — [SkillCorner opendata](https://github.com/SkillCorner/opendata) **[snippet]**
- Excluding SkillCorner's flagged (fully extrapolated) events left a possession-duration vs turnover finding essentially unchanged in one Kaggle write-up — [Kaggle write-up](https://www.kaggle.com/competitions/soccer-feature-engineering-hackathon/writeups/50-aggregated-features-from-skillcorner-tracking-d) **[snippet, single analyst]**
- PLOS ONE study "Event detection in football: Improving the reliability of match analysis": Metrica and Subsequent produce positional data from a single video and are expected to be of lower quality; in the Metrica data set the ball position is typically exactly equal to the position of the player in possession — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0298107) **[peer-reviewed 2024, snippet]**
- "Broadcast analytics — an evaluation of video-based tracking systems with constrained player visibility" (German Sport University Cologne; Science and Medicine in Football 2025): ~97 % of player activity visible in a scouting feed vs under half in TV; visibility gaps distort physical-intensity metrics while formation recognition holds up — [T&F](https://www.tandfonline.com/doi/abs/10.1080/24733938.2025.2533808), [DSHS](https://fis.dshs-koeln.de/en/publications/broadcast-analytics-an-evaluation-of-video-based-tracking-systems/) **[snippet]**
- Mills et al. 2026: auto-event accuracy is tied to tracking quality; ball errors in the box and missing ball height are the named failure modes — [SHURA](https://shura.shu.ac.uk/37410/) **[snippet]**
- A 2026 preprint using PFF World Cup data notes the lack of a validation study for that system as a limitation — [arXiv 2606.19931](https://arxiv.org/pdf/2606.19931) **[snippet]**
- Older context (survey-level): some providers signal dead ball with a boolean while others imply it via missing ball data; missing ball during play and an out-of-sync dead-ball signal are error sources — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804)

### Inferences
- The SkillCorner and PFF figures (17.7 % extrapolated ball; 32.4 % missing ball) bracket the pipeline's 41 % missing-ball regime from below; the competition match is harsher than any published-quality dataset found.
- If Metrica's tracker snaps the ball to the possessor when it is present (PLOS ONE observation, older Metrica data), "ball present and at a player" frames may be partly tracker-imputed, i.e. ball presence itself is a possession clue but its position may be biased toward the last tracked carrier — worth a check on the team's Metrica samples (inference, not verified on the competition data).
- No paper found studies possession inference robustness to identity switches specifically; the pipeline's own damage-and-test bench remains the only evidence.

### Gaps
- No Metrica, SkillCorner, Sportec or Hawk-Eye technical note was found describing how the ball is reported during dead time in automatic tracking (e.g. held at last position, dropped, or snapped to the restart taker).
- No 2025–2026 study quantifying single-camera automatic ball-detection rates by game state (in play vs dead) was found.

## Q6. Public leaderboards, workshops and conferences 2025–2026 with relevant possession work

### Takeaway
The relevant venues in 2025–2026 were KDD 2026 (PathCRF), CIKM 2026 (ELASTIC), MLSA 2025 (ELASTIC workshop version), WACV 2026 (Peral et al.), TPAMI (Capellera et al.), ACCV 2026 (SoccerNet 2026 report), and MMSports 2025 (SoccerTrack Challenge, player MOT only). No possession-specific leaderboard exists; the closest public leaderboards are SoccerNet PCBAS/BAA (video) and SoccerTrack v2 BAS (trajectory-based).

### Cited Findings
- MLSA 2025 (ECML PKDD, Porto): ELASTIC first presented there; proceedings accepted 11 full papers from 24 submissions; MLSA 2026 CfP: submission 5 June 2026, workshop 7 or 11 Sept 2026 — [researchr MLSA](https://researchr.org/conferenceseries/mlsa/publications), [CfP](https://groups.google.com/g/ml-news/c/YOVJDCgGYTk) **[snippet]**
- CVsports 2026 (CVPR 2026): invited talk by Maaike Van Roy (KU Leuven) "Analyzing Soccer Actions and Tactics by Learning and Reasoning"; SoccerNet challenge results session; 3D ball trajectory reconstruction benchmark paper — [CVsports 2026 program](https://vap.aau.dk/cvsports/12th-international-workshop-on-computer-vision-in-sports-cvsports-at-cvpr-2026/program/) **[snippet]**
- MMSports 2025 (Dublin, 27–28 Oct 2025): SoccerTrack Challenge 2025 (fixed-view player MOT) won by GTATrack, HOTA 0.60 — [arXiv 2602.00484](https://arxiv.org/html/2602.00484), [STC 2025](https://sites.google.com/g.sp.m.is.nagoya-u.ac.jp/stc2025); no possession paper found in MMSports 2025 searches.
- SoccerTrack v2 has a released split and evaluation tooling for BAS from trajectories (MIT code) — [GitHub](https://github.com/AtomScott/SoccerTrack-v2)
- SSAC 2026: KAIST's DEFCON ("Better Prevent than Tackle: Valuing Defense in Soccer Based on Graph Neural Networks") — defence valuation, not possession inference; repo has no licence — [GitHub](https://github.com/hyunsungkim-ds/defcon)
- SoccerNet 2026 report accepted to ACCV 2026 — [arXiv 2607.07320](https://arxiv.org/html/2607.07320v1)

### Inferences
- The KAIST group (PathCRF, ELASTIC, BallRadar, DEFCON) is the most active source of usable, licensed code in this niche; watching their GitHub is the highest-yield monitoring action.

### Gaps
- MIT Sloan 2026 research-paper list not found via search; no possession-inference SSAC paper identified.
- StatsBomb/Hudl conference talks and the Big Data Cup (hockey) were not covered by any search result; nothing relevant surfaced.
- Full CVsports 2026 and MMSports 2025 paper lists not retrieved (403 / not in results).

## Triage for the pipeline: usable within days vs next phase (weeks+)

### Takeaway
With the model frozen on 17 Nov 2026 (5 days from this search), only evaluation-side uses are realistic: the ELASTIC frame-accurate benchmark to audit change timing on 3 IDSSE matches, and possibly SoccerTrack v2 as an extra held-out test. Everything else (retraining with SoccerTrack v2, ELASTIC-synced labels, trajectory-TCN touch detector, diffusion imputation) is next-phase; SoccerNet-GAR and PCBAS/FOOTPASS are excluded on licence grounds.

### Cited Findings
- ELASTIC benchmark: CC BY 4.0 event data with `frame_id`/`receive_frame_id` for 3 Sportec matches; MPL-2.0 code; tutorial notebook loads Sportec XML — [ELASTIC README](https://github.com/hyunsungkim-ds/elastic)
- SoccerTrack v2: 10 matches, 25 fps, CC BY 4.0 data / MIT code, team-attributed events, per-frame ball-status flag, ball track interpolated between events — [arXiv 2508.01802v2](https://arxiv.org/html/2508.01802v2), [LICENSE-DATA](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA)
- SoccerTrack v2 trajectory-only TCN: mAP@1 s 0.222 vs 0.599 with ball — [arXiv 2508.01802v2](https://arxiv.org/html/2508.01802v2)
- DataBallPy: MIT, kloppy integration — [GitHub](https://github.com/alek050/databallpy)
- SoccerNet-GAR: gated, licence "other"/NC, code unlicensed — [HF](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR), [GitHub](https://github.com/drishyakarki/pixels_vs_positions)
- SN-PCBAS-2026: gated, licence "other" — [HF](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026)
- PathCRF: no new weights or version beyond what the survey used (last push 2026-06-08) — [GitHub](https://github.com/hyunsungkim-ds/pathcrf)

### Inferences
**Usable within days (evaluation only, no change to the frozen model):**
1. **ELASTIC benchmark as a timing audit** (0.5–1 day): on J03WMX/J03WN1/J03WPY, compare predicted H/A change frames against `receive_frame_id`/`frame_id` instead of DFL's per-frame flags; measures whether the "change timing" and "short possessions at touches" stages are biased early/late, and whether IDSSE's own possession flags lag. Clear rights (CC BY 4.0, MPL-2.0). Caveat: same 3 matches as the training bench, so it is a label-precision check, not an independent test.
2. **SoccerTrack v2 as an extra out-of-domain test** (2–3 days, borderline before 17 Nov): derive H/A/D from team-attributed events + ball-status flag, damage to 41 % ball missing, run the frozen model. Risks: amateur play, fixed panoramic camera, annotated (clean) player tracks, synthetic ball between events, gated download (auto-approval), and label derivation choices. Only worth it if the loader is quick; otherwise next phase.
3. No change is suggested for PathCRF: no newer release exists.

**Next phase (weeks+):**
- Add SoccerTrack v2 (10 matches, CC BY 4.0) to training for the D class and restarts, using its ball-status flag; check domain shift against SkillCorner/Metrica test matches first.
- Use ELASTIC (or DataBallPy) to produce touch-accurate possession labels on any further clear-rights external match with events + ball tracking; forward-fill possession between receptions to make per-frame H/A targets.
- A trajectory-only MS-TCN touch/restart detector following the SoccerTrack v2 baseline (MIT code, 5 Hz, CPU-trainable size) as an extra feature stream; expect weak ±1 s timing without ball but usable ±5 s evidence and restart (Throw-in/Out) signal.
- Method ideas only (no usable data/code): SoccerNet-GAR's role-aware GNN with positional edges for restart classification; PC-SSAS's player-token state-space model with possession heuristics; Intellindust's team-alignment post-processing; uncertainty from heteroscedastic diffusion completion (Capellera et al.) as a per-frame confidence feature once/if code appears.
- Not applicable: event-conditioned reconstruction (Hughes et al. 2026, AgentImputer) because the competition match has no events; last-touch detection (Wang et al. 2025) because it needs skeletal and ball tracking.

### Gaps
- Whether SoccerTrack v2's ball-status flag marks stoppages with the same convention as DFL `BallStatus` (and how throw-in/kick-off restarts are timed) was not verified.
- Whether the ELASTIC `receive_frame_id` convention matches the organisers' possession-change convention in spellF1 labels is unknown.
