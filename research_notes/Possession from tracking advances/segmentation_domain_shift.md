# Segmentation and domain shift: 2024–2026 advances outside sports, and what fits our possession labeller

Scope: newer temporal action segmentation (TAS) models and decoders, test-time adaptation (TTA), label-shift correction, calibration and unsupervised domain adaptation (UDA) for sequence classifiers, and label-free model checks. The team survey of 2026-10-30 (`docs/research/models-literature.md`, `practitioner-advice.md`) already covers HSMM, semi-Markov CRF, change-point detection, MS-TCN, ASRF, ASFormer, DiffAct and Lea et al. 2017. They appear here only as baselines.

Reference point used for every judgement below: per-frame P(H/A/D) from a bidirectional GRU plus gradient-boosted trees (GBDT) on ~97 features, a min-length Viterbi, then learned post stages. Scored by spellF1 (H/A runs matched one-to-one by IoU, averaged over IoU 0.3–0.7). It is trained leave-one-match-out on 10 DFL/Metrica matches, the competition tracker is different, and there are no competition labels. It runs on an M2 laptop and the model is frozen for 17 Nov 2026, five days after these notes.

Licences were checked on 2026-11-12 through the GitHub API (`/repos/{owner}/{repo}` → `license.spdx_id`). LTContext's LICENSE file was read directly.

## Q1. Temporal action segmentation 2024–2026: what beats MS-TCN / ASRF / ASFormer / DiffAct, and how much of the gain is decoding versus backbone?

### Takeaway
Newer backbones (FACT, BaFormer, LTContext, ActFusion, DSA-Net) add only about 0–3 F1@50 points over DiffAct on 50Salads. On Breakfast the stronger backbone (FACT) beats ASFormer by about 11 F1@50, while the newest decoder adds only 1–5 points to a weak backbone and about nothing to a strong one. No Mamba/state-space model has reported Breakfast or 50Salads numbers that beat these. None of this transfers to our setting within days, and two of the strongest repos cannot be used commercially (LTContext is CC BY-NC 4.0, BaFormer has no licence).

### Cited Findings
**What the 2024–2026 papers are (fully supervised list).** The curated awesome-TAS list gives these 2024–2026 fully supervised entries:
- FACT (CVPR 2024), BaFormer (NeurIPS 2024), ActFusion (NeurIPS 2024), "Faster Diffusion Action Segmentation" (arXiv 2024), ASMamba in Video Mamba Suite (arXiv 2024), Dual-Stream Alignment / DSA-Net (arXiv 2025), Difformer (ICCVW 2025), End-to-End Action Segmentation Transformer (ICCVW 2025), "Towards Generalizing TAS to Unseen Views" (IJCV 2025), "Enhancing TAS with LLMs" (2025).
- Constraint-Aware Decoding / CAD (arXiv 2026), LogFA feature-space augmentation (ECCV 2026), and exocentric-to-egocentric adaptation (ECCV 2024 / IJCV 2026).
- The list has no separate decoding or domain-adaptation section.
- Source: [nus-cvml/awesome-temporal-action-segmentation](https://github.com/nus-cvml/awesome-temporal-action-segmentation)

**Benchmark numbers.**
- 50Salads, from an aggregator rather than the papers:

  | Model | F1@50 | Edit |
  |---|---|---|
  | BaFormer | 83.9 | 84.2 |
  | LTContext | 82.0 | 83.2 |
  | DiffAct | 83.7 | 85.0 |

  Source: [HyperAI 50Salads leaderboard](https://hyper.ai/en/sota/tasks/action-segmentation/benchmark/action-segmentation-on-50-salads-1). Third-party figures, not checked against the papers.
- DSA-Net (arXiv 2025, no code listed) is reported at 88.8 Edit / 87.1 F1@50 on 50Salads and 80.4 Edit / 68.1 F1@50 on Breakfast — [search summary of the awesome-TAS list](https://github.com/nus-cvml/awesome-temporal-action-segmentation). Unverified against the paper.
- FACT on Breakfast: F1@{10,25,50} = 81.3 / 76.4 / 66.2, Edit 79.7, Acc 75.2. ASFormer on Breakfast in the same table: 74.1 / 68.7 / 55.5, Edit 72.8 — [CAD paper, arXiv 2605.10149](https://arxiv.org/html/2605.10149). A FACT Breakfast row with Edit 79.7 / F1@50 66.2 also appears in a [PwC mirror](https://paperswithcode2.com/model/FACT%20(efficient%20hybrid%20of%20convolution%20and%20%20transformer%20model)).

**BaFormer (NeurIPS 2024).**
- How it works: instance queries plus one global query for class-agnostic boundaries, then a voting step at inference that classifies each boundary-delimited segment. This is a learned version of ASRF's "boundary + majority vote".
- Speed: it claims to use "only 6% of the running time" of DiffAct with "better or comparable accuracy", and 24% of the inference time of UVAST — [arXiv 2405.15995](https://arxiv.org/html/2405.15995v1); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/42770daf4a3384b712ea9c36e9279998-Abstract.html).
- Third-party figures: F1@50 83.5 on GTEA, average F1 72.4 on Breakfast — [search summary, sotaverified](https://sotaverified.org/papers/efficient-temporal-action-segmentation-via).

**Mamba / state-space models.**
- No dedicated Mamba paper reports frame-level TAS results on Breakfast or 50Salads. The nearest work is on action *detection* (MambaTAD, arXiv 2511.17929), action *localization*, and point-cloud segmentation (Mamba4D, 85.5 vs 84.1 accuracy on HOI4D) — [MambaTAD](https://arxiv.org/html/2511.17929v1); [Mamba4D](https://arxiv.org/pdf/2405.14338).
- ASMamba exists only inside Video Mamba Suite (MIT licence) — [video-mamba-suite](https://github.com/OpenGVLab/video-mamba-suite).

**Decoding versus backbone, from the CAD paper (arXiv, May 2026).**
- With hard transition, start/end and duration-bound constraints, the training-free decoder adds:
  - ASFormer / Breakfast: +4.7 / +4.4 / +2.5 F1@{10,25,50}, +5.1 Edit.
  - ASFormer / 50Salads: +1.4 / +2.3 / +1.4 F1, +2.9 Edit, but −2.1 accuracy.
  - FACT / Breakfast: +0.5 / +0.1 / **−1.4** F1, +1.7 Edit.
- Against the activity-grammar decoder KARI (NeurIPS 2023), as the CAD authors reproduced it, CAD is ahead by +1.0 to +1.5 F1 on ASFormer and behind by −0.8 F1@50 and −1.1 Edit on FACT. KARI took 148.9 s against 2.81 s for CAD on 20 Breakfast videos (A5000 GPU).
- Results are single numbers, with no seeds or variance.
- Source: [arXiv 2605.10149](https://arxiv.org/html/2605.10149)

**Licences (GitHub API, 2026-11-12).**

| Repo | Licence | Stars | Last push |
|---|---|---|---|
| FACT `ZijiaLewisLu/CVPR2024-FACT` | MIT | 116 | Jan 2026 |
| DiffAct `Finspire13/DiffAct` | MIT | — | — |
| KARI `gongda0e/KARI` | MIT | — | — |
| video-mamba-suite | MIT | — | — |
| BaFormer `peiyao-w/BaFormer` | **none** (no licence file, so all rights reserved by default) | — | — |
| LTContext `LTContext/LTContext` | **CC BY-NC 4.0** (non-commercial) | — | — |
| CAD `LUNAProject22/CAD` | **repo returned 404**, though the paper lists it | — | — |

Sources: [FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT), [DiffAct](https://github.com/Finspire13/DiffAct), [KARI](https://github.com/gongda0e/KARI), [video-mamba-suite](https://github.com/OpenGVLab/video-mamba-suite), [BaFormer](https://github.com/peiyao-w/BaFormer), [LTContext](https://github.com/LTContext/LTContext), [CAD](https://github.com/LUNAProject22/CAD).

**Generative refinement.**
- ActFusion (NeurIPS 2024) and "Faster Diffusion Action Segmentation" (2024) carry the DiffAct line forward — [awesome-TAS list](https://github.com/nus-cvml/awesome-temporal-action-segmentation).
- GPDAS (APSIPA 2025) claims state-of-the-art edit score and F1 on Breakfast. The snippet gave no numbers — [GPDAS PDF](https://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P377.pdf).

### Inferences
**Decoding versus backbone.**
- The backbone does most of the work on these benchmarks. ASFormer → FACT is about +11 F1@50 on Breakfast, while decoding adds about 2.5 to ASFormer and 0 to −1.4 to FACT. Decoders help a backbone that over-segments and stop helping once the backbone already produces clean segments.
- Our min-length Viterbi plus post stages already plays the "decoder on top of a weak per-frame model" role. The open question for us is the per-frame model under shift, not the decoder.

**Benchmark saturation.** 2024–2025 backbones are within about ±3 F1@50 of DiffAct (2023) on 50Salads. The benchmark is close to saturated, and none of the 2024–2026 papers tests a change of sensor or domain like ours (except the egocentric/view papers, which need synchronised paired data we do not have).

**Fit, usable within days.** None. Every backbone means retraining a sequence model on our ~97 features. They were designed for 2048-d I3D video features at about 15 fps, and our 199k frames would need downsampling. The model is frozen.

**Fit, next phase (weeks+).**
- **FACT** (MIT, actively maintained, efficient) is the most licence-safe modern backbone to try as a replacement for, or a third member alongside, the GRU. Evidence: strong on Breakfast, one paper, no independent replication found. It should run on an M2 (MPS/CPU) at 5 fps, but that is **unverified**.
- **BaFormer's boundary-plus-voting idea** is worth reimplementing from the paper. It is the learned form of ASRF, which the team already ranked highly. The code itself cannot be used commercially without the authors' permission (no licence).
- **Avoid LTContext code** (CC BY-NC 4.0) for anything shared with commercial partners.
- **Diffusion refinement (ActFusion, GPDAS)** is a poor fit: GPU-hungry, gains are small on saturated benchmarks, and there is no robustness evidence under shift.
- **Mamba** has no TAS evidence yet. Skip.

### Gaps
- Original-paper numbers for BaFormer, LTContext, DSA-Net and ActFusion were not checked; the 50Salads figures above are from aggregators.
- ASMamba's TAS numbers were not retrieved.
- FACT's 50Salads row was not found.
- No 2024–2026 TAS paper reports results under a change of sensor or feature extractor of the kind we face (different tracker), so robustness of these backbones under such shift is unknown.
- M2 training and inference cost for FACT or BaFormer on 199k-frame sequences is unverified.

## Q2. Decoding for segment-level metrics: explicit-duration, MBR / metric-aware, differentiable Viterbi, constrained decoding — 2024–2026 evidence over a tuned min-length Viterbi?

### Takeaway
The only new 2024–2026 decoding evidence found is CAD (arXiv 2026). Hard transition, start/end and duration *bounds* inside Viterbi beat transition-only Viterbi and beat soft constraints, but the gain over the backbone fades with a strong backbone. No 2024–2026 work was found that tests MBR, metric-aware decoding or differentiable Viterbi against a tuned min-length Viterbi on segmental F1. Our min-length Viterbi is already close to the CAD idea.

### Cited Findings
**CAD (2026) ablations.**
- On 50Salads (semi-supervised, C2F-TCN), "classical Viterbi" scored 45.5 / 40.2 / 30.1 F1@{10,25,50} and 39.2 Edit. That is *below* the no-constraint output (51.8 / 47.7 / 37.0, 45.9 Edit). CAD scored 55.8 / 51.4 / 38.5, 48.9 Edit.
- Soft constraints (Breakfast, ASFormer): 75.6 / 70.5 / 56.8, 74.6 Edit. Hard constraints: 78.8 / 73.1 / 58.0, 77.9 Edit.
- The constraints are counted from training labels: allowed transitions weighted by Count(A→B)/Count(A), allowed start and end sets, and per-class [d_min, d_max] *normalised to video length*. Complexity is stated as O(CT).
- The authors acknowledge the method depends on how well training data covers the constraints. Richer multi-parent constraints gave no gain.
- Source: [arXiv 2605.10149](https://arxiv.org/html/2605.10149)

**Grammar decoding (KARI, NeurIPS 2023).** CAD calls grammar methods costly ("complex recursive grammars and costly parsing", worst case O(T³)). On FACT, KARI improved Edit (79.7 → 82.5) but not F1@50 (66.2 → 65.6) — [arXiv 2605.10149](https://arxiv.org/html/2605.10149). Code is MIT — [gongda0e/KARI](https://github.com/gongda0e/KARI).

**Explicit duration.**
- BiasFormer (MDPI Sensors 2026) adds a segmental CRF with explicit duration modelling and Viterbi-style decoding on a transformer. The snippet gave no F1 numbers — [MDPI](https://www.mdpi.com/1424-8220/26/15/4854).
- A targeted search found no 2024–2025 paper with a head-to-head F1 comparison of semi-Markov versus Viterbi decoding for TAS — [search results incl. arXiv 2209.09149](https://arxiv.org/html/2209.09149v1).

**Evidence from sound event detection (SED), a domain with a similar event-based F1.**
- Median-filter post-processing can "erase" high-confidence detections and move boundaries — [arXiv 1904.03841](https://arxiv.org/pdf/1904.03841).
- Per-class median-filter windows tuned on the development set raised event-based F1 (to 48.6% in one system) — [arXiv 2007.00908](https://arxiv.org/pdf/2007.00908).
- Errors near onsets and offsets are "beyond the capacity of the median filter" — [arXiv 2403.13254](https://arxiv.org/html/2403.13254v1).
- Ebbers and Haeb-Umbach argue that differing post-processing biases system comparisons, and propose post-processing-independent evaluation (PSDS; `sed_scores_eval`) — [arXiv 2306.15440](https://arxiv.org/html/2306.15440v1); [DCASE 2023 paper](https://dcase.community/documents/workshop2023/proceedings/DCASE2023Workshop_Ebbers_62.pdf).

### Inferences
**What CAD confirms.** Duration constraints are the part of a decoder that pays off; transition priors alone can hurt. That supports our min-length Viterbi and adds little new.

**What CAD adds that we lack.**
- (a) A **max-duration** bound. This could matter for dead-ball (D) runs, but an H/A possession has no natural upper bound.
- (b) **Durations normalised to sequence length.** This does not help us: our sequences are all about 90-minute matches.
- (c) **Allowed start and end states**, for example D at each kick-off and half-time. The team already restored the second-half kick-off in v12 (git log), so this is mostly covered.

**Fit, usable within days (only if the decoder is not frozen with the model).** A D-run max-duration cap tuned leave-one-match-out on the bench. Evidence is weak: one 2026 preprint, gains of 1–5 F1 on weak backbones only, and no public code (the repo returned 404). Expect a small effect. Accept it only if it holds on every bench match and on the SkillCorner shift test.

**Risk under shift.** CAD's hard bounds come from training statistics. If the competition tracker produces systematically shorter or longer D runs, for example because stoppages are detected differently, hard bounds will force errors. The SED evidence that per-class windows are tuned on a dev set points the same way: post-processing tuned on the bench domain may not transfer. Prefer settings where bench spellF1 is *flat* across a range of min-length and switch prices over the single best point (a post-processing-independent view in the PSDS spirit).

**MBR / metric-aware decoding.**
- The idea: choose the label sequence that maximises *expected* spellF1 under the model's posterior, for example by sampling sequences from forward-filtering backward-sampling and picking the candidate with the highest mean spellF1 against the samples.
- This is a natural next-phase experiment, but the search found no 2024–2026 evidence for it in TAS.
- Weeks of work and only worthwhile if calibration under shift is good (see Q3). Under shift, miscalibrated posteriors make "expected metric" decoding unreliable.

### Gaps
- No 2024–2026 TAS or sequence-labelling paper was found that evaluates MBR or metric-aware decoding, or differentiable Viterbi (structured-loss training), against a tuned min-length Viterbi on segmental F1.
- BiasFormer's numbers were not retrieved.
- CAD has no released code and no variance estimates.

## Q3. Domain shift for sequence models: TTA, label-shift correction, calibration under shift, UDA for time series, and simple robustness tricks — what is reliable with one unlabelled ~2 h, ~199k-frame test sequence?

### Takeaway
With one temporally correlated, unlabelled test sequence and a GRU + GBDT model, entropy-style TTA (TENT and kin) is **not reliable**: benchmarks show it fails or hurts under temporal correlation and non-stationarity, and it does not apply to GBDT at all. EM-based prior (label-shift) correction with calibration is cheap and well studied, but it assumes p(x|y) is unchanged, which a different tracker breaks. Use it as a diagnostic, not an automatic fix. UDA for time series gives modest average gains (+0.05–0.08 accuracy) that are inconsistent across methods. The most reliable next-phase lever is training-time augmentation that simulates the target sensor's artefacts (sensor dropout, missing ball).

### Cited Findings
**TTA under temporal correlation.**
- TTAB (ICML 2023) benchmarked ten TTA methods. Under non-stationary or time-correlated shifts, TENT exceeded 76% error in one setting, and TENT and BN-adapt "still fail" even with good model selection — [Zhao et al., PMLR v202](https://proceedings.mlr.press/v202/zhao23d/zhao23d.pdf).
- NOTE (NeurIPS 2022) shows that batch-norm statistics "cancel out" meaningful instance-wise variation when test data are temporally correlated. It evaluates on real streams including HARTH, a wearable human-activity sensor dataset — [arXiv 2208.05117](https://arxiv.org/pdf/2208.05117).
- TempT (2023, video affect) found standard TTA "consistently hurt" the baseline because test inputs were highly correlated. A temporal-consistency objective instead reduced output flicker — [arXiv 2303.10536](https://arxiv.org/pdf/2303.10536).
- A 2025 survey of continual TTA notes that TENT run continuously without reset accumulates errors progressively — [preprints.org survey](https://www.preprints.org/frontend/manuscript/e4460dc2244544474f8edd7a1ebb3568/download_pub).
- 2026 time-series TTA (arXiv 2602.00073) adapts only the normalisation affine parameters of a frozen backbone, with entropy plus a temporal-consistency term, for forecasting and direction classification — [arXiv 2602.00073](https://arxiv.org/html/2602.00073). Benchmark details were not verified.
- A widely used collection of TTA implementations is [mariodoebler/test-time-adaptation](https://github.com/mariodoebler/test-time-adaptation) (MIT, 294 stars, last push May 2025; GitHub API check). Which methods it includes was not checked in this pass.

**Label shift and prior correction.**
- Alexandari et al. (ICML 2020) found that maximum-likelihood / EM prior adaptation (Saerens et al.'s EM) with *bias-corrected* temperature scaling (BCTS) is "hard to beat" for label-shift adaptation, and that calibration quality drives EM's performance — [arXiv 1901.06852](https://arxiv.org/pdf/1901.06852); [PMLR](http://proceedings.mlr.press/v119/alexandari20a/alexandari20a.pdf). Its reference code repo `kundajelab/abstention` has **no licence** (GitHub API).
- LaSCal (NeurIPS 2024) estimates calibration under label shift *without target labels* and recalibrates accordingly. It reports better calibration than the EM-BCTS pipeline in most settings — [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/783c5986e1d6112cb4688d9b2105609a-Paper-Conference.pdf). Details are from the search summary only; the PDF could not be parsed.
- Newer estimators include a Bayesian MAP estimate of the target label distribution (WACV 2024) — [CVF](https://openaccess.thecvf.com/content/WACV2024/papers/Ye_Label_Shift_Estimation_for_Class-Imbalance_Problem_A_Bayesian_Approach_WACV_2024_paper.pdf) — and an incremental per-sample prior update said to need a weaker calibration assumption (SIAM SDM 2025) — [doi](https://doi.org/10.1137/1.9781611978520.12).
- The search found no paper handling label shift and covariate shift *together* in this line of work.

**UDA for time series.**
- The DMKD 2025 benchmark (Ismail Fawaz et al.) covers 9 methods on 12 datasets, including HAR, HHAR, sleep, sports and WISDM.
  - Source-only InceptionTime averages 0.59–0.60 accuracy under label-free tuning.
  - The best UDA methods add about +0.05 to +0.08 (Raincoat-style, CoDATS / domain-adversarial).
  - Several methods fall *below* source-only: VRADA 0.50–0.52, CoTMix 0.57, InceptionMix 0.56–0.57.
  - "The backbone choice seems to be less important than the UDA strategy."
  - Code: [EricssonResearch/UDA-4-TSC](https://github.com/EricssonResearch/UDA-4-TSC).
  - Source: [arXiv 2312.09857v3](https://arxiv.org/html/2312.09857v3)
- Source-free time-series DA work in 2024–2025 (MAPU, parameter-subspace SFDA, compositional transferability) evaluates on AdaTime-style benchmarks — [arXiv 2307.07542](https://arxiv.org/pdf/2307.07542); [arXiv 2410.02147](https://arxiv.org/pdf/2410.02147); [arXiv 2504.14994](https://arxiv.org/pdf/2504.14994). AdaTime itself is MIT — [emadeldeen24/AdaTime](https://github.com/emadeldeen24/AdaTime) (GitHub API).
- A source-free universal DA benchmark for time series reports that foundation models "do not systematically outperform" classical backbones, and that the threshold for rejecting unknown samples is highly sensitive — [arXiv 2609.39810](https://arxiv.org/abs/2609.39810).

**Robustness through augmentation that simulates sensor artefacts.**
- Input Sensor Dropout masks all features of randomly chosen sensors during training and improves robustness to missing sensors on multi-sensor Earth-observation time series. Ensembles were the most robust — [arXiv 2407.15512](https://arxiv.org/html/2407.15512v2).
- A NeurIPS 2025 paper argues that real sensor faults are *structured* (bias, drift, hard faults), which generic augmentation may not capture, and proposes fault-robust pretraining — [NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/e43ac0bc7daa066aa6861e2c04d5dfdc-Paper-Conference.pdf).

**Domain shift inside TAS.** The only TAS-specific work found concerns view changes (IJCV 2025) and exocentric-to-egocentric transfer using *synchronised paired* unlabelled video (ECCV 2024 / IJCV 2026) — [awesome-TAS list](https://github.com/nus-cvml/awesome-temporal-action-segmentation).

### Inferences
**Why the single-sequence setting rules out TENT-style TTA.**
- One match is one long, strongly autocorrelated stream with long H/A/D runs. That is exactly the setting where TTAB, NOTE and TempT report TTA failures.
- Our GBDT has no gradients or normalisation layers to adapt. Our GRU may have no BatchNorm (check), so TENT's normal-statistics mechanism may not even exist.
- Without labels there is no safe stopping rule.
- Verdict: **do not use within days.**
- Next phase: an LN-affine or normalisation-only TTA with a temporal-consistency term (TempT, arXiv 2602.00073), tested on the bench with leave-one-provider-out. Evidence quality is low to medium.

**EM prior correction: run as a diagnostic now, consider applying later (days).**
- How: apply BCTS-calibrated EM (Saerens/Alexandari) to the per-frame posteriors of the unlabelled competition match. It is about 30 lines of numpy (no licensed code needed).
- What it is mostly about: H/A are roughly symmetric, so the meaningful prior is the dead-ball share D.
- Autocorrelation: the point estimate of the priors does not need independent frames. Only its variance does.
- The catch: a different tracker changes p(x|y), for example ball-missing patterns, which breaks the label-shift assumption. EM may then "explain" covariate shift as a prior change and move D in the wrong direction.
- Test it on our labelled external data first:
  - For each held-out bench match and for the SkillCorner broadcast test, does the EM-estimated D share track the true D share better than the training prior does?
  - Does prior-corrected decoding raise spellF1?
- Apply it automatically only if it helps on SkillCorner, the one cross-tracker case we have, and preferably with shrinkage toward the training prior.
- Either way, the EM-estimated D share is a cheap shift alarm.

**UDA (next phase only).**
- Evidence: average gains of about 5–8 accuracy points, uneven across methods and tuned by label-free selection, which itself lags the oracle by about 10 points (Q4).
- It requires a neural feature extractor shared between domains, which means retraining the GRU and leaving GBDT out.
- Domain-adversarial training (CoDATS/DANN) with the competition match as unlabelled target is the best-evidenced option, and AdaTime / UDA-4-TSC give code.
- Fit: medium. We have exactly one unlabelled target sequence, and the competition rules on using its tracking for training need checking. "Label-free checks on its unlabelled tracking are allowed if automatic" may not cover adversarial training on it.

**Augmentation (next phase, the best fit).**
- Our largest measured drop (0.70 → 0.41 on SkillCorner) is consistent with sensor artefacts: ball missing about 41%, different smoothing and identity swaps.
- Input-sensor-dropout-style training fits this: mask the ball channel in long blocks at the competition rate, and mask or jitter player features.
- Include *structured* faults (drift, frozen ball, ID swaps), per the NeurIPS 2025 point.
- Evidence comes from another domain (Earth observation), but the mechanism matches. It is cheap: retrain plus the bench.
- It cannot change the frozen model before 17 Nov.

**Calibration under shift.**
- If any prior or MBR step is used, recalibrate with temperature scaling on *degraded* bench data (simulated tracker artefacts) rather than clean bench data. Calibration fitted on clean data will be overconfident on the competition tracker.
- LaSCal is the 2024 method to read for label-shift-aware recalibration. The code link and licence were not found.

### Gaps
- No 2024–2026 benchmark specifically for *test-time adaptation* of time-series *classifiers* or sequence labellers was found. The evidence is image/video TTA plus HAR streams in NOTE.
- No study found of label-shift correction under simultaneous covariate shift caused by sensor change.
- LaSCal's method details, numbers and code licence could not be extracted (PDF parse failed).
- The licence of UDA-4-TSC was not checked.
- No evidence was found on GBDT-specific adaptation (for example, importance-weighted refitting with domain-classifier weights) for time series in 2024–2026.

## Q4. Selecting and validating a model without test labels: label-free proxies, agreement-based selection, shift detection

### Takeaway
Label-free selection is clearly worse than oracle selection. On the 2025 time-series UDA benchmark it costs about 9–10 accuracy points against selection with target labels. Agreement-on-the-line (AGL) and confidence methods (ATC, MaNo) can rank models and estimate accuracy, but they hold mostly for families of neural networks and estimate per-sample accuracy, not segmental F1. For us within days, they work best as *alarms* and *tie-breakers* (agreement spellF1 between variants, predicted D share, run-length statistics), not as an objective to tune against.

### Cited Findings
**Cost of label-free selection.** In the DMKD 2025 benchmark, label-free selection by Source Risk and by importance-weighted cross-validation (IWCV) performed alike (Wilcoxon p ≈ 1). Both were "significantly lower" than Target Risk, the oracle with target labels: for example InceptionRain 0.67 / 0.68 vs 0.77, and source-only 0.59–0.60 vs 0.70. IWCV did better under larger shifts, Source Risk under smaller ones — [arXiv 2312.09857v3](https://arxiv.org/html/2312.09857v3).

**Agreement and confidence methods.**
- Agreement-on-the-line (Baek et al., NeurIPS 2022): across a family of models, in-distribution versus OOD *agreement* lies on a line exactly when ID versus OOD *accuracy* does. The line's slope and bias can be estimated from unlabelled OOD data (ALine-S / ALine-D). ALine-D beat ATC, among others. The phenomenon appears to hold only for neural networks — [arXiv 2206.13089](https://arxiv.org/pdf/2206.13089); [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.13089).
- ATC (average thresholded confidence) learns a confidence threshold on ID validation data and predicts OOD accuracy as the share of unlabelled OOD samples above it — described in [arXiv 2206.13089](https://arxiv.org/pdf/2206.13089).
- Kim et al. (NeurIPS 2024): TTA "drastically strengthen[s]" both accuracy-on-the-line and agreement-on-the-line, even on shifts where they were weak. Together with AGL this allows TTA hyperparameter and model selection without OOD labels — [arXiv 2310.04941](https://arxiv.org/abs/2310.04941).

**Other unlabelled accuracy estimators.**
- MaNo (2024): training-free; uses the matrix norm of predictions on unlabelled test data. Code is MIT — [arXiv 2405.18979](https://arxiv.org/html/2405.18979v1); [Renchunzi-Xie/MaNo](https://github.com/Renchunzi-Xie/MaNo) (GitHub API).
- Gradient-norm estimation (2024) — [arXiv 2401.08909](https://arxiv.org/abs/2401.08909).
- IW-GAE: importance-weighted group accuracy for calibration and UDA model selection, with a reported 14% average gain in model selection — [arXiv 2310.10611](https://arxiv.org/html/2310.10611).
- Confidence-based estimators for model monitoring under covariate shift — [arXiv 2407.08649](https://arxiv.org/html/2407.08649v1).
- PAPE: density-ratio estimation plus weighted calibration, said to estimate F1 and AUROC on shifted data. Seen only via a secondary write-up — [Lacuna](https://lacuna.tiptreesystems.com/work/estimating-model-performance-under-covariate-shift-without-labels/wrk_570704a1142062f04352f83ac736a10d).
- A 2026 medical-imaging paper selects foundation models without labels using pseudo-label discrepancy. The arXiv ID looks unusual and was not verified — [arXiv 2608.25810](https://arxiv.org/html/2608.25810v1).

**Post-processing and comparison.** In sound event detection, post-processing choices bias system comparisons. Ebbers and Haeb-Umbach propose metrics independent of post-processing — [arXiv 2306.15440](https://arxiv.org/html/2306.15440v1).

### Inferences
**Usable within days: automatic, label-free checks on the competition tracking.** Each check is compared with the distribution of the same statistic over the 10 bench matches and the SkillCorner test.

1. **Agreement spellF1 between model variants.** Compute spellF1 between the outputs of GRU-only, GBDT-only, the blend, and a min-length sweep. This is an AGL-style signal at the segment level.
   - If variants agree on the bench (and on SkillCorner when they did well) but disagree on the competition match, expect a drop and prefer the most conservative variant. Conservative here means the one closest to the variant consensus, a "medoid" choice.
   - Evidence that agreement tracks performance: AGL (NeurIPS 2022), strengthened in 2024. The *linear* fit needs a neural model family and per-sample accuracy, so treat this as ordinal, not as a calibrated estimate.
2. **ATC-style confidence share.**
   - Set a threshold on max P(H/A/D) so that the share of bench frames above it matches bench frame accuracy, then read the share on the competition match.
   - Also compare spell-level statistics: number of H/A spells per minute, median spell length, D share and run-length histograms. Compare these against the bench and against what 25 fps professional football implies.
   - A big gap in spells per minute means over- or under-segmentation, and spellF1 punishes both.
3. **EM-estimated D share** (Q3) as a prior-shift alarm.
4. **Shift magnitude.**
   - Train a domain classifier (bench frames vs competition frames) on the ~97 features. Its AUC and its top features tell us *which* inputs shifted; ball-related features are expected.
   - Do the same for SkillCorner vs bench. If the competition shift looks more like SkillCorner than like the bench, expect a drop toward SkillCorner levels.
   - The same weights would allow IWCV, which the 2025 benchmark found better under large shifts. IWCV still lagged the oracle, so use it to *rank* only a few candidates.

**Decision rule for the frozen deadline.** Use these checks to *veto* a candidate that looks anomalous on the competition match, not to pick the best one. The 2025 benchmark's 9–10 point gap between label-free and oracle selection says that fine-grained optimisation against a label-free proxy will mostly fit noise.

**Next phase.**
- Build a calibrated label-free *spellF1 estimator* by regressing true spellF1 on (agreement, confidence share, spell-rate gap, domain-classifier AUC) across many leave-one-out and synthetically degraded bench matches.
- Validate it by predicting SkillCorner's 0.41 without its labels. It then becomes a usable model-selection tool, and the same machinery fits TTA hyperparameters (Kim et al. 2024).

### Gaps
- No 2024–2026 work was found on label-free estimation of *segmental* or event-level metrics (F1@IoU, edit). All estimators found target per-sample accuracy.
- No evidence that AGL or ATC hold for GBDT or for GRU+GBDT blends; AGL is reported only for neural networks.
- The original PAPE paper and the 2026 medical-imaging selection paper were not verified.
- No 2024–2026 shift-detection benchmark for multivariate tracking or sensor time series was retrieved in this pass. The domain-classifier approach above is standard practice, but no source for it was collected here.
