# Limitations, and where we have room to move

Written 2026-10-30. Sources: the organisers' RULES.md, FAQ.md, EVALUATION.md, OVERVIEW.md,
SUBMISSION.md, `evaluation/possession_metrics.py`, and the organisers' answers on the challenge's
GitHub issues (cited as #N, linked at the bottom). Where an organiser answer and a file disagree,
the newer one is marked and the point is listed under "grey zones".

**In short**

- One wall matters most: nothing human-made about BAR-ATH may touch the model. That rules out our
  labels, published stats about the match, and choosing settings on its 5 public minutes.
- With no leaderboard, settings and versions are chosen on **other matches** where the clue
  exists there (tracking-based clues), and by reasoning plus label-free checks where it doesn't
  (audio, video, Smart Tagging). The organisers explicitly allow choosing settings on other
  matches, including ones we label ourselves.
- On this match we may still learn **without labels**, use the **future**, rebuild the tracking with
  our own code, and use **every provided asset**: audio, broadcast, Smart Tagging.
- Most of the room left is in the metric: decode for the score, and use `X` only where it can't hurt.

---

## 0. The organisers' clarification of 2026-11-03 (takes precedence over everything below)

Emailed to all participants. It overrides the documentation, earlier emails **and answers to
individual questions**. Where a limit or grey zone below says otherwise, this wins.

| not allowed | allowed |
|---|---|
| checking our output against the video | automatic checks using only the provided data |
| hand labels, human-produced data or any data created manually on the provided game | inspecting the provided data and the video to understand what counts as possession |
| uploading the provided data to AI services | the video, tracking, Smart Tagging or other generated data **as input**, if nothing was trained on the ground truth or the video |
| training, fitting or calibrating on the ground-truth possession data or on the provided video | files of intermediate steps |
| | any dataset we have the rights to use (check its licence) |

**Training and tuning:** only on external data. Every parameter that needs tuning must be tuned on
external data, never on the ground truth or the provided video.

### Audit of our work against it (2026-11-03)

**Fine as it is**
- Every learned model and every bench-chosen setting is fitted on other matches: the decoder (v3),
  the dead-ball model (v4), the restart model (v6), PathCRF's weight (v7), the snap window (v9).
- The public 5 minutes were never used to choose anything. Scoring on them is an automatic check
  on provided data.
- PathCRF: code MPL-2.0, weights trained by its authors on DFL data (CC BY 4.0).
- The keeper repair (v8): constants from the Laws of the Game and reasoning. The idea came from
  watching how the tracking behaves (understanding the data), and the counts on our match are
  automatic checks.
- Viewer notes: the notes file is empty and gitignored. It must stay empty: notes on our match
  would be data created manually on the provided game.

**Needs action** (items B–D handled by [v10](models/v10-clean-base.md), 2026-11-03; A and E wait for the organisers' answer)

| # | what | where it happened | what it touches | options |
|---|---|---|---|---|
| A | **Provided data reached an AI service.** Claude saw screenshots of the viewer with the video, printed tracking values, our labels and statistics computed from the match | this whole project, under our 2026-10-30 reading ([decisions.md](decisions.md)), which this clarification overrides | how we work, not the models | from now on Claude doesn't see the match data (no screenshots, no runs whose output it reads); ask the organisers whether summary statistics are allowed, and disclose what happened |
| B | **Our output checked against the video** | v2 (its 10 changed moments watched); v5 (frames 23697 and 54740 watched: the reason for v5); v9 (Jose spotted late starts by watching our labels on the video); the viewer's model lanes over the video and its "watch changes" mode | v5's whistle weight; v9's origin; the viewer | remove model outputs from the BAR-ATH viewer (the bench viewer stays: it's external data); revert v5's weight or keep it on the automatic check alone (counted whistles not followed by dead: 3 of 11); v9's rule rests on the bench's truth (698/698) and is tuned on external data only. **2026-11-04:** Jose chose to bring the model lanes back behind `viewer.server --models`, to see the versions run, never to choose between them ([decisions.md](decisions.md)) |
| C | **Settings calibrated on the provided video or ground truth** | the whistle detector (v2): its band ("tones found at 2.6–3.9 kHz"), prominence and level thresholds (set from our audio's distribution) and slowdown test (from our 86 events); the set-piece window (15 s) and weight (0.3), from 3 tag offsets and "2 of 5 restarts had no tag" in the public truth | v4–v9 use both clues | set them again from external sources or reasoning, or drop the clues; the bench can't measure either (no audio, no tags) |
| D | **Dataset rights: Metrica's sample data has no licence file** (grey zone G8) | v4 and v6 trained on all 10 bench matches; v7's weight chosen on 4 (3 of them Metrica's); v9's window chosen on all 10 | v4, v6, v7, v9 | ask Metrica (an organiser) for written permission, or redo the training and choices on IDSSE only (CC BY 4.0) |
| E | **Grey: statistics of the provided tracking used to shape external data** | the bench's damage (ball missing 41 %, fragments over 46 slots, missing while stopped) was measured on our tracking, without labels or video; v1's "stopped/active" thresholds too (no longer used since v4) | the bench | ask; the parameters themselves are still tuned on external data |

---

## 1. Hard limits

| # | limit | source | what it rules out for us |
|---|---|---|---|
| 1 | **No hand labels on BAR-ATH, in any amount, for any purpose** | RULES.md §1, #8, #10 | Labelling frames or segments to train, to validate, to estimate our accuracy, or **to choose parameter values**. "The labels never being an input to the pipeline doesn't change this" (#10). Hand-correcting the tracking (ball position, identity switches) counts too. |
| 2 | **No rules or constants tied to specific frames** | RULES.md §1, #14 | "Between 03:06 and 03:26 the ball is dead" in code, even if no label file exists. General rules learned from watching are fine (see §3). |
| 3 | **The 5 public minutes are only for checking the pipeline** | RULES.md §2, EVALUATION.md | Training and tuning on them (RULES.md: "not training data and not a tuning set"). Keeping whichever version scores better there is tuning too; EVALUATION.md says to use the window "never to choose between two versions of it". Using it to catch bugs (misaligned frames, a broken clue) is exactly its purpose. |
| 4 | **No human-produced data about BAR-ATH, from any source** | RULES.md §2 | Event data (Opta, StatsBomb, Wyscout, LaLiga), corrected tracking, possession data, annotations. By the same logic: match reports, goal minutes, published possession %, foul counts. |
| 5 | **No model trained on this match's video using the provided tracking as labels** | #2 | A ball detector trained on BAR-ATH frames with ATD ball positions as targets. Detectors must be trained on other data, with the source documented. (See grey zone G1: RULES.md was loosened a day later.) |
| 6 | **Everything must be accessible and explained** | RULES.md "What you may use" | Private in-house models or datasets. Any labels we make on other matches must be shared with the organisers, with the video if we have the rights (#10). |
| 7 | **Byte-identical regeneration from a public repo** | RULES.md §3, #11 | Any step that may come out differently on their machine. Training does not need to be reproducible; inference does. Advice: commit weights, run inference on CPU, pin versions, fix seeds (#11). |
| 8 | **The data is not ours to publish** | RULES.md "The data" | Committing `data/`, clipped video, or "derived data that would reconstruct it", such as frame-level tracking exports, to the public repo. The viewer cache stays gitignored. |
| 9 | **Match data never goes to a third-party cloud LLM** | OVERVIEW.md, FAQ.md (T&C §8) | Sending videos, datasets "or any extracts thereof" to a hosted LLM. Our working reading is in [decisions.md](decisions.md) (2026-10-30): inspecting is fine, and no LLM ever labels frames. Another team quotes T&C §18 as also forbidding sharing with third parties without written permission (#22, not yet answered), so cloud compute is out too unless the organisers say otherwise. |
| 10 | **External data licences** | #16 | CC BY data (IDSSE, for example) must be cited in the README and the technical document. Other sets have their own terms (see [research/](research/)). |
| 11 | **Team of 3 to 5**, all registered under the same team name **by Nov 10** | RULES.md, OVERVIEW.md | A team of fewer than 3 is not eligible for the Innovation Challenge or the final pitch. |
| 12 | **Submission format** | SUBMISSION.md | Exactly 199,319 rows; `<team_name>.csv` + `<team_name>.pdf`/`.md`; one Google Form response per team (editable until the deadline); a Google account. **Nov 17, 23:59 CET.** |

## 2. Practical limits

- **No feedback before results.** No leaderboard, and the public window may not rank versions
  (rule 3). It couldn't anyway: it has 17 possessions, one possession moves its spellF1 by ~6
  points, and v1 and v2 both score 0.43 there. So the evidence for a change has to come from
  elsewhere, and where depends on the clue:
  - **Clues built from tracking** (nearest player, players slowing, team shape) can be measured on
    other matches' tracking, which comes with per-frame possession and dead-ball flags.
  - **Clues that only our match has** (the audio whistle, Smart Tagging, the video) can't: the open
    datasets have no audio or video. For those the evidence is football reasoning plus label-free
    checks on our own match (does the whistle line up with players slowing down?). The other
    option is a match **with** video that we hand-label ourselves, which is allowed (#8, #10) but
    costs hours of labelling.
- **The public window is unrepresentative** anyway: the ball is visible on 88.2 % of its frames
  against 59.3 % over the match ([data.md](data.md)).
- **Noise band:** spellF1 differences under ~0.04 on the full match are noise (EVALUATION.md). Two
  versions that close are a tie.
- **Hardware:** Apple M2, 8 GB RAM, no NVIDIA GPU. Large video models are slow or don't fit, and
  cloud GPUs are blocked by rule 9 for now.
- **Byte-identity across machines.** The organisers may re-run on Linux/x86. Floating-point results
  can differ across CPUs and library builds, and a tiny difference near a threshold flips a label.
  Mitigations: pinned versions (done), CPU only, keep decisions away from razor-thin margins, and
  commit the outputs of fragile stages (allowed, #14). Our pinned Python 3.14.7 is very new and
  worth checking against what they'll have.
- **Convention not fully pinned down.** CHALLENGE.md says a won header without control doesn't
  change possession, yet the public truth shows a 1.32 s Barça header as a possession (#17, not yet
  answered). The truth follows the tracking's ball owner, so short touches may count.
- **Time:** 18 days to the deadline, 11 to team registration.

## 3. Allowed, but not obvious

Each is backed by a rule or an organiser answer.

| what | why it helps | source |
|---|---|---|
| **Use the future.** Label 08:47 using 09:01. | Two-pass / forward-backward decoding; a hand-over is clearest a second after it happens. | #15 |
| **Learn on this match without labels.** Unsupervised or self-supervised methods, EM-fitting an HMM's transition and emission parameters (Baum-Welch) to the unlabelled match, setting thresholds from how a quantity is spread. | Settings fitted to this match's own tracking quality, with no labels involved. | RULES.md "What remains fine"; [decisions.md](decisions.md) |
| **Rebuild or improve the tracking with our own code.** | The ball is missing on 41 % of frames; filling it is the biggest single lever. Classical CV (colour/motion blobs, optical flow, trajectory physics) needs **no training data at all**, so it avoids limit 5 entirely. | RULES.md "starting point, not a constraint"; FAQ |
| **Train anything on other matches**, including detectors, team-shape models and ball-trajectory imputation. | Pre-trained weights trained on public data are fine to commit. | #2, #16 |
| **Hand-label other matches, in any amount, and choose every setting there**, then apply the settings unchanged to BAR-ATH. | This is the organisers' own suggested route for tuning (#10). It gives us a real validation set. | #8, #10 |
| **Use public tracking with possession labels from other matches** as a validation set with our own copy of the scorer. | IDSSE (Bundesliga, CC BY 4.0) was approved for exactly this kind of use. | #16 |
| **Every provided asset is fair game**: the audio (whistles, kicks), the broadcast video with its own tracking and homography, Smart Tagging. | The broadcast zooms in, but it sees the ball on only 12–13 % of the tactical gaps ([data.md](data.md)), so its value is elsewhere: replays and close-ups are shown mostly while the ball is dead. Set-piece tags mark restarts. | FAQ ("Use it — that is why it is there") |
| **Turn what we see while watching into general rules**, with thresholds set elsewhere. | This is how the whistle clue (v2) came about. | #14 |
| **Commit a slow or non-deterministic stage's small output** (event times and scores), with the script, model version and settings that regenerate it. | It lets us use a GPU or a fragile model once and keep the final CSV byte-identical. | #14 |
| **Runtime is unlimited, and a CUDA GPU is acceptable** for reproduction. | No need to make it fast. | #2, #11 |
| **LLMs anywhere in the pipeline**, including local or open models. | For example, writing code, or reasoning over other matches' data. The match data itself may not go to a cloud LLM (limit 9). | OVERVIEW.md |

## 4. Levers in the metric

These are fair: they're just how the published scorer works (`evaluation/possession_metrics.py`).

- **Every predicted possession is one object.** A spurious 1-second spell costs one false positive,
  exactly like missing a 60-second possession. Insert a short possession only when it's more likely
  real than not. The decoder should aim for the best **expected spellF1**, not the most frames right
  (in jargon, minimum-Bayes-risk decoding).
- **Matching is one-to-one, same team, IoU 0.3–0.7.**
  - Splitting one true possession in two gives at most one match plus one false positive.
  - Missing a stoppage merges two possessions into one prediction, which matches at most one of them.
  - So `D` matters for the ranking through **where possessions end**, not through its own frames.
- **What `X` is worth:**
  - For spellF1 it is replaced by our previous label, so it scores exactly as if we had repeated it.
  - For macroF1 (the tie-break) it counts as a miss but **never as a false positive**, so it beats
    a wrong guess.
  - Rule: write `X` only where we would carry the previous label anyway and that label is probably
    wrong; never across a likely hand-over. Low value, tie-break only.
- **Frames outside the two halves are never graded.** Anything goes there.

## 5. Grey zones: ask the organisers before building on these

| | question | why it's unclear |
|---|---|---|
| G1 | May we **self-train on this match with algorithmic pseudo-labels** (our own pipeline's confident frames, or the provided tracking's ball positions) when no person decides any label? | RULES.md (updated Oct 23) allows "pseudo-labels produced by an algorithm, provided no person decided any of them". #2 (Oct 22) ruled out a detector trained on this match with the provided tracking as labels. |
| G2 | May we use the **broadcast commentary** (speech-to-text)? | It's part of a provided asset, but it's humans describing this match. It may count as "human-produced data about the game". |
| G3 | May we type in **frame facts the organisers published**, such as the 2nd-half kick-off at 113,742? | It's organiser metadata, not something we watched for. Detecting it is safer: players jump from standing still at kick-off, and v2 already sees this. |
| G4 | **Cloud GPU for inference**, uploading only frames? | Asked in #22, not yet answered. We avoid it. |
| G5 | Does a **won header without control** count as a possession? | Asked in #17, not yet answered. It's not a rule, but it changes the target. Metrica's own event definitions count a won challenge as a recovery ([research/practitioner-advice.md](research/practitioner-advice.md)), which points to yes. |
| G6 | May we use **pretrained weights whose training data is "non-commercial research only"** (ISSIA-CNR: WASB, FootAndBall)? | The rules allow anything accessible and documented; the dataset's own licence may not allow a competition with commercial partners. |
| G7 | May we **commit a per-frame ball track** made by our own detector? | #14 allows small stage outputs (event times and scores); a per-frame track looks like a "frame-level tracking export" (rule 8). The safe option is to regenerate it at run time. |
| G8 | Are **PFF World Cup 2022** data and **Metrica's sample data** acceptable? (Metrica's samples are in use since 2026-10-30 for checking and choosing settings only: Jose's decision, [decisions.md](decisions.md)) | PFF sits behind a sign-up form with terms we couldn't find; Metrica's samples have no licence file. IDSSE is already approved (#16). |

The rules ask for these questions as public issues ("ask before you build on it"). G1, G2 and G8
change what we would build; G6 and G7 matter only if we do video ball detection.

## 6. Traps: things that look harmless and aren't

- Keeping v1 or v2 because it scores better on the public 5 minutes (rule 3). Use the window to
  catch bugs; pick by other matches or by reasoning.
- Looking up the match online (score, goal times, possession %) and letting it shape a setting (rule 4).
- "Just a few labels to see how we're doing" on BAR-ATH (rule 1, #10).
- Viewer notes turning into a label row or a frame list the code reads (rule 2, [decisions.md](decisions.md)).
- Committing the viewer cache or per-frame feature dumps to the public repo (rule 8).
- Sending frames or crops to a cloud model or a rented GPU (rule 9).
- A dependency or platform change that moves one number across a threshold (rule 7).
- Not having 3 registered team members by Nov 10 (rule 11).

## Sources

- Repository files: [RULES.md](../RULES.md), [FAQ.md](../FAQ.md), [EVALUATION.md](../EVALUATION.md),
  [OVERVIEW.md](../OVERVIEW.md), [SUBMISSION.md](../SUBMISSION.md),
  [possession_metrics.py](../evaluation/possession_metrics.py)
- Organiser answers (metrica-sports/more-than-a-hack-2027):
  [#2](https://github.com/metrica-sports/more-than-a-hack-2027/issues/2) detector trained on the match; GPU,
  [#8](https://github.com/metrica-sports/more-than-a-hack-2027/issues/8) how much may be hand-labelled,
  [#10](https://github.com/metrica-sports/more-than-a-hack-2027/issues/10) labels to choose parameters,
  [#11](https://github.com/metrica-sports/more-than-a-hack-2027/issues/11) reproducibility,
  [#14](https://github.com/metrica-sports/more-than-a-hack-2027/issues/14) committing stage outputs; watching,
  [#15](https://github.com/metrica-sports/more-than-a-hack-2027/issues/15) using future frames,
  [#16](https://github.com/metrica-sports/more-than-a-hack-2027/issues/16) IDSSE external data
- Open questions: [#17](https://github.com/metrica-sports/more-than-a-hack-2027/issues/17),
  [#20](https://github.com/metrica-sports/more-than-a-hack-2027/issues/20),
  [#22](https://github.com/metrica-sports/more-than-a-hack-2027/issues/22)
