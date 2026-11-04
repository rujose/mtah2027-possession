# Model v10: a clean base for the organisers' clarification

| | |
|---|---|
| **status** | **built 2026-11-03** (Jose approved the page with the eight recommended choices). Not yet run on our match: that's Jose's run (below) |
| **change from v9** | everything calibrated on our match's audio or its public truth is removed; everything trained or chosen on Metrica's samples gets an IDSSE-only twin, ready to switch to; the viewer stops showing our output over the video. The learned models, the keeper repair and v9's restart snap stay |
| **why** | the clarification of 2026-11-03 ([../limitations.md](../limitations.md), section 0, items B–D) |
| **code** | `possession/v10.py` (self-contained: v1–v9 are untouched, so their outputs can't change); `bench/chain.py` and a `--set all \| idsse` option for `bench.train_dead_ball`, `bench.train_restart`, `bench.eval_pathcrf`, `bench.restart_timing`; `bench/check_v10.py`; the viewer (`viewer/build_cache.py`, `viewer/static/app.js`, `viewer/server.py`) |
| **switches** | `BENCH_SET = "all"` (`"idsse"` if Metrica's samples can't be used); `RESTART_SNAP = True` (`False` if the organisers rule out v9's origin) |

## What the clarification asks, and what it touches

- **"Training, fitting or calibrating on the ground-truth possession data or on the provided video"
  is not allowed; tuning only on external data.** Two clues in v9 were calibrated on our match:

| clue | used since | calibrated on | how |
|---|---|---|---|
| **referee whistle** (audio) | v2; its weight raised in v5 | **the provided video's audio** | band "tones found at 2.6–3.9 kHz"; loudness thresholds "set by looking at the audio's distribution"; the slowdown test from our 86 events |
| **set-piece window** (Smart Tagging) | v1 | **the public ground truth** | the 15 s window from 3 tag offsets (2.8–12.2 s), the weak 0.3 weight because "2 of 5 restarts in the public window had no tag" |

- **"Checking your output against the video"** is not allowed. v5's change came from watching two
  moments of our output (frames 23697 and 54740). The viewer is built to do exactly that check: our
  labels and clues over the video, and its "watch changes" mode.
- **"Any dataset you have the rights to use."** Metrica's sample data has no licence file. It helped
  train v4 and v6 and choose the settings for v7 and v9. We've asked the organisers; until they
  answer, we can't be sure it's allowed.

The bench has no audio and no tags, so it never saw the whistle or the set-piece clue. **On the
bench v10 equals v9**, and the bench can't say what removing them costs on our match.

## The changes

### 1. The whistle goes

- **What:** no whistle events reach the decoder. v5's weight goes with it, and so does the audio
  decoding (one less dependency at reproduction time).
- **Why drop it rather than re-set its thresholds:** any threshold we pick now has been seen against
  our audio, and no external data with audio is available to set it again. Its reach was small
  anyway: 11 counted whistles in the whole match. Even v4, where the whistle weighed much less than
  in v5, ended in a stoppage after 8 of them.

### 2. The set-piece window goes

- **What:** Smart Tagging's set-piece tags no longer push the frames after them toward dead.
- **Why drop it:** the window and the weight were set from the public truth. The tag's own window
  (35 s, a fixed template) is too long to use as it is. Choosing any shorter window would mean
  choosing it on our match again.
- **What covers restarts without it:** v4's dead-ball model and v6's restart model, both learned on
  external data.

### 3. What stays from Smart Tagging

| clue | kept? | why it's clean |
|---|---|---|
| **before the 2nd-half kick-off**: dead from the start of the graded 2nd half to the kick-off tag | yes | the tag's meaning ("the half starts here"); the weight is certainty, not a measurement |
| **"a tag says this team has the ball"** (attacking tags for their team, defending tags for the other) | yes | the tags' meaning, learned by inspecting the data (allowed); its weight (0.2, rescaled in v3) is judgement and was never compared with the truth |

### 4. Metrica's samples: an IDSSE-only twin, ready

- **What:** each bench script that uses Metrica's samples gets `--set all | idsse`. Its outputs carry
  the set's name, and v10 has one switch, `BENCH_SET`, that picks which models and settings it uses.
- **Default until the organisers answer:** `"all"`. If the answer is no, the switch flips and the
  submission is rebuilt, with no new code.

| step | with `all` (today) | with `idsse` |
|---|---|---|
| v4's dead-ball model and its decoder settings | trained/chosen on 10 matches | 7 |
| v6's restart model, weight and carry | 10 | 7 |
| v7's PathCRF weight | chosen on the 4 matches PathCRF never saw (3 of them Metrica's) | **only J03WR9 is unseen**: chosen on it alone, so not held out (decision 6) |
| v9's snap window | 10 | 7 |
| v3's decoder | 7 IDSSE matches already | unchanged |

- **Run now, not only if needed:** the reruns use external data only (~40 min), so the numbers are
  ready to compare when the answer comes.

### 5. The viewer on our match shows only what we may inspect

| on our match | before | v10 |
|---|---|---|
| video, tracking dots, Smart Tagging, the public truth (first 5 min) | yes | yes: inspecting to understand what counts as possession is allowed |
| tracking | repaired by v8 | **as delivered** (the repair is our output) |
| label lanes for v1–v9 and the baselines; clue lanes | yes | **no** |
| notes | yes | **off** (notes on our match would be manually created data) |
| **bench matches** | everything | unchanged: external data, no video |

### 6. v9's restart snap stays, on the record

v9 started from Jose watching our labels on the video, which the clarification now forbids. Its
rule, though, rests only on external data:
- the convention comes from the bench's truth (698 of 698 restarts start on the pass);
- the settings were chosen on the bench;
- the error budget on the bench finds the same error on its own (restart timing, +0.018 before v9).

We've asked the organisers (question 5 in the email). If they say no, it's one switch off.

### 7. How we work from here (item A)

- **Claude doesn't run v10 on our match.** Jose runs it and the checks himself.
- **The script prints only what Jose needs:** the file, its SHA-256, the format check. It doesn't
  print label-free statistics to share, until the organisers say summary numbers are fine.
- **Claude doesn't read Jose's terminal** for those runs.
- Everything Claude runs is on external data (the bench).

## How it's checked

- **Older versions unchanged:** v1–v9 regenerate byte-identical to their recorded SHA-256s. Jose runs
  this on our match.
- **The bench:** v10 with `all` gives v9's numbers exactly (no audio or tags there). With `idsse`,
  every held-out number is reported against the `all` one: v4, v6, v7, v9, and the overall result.
- **Our match (Jose, privately):**
  - the file is valid and regenerates byte for byte;
  - the public 5 minutes as a pipeline check only. The organisers' list allows automatic checks
    with the provided data.

## Results (2026-11-03)

### v10 is v9 minus the two clues, exactly

- **`python -m bench.check_v10`:** on J03WR9, J03WMX and metrica1 (damaged tracking, no audio, no
  tags), v9 and v10 give the same labels on every frame (**0 frames differ, 3/3**). So v10 rebuilds
  v9's costs term by term. On our match the only differences are the whistle and the set-piece
  window, by construction.
- **The bench scripts' refactor changes nothing.** The `all` chain, re-run after it, reproduced:
  - both model files (same SHA-256);
  - all 10 bench prediction files;
  - every results file, except one new field (`"held_out": true` in `pathcrf_choice.json`).

### The IDSSE-only twins (7 matches, CC BY 4.0)

| stage | with all 10 (today) | with IDSSE only |
|---|---|---|
| v4 dead-ball model | trees, weight 0.5, dead price 50; held out **0.433** against v3's 0.412 (8/10) | **the same setting**; held out **0.399** against v3's 0.369 (**7/7**) |
| v6 restart model | logistic, weight 1, carry 3 s; **0.443** against 0.433 (8/10) | logistic, weight 1, **carry 5 s**; **0.411** against 0.404 (5/7) |
| v7 PathCRF weight | **0.5**, on every live frame; chosen on 4 unseen matches, held out **0.565** against v6's 0.501 (4/4) | **1.0**, on every live frame; chosen on J03WR9 alone, so **not held out**: 0.572 against v6's 0.472 there; 0.544 against 0.399 on the 6 it saw |
| v9 snap window | 5 s back, 1 s ahead; **+0.0056**, 10/10 | **the same window**; **+0.0016** (0.5497 against 0.5482), **5/7**: passes the bar (70 %, so 5 of 7) narrowly |
| v9 timing | moved starts within 0.2 s: 20 % → 83 % | 27 % → 79 % |

**Read:**
- **Every stage keeps its gain on IDSSE alone.** The settings barely move: the restart clue lasts
  5 s instead of 3, and PathCRF weighs twice as much.
- **Weakest points:** PathCRF's weight is picked on one match, and the snap's gain on IDSSE alone is
  small. Metrica's samples were where it gained most, since their ball behaves most like ours.
- **Smoke test:** with the final models on J03WR9 (seen in training, so not a score), `all` gives
  0.588 and `idsse` 0.585; 3,487 frames differ. Both settings load and run.

### What Jose runs on our match

Claude doesn't run these or read their output (the clarification's item A):

```bash
.venv/bin/python -m possession.v10                 # writes evaluation/results/v10.csv, prints its SHA-256
.venv/bin/python -m possession.v10                 # again: the SHA-256 must be the same
.venv/bin/python evaluation/validate_submission.py evaluation/results/v10.csv
.venv/bin/python -m viewer.server --rebuild        # the viewer for our match: no model output, notes off
```

## Failure modes

1. **Losing real stoppages.** Without the whistle, some of the 11 stoppages it marked may be missed
   (v4, with a weaker whistle, missed 3). Jose's run counts how many. That's the cost we accept for a
   clean entry.
2. **The 2nd-half kick-off tag** sits in the same code as the set-piece window. The change has to keep
   one while removing the other; checked by the clue counts on our match (Jose's run).
3. **IDSSE-only PathCRF weight:** chosen on one match, so it may be off. Reported, not hidden.
4. **"Judgement" weights decided after seeing the data** (v1's clue weights, the tag-team weight)
   could still be read as calibration. They were never compared with the truth or fitted. Variant
   10c (a learned per-frame model trained on external data) would retire them, if there's time.

## Decisions (Jose, 2026-11-03)

| | choice | taken | why |
|---|---|---|---|
| 1 | Drop the whistle, rather than re-set its thresholds | **drop** | No external audio to set them on; small reach (11 counted whistles in the match) |
| 2 | Drop the set-piece window, rather than use the tag's own 35 s with a small weight | **drop** | Any window we pick is picked on our match; the learned models cover restarts |
| 3 | Keep the tag-team clue and the 2nd-half kick-off clue | **keep** | The tags' meaning plus judgement weights, never compared with the truth |
| 4 | Keep v9's restart snap, disclosed in the email, with a switch to drop it | **keep** | Its rule and settings rest on external data only |
| 5 | Run the IDSSE-only twins now and switch on the organisers' answer | **yes** | The numbers are ready when the answer comes |
| 6 | With IDSSE only, PathCRF's weight chosen on J03WR9 alone rather than on all 7 (6 of which PathCRF trained on) | **J03WR9 alone** | The 6 seen matches flatter PathCRF and would push the weight up |
| 7 | The viewer on our match: tracking as delivered, truth, tags, video; no outputs, no notes | **yes** | What the clarification allows us to inspect |
| 8 | Name it v10; the touch line becomes v11 onwards | **yes** | One change per version |
