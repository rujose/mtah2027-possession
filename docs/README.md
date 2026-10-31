# Our documentation

Everything we know, decide and build for More than a Hack 2027 (Phase 1, ball possession), written
so that any team member, or a reviewer, can follow it without having been in the conversation.

| page | what it answers |
|---|---|
| [explained.md](explained.md) | **Start here if you're new:** every term in plain words, and what has been done, with the numbers to judge it |
| [challenge.md](challenge.md) | What is the task, how is it scored, what are the rules and dates? |
| [limitations.md](limitations.md) | Every limit we work under, what's allowed but not obvious, grey zones to ask about, traps |
| [research/](research/) | What models, data and advice exist for this problem, and what we take from it |
| [data.md](data.md) | What is in every data file, measured, and which ones we use and how? |
| [bench.md](bench.md) | The test bench: 7 DFL matches plus 3 Metrica sample games, damaged to look like ours, where versions are compared and settings chosen |
| [convention-check.md](convention-check.md) | Do Metrica's possessions change hands as often as DFL's, the bench's labels? (About 20 % less often, but v3's settings still win on Metrica's labels) |
| [models/](models/) | One page per model version: why, how, every setting and where it comes from, expected failures, results |
| [decisions.md](decisions.md) | Every decision we took, when, and why |
| [viewer.md](viewer.md) | The match viewer: video + tracking + labels + clues on one frame; how to run it |

## Conventions

- **Every number about the data comes from a script**, currently `python tools/explore_data.py`.
  If a doc and the script disagree, the script is right and the doc gets fixed.
- **Every model page is written before the model is built**, and updated with results after.
- **Nothing here is tuned on the public 5 minutes.** They are used only to check that a pipeline is
  not broken (see [decisions.md](decisions.md)).
- These pages are the raw material for the technical document the submission requires (approach,
  assets used and dropped, possession rules, limitations, how to reproduce).

## Models

| version | idea | status | full match | [bench](bench.md), 41 % damage: spellF1 (possessions found / true) | public 5 min (check only) |
|---|---|---|---|---|---|
| [v1](models/v1-clues-and-decider.md) | per-frame clues + Viterbi decider with switch penalties | built 2026-10-30 | 173 possessions, mean 23.9 s | 0.224 ± 0.034 (148 / 437), without its tag clues | macroF1 0.72, spellF1 0.43 |
| [v2](models/v2-whistle.md) | v1 + referee whistle from the audio, counted only if players slow down | built 2026-10-30 | 177 possessions, mean 22.9 s; dead 36.3 % | can't be benched (no audio) | macroF1 0.79, spellF1 0.43 |
| [v3](models/v3-decoder.md) | v2's clues + a decoder with minimum lengths and lower switch prices, settings chosen on the bench | built 2026-10-30 | 429 possessions, mean 8.9 s; dead 39.9 % | **0.369 ± 0.026** held-out (280 / 437), better than v1 on 7/7; Metrica's games: **0.512** (344 / 351) | macroF1 0.70, spellF1 0.34 |
| [v4](models/v4-dead-ball.md) | v3 + a dead-ball probability learned from the players on the bench (gradient-boosted trees), replacing the "stopped / active" rules | built 2026-10-30 | 428 possessions, mean 8.0 s; dead 46.2 % | **0.433 ± 0.065** held-out over all 10 (v3 0.412), better on 8/10; stoppages found 81 % (v3 48 %) | macroF1 0.82, spellF1 0.44 |
| [v5](models/v5-whistle-weight.md) | v4 with the counted whistle's weight = v2's 2.0 + the model's strongest live evidence (4.30) | built 2026-10-30 | 426 possessions; dead 46.3 %; counted whistles followed by a stoppage 11/11 (v4 8/11) | not benchable (no audio); same as v4 there | macroF1 0.82, spellF1 0.44 |
| [v6](models/v6-restart-team.md) | v5 + who restarts, learned from the players' layout at stoppages (logistic regression); revised so only the unlikely team pays and the clue waits for 2 s of dead-looking play | built and revised 2026-10-30 | 439 possessions; dead 46.5 % | **0.443 ± 0.060** held-out (v4/v5 0.433), better on 8/10; right team after a stoppage 62 → 87 % (IDSSE), 72 → 85 % (Metrica) | macroF1 0.82, spellF1 0.44 |
| floor: `baseline_smooth50` | nearest player + majority vote ±2 s | organisers' | 594 possessions, mean 7.2 s; spellF1 **0.2433** | 0.217 ± 0.018 (448 / 437) | macroF1 0.62, spellF1 0.33 |

## Data from other matches

- **IDSSE**: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of
  spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195.
  https://doi.org/10.1038/s41597-025-04505-y. Figshare article 28196177, licence CC BY 4.0. Used
  for the [bench](bench.md) only: never as an input to the submission.
- **Metrica Sports sample data**: three anonymised matches, tracking and events,
  github.com/metrica-sports/sample-data (no licence file; the README asks users to acknowledge the
  source). Used for the [convention check](convention-check.md) and as the bench's second set: never
  as an input to the submission.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-team.txt   # exact versions
.venv/bin/python tools/explore_data.py            # every number in data.md
.venv/bin/python -m possession.v1                 # writes evaluation/results/v1.csv
.venv/bin/python -m possession.v2                 # writes evaluation/results/v2.csv
.venv/bin/python -m possession.v3                 # writes evaluation/results/v3.csv
.venv/bin/python -m possession.v4                 # writes evaluation/results/v4.csv
.venv/bin/python -m possession.v5                 # writes evaluation/results/v5.csv
.venv/bin/python -m possession.v6                 # writes evaluation/results/v6.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/v1.csv
.venv/bin/python evaluation/run_evaluation.py     # public 5 minutes, check only

# the bench (other matches; see bench.md)
.venv/bin/python -m bench.idsse download          # once, 2.6 GB
.venv/bin/python -m bench.idsse parse
.venv/bin/python -m bench.measure                 # our tracking's faults -> bench/damage_stats.json
.venv/bin/python -m bench.run                     # -> bench/results/scores.csv
.venv/bin/python -m bench.tune_decoder            # v3's decoder settings -> bench/results/decoder_*
.venv/bin/python -m bench.metrica download        # once, ~190 MB
.venv/bin/python -m bench.metrica parse
.venv/bin/python -m bench.convention              # Metrica vs DFL convention -> bench/results/convention.csv
.venv/bin/python -m bench.run --set metrica       # the bench on Metrica's games
.venv/bin/python -m bench.tune_decoder --set all  # the decoder search on all 10 matches
.venv/bin/python -m bench.train_dead_ball         # v4's dead-ball model and settings -> possession/models/
.venv/bin/python -m bench.train_restart           # v6's restart model and settings -> possession/models/
```
