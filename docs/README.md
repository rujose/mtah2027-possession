# Our documentation

Everything we know, decide and build for More than a Hack 2027 (Phase 1, ball possession), written
so that any team member, or a reviewer, can follow it without having been in the conversation.

| page | what it answers |
|---|---|
| [challenge.md](challenge.md) | What is the task, how is it scored, what are the rules and dates? |
| [data.md](data.md) | What is in every data file, measured, and which ones we use and how? |
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

| version | idea | status | full match | public 5 min (check only) |
|---|---|---|---|---|
| [v1](models/v1-clues-and-decider.md) | per-frame clues + Viterbi decider with switch penalties | built 2026-10-30 | 173 possessions, mean 23.9 s | macroF1 0.72, spellF1 0.43 |
| floor: `baseline_smooth50` | nearest player + majority vote ±2 s | organisers' | 594 possessions, mean 7.2 s; spellF1 **0.2433** | macroF1 0.62, spellF1 0.33 |

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/explore_data.py            # every number in data.md
.venv/bin/python -m possession.v1                 # writes evaluation/results/v1.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/v1.csv
.venv/bin/python evaluation/run_evaluation.py     # public 5 minutes, check only
```
