# More than a Hack 2027, Phase 1: our ball-possession labels

This repository produces our submission to [More than a Hack 2027](ORGANISERS-README.md), Phase 1: one label per
frame of the tactical video (`H` home, `A` away, `D` dead ball, `X` not graded), 199,319 rows. The submitted file
comes from model **v17**, and the commands below regenerate it **byte for byte** from the organisers' data bundle.

The challenge itself is described in the organisers' documents, kept here unchanged:
[ORGANISERS-README.md](ORGANISERS-README.md) (their README, renamed so this page can be ours),
[CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [RULES.md](RULES.md),
[SUBMISSION.md](SUBMISSION.md), [FAQ.md](FAQ.md), [OVERVIEW.md](OVERVIEW.md).

## Regenerate the submission

**1. Install** (Python 3.14.7; the pins are the exact versions the file was made with):

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements-team.txt
```

**2. Add the organisers' data** to `data/`, with the names it arrives with ([data/README.md](data/README.md)), or
point `MTAH_DATA_DIR` at it. v17 reads these files and nothing else from the bundle (the videos are not read):

```
data/tactical/   BAR-ATH_tactical_atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml
data/broadcast/  BAR-ATH_broadcast_atd.xml, _atd.txt, _homography.csv
```

**3. Run v17 and check the file:**

```bash
.venv/bin/python -m possession.v17
.venv/bin/python evaluation/validate_submission.py evaluation/results/v17.csv
```

`possession.v17` writes `evaluation/results/v17.csv` and prints its SHA-256, which should be:

```
d7b159c5bc392abd0240f5d2d16a893c26694b5d7ab3cb8bb5dafa4dec69ec04
```

Our submitted `<team_name>.csv` is this file, renamed as [SUBMISSION.md](SUBMISSION.md) asks.

- **Run time:** about 4 minutes on a laptop (macOS, Apple silicon).
- **Network:** pip, and on the first run PathCRF's pretrained weights, downloaded once from the pinned upstream commit
  into `data/external/pathcrf/` and refused if their SHA-256 differs. Every other model is in `possession/models/`.
- **If the broadcast folder is missing** or can't be lined up, v17 falls back to v16 and writes v16's file
  (SHA-256 `90599088419d812cbb341168b27ff0ff767549b75c8dfa12a51de290ccc0523c`), so a different hash usually means a
  missing file.
- **Checked:** a fresh clone with a fresh environment and only the organisers' data gives the same SHA-256; so do 1
  and 2 threads (`OMP_NUM_THREADS`). Tested on macOS arm64 only; why it should hold on other CPUs (the parts
  sensitive to rounding run in float64) is in [docs/reproduce.md](docs/reproduce.md).

## How it works

Every learned part, and every setting, was fitted on **other matches** (listed below), never on the provided
game's ground truth, its public 5 minutes or its video, and we made no labels of the provided game. Candidate
versions were compared on a test bench of other matches, damaged to look like ours ([docs/bench.md](docs/bench.md)).
Where the provided game still shaped a choice without labels (the bench's damage, and the simulated broadcast, follow
statistics of our tracking), it is listed in the audit at the top of [docs/limitations.md](docs/limitations.md),
with what happened before the organisers' clarification of 2026-11-03 and what we changed after it.

The short version is our technical document, [docs/technical-document.md](docs/technical-document.md). The method,
version by version, with every setting and where it comes from, is in [docs/](docs/README.md):
start at [docs/explained.md](docs/explained.md); v17 is [docs/models/v17-broadcast.md](docs/models/v17-broadcast.md),
and each earlier version has its own page in [docs/models/](docs/models/). Every decision, with its date and reason,
is in [docs/decisions.md](docs/decisions.md).

## Data from other matches

| dataset | what we used it for | licence |
|---|---|---|
| **IDSSE**: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. [doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y) | 7 matches: training and the bench | CC BY 4.0 |
| **Metrica Sports sample data**: [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), 3 anonymised matches | training and the bench | no licence file; credited as its README asks |
| **SkillCorner open data**: [github.com/SkillCorner/opendata](https://github.com/SkillCorner/opendata), data by SkillCorner | testing candidates on broadcast-quality tracking | MIT |
| **PathCRF** pretrained weights (below) | one input to the per-frame model | MPL-2.0 |

None of this data is in the repository: `bench/` and [docs/bench.md](docs/bench.md) say where to download it to
retrain the models (`bench/train_*.py`), which is not needed to regenerate the file.

## What is where

| path | what |
|---|---|
| `possession/` | our models, one module per version (`v17.py` is the submission) and their shared parts |
| `possession/models/` | trained model files (other matches only) |
| `bench/` | the test bench, the training scripts, the lab's experiments and their results |
| `docs/` | everything we built, measured and decided, and why |
| `viewer/` | a local viewer (video, tracking and labels on one frame) used during development |
| `tools/` | data exploration (`explore_data.py`: every number in [docs/data.md](docs/data.md)) |
| `third_party/pathcrf/` | PathCRF, vendored unchanged ([VENDORED.md](third_party/pathcrf/VENDORED.md)) |
| `reports/`, `research_notes/` | early research on possession from tracking (the survey is [docs/research/](docs/research/README.md)) |
| `evaluation/`, `games.py`, `starter/` | the organisers': scorer, validator, data paths and starter notebooks |

## Credits and licences

- **The organisers' code and documents** (Metrica Sports, FC Barcelona, Mobile World Capital Barcelona): MIT, see
  [LICENSE](LICENSE) and its scope note.
- **PathCRF**: Kim H, Lee K, Seo S, Ko S-K, Yoon J, Park C. *PathCRF: Ball-Free Soccer Event Detection via Possession
  Path Inference from Player Trajectories*. KDD 2026. [doi:10.1145/3770855.3818152](https://doi.org/10.1145/3770855.3818152).
  Source [github.com/hyunsungkim-ds/pathcrf](https://github.com/hyunsungkim-ds/pathcrf), MPL-2.0
  ([third_party/pathcrf/LICENSE](third_party/pathcrf/LICENSE)).
- **The match data** is licensed by the organisers for this competition only. It is not in this repository and must
  not be redistributed.
