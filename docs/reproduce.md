# Reproducing our submission byte for byte

The deliverable is a public repo that regenerates our CSV byte for byte (RULES.md). This page says
how to run it, what has been checked, and where it could still break.

## How to run it

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements-team.txt   # Python 3.14.7
# the organisers' data in data/tactical/ (and data/broadcast/), as DATA.md describes
.venv/bin/python -m possession.v12        # writes evaluation/results/v12.csv; prints its SHA-256
.venv/bin/python evaluation/validate_submission.py evaluation/results/v12.csv
```

- **Nothing else is needed at run time:** the models are in `possession/models/` (trained on the
  bench by `bench/train_v12.py`; the bench data isn't needed to run v12). PathCRF's weights are
  downloaded once from the pinned commit into `data/external/pathcrf/` and refused if their SHA-256
  differs (`possession/pathcrf_adapter.py`).
- **Expected SHA-256** of `evaluation/results/v12.csv` (bench set `all`, macOS arm64):
  `bfd2548d764d0b95290939858fc4d0bc9481e89cc9bb03b441f759c826a88e24`.

## What has been checked (2026-11-04)

| check | result |
|---|---|
| `possession.v12` run twice | same SHA-256 |
| a clean clone of the commit (e2d043c), with only the organisers' data and PathCRF's weights | same SHA-256: nothing outside the repo is read |
| 1 thread, 4 threads, all 8 (`OMP_NUM_THREADS`) | same SHA-256 |
| the pins against the environment that made it | 3 packages pandas needs (`python-dateutil`, `six`, `tzdata`) were missing from `requirements-team.txt`: added, so a fresh install gets the same versions |

## Where it could still break: another CPU

Floating-point results can differ in their last bit between processors (Apple's ARM here, likely
Linux x86 at the organisers'). `python -m bench.check_reproduce` measures how much that matters, on
bench matches (never ours). Nudging v12's inputs by a relative
1e-7, about float32's last bit, changes **321 of 146,811 frames on J03WR9 and 205 of 145,007 on
metrica1**; a 1e-9 nudge changes none. By family:

| inputs nudged by 1e-7 | J03WR9 | metrica1 |
|---|---|---|
| the tracking | 0 | 2 |
| PathCRF's probabilities, used directly | 0 | 0 |
| v6's and v7's decoder costs | 318 | 163 |
| P(dead), P(restart), the restart clue | 116 | 28 |

So what matters is which numbers can differ by ~1e-7 on another CPU:
- values computed in **float64** (every numpy step, the trees, the regressions) differ by ~1e-16,
  and the costs are stored as float32 afterwards, which absorbs that: a flip needs the value within
  1e-16 of a float32 boundary (~1 in 10⁹ values);
- **PathCRF ran in float32** (PyTorch), where 1e-7 differences between CPUs are normal; its
  output feeds v7's costs;
- **the tracking is float32**, but it's read from the files exactly; what could differ is a
  float32 square root or angle computed from it (a nudge there moved 2 frames on one match).

**The fix: v12 runs PathCRF in float64** (`v12.PATHCRF_FLOAT64`; v7-v11 keep float32). Measured:

| | J03WR9 | metrica1 | our match |
|---|---|---|---|
| PathCRF float32 vs float64: largest difference | 1.3·10⁻⁶ (median 6·10⁻⁹) | 2.9·10⁻⁶ (median 1·10⁻⁸) | not looked at |
| v12's letters that change | **0** of 146,811 | **0** of 145,007 | **none**: the same SHA-256, `bfd2548d…a88e24` |

A float32-vs-float64 gap is larger than the gap between two CPUs running float32, and even that
changes nothing; in float64 the CPUs' differences shrink to ~1e-16. So on the evidence, the file
should come out the same on another CPU.

**Not yet tried: Linux.** There's no Docker on this machine, and the only remote is Metrica's. Once
our public repo exists, run the commands above on a Linux machine (a teammate's, or a CI job) and
compare the SHA-256. If it differs, run `python -m bench.check_reproduce` there too: it says which
inputs moved the letters.
