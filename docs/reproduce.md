# Reproducing our submission byte for byte

The deliverable is a public repo that regenerates our CSV byte for byte (RULES.md). This page says
how to run it, what has been checked, and where it could still break.

## How to run it

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements-team.txt   # Python 3.14.7
# the organisers' data in data/tactical/ (and data/broadcast/), as DATA.md describes
.venv/bin/python -m possession.v17        # writes evaluation/results/v17.csv; prints its SHA-256
.venv/bin/python evaluation/validate_submission.py evaluation/results/v17.csv
```

**v17 (since 2026-11-15, evening)**: expected SHA-256 of `evaluation/results/v17.csv` (bench set `all`, macOS arm64):
`d7b159c5bc392abd0240f5d2d16a893c26694b5d7ab3cb8bb5dafa4dec69ec04` (three runs the same: twice with
`OMP_NUM_THREADS=2`, once with 1). v17 is v16 with the broadcast tracking's extra players merged into what branch B
reads ([v17](models/v17-broadcast.md)); **it reads the broadcast set too** (`data/broadcast/`: the `_atd.xml`,
`_atd.txt` and `_homography.csv`; not the video). Without them it falls back to v16 and writes v16's file
(`90599088…0523c`), so a missing broadcast folder shows as v16's SHA-256. No new model file. **Clean copy
(2026-11-15):** a fresh `git clone` of commit 3cce485 with the working tree's changed files and v16's and v17's new
files copied in (`possession/v16.py`, `v17.py`, `broadcast_merge.py`, `models/v16_b_all.joblib`), and only the
organisers' data (tactical and broadcast, no video) and PathCRF's weights in `data/`, gives the same SHA-256, and
`validate_submission.py` passes. Not yet tried on Linux.

**v16 (2026-11-15)**: expected SHA-256 of `evaluation/results/v16.csv` (bench set `all`, macOS arm64):
`90599088419d812cbb341168b27ff0ff767549b75c8dfa12a51de290ccc0523c` (three runs the same: twice with
`OMP_NUM_THREADS=2`, once with 1). v16 is v15 with branch B's network replaced by four GRUs trained on Metrica's
sample games too ([v16](models/v16-netmet.md)); branch B reads `possession/models/v16_b_all.joblib` (3.4 MB, the
four networks run in float64). **Clean copy (2026-11-15):** a fresh `git clone` of commit 3cce485 with the working
tree's changed files and v16's two new files copied in, and only the organisers' data and PathCRF's weights in
`data/`, gives the same SHA-256, and `validate_submission.py` passes. Not yet tried on Linux.

**v15 (2026-11-14 to 11-15)**: expected SHA-256 of `evaluation/results/v15.csv` (bench set `all`, macOS arm64):
`209a2081beae40b1a7971fa958f87e7ca495536eee673af10a37df87069b7f5e` (twice the same at the default threads, and
the same with `OMP_NUM_THREADS=1`; format check passes). v15 is v14 with a new
branch B ([v15](models/v15-branch-b.md)); branch B reads `possession/models/v15_b_all.joblib` (its GRU run in float64,
as v13's). v14's file was v13's (below): the switch takes branch B on our match, where v14 changed nothing.
**Clean clone (2026-11-14):** a fresh `git clone` of commit 3cce485 with only the organisers' data and PathCRF's
weights in `data/` (the same Python environment) gives the same SHA-256, and the format check passes (199,319 rows).
Not yet tried on Linux.

**v13 (2026-11-09 to 11-14; v14 gave the same file)**: expected SHA-256 of `evaluation/results/v13.csv` (bench set `all`, macOS arm64):
`3efbef53e4394c318815fa33d8284af9ab6f675667c09008dfdba2e51a54a7c4` (twice the same at the default threads, and
the same with `OMP_NUM_THREADS=1`; format check passes).
v13 adds a GRU in PyTorch, run in **float64** like PathCRF (float32 against float64 changed 0 letters on four
bench matches, [v13](models/v13-temporal.md)), and the stoppage switch's models; everything it reads at run
time is in `possession/models/` (`v13_all.joblib`, `v13_augment_all.joblib`) or the organisers' data. Not yet
tried from a clean clone or on Linux. The sections below were checked on v12.

- **Nothing else is needed at run time:** the models are in `possession/models/` (trained on the
  bench by `bench/train_v12.py`; the bench data isn't needed to run v12). PathCRF's weights are
  downloaded once from the pinned commit into `data/external/pathcrf/` and refused if their SHA-256
  differs (`possession/pathcrf_adapter.py`).
- **Expected SHA-256** of `evaluation/results/v12.csv` (bench set `all`, macOS arm64), since the
  second-half kick-off was restored on 2026-11-09: `6219237a07110f7e4d0ee8b76f67c4125efb5e4418e35db018ea5032aa6a6067`
  (before: `bfd2548d…a88e24`; the checks below were made on that version, and the change touches only
  the frames before the second-half kick-off).

## Does it behave on our match as on the bench? (label-free)

```bash
.venv/bin/python -m bench.transfer_check          # the bench's ranges (v12 out of fold)
.venv/bin/python -m bench.transfer_check --ours   # Jose: our match against them, PASS / NEAR / WARN only
```

Other teams found bench gains that didn't carry over to the match. Three numbers need no truth: the share
of graded frames called D, possessions per graded minute, and how often v12 names the team of the player
with the ball clearly at his feet (within 1 m, the ball slower than 2 m/s over 0.2 s, nobody from the other
team within 3 m). On the bench (v12 out of fold): **dead share 0.314–0.533, possessions per minute
3.17–4.12, clear control 0.991–0.998**. `--ours` prints only where our match falls against those.

## What has been checked (2026-11-04)

| check | result |
|---|---|
| `possession.v12` run twice | same SHA-256 |
| a clean clone of the commit (20003c6), with only the organisers' data and PathCRF's weights | same SHA-256: nothing outside the repo is read |
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
