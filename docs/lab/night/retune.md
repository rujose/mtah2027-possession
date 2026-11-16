# Night, builder "retune": B's settings re-picked for `netmet5`'s P, leave one match out

| | |
|---|---|
| **status** | **finished 08:06, 2026-11-15. Fails the bar: re-picked leave one match out, `netmet5` is +0.0118 over v15's B on B's bench but better on 6 of 10 (7 needed) and DFL's 7 are down (−0.0008)**; SkillCorner passes (+0.0002). Re-picking costs 0.0021 against keeping trial 1015's settings. Question, candidates and bar written 07:20, before any candidate was scored on `netmet5`'s P. Nothing adopted; Jose decides |
| **code** | `bench/lab_retune.py` (new; imports `lab_netmirror` and `search_b` read only, edits nothing) |
| **outputs** | `data/external/lab_retune/ens4/` (the mirror), `bench/results/lab_retune/` (`candidates.json`, `judge.json`; `ens4/`: the check, every candidate's 30 units, SkillCorner) |
| **our match** | never read, nothing run on it |

## The question (fixed 07:20, before any score)

With settings re-picked for `netmet5`'s P, honestly, leave one match out, does `netmet5` pass the B bar?

`netmet5` (the mean of `netmet`'s seeds 0-3, [network.md](network.md)) was judged with **trial 1015's settings
fixed**: +0.0139 on B's bench, better on 6 of 10 (7 needed). Those settings were picked by round 8's search
([../branch-b.md](../branch-b.md)) for `gru_b`'s P, not `netmet5`'s, which handicaps it. This page re-picks them the
way round 8 did, nested, so the held-out match never helps choose its own settings.

**The candidate settings (fixed now, chosen without `netmet5`'s scores):** trial 1015 plus round 8's top 100 trials
by their round-8 mean over all 30 units, computed from round 8's stored per-trial results for `gru_b`
(`bench/results/search_b/unit_<match>_<variant>.csv`, all 1,500 trials on all 30 units;
`python -m bench.lab_retune candidates`). **Trial 1015 is round 8's #1 by that mean** (0.7326), so the list holds
**100 distinct trials, not 101**: 1015, then round 8's #2 to #100 (round-8 means 0.7326 down to 0.7280; trial 0 there
0.6918). The order (1015 first, then round 8's) is the tie-break order. The list, with each trial's round-8 mean:
`bench/results/lab_retune/candidates.json`:

1015, 1269, 1133, 702, 720, 425, 205, 1497, 183, 554, 913, 327, 1479, 1360, 493, 296, 586, 362, 82, 1387, 1108, 1166,
1280, 988, 509, 1084, 1130, 694, 1242, 1353, 1392, 919, 46, 618, 1151, 851, 70, 726, 773, 303, 733, 431, 801, 622, 953,
976, 1331, 1078, 72, 606, 77, 1111, 271, 333, 1136, 506, 193, 457, 238, 755, 427, 821, 990, 559, 1206, 1173, 136, 532,
1239, 1283, 472, 736, 429, 1105, 209, 1264, 507, 19, 621, 749, 1212, 451, 199, 104, 658, 39, 242, 1254, 897, 30, 234,
957, 53, 420, 1023, 1342, 286, 1320, 1181, 756.

**How each candidate is scored:** on `netmet5`'s P (the network builder's `data/external/lab_netmet/ens4/` files,
cloned into this page's own mirror), each trial as it would deploy, exactly as `search_b confirm` (and the builder's
`lab_netmirror confirm netmet ens4`) does it: its own stage models, trained on its own chain's out-of-fold letters on
its own data, without the held-out match, on all 30 units (10 matches × p25, p50, stressed). Every candidate's stage
tables are computed in this mirror; its cache started empty.

**The nested judgement:** for each of the 10 matches, the candidate with the best mean on the other 9 matches (ties:
trial 1015 first, then round 8's order); that match's score is the picked candidate's score on it (its 3 variants).
The nested scores are judged with the night's bar against v15's B (trial 1015 in round 8's confirm,
`bench/results/search_b/confirm_1015_554_586_1133_1269_-1_-2.csv`: 0.7329; p25 0.7289, p50 0.7319, stressed 0.7379):

1. B's bench (each match's mean of p25, p50, stressed) **+0.005 or more**;
2. **better on 7 of 10**;
3. **no stress variant down more than 0.002** (each variant's mean of the nested picks' scores on it);
4. **neither source (IDSSE's 7, Metrica's 3) down** on the mean;
5. **SkillCorner down no more than 0.005** against v15's B (0.4085): the candidate picked on all 10 (the in-sample
   best, same tie rule), scored on SkillCorner's 20 with `netmet5`'s four `all` nets averaged, as the builder did.

The in-sample best (the candidate best on all 10) is reported for information only, never as the verdict.

**The check, before any other candidate is scored:** this file's scoring path, with trials 0 and 1015, must give the
builder's `bench/results/lab_netmet/ens4/confirm_1015.csv` exactly on all 60 units. If it doesn't, the reason goes
here and nothing else runs.

**Forking paths, said plainly:** `netmet5` itself was registered at 03:37, after `netmet`'s seed 1 was seen
([network.md](network.md)); this re-pick was asked for at 07:20, after `netmet5`'s 6-of-10 result with trial 1015's
settings was seen. The candidate list and the rule are fixed before any of their `netmet5` scores, but the decision
to re-pick at all was made knowing the fixed-settings result missed the bar by one match.

**Time:** if the 100 candidates can't all be scored and judged by 09:25, this page says so, and what was scored.

## The run

Four workers (`python -m bench.lab_retune score --worker I/4`, each through `slot.py`), candidates I, I+4, …; logs in
the scratchpad's `logs/retune/`.

## Results

### The check (07:19): passes, exactly

`python -m bench.lab_retune check`: trials 0 and 1015 through this file's scoring path in the new mirror (trial 1015's
stage tables recomputed there) against the builder's `bench/results/lab_netmet/ens4/confirm_1015.csv`: **60 of 60
units equal, max |difference| 0.0** (B's bench: trial 0 0.6975, trial 1015 0.7468, as the builder). Two trials took
35 s for the stage models and 57 s for the 30 units on one slot. `bench/results/lab_retune/ens4/check.json`.

The four workers started at 07:19 (chunks of 6 trials: 4 to 7 min of stage models, 2.5 to 3 min of units).

At 07:37, with 24 of the 100 candidates scored, the judge was run once on the partial scores to test its code (no
SkillCorner, `complete: false`); nothing in the list, the rule or the bar was changed after it.

All 100 candidates were scored on all 30 units by 07:59 (`bench/results/lab_retune/ens4/scores_w{0..3}of4.csv`;
`python -m bench.lab_retune judge` → `bench/results/lab_retune/judge.json`). Trial 1015 inside the run gives 0.7468
again, as the check and the builder.

### The nested judgement (08:00): fails the bar (better on 6 of 10; DFL's 7 down)

Each match scored with the candidate best on the other 9 (`netmet5`'s P, every stage model trained without it):

| held-out match | picked on the other 9 | its score | v15's B | gain | `netmet5`, trial 1015 fixed (the builder's) |
|---|---|---|---|---|---|
| J03WMX | 919 | 0.6299 | 0.6254 | +0.0046 | +0.0089 |
| J03WN1 | 919 | 0.6596 | 0.6691 | **−0.0095** | −0.0007 |
| J03WOH | 702 | 0.7062 | 0.7199 | **−0.0136** | −0.0117 |
| J03WOY | 702 | 0.7185 | 0.6942 | +0.0243 | +0.0183 |
| J03WPY | 702 | 0.7388 | 0.7589 | **−0.0201** | −0.0124 |
| J03WQQ | 702 | 0.6891 | 0.6939 | **−0.0048** | −0.0011 |
| J03WR9 | 702 | 0.7262 | 0.7128 | +0.0134 | +0.0178 |
| metrica1 | 919 | 0.8676 | 0.8346 | +0.0330 | +0.0335 |
| metrica2 | 702 | 0.8563 | 0.7965 | +0.0598 | +0.0524 |
| metrica3 | 919 | 0.8544 | 0.8233 | +0.0311 | +0.0342 |
| **B's bench** | | **0.7447** | **0.7329** | **+0.0118, better on 6/10** | +0.0139, 6/10 |

| part of the bar | nested `netmet5` | needed | |
|---|---|---|---|
| 1. B's bench | **+0.0118** | +0.005 | passes |
| 2. better on | **6 of 10** | 7 | **fails** |
| 3. variants (p25 / p50 / stressed) | +0.0111 / +0.0106 / +0.0137 | none below −0.002 | passes |
| 4. sources (IDSSE's 7 / Metrica's 3) | **−0.0008** / +0.0413 (3/3) | neither below 0 | **fails** |
| 5. SkillCorner (trial 702, the pick on all 10) | **+0.0002** (0.4087 against 0.4085; better on 13/20) | −0.005 | passes |

- **Re-picking doesn't rescue `netmet5`; it costs.** The nested estimate (+0.0118) is **0.0021 under** the same P with
  trial 1015 kept fixed (+0.0139). Six of DFL's 7 do worse with their nested pick than with 1015 (by 0.002 to
  0.009; only J03WOY gains, +0.006); the losers against v15's B stay the same four (J03WN1, J03WOH, J03WPY, J03WQQ),
  and all four lose more.
- Only two candidates are ever picked: **702** (7 folds) and **919** (3 folds). Both have snap off, short and timing on,
  stage models trained on `both`, lower switch prices and a lower `timing` p_min than 1015 (702: w_gru 0.875,
  f_dead 1.19, switch 1.60 / 31.7; 919: w_gru 0.718, f_dead 1.39, switch 1.38 / 22.9, segments off). In round 8 they
  ranked #4 and #32 on `gru_b`'s P.
- Metrica's 3 gain about as much as with 1015 (+0.041 against +0.040); the re-pick moves DFL's 7 from +0.0027 to
  −0.0008, the part that decides the bar.

### The in-sample best (information only, never the verdict)

The candidate best on all 10 is **trial 702: 0.7494, +0.0165 over v15's B, better on 7 of 10** (p25 / p50 /
stressed +0.0162 / +0.0153 / +0.0180; DFL's 7 +0.0027, Metrica's 3 +0.0486). Trial 1015 ranks 17th of the 100 on
`netmet5`'s P. The best 10 sit between 0.7470 and 0.7494, the median candidate 0.7444, the worst 0.7372.

This number was picked on the 10 matches it is scored on. The gap between it and the nested estimate (+0.0165 against
+0.0118: 0.005) is the winner's curse of picking among 100 on 10 matches, larger than round 8's 0.002 (+0.0353 in
sample, +0.0334 nested) and about the size of the bar's +0.005. With hindsight, 43 of the 100 candidates clear the
bench parts in sample (gain, 7 of 10, DFL's 7 not down); 30 are better on only 6, 2 on 5. Whether `netmet5` beats
v15's B on 7 matches depends on which settings are picked with the answers in hand; the honest pick, without the
held-out match, gives 6.

### SkillCorner (08:05)

`python -m bench.lab_retune skillcorner --trials 1015 702 919 234 618` (started at 07:58 on the first free slot, for
the leaders at that moment, so that the pick on all 10 would be in it; the judged trial is the rule's, 702): `netmet5`'s
four `all` nets averaged, stage models on all 10 from this mirror's tables (`skillcorner_1015_702_919_234_618.csv`).

| trial | SkillCorner (20) | against v15's B (0.4085) |
|---|---|---|
| 0 (B as built) | 0.4077 | as round 8 |
| 1015 | 0.4090 | +0.0004 (the builder's file reproduced on all 20, max \|difference\| 0) |
| **702 (the pick on all 10: the judged one)** | **0.4087** | **+0.0002, better on 13/20: passes** |
| 919 | 0.4099 | +0.0013 (information) |
| 234 | 0.4064 | −0.0021 (information) |
| 618 | 0.4146 | +0.0061 (information) |

## Verdict

**`netmet5` with B's settings re-picked leave one match out fails the bar**: B's bench +0.0118 (passes +0.005) but
**better on 6 of 10** (7 needed) and **DFL's 7 down −0.0008** (neither source may go down); every variant up
(+0.011 to +0.014), Metrica's 3 up +0.041 (3 of 3), SkillCorner +0.0002 (passes). The re-pick does what the night's
noise numbers predicted: picking among 100 settings on 9 matches adds selection noise of about the size it removes
(the nested estimate is 0.0021 under trial 1015 kept fixed, and the in-sample best is 0.005 over the nested one).

What it says about the question behind it: trial 1015's settings were not what held `netmet5` under the bar. With
settings picked for its own P, without the held-out match, the same four DFL matches lose, by more.

## What Jose must decide

Nothing changes in what was already on the table: `netmet5` with trial 1015's settings (+0.0139, 6 of 10, DFL +0.0027,
Metrica +0.040, SkillCorner +0.0004) passes Metrica's rule, not the bar, and the re-pick doesn't add a case for it.

- **Whether to take `netmet5` further at all** under v14's precedent (a pass on Metrica's rule), knowing this page's
  honest re-pick is weaker than the fixed-settings number, and that the in-sample 7-of-10 of trial 702 (+0.0165) is
  hindsight: it would need its own nested or fresh-data confirmation before it could be read as a pass.
- **Whether Metrica's sample games may train a deployed network** (no licence file), as network.md says.
- If `netmet5` goes further: keep trial 1015's settings (this page found no honest reason to change them).

## Cost and footprint

The check (1.5 min, 1 slot), the 100 candidates (07:19-07:59, 4 slots), SkillCorner (07:58-08:05, 1 slot). Disk:
`data/external/lab_retune/ens4/` holds the 60 cloned P files (APFS clones, no extra space) and the 100 trials' stage
tables (about 0.4 GB). Nothing written outside `data/external/lab_retune/`, `bench/results/lab_retune/`,
`bench/lab_retune.py` and this page; no existing file edited; no commit.
