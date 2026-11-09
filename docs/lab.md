# The lab: candidate versions built side by side (from 2026-11-04)

Jose, 2026-11-04: build different versions with subagents, analyse them, keep what's valuable and
discard the rest, mix the winners into a better version, and keep improving. This page is the
protocol they all follow and the log of what each round found. Bench only: nothing here reads our
match.

## The harness: `bench/lab.py`

```bash
.venv/bin/python -m bench.lab --build            # cache the 10 bench matches (after bench.eval_pathcrf)
.venv/bin/python -m bench.lab --build --set idsse
.venv/bin/python -m bench.lab                    # v11 per match: the number to beat
```

- **One cache per match** in `data/external/lab[_idsse]/` (gitignored): the damaged tracking at 41 %
  (as the models see it), the truth, and every out-of-sample output from v4 to v11 on that match:
  v7's letters and its decoder costs, P(dead), the restart and PathCRF probabilities, v9's and
  v11's letters. "Out of sample" means each match was predicted by models trained without it
  (PathCRF itself was pretrained on 6 of the IDSSE matches: see "Unseen" below).
- **Loaded one match at a time** (`lab.each`, ~100 MB each), so several candidates can run at once
  on an 8 GB machine.
- **`lab.judge(grid, columns)`** scores a candidate: each match with the setting best on the other
  matches (leave one match out), against v11 on the same match.

## What a candidate may use

- To predict a match: **only that match's damaged tracking and the out-of-sample outputs on it.**
  Never its truth or its clean tracking (the clean tracking isn't in the cache).
- To train or choose settings: the other bench matches, truth included (they're external data).
- Never our match, its video, its tracking, its labels or anything derived from them.

## The bar (fixed before any candidate ran)

A candidate is kept only if, with each match scored on a setting chosen without it:

1. the held-out mean spellF1 beats v11's on the same matches;
2. at least 70 % of the matches improve (7 of 10; 5 of 7 with `--set idsse`);
3. **by source**: the mean doesn't drop on IDSSE's matches, nor on Metrica's (our match is
   Metrica-tracked, but most of the bench is IDSSE);
4. **unseen**: the mean doesn't drop on the 4 matches PathCRF never saw (J03WR9 and Metrica's 3),
   the only ones where every input is truly out of sample.

**Unseen.** PathCRF was pretrained on 6 of the 7 IDSSE matches, so on those its probabilities are
better than they will be on our match. A candidate that leans on PathCRF must hold on the 4
unseen matches, not only on average.

**Mixing.** Winners are combined in order of their gain, each added only if the mix still clears
the bar against the mix without it. The final version is judged once more on its own: on both bench
sets (`all` and `idsse`), and at 30 % and 55 % damage where it can be.

**Many tries, one bench.** Trying many candidates on the same 10 matches flatters the best of them
(the winner's curse). The bar's per-match and per-source checks are there to catch that; a gain
under ~+0.002 that passes is noted as fragile.

## Rounds

### Round 1 (2026-11-04): five builders, each on v11

| candidate | page | held out vs v11 (0.5727) | matches better | IDSSE / Metrica / unseen | `--set idsse` (v11 0.5625) | verdict |
|---|---|---|---|---|---|---|
| **short**: short possessions at clear touches (a learned clarity check) | [short](lab/short.md) | 0.5909 (+0.0182) | 10/10 | +0.016 / +0.023 / +0.020 | +0.0209, 7/7 | **kept** |
| **segments**: each predicted possession checked; spurious ones merged into the stoppage, wrong teams flipped | [segments](lab/segments.md) | 0.5873 (+0.0146) | 7/10 | +0.004 / +0.040 / +0.037 | +0.0100, 6/7 | **kept** (narrowly) |
| **decoder**: minimum possession and team-switch price chosen again | [decoder](lab/decoder.md) | round 1 +0.0128, 10/10; with its 2nd grid round 0.5793 (+0.0065) | 6/10 | −0.001 / +0.024 / +0.019 | +0.0089, 6/7 | **discarded** (the whole grid fails; round 1 tried in the mix and failed there) |
| **stoppage**: each team → D change moved to just after the ball is seen leaving the pitch | [stoppage](lab/stoppage.md) | 0.5777 (+0.0050) | 10/10 | +0.005 / +0.005 / +0.004 | +0.0032, 5/7 | **kept** |
| **learned**: a per-frame classifier (P of H / A / D, trained on IDSSE) in place of v1's hand weights, then the decoder | [learned](lab/learned.md) | 0.6610 (+0.0882) | 10/10 | +0.076 / +0.117 / +0.108 | +0.0751, 7/7 | **kept** (see its dead-ball caveat) |

**Mixing** (`python -m bench.lab_mix`): post stages in pipeline order, each trained stage fitted on
the other matches' out-of-fold output of the stages before it, each stage on its own fold's setting.

| chain | against | gain | matches better | verdict |
|---|---|---|---|---|
| short | v11 | +0.0182 | 10/10 | kept |
| short → segments | short | +0.0182 | 9/10 | kept |
| decoder → short → segments | short → segments | +0.0038 | 5/10 | decoder out |
| **stoppage → short → segments** | short → segments | +0.0065 | 8/10 | **kept: the mix** |
| decoder on top of the mix | the mix | +0.0017 | 6/10 | out |

**The mix: 0.6156 against v11's 0.5727 (+0.0429), better on 9/10** (IDSSE +0.029, Metrica
+0.074, unseen +0.066; macroF1 0.8775 against 0.8705). On `--set idsse`, with that set's own
settings: 0.6001 against 0.5625 (+0.0376), 7/7, each step passing. The one match it loses, J03WMX
(−0.0075), has the most short possessions in the truth (511): segments merges too many there.
It predicts ~302 possessions a match against ~411 true: fewer, and more of them right.

**What's left** (`python -m bench.lab_budget`, oracles on the mix; on v11 in brackets):

| error | gain if fixed |
|---|---|
| missed possessions (147 a match: median 1.8 s, 57 % under 2 s, 51 % between two of the other team, 49 % next to a stoppage) | +0.252 (+0.245) |
| team-change timing | +0.048 (+0.061) |
| team of each possession | +0.027 (+0.057) |
| spurious possessions | +0.026 (+0.050) |
| missed stoppages | +0.022 (+0.022) |
| restart timing / stoppage start timing / false stoppages | +0.007 / +0.005 / +0.004 |

### Round 2: building on the mix

- **Input:** the mix's out-of-fold letters, `lab_mix.kept_letters(match.id, which)` (each match's
  trained stages fitted without it).
- **Judged against the mix:** `lab.judge(grid, columns, which, base=lab_mix.kept_base(which))`,
  same bar.
- A trained candidate fits on the other matches' kept letters, never the held-out match's truth.

| candidate | page | held out vs the mix (0.6156) | matches better | IDSSE / Metrica / unseen | `--set idsse` (mix 0.6001) | verdict |
|---|---|---|---|---|---|---|
| **recall**: proposals inside each possession (the other team's player within 1 m and nearest, round 1's touch candidates, PathCRF) scored by a classifier | [recall](lab/recall.md) | 0.6329 (+0.0172); nested +0.0162 | 9/10 | +0.016 / +0.021 / +0.019 | +0.0078, 5/7, J03WR9 −0.0008 (fails "unseen") | **kept on `all`** |
| **timing**: each change of team moved to the candidate frame a ranker trusts most (touches, the new team reaching the ball) | [timing](lab/timing.md) | 0.6282 (+0.0126); nested +0.0135 | 10/10 | +0.005 / +0.030 / +0.024 | +0.0039, 5/7 | **kept** |
| **dead**: a dead-ball classifier (ball and players, trained on IDSSE) writes D inside possessions | [dead](lab/dead.md) | 0.6278 (+0.0121); nested +0.0120 | 10/10 | +0.012 / +0.012 / +0.010 | +0.0122, 6/7, J03WR9 −0.0017 (fails "unseen") | **kept on `all`** |
| polish: the mix's own settings chosen again inside the chain (plus stoppage and short run again after segments) | [polish](lab/polish.md) | 0.6207 (+0.0051); fully nested +0.0072 | 8/10 (nested 7/10) | +0.005 / +0.004 / +0.0003 | +0.0046, 5/7 (nested +0.0085, 6/7) | passes narrowly, **superseded**: it tunes the chain that starts from v11, and round 3's chain starts from the learned model |

**A rule made explicit here:** each bench set gets the stages that pass on it. Round 1's stages
passed on both sets, so the question didn't come up; recall and dead pass on `all` but not on
`idsse`, so an IDSSE-only chain leaves them out.

**Chains** (each stage against the same chain without it; `python -m bench.lab_mix --round 2`,
`--round 2-dead`):

| chain | `all`: held out | vs v11 | steps | `idsse` |
|---|---|---|---|---|
| mix → recall → timing | 0.6475 | +0.0747 | each step 10/10 or 9/10 | recall fails "unseen" |
| **mix → timing → recall** | 0.6473 | +0.0745, 10/10 | timing 10/10, recall +0.0190 10/10 | 0.6123 (+0.0498, 7/7), every step passes |
| mix → timing → recall → dead | 0.6586 | +0.0859, 10/10 | dead +0.0114, 10/10 | (dead off on `idsse`) |

Timing before recall was chosen over the other order because it passes every step on both sets;
on `all` the two orders differ by 0.0002.

### Round 3 (2026-11-04): the learned model as the decoder's input, post stages greedily on top

`python -m bench.lab_mix --round 3`: start from the learned model's held-out probabilities, decoded
(then v9's snap and v11's move); add each post stage in pipeline order only if it clears the bar
against the chain so far. Two starts: the **full** model, and its **team half only** (v7's
dead-ball evidence kept), which can't read how the bench's ball behaves in stoppages (learned.md,
caveat 1).

| step | full learned: gain over the chain so far | matches better | team half: gain | matches better |
|---|---|---|---|---|
| start (vs v11) | +0.0883 | 10/10 | +0.0515 | 10/10 |
| + stoppage | +0.0018 | 7/10, kept | +0.0058 | 10/10, kept |
| + short | +0.0235 | 10/10, kept | +0.0216 | 9/10, kept |
| + segments | +0.0021 | 5/10, **out** | +0.0009 | 5/10, **out** |
| + timing | +0.0056 | 8/10, kept | +0.0047 | 8/10, kept |
| + recall | +0.0098 | 9/10, kept | +0.0094 | 8/10, kept |
| + dead | +0.0001 | 5/10, **out** | +0.0050 | 7/10, kept |
| **the chain** | **0.7019 (+0.1291 vs v11), 10/10** | | **0.6707 (+0.0980), 10/10** | |

Full chain by source: IDSSE 0.6727 (+0.105, 7/7), Metrica 0.7700 (+0.184, 3/3), PathCRF-unseen
0.7480 (+0.164, 4/4); macroF1 0.9158 against 0.8705. On top of the full model, segments and the
dead-ball stage no longer pay: the learned model already does their work.

**On `--set idsse`** the greedy chain keeps different marginal stages: **learned → short → segments →
recall, 0.6642 against v11's 0.5625 (+0.1017), 7/7** (stoppage +0.0004 4/7 out, timing +0.0005 2/7
out, segments +0.0041 5/7 in, dead out). Team half: learned_team → stoppage → short → segments →
recall, 0.6273 (+0.0648), 7/7. Learned, short and recall pass on both sets; stoppage, timing and
segments are worth a few thousandths on top of learned and flip with the set.

**The open risk** (learned.md, caveat 1): the ball is the learned model's main input, and IDSSE
freezes its ball in dead time. Our ball is missing 83 % of the time players stand still (bench.md),
so "no ball" should transfer; whether our *seen* ball stands still in stoppages is unknown and not
looked at. Round 4 (`robust`) retrains the model with any ball standing still for a while blanked,
on every match, so it can't lean on the frozen ball.

### Round 4 (2026-11-04): the learned model made robust to the frozen ball

[robust](lab/robust.md): any run of a seen ball standing still (under 0.3 m/s) for at least 1 s
becomes "ball missing", on every match, in training and in prediction; the learned model retrained
the same way. It blanks under 1 % of live frames and 44 % of IDSSE's dead frames, so IDSSE's dead
time now looks like Metrica's and ours (missing). Held out 0.6728 against v11 (+0.1001, 10/10);
`--set idsse` 0.6365 (+0.0739, 7/7). Against learned: level on IDSSE (−0.001), +0.042 on Metrica
(the blanked IDSSE ball now speaks Metrica's convention: not counted as a robustness gain). A ball
that *moves* in our stoppages would still fool it, as it would learned.

**Round 3's greedy chain from robust** (`python -m bench.lab_mix --round 3 --starts robust`):

| step | `all`: gain over the chain so far | better | `idsse`: gain | better |
|---|---|---|---|---|
| robust (vs v11) | +0.1001 | 10/10 | +0.0740 | 7/7 |
| + stoppage | −0.0005 | 3/10, out | −0.0003 | 3/7, out |
| + short | +0.0171 | 10/10, kept | +0.0232 | 6/7, kept |
| + segments | +0.0089 | 10/10, kept | +0.0045 | 6/7, kept |
| + timing | +0.0044 | 8/10, kept | +0.0004 | 4/7, out |
| + recall | +0.0044 | 6/10, out | +0.0046 | 5/7, kept |
| + dead | −0.0006 | 3/10, out | +0.0009 | 5/7, kept |
| **the chain** | **0.7033 (+0.1306 vs v11), 10/10** | | **0.6699 (+0.1074), 7/7** | |

`all` by source: IDSSE 0.6674 (+0.100, 7/7), Metrica 0.7870 (+0.201, 3/3), PathCRF-unseen 0.7617
(+0.178, 4/4); macroF1 0.9182 against 0.8705. Both chains beat their learned twins (0.7019,
0.6642). **Robust → short → segments passes on both sets**; the tail (timing on `all`; recall and
dead on `idsse`) adds a few thousandths and depends on the set.

### Round 5 (2026-11-06): two leads from the X-ray, both discarded

[weighted](lab/weighted.md): robust trained possession by possession, and v12's chain without v9's snap
and v11's move, each judged with every stage on its setting for that match.

| against v12 (0.7033; IDSSE set 0.6699) | all 10 | better | IDSSE's 7 | Metrica's 3 | IDSSE set | better | verdict |
|---|---|---|---|---|---|---|---|
| no snap or move | +0.0022 | 6/10 | −0.0024 | +0.0129 | −0.0016 | 3/7 | fails |
| weighted | −0.0067 | 4/10 | +0.0025 | −0.0282 | −0.0064 | 2/7 | fails |
| weighted, no snap or move | −0.0023 | 5/10 | +0.0059 | −0.0213 | −0.0024 | 4/7 | fails |

v12 stays. What's new: changes that favour one source hurt the other, and gains near 0.002 on this bench
are noise (one comparison moved that much from settings alone).

### Round 6 (2026-11-07 to 11-09): ideas from other teams' repos, and the switch price

A survey of the challenge repo's 258 forks (102 pushed to; about five serious public contenders, all with
a learned per-frame model and a Viterbi decoder) pointed at what we hadn't tried.

| idea | where from | result | verdict |
|---|---|---|---|
| **the second half's kick-off** (FAQ: the 2,934 graded frames before it are all D) | several repos; v1-v10 had it from Smart Tagging's kick-off tag, v12 had dropped it | can't be measured on the bench (no tags); on our match the SHA-256 changed, so v12 hadn't called all of them D | **kept in v12**: the FAQ states the answer |
| **label-free checks on our match**: dead share, possessions per minute, agreement at clear on-the-ball frames | animepuika, Barca-Hackathon | `bench/transfer_check.py`; bench ranges 0.314–0.533, 3.17–4.12, 0.991–0.998 | built; Jose runs `--ours` |
| [**motion**](lab/motion.md): ball-free team movement | ZesterJ | chain +0.0031, 5/10, IDSSE −0.0014; IDSSE set −0.0044 | **discarded** |
| **switch**: the team-switch price set frame by frame by a boundary model (`bench/lab_switch.py`) | lead 2 of inside-v12.md | start alone +0.0045, 6/10 (fails the share); every fold at the grid's edge; the IDSSE set and the chain were cut off | **discarded** |
| broadcast cut-aways as dead ball | animepuika | the bench has no broadcast, so it can't be measured | not tried |
| weighting external frames by how much they look like our match; spotting artefacts with a classifier | another public entry | needs our match's tracking in training: against the 2026-11-03 rules | not tried |

**A pattern across rounds 5-6:** dropping snap and move (+0.013), the switch price (+0.011) and motion
(+0.014) each helped Metrica's 3 matches and not IDSSE's 7. Metrica's labels follow the organisers' rules,
and our match is Metrica-tracked, so this may mean something. But 3 matches, chosen after the fact, can't
carry it: weighing the sources differently would have to be decided before a new test, never to rescue
these.

### Where the lab stands

| | `all` held out | vs v11 | `idsse` held out | vs v11 |
|---|---|---|---|---|
| v11 | 0.5727 | | 0.5625 | |
| round 1's mix (stoppage → short → segments) | 0.6156 | +0.043 | 0.6001 | +0.038 |
| round 2 (mix → timing → recall → dead) | 0.6586 | +0.086 | 0.6123 (no dead, recall) | +0.050 |
| round 3 (learned → …) | 0.7019 | +0.129 | 0.6642 | +0.102 |
| **round 4 (robust → …)** | **0.7033** | **+0.131** | **0.6699** | **+0.107** |

All of it is leave-one-match-out on 10 (or 7) bench matches, every stage judged against the chain
without it. Two things flatter it: several candidates' design choices were explored on all 10
matches' truth, and many candidates were tried on the same matches (the winner's curse). Expect
less on our match; the bar's per-match and per-source checks are what make the direction safe.
