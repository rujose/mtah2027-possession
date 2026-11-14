# Lab round 8: branch B, the chain our match runs

| | |
|---|---|
| **status** | **judged 2026-11-13. A candidate for v15's branch B: search trial 1015 (`gru_b` weighted 0.92 with `augment`, snap off, post stages trained on stressed copies).** **The B bar: all three parts pass**: B's bench **+0.0353, better on 10/10** (nested, held out: **+0.0334, 10/10**), worst variant +0.0339, SkillCorner +0.0008. **The switch's bar, threshold unchanged: parts 1, 2, 4 pass; part 3 fails at 10 % and 25 %**, because B now nearly ties A on the normal bench (−0.0009) and beats it from 10 % up. Our match takes B under the deployed threshold, so the threshold doesn't change its file. **Adopted 2026-11-14 (Jose, "Yes, adopt v15"): [v15](../models/v15-branch-b.md).** The design below was fixed before anything was scored |
| **what** | three tries at branch B of the stoppage switch ([gate.md](gate.md)): a temporal network trained the way `augment` was (`gru_b`), B's start with it (`mix_b`), and a nested random search of B's settings on the stressed benches, including what B's post stages are trained on |
| **why** | v14's file on our match equals v13's ([../models/v14-search.md](../models/v14-search.md)): the switch sends our match to branch B, so B is the chain that sets our score. B was built as a robustness fix and never tuned: its start has no temporal network, its decoder and post-stage settings are v12's (chosen for branch A's bench), and its post stages were trained on `augment`'s letters on the **normal** bench, though B only runs where the ball looks like the stressed bench's |
| **code** | `bench/lab_branch_b.py` (the network, its P on every bench variant, B's chain from each start), `bench/search_b.py` (the search, its confirmation, SkillCorner), `bench/check_v15_switch.py` (the switch's bar with the new B) |
| **results** | `bench/results/lab_branch_b.json`, `lab_branch_b_chains_<variant>.json` (the starts); `bench/results/search_b/` (`unit_<match>_<variant>.csv`: every trial on every unit; `report.json`; `confirm_*.csv`; `skillcorner_*.csv`); `bench/results/check_v15_switch/report_1015.json` |
| **data** | the bench and lab_gate's variants; SkillCorner open data (MIT licence) for the stress templates and the external check. Nothing reads our match |

## What was fixed before anything was scored

**The network (`gru_b`).** lab_temporal's network, settings and seed ([temporal.md](temporal.md)), trained on each
training match's 97 blanked features as they are **and** under `augment`'s training stress (lab_carried's
`TRAIN_SEED`: stoppage balls from SkillCorner's even-indexed matches; the test variants use the odd-indexed ones). Both
orientations, 5 Hz, 60 s chunks, 20 epochs; the fold's normal normaliser. Folds as lab_temporal's: leave one match out,
Metrica's 3 share the network trained on all 7 DFL matches.

**The starts**, each decoded with robust's fold pick (as B), snapped and moved, then v12's post stages refitted out of
fold on the same variant's letters (`lab_mix.add_stage`, as lab_gate ran B):

| start | P |
|---|---|
| `augment` | B as built |
| `gru_b` | the network alone |
| `mix_b` | the mean of both (as v13's `average`: both through float32, clipped, rounded) |

**B's bench**: lab_gate's variants p25, p50 and stressed (the switch sends 5, 10 and 10 of the 10 matches to B there).
Normal and p10 are reported for the switch's sake.

**The B bar** (for any change to B):

1. on the mean of p25, p50 and stressed per match: **+0.005 or more, better on 7 or more of the 10**;
2. **no one of the three variants loses more than 0.002** on the mean;
3. **SkillCorner** (deployment models, 20 matches): **no loss beyond 0.005** on the mean.

Then the switch's bar ([../lab.md](../lab.md) round 7) is rerun with the new B, all four parts, threshold unchanged.

**The search** (`bench/search_b.py`, as [../search-metrica.md](../search-metrica.md)):

- 1,500 trials drawn at random from a fixed seed before any is scored; trial 0 is B as deployed.
- What a trial sets: the GRU's weight in the start (`gru_b` against `augment`; 0 is B), a factor on P(dead), the
  decoder's switch prices and minimum possession, v9's snap and v11's move on/off, each post stage on/off with its
  thresholds, and **the stage models' training data**: `normal` (the other 9 matches as they are: what B deploys),
  `stressed` (the other 9 under `augment`'s **training** stress, never the test variants' templates) or `both`.
- A unit is one match on one of B's 3 variants; a match's score is the mean of its 3.
- Each match's stage models are trained without it; the letters they train on are a reference start's out-of-fold
  chain on that data. The reference start: `mix_b` if it beats `augment` on B's bench in the chains above, else
  `augment`.
- **Nested:** for each match, the trial with the best mean on the other 9 is picked; only its score on the held-out
  match counts. (The other matches' stage models saw the held-out match: a second-order leak, accepted as in
  search_metrica.)
- A setting the search finds is then confirmed the lab's way (every stage refitted out of fold, on each variant) and
  on SkillCorner with deployment models, against the B bar.

## The starts (`python -m bench.lab_branch_b chains VARIANT`, then `report`)

B as built reproduces lab_gate's B on every variant (0.6938 / 0.6921 / 0.6958 / 0.6983 / 0.7057). B's chain, every
post stage refitted out of fold on the same variant:

| bench variant | `augment` (B) | `gru_b` | `mix_b` | v14's A, for scale |
|---|---|---|---|---|
| normal | 0.6938 | 0.7169 | 0.7184 | 0.7241 |
| 10 % stressed | 0.6921 | 0.7203 | 0.7197 | 0.7124 |
| **25 %** | 0.6958 | **0.7230** | 0.7225 | 0.6979 |
| **50 %** | 0.6983 | **0.7264** | 0.7255 | 0.6699 |
| **stressed** | 0.7057 | **0.7304** | 0.7301 | 0.6107 |

- **The B bar, parts 1–2: both pass by a wide margin.** `gru_b` +0.0267 on B's bench, better on 9/10 (each variant
  +0.025 to +0.028); `mix_b` +0.0261, 9/10 (+0.024 to +0.027). The one match that loses is J03WOH (−0.013 / −0.004).
  Metrica's 3 gain the most (`mix_b` +0.040, +0.035, +0.056).
- **Part 3 (SkillCorner, deployment models, `search_b skillcorner --trials -1 -2`): both pass.** B as built
  reproduces v13's branch B there exactly (0.4077); `mix_b` 0.4146 (+0.0069, better on 13/20), `gru_b` 0.4063
  (−0.0014, 9/20).
- The frame check on one match (J03WMX, stressed): log loss `gru_b` 0.151, `augment` 0.179, A's GRU 0.608.
- **B is now close to A on the normal bench** (`mix_b` 0.718 against v14's A 0.724) and beats it from 10 % up. So the
  switch's threshold, set where the old B overtook A, now sits late: see "The switch" below.
- `mix_b` beat `augment` on B's bench, so the search's reference start (the letters its stage models train on) is
  `mix_b`.

## The search (`python -m bench.search_b run --worker I/7`, then `report`)

All 1,500 trials scored on all 30 units (10 matches × B's 3 variants; ~2 h on 7 workers). Trial 0 here (B's
settings on the search's shared stage models) is 0.6918.

| held-out match | picked on the other 9 | its score | trial 0 | gain |
|---|---|---|---|---|
| J03WMX | 1015 | 0.6229 | 0.5863 | +0.037 |
| J03WN1 | 554 | 0.6505 | 0.6129 | +0.038 |
| J03WOH | 1269 | 0.7155 | 0.6994 | +0.016 |
| J03WOY | 1015 | 0.6851 | 0.6830 | +0.002 |
| J03WPY | 1133 | 0.7505 | 0.7084 | +0.042 |
| J03WQQ | 1269 | 0.6877 | 0.6545 | +0.033 |
| J03WR9 | 586 | 0.6946 | 0.6808 | +0.014 |
| metrica1 | 1133 | 0.8284 | 0.7758 | +0.053 |
| metrica2 | 1133 | 0.7922 | 0.7710 | +0.021 |
| metrica3 | 1015 | 0.8243 | 0.7459 | +0.078 |
| **mean** | | **0.7252** | **0.6918** | **+0.0334, 10/10** |

The best trial on all 10, in sample: **1015 at 0.7326** (the winner's curse inside the search: ~0.007).

**What the trials agree on** (all 1,500, and the top 100 by the mean of the 10):

| setting | B as built | across the trials | trial 1015 |
|---|---|---|---|
| **`gru_b`'s weight in the start** | 0 | **rank correlation with the score +0.85**; top-100 median 0.81 | 0.915 |
| **v9's snap and v11's move** | on | **off in 79 of the top 100** (48 % of all) | off |
| short, timing | on | on in 100 and 94 of the top 100 | on (p_min 0.340, 0.450) |
| segments | on | on in 75 of the top 100 | on (flip 0.910, merge 0.661) |
| the stage models' data | normal | top 100: stressed 42, both 34, normal 24 (a third each overall) | stressed |
| P(dead) factor, switch prices, minimum possession | 1, 4.0 / 25, 10 | weak (rank correlations −0.07 to +0.03) | 0.909, 2.82 / 48.7, 7 |

The top is flat: the best 10 trials sit within 0.002 of each other, all with the weight 0.76–0.98 and snap off; the
124 trials with the weight above 0.7, snap off and short and timing on have a median of 0.7278 (all trials: 0.7159).
The gain is a region of settings, not one lucky draw.

## The confirmation (`search_b confirm --trials …`, `search_b skillcorner --trials …`)

Each trial as it would deploy: its own settings for every match, its stage models trained on its own chain's
out-of-fold letters on its own data, without the held-out match. Trial 0 here is v13's B as it deploys (0.6975 on
B's bench; lab_gate's B, 0.6999, decoded each match with its fold's own pick and refitted every stage on the same
variant's letters, a protocol a deployed program can't follow). `-1` and `-2` are `mix_b` and `gru_b` with all of
B's other settings.

| match | B (trial 0) | `mix_b` | `gru_b` | **trial 1015** | gain |
|---|---|---|---|---|---|
| J03WMX | 0.5901 | 0.6110 | 0.6200 | 0.6254 | +0.035 |
| J03WN1 | 0.6287 | 0.6471 | 0.6554 | 0.6691 | +0.041 |
| J03WOH | 0.7097 | 0.7153 | 0.7068 | 0.7199 | +0.010 |
| J03WOY | 0.6855 | 0.7057 | 0.6977 | 0.6942 | +0.009 |
| J03WPY | 0.7154 | 0.7421 | 0.7436 | 0.7589 | +0.044 |
| J03WQQ | 0.6564 | 0.6884 | 0.6791 | 0.6939 | +0.038 |
| J03WR9 | 0.6804 | 0.7007 | 0.7138 | 0.7128 | +0.032 |
| metrica1 | 0.7849 | 0.8261 | 0.8321 | 0.8346 | +0.050 |
| metrica2 | 0.7735 | 0.8033 | 0.8009 | 0.7965 | +0.023 |
| metrica3 | 0.7507 | 0.8179 | 0.8220 | 0.8233 | +0.073 |
| **B's bench** | **0.6975** | 0.7258 | 0.7271 | **0.7329** | **+0.0353, 10/10** |
| p25 / p50 / stressed | 0.6949 / 0.6958 / 0.7019 | | | 0.7289 / 0.7319 / 0.7379 | +0.034 / +0.036 / +0.036 |
| **SkillCorner** (deployment models, 20) | **0.4077** | 0.4146 | 0.4063 | **0.4085** | **+0.0008, 11/20** |

**The B bar, every nested pick:**

| trial | B's bench | better on | worst variant | SkillCorner | verdict |
|---|---|---|---|---|---|
| **1015** (the best on all 10) | **+0.0353** | **10/10** | +0.0339 | +0.0008 (11/20) | **passes** |
| 1133 | +0.0339 | 10/10 | +0.0320 | +0.0035 (12/20) | passes |
| 1269 | +0.0337 | 10/10 | +0.0325 | +0.0031 (11/20) | passes |
| 586 | +0.0316 | 10/10 | +0.0298 | +0.0071 (13/20) | passes |
| 554 | +0.0315 | 10/10 | +0.0305 | +0.0097 (15/20) | passes |
| `gru_b`, B's settings | +0.0296 | 9/10 | +0.0290 | −0.0014 (9/20) | passes |
| `mix_b`, B's settings | +0.0282 | 10/10 | +0.0275 | +0.0069 (13/20) | passes |

- **Most of the gain is the network** (+0.028 to +0.030 with B's old settings); the search's settings add +0.006.
- **Trial 1015 is what the search's rule picks** (the best on all 10; the rule whose held-out estimate is the
  nested +0.033). SkillCorner is a guard, not a judge: picking 554 for its SkillCorner score would make SkillCorner
  part of the tuning. All picks lie within 0.01 of each other there.

## The switch (`python -m bench.check_v15_switch --trial 1015`)

Branch A is v14's (every post stage refitted the lab's way, as `check_v14_switch` scored it); B is trial 1015 as it
deploys (above, plus `search_b confirm --trials 1015 -1 --variants normal p10`); the threshold is unchanged (0.1445).

| bench variant | v14's A | v13's B | **v15's B** | B better than A on | the switch picks B on | switch | switch − the better branch |
|---|---|---|---|---|---|---|---|
| normal | 0.7241 | 0.6894 | **0.7233** | 4/10 | 0 | 0.7241 | ±0 |
| 10 % stressed | 0.7124 | 0.6921 | 0.7255 | 9/10 | 0 | 0.7124 | **−0.0131** |
| 25 % | 0.6979 | 0.6949 | 0.7289 | 10/10 | 5 | 0.7115 | **−0.0174** |
| 50 % | 0.6699 | 0.6958 | 0.7319 | 10/10 | 10 | 0.7319 | ±0 |
| stressed | 0.6107 | 0.7019 | 0.7379 | 10/10 | 10 | 0.7379 | ±0 |
| SkillCorner (20) | 0.3196 | 0.4077 | 0.4085 | 20/20 | 20 | 0.4085 | ±0 |

1. **Part 1 (normal: A everywhere, or no loss beyond 0.002): passes** (A on 10/10).
2. **Part 2 (stressed: B on 9+, +0.02): passes**, B on 10/10, +0.127.
3. **Part 3 (10 / 25 / 50 %: within 0.005 of the better branch): fails at 10 % (−0.013) and 25 % (−0.017)**,
   passes at 50 %. Not because B lost anywhere: B got better, and the threshold, placed where the old B overtook A
   (between 25 % and 50 %), now keeps A on matches where the new B is the better branch.
4. **Part 4 (SkillCorner: +0.02, 15/20): passes**, B on 20/20, +0.089, better on 20/20.

**The alternatives** (information; not adopted):

| threshold | normal: switch − A | 10 % | 25 % | parts failing |
|---|---|---|---|---|
| **0.1445 (deployed, kept)** | ±0 | −0.013 | −0.017 | 3 (10 %, 25 %) |
| 0.0485 (round 7's rule, rerun with the new B; left out: −∞ to 0.065) | −0.0034 (Metrica's 3 go to B, which loses on all 3: −0.014, −0.001, −0.019) | ±0 | ±0 | 1 |
| none: B everywhere | −0.0009 | ±0 | ±0 | none |

- **None of these changes our match's file.** v14's file on our match equals v13's, which means the switch took B
  there (v14 changed only A); with the signal above 0.1445, any lower threshold also gives B. So the choice is about
  what the program does on other matches, not about our submission.
- B everywhere passes on paper, but by 0.0009 against a 0.002 allowance, and B loses on all 3 of Metrica's normal
  games, the convention our labels follow. Keeping the threshold fixed before scoring is the choice that needs no
  new judgement.

## Verdict

**Trial 1015 passes the B bar** (+0.0353 on B's bench, 10/10, every variant +0.034 or more; SkillCorner +0.0008)
and the switch keeps 3 of its 4 parts with the threshold unchanged, part 3 failing because B improved. **A candidate
for v15's branch B; adopted on Jose's go-ahead (2026-11-14): [v15](../models/v15-branch-b.md).**

What to keep in mind:

- **The gain is measured on a synthetic ball.** B's bench is the 10 matches with SkillCorner's stoppage balls put
  into their true stoppages; on SkillCorner's own 20 broadcast matches (real data of the kind B runs on, another
  labelling convention: every branch scores ~0.41 there) the new B gains +0.0008, 11/20, about nothing. What
  carries the case is Metrica's 3 games, our labels' convention, gaining +0.023 to +0.073.
- **The honest size is the nested +0.033**; trial 1015's +0.035 is in sample (it was picked on these 10).
- **The nested picks leak a little**: the other matches' stage models saw the held-out match (as in search_metrica).
- **J03WOH and J03WOY gain least** (+0.010, +0.009); no match loses.

## The build (`python -m bench.train_v15`, `python -m bench.check_v15`)

`possession/models/v15_b_all.joblib` (1.9 MB) was built and checked on external data before the decision; v15's run
on our match came after Jose's go-ahead ([v15](../models/v15-branch-b.md)).

- **The network**: `gru_b` retrained on the 7 DFL matches is **bit-identical** to the lab's fold `net_metrica1.pt`.
- **The port** (`check_v15 port`: v15's code with the lab's fold models on 3 bench units, against `search_b`): in
  float32 the same P and **0 frames** differing on all 3; in float64 (as deployed) P differs by one rounding step and
  1 frame differs on J03WR9, same scores (as v13's and v14's ports).
- **SkillCorner** (`check_v15 skillcorner`: the built file on the 20 matches): **0.4085**, `search_b`'s scores
  reproduced to 5e-6 on every match; the switch picks B on 20/20.
