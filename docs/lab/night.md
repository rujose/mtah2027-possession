# The night of 2026-11-14 to 11-15: new ideas, tried by agents

Jose, 23:05: "after that is over, have subagents think in other ways to improve and the best ways to implement,
new ideas. after they get to the best or more promessing, commit to those and start creating subagents to achieve
those … try that until 10 am tomorrow".

This page is the night's plan, its rules and its log.

## Morning summary (08:05)

**Nothing passed its bar; v15 stays the upload.** Nine candidates were built and judged on the bench (every bar
fixed before the first score), and round 9 (cleaning) was closed. One is a near miss worth Jose's decision: `netmet5`.

| candidate | what it changes | B's bench vs v15's B | better on | SkillCorner | verdict |
|---|---|---|---|---|---|
| **`netmet5`** | B's network trained with Metrica's 3 games too, 4 seeds averaged | **+0.0139** (Metrica +0.040, DFL +0.0027) | **6/10** | +0.0004 | fails on "7 of 10" only; passes Metrica's rule (as v14 did); audited clean |
| `netmet5`, settings re-picked leave one out | the same, B's settings re-picked among trial 1015 + round 8's top 100 | +0.0118 (DFL −0.0008) | 6/10 | +0.0002 | fails; re-picking costs 0.002 |
| `netmet` (seed 0 / seed 1) | the same, one network | +0.0130 / +0.0069 | 8/10 / 6/10 | −0.0017 | fails the noise rule |
| `netown` | a training-only clean-owner head | +0.0036 | 7/10 | −0.0031 | fails |
| `netens` | 5 seeds of B's network averaged | +0.0005 | 6/10 | −0.0005 | fails (better P, absorbed by the chain) |
| `pstage` | B's own P given to its post stages, `recall` added | +0.0011 | 8/10 | −0.0035 | fails |
| `netsc2` | SkillCorner's real stoppage balls in training, dead/live only | −0.0074 | 4/10 | odd 10: +0.018 | fails the robustness bar |
| `netsc` | SkillCorner's 10 in training | −0.0132 | 2/10 | odd 10: +0.059 | fails the robustness bar |
| `netavg` | B's network averaged with A's | −0.0112 | 2/10 | −0.0164 | fails |
| ghost features (screen) | inputs pointing to missing players near the ball | −0.0031 (trees) | 4/10 | | fails the screen |
| round 9: cleaning | joins, fills, ball, smoothing | −0.0098 (trees) | 3/10 | | nothing kept |

**What the night established**
- **The only lever that moved the bench is what B's network trains on.** Adding Metrica's 3 games gains with every
  seed tried (+0.007 to +0.014; Metrica +0.029 to +0.039 each time, 6 of 6 match-seeds), but DFL's 7 sit within the
  seed noise (−0.0025 to +0.0047), and J03WOH and J03WPY lose with every seed. Each held-out Metrica game is scored
  by a net trained on the other two, which is exactly how our match (Metrica's tracker, Metrica's convention) would be
  scored. The audit ([audit_netmet.md](night/audit_netmet.md)) found no leak.
- **Seed noise is 0.0035 on B's bench**, and single matches move ±0.02 with the seed alone; about 0.003 of v15's
  margin is seed 0's luck. Averaging seeds removes the swings but doesn't raise the score (the chain absorbs a better
  P, as in round 7).
- **Real stoppage balls (SkillCorner) in training help on real balls and cost on the bench**, either way the loss is
  written; the bench, which follows the answer key's convention, keeps the last word.
- **The value in the missing players near the ball can't be built from the tracking** (round 9, ghosts, the
  denoiser); 82 % of it sits where the ball is seen, which points to the broadcast tracking as the next-phase source.
- The isolated rebuild of B reproduces v15's B exactly (all 30 units and SkillCorner), so every comparison above is
  against the real v15.

**For Jose to decide**
1. **`netmet5`: take it further or not.** It misses the bar on "7 of 10" (6), passes Metrica's rule, and carries
   most of its gain on Metrica, the provider our match comes from. Taking it further means: using Metrica's sample
   games (no licence file) to train a deployed network; four networks in the model file; a v16 port, its
   reproduction checks and the transfer check. The switch doesn't change our file (our match takes B either way).
2. If not: v15 stays, and the night's pages stand as the record.

Pages: [network.md](night/network.md), [audit_netmet.md](night/audit_netmet.md), [retune.md](night/retune.md),
[netsc.md](night/netsc.md), [netsc2.md](night/netsc2.md), [pstage.md](night/pstage.md), [ghost.md](night/ghost.md),
[cleaning.md](cleaning.md); the four idea pages in [night/](night/). Nothing committed; nothing read or run on our
match.

## The rules every agent follows

- **Our match (BAR-ATH) is never read**: not its tracking, its labels, the public truth, nor any statistic of our
  output. Nothing is run on it, except what Jose already allows (output that is only paths, SHA-256, row counts or
  lane names), and nothing tonight needs that. The viewer page is never opened.
- **Everything is trained, tuned and judged on external data**: the bench (7 IDSSE + 3 Metrica matches, damaged
  as in `bench/`), SkillCorner's open data for the real-stoppage-ball check. Leave one match out, as the lab does.
- **Nothing is adopted tonight.** `possession/v15.py` and what it reads stay as they are; the upload stays v15
  (SHA `209a2081…7f5e`). A candidate that passes its bar gets a page; Jose decides in the morning. A model change
  needs the page first, then Jose's go-ahead.
- **No commits, no pushes, no messages to anyone.**
- **Shared machine (8 GB, 8 cores):** every heavy job goes through the scratchpad's `slot.py` (4 slots; one heavy
  job = one slot at `OMP_NUM_THREADS=2`). Each agent writes only its own new files (`bench/lab_<name>.py`,
  `docs/lab/<name>.md`, `bench/results/<name>_*`, `data/external/<name>/`), and rebuilds of branch B happen in an
  isolated copy, as `stage3.sh` does, never in the shared caches.
- **Each bar is written before the first score** (the lab's habit, [lab.md](../lab.md)), and every result,
  including what didn't help, goes on the candidate's page.

## The plan

1. Finish round 9 (the cleaning grid): the judge, the verdict, the docs.
2. Ideas: agents look at the work from different sides (the model behind branch B, the inputs, the output side and
   the decoder, what other teams and the literature do) and each proposes its best few, with a test plan and a cost.
3. Pick the most promising ones that fit in the night, write their bars here, and give each to a builder agent.
4. Keep going: as results come in, drop what fails and start the next idea, until about 09:45.
5. Morning summary at the top of this page.

## What was picked (23:35), and the bar, fixed before any score

Three of the four idea pages agree on the lever: branch B is 92 % one network (`gru_b`), trained with one seed on
DFL's 7 matches only. Picked, in this order:

| builder | ideas | page | why |
|---|---|---|---|
| **network** | `netens` (five seeds of `gru_b`, averaged); `netmet` (Metrica's labels in `gru_b`'s training); then, if slots allow, the clean-owner training head | `night/network.md` | proposed by two agents independently; F3 (two GRUs averaged: log loss 0.229 → 0.212, 10/10) and F4-F5 (Metrica's dead/live convention, which the answer key follows, never seen by `gru_b`) in [ideas_network.md](night/ideas_network.md) |
| **ghost** | ghost features: where a missing player's piece ended or starts near the ball, as new inputs (positions untouched) | `night/ghost.md` | [ideas_inputs.md](night/ideas_inputs.md): a ghost within 5 m of the ball names the right team 85-92 %; screened on the trees first, B rebuilt only if the screen passes |
| **output** | `pstage`: B's `timing` stage, and `recall` added as a 4th stage, given B's own start probabilities as features | `night/pstage.md` | [ideas_output.md](night/ideas_output.md): the output side is near saturated (decoder misses are short possessions with no evidence in P), but the stages' candidate sets hold room a better ranker can reach; in-sample screens +0.002 to +0.003. Cheap: cached data, no network rebuild |

**The bar for every candidate tonight** (the round-8 B bar, against v15's B as confirmed, trial 1015's settings
fixed): on B's bench (each match's mean of p25, p50, stressed) **+0.005 or more** (0.7379), **better on 7 of 10**,
**no stress variant down more than 0.002**, **neither source (IDSSE, Metrica) down on the mean**, **SkillCorner down
no more than 0.005** (v15's B 0.4085, so 0.4035). **Noise rule:** if a second seed of the single network moves B's bench by 0.003 or more,
a single-network pass must repeat with a second seed. `netmet` also reports Metrica's 3 and DFL's 7 apart; a pass
on Metrica only is a decision for Jose (as v14 was), and so is using Metrica's sample games in training at all
(they carry no licence file, [limitations.md](../limitations.md)).

## Log

- 23:05 start. The Mac stays awake until ~10:15 (`caffeinate`). A check-in every 30 minutes restarts the work if a
  usage limit stopped it.
- 23:10 four idea agents started (read-only): the network behind branch B, what the models see, the decoder and
  output side, other teams and the literature. Their proposals go to [night/](night/).
- 23:15 round 9 closed: no cleaner passed stage 2 (the judge: −0.0098, 3 of 10), so stage 3 doesn't run
  ([cleaning.md](cleaning.md)); written into lab.md, decisions.md and the README.
- 23:12 a full no-change rebuild of branch B started in an isolated copy (`stage3.sh c0`), to check the rebuild
  reproduces v15's B and to time it.
- 23:20-23:35 three idea pages in (outside, inputs, network); picks and the bar written above.
- 23:40 two builders started: **network** (`netens`, `netmet`, then the clean-owner head) and **ghost**.
- 23:40 the output-side page in (it reproduced B's 0.7329 on all 30 units from cached data; MBR/n-best decoding,
  ball-exit snap, temperature, consensus picks measured or argued not worth building); the **output** builder
  started on `pstage`.
- 00:48 the no-change rebuild of branch B in an isolated copy (`stage3.sh c0`, every stage retrained from the
  tracking: stress, 8 GRU folds, the carried and gate fits, the trial tables) **reproduces v15's B exactly**: all 30
  units of B's bench equal to the reference (largest difference 0.0), SkillCorner 0.4085 (+0.0008, 11 of 20) as
  before. A full rebuild takes 1 h 36 min with the machine shared (the GRU folds are most of it).
- 01:05 **`pstage` fails the bar** ([pstage.md](night/pstage.md)): B's bench **+0.0011**, better on 8 of 10 (p25
  +0.0011, p50 +0.0014, stressed +0.0007; IDSSE +0.0008, Metrica +0.0016), SkillCorner **−0.0035** (inside the
  0.005 allowance). It reproduced trial 1015 exactly first. The probabilities don't help `timing` once it's trained
  the way B deploys (the idea page's +0.0028 was trained on the test variant's own letters); `recall` adds about
  +0.001 to +0.002 with or without them, and loses on SkillCorner the more it writes. Best of all 13 settings, picked
  with hindsight: +0.0019. Not kept. No new builder takes its place for now: the network and ghost builders hold
  all four slots.
- 01:00 **ghost features fail the screen gate** ([ghost.md](night/ghost.md)): `ghost_a` +0.0006 (5/5), `ghost_b`
  −0.0004 (7/3); the judge, leave one match out: **−0.0031, 4 of 10, Metrica −0.0090**. No B rebuild. The features
  do what they were built for (on the 4.7 % of live frames whose on-ball player is missing, log loss −0.004 to
  −0.036), but the trees already had those frames right 86-89 % of the time, and the extra columns cost a little
  elsewhere. The builder goes on with its pre-registered extras (the near-ball denoiser and two diagnostics).
- 01:30 the ghost builder finished. Its extras: the learned near-ball **denoiser fails the same gate** (−0.0005, 6
  of 10, Metrica −0.0043). Two diagnostics (they need the truth): the true near-ball players **only where the ball is
  seen** give +0.0178 (82 % of round 9's +0.0216, 9 of 10); with PathCRF left as delivered +0.0170 (79 %), so most of
  the value flows through robust's own features. The ghost columns reach about a tenth of what the true positions
  give where the on-ball player is missing, and nothing on loose-ball frames (38.5 % of live, 45 % of the gain).
  Round 9's conclusion stands: only a source that sees around the ball can reach it.
- 01:35 a fourth builder started, **netsc**: SkillCorner's even 10 (real moving stoppage balls) in `gru_b`'s
  training ([ideas_outside.md](night/ideas_outside.md), idea 4), judged on **a robustness bar** fixed on that page
  (B's bench no loss beyond 0.002, no variant down more than 0.003, neither source down more than 0.005; SkillCorner's
  untouched odd 10 +0.010 or more, 7 of 10). Accepting that kind of bar is Jose's call. Page: `night/netsc.md`.
- 01:44 **`netavg` fails** ([network.md](night/network.md)): `gru_b` averaged with A's GRU, trial 1015's settings
  fixed: B's bench **−0.0112** (2 of 10; stressed −0.024), SkillCorner −0.0164. A's GRU never saw a moving stoppage
  ball, so it pulls the mean the wrong way exactly where B runs (the idea page's gain was on the normal variant).
- 01:52 **seed noise measured:** `gru_b`'s recipe with seed 1 instead of 0 gives B's bench **−0.0035** (4 of 10;
  single matches move up to ±0.02). The noise rule is on: a single-network pass must repeat with seed 1. About 0.003
  of v15's +0.035 margin is seed 0's luck (with seed 1, the same recipe and settings give +0.032 over B as built).
- 02:20 **`netmet` passes every part of the bar** ([network.md](night/network.md)): `gru_b` trained with Metrica's 3
  games too, trial 1015's settings fixed: B's bench **0.7459, +0.0130, better on 8 of 10** (p25 +0.0108, p50 +0.0120,
  stressed +0.0163); **DFL's 7 +0.0047, Metrica's 3 +0.0324** (3 of 3); SkillCorner −0.0017 (allowed −0.005). The
  margin is 3.7 times the seed noise. The noise rule's second seed (seed 1) is training; the stoppage switch's check
  with `netmet` as B is asked for next (for Jose's decision). Using Metrica's sample games in training is Jose's call.
- 03:03 **the stoppage switch with `netmet` as B** (information, threshold 0.1445 unchanged): parts 1, 2 and 4 pass,
  part 3 fails at 10 % and 25 % (−0.024, −0.027): the same failure v15 had (−0.013, −0.017), for the same reason (B
  improved and the threshold keeps A where B is now better). Our match takes B under the deployed threshold, so the
  switch doesn't change our file. New: `netmet`'s B beats A on the normal bench too (+0.0113, 7 of 10, Metrica's 3
  all up), though on Metrica that comparison favours B (it trained on Metrica's other games, A didn't).
- 03:40 **`netsc` fails the robustness bar** ([netsc.md](night/netsc.md)): B's bench **−0.0132** (2 of 10; IDSSE
  −0.0143, Metrica −0.0107), while SkillCorner's untouched odd 10 gain **+0.0593, 10 of 10** (15 times the seed
  spread there). The network learns real moving stoppage balls (called dead 0.88 → 0.94 on the odd 10), but it also
  learns SkillCorner's timing of possession changes, which doesn't follow a touch: on the bench the team is right
  about 3 points less often within a second of a true change. The bench (Metrica's convention, the answer key's)
  keeps the last word, as the bar says. A narrower recipe follows (below).
- 03:35 **`netmet` doesn't pass under the noise rule:** with seed 1 the same recipe gives +0.0069 (6 of 10; DFL's 7
  −0.0025, Metrica's 3 +0.0288). Within each seed, training on Metrica helps by +0.013 (seed 0) and +0.010 (seed 1);
  Metrica's 3 gain +0.02 to +0.05 a match with both seeds (6 of 6); DFL's 7 swing ±0.02 a match with the seed alone.
  The builder now judges **`netmet5`** (the mean of `netmet`'s seeds 0-4) with the bar, a choice made after seeing
  seed 1 and said so on its page. Also queued: `netens` (five seeds of `gru_b`), then the clean-owner head if time
  allows.
- 03:50 a fifth builder, **netsc2**: netsc's recipe with SkillCorner's frames teaching **dead or live only** (their
  possession timing left out of the loss), the same robustness bar against v15's B, one variant, fixed before any
  score. Page: `night/netsc2.md`. It holds at most 2 slots, so the network builder's runs come first.
- 04:30 **`netens` fails the bar** ([network.md](night/network.md)): five seeds of `gru_b` averaged give B's bench
  **+0.0005** (6 of 10), SkillCorner −0.0005. At frame level the ensemble is clearly better (log loss −3 to −4 %), and
  the decoder and stages absorb it, as in round 7. Its value is robustness (it removes the ±0.02-a-match seed swings:
  +0.004 over a seed-1 network), not a higher score than the seed v15's settings were tuned on. Next on the network
  builder: `netmet`'s seed 2, the clean-owner head (`netown`), `netmet`'s seeds 3-4; `netmet5` takes the seeds
  trained by 07:00.
- 04:59 (information, not the registered candidate) `netmet`'s seeds 0 and 1 averaged: B's bench **+0.0119, 8 of
  10** (p25 +0.0102, p50 +0.0106, stressed +0.0149; DFL's 7 +0.0027, 5 of 7; Metrica's 3 +0.0334, 3 of 3). Every
  bench part of the bar passes; SkillCorner not yet run for it.
- 05:48 **`netsc2` fails the robustness bar** ([netsc2.md](night/netsc2.md)): B's bench **−0.0074** (4 of 10; every
  variant and both sources down beyond the allowance), SkillCorner's untouched odd 10 **+0.0180, 10 of 10**. Leaving
  SkillCorner's team out of the loss gives back 44 % of `netsc`'s bench loss and costs 70 % of its SkillCorner gain.
  So real moving stoppage balls in training help on real balls and cost on the bench either way; the bench (the answer
  key's convention) keeps the last word. Two matches carry most of the loss (J03WPY −0.036, metrica3 −0.025).
  Most of `netsc`'s +0.059 came from learning SkillCorner's own team calls; the stoppage balls are worth about +0.018.
- 06:08 **the clean-owner head (`netown`) fails the bar** ([network.md](night/network.md)): B's bench **+0.0036**,
  7 of 10 (every variant +0.003 to +0.004; Metrica's 3 +0.0091, 3 of 3), SkillCorner −0.0031. Under +0.005 and the
  size of the seed noise. The same direction as Barca-Hackathon's, smaller.
- 06:12 by the clock: `netmet`'s seed 4 can't finish by 07:00, so it's dropped; `netmet5`'s K is seeds 0-3 if seed 3
  is done by 07:00, else 0-2 (the rule fixed at 02:47).
- 06:55 **`netmet5` (`netmet`'s seeds 0-3 averaged; K = 4 by the clock) fails the bar on one part only**
  ([network.md](night/network.md)): B's bench **+0.0139**, but better on **6 of 10** (7 needed); p25 / p50 / stressed
  +0.014 / +0.013 / +0.015; DFL's 7 +0.0027, **Metrica's 3 +0.040** (3 of 3); SkillCorner **+0.0004**. It passes
  Metrica's rule (as v14 did). J03WOH and J03WPY lose about −0.012 with every `netmet` seed; two more are level
  (−0.001). The switch with `netmet5` as B: parts 1, 2, 4 pass, part 3 fails at 10 % and 25 % (as v15's did), and B
  now beats A on the normal bench (+0.0153, 7 of 10). Registered at 03:37, after `netmet`'s seed 1 was seen.
- 07:00 an independent audit of the `netmet` results started (leakage, held-out contamination, fairness of the
  comparison), so that Jose's decision rests on checked numbers. Read-only, one slot at most.
- 07:10 **the audit finds no problem** ([audit_netmet.md](night/audit_netmet.md)): held-out exclusion clean (all 44
  nets' training and normaliser lists leave out their match; all 80 held-out predictions re-predict exactly from the
  net that never saw it), the comparison fair (same trial 1015 settings and code; B as built identical on all 30 + 20
  units; every number on the page recomputes), SkillCorner as v15 does it, no stale cache. How to read the gain:
  about +0.012 of `netmet5`'s +0.0139 is Metrica's 3 (each scored by a net trained on the other two, which is how our
  match would be scored); DFL's 7 +0.0027 and SkillCorner's untouched odd 10 +0.0025 are the guides for a provider
  it never trained on. With v15's own stage models Metrica still gains +0.041, so the second-order path doesn't carry
  it. Seeds 2 and 3 alone: the four seeds run +0.0069 to +0.0141; Metrica +0.029 to +0.039 with every seed.
- 07:20 a last builder, **retune**, with a hard stop at 09:25: `netmet5` was judged on trial 1015's settings, which
  were picked for `gru_b`'s P. It re-picks B's settings for `netmet5` **leave one match out** among a list fixed now
  without `netmet5`'s scores (trial 1015 plus round 8's top 100 trials), and judges the nested scores with the same
  bar. Page: `night/retune.md`.
- 08:00 **retune: re-picking doesn't rescue `netmet5`** ([retune.md](night/retune.md)): nested **+0.0118, 6 of 10**,
  DFL's 7 **−0.0008** (fails "7 of 10" and "neither source down"); 0.002 under the same P with trial 1015 fixed. The
  in-sample best (trial 702: +0.0165, 7 of 10) is the winner's curse of picking among 100 on 10 matches (0.005).
  SkillCorner for the pick (trial 702): +0.0002, 13 of 20. If `netmet5` goes on, keep trial 1015's settings.
- 08:05 morning summary written at the top; the check-ins stopped.
