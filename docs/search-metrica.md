# A nested search of v13's settings on Metrica's convention (2026-11-12)

| | |
|---|---|
| **why** | The answer key follows Metrica's owner convention ([owner-rule.md](owner-rule.md)); every lab verdict so far was decided mostly by DFL's 7 matches. Jose: "brute force" the settings to match Metrica's 3 games, and compare training on 2.5 matches with testing on 45 minutes of one |
| **code** | `bench/search_metrica.py`: `prepare` (stage models per game, ~10 min), `run --worker I/N` (2,000 trials, ~25 min on 4 workers), `report`, `confirm` (the lab's full protocol) |
| **rule, fixed before the confirmation ran** | Metrica's 3 games are the main judge (their labels follow the answer key's convention); DFL's 7 are a guard: the candidate may cost them at most 0.005 on average |
| **answer** | **Two settings change, consistently across all 2,000 trials: the GRU's weight in the start 0.5 → 0.8, and v9's snap and v11's move off.** With every stage refitted the lab's way: **`all` 0.7241 against v13's 0.7162 (+0.0080, 8/10), Metrica +0.0156 (3/3), DFL +0.0047 (5/7), PathCRF-unseen +0.0142 (4/4): it passes the lab's full bar on `all`.** On the IDSSE-only set −0.0008 (3/7): within the guard, not a gain |

## The search

- **2,000 settings drawn at random from a fixed seed before any was scored** (an adaptive search guided by all 3
  games would leak the held-out one). Trial 0 is v13 as built. Each trial sets 13 things: the GRU's weight in the
  start, a factor on P(dead), the decoder's two switch prices and minimum possession, snap and move on/off, and each
  post stage on/off with its thresholds.
- **Each trial is scored on each game with stage models trained without that game** (trained once on v13's
  out-of-fold letters, then applied with the trial's thresholds). A trial takes ~1 s a game; trial 0 reproduces v13
  (0.821 / 0.799 / 0.829 against the lab's 0.815 / 0.795 / 0.816; the gap is v13's deployment settings against the
  lab's per-match picks).

## Holding out a match or a half (Jose's question)

| protocol | held-out score | v13 | gain | units better |
|---|---|---|---|---|
| **match**: pick on the other 2 games' halves, score the held-out game's halves | 0.8220 | 0.8143 | **+0.0077** | 5/6 |
| **half**: pick on the other 5 halves (the sibling half included), score the held-out half | 0.8218 | 0.8143 | +0.0075 | 5/6 |
| the best trial on all 6 halves, in sample | 0.8329 | 0.8143 | +0.019 | |

- **Holding out a half didn't flatter here**: the halves bounce so much within a game (0.72 to 0.86 on game 2) that
  the sibling half's information drowned in that noise. It cost nothing and bought nothing; the match stays the unit
  to hold out, since our test is another match.
- **The winner's curse is real**: the best of 2,000 looked +0.019 in sample, +0.008 held out.

## What the 2,000 trials agree on

| setting | across all trials | v13 | the top 10 % |
|---|---|---|---|
| **the GRU's weight in the start** | rank correlation with each game's score **+0.81, +0.58, +0.75** | 0.5 | 0.78 (median) |
| **snap and move** | on costs **0.004–0.005 on every game** | on | on in 21 % |
| short, segments, timing | each on helps every game (+0.009 to +0.017, +0.005 to +0.007, +0.006 to +0.010) | on | on in 96 %, 82 %, 98 % |
| P(dead) factor | weak (+0.16, +0.18, +0.04) | 1.0 | 1.27 |
| the decoder's prices and minimum possession, the stages' thresholds | weak and inconsistent | | |

Two interpretable changes, not one lucky combination of 13 numbers.

## The confirmation (`python -m bench.search_metrica confirm --w-gru 0.8 --no-snap [--set idsse]`)

v13's chain and the candidate's, every post stage refitted on each start's own letters out of fold, each match on
its fold settings, as `lab_temporal.chains` judged v13 (v13 recomputed: 0.7162 / 0.6760, as recorded).

| match | v13 | candidate | gain | | IDSSE set: v13 | candidate | gain |
|---|---|---|---|---|---|---|---|
| J03WMX | 0.6270 | 0.6286 | +0.002 | | 0.6295 | 0.6314 | +0.002 |
| J03WN1 | 0.6391 | 0.6404 | +0.001 | | 0.6260 | 0.6242 | −0.002 |
| J03WOH | 0.7064 | 0.7011 | −0.005 | | 0.7210 | 0.7041 | −0.017 |
| J03WOY | 0.6886 | 0.6772 | −0.012 | | 0.6785 | 0.6714 | −0.007 |
| J03WPY | 0.7101 | 0.7172 | +0.007 | | 0.7280 | 0.7268 | −0.001 |
| J03WQQ | 0.6690 | 0.6985 | +0.030 | | 0.6643 | 0.6792 | +0.015 |
| J03WR9 | 0.6952 | 0.7051 | +0.010 | | 0.6846 | 0.6892 | +0.005 |
| metrica1 | 0.8147 | 0.8419 | +0.027 | | | | |
| metrica2 | 0.7955 | 0.7987 | +0.003 | | | | |
| metrica3 | 0.8163 | 0.8328 | +0.016 | | | | |
| **mean** | **0.7162** | **0.7241** | **+0.0080 (8/10)** | | **0.6760** | **0.6752** | **−0.0008 (3/7)** |

## Caveats

- **The two settings were spotted on Metrica's 3 games**, so their +0.016 there is partly in sample; the search's
  held-out estimate is +0.008. The DFL matches and the PathCRF-unseen ones took no part in choosing, and they don't
  lose on `all` (+0.005, +0.014).
- **On the IDSSE-only set it's not a gain** (−0.0008): that set stays as v13 if this is adopted.
- **The stoppage switch was judged with v13's branch A**; a changed branch A could move where `augment`'s branch
  starts to win on the partly stressed benches. Its threshold sits far above the normal bench's signals (0.1445
  against 0.02–0.10), so normal matches still get A.
