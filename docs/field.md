# The field: how we compare with other teams' public repos

| | |
|---|---|
| **as of** | 2026-11-14 (surveys 2026-11-08 in [lab.md](lab.md) round 6, 2026-11-10, and this one) |
| **how** | the challenge repo's 293 forks listed with `gh api` (263 on 11-10); the 61 pushed since 2026-11-10 read on every branch with recent commits: READMEs, technical documents, experiment logs, commit messages. **No data, weights or prediction files downloaded; no file holding labels or predictions for our match opened** (two teams' files holding labels for our match skipped on purpose) |
| **answer** | on the external tests that can be compared, **we are still at or near the top of what's public, not clearly ahead**. The closest new entry is TEKAB2 (Metrica 0.805, level with our 0.809–0.825). Nobody new reports a SkillCorner number |

## Why most numbers can't be compared

The label rule on the external matches moves a score more than the model does:

- Barca-Hackathon's Metrica score is 0.853 with one-touch recoveries removed from the labels and 0.714–0.721 with them;
- Ryu2804's second model ("jeremy") goes from 0.664 to 0.748 when only its labels change;
- MikolajEkiert's Metrica score is 0.643 with every touch counted and 0.675 with "controlled" possession;
- Alberstation's IDSSE score swung from 0.495 to 0.680 between the provider's flags and a control rule (2026-11-10).

Many entries merge short possessions or use a "controlled for N s" rule, which makes the test easier. **Our bench
scores against the providers' own possession flags with nothing merged** ([bench.md](bench.md)), the harder
version. How hard the simulated damage is moves a score by about 0.1 more.

## The comparisons that are fair

| test | ours | public entries on a similar test |
|---|---|---|
| **Metrica's 3 sample matches, held out, ~41 % of the ball missing, nothing merged** | **v13 0.809; v14 and v15 0.825** (v15 runs branch A there; settings found on these 3 games, held-out estimate ≈0.817); v15's branch B under stoppage stress 0.818 | **TEKAB2 0.805** (other two Metrica games + PFF in training, event labels, best of 4 variants); Barca-Hackathon 0.714–0.721 with one-touch recoveries (0.853 without); nuggetofwisdom 0.815 with 1 s merging (unchanged since 11-09); Alberstation 0.691–0.714 (control rule); MikolajEkiert 0.659 (never trained on Metrica); RayhanSK 0.626 (2 games); Tejeshhh78 0.543; vuleanhduc1604 0.495; yania-n 0.474 |
| IDSSE's 7 matches, provider labels, held out | **0.676** (IDSSE-only set); 0.681 on the `all` set | Ryu2804 "jeremy" 0.664 (their common benchmark 0.598); MikolajEkiert 0.609; Jamoser123 0.567; yelhakeem 0.553; RayhanSK 0.551. On easier labels: TEKAB2 0.763 (opponent spells under 1.6 s handed back), zka2038 0.733 (2 matches, controlled labels), ZesterJ 0.703 (under 3 s merged), Alberstation 0.780 and Barca-Hackathon 0.779 (control rules) |
| SkillCorner's 20 broadcast matches | **0.408–0.409**, never trained on SkillCorner's labels ([lab/skillcorner.md](lab/skillcorner.md)) | Ryu2804 0.679 pooled over 30 matches, trained on SkillCorner and labelled differently; ZesterJ 0.675 on 3 SkillCorner matches (merged labels) |
| other unseen providers | | Alberstation 0.731 on SoccerTrack v2 (8 university matches, control-rule labels); MikolajEkiert 0.773 on 16 PFF matches (PFF's terms forbid benchmarking use for us: limitations G8) |
| the public 5 minutes (17 possessions) | not scored by us | 0.39–0.75; Alberstation 0.707, Ryu2804 up to 0.747, ZesterJ 0.648, jekkiechan 0.624, MikolajEkiert 0.594, Barca-Hackathon 0.593. It can't rank anyone (±0.3 noise band) |

## The entries

Serious entries (an external held-out claim, or a learned model):

| team | method | data | best held-out claim (protocol) | since 2026-11-10 |
|---|---|---|---|---|
| **TEKAB2** | gradient-boosted trees + 5 two-stage temporal CNNs (±10 s, per-player channels, a physics ball fill, heads for possession edges), semi-Markov decoder | Metrica 3 + IDSSE 7; PFF World Cup 2022 (64 matches) for pre-training | **0.775 over 10; Metrica 0.805, DFL 0.763** (5 folds of 2 matches, game-state damage 13 % live / 73 % dead missing; best of 4 variants on the same folds) | new; frozen 11-12, submit planned 11-16 |
| Barca-Hackathon | temporal CNN ensemble of 8, team swap, Viterbi; new: a training-only "clean-owner" head | Metrica 3 + DFL 7 + SkillCorner 20 | Metrica 0.850 fully held out without one-touch recoveries, 0.714 with them; DFL 0.779 (control labels) | v3 re-frozen 11-13 (clean-owner head, second-half kick-off wait as D) |
| Alberstation | trees (0.25) + per-player set encoder (0.75) + Viterbi | IDSSE 7 + Metrica 3 | Metrica 0.691 / 0.714, IDSSE 0.780 / 0.798 (1.5 s control rule); SoccerTrack v2 0.731 | v6 final; v7 (+SkillCorner) tried and reverted; video and audio inputs dropped |
| MikolajEkiert ("culers") | LightGBM + a formation kick-off rule + switch-cost Viterbi | IDSSE 7; Metrica for validation; PFF 16 as a veto set | Metrica 0.659 (never trained on), IDSSE 0.609 provider flags, PFF 0.773 | new, ~90 commits; version 1 frozen |
| zka2038 | 5 BiGRUs over team-level features + exact Viterbi + kick-off rule | IDSSE 7 + Metrica 3 | 0.755 in 4-fold CV on 8 matches; 0.733 on 2 held-out DFL matches (flips under 1 s folded back) | new; a careful rules stance |
| nuggetofwisdom | 3 small TCNs + team swap + Viterbi | Metrica 3 + DFL 7 | 0.815 on Metrica's 3 with 1 s merging | nothing since 11-09 |
| ZesterJ | LightGBM + Viterbi; new temporal, set-piece and ball-free team features; a dead-ball detector | IDSSE 7 (+ SkillCorner, licence pending) | IDSSE 0.703 (under 3 s merged) | integration branch unmerged; final not stated |
| Ryu2804 | TCN + set encoder; "jeremy" DeepSets + BiGRU | Metrica 3 + SkillCorner 20 + IDSSE 7 | IDSSE 0.664 provider / 0.748 control; common benchmark 0.598 | exp_4–7 (mostly failed), a candidate registry |
| jekkiechan | GRU + trees, calibrated dead ball, Viterbi; new: stoppage starts snapped to the ball leaving the pitch ("ball-exit snap") | 5 dev matches + SoccerNet-GAR | SoccerNet-GAR 0.709 (silver labels); Metrica game 3 only 0.715 | the snap adopted; refiners rejected |
| Pompeu-Data-Team | final: a dead-ball HMM + a touch state machine; candidates: a 4-state semi-Markov model decoded for expected spellF1, a temporal CNN | IDSSE + Metrica + SkillCorner (splits) | pooled test 0.459–0.513 (5 s merge) | new, ~70 commits; freeze 11-16 |
| RayhanSK | small neural net over clues + HMM | Metrica 1–2 + IDSSE 7 | Metrica 2 games 0.626, IDSSE 0.551 provider flags | new |
| Tejeshhh78 | controlled evidence + an EM movement mixture + Viterbi | Metrica 3 | Metrica 0.543, IDSSE 0.529 (2 matches) | final 11-11 |
| yelhakeem, Jamoser123, MRH2818, stefrazv, Rushilkr77, vuleanhduc1604, yania-n, GTT-TalentArena, ZainnB | trees/HMM/rules; MRH2818 adds video ball detection; stefrazv uses PathCRF | IDSSE or Metrica | IDSSE 0.42–0.60 raw flags (or 0.595–0.664 on merged labels); Metrica 0.47–0.50 | various; GTT-TalentArena docs only |

About 20 more entries were rule-based, with no external held-out claim; their public-window scores run 0.25–0.58.

## What it means for us

- **On Metrica's convention (the answer key's) we are at the top with TEKAB2.** Their 0.805 trains on the other two
  Metrica games and on PFF and picks the best of 4 variants; our 0.825 picked two settings on the same 3 games
  (held-out estimate ≈0.817). Within the noise of 3 games, level.
- **On DFL's provider flags, nothing merged, nobody is above our 0.676.**
- **Nobody else has a stoppage switch or reports a stressed or SkillCorner test** of the kind behind v15's branch B;
  ZesterJ and jekkiechan found the same weakness (the frozen dead ball as a cue) and dropped the cue instead.
- **The real match ranks everyone, not the benches.** Teams keep writing down bench gains that didn't carry
  (Barca-Hackathon's 0.846 bench and 0.593 public window).

## Rules problems seen in public repos

Under the 2026-11-03 clarification (training and tuning on external data only; no checking against the video),
public repos show these kinds of problem. We record the kinds, not the teams: judging other entries is the
organisers' job, and a reading of someone else's repo can be wrong.

- a model trained on the provided match's video (a ball detector), or the video uploaded to a third-party service;
- the video watched to accept or reject a candidate, or labels drawn on the provided match by hand;
- training on pseudo-labels of the provided match (self-training);
- choosing or fitting on the public window, or keeping a change only if the public window improves;
- damage or candidates chosen from statistics of the team's own predictions on the provided match.

On 11-14 about 12 teams showed one of these. Ours stay audited and documented
([limitations.md](limitations.md), section 0). Our damage uses gap statistics of our match's tracking (no labels),
a grey zone that other entries share.

## Ideas they use that we don't

| idea | who | evidence | our view |
|---|---|---|---|
| **a training-only "clean-owner" head** (which team's player is within 1.5 m of the ball in the clean tracking) | Barca-Hackathon | +0.007 to +0.009 held out, every seed up; Alberstation's similar head failed its screen | a cheap auxiliary target for our GRU; testable on the bench |
| **ball-exit snap**: a predicted stoppage starts when the ball is seen leaving the pitch | jekkiechan | +0.011 clean, +0.005 damaged; up on DFL and Metrica, down on SkillCorner | close to v9's snap (restarts on the seen pass), at the other end of the stoppage; testable |
| decoding for the expected spellF1 (sampling paths) | Pompeu-Data-Team | candidate only | MBR decoding: set aside in round 7 because the frames, not the decoder, limit recall |
| PFF World Cup 2022 for pre-training or testing | TEKAB2, MikolajEkiert, joaopecomonteiro | TEKAB2's headline uses it | PFF's terms forbid benchmarking use (limitations G8) |
| SoccerTrack v2 as an unseen provider | Alberstation | 0.731 frozen | needs a Hugging Face account; a test, not training |
| ball-free set-piece features; dropping the frozen-ball cue | ZesterJ | +0.011; dead recall 0.51 → 0.90 when the dead ball moves | the same problem our switch and branch B answer |
| video ball detection to fill gaps | MRH2818, Alberstation, ZainnB (broadcast ball) | Alberstation dropped it; Barca-Hackathon puts the ceiling at +0.010 | reading the video as an input is allowed, but small gain, much work |
| a competitor survey (Polish) | MikolajEkiert's `research/COMPETITORS.md` | ~20 teams | a lead list |
| already ours | | | team swap, damage tied to game state, label-free checks, a kick-off rule, semi-Markov decoding (tried: [lab/hsmm.md](lab/hsmm.md)), ball-free team movement (tried: [lab/motion.md](lab/motion.md)) |
