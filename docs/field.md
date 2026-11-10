# The field: how we compare with other teams' public repos

| | |
|---|---|
| **as of** | 2026-11-10 (first survey 2026-11-08, in [lab.md](lab.md) round 6) |
| **how** | the challenge repo's 263 forks listed with `gh api`, those pushed since 2026-11-08 and the earlier leaders read: READMEs, technical documents, experiment logs, pull requests and commit messages. **No data, weights or prediction files were downloaded** |
| **answer** | on the external tests that can be compared, **v13 is at or near the top of what's public, not clearly ahead**; on the public 5 minutes we have no number (we don't score on it), and nobody's number there can rank teams |

## Why most numbers can't be compared

The label rule on the external matches moves a score more than the model does:

- one team's model goes from 0.664 to 0.748 when only its labels change (Ryu2804's second model, "jeremy");
- Barca-Hackathon's Metrica score is 0.846 with one-touch recoveries removed from the labels and 0.710 with them;
- Alberstation's IDSSE score swings from 0.495 to 0.680 between the provider's flags and a control rule.

Barca-Hackathon, nuggetofwisdom, Alberstation, ZesterJ and "jeremy" merge short possessions or use a "controlled
for 1.5 s" rule, which makes the test easier. **Our bench scores against the providers' own possession flags
with nothing merged** ([bench.md](bench.md)), the harder version. How hard the simulated damage is moves a score
by about 0.1 more.

## The comparisons that are fair

| test | v13 | public entries on a similar test |
|---|---|---|
| Metrica's 3 sample matches, held out, ~41 % of the ball missing, nothing merged | **0.809** | Barca-Hackathon 0.710 (0.846 with one-touch recoveries removed); nuggetofwisdom 0.815 with 1 s merging; Alberstation 0.691–0.714 |
| IDSSE's 7 matches, provider labels, held out | **0.676** | Ryu2804 "jeremy" 0.664; Ryu2804 main model 0.572 on its IDSSE subset; ZesterJ 0.688 and Alberstation 0.780 on easier merged or control-rule labels |
| SkillCorner's 20 broadcast matches | **0.408**, never trained on SkillCorner's labels ([lab/skillcorner.md](lab/skillcorner.md)) | Ryu2804 0.679 pooled over 30 matches, two-thirds SkillCorner, trained on it and labelled differently |
| the public 5 minutes (17 possessions) | not scored by us | 0.57–0.75: Ryu2804 0.747, Alberstation 0.645, jekkiechan 0.611–0.630, animepuika 0.612, Barca-Hackathon 0.593, ZesterJ ~0.57 |

## The entries

| team | method | data | best held-out claim (protocol) | public window | since 2026-11-08 |
|---|---|---|---|---|---|
| jekkiechan | GRU + gradient-boosted trees, calibrated dead ball, Viterbi, 5-phase averaging | IDSSE, Metrica, SkillCorner (5 dev matches) + SoccerNet-GAR (45 train, 10 test) | 0.594, leave one provider out, damage tied to game state (75 % of the ball missing in dead time, 12 % live); SoccerNet-GAR test 0.702 clean | 0.630, then 0.612 | a 0.594 candidate, not adopted |
| Barca-Hackathon | temporal CNN (±12.6 s), 8-network ensemble, team-swap averaging, Viterbi with a 1 s minimum stoppage | Metrica 3 + IDSSE 7 + SkillCorner 20, damaged at our match's rates | 0.846 leave one out on Metrica's 3 only, one-touch recoveries removed (0.710 with them); IDSSE 0.789 | 0.593 | nothing |
| animepuika | boosted trees for turnovers + a ball-free gap model + a formation dead-ball model + broadcast cut-aways as D | Metrica 3 + IDSSE 7 | submitted version unscored on its bench; a BiGRU reached 0.638 (10 matches, 25 % ball hidden) | 0.612 (the BiGRU 0.568) | the BiGRU rejected |
| ZesterJ | LightGBM, 3 s minimum; a branch adds Viterbi and team-movement features | IDSSE 7 + our match's gap masks | 0.688 leave one out on IDSSE, possessions under 3 s merged; branch 0.694 | ~0.57 | a feature-importance branch |
| 7dracoder | BiGRU over contacts | IDSSE: 4 train, 1 dev | 0.857 on one dev match reused for every choice; 0.732 on 2 untouched matches (older model) | none current | nothing |
| Alberstation | gradient-boosted trees + a per-player set encoder (DeepSets + TCN, ±50 s) + Viterbi | IDSSE 7 + Metrica 3, a moving dead ball simulated | 0.780 / 0.798 on IDSSE (control-rule labels), Metrica 0.691 / 0.714 | 0.645 (older version) | v6 adopted 2026-11-10 |
| nuggetofwisdom | 3 small TCNs + team swap + Viterbi | Metrica 3 + DFL 7 | 0.815 leave one out on Metrica's 3 only (IDSSE always in training), ~41 % ball missing, 1 s merge | not published | v5 on 2026-11-09 |
| Ryu2804 | TCN + per-player set encoder, Viterbi tuned to a target possession rate; a second model (DeepSets + BiGRU) | Metrica 3 + SkillCorner 20 + IDSSE 7 | 0.679 over 30 matches, 6-fold by match (IDSSE subset 0.572); second model 0.664 (provider labels) / 0.748 (control rule) | 0.747 | all new, 2026-11-08 to 11-10 |
| GTT-TalentArena | hand-built evidence + exact Viterbi + a logistic ball-free model | IDSSE 5 | 0.359 / 0.429 on 2 untouched IDSSE matches, 41 % gaps, raw flags | not published | docs only |
| ZainnB | rules + hysteresis, dead-ball classifier, whistle | IDSSE 7 | ~0.355 on IDSSE, raw flags | 0.503 | IDSSE tuning |

Not scored yet or unchanged: nerirulu, Rakeshkumar-dhanarasu, vuleanhduc1604, Venommighty123 (exploration, video
work); Musoye, anisabdali (unchanged since before 2026-11-08).

## What it means for us

- **The real match ranks everyone, not the benches.** Several teams wrote down that bench gains didn't carry
  over: ZesterJ's +0.046 left its public score flat; animepuika's best bench model scored lower on the public
  window and failed its own label-free check (agreement with clear ball control 85.5 % against 90.6 %);
  Barca-Hackathon's 0.846 bench became 0.593. Our answer is [v13](models/v13-temporal.md)'s stoppage switch and
  the label-free checks (`bench/transfer_check.py`).
- **The public window can't separate teams**: 17 possessions, a noise band of about ±0.3, and the leaders'
  0.57–0.75 mostly fit inside it. Jose may score v13 on it as a pipeline check
  (`evaluation/run_evaluation.py`); it never changes v13 ([decisions.md](decisions.md)).
- **Where we may trail: broadcast-style tracking.** Our SkillCorner 0.408 is well under Ryu2804's pooled 0.679,
  but they train on SkillCorner and label it differently, and our match is tactical-camera tracking, which our
  bench resembles more.
- **The rules.** Several entries have problems under the 2026-11-03 clarification (choosing on the
  public window, hand labels or video work on our match, weighting training data by our match's tracking);
  we record the kinds, not the teams. Ours are audited and documented ([limitations.md](limitations.md),
  section 0).

## Ideas they use that we don't

| idea | who | our view |
|---|---|---|
| broadcast cut-aways (no usable broadcast homography: 34 % of broadcast frames) as dead ball | animepuika (on SkillCorner, 91.5 % of cut-away time is dead) | a dead-ball signal that doesn't need the ball, our weak spot; but nothing on our bench can judge it and ~8 % of cut-away time is live. Not before the 2026-11-12 freeze |
| per-player set encoders (DeepSets) | Alberstation, Ryu2804 | a new model family; no time to judge it before the freeze |
| a decoder tuned to a target possession rate | Ryu2804 | the target would come from our match or the organisers' hint; not on external data alone |
| an audit separating our match's tracking from the external matches' (caught an artefact feature) | ZesterJ | uses our match's tracking in an analysis; we treat that as off limits |
| SoccerNet-GAR as a source (55 games) | jekkiechan | licence not checked |
| already ours | | team-swap averaging, ball-free team movement (tried, [lab/motion.md](lab/motion.md), discarded), damage that depends on game state ([bench.md](bench.md)), label-free checks |
