# Tracking errors: how much would wrong teams and misplaced players cost v15's branch B?

| | |
|---|---|
| **status** | error model fixed 2026-11-15, 08:40 (machine clock), before any score; **finished 09:51.** A sensitivity measurement, not a candidate: nothing is adopted, no model changes |
| **decision** | Jose (2026-11-15), from what he has seen on our match: **rare, a few seconds here and there.** At the measured cost (about −0.0005 per half-minute of one player in the wrong colour) that is under 0.005, so nothing is built before Nov 17; wrong teams near the ball stay a first priority for the next phase, beside the missing players near the ball ([cleaning.md](cleaning.md)) |
| **in one line** | **every setting costs B on all 10 matches**: T1 −0.018, T3 −0.041, T10 −0.138, K10 −0.013 (−0.016 without v8's repair, which undoes a sixth of the keeper's wrong frames), G3 −0.014, against v15's B at 0.7329 (the harness reproduces it to 5e-6). The loss sits where the wrongly coloured player is near the ball. Whether to fix it before Nov 17 depends on how often it happens on our match (not measured here) |
| **question** | Jose, watching our match's video with the delivered tracking, sees players drawn in the wrong team's colour (the goalkeeper too) and players placed absurdly. The bench's damage (`bench/degrade.py`) cuts players into anonymous pieces and blanks the ball, but every piece keeps its true team. **If such errors happened at realistic-looking rates, how much would v15's branch B lose?** If it barely moves, there is nothing to fix before Nov 17; if it drops a lot, it is a priority for the next phase |
| **code** | `bench/lab_trackerr.py` (new; no existing file edited). Runs in isolated copies of the repo under the scratchpad, with b16_c0's trained models (the verified rebuild of v15's B) read-only |
| **results** | `bench/results/lab_trackerr/` |
| **nothing here reads our match** | the 10 bench matches only; the error rates are fixed guesses, not measured on our match |

## The error model (fixed before any score; as given)

Errors are injected into the bench's **damaged tracking as the models see it** (the lab cache's players/team:
anonymous slots cut into pieces; a "piece" = a maximal run of frames in which one slot holds a position), per match
with a fixed seed (e.g. derived from the match id), before everything downstream reads it. You may use the clean
tracking/truth to decide *what* to corrupt (e.g. which pieces are goalkeepers — bench/lab_clean.py's
`person_of_slots` maps pieces to true persons); the models never see it.

- **T1, T3, T10 (wrong team, outfield):** random outfield pieces get their team flipped (home↔away) for the whole
  piece, until the flipped share of outfield player-frames reaches 1 %, 3 %, 10 %.
- **K10 (keeper in the wrong team):** for each goalkeeper, random pieces of his get their team flipped until ~10 % of
  his frames are flipped. (This is what Jose saw. If the deployed chain runs the keeper repair, possession/repair.py,
  v8, before the models — check how the bench pipeline builds B's inputs — it runs here too, as it would on our
  match; report how much it undoes.)
- **G3 (ghost positions, only if time allows):** random outfield pieces displaced by a constant offset of 10-20 m in
  a random direction (clipped to the pitch), until 3 % of outfield player-frames are affected.

Priority if time runs short: T3 and K10 first, then T1 and T10, then G3.

### How it is implemented (fixed with the model, before any score)

- **Pieces**: `lab_clean.pieces` on the lab cache's damaged players (per half, per slot, every unbroken run).
  Each piece's true person: `lab_clean.person_of_slots` against the clean tracking (`sets.load`).
- **Goalkeepers**: per half and team, the person seen on at least half the half's frames whose mean distance to his
  own goal is smallest (own goal: the end his team stands nearer to on average, as repair.py decides the defending
  team). Checked by his share of frames inside his own penalty area (reported). Every other person is outfield.
- **Seed**: `np.random.default_rng([20261115, crc32(match), crc32(setting)])`.
- **Picking pieces**: the eligible pieces (outfield for T and G, one keeper's for K) in a random order; taken one by
  one until the affected frames reach the target share of the eligible frames (outfield player-frames for T and G,
  each keeper's own frames for K). The share reached is reported. *Changed 08:48, before any corrupted score (a dry
  run of the errors alone, no model run): taking pieces until the target is reached overshot by a whole piece, up
  to 17 % for a keeper whose pieces are long; now a piece that would cross the target is taken only if it brings
  the total nearer to it, and otherwise the next pieces in the order are tried.*
- **Flips keep a piece on its own**: the receiving slot must also be empty on the frame just before and just after
  the piece, so the flipped piece stays one piece (not glued to another person's run).
- **A flip**: the piece's positions leave its slot and go to a slot of the other team that is empty on all of the
  piece's frames, the one free longest (`degrade`'s rule for dealing pieces); a new slot of that team is appended if
  none is free. Nothing else about the piece changes.
- **A ghost**: one offset per piece, length uniform in 10-20 m, direction uniform; positions clipped to the pitch.
- **Then, as the deployed chain** (`possession/v12.inputs` runs v8's keeper repair first, then everything reads the
  repaired tracking): v8's repair on the corrupted tracking (with the lab cache's ball), then every input computed
  from the tracking is recomputed on it, out of fold, as `bench/eval_pathcrf.py` and `bench/lab.py` built them: v4's
  P(dead), v6's restart P, cost and letters, PathCRF's P (on v6's live frames), v7's cost, v9's and v11's letters
  (the fold dead-ball and restart models refitted on the other 9 matches' **clean** rows, as eval_pathcrf fits them);
  then lab_gate's variants (p25, p50, stressed: the same stoppage balls, v6's and v7's costs recomputed on them),
  robust's 97 features, `augment`'s fold trees and `gru_b`'s fold nets (b16_c0's, trained on clean data), trial
  1015's decoder, and its post stages (models from b16_c0's `tables_trial1015.joblib`, trained on the clean
  training-stress copies) reading the corrupted match. Nothing is retrained or refitted on corrupted data.
- **Scored** on B's bench: 10 matches × (p25, p50, stressed); the mean is "B's bench" (v15's B: 0.7329). Reported
  per match, per variant and per source (IDSSE's 7, Metrica's 3), against the same harness with no corruption.
- **A diagnostic, `K10nr`** (added 08:55, before any corrupted score): K10's very errors (same seed) with v8's repair
  switched off, to put a score on what the repair saves. Not part of the error model.
- **Diagnostics `T1b`, `T3b`** (added 09:28, after T1 and T3 were scored): the same rates with another draw of pieces
  (the setting's name seeds it), for how much a drop depends on which pieces happen to be hit. Not part of the model.
- **Where the points go**: frames where a corrupted piece is within 15 m of the true ball ("near", `o_near`'s
  radius), frames with a corrupted piece but none that near ("far"), frames with none; frame accuracy of the
  letters against the uncorrupted run in each.

## The reproduction check (08:55): passes

The harness with no corruption (`none`: the same copy, v8's repair run, every input recomputed from the tracking):

| check | units | max \|difference\| | |
|---|---|---|---|
| the recomputed inputs (v4's P(dead), v6's restart P and costs, v4 / v6 / v7 letters, PathCRF's P and clue, v7's cost, v9 / v11 letters, players and teams) against b16_c0's lab cache | 10 matches | **0** (bit-identical) | passes |
| `gru_b`'s P on p25, p50, stressed (J03WR9, checked file by file) | 3 | **0** | passes |
| trial 1015's spellF1 against round 8's confirm CSV (`confirm_1015_554_586_1133_1269_-1_-2.csv`, written to 5 decimals) | **30** | **4.8e-6** | passes (≤ 1e-5) |

B's bench in the harness: **0.7329**, as v15. The keeper repair fires on no bench match without errors (as
`bench/keeper_repair.py` found), so running it changes nothing there. Nothing was rebuilt from the copy's data:
its `lab_robust`, `lab_dead` stayed empty and `search_b` held only b16_c0's two table files.

## Results (09:33; second draws and K10nr 09:51)

For scale: v15's B gained **+0.0353** on B's bench over v13's B (round 8); a second seed of `gru_b` alone moves B's
bench by 0.0035 (the night's noise yardstick); the B bar asks +0.005 of a change.

### Every setting, on B's bench (10 matches × p25, p50, stressed)

| setting | share corrupted | B's bench | drop | better / worse | p25 / p50 / stressed | IDSSE's 7 / Metrica's 3 |
|---|---|---|---|---|---|---|
| none (v15's B) | 0 | 0.7329 | | | | |
| **T1** (1 % of outfield frames in the wrong team) | 1.0 % | 0.7146 | **−0.0183** | 0 / 10 | −0.0184 / −0.0179 / −0.0185 | −0.0145 / −0.0272 |
| **T3** | 3.0 % | 0.6921 | **−0.0408** | 0 / 10 | −0.0393 / −0.0406 / −0.0424 | −0.0286 / −0.0691 |
| **T10** | 10.1 % | 0.5946 | **−0.1382** | 0 / 10 | −0.1363 / −0.1377 / −0.1407 | −0.1126 / −0.1981 |
| **K10** (each keeper ~10 % of his frames in the wrong team; v8's repair on) | 11.1 % of the keepers' frames | 0.7199 | **−0.0129** | 0 / 10 | −0.0138 / −0.0128 / −0.0122 | −0.0124 / −0.0142 |
| **G3** (3 % of outfield frames moved 10-20 m) | 3.0 % | 0.7188 | **−0.0140** | 0 / 10 | −0.0146 / −0.0140 / −0.0136 | −0.0161 / −0.0092 |

**Every setting loses on every one of the 10 matches**, on every variant alike. The loss grows with the rate, a
little less than in proportion (1 % → −0.018, 3 % → −0.041, 10 % → −0.138). **1 % of the outfield frames in the
wrong team costs half of v15's whole gain; 3 % costs more than all of it.** Metrica's 3 (our labels' convention)
lose about twice what IDSSE's lose on wrong teams. Per frame corrupted, a wrong team costs about three times a
10-20 m misplacement (T3 −0.041 against G3 −0.014).

Per match (B's bench, the mean of the 3 variants; drop against `none`):

| match | none | T1 | T3 | T10 | K10 | G3 |
|---|---|---|---|---|---|---|
| J03WMX | 0.6254 | −0.0100 | −0.0112 | −0.0862 | −0.0048 | −0.0273 |
| J03WN1 | 0.6691 | −0.0013 | −0.0234 | −0.1154 | −0.0101 | −0.0120 |
| J03WOH | 0.7199 | −0.0078 | −0.0270 | −0.1026 | −0.0171 | −0.0182 |
| J03WOY | 0.6942 | −0.0132 | −0.0224 | −0.0762 | −0.0204 | −0.0096 |
| J03WPY | 0.7589 | −0.0233 | −0.0573 | −0.1581 | −0.0105 | −0.0125 |
| J03WQQ | 0.6939 | −0.0178 | −0.0450 | −0.1152 | −0.0175 | −0.0133 |
| J03WR9 | 0.7128 | −0.0278 | −0.0141 | −0.1343 | −0.0064 | −0.0198 |
| metrica1 | 0.8346 | −0.0367 | −0.0850 | −0.1787 | −0.0160 | −0.0124 |
| metrica2 | 0.7965 | −0.0279 | −0.0572 | −0.2406 | −0.0206 | −0.0058 |
| metrica3 | 0.8233 | −0.0171 | −0.0649 | −0.1749 | −0.0060 | −0.0096 |

A single match's drop depends on which pieces the draw hits (J03WR9 loses more at 1 % than at 3 %); the mean over
10 does not move much with the draw (the second draws below).

### Where the points go

Frames graded by the truth, pooled over the 30 units (each match counted once per variant), split by whether a
corrupted piece is on the frame and whether one is within 15 m of the true ball; the letters' frame accuracy with
and without the errors, and the log loss of the start's P (`gru_b` 0.915 with `augment`, the P the decoder reads):

| setting | zone | share of frames | accuracy none → setting | change | letters changed | right→wrong / wrong→right | start's log loss none → setting | share of the net frames lost |
|---|---|---|---|---|---|---|---|---|
| T1 | **near** | 4.7 % | 0.915 → 0.853 | **−0.062** | 10.3 % | 16,166 / 3,805 | 0.249 → 0.325 | **80 %** |
| T1 | far | 13.0 % | 0.932 → 0.929 | −0.004 | 1.4 % | 4,700 / 2,738 | 0.237 → 0.240 | 13 % |
| T1 | none | 82.3 % | 0.933 → 0.933 | −0.000 | 0.05 % | 1,480 / 361 | 0.209 → 0.209 | 7 % |
| T3 | **near** | 12.9 % | 0.919 → 0.860 | **−0.059** | 9.0 % | 40,815 / 7,945 | 0.221 → 0.317 | **92 %** |
| T3 | far | 31.7 % | 0.950 → 0.948 | −0.002 | 1.5 % | 11,026 / 8,354 | 0.160 → 0.166 | 7 % |
| T3 | none | 55.4 % | 0.926 → 0.925 | −0.000 | 0.1 % | 1,391 / 1,104 | 0.244 → 0.244 | 1 % |
| T10 | **near** | 33.6 % | 0.929 → 0.842 | **−0.087** | 11.8 % | 146,589 / 20,919 | 0.203 → 0.344 | **89 %** |
| T10 | far | 54.0 % | 0.937 → 0.931 | −0.007 | 2.1 % | 30,457 / 15,250 | 0.209 → 0.225 | 11 % |
| T10 | none | 12.4 % | 0.921 → 0.920 | −0.000 | 0.3 % | 853 / 686 | 0.272 → 0.274 | 0 % |
| K10 | **near** | 2.4 % | 0.959 → 0.874 | **−0.085** | 10.3 % | 9,293 / 623 | 0.157 → 0.313 | **53 %** |
| K10 | **far** | 16.6 % | 0.930 → 0.919 | **−0.011** | 2.5 % | 12,234 / 4,566 | 0.222 → 0.242 | **47 %** |
| K10 | none | 81.1 % | 0.932 → 0.932 | −0.000 | 0.03 % | 627 / 477 | 0.215 → 0.215 | 1 % |
| G3 | **near** | 12.8 % | 0.925 → 0.910 | **−0.015** | 3.8 % | 14,288 / 5,932 | 0.208 → 0.240 | **75 %** |
| G3 | far | 32.8 % | 0.943 → 0.941 | −0.002 | 1.2 % | 8,823 / 6,134 | 0.192 → 0.198 | 24 % |
| G3 | none | 54.4 % | 0.928 → 0.928 | −0.000 | 0.1 % | 890 / 763 | 0.230 → 0.230 | 1 % |

- **A wrong-team outfield piece hurts where it is near the ball, almost nowhere else.** Within 15 m of the ball the
  letters lose 6-9 points of frame accuracy and one frame in ten changes letter; away from the ball 0.2-0.7 points;
  on frames with no corrupted piece nothing moves. 80-92 % of the frames lost are near-ball frames, which hold only
  5-34 % of the frames. That is the same place the cleaning round found nine tenths of the missing players' value
  ([cleaning.md](cleaning.md): `o_near`): **B reads the team of the players around the ball, and believes it.**
- **The keeper is different: he hurts far from the ball too.** Half of K10's lost frames are frames where the
  wrongly coloured keeper is more than 15 m from the ball (accuracy −1.1 points there, three times T1's far-ball
  cost), and the start's P is worse there (log loss +0.019). Likely (not tested here) because a keeper in the wrong
  team changes where a team appears to stand (its deepest player, its block; PathCRF takes each team's keeper to be
  its player furthest from halfway), not only who is near the ball.
- **The spells**: B's letters gain spells (H or A runs) with the errors: T1 +2.7 %, T3 +6.6 %, T10 +18.3 %, K10
  +1.5 %, G3 +0.9 % (none: 10,521 over the 30 units; the truth 12,342). About a quarter of the new ones are under 2 s (T1:
  +73 of +282); the rest are longer stretches handed to the wrong team, not flicker.

### What v8's keeper repair undid in K10

v12.inputs runs v8's keeper repair before every model, so it ran here, on the corrupted tracking.

| | pieces | keeper frames |
|---|---|---|
| keeper pieces put in the wrong team (10 matches) | 157 | 296,221 (11.1 % of the keepers' frames) |
| the rule looked at (5 s in a row in his 6-yard box, the ball more than 35 m away, no defender there) | 21 | |
| **repaired** (also ≥ 90 % of the piece inside the penalty area) | **12** | **46,903 (16 %)** |
| the same rule with the stressed variant's ball (more of it seen in stoppages; diagnostic) | 15 | 48,102 (16 %) |
| repairs of pieces that were not flipped (false repairs) | 0 | |

**The repair undoes about a sixth of the keeper's wrong frames.** It fires only where a keeper stands in his own
6-yard box for 5 s with the ball far away; the flipped pieces are short (median 20 s, a quarter of them 5 s or less) and
keepers spend much of a match outside the 6-yard box, so 136 of the 157 never meet that condition, and 9 of the 21
that do fail the 90 %-in-the-area test (their shares: 0.38-0.75). It never fired on a flipped outfield piece in
T1, T3 or T10, and never falsely.

**In points (`K10nr`: K10's very errors with the repair switched off):** B's bench 0.7168, **−0.0161** (0/10;
p25 / p50 / stressed −0.0167 / −0.0157 / −0.0157), against K10's −0.0129: **the repair saves +0.0032, a fifth of
the keeper's cost**, all of it on 4 matches (J03WMX +0.004, J03WR9 +0.009, metrica2 +0.003, metrica3 +0.015, where
it undid 75 % of the wrong frames); where it undid little or nothing, K10 and K10nr are identical to the last digit.

| match | K10 | K10nr (no repair) | the repair saves | wrong keeper frames it undid |
|---|---|---|---|---|
| J03WMX | −0.0048 | −0.0091 | +0.0042 | 14 % |
| J03WN1 | −0.0101 | −0.0101 | 0 | 6 % |
| J03WOH | −0.0171 | −0.0171 | 0 | 0 % |
| J03WOY | −0.0204 | −0.0204 | 0 | 3 % |
| J03WPY | −0.0105 | −0.0105 | 0 | 0 % |
| J03WQQ | −0.0175 | −0.0175 | 0 | 0 % |
| J03WR9 | −0.0064 | −0.0151 | +0.0087 | 45 % |
| metrica1 | −0.0160 | −0.0160 | 0 | 0 % |
| metrica2 | −0.0206 | −0.0236 | +0.0029 | 2 % |
| metrica3 | −0.0060 | −0.0212 | +0.0152 | 75 % |
| **mean** | **−0.0129** | **−0.0161** | **+0.0032** | 16 % |

### How much the drop depends on which pieces are hit (second draws)

| setting | first draw | second draw (`T1b`, `T3b`) | per match, the two draws differ by |
|---|---|---|---|
| T1 | −0.0183 (0/10) | −0.0179 (0/10; p25 / p50 / stressed −0.0184 / −0.0185 / −0.0169) | up to 0.027 (metrica1 −0.037 / −0.009) |
| T3 | −0.0408 (0/10) | −0.0455 (0/10; p25 / p50 / stressed −0.0439 / −0.0456 / −0.0468; IDSSE / Metrica −0.0367 / −0.0658) | up to 0.050 (J03WR9 −0.014 / −0.065) |

The mean over 10 matches holds to within 0.005 from one draw to the next (0.0004 at 1 %, 0.0047 at 3 %), and every
match loses on every draw; a single match's size does not hold (which pieces land near the ball decides it).

## What it means (plain words, for Jose)

**Yes: wrong teams cost v15's branch B a lot, and the keeper repair barely covers it.** On the bench, with every
model as deployed and only the test match's tracking spoiled:

- **1 % of the outfield players' frames in the wrong team costs B's bench about −0.018** (−0.0183 and −0.0179 on
  two draws, 10 of 10 matches down). 1 % is roughly 18 player-minutes in a 90-minute match: one outfield player in
  the wrong colour for 18 minutes, or 36 half-minute episodes. **That is half of what v15 gained over v13's B, and
  five times the noise of a retrained network.** 3 % costs −0.041 to −0.046 (two draws), more than v15's whole
  gain; 10 % −0.138.
  A rough rule from the low end of the curve: **each half-minute of one outfield player in the wrong colour costs
  about −0.0005**, so ten such episodes cost what the B bar asks a change to win (0.005).
- **The keeper in the wrong team for ~10 % of the match (about 9 minutes each) costs −0.013 with v8's repair, −0.016
  without it.** The repair only looks at a keeper standing in his 6-yard box for 5 s with the ball far away: 21 of
  157 wrong pieces met that, 12 were repaired, a sixth of the wrong frames, a fifth of the cost.
- **Players misplaced by 10-20 m on 3 % of the outfield frames cost −0.014** (a wrong team hurts about three times
  more than a misplacement, frame for frame).
- **The cost is where the wrong player is near the ball** (80-92 % of the frames lost for outfield errors): B
  decides who has the ball largely from the team of the players around it, and believes the colours. A wrong keeper
  also costs away from the ball.

**Worth fixing before the deadline?** That depends on how often it happens on our match, which this page does not
measure (and must not: it is not measured on our match). The numbers say the cost is roughly proportional at low
rates: a handful of short episodes per half costs a few thousandths; wrong colours seen for many minutes of the
match cost more than anything the lab has gained since v15. So:

1. **If what Jose sees adds up to minutes** (say, ten or more half-minute episodes of an outfield player in the wrong
   colour, or the keeper wrong for minutes at a time), it is the largest known loss on our match and worth a narrow
   try before Nov 17. **A fix still needs its own test** before it goes in: this harness is that test, ready to run
   (a fix must win back a clear part of T1 / T3 / K10's loss **and** change nothing on `none`, where every team is
   right, since a wrong re-teaming costs exactly what the errors here cost). The most tractable piece is the keeper
   (a wider keeper rule than v8's; its test is K10 against K10nr against `none`). Re-teaming outfield pieces from
   positions alone is a new model with a real risk of wrong flips: less likely to be ready and tested in two days.
2. **If it is rare** (a few seconds here and there), it costs under 0.005 and there is nothing to fix before the
   deadline; it stays a first priority for the next phase, next to the missing players near the ball
   ([cleaning.md](cleaning.md)): both live in the same place, the players around the ball.

**Caveats.** The errors here hit random pieces anywhere on the pitch; a real tracker more likely swaps colours in
crowds, which are near the ball, so the cost per wrong frame on our match may be higher than here. Only branch B was
measured (our match runs B); whether such errors could move the stoppage switch's signal (it reads v4's players-only
P(dead), which reads the teams) was not measured. Keepers were found from the clean tracking (the person of each
team nearest his own goal; the nearest outfield player stands about twice as far away).

## Files

- `bench/lab_trackerr.py` (new): the errors, the inputs recomputed, the scoring, the report.
- `bench/results/lab_trackerr/`: `report.json` (every setting: B's bench, drop, per variant, per source, per match,
  zones, repair), `scores_<setting>.csv` (every unit), `reproduction.csv` (the check against round 8's CSV).
- `data/external/lab_trackerr/` (gitignored): each setting's letters, start log loss, zones and what was corrupted
  (`info_<match>.json`); the clean training rows and fold models shared by all settings. The copies' bulky caches
  under the scratchpad (`trackerr/wc_<setting>/`) were deleted once each setting was scored.
