# Models v11–v13: the touch line

| | |
|---|---|
| **status** | **v11 built 2026-11-03 and adopted** (held out 0.571 against v9's 0.561, better on 10/10; IDSSE only 7/7). v12 and v13 proposed, not built |
| **what** | three steps, each its own version and each judged on its own: **v11** moves every change of team onto the new team's touch; **v12** adds short possessions at clear touches by the other team; **v13** re-chooses the decoder's team settings with those in the loop |
| **change from v10** | v11 is one step after v10 (and after v9's restart snap): H ↔ A changes move to a seen touch; the team and the number of possessions never change. Everything in v10 stays, including its two switches |
| **new code** | `possession/touches.py` (the touch detector), `possession/v11.py` (later `v12.py`, `v13.py`), `bench/touch_timing.py` (measurements and the leave-one-out choice, `--set all \| idsse`), `bench/check_v11.py` (v11 end to end on a bench match) |
| **rules** | everything is tuned on the bench (external data) only; Claude never runs it on our match (Jose does) |
| **effort** | v11 ~3 h; v12 ~1 h after it; v13 ~2 h plus ~30 min of bench runs |

## Why

**The error budget** ([../ideas.md](../ideas.md), v10 candidates): of what v9 leaves on the bench,
two things are about the moment the ball changes team:

| error | if fixed perfectly |
|---|---|
| **team-change timing** (H ↔ A changes on the true frame) | **+0.096** |
| **missed possessions** (35 % of true ones; median 1.6 s; half of them between two possessions of the other team) | **+0.250** |

Our team changes are off by ±0.6 s, as often early as late. That's the same picture v9 found for
restarts, and the same cause: the decoder has no sharp moment to land on.

**The truth's convention, checked on the bench** (IDSSE, clean ball, 2,349 team changes):
- **at every change** a player of the new team is within 2 m of the ball (**100 %**);
- in 71 % the ball was coming toward that player over the 0.4 s before.

So a possession changes hands **at the new team's touch**: the reception or interception.

## The touch detector (`possession/touches.py`)

Frame *k* is **a touch by team T** when:
1. the ball is seen at *k*, 0.2 s before and 0.2 s after;
2. a player of T is within **2 m** of it at *k*;
3. the ball's velocity over the 0.2 s after *k* differs from its velocity over the 0.2 s before by
   at least **ΔV**: it was stopped, deflected, or played.

A touch shows as a short run of such frames; the touch is the frame in the run where the change is
largest.

| setting | value | why |
|---|---|---|
| player distance | 2 m | v1's "on the ball" distance: a stride plus tracking error |
| look either side | 0.2 s (5 frames) | short enough to separate two touches; long enough to see a change through the tracking's jitter |
| ΔV | **3 or 5 m/s, chosen on the bench** | a trap or a kick changes the ball's velocity by much more; the damaged ball's jitter makes about 2 m/s of apparent change |

**What it sees on the bench** (damaged tracking, 41 %), measured by `bench/touch_timing.py`:

| ΔV | new-team touch within 0.2 s of a true change | within 0.5 s | touches a minute |
|---|---|---|---|
| 3 m/s | **62 %** (IDSSE only 58 %) | 65 % | 21.0 |
| 5 m/s | 58 % (53 %) | 60 % | 17.4 |

The rest happen while the ball is missing.

`possession/passes.py` (v9) stays as it is: a pass is a touch that sends the ball away, with its own
settings.

## v11: team changes at the touch

After v10 (and v9's restart snap), at each change from team O to team N at frame *g*:
1. **Look** for touches by N from *g* − **window** to *g* + **window**.
2. **Take the nearest to *g*.**
3. **Move the change there:**
   - **touch before *g*:** O's frames in between become N's. Only if they're all O's and O keeps
     at least the frame before the touch: the move never crosses another change and never swallows
     O's possession.
   - **touch after *g*:** N's frames before the touch become O's. Only if they're all N's and the
     touch itself is N's: N's possession is never swallowed either.
   - **no touch seen:** nothing moves.

| setting | grid | chosen by |
|---|---|---|
| ΔV | 3, 5 m/s | leave-one-out on the bench |
| window | ±0.5 s, ±1 s, 2 s back / 1 s ahead | leave-one-out on the bench |

**Why the nearest touch, not the earliest:** with a wider look back, "earliest" grabs an older touch
by N, from before O even had the ball (probe: −0.003 to −0.008 at 2 s back). "Nearest" holds up in every
window tried.

## v11: what the bench says (built 2026-11-03)

`python -m bench.touch_timing --set all` and `--set idsse`, about 30 s each; results in
`bench/results/touch_timing_*`. The input is v9's labels on the bench (v7's out-of-sample predictions
with the restart snap).

| | all 10 matches | IDSSE only (7) |
|---|---|---|
| **chosen** | **ΔV 3 m/s, ±1 s** (8 of 10 folds picked exactly this) | **ΔV 3 m/s, 2 s back / 1 s ahead** (4 of 7 exactly; 6 of 7 picked 2 s back) |
| **held-out spellF1** | **0.5711** against v9's 0.5613 (**+0.0098**), better on **10/10** | **0.5607** against 0.5497 (**+0.0109**), **7/7** |
| macroF1 | 0.870 against 0.868 | 0.876 against 0.873 |
| changes moved | about half of them (1,465): 645 earlier, 820 later; quartiles −0.20 / +0.40 s | 1,044: 514 earlier, 530 later; −0.32 / +0.36 s |
| **moved changes within 0.2 s of the true one** | **31 % → 70 %**; median distance 0.44 s → **0.04 s** | 28 % → 70 %; 0.44 s → 0.04 s |
| without v9's snap (v10's `RESTART_SNAP = False`) | 0.5685 against v7's 0.5557, 10/10 | 0.5608 against 0.5482, 7/7 |
| detector on tracking damaged at 30 % / 55 % | 0.5732 / 0.5729 against 0.5613, 10/10 both | 0.5656 / 0.5619 against 0.5497, 7/7 both |
| **adopted** | **yes** | **yes** |

- **The truth's convention, re-measured by the script:** a new-team player within 2 m at all 2,349
  IDSSE changes; the ball coming toward them in 71 %.
- **ΔV 3 and 5 almost tie** (in sample 0.5727 against 0.5725 at ±1 s): the move is robust to it.
  The window matters more: ±0.5 s gains about half as much.
- **The moved changes land on the truth's frame:** median 0.04 s away, from 0.44 s.
- **It works with or without v9's snap,** and at every damage level tried, so v10's two switches
  keep working.
- **Below the probe's +0.013, and why:** the probe let a move swallow a whole possession; the build
  doesn't (the rule above). On the same labels (ΔV 5, ±1 s): 0.5743 with the probe's rule, 0.5725
  with the build's. The probe's rule removed 28 possessions in 10 matches. Removing possessions is a
  different change (their number), so it isn't hidden inside v11; it's noted under v12.
- **End to end** (`python -m bench.check_v11`, ~3 min: the full v11 on J03WR9, the IDSSE match
  PathCRF never saw, with each set's settings):

  | | changes moved | frames differ from v10 | all within the window of a change | same possessions, same teams in order | same labels on a second run |
  |---|---|---|---|---|---|
  | all | 132 | 1,242 | yes | yes | yes |
  | idsse | 140 | 1,564 | yes | yes | yes |

### Our match

Jose runs it; Claude doesn't see the output:
```
.venv/bin/python -m possession.v11      # twice: the SHA-256 must repeat
.venv/bin/python evaluation/validate_submission.py evaluation/results/v11.csv
```

## v12: short possessions at clear touches (judged separately)

Inside a possession of O, where N **clearly** touches the ball and O touches it again 1–3 s later,
N gets those frames.

| setting | value | why |
|---|---|---|
| clearly | N's player within 2 m, and O's nearest at least **1 m further** | in a duel, "the nearest team" is a coin toss (Jose: "the tracking goes wrong when a player has the ball close to another") |
| length | **1–3 s** | the missed possessions' quartiles are 0.9 and 3.0 s; under 1 s the touch is more likely a failed tackle |
| ΔV | v11's | |

All fixed by reasoning. **Found while building v11:** the probe's looser move, which let a touch
swallow a short possession, removed 28 possessions in 10 matches and added +0.0018. Spurious
possessions are on the error budget too (+0.043), so v12 may test that as its own variant.

**Expect it may not pass:** in the probe, a loose version (any touch, up to
5 s) was a disaster (**−0.14 to −0.20**, worse on 10/10); this strict one added only about +0.003
(7/10).

## v13: the decoder re-chosen with touches in the loop

v3 chose the decoder's team settings (minimum possession 0.8 s, team switch price 0.5) before any
of the later evidence existed. With touches placing the changes, a shorter minimum may now pay.

| setting | grid |
|---|---|
| minimum possession | 10, 15, 20, 30 frames (0.4–1.2 s) |
| team switch price | 0.25, 0.5, 1 |

- **What it needs:** v7's out-of-sample costs per bench match. `bench/eval_pathcrf.py` gets an
  option to save them (gitignored, ~4 MB per match) and is re-run for both sets (~30 min). The
  decoder is then re-run for each setting, with v9's snap and v11 (and v12, if adopted) applied after.
- **Untouched:** the dead-ball settings, the restart and PathCRF weights stay as chosen.

## How each step is chosen and judged (bench)

- **Same procedure as v9:**
  - leave one match out over the 10 matches (or 7 with `--set idsse`);
  - held-out spellF1 and macroF1;
  - the timing check (moved changes within 0.2 s of the truth, before and after);
  - the same sensitivity check at 30 % and 55 % damage.
- **Adopted only if** the held-out mean beats the previous version's and at least 70 % of the
  matches improve (7 of 10, 5 of 7). Each step is judged on top of the one before it.
- **Both sets:** every step runs with `all` and with `idsse`, so the switch in v10 keeps working.
- **Our match:** Jose runs `python -m possession.v11`; it prints only the file, its SHA-256 and the
  format check.

## What a probe already shows (exploration, not the procedure above)

All 10 bench matches, on v9's bench labels, one setting at a time, no leave-one-out (v9 = 0.5613):

| variant | mean spellF1 | matches better |
|---|---|---|
| **v11, nearest touch, ±1 s, ΔV 5 m/s** (the probe's rule: may swallow a possession) | **0.5743 (+0.0130)** | **10/10** |
| v11, nearest, ±1 s, ΔV 3 m/s | 0.5735 (+0.0122) | 10/10 |
| v11, nearest, 2 s back / 1 s ahead, ΔV 5 | 0.5717 (+0.0104) | 10/10 |
| v11, nearest, ±0.5 s, ΔV 5 | 0.5703 (+0.0089) | 9/10 |
| v11, earliest, 2 s back / 1 s ahead, ΔV 5 | 0.5588 (−0.0026) | 3/10 |
| v11 with a 1 m clarity margin, ±1 s | 0.5696 (+0.0083) | 10/10 |
| v12 loose (any touch, up to 5 s) on v9 | 0.39–0.42 (−0.14 to −0.20) | 0/10 |
| v12 strict, on v11 with a 1 m margin | 0.5728 (+0.0032 over that v11) | 7/10 |

- **v11 is the solid step,** about twice v9's gain, with 10/10 in every reasonable window. A clarity
  margin hurts it: it drops real touches.
- **v12 is fragile.** It's worth the bar only in its strictest form.
- **v13 is unmeasured.**

## Failure modes

1. **Duels.** With players of both teams close, the touch goes to whichever team has a player
   within 2 m; v11 then picks the new team's touch even if the old team made it. Bounded by the
   window, and the probe holds 10/10.
2. **High balls** (Jose: the ball dot is its shadow on the grass). A ball landing or bouncing next
   to a player changes its ground speed, which can look like a touch. It only matters inside the
   window around a change.
3. **Unseen touches:** about a third of team changes have none in view. Nothing moves there, which
   costs nothing.
4. **The second touch:** if N's first touch is unseen and its second is seen, the change moves to
   the second (a fraction of a second late).
5. **Our tracking's jitter** may differ from the bench's damage, so touches per minute could differ
   on our match. Only Jose's run can show that, and only if the organisers allow summary numbers.

## Decisions for Jose before building

**Decided 2026-11-03:** Jose chose all seven recommended options ("Yes, build v11").

| | choice | recommended | why |
|---|---|---|---|
| 1 | Move changes after decoding (as v9), rather than make touches a decoder clue | **after decoding** | Transparent; can't change the team or the number of possessions; easy to measure. v13 brings touches into the decoder's settings anyway |
| 2 | The detector's distance (2 m) and look (0.2 s) fixed by reasoning; ΔV and the window chosen on the bench | **yes** | Few knobs; the two that matter are chosen out of sample |
| 3 | The nearest touch in the window, not the earliest | **nearest** | Earliest collapses with a wider window (probe) |
| 4 | v12 built and judged on its own, with its settings fixed by reasoning, expecting it may fail | **yes** | The loose version is a disaster; only the bar decides |
| 5 | v13: save v7's out-of-sample costs on the bench (both sets) to re-choose the decoder's team settings | **yes** | The decoder was chosen before any of the later evidence |
| 6 | Each step its own version (v11, v12, v13), same bar as v9 (better mean, at least 70 % of matches) | **yes** | One change per version |
| 7 | Every step run with both bench sets (`all`, `idsse`) | **yes** | v10's switch keeps working whatever the organisers answer |
