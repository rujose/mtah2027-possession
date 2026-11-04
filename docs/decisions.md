# Decision log

Newest at the bottom. Each entry: what was decided, why, and what it rules out.

### 2026-10-30: work inside a clone of the organisers' repository

- **What:** our code and docs live in `~/Code/more-than-a-hack-2027`, a clone of
  `metrica-sports/more-than-a-hack-2027`, in folders the organisers don't use (`docs/`, `tools/`,
  later the model code).
- **Why:** the scorer, the validator, `games.py` (frame alignment, paths) and the loaders come with
  it. The submission must be a public repo that regenerates our file, and a fork of theirs makes
  "run the official scorer on our output" trivial for anyone checking.
- **Open:** where the public repo will live (fork on GitHub, team account) is not decided.

### 2026-10-30: data stays out of git

- **What:** the full bundle (9 files, ~19.5 GB) sits in `data/tactical/` and `data/broadcast/`,
  which the repo's `.gitignore` already excludes.
- **Why:** the data terms forbid redistributing it, and our repo will be public.

### 2026-10-30: how we read the "no cloud LLM" clause

> **Superseded 2026-11-03** by the organisers' clarification ("uploading the provided data to AI
> services" is not allowed): see [limitations.md](limitations.md), section 0, item A.

- **What:** we inspect the data normally (including with Claude). No LLM ever labels frames.
- **Why:** Jose's reading: the clause targets extracting frames to an LLM as the labelling method.

### 2026-10-30: never tune on the public 5 minutes

- **What:** the public ground truth is used only to check the pipeline is not broken. Settings come
  from football reasoning, or from public data on *other* matches.
- **Why:** it is a rule (RULES.md), and the organisers show it picks the wrong setting anyway.
  Those minutes are also unrepresentative: the ball is visible on 88.2 % of them versus 59.3 % over
  the match.

### 2026-10-30: everything is explained and documented before it is built

- **What:** every model gets a page in `docs/models/` with why, how, every setting and where its
  value comes from, expected failures, and results. Every number about the data comes from
  `tools/explore_data.py`.
- **Why:** Jose's requirement. It also builds the technical document the submission needs.

### 2026-10-30: thresholds may be set from unlabelled distributions

- **What:** looking at how a quantity is spread over the match (for example the team's mean speed)
  to place a threshold is allowed. Looking at how it relates to the true labels is not.
- **Why:** it uses no labels, so it is the same as reading the data. It is how v1's "stopped"
  threshold moved from 1.5 to 1.0 m/s.

### 2026-10-30: v1 built; its settings are a reasoned first cut, frozen

- **What:** v1 is built as documented, with the four changes listed in its page. Its weights and
  switch penalties stay as they are. The public-window numbers (spellF1 0.43, D recall 0.44, 12
  possessions against 17) are recorded but not used to change anything.
- **Why:** moving a setting because the public window looks better is tuning on it. Changes to the
  weights come from reasoning or from other matches' data.

### 2026-10-30: ideas may come from looking at errors; settings may not

- **What:** the whistle idea came from Jose watching a v1 error in the public 5 minutes (1st half
  03:06). Using that to add a *kind* of clue (a referee's whistle means the ball is dead) is
  football reasoning. Choosing the whistle detector's *thresholds* by how well they match the public
  truth would be tuning, and is not allowed.
- **Why:** the line in RULES.md is between understanding and fitting. Any thresholds get set from
  audio alone or from other matches.

### 2026-10-30: v2 built as proposed; exact dependency versions pinned

- **What:** v2 = v1 + whistle clue with the slowdown test, settings exactly as planned. Our exact
  library versions go in `requirements-team.txt`, and the audio is decoded with PyAV (which bundles
  its own ffmpeg) instead of the system ffmpeg.
- **Why:** the organisers have to regenerate our file byte for byte. A different numpy, pandas or
  audio decoder could shift a number across a threshold.
- **Not acted on:** the false alarm at 1st half 03:57 and the three 3-second dead stretches. They're
  recorded in the v2 page. Changing settings because of the public-window one would be tuning.

### 2026-10-30: notes while watching, but never labels

- **What:** the viewer lets Jose pin free-text notes to frames (what he sees: tracking problems,
  whistles, restarts, questions). It does **not** offer a row of his own H/A/D calls. Notes live in
  `viewer/notes/notes.json` (gitignored) and no model code reads them.
- **Why:** Jose asked to "add what you see" and compare with the data while watching. RULES.md:
  "No hand labels on the provided game, in any amount and for any purpose: not for the submission,
  not to train a model and not to validate one", and "watching is not the problem; turning what you
  saw into labels, corrections, or rules and constants tied to specific frames of the game is."
  Free-text notes stay on the watching side. A label row would be hand-labelling.

### 2026-10-30: a match viewer for diagnosis

- **What:** a local web app (`python -m viewer.server`) showing video, tracking overlay, pitch map,
  label and clue timelines on one frame. Its cache, derived from the gated data, is gitignored.
- **Why:** Jose wants to see the match next to the data. Watching to understand failures is allowed
  (RULES.md). What stays off-limits is turning what we see on specific frames into labels, corrections
  or frame-specific constants.

### 2026-10-30: limits mapped; other matches become our measuring stick

- **What:** [limitations.md](limitations.md) lists every hard limit, what the rules and the
  organisers' issue answers allow that isn't obvious, the metric's levers, open grey zones and traps.
  The main consequence: there is no leaderboard and the public window may not choose between
  versions (EVALUATION.md), so tracking-based clues are compared on **other matches** (allowed,
  #10, #16), and clues only our match has (audio, video, Smart Tagging) by reasoning plus
  label-free checks on our match.
- **Why:** Jose asked for room to extend the work in ways that are allowed but not obvious.
- **Open:** grey zones G1 (self-training with algorithmic pseudo-labels) and G2 (broadcast
  commentary) would need a public question to the organisers before we build on them.

### 2026-10-30: research survey done; nothing built from it yet

- **What:** four research reports in [research/](research/) (models, data and code, practitioner
  advice, extra signals) and a synthesis with a proposed order of work in
  [research/README.md](research/README.md). The proposed core is: an IDSSE test bench, a duration-aware
  decoder, then a dead-ball classifier trained on IDSSE. (A first draft put the broadcast ball fill
  second; data.md had already measured it at 12–13 % of the gaps, so it became a side item.)
- **Why:** Jose asked what models exist, what we can start from and what people recommend, all
  documented.
- **Not decided:** the order of work and any GitHub question to the organisers wait for Jose.

### 2026-10-30: the bench is built; on it, v1 ties the floor because it merges

- **What:** [bench.md](bench.md) built as approved (own parser, raw DFL labels plus a merged
  sensitivity run, Metrica samples later, damage statistics committed). The damage model changed
  while building: the ball became a seen ↔ missing chain whose chances depend on crowding, players
  stopping, and the lines, because our ball is missing 83 % of the time while players stand still
  and 29.5 % while they move. The bench reaches 66 % and 34 %; that gap is recorded as a known limit.
- **Result:** at our damage level v1 scores spellF1 0.224 ± 0.034 against the floor's 0.217 ±
  0.018, a tie. v1 finds ~148 possessions per match where there are ~437.
- **Why it matters:** it is the first evidence about v1 that doesn't come from the public window,
  and it points at the decoder (step 1) as the next change.
- **Rules:** no BAR-ATH labels touched; only its input tracking fed the damage statistics. The
  damage was not adjusted to reproduce the organisers' published baseline scores.

### 2026-10-30: v3 built; decoder settings chosen on the bench

- **What:** v3 = v2's clues + a decoder where a label must last a minimum number of frames, with
  four settings chosen on the bench by leave-one-match-out and the cautious tie rule: team price
  0.5 (v1: 15), dead-ball price 50 (25), minimum possession 0.8 s (none), no minimum stoppage. All 7
  folds chose the same setting. Held-out spellF1 0.369 ± 0.026 against v1's 0.224, better on 7/7
  matches, and the gain holds at 30 % and 55 % damage and with merged labels. The tag and set-piece
  weights were rescaled with the prices so they keep v1's meaning (reasoning, not fitting).
- **On our match:** 429 possessions (v2: 177), mean 8.9 s.
- **Public 5 minutes, not acted on:** v3 0.335 against v2 0.427 (20 possessions against a true 17).
  About 1.5 possessions; EVALUATION.md rules the window out for choosing versions. Recorded as a hint
  for the convention risk, not as a reason to change anything.
- **Open:** whether to widen the grid (two settings sit on its edges), and how to test Metrica's
  convention on labelled data from another match.

### 2026-10-30: convention check done; v3 stands

- **What:** labels built from Metrica's public sample games (3 matches, events tagged by Metrica)
  and from DFL's events with the same recipe, compared during live play
  ([convention-check.md](convention-check.md)). Jose's choices: use the sample data now (no licence
  file; Metrica's own public repo, checking only), a won challenge doesn't flip possession on its
  own, and put the games on the bench regardless.
- **Result:** like for like, Metrica's events change hands ~21 % less often than DFL's (52.8 against
  67.1 per 10 live minutes); the fixed rule says "in between" (2 games inside DFL's range, 1 below).
  On the bench, with Metrica's labels, v3 scores 0.512 against v1's 0.384 and finds 344 possessions
  where there are 351. The decoder search picks v3's exact setting on Metrica's games alone and in
  all 10 folds over all 10 matches.
- **So:** no v4; v3's settings stand. The bench now has a second set (`--set metrica|all`).
- **Still unknown:** the truth's own convention (corrected tracking, not events). Nothing public
  measures it.

### 2026-10-30: v4 built; the dead-ball model is gradient-boosted trees

- **What:** v4 = v3 with v1's "stopped / active" rules replaced by P(dead) from a players-only
  model trained on all 10 bench matches ([models/v4-dead-ball.md](models/v4-dead-ball.md)). Jose's
  choices: try logistic regression and trees, keep logistic unless trees win by more than 0.01;
  scikit-learn 1.9.1 pinned; no ball features; replace rather than add; name v4.
- **Result:** trees won by 0.013 held out, so scikit-learn and the committed model file are needed
  at run time. Held-out spellF1 0.433 against v3's 0.412 (better on 8/10; on all 7 IDSSE, level on
  Metrica). Stoppages found rose from 48 % to 81 % with almost no new false ones.
- **Not acted on:** on our match two whistles that v2 counts are no longer followed by a stoppage
  (frames 23697, 54740), and stoppages are fewer and longer than the bench suggests (74 over 106
  min). Both recorded as open questions on the v4 page.

### 2026-10-30: v5, the whistle gets its weight back

- **What:** Jose watched frames 23697 and 54740 in the viewer, the two counted whistles v4 no longer
  followed with a stoppage: both were real stoppages. The cause: the dead-ball model, confident play
  goes on (P = 0.01), outweighed the whistle (2.30 against 2.0 per frame). v5 sets the whistle to
  v2's 2.0 plus the model's strongest live evidence (0.5 × −log 0.01), 4.30, as a formula.
- **Why it is within the rules:** the viewing gave the idea; the value comes from the costs, applies
  to all 11 counted whistles alike, and was set before looking at the result. The two frames were
  not used to choose or to accept it (#10, #14).
- **Result (label-free):** counted whistles followed by a stoppage 8/11 → 11/11; 488 frames changed.
  The two stoppages now last exactly 3 s; their true end is left to step 3 (restart shapes).

### 2026-10-30: v6 built; restart team much better, ranking gain inside the noise

- **What:** v6 = v5 + a clue for which team takes the restart, from a players-only layout model
  trained on all 10 bench matches with four-way mirroring ([models/v6-restart-team.md](models/v6-restart-team.md)).
  Jose's choices: aim at the restart team (timing later), one learned model, Smart Tagging's team as
  a cross-check only, name v6. The family rule kept logistic regression (trees ahead by 0.004).
- **Result:** the right team after a found stoppage rose from 62 % to 87 % (IDSSE) and 72 % to 84 %
  (Metrica), but held-out spellF1 is 0.435 against v4/v5's 0.433 (better on 6/10; worse on Metrica).
  On our match the model agrees with Smart Tagging's set-piece team on 23/25 tags, v6's decoded
  first team on 14/25 (v5: 10/25).
- **Corrected:** the clue does affect dead ball indirectly (dead share 46.3 % → 48.3 %).
- **Open:** whether v6 replaces v5 as the main line, and why a much better restart team barely moves
  spellF1.

### 2026-10-30: v6 revised after the diagnosis

- **What:** the diagnosis (v6 page) showed the decoder follows the restart clue (99 %), but the clue
  made staying dead cheaper and fired in open play, so the gain right after restarts (+394) was
  mostly lost elsewhere (−243). Jose approved two reasoned changes: only the unlikely team pays, by
  the odds (no new constant); the clue waits for 2 s of dead-looking play (1 % of bench stoppages
  are shorter). Weight and carry re-chosen with the same procedure: unchanged (1, 3 s).
- **Result:** after restarts +429, elsewhere −62; held-out spellF1 0.443 against 0.433, better on
  8/10, holding at 30 %/55 % damage and with merged labels. Dead share on our match back to 46.5 %.
- **So:** the revised v6 is the best-supported version on the bench, by a small, consistent margin.

### 2026-10-30: v7 built (PathCRF); the viewer shows v6/v7's signals and the bench matches

- **What:** v7 = v6 + PathCRF's estimate of who has the ball, from the players' movement, as a
  clue where only the unlikely team pays. PathCRF is a published model trained by its authors on
  other matches; its code is vendored unchanged (MPL-2.0), its weights downloaded at a pinned
  commit with a hash check. Jose's choices: published weights, run at reproduction time, vendored
  code, settings on the 4 matches it never saw, PyTorch pinned, name v7. Added while building:
  PyTorch Geometric (imported by PathCRF's unchanged code).
- **Result:** reproduced at 91.0 % on clean J03WR9 (paper 92.4 %); on the unseen matches with our
  damage, right team inside ball gaps 85.7 % (v6 69.5 %), changes caught there 55 % (v6 8 %).
  Held-out spellF1 0.565 against 0.501, better on 4/4, weight 0.5 on every live frame.
- **Watch:** on our match v7 reports 347 possessions (v6 439), and PathCRF agrees with the direct
  on-the-ball evidence 86 % there against 91 % on the bench: some domain shift.
- **Viewer:** v6/v7's signals as lanes; the 10 bench matches viewable with their true labels and
  out-of-sample predictions, without video (Jose asked for both).

### 2026-10-30: v8 built (keeper repair)

- **What:** Jose saw a keeper tracked as the other team. A general rule (6-yard box, ball far, 5 s,
  90 % of the piece inside the area; boxes from the Laws, the rest reasoned) moves such pieces to a
  new slot of the defending team; v7 runs unchanged on top. Jose's choices: repair the whole piece,
  name it v8.
- **Result:** on our match, 1 piece repaired (Barça's keeper in Athletic slot 45, 230 s graded);
  on the bench, 0 false repairs on clean tracking and at every damage level, so v8 equals v7 there.
  446 frames differ from v7, possessions 347 → 351; the restart model's P(Barça restarts) inside
  the piece 0.449 → 0.689.
- **So:** kept because it fixes the input, not for a measured gain (the bench can't score it). The
  viewer now draws the repaired tracking.
- **Next (Jose's observation, same day):** possessions after a stoppage should start at the restart
  pass. The bench's truth confirms it exactly; the ceiling for fixing it is +0.018 (ideas.md, idea 10).

### 2026-10-30: v9 built (possessions start at the restart pass)

- **What:** Jose saw our possessions after a stoppage start when the ball reaches the receiver; the
  bench's truth starts them on the pass, exactly (698/698 IDSSE restarts). v9 moves each dead → team
  change to the earliest seen pass by that team (player within 2 m, ball leaving him at over 4 m/s;
  fixed by reasoning) within a window chosen on the bench. Snapping to the ball's reappearance was
  tried and rejected: it hurt the IDSSE matches. Jose's choices: snap after decoding, earliest pass,
  detector fixed by reasoning, adoption bar set in advance (better mean and at least 7/10), name v9.
- **Result:** window 5 s back / 1 s ahead, picked by every fold. Held-out spellF1 0.5613 against
  0.5557 (+0.0056), better on 10/10; macroF1 0.868 against 0.864; moved starts within 0.2 s of the
  truth 20 % → 83 %. Adopted. On our match 40 of 78 starts move, 34 earlier (median −0.94 s).
- **Watch:** the largest moves (up to −4.8 s) may be passes during the stoppage; they're listed on
  the page for the viewer.

### 2026-11-03: the organisers' clarification, and v10

- **What:** the organisers clarified the rules for everyone, overriding earlier answers: no checking
  output against the video, no data made by hand on the match, no provided data to AI services,
  tuning only on external data, datasets only with rights. Our audit ([limitations.md](limitations.md),
  section 0) found five items.
- **Done (v10):**
  - the whistle and the set-piece window are dropped (calibrated on our audio and the public truth);
  - IDSSE-only twins of every Metrica-trained stage are ready behind one switch;
  - the viewer shows no model output on our match, and notes are off there;
  - Claude no longer runs anything on our match or reads its output.

  Jose's choices: all eight recommended options on the v10 page.
- **Result:** v10 equals v9 wherever there's no audio or tags (0 frames differ on 3 bench matches).
  The IDSSE-only twins keep each stage's gain, with PathCRF's weight 1.0 (chosen on one match) and
  the restart carry 5 s.
- **Open:** the organisers' answer on items A and E, Metrica's samples, and v9's origin. Jose emails
  them.


### 2026-11-03: v11, each change of team on the new team's touch

- **What:** after v10, each change from one team to the other moves to the nearest seen touch by the
  new team: a player of it within 2 m while the ball's velocity changes by at least ΔV. Frames move
  only between the change and the touch, and never swallow a possession. On the bench the truth
  changes team at the new team's touch (a new-team player within 2 m at all 2,349 IDSSE changes),
  and ours were ±0.6 s off. Jose's choices: all seven recommended options on the page (move after
  decoding, nearest touch, distance and look fixed by reasoning, ΔV and window chosen on the bench,
  same bar as v9, both bench sets).
- **Result:** ΔV 3 m/s, ±1 s (IDSSE only: 2 s back / 1 s ahead). Held-out spellF1 0.5711 against
  v9's 0.5613 (+0.0098), better on 10/10; IDSSE only 0.5607 against 0.5497, 7/7. Moved changes within
  0.2 s of the truth 31 % → 70 %. It also gains without v9's snap and at 30 % and 55 % damage.
  Adopted.
- **Noted:** the probe's looser rule, which could swallow a short possession, scored 0.0018 more by
  removing 28 possessions in 10 matches. That changes the number of possessions, so it's left to
  v12's judging, not folded into v11.


### 2026-11-04: our versions back on our match's viewer, behind a switch

- **What:** `python -m viewer.server --models` rebuilds the cache with a label lane for every CSV in
  `evaluation/results/` (the baselines, then v1, v2, ... v11 by number) on our match. Without the
  switch the viewer is as v10 left it: video, tracking as delivered, tags, public truth, ball seen.
  Notes stay off on our match either way.
- **Why:** Jose's call, to see the versions run on the match. His reading of the clarification:
  looking is not checking, as long as nothing seen there changes a model. Claude flagged item B of
  the audit ([limitations.md](limitations.md), section 0: "our output checked against the video"),
  and Jose decided. Not asked of the organisers.
- **How we keep it apart:** Jose watches it. Claude may run the commands that write the CSVs and
  build the cache, whose output is only file paths, hashes, row counts and lane names (Jose's call,
  same day); it never opens our match in the viewer, reads the CSVs or the lanes, or prints
  statistics of them (the format check's label counts stay hidden). No setting, version choice or
  idea comes from what is seen there: those stay on the bench, whose matches show every version
  with their true labels.


### 2026-11-04: the lab, and v12

- **What:** Jose asked for versions built side by side by subagents, the valuable ones kept, the
  rest discarded, the winners mixed, and the whole thing improving itself. One harness
  (`bench/lab.py`), one bar fixed before any run (the held-out mean beats the chain without the
  stage; at least 70 % of the matches better; no drop on either source or on the matches PathCRF
  never saw), four rounds, ten builders ([lab.md](lab.md)). Kept: robust (a learned per-frame
  model, the dead-time ball blanked), short, segments, timing (`all`) or recall and dead (`idsse`).
  Discarded: the decoder re-chosen, stoppage on top of the learned model, polish (superseded).
- **Result:** v12 = the lab's chain, **0.703 held out against v11's 0.573 (+0.131), 10/10**; IDSSE
  set 0.670 against 0.563, 7/7. Built with the five recommended decisions on its page: robust as
  the start, the full chain per set, Smart Tagging off (no model has seen it), the models committed
  (trained on bench data only), and the port checked frame for frame on two bench matches (0
  frames differ).
- **Rule made explicit:** each bench set gets the stages that pass on it.
- **Open:** whether our seen ball is carried around in stoppages (robust's remaining exposure; a
  label-free check exists in lab/learned.md, not run); the gain on our match will be smaller than
  the bench's.
