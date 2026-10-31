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
