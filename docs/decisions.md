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
