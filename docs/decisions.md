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

### 2026-10-30: a match viewer for diagnosis

- **What:** a local web app (`python -m viewer.server`) showing video, tracking overlay, pitch map,
  label and clue timelines on one frame. Its cache, derived from the gated data, is gitignored.
- **Why:** Jose wants to see the match next to the data. Watching to understand failures is allowed
  (RULES.md). What stays off-limits is turning what we see on specific frames into labels, corrections
  or frame-specific constants.
