# Draft: questions to the organisers (2026-11-12)

RULES.md says to ask by opening an issue, but issues are now disabled on
`metrica-sports/more-than-a-hack-2027` (checked 2026-11-12), so this is written to go by email, in reply to the
organisers' emails to registered teams. If issues reopen, it can be pasted as one. Jose sends it; nothing here is
sent automatically. Fill in the team name.

---

**Subject:** Questions on external data: Metrica's sample data, and tracking from other matches

Hello,

We're [team name]. Before the deadline we'd like to confirm a few points about external data, so that what we
submit stays within the rules. Each can be answered yes or no.

1. **Metrica's public sample data** ([github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)).
   Its repository has no licence file. May we use those three games to train and validate our method, credited in
   our technical document? (If not, we have a version trained on IDSSE's CC BY 4.0 data only.)

2. **Automatic tracking from other matches.** Every open dataset we found treats the ball differently from an
   automatic tracker: DFL's data freezes it in dead time, and the 2019 samples remove it. Could Metrica share
   automatic, uncorrected tracking (players and ball, the same pipeline as the provided ATD) from a few other
   matches, with their events or possession, for all teams?

3. **If not: Metrica's own products on our own footage.** May a team run a Metrica product with automatic
   tracking (for example a PlayBase or Nexus trial) on its own recording of another game, and use that tracking for
   training or validation, explained in the technical document?

4. **A general question about the tracker, not about the provided match.** While play is stopped (the ball carried
   to a throw-in, or held before a free kick), does Metrica's automatic tracking usually keep reporting the ball,
   freeze it, or drop it? A pointer to documentation would be enough.

5. **Licensed data.** Is a dataset whose licence forbids redistribution acceptable if the organisers can obtain it
   themselves (for example on request to the provider), with only the models trained on it in our repository?

6. **A won header without control** (asked earlier in issue #17, before issues closed). Does winning a header with
   no control of the ball start a possession for that team, or does the ball stay with the last team until someone
   controls it?

Answers that apply to all teams, published wherever suits you, would be ideal.

Thank you,
[name], for [team name]

---

## Why each question (not part of the message)

| # | what it decides for us |
|---|---|
| 1 | v13's `all` chain trains its post stages on Metrica's samples (limitations.md, audit item D, grey zone G8); a "no" means submitting with bench set `idsse` (`v13_idsse.joblib`, branch A only) |
| 2–3 | the only way to see how Metrica's own tracker treats the ball in stoppages on matches other than ours: v13's switch (docs/lab/gate.md) exists because we can't know |
| 4 | the same, answered in words; an answer goes into the technical document; v13's switch is never overridden by hand |
| 5 | paid tracking (docs/research/README.md, "Paid data"); only for after Phase 1 |
| 6 | grey zone G5: what counts as a change of possession, which the bench's labels may not share |
