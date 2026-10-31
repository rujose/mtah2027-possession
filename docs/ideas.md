# Ideas queue

Ideas waiting their turn, with where each came from. When one is picked up it gets a model page
written before the code, and its settings come from the bench, never from our match. Observations
from watching become **general rules**, never constants tied to specific frames (RULES.md, #14).

| # | idea | from | why it might help | how it would be tested | status |
|---|---|---|---|---|---|
| 1 | **Players' movement relative to the ball says whether play is live.** In play, the nearest players close in on the ball and the team shifts with it. When it's dead, they stop caring: the ball goes one way and nobody follows, and players walk or turn away | Jose, 2026-10-30 | Catches stoppages where players keep jogging (v4's weak spot: the whistles at 23697 and 54740) and slow live play that looks stopped. Only where the ball is seen, so mostly the first seconds of a stoppage | Ball-relative features (nearest players moving toward the ball; team centre moving with the ball; ball moving with nobody reacting) added to v4's dead-ball model, trained only on frames where the ball is seen, on the bench | queued after step 4 |
| 2 | **"No ball" is a real dead-ball sign on our data too**, because our tracking is Metrica's as well: missing on 83 % of frames where players stand still against 29.5 % where they move. Weaker than Metrica's cleaned samples (92–99 % against 3–10 %) | Jose, 2026-10-30 | v4 excludes the ball, to avoid learning the samples' too-strong version, and so throws away a real signal | First close the bench's known gap (its damaged ball is missing 66 % while stopped, against our 83 %). Then teach the dead-ball model how much "no ball" means, on IDSSE damaged like ours, not on Metrica's blanked samples | queued after step 4 |
| 3 | Jose's answers to the questions about stoppages, restarts, crowded moments, long balls, pressing, tracking errors and goalkeepers | Jose, pending | Football knowledge the data can't give | Each answer becomes a general rule or a feature idea, tested on the bench where possible | waiting for Jose, after step 4 |
