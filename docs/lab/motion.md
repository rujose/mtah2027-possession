# Lab candidate `motion`: how each team moves, without the ball

| | |
|---|---|
| **status** | **discarded** (2026-11-09). v12 stays as it is |
| **what** | robust plus 27 ball-free numbers per frame: for each team, with x measured from its own goal, the shift of its centroid, its front three and its back line (2nd–4th deepest, keeper left out) over the next and last 2, 4 and 6 s; and home minus away |
| **why** | another team's public repo (ZesterJ) reported +0.07 from team-movement features; a team in possession pushes forward and the other drops back, and that holds while the ball is missing |
| **code** | `bench/lab_motion.py`; results in `bench/results/lab_motion[_idsse].json`, `lab_motion_chains[_idsse].json` |

Everything but the features is robust's: blanked ball, IDSSE-only training, every 10th graded frame,
teams swapped, the same trees and decoder grid. The signal is real: on J03WR9 the 4-second centroid
shift, home minus away, averages +7.3 m while home has the ball and −10.6 m while away does.

## Results

| | start alone, against robust | the full chain, against v12 |
|---|---|---|
| all 10 | +0.0049, 7/10, **passes** (IDSSE +0.0011, Metrica +0.0139) | +0.0031, 5/10, **fails** (IDSSE −0.0014 2/7, Metrica +0.0135 3/3) |
| IDSSE set | −0.0028, 3/7, **fails** | −0.0044, 4/7, **fails** |

Robust's features already carry most of it (the teams' shape and position, PathCRF, the costs), and
the chain's later stages take the rest. Like three candidates before it, it helps Metrica's matches and
not IDSSE's (see lab.md, round 6).
