# Lab candidate `temporal`: a bidirectional GRU over robust's features

| | |
|---|---|
| **status** | **built 2026-11-09, lab round 7. The `average` variant clears the round-7 bar on both sets; the network alone doesn't.** Start against robust: `temporal` **0.7097 (+0.0369, 9/10)**, IDSSE set 0.6611 (+0.0247, 7/7); `average` 0.7030 (+0.0301, 10/10), IDSSE set 0.6635 (+0.0271, 7/7). v12's chain from it against v12: `average` **0.7162 (+0.0129, 8/10)**, IDSSE set **0.6760 (+0.0061, 5/7)**, every check passing; `temporal` 0.7200 (+0.0167, 9/10) but IDSSE set 0.6771 (+0.0072, **4/7**, fails the share) |
| **what** | a small temporal network in place of robust's trees: a 2-layer bidirectional GRU reads robust's 97 blanked features every 5th frame (5 Hz) and gives P(H / A / D) for every frame; decoded with robust's decoder grid. Two variants, both fixed before any run: **`temporal`**, the network's P; **`average`**, the mean of the network's P and robust's |
| **why** | the leading public entries (round 6's survey) put a temporal deep model (bidirectional GRUs, temporal CNNs seeing about ±12 s) over per-frame features, then Viterbi. Robust is a per-frame tree model whose only context is hand-made 1 s and 3 s windows |
| **stage** | `decode`: a new start, judged against robust, then as the start of v12's chain against v12 |
| **code** | `bench/lab_temporal.py` (new; reads robust's cached features and held-out P, changes nothing) |
| **run** | `OMP_NUM_THREADS=2 python -m bench.lab_temporal --train [--set idsse]` (one network per fold, held-out P saved), then `python -m bench.lab_temporal [--set idsse]` (decoded and judged against robust), then `python -m bench.lab_temporal --chains [--set idsse]` (v12's chain from each variant, against v12) |
| **results** | `bench/results/lab_temporal[_idsse].json` and `lab_temporal_average[_idsse].json` (each judged against robust), `lab_temporal_grid[_idsse].csv`, `lab_temporal_average_grid[_idsse].csv`, `lab_temporal_frames[_idsse].csv` (per fold: training time, held-out log loss and argmax accuracy against robust's), `lab_temporal_chains[_idsse].json`; held-out P in `data/external/lab_temporal[_idsse]/p_<match>.npy`, each fold's weights in `net_<match>.pt` (gitignored) |

## How it works

1. **Input:** robust's 97 features on the blanked tracking (`lab_robust.cached`), as is and with the
   teams exchanged. Standardised with the mean and sd of the fold's training matches (both
   orientations, every frame inside the halves), clipped at ±5 sd. Missing values become 0, and each
   feature missing on more than 0.1 % of the training frames gets a missing flag (44 flags: the
   ball's family, PathCRF's, the restart clue, the next seen ball): 141 inputs.
2. **5 Hz, five phases:** the network steps every 5th frame (0.2 s). Each of the 5 phases (frames
   o, o + 5, o + 10, ... within a half) is its own sequence, so every frame gets its own P and nothing
   is upsampled. Training draws one phase at random for each half and pass.
3. **The network:** a linear layer to 64 (ReLU, dropout 0.2), a 2-layer bidirectional GRU with 64
   units each way (dropout 0.2 between layers), a linear layer to 3 classes; ~140k weights.
4. **Training:** IDSSE matches only (as robust: Metrica's samples blank the ball in dead time,
   learned.md), each half in both orientations (labels H ↔ A), cut into 60 s sequences (300 steps)
   at a random offset, 16 a batch, cross-entropy on the graded frames, AdamW (learning rate 1e-3,
   cosine decay to 0, weight decay 1e-4), gradients clipped at 1, **20 passes**, seed 0, 2 threads,
   deterministic algorithms. Leave one match out; Metrica's 3 share the network trained on all 7 IDSSE
   matches, as robust's.
5. **Prediction:** each phase read in 300-step windows every 150 steps, overlaps averaged with
   triangular weights (so the network always reads a sequence as long as it was trained on); P is
   the mean of the match as it is and with the teams exchanged, clipped to 0.01–0.99 and rounded to
   3 decimals, as robust's.
6. **`average`:** the mean of this P and robust's held-out P (`blank`), clipped and rounded the same.
7. **Decoded** over robust's decoder grid (team switch 0.5–16, dead switch 50, 25, 10, minimum 20 or
   10 frames; no blend), then v9's snap and v11's move (`L.finish`). Each variant judged on its own
   against robust (`lab.judge`, base `lab_weighted.robust_base`), then as the start of v12's chain
   against v12, every stage refitted on its output, each on its setting for that match
   (`lab_mix.fold_settings`).

## Every setting, and why

| setting | value | why |
|---|---|---|
| step | every 5th frame (5 Hz) | the brief; ±60 s of context is 300 steps, cheap for a GRU on CPU |
| phases | all 5, each its own sequence | gives every frame its own P (no repeat or interpolation) and five times the distinct training sequences |
| sequence | 60 s (300 steps) in training and prediction | long enough for a possession and its neighbours; the same length both ways, so a GRU never reads longer than it learned |
| network | 64 → BiGRU 2 × 64 → 3 | the brief's size; small against ~6 matches of training data |
| missing | 0 plus a flag | the brief; 0 is the training mean after standardising |
| clip | ±5 sd | a few long-tailed inputs (seconds since the ball was seen, its course) mustn't dominate |
| optimiser | AdamW 1e-3, cosine to 0, weight decay 1e-4, clip 1 | standard for a small recurrent network; the cosine decay ends on a settled model without early stopping (which would need a held-out match) |
| **passes** | **20** | fixed from a one-fold timing test that read only the training loss: 9–10 s a pass, so ~3.5 min a fold, inside the ~5 min the brief allows. With four other builders on the machine the folds took 3.2–8.5 min (mean 6.2 min on `all`, 4.4 min on `idsse`; 49 and 31 min in all); the result doesn't depend on the speed |
| seed, threads | 0, 2; `torch.use_deterministic_algorithms(True)` | repeatable on one machine: a pass repeated gave the same training loss to 4 decimals |
| variants | `temporal`, `average`, judged separately | fixed before the run; the average is the usual guard when a new model is about as good as the old one but wrong in different places |

## Results

### The start alone, against robust

Each variant decoded over robust's grid, each match on the setting best on the others.

| | `temporal`, all 10 | `average`, all 10 | `temporal`, IDSSE set | `average`, IDSSE set |
|---|---|---|---|---|
| held out (robust 0.6728; IDSSE set 0.6365) | **0.7097 (+0.0369)** | 0.7030 (+0.0301) | **0.6611 (+0.0247)** | 0.6635 (+0.0271) |
| matches better | 9/10 | 10/10 | 7/7 | 7/7 |
| IDSSE's matches | +0.0246, 6/7 | +0.0216, 7/7 | +0.0247, 7/7 | +0.0271, 7/7 |
| Metrica's matches | +0.0656, 3/3 | +0.0501, 3/3 | — | — |
| PathCRF-unseen | +0.0571, 4/4 | +0.0419, 4/4 | +0.0206, 1/1 | +0.0140, 1/1 |
| macroF1 (robust 0.9145; 0.9115) | 0.9198 | 0.9202 | 0.9174 | 0.9181 |
| possessions a match (true 411; 437) | 340 | 336 | 345 | 351 |
| bar | **passes** | **passes** | **passes** | **passes** |

Per match, `temporal` on all 10: J03WMX +0.041, J03WN1 +0.020, J03WOH +0.012, **J03WOY −0.003**,
J03WPY +0.026, J03WQQ +0.045, J03WR9 +0.032, metrica1 +0.073, metrica2 +0.060, metrica3 +0.064. The
folds picked a lower team-switch price than robust's (2 in 8 of 10 folds; robust's is 4) and the
minimum of 10 frames: the network's P already changes team less often, so the decoder needs to
charge less for it. **The start alone (0.7097) is above v12's whole chain (0.7033).**

### Where it comes from

Held out, frame by frame (truth read only to report; `lab_temporal_frames[_idsse].csv`):

| | log loss: temporal / average / robust | argmax accuracy: temporal / average / robust |
|---|---|---|
| IDSSE's 7 (all) | 0.190 / 0.184 / 0.199 | 92.9 % / 92.9 % / 92.4 % |
| Metrica's 3 | 0.272 / 0.272 / 0.292 | 92.4 % / 92.5 % / 91.7 % |
| IDSSE set | 0.187 / 0.183 / 0.198 | 92.9 % / 92.9 % / 92.4 % |

Lower log loss on 9 of 10 (J03WN1 level: 0.2345 against 0.2339), and on all 7 on the IDSSE set. The frames
improve a little; the possessions improve a lot. The network's P is coherent in time: its argmax alone
(no decoder) makes 625–843 possessions a match against robust's 828–1161 (true 318–511), and scores
higher spellF1 undecoded on every match (e.g. J03WQQ 0.511 against 0.378). The decoder then has less
flicker to remove, and what it keeps has better edges. A per-frame model can only see that context
through hand-made 1 s and 3 s windows; the GRU reads ±60 s.

### v12's chain from each variant, against v12

`python -m bench.lab_temporal --chains [--set idsse]`: v12's post stages (short → segments → timing on
`all`; short → segments → recall → dead on `idsse`), every stage refitted on each start's out-of-fold
letters, each on its setting for that match. v12's own chain, computed in the same run, came out at
0.7033 and 0.6699, as the lab records.

| against v12 | `temporal`, all 10 | `average`, all 10 | `temporal`, IDSSE set | `average`, IDSSE set |
|---|---|---|---|---|
| held out (v12 0.7033; 0.6699) | **0.7200 (+0.0167)** | 0.7162 (+0.0129) | 0.6771 (+0.0072) | 0.6760 (+0.0061) |
| matches better | 9/10 | 8/10 | **4/7** | 5/7 |
| IDSSE's matches | +0.0102, 6/7 | +0.0090, 5/7 | +0.0072, 4/7 | +0.0061, 5/7 |
| Metrica's matches | +0.0320, 3/3 | +0.0219, 3/3 | — | — |
| PathCRF-unseen | +0.0288, 4/4 | +0.0188, 4/4 | +0.0036, 1/1 | +0.0002, 1/1 |
| macroF1 (v12 0.9182; 0.9162) | 0.9216 | 0.9222 | 0.9205 | 0.9194 |
| four checks, gain ≥ +0.005 | **pass** | **pass** | **fails** (share: 4/7, needs 5) | **pass** (narrowly) |

Per match, chain against v12:

| | J03WMX | J03WN1 | J03WOH | J03WOY | J03WPY | J03WQQ | J03WR9 | metrica1 | metrica2 | metrica3 |
|---|---|---|---|---|---|---|---|---|---|---|
| `temporal`, all | +0.028 | +0.011 | +0.007 | **−0.024** | +0.004 | +0.026 | +0.019 | +0.042 | +0.026 | +0.028 |
| `average`, all | +0.027 | +0.021 | +0.005 | −0.007 | −0.001 | +0.009 | +0.010 | +0.027 | +0.009 | +0.030 |
| `temporal`, IDSSE set | +0.059 | −0.009 | −0.010 | **−0.026** | +0.002 | +0.031 | +0.004 | | | |
| `average`, IDSSE set | +0.045 | −0.005 | +0.007 | −0.024 | +0.003 | +0.016 | +0.000 | | | |

## Verdict

**`average` clears the round-7 bar on both sets; `temporal` alone does not.** On all 10 both
variants pass every check with room (+0.0167 and +0.0129). On the IDSSE set the network alone gains
+0.0072 but improves only 4 of 7 matches. The average improves 5 of 7, by +0.0061, with the
PathCRF-unseen match level (+0.0002). By the protocol that is a pass, but every margin on the IDSSE
set is thin: +0.0061 against the +0.005 floor, 5 of 7 against the 5-of-7 floor, the unseen match
+0.0002.

What to weigh before mixing it in:

1. **Two variants were tried**, so the better of them is slightly flattered (lab.md, "Many tries,
   one bench"). Both were fixed before the run, and both pass on all 10. The network alone, the
   stronger on all 10, is the one that fails on the IDSSE set.
2. **The post stages give back much of the gain.** At the start the network is +0.037 on all and
   +0.025 on the IDSSE set (7/7). After v12's stages it's +0.017 and +0.007. v12's stages add +0.031
   to robust (+0.033 on the IDSSE set) but +0.010 to the network (+0.016): the GRU already does part
   of what short and segments do. On the IDSSE set the stages' settings, chosen on robust's letters,
   turn a 7/7 start into a 4/7 chain. Those settings were not chosen again here; a fair next test
   is a greedy chain from this start (round 3's procedure), which may well drop a stage.
3. **J03WOY loses in every chain** (−0.024 and −0.026 with the network alone), though its start is
   level (−0.003 on all, +0.010 on the IDSSE set): the post stages undo something the network got
   right there. Not looked into.
4. **It leans toward Metrica, like rounds 5-6, but not only:** the start gains +0.066 on Metrica
   against +0.025 on IDSSE; the chain +0.032 against +0.010. Unlike motion, weighted or switch, it
   also gains on IDSSE's matches at every step on all 10 (start 6/7 and 7/7, chain 6/7 and 5/7). Its training is IDSSE-only,
   so the Metrica gain isn't learned from Metrica's labels.
5. **The frozen-ball risk is robust's, unchanged:** the network reads robust's blanked features, so
   a still ball doesn't exist for it either. With ±60 s of context it can learn longer patterns of
   "ball missing", and IDSSE's dead time is mostly missing ball after the blanking: a stoppage in
   our match where the ball keeps being seen and moving would fool it as it fools robust.
6. **Reproducibility.** PyTorch on the CPU with a fixed seed, 2 threads and
   `torch.use_deterministic_algorithms(True)`: J03WR9's fold retrained from scratch (on a busy machine)
   gave the same final training loss (0.16694) and the same P on every frame after rounding (largest
   difference under 1e-6, on a 0.001 grid). Another machine, another PyTorch
   version or another thread count may change the last digits of P; the 3-decimal rounding absorbs
   most of that, as it does for robust's trees. Each fold's weights are saved (`net_<match>.pt`), so
   the held-out P can be recomputed without training.

## Using it

- Held-out P on a bench match: `bench.lab_temporal.held_out(match, which, variant)` (`"temporal"` or
  `"average"`; the average adds `lab_robust.held_out_proba(match, which, "blank")`).
- Letters at v11's level: `bench.lab_temporal.letters_of(match, which, variant, setting)` (decode, then
  v9's snap and v11's move). Each fold's setting is in `bench/results/lab_temporal[_average][_idsse].json`;
  chosen on all matches: `average` team 2, dead 25, minimum 10 on `all`; team 2, dead 50, minimum 10
  on `idsse` (`temporal`: team 2, dead 25, minimum 10; team 4, dead 25, minimum 10).
- v12's chain from it: `bench.lab_temporal.chains(which)` (as `lab_motion.chains`).
- A new match: `fit(idsse_matches, which)` (or the saved `net_metrica1.pt`, trained on all 7 IDSSE
  matches, with `normaliser(idsse_matches, which)`), then `proba(match_id, model, which)` on robust's
  cached features for that match. Nothing here has been run on our match.
