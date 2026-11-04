"""Model v10: v9 made clean for the organisers' clarification of 2026-11-03.

    python -m possession.v10           # Jose runs it on our match: writes evaluation/results/v10.csv

Full explanation: docs/models/v10-clean-base.md. The same pieces as v9, composed here with their
settings in one table, so the bench matches they were chosen on can be switched:
  - no referee whistle (its thresholds were set on our match's audio) and no set-piece window (set
    from the public ground truth); Smart Tagging's team hints and the 2nd-half kick-off stay;
  - the keeper repair (v8), the dead-ball model (v4), the restart model (v6), PathCRF (v7) and the
    restart snap (v9), with the settings chosen on the bench matches in BENCH_SET.
Without audio and set-piece tags (as on the bench) and with BENCH_SET = "all", this is exactly v9
(`python -m bench.check_v10`).
"""

import hashlib
import os

import numpy as np

from possession import dead_ball, features, pathcrf_adapter, repair, restart, v1, v3, v4, v6, v7, v9
from possession.data import AWAY, FPS, HOME, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking
from possession.decoder import viterbi_min_length

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v10.csv")
BENCH_SET = "all"          # "idsse" if Metrica's samples can't be used (bench/chain.py)
RESTART_SNAP = True        # v9's snap; False if the organisers rule out where the idea came from

# Per bench set: the models and settings each stage chose (bench/results/*_choice[_idsse].json).
SETTINGS = {
    "all": {
        "dead_ball": (v4.FAMILY, "dead_ball"), "w_dead_ball": v4.W_DEAD_BALL,
        "switch_dead": v4.SWITCH_DEAD, "min_dead": v4.MIN_DEAD,
        "restart": (v6.FAMILY, "restart"), "w_restart": v6.W_RESTART, "carry": v6.CARRY,
        "w_pathcrf": v7.W_PATHCRF, "where": v7.WHERE,
        "back": v9.BACK, "ahead": v9.AHEAD,
    },
    # The 7 IDSSE matches alone (CC BY 4.0), from `python -m bench.<stage> --set idsse`:
    # dead ball unchanged (held out 0.399); restart carry 5 s (0.411, 5/7); PathCRF weight 1.0, chosen
    # on J03WR9, the only IDSSE match PathCRF never saw, so not held out (0.572); snap unchanged
    # (0.550 against v7's 0.548, 5/7).
    "idsse": {
        "dead_ball": ("trees", "dead_ball_idsse"), "w_dead_ball": 0.5, "switch_dead": 50.0, "min_dead": 1,
        "restart": ("logistic", "restart_idsse"), "w_restart": 1.0, "carry": 5.0,
        "w_pathcrf": 1.0, "where": "live",
        "back": 5, "ahead": 1,
    },
}
# The team's decoder settings and the tag weight: v3's, chosen on the 7 IDSSE matches alone.
SWITCH_TEAM = v3.SWITCH_TEAM
MIN_POSSESSION = v3.MIN_POSSESSION
W_TAG_TEAM = v3.W_TAG_TEAM


def base_costs(clues):
    """v1's clues at v4's weights, without "stopped"/"active" (replaced by the dead-ball model), the
    whistle and the set-piece window. What stays: on the ball, someone on the ball, ball out, before
    the 2nd-half kick-off, Smart Tagging's team hints."""
    off = {name: np.zeros_like(clues[name]) for name in ("stopped", "active", "set_piece")}
    cost = v1.build_costs(dict(clues, **off))
    cost[:, v1.A] += (W_TAG_TEAM - v1.W_TAG_TEAM) * clues["tag_supports"][:, HOME]
    cost[:, v1.H] += (W_TAG_TEAM - v1.W_TAG_TEAM) * clues["tag_supports"][:, AWAY]
    return cost


def decode(cost, halves, setting):
    switch = np.zeros((3, 3))
    switch[v1.H, v1.A] = switch[v1.A, v1.H] = SWITCH_TEAM
    switch[v1.H, v1.D] = switch[v1.A, v1.D] = setting["switch_dead"]
    switch[v1.D, v1.H] = switch[v1.D, v1.A] = setting["switch_dead"]
    labels = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in halves:
        path = viterbi_min_length(cost[first:last + 1], switch, [MIN_POSSESSION, MIN_POSSESSION, setting["min_dead"]])
        labels[first:last + 1] = v1.LABELS[path]
    return labels


def add_team_clue(cost, belief, weight):
    on_h, on_a = restart.costs(belief, weight)
    cost[:, v1.H] += on_h
    cost[:, v1.A] += on_a


def predict(tracking, tags, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH), bench_set=None):
    """(labels for every video frame, what went into them: keeper pieces, snap moves, P(dead),
    P(home restarts), PathCRF's P(home), the repaired tracking)."""
    halves = halves or graded_halves()
    setting = SETTINGS[bench_set or BENCH_SET]
    if setting is None:
        raise ValueError("no settings for bench set {!r} yet".format(bench_set or BENCH_SET))
    fixed, pieces = repair.repair(tracking, halves, pitch)                        # v8
    clues = v1.compute_clues(fixed, tags, halves, pitch)
    cost = base_costs(clues)

    names, X = features.frame_features(fixed, halves, pitch)                        # v4
    model = dead_ball.load(setting["dead_ball"][0], name=setting["dead_ball"][1])
    if model["features"] != names:
        raise ValueError("the saved model's features differ from possession/features.py")
    p_dead = dead_ball.predict(model, X)
    alive, dead = dead_ball.costs(p_dead)
    cost[:, v1.H] += setting["w_dead_ball"] * alive
    cost[:, v1.A] += setting["w_dead_ball"] * alive
    cost[:, v1.D] += setting["w_dead_ball"] * dead

    start_model = dead_ball.load(setting["restart"][0], name=setting["restart"][1])  # v6
    p_start = restart.p_home(start_model, restart.all_orientations(fixed, halves, pitch))
    add_team_clue(cost, restart.restart_belief(p_start, p_dead, halves, int(setting["carry"] * FPS)), setting["w_restart"])

    live = np.isin(decode(cost, halves, setting), ["H", "A"])                      # v7
    p_pathcrf = pathcrf_adapter.p_home(pathcrf_adapter.load_model(), fixed, halves, pitch, live)
    belief = np.round(p_pathcrf, 3)
    if setting["where"] == "gaps":
        belief[~np.isnan(fixed["ball"][:, 0])] = np.nan
    add_team_clue(cost, belief, setting["w_pathcrf"])
    labels = decode(cost, halves, setting)

    moves = []
    if RESTART_SNAP:                                                                # v9
        labels, moves = v9.snap(labels, fixed, halves, setting["back"], setting["ahead"])
    return labels, {"pieces": pieces, "moves": moves, "p_dead": p_dead, "p_start": p_start, "p_pathcrf": p_pathcrf,
                    "fixed": fixed}


def main():
    """Only what Jose needs to check the file: no statistics about the match (the clarification's
    "no provided data to AI services": this output is never pasted to Claude)."""
    labels, _ = predict(load_tracking(), load_tags())
    v1.write_submission(OUTPUT, labels)
    with open(OUTPUT, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    path = os.path.relpath(OUTPUT, ROOT)
    print("wrote {} (bench set {}, restart snap {})".format(path, BENCH_SET, "on" if RESTART_SNAP else "off"))
    print("SHA-256 {}".format(digest))
    print("format check: .venv/bin/python evaluation/validate_submission.py {}".format(path))


if __name__ == "__main__":
    main()
