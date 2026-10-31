"""Model v1: per-frame clues + a Viterbi decider that charges for switching label.

    python -m possession.v1            # writes evaluation/results/v1.csv and prints diagnostics

Full explanation, with the reasoning behind every setting below: docs/models/v1-clues-and-decider.md

Costs are in "frames of evidence": a clue at weight 1.0 adds 1 to the cost of every label it
argues against, on every frame it is active. A switch penalty of 25 therefore means "changing label
needs about one second (25 frames) of clear evidence against the current one".
"""

import os

import numpy as np

from possession.data import (AWAY, FPS, HOME, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves,
                             load_tags, load_tracking)
from possession.decoder import viterbi

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v1.csv")
LABELS = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2

# Piece 2: ball cleaning
JUMP_SPEED = 40.0          # m/s. Faster than the hardest shots (~35-40 m/s): a re-acquisition glitch.

# Clue A: who is on the ball
ON_BALL_DISTANCE = 2.0     # m. One stride (~1-1.5 m) plus tracking error.
FLIGHT_SPEED = 8.0         # m/s. Players dribble at up to ~7-8 m/s; passes travel at 10-30 m/s.
W_ON_BALL = 1.0            # against the other team, scaled by how clearly this team is closer
W_ON_BALL_ALIVE = 0.5      # against D: someone is playing a slow ball inside the pitch

# Clue B: is the ball dead or alive
OUT_MARGIN = 1.0           # m beyond a line before the ball counts as out (absorbs tracking error)
W_OUT = 2.0                # against H and A
STOPPED_SPEED = 1.0        # m/s, mean speed of all visible players over 1 s: strolling pace
STOPPED_MIN = 3 * FPS      # frames the team must stay that slow before it counts
W_STOPPED = 1.0            # against H and A
ACTIVE_SPEED = 2.0         # m/s, same mean: clearly above a stroll
W_ACTIVE = 0.5             # against D
SET_PIECE_WINDOW = 15 * FPS  # frames after a set-piece tag starts (tags start 2.8-12.2 s early)
W_SET_PIECE = 0.3          # against H and A; weak, 2 of 5 restarts in the public window had no tag
W_BEFORE_KICKOFF = 1000.0  # against H and A, before the 2nd-half kick-off tag: certain

# Clue C: who is attacking, from Smart Tagging
WITH_BALL_CODES = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION"}
WITHOUT_BALL_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                      "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
TEAM_NAMES = {"BARCELONA": HOME, "ATLETIC CLUB": AWAY}
W_TAG_TEAM = 0.2           # against the other team; weak, fixed-length windows and missed events

# Piece 4: switch penalties, in frames of evidence
SWITCH_TEAM = 15.0         # H <-> A: ~0.6 s of one-sided control
SWITCH_TO_DEAD = 25.0      # H/A -> D: ~1 s of dead evidence
SWITCH_FROM_DEAD = 25.0    # D -> H/A: ~1 s of alive evidence


def runs(mask):
    """(starts, ends) of the runs of True in a boolean array; ends are exclusive."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.where(edges == 1)[0], np.where(edges == -1)[0]


def rolling_mean(values, window):
    """Centred mean over `window` frames, ignoring NaN; NaN where the whole window is NaN."""
    valid = ~np.isnan(values)
    kernel = np.ones(window)
    total = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    count = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


# --------------------------------------------------------------------------------------------
# Piece 2: clean the ball
# --------------------------------------------------------------------------------------------

def ball_speed(ball):
    """m/s from consecutive positions; NaN where unknown or where the step is a glitch."""
    step = np.r_[np.nan, np.hypot(*np.diff(ball, axis=0).T) * FPS]
    step[step > JUMP_SPEED] = np.nan
    return step


def ball_out(ball, pitch=(PITCH_LENGTH, PITCH_WIDTH)):
    """True where the ball is visible and more than OUT_MARGIN beyond a pitch line."""
    x, y = ball[:, 0], ball[:, 1]
    length, width = pitch
    with np.errstate(invalid="ignore"):
        return ((x < -OUT_MARGIN) | (x > length + OUT_MARGIN)
                | (y < -OUT_MARGIN) | (y > width + OUT_MARGIN))


# --------------------------------------------------------------------------------------------
# Piece 3: clues
# --------------------------------------------------------------------------------------------

def clue_on_ball(tracking, speed, out):
    """Clue A. Returns (strength [frames, 2] per team in 0..1, someone_on_ball [frames])."""
    players, team, ball = tracking["players"], tracking["team"], tracking["ball"]
    distance = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))  # [frames, slots]
    nearest = np.stack([np.nanmin(np.where(team == side, distance, np.inf), axis=1)
                        for side in (HOME, AWAY)], axis=1)                  # inf = nobody seen
    nearest[np.isnan(ball[:, 0])] = np.inf

    with np.errstate(invalid="ignore"):
        slow = speed < FLIGHT_SPEED          # NaN speed (unknown) is not slow
    strength = np.zeros((len(ball), 2))
    for side in (HOME, AWAY):
        own, other = nearest[:, side], nearest[:, 1 - side]
        active = slow & (own <= ON_BALL_DISTANCE) & (own < other)
        with np.errstate(invalid="ignore"):  # inf - inf where nobody is seen; never active there
            gap = np.where(np.isinf(other), ON_BALL_DISTANCE, other - own)
        strength[active, side] = np.clip(gap[active] / ON_BALL_DISTANCE, 0.0, 1.0)
    someone = (strength.max(axis=1) > 0) & ~out
    return strength, someone


def clue_team_speed(tracking):
    """Clue B2 and the 'active' half of B. Returns (stopped [frames], active [frames]).

    'stopped' marks the WHOLE of every stretch where the 1-second mean speed of all visible players
    stays under STOPPED_SPEED for at least STOPPED_MIN frames, including its first 3 seconds.
    """
    speed = tracking["speed"]
    seen = ~np.isnan(speed)
    count = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.where(count > 0, np.where(seen, speed, 0.0).sum(axis=1) / count, np.nan)
    smooth = rolling_mean(mean_speed, FPS)
    with np.errstate(invalid="ignore"):
        slow = smooth < STOPPED_SPEED
        active = smooth > ACTIVE_SPEED
    stopped = np.zeros(len(slow), bool)
    for start, end in zip(*runs(slow)):
        if end - start >= STOPPED_MIN:
            stopped[start:end] = True
    return stopped, active


def clue_tags(tags, n_frames, halves):
    """Clue B3, B4 and C from Smart Tagging.

    Returns (set_piece [frames], before_kickoff [frames], supports [frames, 2]).
    supports[:, team] is True where some tag says that team has the ball, and the other team has no
    such tag at the same time (conflicting tags cancel out).
    """
    set_piece = np.zeros(n_frames, bool)
    before_kickoff = np.zeros(n_frames, bool)
    says = np.zeros((n_frames, 2), bool)
    second_half_start = halves[1][1]
    for tag in tags:
        if tag["code"] == "SET PIECES":
            set_piece[tag["start"]:tag["start"] + SET_PIECE_WINDOW] = True
            if tag.get("Type") == "KICK OFF (START)" and tag["start"] >= second_half_start:
                before_kickoff[second_half_start:tag["start"]] = True
        team = TEAM_NAMES.get(tag.get("Team"))
        if team is None:
            continue
        if tag["code"] in WITH_BALL_CODES:
            says[tag["start"]:tag["end"], team] = True
        elif tag["code"] in WITHOUT_BALL_CODES:
            says[tag["start"]:tag["end"], 1 - team] = True
    supports = says & ~says[:, ::-1]
    return set_piece, before_kickoff, supports


# --------------------------------------------------------------------------------------------
# Piece 4: costs and the decider
# --------------------------------------------------------------------------------------------

def build_costs(clues):
    """Per-frame cost of H, A and D: each clue adds weight to the labels it argues against."""
    n = len(clues["out"])
    cost = np.zeros((n, 3))
    cost[:, A] += W_ON_BALL * clues["on_ball"][:, HOME]
    cost[:, H] += W_ON_BALL * clues["on_ball"][:, AWAY]
    cost[:, D] += W_ON_BALL_ALIVE * clues["someone_on_ball"]
    dead = (W_OUT * clues["out"] + W_STOPPED * clues["stopped"] + W_SET_PIECE * clues["set_piece"]
            + W_BEFORE_KICKOFF * clues["before_kickoff"])
    cost[:, H] += dead
    cost[:, A] += dead
    cost[:, D] += W_ACTIVE * clues["active"]
    cost[:, A] += W_TAG_TEAM * clues["tag_supports"][:, HOME]
    cost[:, H] += W_TAG_TEAM * clues["tag_supports"][:, AWAY]
    return cost


def switch_matrix():
    switch = np.zeros((3, 3))
    switch[H, A] = switch[A, H] = SWITCH_TEAM
    switch[H, D] = switch[A, D] = SWITCH_TO_DEAD
    switch[D, H] = switch[D, A] = SWITCH_FROM_DEAD
    return switch


def compute_clues(tracking, tags, halves, pitch=(PITCH_LENGTH, PITCH_WIDTH)):
    speed = ball_speed(tracking["ball"])
    out = ball_out(tracking["ball"], pitch)
    on_ball, someone = clue_on_ball(tracking, speed, out)
    stopped, active = clue_team_speed(tracking)
    set_piece, before_kickoff, supports = clue_tags(tags, len(out), halves)
    return {"on_ball": on_ball, "someone_on_ball": someone, "out": out, "stopped": stopped,
            "active": active, "set_piece": set_piece, "before_kickoff": before_kickoff,
            "tag_supports": supports}


def predict(tracking, tags, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH)):
    """Labels for every video frame: H/A/D on graded frames, X elsewhere.

    halves and pitch default to BAR-ATH's; the test bench passes another match's (docs/bench.md).
    """
    halves = halves or graded_halves()
    clues = compute_clues(tracking, tags, halves, pitch)
    cost = build_costs(clues)
    switch = switch_matrix()
    labels = np.full(len(cost), "X", dtype="<U1")
    for _, first, last in halves:
        labels[first:last + 1] = LABELS[viterbi(cost[first:last + 1], switch)]
    return labels, clues


# --------------------------------------------------------------------------------------------
# Piece 5: write, and describe what came out (no ground truth involved)
# --------------------------------------------------------------------------------------------

def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def describe(labels, clues):
    graded = labels != "X"
    n = graded.sum()
    print("clue coverage over graded frames:")
    print("  A  someone on the ball          {:5.1f}%".format(100 * clues["on_ball"].max(axis=1)[graded].astype(bool).mean()))
    for name in ("out", "stopped", "active", "set_piece", "before_kickoff"):
        print("  B  {:28s} {:5.1f}%".format(name, 100 * clues[name][graded].mean()))
    print("  C  a tag says Barcelona         {:5.1f}%".format(100 * clues["tag_supports"][graded, HOME].mean()))
    print("  C  a tag says Athletic          {:5.1f}%".format(100 * clues["tag_supports"][graded, AWAY].mean()))
    print("output over graded frames: " + "  ".join(
        "{} {:.1f}%".format(l, 100 * np.mean(labels[graded] == l)) for l in "HAD"))
    possessions = []
    for _, first, last in graded_halves():
        half = labels[first:last + 1]
        for label in "HA":
            starts, ends = runs(half == label)
            possessions += list(ends - starts)
    possessions = np.array(possessions) / FPS
    print("possessions {}  mean {:.1f} s  median {:.1f} s  shorter than 2 s: {}".format(
        len(possessions), possessions.mean(), np.median(possessions), int(np.sum(possessions < 2))))
    alive = np.isin(labels, ["H", "A"])
    print("Barcelona share of ball-alive frames: {:.1f}%".format(100 * np.mean(labels[alive] == "H")))


def main():
    tracking = load_tracking()
    tags = load_tags()
    labels, clues = predict(tracking, tags)
    write_submission(OUTPUT, labels)
    print("wrote " + os.path.relpath(OUTPUT, ROOT))
    describe(labels, clues)


if __name__ == "__main__":
    main()
