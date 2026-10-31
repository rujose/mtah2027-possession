"""Damage a clean IDSSE match so its tracking looks like ours (docs/bench.md, "Damage").

Every fault is imitated from bench/damage_stats.json, measured on our own input tracking by
bench/measure.py. Seeded per match and level, so every run sees exactly the same damage.

Levels are the share of frames without a ball: 0 (no damage at all: the clean reference), 30, 41
(ours) and 55. The ball's losses are scaled up or down to reach a level; everything else is
imitated as measured.
"""

import json
import os
import zlib

import numpy as np

from bench.measure import edge_zone, stopping_soon
from possession import v1
from possession.data import AWAY, HOME, ROOT

# Params
STATS = os.path.join(ROOT, "bench", "damage_stats.json")
LEVELS = (0, 30, 41, 55)
PLAYERS_ON_PITCH = 22


def load_stats():
    with open(STATS) as handle:
        return json.load(handle)


def rng_for(match, level, part):
    """One independent random stream per match, level and damage step."""
    return np.random.default_rng([level, zlib.crc32(match.encode()), zlib.crc32(part.encode())])


# --------------------------------------------------------------------------------------------
# Players: anonymous, in pieces, spread over our number of slots
# --------------------------------------------------------------------------------------------

def fragment(data, stats, rng):
    """Cut every player's track into pieces with our lengths, separated by holes, and deal the
    pieces into a pool of anonymous slots per team (our slot counts).

    Hole length: geometric with the mean that makes the visible share of players equal ours,
    mean piece / (mean piece + mean hole) = players visible / 22.
    """
    pieces_from = np.array(stats["fragment_frames"])
    visible_share = stats["players_visible_mean"] / PLAYERS_ON_PITCH
    mean_hole = pieces_from.mean() * (1 / visible_share - 1)
    slots = {HOME: stats["slots"]["home"], AWAY: stats["slots"]["away"]}

    players, speed, team = data["players"], data["speed"], data["team"]
    n_frames = len(players)
    pieces = {HOME: [], AWAY: []}   # (start, stop, person)
    for _, first, last in data["halves"]:
        for person in range(players.shape[1]):
            on = ~np.isnan(players[first:last + 1, person, 0])
            for start, stop in zip(*v1.runs(on)):
                cursor = first + start
                while cursor < first + stop:
                    end = min(cursor + int(rng.choice(pieces_from)), first + stop)
                    pieces[team[person]].append((cursor, end, person))
                    cursor = end + int(rng.geometric(1 / (mean_hole + 1)))

    out_players = np.full((n_frames, slots[HOME] + slots[AWAY], 2), np.nan, dtype=np.float32)
    out_speed = np.full((n_frames, slots[HOME] + slots[AWAY]), np.nan, dtype=np.float32)
    out_team = np.array([HOME] * slots[HOME] + [AWAY] * slots[AWAY], dtype=np.int8)
    for side, offset in ((HOME, 0), (AWAY, slots[HOME])):
        free_at = np.full(slots[side], -1)
        for start, stop, person in sorted(pieces[side]):
            # the slot that has been free longest; a busy pool means our slot count is too small
            slot = int(np.argmin(free_at))
            if free_at[slot] > start:
                raise RuntimeError("more simultaneous pieces than slots")
            free_at[slot] = stop
            out_players[start:stop, offset + slot] = players[start:stop, person]
            out_speed[start:stop, offset + slot] = speed[start:stop, person]
    return out_players, out_speed, out_team


def blank_speed(speed, stats, rng):
    """Our file leaves a player's speed empty on some frames where the position exists."""
    blank = rng.random(speed.shape) < stats["player_speed_missing_share"]
    return np.where(blank, np.nan, speed).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Ball: a seen <-> missing chain with our chances, then our jitter
# --------------------------------------------------------------------------------------------

class BallChain:
    """Everything about the ball's losses that does not depend on the scale, computed once.

    Chance of losing a seen ball on the next frame: ours, by crowd size (players of the DAMAGED
    tracking within the crowd radius of the true ball) and by whether players stop soon (v1's own
    'stopped' clue within the next few seconds, on the damaged tracking, as on ours), times how much likelier ours is to
    lose the ball beyond or near a line than on average (taken as independent of the other two).
    Chance of finding a missing ball: ours, by 'stopped'. `simulate(scale)` multiplies the losing
    chance by `scale`.
    """

    def __init__(self, ball, players, speed, halves, pitch, stats, rng):
        visible = np.array(stats["loss_visible_frames"], dtype=float)
        lose = np.divide(stats["loss_events"], visible, out=np.zeros(visible.shape), where=visible > 0)
        find = np.array(stats["found_events"], dtype=float) / np.array(stats["missing_frames"])
        bins = lose.shape[1]

        stopped, _ = v1.clue_team_speed({"speed": speed})
        distance = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))
        with np.errstate(invalid="ignore"):
            crowd = np.minimum((distance <= stats["crowd_radius_m"]).sum(axis=1), bins - 1)
        edge_visible = np.array(stats["loss_visible_frames_by_edge"], dtype=float)
        edge_lost = np.array(stats["loss_events_by_edge"], dtype=float)
        edge_factor = (edge_lost / edge_visible) / (edge_lost.sum() / edge_visible.sum())
        soon = stopping_soon(stopped).astype(int)
        self.lose = lose[soon, crowd] * edge_factor[edge_zone(ball, pitch)]
        self.find = find[stopped.astype(int)]
        self.draw = rng.random(len(ball))
        self.halves = halves

    def simulate(self, scale):
        """Boolean [frames]: True where the ball is missing. Every half starts with a seen ball."""
        missing = np.zeros(len(self.draw), bool)
        lose = np.minimum(self.lose * scale, 1.0).tolist()
        find, draw = self.find.tolist(), self.draw.tolist()
        for _, first, last in self.halves:
            gone = False
            for t in range(first + 1, last + 1):
                gone = draw[t] >= find[t - 1] if gone else draw[t] < lose[t - 1]
                missing[t] = gone
        return missing


def prepare(match, data, level, stats):
    """The damaged players and the ball chain for one match and level (scale applied later)."""
    if level == 0:
        return None
    players, speed, team = fragment(data, stats, rng_for(match, level, "fragment"))
    speed = blank_speed(speed, stats, rng_for(match, level, "speed"))
    chain = BallChain(data["ball"], players, speed, data["halves"], data["pitch"], stats,
                      rng_for(match, level, "ball"))
    # Metrica's samples already blank the ball in dead time: those frames count as missing too.
    already = np.isnan(data["ball"][:, 0])
    return {"players": players, "speed": speed, "team": team, "chain": chain, "already": already}


def in_halves(mask, halves):
    """The part of a per-frame mask that falls inside the halves."""
    keep = np.zeros(len(mask), bool)
    for _, first, last in halves:
        keep[first:last + 1] = True
    return mask & keep


def calibrate(prepared, level, iterations=30):
    """The scale on the losing chance that makes `level` % of all frames (pooled over matches)
    miss the ball. Bisection on log(scale); the random draws are fixed, so this is repeatable."""
    if level == 0:
        return None
    low, high = np.log(1e-2), np.log(1e2)
    for _ in range(iterations):
        middle = (low + high) / 2
        missing = sum(int(in_halves(p["chain"].simulate(np.exp(middle)) | p["already"],
                                    p["chain"].halves).sum()) for p in prepared)
        frames = sum(sum(last - first + 1 for _, first, last in p["chain"].halves) for p in prepared)
        if missing / frames < level / 100:
            low = middle
        else:
            high = middle
    return float(np.exp((low + high) / 2))


def tracking(match, data, level, prepared, scale, stats):
    """Tracking dict in the shape of possession.data.load_tracking, damaged to `level`."""
    if level == 0:
        return {"players": data["players"], "speed": data["speed"], "ball": data["ball"],
                "team": data["team"]}
    ball = data["ball"].copy()
    ball[prepared["chain"].simulate(scale)] = np.nan
    jitter = rng_for(match, level, "jitter").normal(0.0, stats["ball_noise_m"], ball.shape)
    return {"players": prepared["players"], "speed": prepared["speed"],
            "ball": (ball + jitter).astype(np.float32), "team": prepared["team"]}


def describe(level=41):
    """Compare the damaged ball with ours: missing share while players are stopped or moving,
    and gap lengths. Printed by `python -m bench.degrade`."""
    from bench import idsse
    stats = load_stats()
    matches = idsse.matches()
    data = {m: idsse.load(m) for m in matches}
    prepared = {m: prepare(m, data[m], level, stats) for m in matches}
    scale = calibrate(list(prepared.values()), level)
    stopped_missing, moving_missing, gaps = [], [], []
    for m in matches:
        chain = prepared[m]["chain"]
        missing = chain.simulate(scale)
        stopped, _ = v1.clue_team_speed({"speed": prepared[m]["speed"]})
        for _, first, last in data[m]["halves"]:
            span = slice(first, last + 1)
            stopped_missing.append(missing[span][stopped[span]])
            moving_missing.append(missing[span][~stopped[span]])
            starts, ends = v1.runs(missing[span])
            gaps += (ends - starts).tolist()
    gaps = np.array(gaps)
    ours = np.array(stats["gap_frames"]["moving"] + stats["gap_frames"]["stopped"])
    print("level {}%, scale {:.3f}".format(level, scale))
    print("  ball missing while stopped: bench {:.1%}, ours {:.1%}".format(
        np.concatenate(stopped_missing).mean(), stats["ball_missing_share_stopped"]))
    print("  ball missing while moving:  bench {:.1%}, ours {:.1%}".format(
        np.concatenate(moving_missing).mean(), stats["ball_missing_share_moving"]))
    print("  gaps per 90 min: bench {:.0f}, ours {:.0f}".format(
        len(gaps) / sum(sum(b - a + 1 for _, a, b in data[m]["halves"]) for m in matches) * 135000,
        len(ours) / 158857 * 135000))
    print("  gap length median / mean / p90 (s): bench {:.1f} / {:.1f} / {:.1f}, ours {:.1f} / {:.1f} / {:.1f}".format(
        *(np.array([np.median(gaps), gaps.mean(), np.percentile(gaps, 90)]) / 25),
        *(np.array([np.median(ours), ours.mean(), np.percentile(ours, 90)]) / 25)))


if __name__ == "__main__":
    describe()
