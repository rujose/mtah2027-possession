"""Measure the faults of OUR tracking (BAR-ATH, tactical), which bench/degrade.py imitates.

    python -m bench.measure         # writes bench/damage_stats.json and prints it

Input tracking only: no labels, no public ground truth, nothing human-made about the match. The
output holds rates and distributions, never frames (docs/bench.md, decision 4).
"""

import json
import os

import numpy as np

from possession import v1
from possession.data import (AWAY, FPS, HOME, PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves,
                             load_tracking)

# Params
OUTPUT = os.path.join(ROOT, "bench", "damage_stats.json")
CROWD_RADIUS = 3.0       # m: players this close to the ball make it hard to see
CROWD_BINS = 7           # 0, 1, ..., 5, and 6 or more players
NOISE_WINDOW = 5         # frames in the centred mean the ball's jitter is measured against
SLOW_STEP = 0.1          # m per frame (2.5 m/s) of the smoothed ball: "slow", so jitter dominates
EDGE_ZONES = (0.0, 5.0)  # m inside the nearest line: outside the pitch / within 5 m / further in
STOPPING_AHEAD = 5 * FPS  # frames: "players stop soon" looks this far ahead for v1's 'stopped'


def runs(mask):
    starts, ends = v1.runs(mask)
    return list(zip(starts.tolist(), ends.tolist()))


def stopping_soon(stopped):
    """True where v1's 'stopped' holds now or at any frame in the next STOPPING_AHEAD frames.

    Our tracker tends to lose the ball just BEFORE players slow down (it goes out, the whistle
    goes), so the chance of losing it is measured against this rather than against 'stopped'."""
    ahead = np.r_[stopped, np.zeros(STOPPING_AHEAD, bool)]
    count = np.convolve(ahead.astype(int), np.ones(STOPPING_AHEAD + 1, int))[STOPPING_AHEAD:]
    return count[:len(stopped)] > 0


def ball_loss(ball, players, stopped, stopping, halves):
    """The ball as a two-state chain, seen <-> missing, measured frame to frame:

    - losing it: chance a seen ball is missing on the next frame, by crowd size and by whether
      players stop soon (stopping_soon);
    - finding it: chance a missing ball is seen on the next frame, by 'stopped' (while it is
      missing we do not know where it is, so there is no crowd to count).
    Also every gap's length, kept to check the imitation against."""
    seen = ~np.isnan(ball[:, 0])
    distance = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))
    with np.errstate(invalid="ignore"):
        crowd = np.minimum((distance <= CROWD_RADIUS).sum(axis=1), CROWD_BINS - 1)

    visible = np.zeros((2, CROWD_BINS), np.int64)   # [stopped?, crowd] visible frames
    lost = np.zeros((2, CROWD_BINS), np.int64)      # ... followed by a missing frame
    missing = np.zeros(2, np.int64)                 # [stopped?] missing frames
    found = np.zeros(2, np.int64)                   # ... followed by a seen frame
    gaps = {"moving": [], "stopped": []}
    for _, first, last in halves:
        s, b, c = stopped[first:last + 1], seen[first:last + 1], crowd[first:last + 1]
        soon = stopping[first:last + 1]
        now, nxt = b[:-1], ~b[1:]
        np.add.at(visible, (soon[:-1][now].astype(int), c[:-1][now]), 1)
        np.add.at(lost, (soon[:-1][now & nxt].astype(int), c[:-1][now & nxt]), 1)
        np.add.at(missing, s[:-1][~now].astype(int), 1)
        np.add.at(found, s[:-1][~now & ~nxt].astype(int), 1)
        for start, end in runs(~b):
            if start == 0:
                continue  # a half that starts without the ball: no loss to explain
            gaps["stopped" if s[start - 1] else "moving"].append(end - start)
    return visible, lost, missing, found, gaps


def edge_zone(ball, pitch):
    """0 = beyond a line, 1 = within EDGE_ZONES[1] m inside it, 2 = further in. NaN ball -> 2."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        inside = np.minimum.reduce([x, pitch[0] - x, y, pitch[1] - y])
        return np.where(inside < EDGE_ZONES[0], 0, np.where(inside < EDGE_ZONES[1], 1, 2))


def loss_by_edge(ball, halves):
    """Visible frames and losses by how close the ball is to a line: our tracker loses the ball
    more often there (it leaves the pitch, and play stops)."""
    seen = ~np.isnan(ball[:, 0])
    zone = edge_zone(ball, (PITCH_LENGTH, PITCH_WIDTH))
    visible, lost = np.zeros(3, np.int64), np.zeros(3, np.int64)
    for _, first, last in halves:
        s, z = seen[first:last + 1], zone[first:last + 1]
        np.add.at(visible, z[:-1][s[:-1]], 1)
        np.add.at(lost, z[:-1][s[:-1] & ~s[1:]], 1)
    return visible, lost


def ball_noise(ball, halves):
    """Spread (m, per axis) of the ball around its centred NOISE_WINDOW-frame mean, on slow
    stretches, corrected for the mean itself containing the frame (factor sqrt(n / (n - 1)))."""
    residuals = []
    kernel = np.ones(NOISE_WINDOW) / NOISE_WINDOW
    for _, first, last in halves:
        b = ball[first:last + 1].astype(np.float64)
        smooth = np.stack([np.convolve(b[:, k], kernel, mode="same") for k in (0, 1)], axis=1)
        step = np.r_[np.nan, np.hypot(*np.diff(smooth, axis=0).T)]
        with np.errstate(invalid="ignore"):
            slow = step < SLOW_STEP   # NaN anywhere in the window makes the step NaN: excluded
        residuals.append((b - smooth)[slow])
    residuals = np.concatenate(residuals)
    return float(np.std(residuals) * np.sqrt(NOISE_WINDOW / (NOISE_WINDOW - 1))), len(residuals)


def fragments(players, team, halves):
    """Lengths of every unbroken run of a slot, slots per team, players visible per frame."""
    lengths = []
    for _, first, last in halves:
        seen = ~np.isnan(players[first:last + 1, :, 0])
        for slot in range(seen.shape[1]):
            lengths += [end - start for start, end in runs(seen[:, slot])]
    graded = np.concatenate([np.arange(first, last + 1) for _, first, last in halves])
    visible = (~np.isnan(players[graded, :, 0])).sum(axis=1)
    return lengths, {"home": int((team == HOME).sum()), "away": int((team == AWAY).sum())}, visible


def main():
    tracking = load_tracking("tactical")
    halves = graded_halves()
    graded = np.concatenate([np.arange(first, last + 1) for _, first, last in halves])
    ball, players = tracking["ball"], tracking["players"]
    stopped, _ = v1.clue_team_speed(tracking)

    visible, lost, missing_frames, found, gaps = ball_loss(ball, players, stopped,
                                                           stopping_soon(stopped), halves)
    noise, noise_frames = ball_noise(ball, halves)
    edge_visible, edge_lost = loss_by_edge(ball, halves)
    lengths, slots, seen_players = fragments(players, tracking["team"], halves)
    missing = np.isnan(ball[graded, 0])
    position = ~np.isnan(players[graded, :, 0])
    speed_missing = position & np.isnan(tracking["speed"][graded])

    stats = {
        "source": "BAR-ATH tactical tracking (input only), graded frames; bench/measure.py",
        "crowd_radius_m": CROWD_RADIUS,
        "ball_missing_share": float(missing.mean()),
        "ball_missing_share_stopped": float(missing[stopped[graded]].mean()),
        "ball_missing_share_moving": float(missing[~stopped[graded]].mean()),
        "stopped_share": float(stopped[graded].mean()),
        "stopping_ahead_frames": STOPPING_AHEAD,
        "loss_visible_frames": visible.tolist(),      # [not stopping soon, stopping soon] x crowd 0..6+
        "loss_events": lost.tolist(),
        "edge_zones_m": list(EDGE_ZONES),
        "loss_visible_frames_by_edge": edge_visible.tolist(),  # [outside, near a line, further in]
        "loss_events_by_edge": edge_lost.tolist(),
        "missing_frames": missing_frames.tolist(),    # [moving, stopped]
        "found_events": found.tolist(),
        "gap_frames": gaps,
        "ball_noise_m": noise,
        "ball_noise_frames": noise_frames,
        "fragment_frames": sorted(lengths),
        "slots": slots,
        "players_visible_mean": float(seen_players.mean()),
        "player_speed_missing_share": float(speed_missing.sum() / position.sum()),
    }
    with open(OUTPUT, "w") as handle:
        json.dump(stats, handle, indent=1)

    hazard = np.divide(lost, visible, out=np.zeros(lost.shape), where=visible > 0)
    print("ball missing {:.1%} (players stopped {:.1%}, moving {:.1%}); stopped frames {:.1%}".format(
        stats["ball_missing_share"], stats["ball_missing_share_stopped"],
        stats["ball_missing_share_moving"], stats["stopped_share"]))
    for name, row in zip(("moving", "stopped"), range(2)):
        print("  chance of losing the ball next frame, {} (soon), crowd 0..6+: {}".format(
            name, " ".join("{:.4f}".format(h) for h in hazard[row])))
        print("    visible frames per crowd size: {}".format(visible[row].tolist()))
        g = np.array(gaps[name])
        print("  gaps starting while {}: {}, median {:.1f} s, mean {:.1f} s".format(
            name, len(g), np.median(g) / FPS, g.mean() / FPS))
    print("  chance of finding a missing ball next frame: moving {:.4f}, stopped {:.4f}".format(
        *(found / missing_frames)))
    print("  chance of losing the ball next frame, by zone (outside, within 5 m of a line, further in): "
          + " ".join("{:.4f}".format(l / v) for l, v in zip(edge_lost, edge_visible)))
    print("ball noise {:.3f} m per axis ({} slow frames)".format(noise, noise_frames))
    print("fragments {}, median {:.1f} s, mean {:.1f} s; slots {}".format(
        len(lengths), np.median(lengths) / FPS, np.mean(lengths) / FPS, slots))
    print("players visible per frame {:.2f}; player speed missing {:.1%}".format(
        stats["players_visible_mean"], stats["player_speed_missing_share"]))
    print("wrote {}".format(os.path.relpath(OUTPUT, ROOT)))


if __name__ == "__main__":
    main()
