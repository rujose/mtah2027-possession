"""IDSSE's own fields -> H/A/D per frame, plus three checks of how far DFL's convention sits from
Metrica's (docs/bench.md, "Label").

    python -m bench.labels          # prints the checks for every cached match

Labels are used raw (decision 2): BallStatus 0 -> D, otherwise BallPossession 1 -> H, 2 -> A.
The "merged" variant exists only for the sensitivity run: if a ranking of two versions flips
between raw and merged labels, that comparison is too close to trust.
"""

import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np

from bench import idsse
from possession.data import HOME, ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
SHORT = 13             # frames: a possession under 0.5 s (12.5 frames) counts as "short"
VERY_SHORT_S = 1.0     # seconds, the second line reported by check 1
CROSS_LOOKBACK = 5 * idsse.FPS  # frames searched back for the moment the ball crossed a line
RESTART_TOLERANCE = 5 * idsse.FPS  # frames between a dead stretch's end and its restart event
RESTARTS = ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty")
CONTROL_RADIUS = 1.0   # m: a player this close to a slow ball is very likely controlling it
CONTROL_SPEED = 2.0    # m/s: a ball slower than this is at someone's feet, not in flight


def runs_of(values):
    """[(start, stop, value)] maximal runs of equal values; stop exclusive."""
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(values)]
    return [(int(a), int(b), values[a]) for a, b in zip(starts, stops)]


def codes(data, merged=False):
    """Scorer codes (pm.H, pm.A, pm.D, pm.X) for every frame of a cached match.

    Frames outside the halves (the one empty frame between them) are X, so the scorer never lets
    a possession run across half-time. Where the ball is alive but no team is written, or the
    status itself is missing, the label of the previous frame carries on ("a loose ball stays with
    the last team"); counts are returned so that this is never silent.
    """
    status, possession = data["status"], data["possession"]
    out = np.full(len(status), pm.X, dtype=np.int8)
    filled = 0
    for _, first, last in data["halves"]:
        s, p = status[first:last + 1], possession[first:last + 1]
        label = np.where(s == 0, pm.D, np.where(p == 1, pm.H, np.where(p == 2, pm.A, -1)))
        label[(s == -1)] = -1
        known = label >= 0
        filled += int((~known).sum())
        carry = np.maximum.accumulate(np.where(known, np.arange(len(label)), -1))
        first_known = np.flatnonzero(known)[0]
        carry[carry < 0] = first_known
        out[first:last + 1] = label[carry]
    if merged:
        out = merge_short(out)
    return out, filled


def merge_short(out):
    """Possessions shorter than SHORT frames take the label of whatever came before them.

    One pass in order, so a chain of short possessions collapses into the label before the chain:
    a touch does not change possession, and after a dead ball nobody has it until someone controls.
    """
    out = out.copy()
    previous = None
    for start, stop, value in runs_of(out):
        if value in (pm.H, pm.A) and stop - start < SHORT and previous is not None:
            out[start:stop] = previous
        else:
            previous = out[start] if value != pm.X else None
    return out


def check_short_possessions(gt):
    """Check 1: how many possessions are under 0.5 s and under 1 s."""
    spells = [b - a for a, b, v in runs_of(gt) if v in (pm.H, pm.A)]
    spells = np.array(spells)
    return {"possessions": len(spells),
            "median_s": float(np.median(spells)) / idsse.FPS,
            "under_0.5s": int((spells < SHORT).sum()),
            "under_1s": int((spells < VERY_SHORT_S * idsse.FPS).sum())}


def outside(ball, pitch):
    """True where the ball is beyond a pitch line (no margin: IDSSE's ball is precise)."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        return (x < 0) | (x > pitch[0]) | (y < 0) | (y > pitch[1])


def check_dead_ball_start(gt, data):
    """Check 2: is the ball out when D starts, and if so how long after it crossed the line?

    For every dead stretch: where was the ball on its first frame? If outside the pitch, go back
    to the last frame it was inside (up to CROSS_LOOKBACK) and report the lag in frames. A dead
    stretch that starts with the ball inside is a whistle stoppage (foul, offside, injury).
    """
    out = outside(data["ball"], data["pitch"])
    lags, inside = [], 0
    for start, stop, value in runs_of(gt):
        if value != pm.D or start == 0 or gt[start - 1] == pm.X:
            continue  # the first frames of a half are not a stoppage
        if not out[start]:
            inside += 1
            continue
        back = out[max(0, start - CROSS_LOOKBACK):start + 1][::-1]
        crossed = np.flatnonzero(~back)
        lags.append(int(crossed[0]) - 1 if len(crossed) else CROSS_LOOKBACK)
    lags = np.array(lags)
    return {"stoppages": len(lags) + inside, "ball_inside_at_start": inside,
            "ball_out_at_start": len(lags),
            "lag_frames_median": float(np.median(lags)) if len(lags) else None,
            "lag_frames_p90": float(np.percentile(lags, 90)) if len(lags) else None}


def restart_events(match, data):
    """[(frame, kind, H or A)] for every restart in the event file.

    Event times carry a local offset (+02:00) while the tracking's are UTC, so both are taken to
    UTC; then each event goes to the frame with the nearest tracking time. A kick-off names its
    team on the pass inside it, the other restarts on themselves.
    """
    home_id, guest_id, _, _ = idsse.match_information(match)
    side = {home_id: pm.H, guest_id: pm.A}
    found = []
    for event in ET.parse(idsse.raw_file("events_raw", match)).getroot().iter("Event"):
        restart = next((child for child in event if child.tag in RESTARTS), None)
        if restart is None:
            continue
        team = restart.get("Team") or next((p.get("Team") for p in restart.iter("Play")), None)
        when = datetime.fromisoformat(event.get("EventTime")).astimezone(timezone.utc)
        found.append((np.datetime64(when.replace(tzinfo=None), "ms"), restart.tag, side.get(team)))

    out = []
    for _, first, last in data["halves"]:
        times = data["time"][first:last + 1]
        for when, kind, team in found:
            index = int(np.clip(np.searchsorted(times, when), 0, len(times) - 1))
            if times[0] - np.timedelta64(10, "s") <= when <= times[-1] + np.timedelta64(10, "s"):
                out.append((first + index, kind, team))
    return out


def check_restart_team(gt, events):
    """Check 3: after a dead stretch, is the first team with the ball the team that restarts?

    Only dead stretches with a restart event within RESTART_TOLERANCE of their end are counted.
    """
    starts = np.array(sorted(frame for frame, _, _ in events))
    by_frame = {frame: team for frame, _, team in events}
    agree = checked = 0
    for start, stop, value in runs_of(gt):
        if value != pm.D or stop >= len(gt) or gt[stop] == pm.X:
            continue
        near = starts[np.abs(starts - stop) <= RESTART_TOLERANCE]
        if not len(near):
            continue
        checked += 1
        agree += int(by_frame[int(near[np.argmin(np.abs(near - stop))])] == gt[stop])
    return {"restart_events": len(events), "stoppages_with_a_restart_event": checked,
            "first_team_is_restart_team": agree}


def check_close_control(gt, data):
    """Check of our own reading (coordinates, teams, labels): where the ball is slow (under
    CONTROL_SPEED) and a player is within CONTROL_RADIUS of it, that player's team should almost
    always be the labelled team. A low share means something is mis-read, not a convention gap."""
    ball, players, team = data["ball"], data["players"], data["team"]
    distance = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))
    with np.errstate(invalid="ignore"):
        nearest = np.nanargmin(np.where(np.isnan(distance), np.inf, distance), axis=1)
        close = np.nanmin(np.where(np.isnan(distance), np.inf, distance), axis=1) <= CONTROL_RADIUS
        speed = np.r_[np.nan, np.hypot(*np.diff(ball, axis=0).T) * idsse.FPS]
        slow = speed < CONTROL_SPEED
    alive = (gt == pm.H) | (gt == pm.A)
    chosen = alive & close & slow
    nearest_code = np.where(team[nearest] == HOME, pm.H, pm.A)
    return {"frames": int(chosen.sum()), "team_agrees": float((nearest_code[chosen] == gt[chosen]).mean())}


def main():
    for match in idsse.matches():
        data = idsse.load(match)
        gt, filled = codes(data)
        scored = gt != pm.X
        shares = {name: float((gt[scored] == code).mean())
                  for name, code in (("H", pm.H), ("A", pm.A), ("D", pm.D))}
        print("{}  {}".format(match, data["title"]))
        print("  labels  H {H:.1%}  A {A:.1%}  D {D:.1%}  frames filled from the previous label: "
              "{filled}".format(filled=filled, **shares))
        print("  check 1 {}".format(check_short_possessions(gt)))
        print("  check 2 {}".format(check_dead_ball_start(gt, data)))
        print("  check 3 {}".format(check_restart_team(gt, restart_events(match, data))))
        print("  reading check {}".format(check_close_control(gt, data)))
        merged = merge_short(gt)
        print("  merged: {} frames change".format(int((merged != gt).sum())))


if __name__ == "__main__":
    main()
