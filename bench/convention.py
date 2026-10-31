"""Does Metrica's possession change hands as often as DFL's? (docs/convention-check.md, pieces 1-3)

    python -m bench.convention      # prints the checks and the comparison, writes bench/results/convention.csv

Three label sources, compared only during live play:
  (a) dfl_frames      DFL's own per-frame BallPossession / BallStatus (the bench's labels)
  (b) dfl_events      DFL's events through our recipe, with DFL's per-frame status for dead ball
  (c) metrica_events  Metrica's events through the same recipe (bench/metrica.py)
(b) vs (a) shows what the recipe does; (c) vs (b) compares the conventions like for like.
"""

import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bench import idsse, labels, metrica
from possession.data import ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
OUTPUT = os.path.join(ROOT, "bench", "results", "convention.csv")
FPS = 25
CONTROLLING_DFL = {"Play", "ShotAtGoal", "OtherBallAction", "BallClaiming"}
RESTARTS_DFL = {"KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty"}


# --------------------------------------------------------------------------------------------
# (b): the recipe on DFL's events
# --------------------------------------------------------------------------------------------

def dfl_controlling_events(match, data):
    """[(frame, H or A)] for every event that gives a team the ball, placed on the frame with the
    nearest tracking time (event times carry a +02:00 offset; tracking times are UTC)."""
    home_id, guest_id, _, _ = idsse.match_information(match)
    side = {home_id: pm.H, guest_id: pm.A}
    found = []
    for event in ET.parse(idsse.raw_file("events_raw", match)).getroot().iter("Event"):
        child = next(iter(event), None)
        if child is None:
            continue
        if child.tag in CONTROLLING_DFL or child.tag in RESTARTS_DFL:
            team = child.get("Team") or next((p.get("Team") for p in child.iter("Play")), None)
        elif child.tag == "TacklingGame" and child.get("PossessionChange") == "true":
            team = child.get("WinnerTeam")
        else:
            continue
        if team not in side:
            continue
        when = datetime.fromisoformat(event.get("EventTime")).astimezone(timezone.utc)
        found.append((np.datetime64(when.replace(tzinfo=None), "ms"), side[team]))

    out = []
    for _, first, last in data["halves"]:
        times = data["time"][first:last + 1]
        for when, team in found:
            if times[0] - np.timedelta64(10, "s") <= when <= times[-1] + np.timedelta64(10, "s"):
                out.append((first + int(np.clip(np.searchsorted(times, when), 0, len(times) - 1)), team))
    return sorted(out, key=lambda item: item[0])


def dfl_event_labels(match, data):
    """Owner from the last controlling event (back-filled to the start of each half), D from
    DFL's own per-frame status."""
    frames = labels.codes(data)[0]
    out = frames.copy()
    events = dfl_controlling_events(match, data)
    for _, first, last in data["halves"]:
        owner = np.full(last - first + 1, -1, dtype=np.int8)
        for frame, team in events:
            if first <= frame <= last:
                owner[frame - first:] = team
        known = np.flatnonzero(owner >= 0)
        owner[:known[0]] = owner[known[0]]
        half = frames[first:last + 1]
        out[first:last + 1] = np.where(half == pm.D, pm.D, owner)
    return out


# --------------------------------------------------------------------------------------------
# The measures
# --------------------------------------------------------------------------------------------

def measures(codes, halves):
    """The comparison's measures for one match and label source (docs/convention-check.md, piece 3)."""
    changes, live_frames, stretches, possessions, stoppages, frames = 0, 0, [], [], [], 0
    for _, first, last in halves:
        half = codes[first:last + 1]
        frames += len(half)
        live = half[(half == pm.H) | (half == pm.A)]
        live_frames += len(live)
        runs = labels.runs_of(live)
        changes += max(len(runs) - 1, 0)
        stretches += [b - a for a, b, _ in runs]
        for i, (a, b, v) in enumerate(labels.runs_of(half)):
            if v in (pm.H, pm.A):
                possessions.append(b - a)
            elif v == pm.D and i > 0:
                stoppages.append(b - a)
    possessions, stretches = np.array(possessions) / FPS, np.array(stretches) / FPS
    per_90 = frames / FPS / 5400
    return {"changes_per_10_live_min": changes / (live_frames / FPS / 600),
            "team_stretch_median_s": float(np.median(stretches)),
            "team_stretch_mean_s": float(stretches.mean()),
            "possessions_per_90": len(possessions) / per_90,
            "possession_median_s": float(np.median(possessions)),
            "possessions_under_1s": float((possessions < 1).mean()),
            "possessions_under_2s": float((possessions < 2).mean()),
            "stoppages_per_90": len(stoppages) / per_90,
            "stoppage_median_s": float(np.median(stoppages)) / FPS,
            "dead_share": 1 - live_frames / frames}


def team_agreement(one, other):
    """Share of frames where both say a team (H or A) on which they name the same one."""
    both = np.isin(one, (pm.H, pm.A)) & np.isin(other, (pm.H, pm.A))
    return float((one[both] == other[both]).mean())


def metrica_checks(name, data):
    """Piece 1's checks: D against their blanked ball, and the reading check."""
    codes = data["labels"]
    in_halves = codes != pm.X
    missing = np.isnan(data["ball"][:, 0])
    dead, live = in_halves & (codes == pm.D), in_halves & np.isin(codes, (pm.H, pm.A))
    return {"match": name, "dead_with_ball_missing": float(missing[dead].mean()),
            "live_with_ball_missing": float(missing[live].mean()),
            "reading_check": labels.check_close_control(codes, data)["team_agrees"],
            "reading_frames": labels.check_close_control(codes, data)["frames"],
            "dead_ended_by_play": int(data["dead_ended_by_play"])}


def main():
    rows, agreement = [], {}
    for match in idsse.matches():
        data = idsse.load(match)
        frames = labels.codes(data)[0]
        events = dfl_event_labels(match, data)
        agreement[match] = team_agreement(frames, events)
        rows.append({"source": "dfl_frames", "match": match, **measures(frames, data["halves"])})
        rows.append({"source": "dfl_events", "match": match, **measures(events, data["halves"])})
    checks = []
    for name in metrica.match_names():
        data = metrica.load(name)
        rows.append({"source": "metrica_events", "match": name, **measures(data["labels"], data["halves"])})
        checks.append(metrica_checks(name, data))

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    table.to_csv(OUTPUT, index=False, float_format="%.4f")

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print("Metrica labels, checks:")
        print(pd.DataFrame(checks).to_string(index=False))
        print("\nDFL: team agreement, events recipe vs per-frame labels (both alive): " +
              " ".join("{} {:.3f}".format(m, a) for m, a in agreement.items()))
        print("\nper match:")
        print(table.to_string(index=False))
        print("\nmean per source:")
        print(table.groupby("source", sort=False).mean(numeric_only=True).T)

    key = "changes_per_10_live_min"
    dfl = table[table["source"] == "dfl_events"][key]
    mine = table[table["source"] == "metrica_events"][key]
    low, high = dfl.min(), dfl.max()
    inside = int(((mine >= low) & (mine <= high)).sum())
    below, above = int((mine < low).sum()), int((mine > high).sum())
    if inside == len(mine):
        verdict = "no evidence of a gap: all Metrica games inside DFL's range"
    elif below == len(mine) or above == len(mine):
        verdict = "a gap: all Metrica games {} DFL's range".format("below" if below else "above")
    else:
        verdict = "in between: {} inside, {} below, {} above".format(inside, below, above)
    print("\ndecision rule on {}: DFL events range {:.2f}-{:.2f}; Metrica {}; -> {}".format(
        key, low, high, " ".join("{:.2f}".format(v) for v in mine), verdict))
    print("wrote {}".format(os.path.relpath(OUTPUT, ROOT)))


if __name__ == "__main__":
    main()
