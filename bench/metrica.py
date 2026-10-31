"""Metrica's public sample games: download, events -> H/A/D, tracking -> arrays like the bench's.

    python -m bench.metrica download   # ~190 MB from GitHub, each file checked against its git hash
    python -m bench.metrica parse      # one cache file per game, plus the label checks

github.com/metrica-sports/sample-data: 3 anonymised matches, tracking and events "synchronized",
pitch 105 x 68 m, 25 fps. No licence file; the README asks users to acknowledge the source.
Full explanation and the recipe's reasoning: docs/convention-check.md, piece 1.
"""

import csv
import hashlib
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from possession.data import AWAY, HOME, ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
REPO = "metrica-sports/sample-data"
FILES = {
    1: ["Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"],
    2: ["Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"],
    3: ["Sample_Game_3/Sample_Game_3_events.json",
        "Sample_Game_3/Sample_Game_3_metadata.xml",
        "Sample_Game_3/Sample_Game_3_tracking.txt"],
}
FOLDER = os.path.join(ROOT, "data", "external", "metrica")
RAW = os.path.join(FOLDER, "raw")
CACHE = os.path.join(FOLDER, "cache")
PITCH = (105.0, 68.0)
FPS = 25
SPEED_SPAN = 2          # frames each side: player speed = distance over 4 frames (0.16 s) / time

# The recipe (docs/convention-check.md, piece 1)
CONTROLLING = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}


def match_names():
    return ["metrica{}".format(game) for game in sorted(FILES)]


def cache_path(name):
    return os.path.join(CACHE, "{}.npz".format(name))


# --------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------

def git_blob_sha(path):
    """The hash git itself gives a file, which GitHub's API lists for every file in the repo."""
    digest = hashlib.sha1()
    digest.update("blob {}\0".format(os.path.getsize(path)).encode())
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download():
    os.makedirs(RAW, exist_ok=True)
    for game, files in FILES.items():
        folder = os.path.dirname(files[0])
        url = "https://api.github.com/repos/{}/contents/data/{}".format(REPO, folder)
        with urllib.request.urlopen(url) as response:
            listing = {item["name"]: item for item in json.load(response)}
        for name in files:
            item = listing[os.path.basename(name)]
            path = os.path.join(RAW, item["name"])
            if os.path.exists(path) and git_blob_sha(path) == item["sha"]:
                print("  have  {}".format(item["name"]))
                continue
            print("  fetch {} ({:.0f} MB)".format(item["name"], item["size"] / 1e6), flush=True)
            urllib.request.urlretrieve(item["download_url"], path + ".part")
            if git_blob_sha(path + ".part") != item["sha"]:
                raise RuntimeError("hash mismatch for {}".format(item["name"]))
            os.replace(path + ".part", path)
    print("all files present and hash-checked")


# --------------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------------

def tokens_from(subtype):
    """Subtype -> set of words. CSV joins them with '-' ('GROUND-FAULT-WON'); JSON gives a dict or
    a list of dicts."""
    if not subtype:
        return set()
    if isinstance(subtype, str):
        return {part.strip() for part in subtype.split("-") if part.strip()}
    if isinstance(subtype, dict):
        return {subtype["name"]}
    return {item["name"] for item in subtype}


def read_events(game):
    """[{team, type, tokens, period, start, end}], team as HOME or AWAY, frames as in the tracking."""
    raw = os.path.join(RAW, os.path.basename(FILES[game][0]))
    events = []
    if raw.endswith(".csv"):
        side = {"Home": HOME, "Away": AWAY}
        with open(raw) as handle:
            for row in csv.DictReader(handle):
                events.append({"team": side.get(row["Team"]), "type": row["Type"].strip(),
                               "tokens": tokens_from(row["Subtype"]), "period": int(row["Period"]),
                               "start": int(row["Start Frame"]), "end": int(row["End Frame"])})
    else:
        side = team_sides(game)
        with open(raw) as handle:
            for item in json.load(handle)["data"]:
                events.append({"team": side.get(item["team"]["id"]), "type": item["type"]["name"],
                               "tokens": tokens_from(item.get("subtypes")),
                               "period": int(item["period"]), "start": int(item["start"]["frame"]),
                               "end": int(item["end"]["frame"] or item["start"]["frame"])})
    return events


def dead_start(event):
    """Frame where this event stops play, or None (docs/convention-check.md, piece 1, step 2)."""
    kind, tokens = event["type"], event["tokens"]
    if kind == "BALL OUT":
        return max(event["end"], event["start"])
    if "ADVANTAGE" in tokens:
        return None
    if (kind == "CHALLENGE" and "FAULT" in tokens) or kind == "FAULT RECEIVED" or "OFFSIDE" in tokens:
        return event["start"]
    if kind == "SHOT" and ({"GOAL", "OUT"} & tokens):
        return max(event["end"], event["start"])
    return None


def is_restart(event):
    return event["type"] == "SET PIECE" or (event["type"] == "PASS" and "GOAL KICK" in event["tokens"])


def labels_from_events(events, periods):
    """Scorer codes per ORIGINAL tracking frame for the frames inside `periods`
    ({period: (first, last)} inclusive), X elsewhere. Also counts, for the report, how many dead
    stretches ended because play visibly went on (a controlling event) without a tagged restart.
    """
    n = max(last for _, last in periods.values()) + 1
    out = np.full(n, pm.X, dtype=np.int8)
    ended_by_play = 0
    for period, (first, last) in periods.items():
        mine = sorted((e for e in events if e["period"] == period and first <= e["start"] <= last),
                      key=lambda e: e["start"])
        owner = np.full(last - first + 1, -1, dtype=np.int8)
        dead = np.zeros(last - first + 1, bool)
        for e in mine:
            if e["type"] in CONTROLLING and e["team"] is not None:
                owner[e["start"] - first:] = e["team"]
        kickoff = next((e["start"] for e in mine
                        if e["type"] == "SET PIECE" and "KICK OFF" in e["tokens"]), first)
        dead[:kickoff - first] = True
        for i, e in enumerate(mine):
            start = dead_start(e)
            if start is None or not first <= start <= last:
                continue
            # until the next restart, or the next controlling event if play went on without one
            end = last + 1
            for later in mine[i + 1:]:
                if later["start"] <= start:
                    continue
                if is_restart(later):
                    end = later["start"]
                    break
                if later["type"] in CONTROLLING:
                    end = later["start"]
                    ended_by_play += 1
                    break
            dead[start - first:end - first] = True
        label = np.where(dead, pm.D, np.where(owner == HOME, pm.H, np.where(owner == AWAY, pm.A, pm.D)))
        out[first:last + 1] = label
    return out, ended_by_play


# --------------------------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------------------------

def team_sides(game):
    """Game 3: team id -> HOME/AWAY, from the metadata's local (home) and visiting team."""
    score = ET.parse(os.path.join(RAW, "Sample_Game_3_metadata.xml")).getroot().find(".//Score")
    return {score.get("idLocalTeam"): HOME, score.get("idVisitingTeam"): AWAY}


def read_csv_tracking(path):
    """Games 1-2: (period [frames], frame [frames], xy [frames, players, 2], ball [frames, 2])."""
    with open(path) as handle:
        rows = list(csv.reader(handle))
    header = rows[2]
    data = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in row] for row in rows[3:]])
    players = [i for i, name in enumerate(header) if name.startswith("Player")]
    ball = header.index("Ball")
    xy = np.stack([data[:, [i, i + 1]] for i in players], axis=1)
    return data[:, 0].astype(int), data[:, 1].astype(int), xy, data[:, [ball, ball + 1]]


def read_epts_tracking():
    """Game 3: EPTS (FIFA format). Each line holds the 22 players on the pitch, in the order its
    frame range's DataFormatSpecification lists them (it changes at substitutions); halves come
    from the provider parameters. Returns (period, frame, xy [frames, players, 2], ball, team)."""
    root = ET.parse(os.path.join(RAW, "Sample_Game_3_metadata.xml")).getroot()
    side = team_sides(3)
    people = [p.get("id") for p in root.iter("Player")]
    team = np.array([side[p.get("teamId")] for p in root.iter("Player")], dtype=np.int8)
    column = {person: i for i, person in enumerate(people)}
    channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [channel[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")]
        order = [column[person] for person in refs[::2]]     # x then y per player
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    halves = {1: (int(params["first_half_start"]), int(params["first_half_end"])),
              2: (int(params["second_half_start"]), int(params["second_half_end"]))}

    frames, xys, balls = [], [], []
    with open(os.path.join(RAW, "Sample_Game_3_tracking.txt")) as handle:
        for line in handle:
            frame, players, ball = line.rstrip("\n").split(":")
            frame = int(frame)
            order = next(o for a, b, o in specs if a <= frame <= b)
            values = np.array(players.replace(";", ",").split(","), dtype=float).reshape(-1, 2)
            row = np.full((len(people), 2), np.nan)
            row[order] = values
            frames.append(frame)
            xys.append(row)
            balls.append(np.array(ball.split(","), dtype=float))
    frames = np.array(frames)
    period = np.zeros(len(frames), int)
    for half, (first, last) in halves.items():
        period[(frames >= first) & (frames <= last)] = half
    return period, frames, np.stack(xys), np.stack(balls), team


def player_speed(xy):
    """m/s from positions over 2 * SPEED_SPAN frames (the samples have no speed column).
    NaN where either end is missing."""
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    step = np.hypot(*(xy[2 * SPEED_SPAN:] - xy[:-2 * SPEED_SPAN]).transpose(2, 0, 1))
    speed[SPEED_SPAN:-SPEED_SPAN] = step / (2 * SPEED_SPAN / FPS)
    return speed


def build(game):
    """One game as the bench's arrays (see bench/idsse.py build): halves laid end to end with one
    empty frame between them, positions in metres, plus its own H/A/D from the events."""
    events = read_events(game)
    if game == 3:
        period, frame, xy, ball, team = read_epts_tracking()
    else:
        p_home, frame, xy_home, ball = read_csv_tracking(os.path.join(RAW, os.path.basename(FILES[game][1])))
        p_away, frame_away, xy_away, _ = read_csv_tracking(os.path.join(RAW, os.path.basename(FILES[game][2])))
        if not (frame == frame_away).all():
            raise ValueError("home and away tracking frames differ")
        period, xy = p_home, np.concatenate([xy_home, xy_away], axis=1)
        team = np.array([HOME] * xy_home.shape[1] + [AWAY] * xy_away.shape[1], dtype=np.int8)

    scale = np.array(PITCH, dtype=np.float32)
    xy = (xy * scale).astype(np.float32)
    ball = (ball * scale).astype(np.float32)
    spans = {p: (int(frame[period == p].min()), int(frame[period == p].max()))
             for p in (1, 2) if (period == p).any()}
    codes, ended_by_play = labels_from_events(events, spans)

    # lay the two halves end to end, one empty frame between, as the IDSSE cache does
    row_of = {f: i for i, f in enumerate(frame)}
    pieces, halves, length = [], [], 0
    for half, (first, last) in sorted(spans.items()):
        rows = [row_of[f] for f in range(first, last + 1)]
        pieces.append(rows)
        halves.append((half, length, length + last - first))
        length += last - first + 2
    n_frames = length - 1
    players = np.full((n_frames, xy.shape[1], 2), np.nan, dtype=np.float32)
    ball_out = np.full((n_frames, 2), np.nan, dtype=np.float32)
    labels = np.full(n_frames, pm.X, dtype=np.int8)
    for (half, start, stop), rows in zip(halves, pieces):
        players[start:stop + 1] = xy[rows]
        ball_out[start:stop + 1] = ball[rows]
        labels[start:stop + 1] = codes[spans[half][0]:spans[half][1] + 1]
    speed = np.full((n_frames, xy.shape[1]), np.nan, dtype=np.float32)
    for _, start, stop in halves:
        speed[start:stop + 1] = player_speed(players[start:stop + 1])
    return {"players": players, "speed": speed, "ball": ball_out, "team": team,
            "labels": labels, "halves": np.array(halves), "pitch": np.array(PITCH),
            "title": "Metrica sample game {}".format(game), "dead_ended_by_play": ended_by_play}


def load(name):
    with np.load(cache_path(name)) as data:
        out = {key: data[key] for key in data.files}
    out["halves"] = [tuple(int(v) for v in row) for row in out["halves"]]
    out["pitch"] = tuple(float(v) for v in out["pitch"])
    out["title"] = str(out["title"])
    return out


def parse():
    os.makedirs(CACHE, exist_ok=True)
    for game in sorted(FILES):
        data = build(game)
        np.savez_compressed(cache_path("metrica{}".format(game)), **data)
        print("metrica{}: {} frames in halves, {} players".format(
            game, sum(b - a + 1 for _, a, b in data["halves"]), data["players"].shape[1]), flush=True)


if __name__ == "__main__":
    {"download": download, "parse": parse}[sys.argv[1]]()
