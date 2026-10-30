"""Load the provided assets into plain numpy arrays on the submission timeline.

Everything returned here is indexed by VIDEO frame (row i of a submission), not by the tracking's
own frame id: tracking line N is video frame N - 1 (DATA.md, "Frame alignment"). Positions are in
metres, with the pitch size from games.py, so distances along and across the pitch are comparable.

No model logic lives here: only reading, unit conversion and the team of each track slot.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]  # metres
HOME, AWAY = 0, 1


def slot_teams(meta_path):
    """Team of each track slot (HOME/AWAY), and each team's attack direction in the first half.

    The first team in the metadata is the home side (EPTS convention; Barcelona here).
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    teams = np.array([HOME if p.get("teamId") == team_ids[0] else AWAY for p in root.iter("Player")],
                     dtype=np.int8)
    directions = {}
    for index, team in enumerate(root.iter("Team")):
        for param in team.iter("ProviderParameter"):
            if param.findtext("Name") == "attack_direction_first_half":
                directions[index] = param.findtext("Value")
    return teams, directions


def load_tracking(source="tactical", game=GAME, data_dir=None):
    """Tracking as arrays over video frames.

    Returns a dict:
        players  float32 [frames, slots, 2]  x, y in metres (NaN where not seen)
        speed    float32 [frames, slots]     metres per second, from the file (NaN where missing)
        ball     float32 [frames, 2]         x, y in metres (NaN where not seen)
        team     int8    [slots]             HOME or AWAY
        directions                           {team: 'left_to_right' | 'right_to_left'} (1st half)

    The ball's own speed column is dropped: it is empty on 91% of ball frames and wrong when
    filled (docs/data.md). Ball speed is computed from positions by the models instead.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    team, directions = slot_teams(meta_path)
    n_frames = source_frames(source, game)

    rows = np.full((n_frames, len(team) * 3 + 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame_id, players, ball = line.rstrip("\n").split(":")
            rows[int(frame_id) - 1] = np.array((players.replace(";", ",") + "," + ball).split(","),
                                               dtype=np.float32)

    values = rows[:, :-3].reshape(n_frames, len(team), 3)
    scale = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float32)
    return {
        "players": values[:, :, :2] * scale,
        "speed": values[:, :, 2],
        "ball": rows[:, -3:-1] * scale,
        "team": team,
        "directions": directions,
    }


def load_tags(game=GAME, data_dir=None):
    """Smart Tagging instances as dicts: start/end video frame, code, and every label group.

    Seconds become video frames with round(seconds * 25) (DATA.md, "Time base -- verified").
    Codes and label values are upper-cased so matching never depends on the export's casing.
    """
    (path,) = require(["smart_tagging"], game, data_dir)
    tags = []
    for instance in ET.parse(path).getroot().iter("instance"):
        tag = {"start": int(round(float(instance.findtext("start")) * FPS)),
               "end": int(round(float(instance.findtext("end")) * FPS)),
               "code": instance.findtext("code").strip().upper()}
        for label in instance.iter("label"):
            tag[label.findtext("group").strip()] = (label.findtext("text") or "").strip().upper()
        tags.append(tag)
    return tags


def graded_halves(game=GAME):
    """[(half, first_video_frame, last_video_frame)] for the two graded halves, inclusive."""
    return half_video_blocks(game)
