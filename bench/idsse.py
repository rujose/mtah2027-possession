"""IDSSE (open DFL data): download, read and cache as arrays shaped like our own tracking.

    python -m bench.idsse download     # 2.63 GB from Figshare, MD5-checked; skips files already there
    python -m bench.idsse parse        # one cache file per match, ~1 min per match

Bassek, Rein, Weber, Memmert (2025). An integrated dataset of spatiotemporal and event data in
elite soccer. Scientific Data 12:195. Figshare article 28196177, CC BY 4.0.

Our own streaming reader, not kloppy: it keeps memory small on 8 GB, and it reads BallPossession
exactly as written (kloppy turns a missing value into "away"). See docs/bench.md, decision 1.
"""

import hashlib
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from possession.data import AWAY, HOME, ROOT

# Params
ARTICLE_URL = "https://api.figshare.com/v2/articles/28196177"
FOLDER = os.path.join(ROOT, "data", "external", "idsse")
RAW = os.path.join(FOLDER, "raw")
CACHE = os.path.join(FOLDER, "cache")
FPS = 25
SECTIONS = ("firstHalf", "secondHalf")  # extra-time sections, if any, are not read
CHUNK = 1 << 20


def matches():
    """Match ids present in the raw folder, e.g. 'J03WOH', sorted."""
    names = os.listdir(RAW) if os.path.isdir(RAW) else []
    return sorted({m.group(1) for m in map(re.compile(r"DFL-MAT-(\w+)\.xml$").search, names) if m})


def raw_file(kind, match):
    """Path of a match's 'matchinformation', 'events_raw' or 'positions_raw_observed' file."""
    for name in os.listdir(RAW):
        if kind in name and name.endswith("DFL-MAT-{}.xml".format(match)):
            return os.path.join(RAW, name)
    raise FileNotFoundError("{} for {} not in {}".format(kind, match, RAW))


def cache_path(match):
    return os.path.join(CACHE, "{}.npz".format(match))


# --------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------

def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download():
    os.makedirs(RAW, exist_ok=True)
    with urllib.request.urlopen(ARTICLE_URL) as response:
        article = json.load(response)
    with open(os.path.join(FOLDER, "figshare_article.json"), "w") as handle:
        json.dump(article, handle, indent=1)
    print("{}: {} files, {:.2f} GB, licence {}".format(
        article["title"], len(article["files"]), sum(f["size"] for f in article["files"]) / 1e9,
        article["license"]["name"]))

    for item in article["files"]:
        path = os.path.join(RAW, item["name"])
        if os.path.exists(path) and os.path.getsize(path) == item["size"] \
                and md5_of(path) == item["computed_md5"]:
            print("  have  {}".format(item["name"]))
            continue
        print("  fetch {} ({:.0f} MB)".format(item["name"], item["size"] / 1e6), flush=True)
        partial = path + ".part"
        with urllib.request.urlopen(item["download_url"]) as response, open(partial, "wb") as out:
            for block in iter(lambda: response.read(CHUNK), b""):
                out.write(block)
        if md5_of(partial) != item["computed_md5"]:
            raise RuntimeError("MD5 mismatch for {}".format(item["name"]))
        os.replace(partial, path)
    print("all files present and MD5-checked")


# --------------------------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------------------------

def match_information(match):
    """(home team id, guest team id, (pitch length, pitch width) in metres, title)."""
    root = ET.parse(raw_file("matchinformation", match)).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    return general.get("HomeTeamId"), general.get("GuestTeamId"), pitch, general.get("MatchTitle")


def read_positions(match):
    """Every FrameSet of the positions file, streamed.

    Returns a list of dicts: section, team id, person id, and numpy arrays of the frame number N,
    X, Y, S; the ball's FrameSets also carry BallPossession and BallStatus, with -1 where an
    attribute is absent so that missing values are counted, never guessed, and the wall-clock time
    T of each frame, which is how events are placed on frames.
    """
    sets, current, rows = [], None, None
    for event, element in ET.iterparse(raw_file("positions_raw_observed", match),
                                       events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            current = {"section": element.get("GameSection"), "team": element.get("TeamId"),
                       "person": element.get("PersonId")}
            rows, times = [], []
            is_ball = current["team"].upper() == "BALL"
        elif event == "end" and element.tag == "Frame":
            get = element.get
            rows.append((int(get("N")), float(get("X")), float(get("Y")), float(get("S") or "nan"),
                         int(get("BallPossession") or -1), int(get("BallStatus") or -1)))
            if is_ball:
                times.append(get("T")[:23])  # drop the "+00:00": every file is in UTC
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            values = np.array(rows, dtype=np.float64)
            current.update(n=values[:, 0].astype(np.int64), x=values[:, 1], y=values[:, 2],
                           s=values[:, 3], possession=values[:, 4].astype(np.int8),
                           status=values[:, 5].astype(np.int8),
                           time=np.array(times, dtype="datetime64[ms]") if is_ball else None)
            sets.append(current)
            element.clear()
    return sets


def build(match):
    """One match as arrays shaped like possession.data.load_tracking, plus its own labels.

    The two halves are laid end to end with ONE empty frame between them, so that nothing (a
    possession, a smoothing window) can run across half-time. `halves` gives the inclusive frame
    range of each, in the same form as possession.data.graded_halves().
    Positions move from DFL's centre-spot origin to our corner origin; speed goes km/h -> m/s.
    """
    home_id, guest_id, pitch, title = match_information(match)
    # Referees are tracked too (TeamId "referee"); our tracking has no referee slots, so drop them.
    sets = [s for s in read_positions(match)
            if s["section"] in SECTIONS and s["team"] in (home_id, guest_id, "BALL")]
    is_ball = [s["team"].upper() == "BALL" for s in sets]
    if sum(is_ball) != len(SECTIONS):
        raise ValueError("{}: expected one ball FrameSet per half, found {}".format(match, sum(is_ball)))

    # Frame range of each half, from the ball (it is tracked through dead time as well).
    offsets, halves, length = {}, [], 0
    for half, section in enumerate(SECTIONS, start=1):
        ball = next(s for s, b in zip(sets, is_ball) if b and s["section"] == section)
        first, last = int(ball["n"].min()), int(ball["n"].max())
        offsets[section] = length - first
        halves.append((half, length, length + last - first))
        length += last - first + 2  # + 1 empty frame between halves
    n_frames = length - 1

    people = sorted({s["person"] for s, b in zip(sets, is_ball) if not b})
    team_of = {s["person"]: s["team"] for s, b in zip(sets, is_ball) if not b}
    slot = {person: i for i, person in enumerate(people)}
    players = np.full((n_frames, len(people), 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, len(people)), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    time = np.full(n_frames, np.datetime64("NaT", "ms"), dtype="datetime64[ms]")
    possession = np.full(n_frames, -1, dtype=np.int8)
    status = np.full(n_frames, -1, dtype=np.int8)
    half_x, half_y = pitch[0] / 2, pitch[1] / 2

    for s, b in zip(sets, is_ball):
        frames = s["n"] + offsets[s["section"]]
        keep = (frames >= 0) & (frames < n_frames)
        frames = frames[keep]
        xy = np.stack([s["x"][keep] + half_x, s["y"][keep] + half_y], axis=1)
        if b:
            ball_xy[frames] = xy
            possession[frames] = s["possession"][keep]
            status[frames] = s["status"][keep]
            time[frames] = s["time"][keep]
        else:
            players[frames, slot[s["person"]]] = xy
            speed[frames, slot[s["person"]]] = s["s"][keep] / 3.6

    team = np.array([HOME if team_of[p] == home_id else AWAY if team_of[p] == guest_id else -1
                     for p in people], dtype=np.int8)
    if (team < 0).any():
        raise ValueError("{}: players of an unknown team".format(match))
    return {"players": players, "speed": speed, "ball": ball_xy, "team": team,
            "possession": possession, "status": status, "time": time, "halves": np.array(halves),
            "pitch": np.array(pitch), "title": title}


def load(match):
    """A cached match (see build). Missing cache -> run `python -m bench.idsse parse` first."""
    with np.load(cache_path(match)) as data:
        out = {key: data[key] for key in data.files}
    out["halves"] = [tuple(int(v) for v in row) for row in out["halves"]]
    out["pitch"] = tuple(float(v) for v in out["pitch"])
    out["title"] = str(out["title"])
    return out


def parse():
    os.makedirs(CACHE, exist_ok=True)
    total = 0
    for match in matches():
        data = build(match)
        np.savez_compressed(cache_path(match), **data)
        in_halves = sum(last - first + 1 for _, first, last in data["halves"])
        total += in_halves
        on_pitch = (~np.isnan(data["players"][:, :, 0])).sum(axis=1)
        status = data["status"]
        alive = status == 1
        print("{}  {:<45s} frames {:>7d}  players on pitch (median) {:>2.0f}  dead {:5.1f}%  "
              "status missing {}  alive without possession {}".format(
                  match, data["title"], in_halves, np.median(on_pitch), 100 * (status == 0).mean(),
                  int((status == -1).sum()) - (len(data["halves"]) - 1),
                  int((alive & (data["possession"] < 1)).sum())), flush=True)
    print("total frames in halves: {} (the paper counts 1,002,644 position frames)".format(total))


if __name__ == "__main__":
    {"download": download, "parse": parse}[sys.argv[1]]()
