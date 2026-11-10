"""SkillCorner's open data (20 A-League 2024/25 matches, broadcast tracking): download, read and
cache as arrays shaped like our own tracking, with labels from SkillCorner's own fields.

    python -m bench.skillcorner download   # ~1.8 GB (tracking via the Git LFS media URLs), SHA-256-checked
    python -m bench.skillcorner parse      # one cache file per match, plus the label checks (~1 min each)

github.com/SkillCorner/opendata, MIT licence ("we kindly ask that you credit SkillCorner"). Data by
SkillCorner. Used here as a test only: nothing is trained or tuned on it (docs/lab/skillcorner.md).

The loader's choices, fixed before anything was scored:

- **Positions:** SkillCorner's metres with the centre spot at (0, 0) -> our corner origin (+ half the
  pitch on each axis, no flip: the same as bench/idsse.py does with DFL's centre origin). Pitch size
  from the match file.
- **25 fps:** SkillCorner tracks at 10 fps. Each 25 fps frame is placed at its time on the 10 fps clock
  and positions are interpolated linearly between the two 10 fps frames around it. A player or the
  ball is only placed where both of those frames have it; **the ball stays missing where SkillCorner
  has none.** Off-screen players keep SkillCorner's extrapolated positions (that is its tracking).
- **Speed:** from positions, over 4 frames (0.16 s) at 25 fps, as bench/metrica.py does.
- **Halves:** SkillCorner's two periods, laid end to end with ONE empty frame between them, as the
  bench's caches do; `halves` = [(half, first, last)] inclusive.
- **Labels:** read raw from each frame's `possession.group` ("home team" / "away team" / null: kloppy
  would turn null into dead). **D** wherever no phase of play covers the frame (SkillCorner: phases "are
  only defined when the ball is in play"). Live frames with no team written carry the last team of
  that live stretch ("a loose ball stays with the last team"); a live stretch that starts with no team
  takes the first team written in it. Labels go to 25 fps by the nearest 10 fps frame.
"""

import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

from possession.data import AWAY, HOME, ROOT

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
REPO = "SkillCorner/opendata"
BRANCH = "master"
RAW_URL = "https://raw.githubusercontent.com/{}/{}/data/".format(REPO, BRANCH)
LFS_URL = "https://media.githubusercontent.com/media/{}/{}/data/".format(REPO, BRANCH)
FOLDER = os.path.join(ROOT, "data", "external", "skillcorner")
RAW = os.path.join(FOLDER, "raw")
CACHE = os.path.join(FOLDER, "cache")
FILES = ("match.json", "phases_of_play.csv", "tracking_extrapolated.jsonl")
SOURCE_FPS = 10
FPS = 25
SPEED_SPAN = 2          # frames each side at 25 fps: speed = distance over 0.16 s (as bench/metrica.py)
CHUNK = 1 << 20
TIMEOUT = 120         # s: a stalled connection fails instead of hanging (one did, at 87 MB)


def match_ids():
    """The 20 match ids listed in matches.json, sorted."""
    with open(os.path.join(RAW, "matches.json")) as handle:
        return sorted(str(m["id"]) for m in json.load(handle))


def raw_path(match, kind):
    return os.path.join(RAW, match, "{}_{}".format(match, kind))


def cache_path(match):
    return os.path.join(CACHE, "{}.npz".format(match))


# --------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, path):
    partial = path + ".part"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response, open(partial, "wb") as out:
        for block in iter(lambda: response.read(CHUNK), b""):
            out.write(block)
    os.replace(partial, path)


def lfs_pointer(text):
    """{'oid': sha256, 'size': bytes} from a Git LFS pointer file, or None if `text` isn't one."""
    if not text.startswith("version https://git-lfs"):
        return None
    fields = dict(line.split(" ", 1) for line in text.strip().splitlines())
    return {"oid": fields["oid"].split(":", 1)[1], "size": int(fields["size"])}


def download():
    os.makedirs(RAW, exist_ok=True)
    for name in ("data/matches.json", "LICENSE", "README.md"):
        fetch(RAW_URL[:-len("data/")] + name, os.path.join(RAW, os.path.basename(name)))
    total = 0
    for match in match_ids():
        os.makedirs(os.path.join(RAW, match), exist_ok=True)
        for kind in FILES:
            path = raw_path(match, kind)
            rel = "matches/{0}/{0}_{1}".format(match, kind)
            with urllib.request.urlopen(RAW_URL + rel, timeout=TIMEOUT) as response:
                head = response.read(400).decode("utf-8", "replace")
            pointer = lfs_pointer(head)
            if pointer is None:                      # a plain file in git: fetch it as is
                fetch(RAW_URL + rel, path)
                total += os.path.getsize(path)
                continue
            if os.path.exists(path) and os.path.getsize(path) == pointer["size"] and sha256_of(path) == pointer["oid"]:
                print("  have  {}".format(os.path.basename(path)))
                total += pointer["size"]
                continue
            print("  fetch {} ({:.0f} MB)".format(os.path.basename(path), pointer["size"] / 1e6), flush=True)
            fetch(LFS_URL + rel, path)
            if os.path.getsize(path) != pointer["size"] or sha256_of(path) != pointer["oid"]:
                raise RuntimeError("SHA-256 mismatch for {}".format(path))
            total += pointer["size"]
    print("all files present, LFS files SHA-256-checked: {:.2f} GB".format(total / 1e9))


# --------------------------------------------------------------------------------------------
# Read (10 fps, SkillCorner's own frames)
# --------------------------------------------------------------------------------------------

def match_info(match):
    """(pitch (length, width) in metres, {player id: HOME/AWAY}, [(period, first, last) 10 fps frames],
    title)."""
    with open(raw_path(match, "match.json")) as handle:
        info = json.load(handle)
    side = {info["home_team"]["id"]: HOME, info["away_team"]["id"]: AWAY}
    team_of = {p["id"]: side[p["team_id"]] for p in info["players"] if p["team_id"] in side}
    periods = [(int(p["period"]), int(p["start_frame"]), int(p["end_frame"])) for p in info["match_periods"]
               if int(p["period"]) in (1, 2)]
    title = "{} - {} ({})".format(info["home_team"]["short_name"], info["away_team"]["short_name"],
                                  info["date_time"][:10])
    return (float(info["pitch_length"]), float(info["pitch_width"])), team_of, periods, title


def read_tracking(match, team_of):
    """Every 10 fps frame of the tracking file, as arrays indexed by SkillCorner's frame number:
    players [n, slots, 2] (NaN where not listed), player detected [n, slots], ball [n, 2] (NaN where
    null), ball detected [n], group [n] (HOME / AWAY / -1 for null), no data [n] (no player listed),
    and the slots' player ids. Coordinates are SkillCorner's (centre origin)."""
    ids = sorted(team_of)
    slot = {pid: i for i, pid in enumerate(ids)}
    group_code = {"home team": HOME, "away team": AWAY, None: -1}
    records, unknown = [], 0
    with open(raw_path(match, "tracking_extrapolated.jsonl")) as handle:
        for line in handle:
            r = json.loads(line)
            ball = r.get("ball_data") or {}
            people = [(slot[p["player_id"]], p["x"], p["y"], bool(p["is_detected"])) for p in r.get("player_data") or []
                      if p["player_id"] in slot and p["x"] is not None]
            unknown += sum(1 for p in r.get("player_data") or [] if p["player_id"] not in slot)
            records.append((int(r["frame"]), ball.get("x"), ball.get("y"), bool(ball.get("is_detected")),
                            group_code[(r.get("possession") or {}).get("group")], people))
    n = max(f for f, *_ in records) + 1
    players = np.full((n, len(ids), 2), np.nan, dtype=np.float64)
    detected = np.zeros((n, len(ids)), bool)
    ball = np.full((n, 2), np.nan, dtype=np.float64)
    ball_detected = np.zeros(n, bool)
    group = np.full(n, -1, dtype=np.int8)
    no_data = np.ones(n, bool)
    for f, bx, by, bdet, g, people in records:
        if bx is not None and by is not None:
            ball[f] = (bx, by)
        ball_detected[f] = bdet and bx is not None
        group[f] = g
        no_data[f] = not people
        for s, x, y, det in people:
            players[f, s] = (x, y)
            detected[f, s] = det
    return {"players": players, "detected": detected, "ball": ball, "ball_detected": ball_detected,
            "group": group, "no_data": no_data, "ids": np.array(ids), "unknown_people": unknown}


def read_phases(match, n):
    """bool [n]: 10 fps frames inside a phase of play (frame_start..frame_end inclusive), and the phase's
    team in possession per frame (HOME / AWAY / -1), from the phases file."""
    import csv
    with open(raw_path(match, "match.json")) as handle:
        info = json.load(handle)
    side = {str(info["home_team"]["id"]): HOME, str(info["away_team"]["id"]): AWAY}
    live = np.zeros(n, bool)
    phase_team = np.full(n, -1, dtype=np.int8)
    with open(raw_path(match, "phases_of_play.csv")) as handle:
        for row in csv.DictReader(handle):
            a, b = int(row["frame_start"]), int(row["frame_end"])
            live[a:b + 1] = True
            phase_team[a:b + 1] = side.get(str(row["team_in_possession_id"]), -1)
    return live, phase_team


def fill_players(players, no_data, first, last):
    """Within one period (10 fps frames first..last), each player's position across the frames with no
    tracking at all (broadcast cut-aways): linear between the frames around, held at the period's ends.
    A player absent from a frame WITH tracking (not yet on, or off) is left absent."""
    span = slice(first, last + 1)
    p, gap = players[span], no_data[span]
    t = np.arange(len(p))
    with_data = np.flatnonzero(~gap)
    if not len(with_data):
        return
    for s in range(p.shape[1]):
        seen = ~np.isnan(p[:, s, 0])
        if not seen.any():
            continue
        lo, hi = np.flatnonzero(seen)[[0, -1]]
        # hold at the period's ends only if no frame with tracking before / after lacks him
        a = 0 if lo <= with_data[0] else lo
        b = len(p) - 1 if hi >= with_data[-1] else hi
        inside = np.arange(a, b + 1)
        todo = inside[~seen[inside]]
        if len(todo):
            for k in (0, 1):
                p[todo, s, k] = np.interp(todo, t[seen], p[seen, s, k])
    players[span] = p


def to_25fps(values, first, last):
    """values over 10 fps frames first..last -> 25 fps: each 25 fps frame at its time on the 10 fps clock,
    linear between the two 10 fps frames around it; NaN if either is NaN (exact 10 fps times copy)."""
    v = values[first:last + 1]
    n25 = int(np.floor((len(v) - 1) * FPS / SOURCE_FPS + 1e-9)) + 1
    s = np.arange(n25) * (SOURCE_FPS / FPS)
    lo = np.floor(s + 1e-9).astype(int)
    hi = np.minimum(lo + 1, len(v) - 1)
    w = (s - lo).reshape((-1,) + (1,) * (v.ndim - 1))
    out = (1 - w) * v[lo] + w * v[hi]
    exact = (np.abs(s - lo) < 1e-9).reshape(w.shape)
    return np.where(exact, v[lo], out)


def nearest_25fps(values, first, last):
    """Categorical values over 10 fps frames first..last -> 25 fps by the nearest 10 fps frame."""
    v = values[first:last + 1]
    n25 = int(np.floor((len(v) - 1) * FPS / SOURCE_FPS + 1e-9)) + 1
    return v[np.minimum(np.rint(np.arange(n25) * (SOURCE_FPS / FPS)).astype(int), len(v) - 1)]


def labels_10fps(group, live, phase_team, first, last):
    """Scorer codes over 10 fps frames first..last: D outside phases of play; inside, the raw group,
    nulls carrying the last team of that live stretch (leading nulls: the first team written in it; a
    stretch with no team at all: the phase's team). Returns (codes, counts of what was filled)."""
    g, alive, pt = group[first:last + 1], live[first:last + 1], phase_team[first:last + 1]
    out = np.full(len(g), pm.D, dtype=np.int8)
    counts = {"null_in_live": 0, "backfilled": 0, "from_phase": 0, "team_in_dead": int(np.sum(~alive & (g >= 0)))}
    edges = np.flatnonzero(np.diff(np.r_[0, alive.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        team = g[a:b].astype(np.int16)
        known = team >= 0
        counts["null_in_live"] += int((~known).sum())
        if not known.any():
            team = pt[a:b].astype(np.int16)
            counts["from_phase"] += b - a
        else:
            carry = np.maximum.accumulate(np.where(known, np.arange(b - a), -1))
            lead = carry < 0
            counts["backfilled"] += int(lead.sum())
            carry[lead] = np.flatnonzero(known)[0]
            team = team[carry]
        out[a:b] = np.where(team == HOME, pm.H, np.where(team == AWAY, pm.A, pm.D))
    return out, counts


def player_speed(xy):
    """m/s from positions over 2 * SPEED_SPAN frames at 25 fps (bench/metrica.py's)."""
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    step = np.hypot(*(xy[2 * SPEED_SPAN:] - xy[:-2 * SPEED_SPAN]).transpose(2, 0, 1))
    speed[SPEED_SPAN:-SPEED_SPAN] = step / (2 * SPEED_SPAN / FPS)
    return speed


def build(match):
    """One match as arrays shaped like possession.data.load_tracking, at 25 fps, plus its labels.

    players float32 [frames, slots, 2] metres, corner origin; speed float32 [frames, slots] m/s;
    ball float32 [frames, 2] (NaN where SkillCorner has none); team int8 [slots]; labels int8 (scorer
    codes, X between the halves); halves [(half, first, last)]; pitch (length, width); and for the
    report: no_data (no tracking at all on the nearest 10 fps frame), ball_detected, players_detected
    (share of slots on the pitch that are detected), and the label fill counts."""
    pitch, team_of, periods, title = match_info(match)
    raw = read_tracking(match, team_of)
    n10 = len(raw["group"])
    live, phase_team = read_phases(match, n10)
    offset = np.array([pitch[0] / 2, pitch[1] / 2])
    raw["players"] += offset
    raw["ball"] += offset

    pieces, halves, length, counts = [], [], 0, {}
    for half, first, last in sorted(periods):
        fill_players(raw["players"], raw["no_data"], first, last)
        codes, c = labels_10fps(raw["group"], live, phase_team, first, last)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + int(v)
        players = to_25fps(raw["players"], first, last)
        on = ~np.isnan(raw["players"][first:last + 1, :, 0])
        det_share = np.where(on.sum(axis=1) > 0, (raw["detected"][first:last + 1] & on).sum(axis=1) / np.maximum(on.sum(axis=1), 1), 0.0)
        piece = {"players": players, "ball": to_25fps(raw["ball"], first, last),
                 "labels": nearest_25fps(codes, 0, len(codes) - 1),
                 "no_data": nearest_25fps(raw["no_data"], first, last),
                 "ball_detected": nearest_25fps(raw["ball_detected"], first, last),
                 "players_detected": nearest_25fps(det_share.astype(np.float32), 0, len(det_share) - 1),
                 "labels_10fps": codes}
        pieces.append(piece)
        n = len(players)
        halves.append((half, length, length + n - 1))
        length += n + 1
    n_frames = length - 1
    slots = raw["players"].shape[1]
    out = {"players": np.full((n_frames, slots, 2), np.nan, dtype=np.float32),
           "ball": np.full((n_frames, 2), np.nan, dtype=np.float32),
           "labels": np.full(n_frames, pm.X, dtype=np.int8),
           "no_data": np.zeros(n_frames, bool), "ball_detected": np.zeros(n_frames, bool),
           "players_detected": np.zeros(n_frames, np.float32)}
    for (_, a, b), piece in zip(halves, pieces):
        for key in out:
            out[key][a:b + 1] = piece[key]
    speed = np.full((n_frames, slots), np.nan, dtype=np.float32)
    for _, a, b in halves:
        speed[a:b + 1] = player_speed(out["players"][a:b + 1])
    team = np.array([team_of[pid] for pid in raw["ids"]], dtype=np.int8)
    keep = ~np.isnan(out["players"][:, :, 0]).all(axis=0)      # drop players never on the pitch
    out.update(players=out["players"][:, keep], speed=speed[:, keep], team=team[keep], player_ids=raw["ids"][keep],
               halves=np.array(halves), pitch=np.array(pitch), title=title,
               labels_10fps=np.concatenate([p["labels_10fps"] for p in pieces]),
               fill_counts=json.dumps(dict(counts, unknown_people=int(raw["unknown_people"]))))
    return out


def load(match):
    """A cached match (see build). Missing cache -> run `python -m bench.skillcorner parse` first."""
    with np.load(cache_path(match)) as data:
        out = {key: data[key] for key in data.files}
    out["halves"] = [tuple(int(v) for v in row) for row in out["halves"]]
    out["pitch"] = tuple(float(v) for v in out["pitch"])
    out["title"] = str(out["title"])
    out["fill_counts"] = json.loads(str(out["fill_counts"]))
    return out


def tracking(data):
    """The dict v12.predict reads, as possession.data.load_tracking returns it."""
    return {"players": data["players"], "speed": data["speed"], "ball": data["ball"], "team": data["team"]}


# --------------------------------------------------------------------------------------------
# Label checks (no model involved)
# --------------------------------------------------------------------------------------------

REACH = 1.0          # m (bench/labels.py's reading check, bench/transfer_check.py)
SLOW = 2.0           # m/s
LOOK = 5             # frames: ball speed over 0.2 s (transfer_check's)
CLEAR = 3.0          # m: nobody of the other team this close (transfer_check's)


def label_checks(data):
    """Possessions per graded minute, dead share, possession lengths, and two readings of the labels
    at clear on-the-ball frames (ball detected by SkillCorner, slow, a player within 1 m): the reading
    check (bench/labels.py: the nearest player's team is the labelled team) and the clear one
    (bench/transfer_check.py: nobody of the other team within 3 m)."""
    from bench.labels import runs_of
    gt, halves = data["labels"], data["halves"]
    graded = gt != pm.X
    spells = [b - a for _, f, l in halves for a, b, v in runs_of(gt[f:l + 1]) if v in (pm.H, pm.A)]
    spells = np.array(spells) / FPS
    minutes = graded.sum() / FPS / 60
    ball, players, team = data["ball"], data["players"], data["team"]
    distance = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))
    dist = np.where(np.isnan(distance), np.inf, distance)
    speed = np.full(len(ball), np.nan)
    speed[:-LOOK] = np.hypot(*(ball[LOOK:] - ball[:-LOOK]).T) * (FPS / LOOK)
    near_home = np.min(np.where(team[None, :] == HOME, dist, np.inf), axis=1)
    near_away = np.min(np.where(team[None, :] == AWAY, dist, np.inf), axis=1)
    alive = np.isin(gt, (pm.H, pm.A))
    with np.errstate(invalid="ignore"):
        slow = (speed < SLOW) & data["ball_detected"] & ~data["no_data"]
    nearest_home = near_home < near_away
    close = np.minimum(near_home, near_away) <= REACH
    reading = alive & slow & close
    home_clear = (near_home < REACH) & (near_away > CLEAR) & slow
    away_clear = (near_away < REACH) & (near_home > CLEAR) & slow
    clear = alive & (home_clear | away_clear)
    return {"minutes": float(minutes), "possessions": int(len(spells)),
            "possessions per minute": float(len(spells) / minutes),
            "dead share": float(np.mean(gt[graded] == pm.D)),
            "median possession s": float(np.median(spells)), "under 1 s": float(np.mean(spells < 1.0)),
            "reading check": float(np.mean(np.where(nearest_home, pm.H, pm.A)[reading] == gt[reading])),
            "reading frames": int(reading.sum()),
            "clear control": float((np.sum(alive & home_clear & (gt == pm.H)) + np.sum(alive & away_clear & (gt == pm.A)))
                                   / max(clear.sum(), 1)),
            "clear frames": int(clear.sum()),
            "no tracking share": float(np.mean(data["no_data"][graded])),
            "no tracking, live share": float(np.mean(alive[graded & data["no_data"]])) if (graded & data["no_data"]).any() else 0.0,
            "ball missing share": float(np.mean(np.isnan(ball[graded, 0]))),
            "ball detected share": float(np.mean(data["ball_detected"][graded])),
            "players detected share": float(np.mean(data["players_detected"][graded & ~data["no_data"]])),
            **data["fill_counts"]}


def parse(only=None):
    os.makedirs(CACHE, exist_ok=True)
    rows = {}
    for match in only or match_ids():
        if not os.path.exists(raw_path(match, "tracking_extrapolated.jsonl")):
            print("{}: tracking not downloaded yet".format(match), flush=True)
            continue
        data = build(match)
        np.savez_compressed(cache_path(match), **data)
        data = load(match)
        rows[match] = dict(label_checks(data), title=data["title"])
        r = rows[match]
        print("{} {:<40s} {:5.1f} min  poss/min {:.2f}  dead {:.3f}  median {:.1f} s  <1 s {:.1%}  reading {:.3f} ({})  "
              "clear {:.3f} ({})  no tracking {:.1%}  ball missing {:.1%}  null-in-live {}  backfilled {}  from phase {}".format(
                  match, r["title"], r["minutes"], r["possessions per minute"], r["dead share"], r["median possession s"],
                  r["under 1 s"], r["reading check"], r["reading frames"], r["clear control"], r["clear frames"],
                  r["no tracking share"], r["ball missing share"], r["null_in_live"], r["backfilled"], r["from_phase"]), flush=True)
    path = os.path.join(ROOT, "bench", "results", "lab_skillcorner_labels.json")
    old = {}
    if os.path.exists(path):
        with open(path) as handle:
            old = json.load(handle)
    old.update(rows)
    with open(path, "w") as handle:
        json.dump(old, handle, indent=1)


if __name__ == "__main__":
    {"download": download, "parse": lambda: parse(sys.argv[2:] or None)}[sys.argv[1]]()
