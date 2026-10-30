"""Measure every provided data file. Every number in docs/data.md comes from this script.

    python tools/explore_data.py              # tactical + broadcast + public ground truth
    python tools/explore_data.py --no-video   # skip the ffprobe calls (no ffmpeg installed)

It reads only the provided files and the public 5-minute ground truth that ships in this
repository. The ground truth is used here to *describe* what the labels mean (for example "is the
ball visible during dead-ball frames?"), never to choose a setting of any model.

Takes about 15 seconds: the two raw tracking files are parsed into arrays first.
"""

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]  # metres; used to turn 0-1 coordinates into metres
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
# How far (seconds) a set-piece tag's start may sit from a restart and still be counted as its tag.
RESTART_MATCH_S = 15


def section(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)


# --------------------------------------------------------------------------------------------
# Tracking: metadata XML + raw text
# --------------------------------------------------------------------------------------------

def read_metadata(meta_path):
    """(team names in file order, team index per slot, attack directions, ball channels)."""
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    team_index = {tid: i for i, (tid, _) in enumerate(teams)}
    slot_team = np.array([team_index[p.get("teamId")] for p in root.iter("Player")])
    directions = {}
    for t in root.iter("Team"):
        for p in t.iter("ProviderParameter"):
            if p.findtext("Name") == "attack_direction_first_half":
                directions[t.findtext("Name")] = p.findtext("Value")
    units = {c.get("id"): c.findtext("Unit") for c in root.iter("Channel")}
    ball_channels = [b.get("channelId") for b in root.iter("BallChannelRef")]
    return [name for _, name in teams], slot_team, directions, units, ball_channels


def read_raw(raw_path):
    """Parse '<frame>:<x,y,s;x,y,s;...>:<x,y,s>' lines into (ids, players[T,slots,3], ball[T,3])."""
    ids, rows = [], []
    with open(raw_path) as handle:
        for line in handle:
            frame, players, ball = line.rstrip("\n").split(":")
            ids.append(int(frame))
            rows.append(players.replace(";", ",") + "," + ball)
    values = np.array([np.array(r.split(","), dtype=np.float32) for r in rows])
    return np.array(ids), values[:, :-3].reshape(len(rows), -1, 3), values[:, -3:]


def runs(mask):
    """Lengths of the runs of True in a boolean array."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.where(edges == -1)[0] - np.where(edges == 1)[0]


def describe_tracking(source, graded):
    paths = asset_paths(source=source)
    names, slot_team, directions, units, ball_channels = read_metadata(paths["atd_meta"])
    ids, players, ball = read_raw(paths["atd_raw"])
    section("TRACKING ({}): {} + {}".format(source, os.path.basename(paths["atd_meta"]),
                                            os.path.basename(paths["atd_raw"])))
    print("teams in file order: {}  attack direction 1st half: {}".format(names, directions))
    print("slots: {} total, per team {}".format(len(slot_team),
                                                 dict(collections.Counter(names[t] for t in slot_team))))
    print("channel units claimed by the metadata: {}   ball channels: {}".format(units, ball_channels))
    print("lines {}  frame ids {}..{}  contiguous: {}".format(
        len(ids), ids[0], ids[-1], bool(np.all(np.diff(ids) == 1))))
    # The broadcast has no graded window of its own (it is not on the submission timeline), so its
    # numbers are over every frame.
    scope = "graded frame" if graded is not None else "frame"
    if graded is None:
        graded = np.ones(len(ids), bool)

    seen = ~np.isnan(players[:, :, 0])
    per_frame = seen[graded].sum(1)
    print("players visible per {}: mean {:.1f}  min {}  max {}".format(
        scope, per_frame.mean(), per_frame.min(), per_frame.max()))
    for t, name in enumerate(names):
        count = seen[graded][:, slot_team == t].sum(1)
        print("  {:13s} mean {:.1f}   frames with <8 visible {:.1f}%   with >11 visible {:.1f}%".format(
            name, count.mean(), 100 * np.mean(count < 8), 100 * np.mean(count > 11)))

    speed = players[:, :, 2][seen]
    print("player speed column: median {:.2f}  p99 {:.2f}  max {:.2f}  -> metres per second, "
          "whatever the metadata says   (empty where position exists: {:.1f}%)".format(
              np.nanmedian(speed), np.nanpercentile(speed, 99), np.nanmax(speed),
              100 * np.mean(np.isnan(speed))))
    step = np.hypot(np.diff(players[:, :, 0] * PITCH_LENGTH, axis=0),
                    np.diff(players[:, :, 1] * PITCH_WIDTH, axis=0)) * FPS
    column = players[1:, :, 2]
    both = ~np.isnan(step) & ~np.isnan(column)
    print("player speed column vs speed computed from positions: median ratio {:.2f}, correlation {:.2f}".format(
        np.median(column[both] / np.maximum(step[both], 1e-3)), np.corrcoef(column[both], step[both])[0, 1]))
    team_mean = np.nanmean(np.where(seen, players[:, :, 2], np.nan)[graded], axis=1)
    print("mean speed of all visible players per {}: percentiles 5/25/50/75/95 = {}".format(
        scope, np.round(np.nanpercentile(team_mean, [5, 25, 50, 75, 95]), 2)))
    x, y = players[:, :, 0][seen], players[:, :, 1][seen]
    print("player x {:.3f}..{:.3f}  y {:.3f}..{:.3f}  (0-1 is the pitch; outside = off the pitch)".format(
        x.min(), x.max(), y.min(), y.max()))

    tracklets = [runs(seen[:, s]) for s in range(seen.shape[1])]
    lengths = np.concatenate(tracklets)
    print("tracklets (unbroken stretches of one slot): {} total, mean {:.0f} s, "
          "per slot median {:.0f}, max {}".format(
              len(lengths), lengths.mean() / FPS, np.median([len(t) for t in tracklets]),
              max(len(t) for t in tracklets)))

    has_ball = ~np.isnan(ball[:, 0])
    print("ball visible: {:.1f}% of all frames, {:.1f}% of graded frames".format(
        100 * has_ball.mean(), 100 * has_ball[graded].mean()))
    gap_mask = ~has_ball & graded
    gaps = runs(gap_mask)
    print("ball gaps inside graded frames: {} gaps, median {:.1f} s, p90 {:.1f} s, longest {:.1f} s; "
          "gaps longer than 2 s hold {:.1f}% of the missing frames".format(
              len(gaps), np.median(gaps) / FPS, np.percentile(gaps, 90) / FPS, gaps.max() / FPS,
              100 * gaps[gaps > 2 * FPS].sum() / gaps.sum()))
    print("ball speed column: filled on {:.1f}% of ball frames, median {:.2f}, max {:.2f} "
          "-> not usable".format(100 * np.mean(~np.isnan(ball[has_ball, 2])),
                                 np.nanmedian(ball[has_ball, 2]), np.nanmax(ball[has_ball, 2])))
    bx, by = ball[:, 0] * PITCH_LENGTH, ball[:, 1] * PITCH_WIDTH
    step = np.hypot(np.diff(bx), np.diff(by)) * FPS
    print("ball speed computed from positions (m/s): median {:.1f}  p90 {:.1f}  p99 {:.1f}  max {:.0f}".format(
        *np.nanpercentile(step, [50, 90, 99, 100])))
    wobble = np.hypot(np.diff(bx, 2), np.diff(by, 2))
    print("ball frame-to-frame wobble (second difference) median {:.3f} m -> the track is smoothed".format(
        np.nanmedian(wobble)))
    out = (ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1)
    print("ball outside the pitch lines: {:.1f}% of ball frames".format(100 * np.mean(out[has_ball])))
    return paths, has_ball


# --------------------------------------------------------------------------------------------
# Homography
# --------------------------------------------------------------------------------------------

def describe_homography(source, graded):
    path = asset_paths(source=source)["homography"]
    section("HOMOGRAPHY ({}): {}".format(source, os.path.basename(path)))
    table = pd.read_csv(path)
    print("columns: {}".format(list(table.columns)))
    error = table.reprojection_error
    print("rows {}  valid {:.1f}%  reprojection error median {:.2f} px, p99 {:.2f}, max {:.1f}".format(
        len(table), 100 * table.valid.mean(), error.median(), error.quantile(0.99), error.max()))
    if source == "tactical":
        matrices = table[["h{}{}".format(i, j) for i in range(3) for j in range(3)]].values.reshape(-1, 3, 3)
        first_graded = int(np.argmax(graded))
        print("where the pitch lands in the picture at the first graded frame ({}):".format(first_graded))
        for name, (px, py) in (("corner x=0,y=0", (0, 0)), ("corner x=1,y=0", (1, 0)),
                               ("corner x=0,y=1", (0, 1)), ("corner x=1,y=1", (1, 1)),
                               ("centre spot", (0.5, 0.5))):
            p = matrices[first_graded] @ np.array([px, py, 1.0])
            print("  {:15s} -> pixel ({:6.0f}, {:5.0f})   (picture is 1920 x 1080)".format(name, p[0] / p[2], p[1] / p[2]))
    return table


# --------------------------------------------------------------------------------------------
# Smart Tagging
# --------------------------------------------------------------------------------------------

def read_tags(path):
    root = ET.parse(path).getroot()
    tags = []
    for inst in root.iter("instance"):
        labels = {l.findtext("group"): l.findtext("text") for l in inst.iter("label")}
        tags.append({"start": float(inst.findtext("start")), "end": float(inst.findtext("end")),
                     "code": inst.findtext("code"), **labels})
    return root, tags


def describe_tagging(graded, gt):
    path = asset_paths()["smart_tagging"]
    section("SMART TAGGING (tactical): " + os.path.basename(path))
    root, tags = read_tags(path)
    print("top-level sections: {}   instances: {}".format([c.tag for c in root], len(tags)))
    print("ROWS section holds display colours only: {}".format(
        [(r.findtext("code"), r.findtext("R")) for r in root.find("ROWS")][:3]))

    durations = collections.defaultdict(list)
    for t in tags:
        durations[t["code"]].append(t["end"] - t["start"])
    print("\n{:30s} {:>4s}  {:>8s}  {:>12s}".format("code", "n", "median s", "range s"))
    for code, d in sorted(durations.items(), key=lambda kv: -len(kv[1])):
        print("{:30s} {:4d}  {:8.1f}  {:5.1f}-{:5.1f}".format(code, len(d), np.median(d), min(d), max(d)))

    print("\nteam label counts: {}".format(dict(collections.Counter(t.get("Team") for t in tags))))
    print("set-piece types: {}".format(dict(collections.Counter(t.get("Type") for t in tags if t["code"] == "SET PIECES"))))
    print("set-piece window lengths by type:")
    lengths = collections.defaultdict(collections.Counter)
    for t in tags:
        if t["code"] == "SET PIECES":
            lengths[t["Type"]][round(t["end"] - t["start"], 1)] += 1
    for kind, c in lengths.items():
        print("  {:22s} {}".format(kind, dict(c)))

    by_interval = collections.defaultdict(list)
    for t in tags:
        by_interval[(t["start"], t["end"])].append(t)
    pairs = collections.Counter()
    for group in by_interval.values():
        if len(group) == 2:
            a, b = group
            pairs[(" + ".join(sorted([a["code"], b["code"]])),
                   "opposite teams" if a.get("Team") != b.get("Team") else "SAME team")] += 1
    print("\nmirrored pairs (two tags with identical start and end):")
    for (codes, teams), n in pairs.most_common():
        print("  {:3d}  {:55s} {}".format(n, codes, teams))

    n = len(graded)
    any_tag = np.zeros(n, bool)
    team_cover = collections.defaultdict(lambda: np.zeros(n, bool))
    for t in tags:
        a, b = round(t["start"] * FPS), round(t["end"] * FPS)
        any_tag[a:b] = True
        if t.get("Team") not in (None, "N/A"):
            team_cover[t["Team"]][a:b] = True
    names = sorted(team_cover)
    print("\ngraded frames under any tag {:.1f}%; under a {} tag {:.1f}%; under a {} tag {:.1f}%; "
          "under both {:.1f}%".format(100 * any_tag[graded].mean(), names[0], 100 * team_cover[names[0]][graded].mean(),
                                      names[1], 100 * team_cover[names[1]][graded].mean(),
                                      100 * (team_cover[names[0]] & team_cover[names[1]])[graded].mean()))

    # Restarts in the public window: the first alive frame after a dead stretch. Compared with the
    # set-piece tags to see where a tag sits relative to the restart it describes.
    window = np.where(gt != "X")[0]
    labels = gt[window]
    restarts = [int(window[i]) for i in range(1, len(window)) if labels[i - 1] == "D" and labels[i] != "D"]
    restarts = [int(window[0])] + restarts  # the opening kickoff is a restart too
    print("\npublic window restarts (first alive frame after D, plus the kickoff): {}".format(restarts))
    # A tag "belongs" to a restart if it starts within RESTART_MATCH_S seconds before or after it.
    for r in restarts:
        near = [t for t in tags if t["code"] == "SET PIECES"
                and abs(round(t["start"] * FPS) - r) <= RESTART_MATCH_S * FPS]
        if near:
            t = min(near, key=lambda t: abs(round(t["start"] * FPS) - r))
            print("  restart {:6d}: SET PIECES {:22s} team {:13s} tag starts {:+.1f} s from the restart".format(
                r, t["Type"], t.get("Team"), (round(t["start"] * FPS) - r) / FPS))
        else:
            print("  restart {:6d}: no set-piece tag within {} s".format(r, RESTART_MATCH_S))
    return tags


# --------------------------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------------------------

def describe_video(source):
    path = asset_paths(source=source)["video"]
    section("VIDEO ({}): {}".format(source, os.path.basename(path)))
    if not os.path.exists(path):
        print("not downloaded")
        return
    print("size {:.1f} GB".format(os.path.getsize(path) / 1e9))
    if not shutil.which("ffprobe"):
        print("ffprobe not installed; skipping stream details")
        return
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-show_entries",
                            "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_packets:format=duration",
                            "-of", "json", path], capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    for s in info["streams"]:
        if s["codec_type"] == "video":
            print("video: {} {}x{} at {} fps, {} frames".format(s["codec_name"], s["width"], s["height"],
                                                               s["r_frame_rate"], s["nb_read_packets"]))
        else:
            print("{}: {}".format(s["codec_type"], s["codec_name"]))
    print("duration {:.0f} s ({:.1f} h)".format(float(info["format"]["duration"]),
                                                float(info["format"]["duration"]) / 3600))


# --------------------------------------------------------------------------------------------
# Public ground truth, and how the ball relates to it
# --------------------------------------------------------------------------------------------

def describe_public_gt(gt, has_ball):
    section("PUBLIC GROUND TRUTH: " + os.path.basename(PUBLIC_GT))
    window = np.where(gt != "X")[0]
    print("labelled frames {}..{} ({} frames, {:.1f} min); counts {}".format(
        window[0], window[-1], len(window), len(window) / FPS / 60,
        dict(collections.Counter(gt[window]))))
    segments, start = [], window[0]
    for i in range(window[0] + 1, window[-1] + 2):
        if i > window[-1] or gt[i] != gt[start]:
            segments.append((start, i - 1, gt[start]))
            start = i
    possessions = [(a, b) for a, b, l in segments if l in "HA"]
    print("segments {}  possessions {}  mean possession {:.1f} s  shortest {:.1f} s  longest {:.1f} s".format(
        len(segments), len(possessions), np.mean([b - a + 1 for a, b in possessions]) / FPS,
        min(b - a + 1 for a, b in possessions) / FPS, max(b - a + 1 for a, b in possessions) / FPS))
    print("ball visible in the tracking over the whole window: {:.1f}%".format(100 * has_ball[window].mean()))
    print("ball visible in the tracking, by true label:")
    for label in "HAD":
        mask = gt == label
        print("  {}: {:5d} frames, ball visible {:.1f}%".format(label, mask.sum(), 100 * has_ball[mask].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-video", action="store_true", help="skip ffprobe")
    args = parser.parse_args()

    n = GAMES[GAME]["n_video_frames"]
    graded = np.zeros(n, bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    gt = pd.read_csv(PUBLIC_GT).label.values

    _, has_ball = describe_tracking("tactical", graded)
    describe_homography("tactical", graded)
    describe_tagging(graded, gt)
    describe_public_gt(gt, has_ball)
    _, bro_ball = describe_tracking("broadcast", None)
    bro_h = describe_homography("broadcast", None)
    valid = bro_h.valid.values.astype(bool)
    print("broadcast ball visible on valid-homography frames (camera shows the pitch): {:.1f}%".format(
        100 * bro_ball[valid].mean()))
    if not args.no_video:
        describe_video("tactical")
        describe_video("broadcast")


if __name__ == "__main__":
    main()
