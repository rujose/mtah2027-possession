"""Referee whistles from the tactical video's audio track (used by model v2).

The track is very quiet (mean around -56 to -63 dB) but clean, so a whistle stands out as one sharp
tone between 2.5 and 4.5 kHz. Steps, each explained in docs/models/v2-whistle.md:

  1. decode to 16 kHz mono with PyAV (pinned: it ships its own ffmpeg, so every machine decodes
     the same numbers)
  2. one spectrum per video frame (640 samples = 40 ms), Hann window
  3. per frame: prominence (loudest bin in the band vs the band's median) and level (that peak vs
     the match's typical peak)
  4. tone = >= 3 consecutive graded frames over both thresholds; event = tones < 2 s apart, joined
  5. an event COUNTS only if the players slow down after it (crowd whistles don't stop players)

The audio starts at time 0 like the video, so sample i belongs to video frame i // 640.
"""

import av
import numpy as np

from possession.data import FPS, graded_halves
from games import asset_paths

# Params
RATE = 16000                 # samples per second after decoding
HOP = RATE // FPS            # 640 samples: exactly one video frame
BAND = (2500.0, 4500.0)      # Hz, where referee whistles sound (tones found at 2.6-3.9 kHz)
MIN_PROMINENCE = 25.0        # dB above the band's median; far tail of the audio's own distribution
MIN_LEVEL = 15.0             # dB above the match's typical peak
MIN_TONE_FRAMES = 3          # 0.12 s; single-frame clicks are noise
MERGE_GAP = 2 * FPS          # referees give 2-3 short blasts for one stoppage
SPEED_BEFORE = 3 * FPS       # mean player speed over the 3 s before the whistle...
SPEED_AFTER = (2 * FPS, 6 * FPS)  # ...against 2-6 s after it
MIN_SLOWDOWN = 0.5           # m/s; 66 of 86 events change by less than +-0.5, the noise band
CHUNK = 10000                # frames per FFT batch, to bound memory


def decode(path):
    """The whole audio track as int16 mono samples at RATE."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    chunks = []
    with av.open(path) as container:
        for frame in container.decode(container.streams.audio[0]):
            chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))  # flush
    return np.concatenate(chunks)


def frame_features(samples, n_frames):
    """(prominence, peak) in dB per video frame; NaN past the end of the audio."""
    usable = min(len(samples) // HOP, n_frames)
    window = np.hanning(HOP)
    freqs = np.fft.rfftfreq(HOP, 1.0 / RATE)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    prominence = np.full(n_frames, np.nan)
    peak = np.full(n_frames, np.nan)
    for start in range(0, usable, CHUNK):
        stop = min(start + CHUNK, usable)
        block = samples[start * HOP:stop * HOP].astype(np.float64).reshape(-1, HOP)
        db = 20.0 * np.log10(np.abs(np.fft.rfft(block * window, axis=1))[:, band] + 1e-6)
        prominence[start:stop] = db.max(axis=1) - np.median(db, axis=1)
        peak[start:stop] = db.max(axis=1)
    return prominence, peak


def mean_player_speed(tracking):
    speed = tracking["speed"]
    seen = ~np.isnan(speed)
    count = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, np.where(seen, speed, 0.0).sum(axis=1) / count, np.nan)


def runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.where(edges == 1)[0], np.where(edges == -1)[0]


def whistle_events(tracking, samples=None):
    """Whistle events over the graded frames.

    Returns a list of dicts: start, end (exclusive) video frame, speed_before, speed_after, counted.
    `samples` can be passed in to avoid decoding twice.
    """
    n_frames = len(tracking["ball"])
    graded = np.zeros(n_frames, bool)
    for _, first, last in graded_halves():
        graded[first:last + 1] = True
    if samples is None:
        samples = decode(asset_paths()["video"])
    prominence, peak = frame_features(samples, n_frames)
    level = peak - np.nanmedian(peak[graded])
    with np.errstate(invalid="ignore"):
        loud = (prominence > MIN_PROMINENCE) & (level > MIN_LEVEL) & graded

    events = []
    for start, end in zip(*runs(loud)):
        if end - start < MIN_TONE_FRAMES:
            continue
        if events and start - events[-1]["end"] < MERGE_GAP:
            events[-1]["end"] = int(end)
        else:
            events.append({"start": int(start), "end": int(end)})

    speed = mean_player_speed(tracking)
    for event in events:
        s = event["start"]
        before = speed[max(0, s - SPEED_BEFORE):s]
        after = speed[s + SPEED_AFTER[0]:s + SPEED_AFTER[1]]
        event["speed_before"] = float(np.nanmean(before)) if np.any(~np.isnan(before)) else float("nan")
        event["speed_after"] = float(np.nanmean(after)) if np.any(~np.isnan(after)) else float("nan")
        event["counted"] = bool(event["speed_before"] - event["speed_after"] >= MIN_SLOWDOWN)
    return events
