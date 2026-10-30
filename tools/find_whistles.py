"""List whistle-like tones in the tactical video's audio, as video frames.

    python tools/find_whistles.py 11000 7800     # start frame, number of frames

The tactical audio is very quiet (mean around -56 to -63 dB, peaks around -44 dB) but clean, so a
referee's whistle stands out as a narrow tone. One spectrum per video frame (40 ms window, 16 kHz
mono): the score is how far the loudest bin between 2.5 and 4.5 kHz stands above that band's median,
and how loud that peak is compared with the whole clip. Exploration only: the thresholds below are
eyeballed, not fitted, and crowd whistles also trigger it.
"""

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402

# Params
RATE, FPS = 16000, 25
BAND = (2500, 4500)      # Hz: where referee whistles put their energy
MIN_PROMINENCE = 25      # dB the peak stands above the band median
MIN_LEVEL = 15           # dB the peak stands above the clip's typical peak
MIN_FRAMES = 3           # consecutive frames (0.12 s)


def main():
    start, count = int(sys.argv[1]), int(sys.argv[2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start / FPS), "-t", str(count / FPS),
                          "-i", asset_paths()["video"], "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, np.int16).astype(np.float32)
    hop = RATE // FPS
    frames = len(audio) // hop
    spectrum = np.abs(np.fft.rfft(audio[:frames * hop].reshape(frames, hop) * np.hanning(hop), axis=1)) + 1e-6
    freqs = np.fft.rfftfreq(hop, 1 / RATE)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    db = 20 * np.log10(spectrum[:, band])
    prominence = db.max(axis=1) - np.median(db, axis=1)
    level = db.max(axis=1) - np.median(db.max(axis=1))
    hit = np.r_[(prominence > MIN_PROMINENCE) & (level > MIN_LEVEL), False]
    print("{} frames analysed".format(frames))
    begin = None
    for i, h in enumerate(hit):
        if h and begin is None:
            begin = i
        elif not h and begin is not None:
            if i - begin >= MIN_FRAMES:
                hz = freqs[band][spectrum[begin:i, band].mean(axis=0).argmax()]
                print("  whistle-like tone at video frame {} ({:.2f} s, ~{:.0f} Hz)".format(start + begin, (i - begin) / FPS, hz))
            begin = None


if __name__ == "__main__":
    main()
