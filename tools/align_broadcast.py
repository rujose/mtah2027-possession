"""Line the broadcast up with the tactical video, using only the two ball tracks.

    python tools/align_broadcast.py      # ~15 s

Both trackings give the ball in the same pitch coordinates, so at the right time shift the two balls
sit on top of each other. For each graded half it:
  1. tries every possible shift at once (FFT cross-correlation of the two ball tracks) and keeps the
     one with the smallest mean squared ball distance
  2. checks the shift holds: best local correction in every 2-minute window (within +-50 frames)
  3. measures what the broadcast adds: how often it sees the ball when the tactical tracking lost it

Result (2026-10-30): broadcast frame = tactical frame - 4014 in the 1st half, - 26919 in the 2nd.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402
from possession.data import graded_halves, load_tracking  # noqa: E402

# Params
MIN_OVERLAP = 0.2       # a shift needs shared ball frames >= 20 % of the half's ball frames to count
WINDOW = 3000           # frames: 2-minute windows for the stability check
LOCAL_SEARCH = 50       # frames either side of the global shift
CLOSE = 3.0             # metres: "the two balls agree"


def xcorr(a, b):
    """r[k] = sum_t a[t] * b[t - k] for every lag k from -(len(b)-1) to len(a)-1."""
    size = 1 << (len(a) + len(b) - 2).bit_length()
    r = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    return np.concatenate([r[-(len(b) - 1):], r[:len(a)]])


def global_shift(tac_ball, bro_ball, first, last):
    """Shift s such that broadcast frame = tactical frame - s, for one tactical half."""
    tb = tac_ball[first:last + 1]
    mt, mb = ~np.isnan(tb[:, 0]), ~np.isnan(bro_ball[:, 0])
    xt, xb = np.nan_to_num(tb), np.nan_to_num(bro_ball)
    count = xcorr(mt.astype(float), mb.astype(float))
    squares, cross = np.zeros_like(count), np.zeros_like(count)
    for axis in range(2):
        squares += xcorr(mt * xt[:, axis] ** 2, mb.astype(float)) + xcorr(mt.astype(float), mb * xb[:, axis] ** 2)
        cross += xcorr(mt * xt[:, axis], mb * xb[:, axis])
    mean_sq = np.where(count > MIN_OVERLAP * mt.sum(), (squares - 2 * cross) / np.maximum(count, 1), np.inf)
    lags = np.arange(-(len(mb) - 1), len(mt))
    order = np.argsort(mean_sq)
    best = order[0]
    runner = next(i for i in order if abs(lags[i] - lags[best]) > 250)
    return first + lags[best], np.sqrt(mean_sq[best]), int(count[best]), first + lags[runner], np.sqrt(mean_sq[runner])


def main():
    tac, bro = load_tracking("tactical"), load_tracking("broadcast")
    valid = pd.read_csv(asset_paths(source="broadcast")["homography"]).valid.values.astype(bool)
    for half, first, last in graded_halves():
        shift, rms, shared, runner, runner_rms = global_shift(tac["ball"], bro["ball"], first, last)
        print("half {}: broadcast frame = tactical frame - {}  (rms ball distance {:.2f} m over {} shared "
              "frames; next-best shift {} has {:.1f} m)".format(half, shift, rms, shared, runner, runner_rms))

        corrections = []
        for start in range(first, last, WINDOW):
            frames = np.arange(start, min(start + WINDOW, last))
            best = None
            for d in range(-LOCAL_SEARCH, LOCAL_SEARCH + 1):
                b = frames - shift - d
                inside = (b >= 0) & (b < len(bro["ball"]))
                t, b = frames[inside], b[inside]
                both = ~np.isnan(tac["ball"][t, 0]) & ~np.isnan(bro["ball"][b, 0]) & valid[b]
                if both.sum() < 200:
                    continue
                median = np.median(np.hypot(*(tac["ball"][t[both]] - bro["ball"][b[both]]).T))
                if best is None or median < best[1]:
                    best = (d, median)
            if best:
                corrections.append(best)
        d = np.array([c[0] for c in corrections])
        print("  {} two-minute windows: local correction {}..{} frames, within +-2 frames in {:.0f}%; "
              "median ball distance {:.2f} m".format(len(d), d.min(), d.max(), 100 * np.mean(np.abs(d) <= 2),
                                                      np.median([c[1] for c in corrections])))

        t = np.arange(first, last + 1)
        b = t - shift
        inside = (b >= 0) & (b < len(bro["ball"]))
        t, b = t[inside], b[inside]
        both = ~np.isnan(tac["ball"][t, 0]) & ~np.isnan(bro["ball"][b, 0]) & valid[b]
        agree = np.hypot(*(tac["ball"][t[both]] - bro["ball"][b[both]]).T) < CLOSE
        lost = np.isnan(tac["ball"][t, 0])
        filled = lost & ~np.isnan(bro["ball"][b, 0]) & valid[b]
        print("  broadcast shows the pitch on {:.0f}% of these frames; both balls seen on {:.0f}%, within {:.0f} m on "
              "{:.0f}% of those".format(100 * valid[b].mean(), 100 * both.mean(), CLOSE, 100 * agree.mean()))
        print("  tactical ball lost on {:.0f}% of frames; the broadcast sees it on {:.0f}% of those "
              "(before removing replays)".format(100 * lost.mean(), 100 * filled.sum() / max(lost.sum(), 1)))


if __name__ == "__main__":
    main()
