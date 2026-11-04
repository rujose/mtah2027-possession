"""v10 is v9 without the whistle and the set-piece window: check it on the bench.

    python -m bench.check_v10          # ~10 min; prints one line per match

The bench matches have no audio and no Smart Tagging, so there v9 and v10 (bench set "all") must give
the same labels, frame for frame. That checks v10's composition (possession/v10.py rebuilds v9's
costs term by term) without running anything on our match.
"""

import numpy as np

from bench import degrade, sets
from bench.train_restart import prepare_level
from possession import v9, v10

# Params
MATCHES = ("J03WR9", "J03WMX", "metrica1")   # one unseen by PathCRF, one seen, one Metrica
LEVEL = 41


def main():
    stats = degrade.load_stats()
    data = {m: sets.load(m) for m in sets.matches("all")}
    prepared, scale = prepare_level(LEVEL, list(data), data, stats)
    same = 0
    for m in MATCHES:
        track = degrade.tracking(m, data[m], LEVEL, prepared[m], scale[sets.source(m)], stats)
        halves, pitch = data[m]["halves"], data[m]["pitch"]
        before = v9.predict(track, [], [], halves, pitch)[0]
        after = v10.predict(track, [], halves, pitch, bench_set="all")[0]
        differ = int(np.sum(before != after))
        same += differ == 0
        print("{}: {} frames differ between v9 and v10".format(m, differ), flush=True)
    print("identical on {}/{} matches".format(same, len(MATCHES)))


if __name__ == "__main__":
    main()
