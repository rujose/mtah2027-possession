"""Does the keeper repair (possession/repair.py) ever fire on the bench? It should not.

    python -m bench.keeper_repair      # ~3 min; writes bench/results/keeper_repair.json

The bench's teams are always right: its damage cuts tracks into pieces and drops the ball, but never
swaps a slot's team. So every piece the rule would repair here is a false repair. Checked on the
published tracking and at the three damage levels the bench uses.
"""

import json
import os

from bench import degrade, sets
from bench.train_restart import prepare_level
from possession import repair
from possession.data import ROOT

# Params
OUT = os.path.join(ROOT, "bench", "results", "keeper_repair.json")
LEVELS = (30, 41, 55)


def main():
    stats = degrade.load_stats()
    matches = sets.matches("all")
    data = {m: sets.load(m) for m in matches}
    found = {"clean": {m: repair.keeper_pieces(data[m], data[m]["halves"], data[m]["pitch"]) for m in matches}}
    for level in LEVELS:
        prepared, scale = prepare_level(level, matches, data, stats)
        found[str(level)] = {}
        for m in matches:
            track = degrade.tracking(m, data[m], level, prepared[m], scale[sets.source(m)], stats)
            found[str(level)][m] = repair.keeper_pieces(track, data[m]["halves"], data[m]["pitch"])
    summary = {}
    for level, per_match in found.items():
        summary[level] = {"pieces_looked_at": sum(len(p) for p in per_match.values()),
                          "false_repairs": sum(q["repaired"] for p in per_match.values() for q in p),
                          "matches": len(per_match)}
        print("{:>5}: {} pieces looked at, {} false repairs, over {} matches".format(
            level, summary[level]["pieces_looked_at"], summary[level]["false_repairs"], len(per_match)))
    with open(OUT, "w") as handle:
        json.dump({"summary": summary, "pieces": found}, handle, indent=1)
    print("wrote " + os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
