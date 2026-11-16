"""Model v17: v16 with the broadcast's players near the ball merged into the tactical tracking that branch B reads
(push track 3, docs/lab/push/broadcast.md; docs/models/v17-broadcast.md).

    python -m possession.v17           # writes evaluation/results/v17.csv; prints only its path and SHA-256

v16 (possession/v16.py) runs v14's branch A unless the stoppage switch's signal is above its threshold; then its
branch B (netmet5's four networks averaged, augment's trees, trial 1015's decoder and post stages). v17 changes what
branch B reads, nothing else:

- **the switch reads the tactical tracking as delivered**, exactly as v16 (so it takes the same branch);
- **if it takes B**, the broadcast set (files 6-8) is read, lined up with the tactical tracking by the ball
  (tools/align_broadcast.py's method) and its players that the tactical tracking lacks are added with rule M2
  (possession/broadcast_merge.py: rule M, fixed before any score, plus "a team never has more than 11", chosen after
  the pessimistic run; Jose's choice 2026-11-15). v8's repair and every input B reads are then computed from the
  merged tracking (v12.inputs), and v16's branch B runs on it with v16's models, unchanged (no retraining: track 3's
  judge; a B retrained on merged data was worse);
- **if anything about the broadcast fails** (a file missing, the alignment not trusted, any error), branch B reads
  the delivered tracking and v17's file is v16's, byte for byte.

Branch A, the switch, the kick-off rule and every model are v16's (possession/models/v16_b_all.joblib and v14's).
`python -m bench.check_v17` checks the glue on a bench match with a simulated broadcast written as files.
"""

import argparse
import hashlib
import os

import numpy as np

from possession import broadcast_merge, v1, v10, v12, v13, v14, v16
from possession.data import PITCH_LENGTH, PITCH_WIDTH, ROOT, graded_halves, load_tags, load_tracking

# Params
OUTPUT = os.path.join(ROOT, "evaluation", "results", "v17.csv")
RULE = broadcast_merge.RULE_M2       # Jose, 2026-11-15: "Yes, with M2"
SETS = v16.SETS


def broadcast_paths(data_dir=None):
    """Files 6-8 of our match (games.asset_paths' broadcast set). Only main() uses them: predict() merges nothing
    unless it is given paths, so a bench match can never be merged with our match's broadcast by accident."""
    from games import asset_paths
    p = asset_paths(data_dir=data_dir, source="broadcast")
    return {k: p[k] for k in ("atd_meta", "atd_raw", "homography")}


def b_match(tracking, halves, pitch, which, paths, delivered=None):
    """(the match branch B reads, the merge's status). paths None: no broadcast ("off"); the delivered match
    (v12.inputs of the delivered tracking) is reused when given. A failed merge falls back to the delivered match."""
    if paths is not None:
        merged, status = broadcast_merge.merge_broadcast(tracking, halves, paths=paths, rule=RULE)
        if status == "merged":
            fixed, pred = v12.inputs(merged, halves, pitch, which)
            return v12.as_match(fixed, pred, halves, pitch), status
    else:
        status = "off"
    if delivered is None:
        fixed, pred = v12.inputs(tracking, halves, pitch, which)
        delivered = v12.as_match(fixed, pred, halves, pitch)
    return delivered, status


def b_letters(match, which):
    """v16's branch B on a match: (letters, B's start P)."""
    bundle = v16.load_branch_b(which)
    p = v16.b_start_proba(match, bundle, which)
    return v16.b_chain_from(match, p, bundle, which), p


def predict(tracking, tags=None, halves=None, pitch=(PITCH_LENGTH, PITCH_WIDTH), bench_set=None, paths=None):
    """(labels for every video frame, {"pred", "p_start", "branch", "broadcast"}): v16's predict, with branch B on
    the tracking merged with the broadcast in `paths` (None: no broadcast, and v17 is v16)."""
    halves = halves or graded_halves()
    which = bench_set or v10.BENCH_SET
    if which not in SETS:
        labels, info = v14.predict(tracking, tags, halves, pitch, which)
        return labels, dict(info, broadcast="off")
    fixed, pred = v12.inputs(tracking, halves, pitch, which)                    # the delivered tracking
    match = v12.as_match(fixed, pred, halves, pitch)
    chosen = v13.branch(fixed, pred, halves, pitch, which)                      # the switch: as v16
    if chosen == "B":
        del fixed
        match, status = b_match(tracking, halves, pitch, which, paths, delivered=match)
        pred = match.pred
        labels, p = b_letters(match, which)
    else:                                                                       # v14's branch A, unchanged
        status = "not-used"
        models = v14.load_models(which)
        p = v14.start_proba(match, models, which)
        labels = v14.run_chain_from(match, p, models, which)
    labels[v12.before_kickoff(tags, len(labels), halves)] = "D"
    return labels, {"pred": pred, "p_start": p, "branch": chosen, "broadcast": status}


def main():
    """Only what Jose needs to check the file, as v10-v16: no statistics about the match. --show-branch adds the
    branch the switch took and whether the broadcast was merged (Jose's to read; Claude doesn't run it on our match)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--show-branch", action="store_true",
                        help="also print the branch the stoppage switch took and the broadcast merge's status")
    args = parser.parse_args()
    labels, info = predict(load_tracking(), load_tags(), paths=broadcast_paths())
    v1.write_submission(OUTPUT, labels)
    with open(OUTPUT, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    path = os.path.relpath(OUTPUT, ROOT)
    print("wrote {} (bench set {})".format(path, v10.BENCH_SET))
    print("SHA-256 {}".format(digest))
    if args.show_branch:
        print("stoppage switch: branch {}; broadcast: {}".format(info["branch"], info["broadcast"]))
    print("format check: .venv/bin/python evaluation/validate_submission.py {}".format(path))


if __name__ == "__main__":
    main()
