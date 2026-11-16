"""Night 2026-11-14, idea `netown` (docs/lab/night/network.md; docs/lab/night/ideas_outside.md idea 2, the training-only
clean-owner head of Barca-Hackathon's public plan: the idea is theirs, the code is ours).

    python -m bench.lab_netown targets                # the head's target on the 7 DFL matches (~1 min)
    python -m bench.lab_netown train --fold HELD      # gru_b's fold with the head (~8 min), then its held-out P

gru_b's recipe, seed 0, its 8 folds (lab_branch_b.held_list), plus a second head on the BiGRU's output, used in
training only: for each frame, the team of the nearest player within 1.5 m of the ball in the bench's CLEAN tracking
(sets.load: 0 home, 1 away, 2 nobody), ignored where the clean ball is missing; home and away exchanged on the swapped
orientation; the same target on the training-stress copy (the stress replaces only the ball the model sees). Loss =
main + 0.5 x head. The head is built with torch's random state saved and restored (lab_netmirror.fit), so the main
net's initial weights, its dropout draws and the batch order are gru_b's: the only difference is the head's gradient.
P is read from the main head only, so the deployed program and its inputs would not change.
Files: data/external/lab_netown/main/ (targets/, nets/, lab_branch_b/, search_b/).

Nothing here reads our match.
"""

import argparse
import os

import numpy as np

from bench import lab, lab_branch_b, lab_netmirror, sets

# Params
B = lab_branch_b
N = lab_netmirror
NAME = "netown"
SUB = "main"
WHICH = "all"
SEED = 0
NEAR = 1.5            # m: a player this close to the clean ball owns it (Barca-Hackathon's recipe)
WEIGHT = 0.5          # the head's weight in the loss (theirs; their 1.5 leaned to DFL)


def root(seed=SEED):
    return N.root_of(NAME, SUB if seed == SEED else "seed{}".format(seed))


def target_path(m):
    return os.path.join(root(), "targets", "owner_{}.npy".format(m))


def target(m):
    """int16 per frame: 0 home, 1 away, 2 nobody within NEAR of the clean ball, -100 where the clean ball is missing."""
    path = target_path(m)
    if os.path.exists(path):
        return np.load(path)
    clean = sets.load(m)
    players = clean["players"].astype(np.float32)
    ball = clean["ball"][:, :2].astype(np.float32)
    team = np.asarray(clean["team"]).astype(np.int16)
    n = len(ball)
    if n != lab.load(m, WHICH).frames:
        raise RuntimeError("{}: clean frames {} differ from the lab cache's".format(m, n))
    out = np.full(n, -100, np.int16)
    for a in range(0, n, 20000):
        b = min(a + 20000, n)
        d = np.hypot(players[a:b, :, 0] - ball[a:b, None, 0], players[a:b, :, 1] - ball[a:b, None, 1])
        d = np.where(np.isnan(d), np.inf, d)
        k = d.argmin(axis=1)
        near = d[np.arange(b - a), k] <= NEAR
        t = np.where(near, team[k], 2).astype(np.int16)
        t[np.isnan(ball[a:b, 0])] = -100
        out[a:b] = t
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, out)
    return out


def targets():
    for m in lab.matches(WHICH):
        if sets.source(m) not in B.L.TRAIN_SOURCES:
            continue
        t = target(m)
        seen = t != -100
        print("{}: clean ball seen {:.1%}; of those, home {:.1%}, away {:.1%}, nobody within {} m {:.1%}".format(
            m, seen.mean(), (t == 0).sum() / seen.sum(), (t == 1).sum() / seen.sum(), NEAR, (t == 2).sum() / seen.sum()),
            flush=True)


def net_path(held, seed=SEED):
    return os.path.join(root(seed), "nets", "net_{}.pt".format(held))


def train(held, seed=SEED):
    path = net_path(held, seed)
    used = B.used_for(held)
    if not os.path.exists(path):
        owner = {m: target(m) for m in used}
        print("netown fold {}: training on {} (as they are and stressed), owner head weight {}".format(
            held, " ".join(used), WEIGHT), flush=True)
        model = N.fit(used, seed=seed, owner=owner, owner_weight=WEIGHT, log=lambda s: print(s, flush=True))
        N.save_net(model, path, {"near": NEAR, "weight": WEIGHT})
        print("netown fold {}: saved ({:.0f} s)".format(held, model["seconds"]), flush=True)
    ms = [held] if sets.source(held) in B.L.TRAIN_SOURCES else \
        [m for m in lab.matches(WHICH) if sets.source(m) not in B.L.TRAIN_SOURCES]
    for m in ms:
        N.predict_into(root(seed), m, path, used)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("targets", "train"))
    parser.add_argument("--fold")
    parser.add_argument("--seed", type=int, default=SEED, help="train: the seed (not 0: files under seedS/, the noise rule)")
    args = parser.parse_args()
    if args.step == "targets":
        return targets()
    return train(args.fold, args.seed)


if __name__ == "__main__":
    main()
