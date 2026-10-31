"""Viterbi: the cheapest sequence of labels given per-frame costs and a price for switching.

The problem it solves: every frame t has a cost for each label k (cost[t, k], low = the clues agree
with k), and moving from label i on frame t-1 to label j on frame t costs switch[i, j]. Among all
possible label sequences, find the one with the lowest total cost. Trying them all is impossible
(3 labels over 80,000 frames), but the cheapest path to label j at frame t only depends on the
cheapest paths to each label at frame t-1, so one pass forward plus one pass back finds the exact
answer.

A tiny example, labels (H, A), cost 1 for disagreeing with the clue, switch price 3, clues H H A H:
    following every clue   H H A H   0 + 3 + 3 = 6
    ignoring the blip      H H H H   1         = 1   <- what this returns
"""

import numpy as np


def viterbi(cost, switch):
    """Lowest-total-cost label index for every frame.

    cost    float [frames, labels]   per-frame cost of each label
    switch  float [labels, labels]   switch[i, j] = price of going from label i to label j
                                     (the diagonal is the price of staying, normally 0)
    Returns int [frames].

    Ties are broken towards the lower label index, so the result is deterministic.
    """
    cost = np.asarray(cost, dtype=np.float64)
    switch = np.asarray(switch, dtype=np.float64)
    n_frames, n_labels = cost.shape
    best = cost[0].copy()                            # cheapest total ending in each label so far
    came_from = np.zeros((n_frames, n_labels), dtype=np.int8)
    for t in range(1, n_frames):
        # candidates[i, j]: be in label i at t-1, then move to label j at t
        candidates = best[:, None] + switch
        came_from[t] = np.argmin(candidates, axis=0)
        best = candidates[came_from[t], np.arange(n_labels)] + cost[t]

    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = np.argmin(best)
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = came_from[t, path[t]]
    return path


INF = float("inf")


def viterbi_min_length(cost, switch, min_frames):
    """viterbi(), plus one rule: once a label starts, it lasts at least min_frames[label] frames.

    cost, switch  as in viterbi()
    min_frames    int [labels]   shortest allowed stretch of each label (1 = no minimum)
    Returns int [frames].

    The first and last stretch may be shorter: the edges of the sequence (a half) cut them. With
    every min_frames == 1 the answer, ties included, is exactly viterbi()'s.

    How: a label that must last k frames is a chain of k steps with the same per-frame cost. The
    cheapest total on its last step at frame t either stayed there from t-1, or entered the label
    at t-k+1 and paid k frames of its cost since, which a cumulative sum gives in one subtraction.
    So per label only two totals per frame are kept ("entered at t", "on the last step at t") and
    the work per frame does not grow with k. Ties: lower label index first, as in viterbi(); on the
    last step, "came along the chain" beats "stayed".
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_frames, n_labels = cost.shape
    k = [int(v) for v in min_frames]
    labels = range(n_labels)
    c = cost.tolist()
    total = np.vstack([np.zeros(n_labels), np.cumsum(cost, axis=0)]).T.tolist()  # total[j][t] = cost[:t, j].sum()
    sw = np.asarray(switch, dtype=np.float64).tolist()

    enter = [[INF] * n_frames for _ in labels]       # cheapest total with label j starting at t
    last = [[INF] * n_frames for _ in labels]        # cheapest total on label j's last step at t
    enter_from = [[0] * n_frames for _ in labels]    # the label before the switch into j at t
    stayed = [[False] * n_frames for _ in labels]    # last[j][t] came from last[j][t - 1]
    for j in labels:
        last[j][0] = c[0][j]                         # the first stretch has no minimum
        enter[j][0] = c[0][j]

    for t in range(1, n_frames):
        for j in labels:
            # Into j at t, from the last step of a label at t - 1. For k == 1 that includes j itself,
            # which is simply staying, exactly as in viterbi().
            best_i, best = 0, INF
            for i in labels:
                if i == j and k[j] > 1:
                    continue
                value = last[i][t - 1] + sw[i][j]
                if value < best:
                    best_i, best = i, value
            enter[j][t] = best + c[t][j]
            enter_from[j][t] = best_i
            if k[j] == 1:
                last[j][t] = enter[j][t]
                continue
            stay = last[j][t - 1] + sw[j][j] + c[t][j]
            start = t - k[j] + 1
            chain = enter[j][start] + total[j][t + 1] - total[j][start + 1] if start >= 1 else INF
            stayed[j][t] = stay < chain
            last[j][t] = stay if stayed[j][t] else chain

    # The last stretch may be cut by the end: it can finish on any step of its chain.
    end = n_frames - 1
    best_value, best_label, best_start = INF, 0, None   # best_start None = on the last step
    for j in labels:
        for steps in range(1, k[j]):                     # entered `steps - 1` frames before the end
            start = end - steps + 1
            if start < 1:
                break
            value = enter[j][start] + total[j][end + 1] - total[j][start + 1]
            if value < best_value:
                best_value, best_label, best_start = value, j, start
        if last[j][end] < best_value:
            best_value, best_label, best_start = last[j][end], j, None

    path = np.empty(n_frames, dtype=np.int8)
    t, j = end, best_label
    if best_start is not None:                           # finish inside a chain
        path[best_start:] = j
        t, j = best_start - 1, enter_from[j][best_start]
    while t >= 0:                                        # (t, j): on j's last step at frame t
        if t == 0 or k[j] == 1:
            path[t] = j
            if t > 0:
                j = enter_from[j][t]
            t -= 1
        elif stayed[j][t]:
            path[t] = j
            t -= 1
        else:
            start = t - k[j] + 1
            path[start:t + 1] = j
            t, j = start - 1, enter_from[j][start]
    return path
