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
