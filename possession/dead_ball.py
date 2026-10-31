"""v4's dead-ball model: fit (on the bench only), save, load, predict
(docs/models/v4-dead-ball.md, pieces 2, 3 and 5).

Two families: logistic regression (stored as plain numbers; inference is numpy only) and
gradient-boosted trees (scikit-learn, stored with joblib). Probabilities are clipped and rounded
before they become costs, so a last-digit floating-point difference on another machine can't flip
a label.
"""

import json
import os

import numpy as np

from possession.data import ROOT

# Params
MODELS = os.path.join(ROOT, "possession", "models")
CLIP = (0.01, 0.99)     # one frame can't cost more than -log(0.01) = 4.6 x the weight
DECIMALS = 3            # rounding of P before it becomes a cost
SEED = 0


def fit(family, X, y, names):
    """A model of P(dead | features). X float [rows, features], y 1 = dead."""
    if family == "logistic":
        from sklearn.linear_model import LogisticRegression
        mean = np.nanmean(X, axis=0, dtype=np.float64)
        std = np.nanstd(X, axis=0, dtype=np.float64)
        std[std == 0] = 1.0
        z = np.nan_to_num((X - mean) / std)            # a missing feature counts as average
        clf = LogisticRegression(max_iter=2000).fit(z, y)
        return {"family": family, "features": list(names), "mean": mean.tolist(), "std": std.tolist(),
                "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(random_state=SEED).fit(X, y)
    return {"family": family, "features": list(names), "model": clf}


def predict(model, X):
    """P(dead) per frame, clipped and rounded; 0.5 (no opinion) on frames with no features at all."""
    if model["family"] == "logistic":
        z = np.nan_to_num((X - np.array(model["mean"])) / np.array(model["std"]))
        # an elementwise product and numpy's own sum, not a BLAS dot, so the order of additions
        # does not depend on the machine's linear-algebra library
        logit = (z * np.array(model["coef"])).sum(axis=1) + model["intercept"]
        with np.errstate(over="ignore"):
            p = 1.0 / (1.0 + np.exp(-logit))
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            p = model["model"].predict_proba(X)[:, 1]
    p = np.round(np.clip(p, *CLIP), DECIMALS)
    p[np.isnan(X).all(axis=1)] = 0.5
    return p


def costs(p):
    """(cost of saying live, cost of saying dead) per frame: how surprised we'd be."""
    return -np.log(1.0 - p), -np.log(p)


def save(model, name="dead_ball"):
    """Also used for v6's restart model (name="restart"): same two families, same format."""
    os.makedirs(MODELS, exist_ok=True)
    if model["family"] == "logistic":
        path = os.path.join(MODELS, "{}_logistic.json".format(name))
        with open(path, "w") as handle:
            json.dump(model, handle, indent=1)
    else:
        import joblib
        path = os.path.join(MODELS, "{}_trees.joblib".format(name))
        joblib.dump(model, path)
    return path


def load(family, name="dead_ball"):
    if family == "logistic":
        with open(os.path.join(MODELS, "{}_logistic.json".format(name))) as handle:
            return json.load(handle)
    import joblib
    return joblib.load(os.path.join(MODELS, "{}_trees.joblib".format(name)))
