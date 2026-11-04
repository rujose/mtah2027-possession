"""How fragile is v12's output to another CPU? Bench matches only (docs/reproduce.md).

    python -m bench.check_reproduce            # ~10 min; after `python -m bench.train_v12`

1. Each family of v12's inputs nudged by a relative 1e-7 (about float32's last bit) with a fixed
   random sign: frames of v12's letters that change (the deployment models).
2. PathCRF in float32 against float64: the largest difference, and the frames of v12's letters that
   change between the two.
"""

import numpy as np

from bench import chain, lab
from bench.eval_pathcrf import v7_cost
from bench.tune_decoder import decode
from possession import pathcrf_adapter, v12

# Params
MATCHES = ("J03WR9", "metrica1")
WHICH = "all"
NUDGE = 1e-7
FAMILIES = {"tracking": None, "PathCRF's P": ["p_pathcrf"], "decoder costs": ["v6_cost", "v7_cost"],
            "P(dead), P(restart), restart clue": ["p_dead", "p_start", "restart_clue"]}


def letters(m, models, track=None, pred=None):
    return v12.run_chain(v12.as_match(track or m.track, pred or m.pred, m.halves, m.pitch, m.id), models, WHICH)


def main():
    models = v12.load_models(WHICH)
    pathcrf = pathcrf_adapter.load_model()
    pc = chain.pathcrf(WHICH)
    for m_id in MATCHES:
        m = lab.load(m_id, WHICH)
        base = letters(m, models)
        for name, keys in FAMILIES.items():
            rng = np.random.default_rng(0)
            nudge = lambda v: (v * (1 + NUDGE * rng.choice([-1, 1], size=v.shape))).astype(v.dtype)
            track, pred = dict(m.track), dict(m.pred)
            if keys is None:
                track["players"], track["ball"] = nudge(m.track["players"]), nudge(m.track["ball"])
            for k in keys or ():
                pred[k] = nudge(m.pred[k])
            print("{}: {} nudged by {:g}: {} of {} frames change".format(
                m_id, name, NUDGE, int(np.sum(letters(m, models, track, pred) != base)), m.frames), flush=True)
        live = np.isin(decode(m.pred["v6_cost"].astype(np.float64), m.halves, chain.decoder(WHICH)), ["H", "A"])
        missing = np.isnan(m.track["ball"][:, 0])
        out, p = {}, {}
        for name in ("float32", "float64"):
            model = pathcrf.float() if name == "float32" else pathcrf.double()   # in place: one at a time
            p[name] = pathcrf_adapter.p_home(model, m.track, m.halves, m.pitch, live)
            cost = v7_cost(m.pred["v6_cost"].astype(np.float64), p[name], missing, pc["weight"], pc["where"])
            out[name] = letters(m, models, pred=dict(m.pred, p_pathcrf=p[name], v7_cost=cost.astype(np.float32)))
        both = ~np.isnan(p["float32"])
        print("{}: PathCRF float32 vs float64: largest difference {:.1e}; frames of v12 that change: {} of {}".format(
            m_id, np.max(np.abs(p["float32"][both] - p["float64"][both])), int(np.sum(out["float32"] != out["float64"])),
            m.frames), flush=True)


if __name__ == "__main__":
    main()
