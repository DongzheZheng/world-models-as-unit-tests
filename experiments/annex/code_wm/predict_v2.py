"""In-world success prediction (v2 world) for a controller over snapshot situations, 6 worker processes (spawn).
usage: python predict_v2.py <factory> <first> <count> <out.json> [ens_spread] [kwargs.json]
ens_spread > 0: every imagined episode runs in its own perturbed 'reality': each uncertain world parameter is
multiplied by a factor drawn uniformly from [1 - s, 1 + s] (seeded by the snapshot index), and the initial node
velocities are drawn as a random rigid spin of 0-0.3 m/s at the free end (the real reset reading is not at rest).
The controller always plans in its own nominal world."""
import json
import os
import sys
import time

KEYS = ("k_stretch", "c_stretch", "k_bend", "c_bend", "mu", "aniso", "c_visc", "c_quad", "lat_mass", "tip_mass",
        "mu_lo", "v_s", "mu_post")


def _work(args):
    fac, idx, spread, kw = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from experiments.annex.code_wm.screen_v2 import run_in_world, snapshot_obs, world_params
    out = []
    for i in idx:
        o = snapshot_obs([i])
        rng = np.random.default_rng(1000 + i)
        over, vel = None, None
        if spread > 0:
            nom = world_params()
            over = {k: nom[k] * float(rng.uniform(1 - spread, 1 + spread)) for k in KEYS}
            P = o[0]["points"][:, :2]
            rel = P - P[-1]
            w = float(rng.uniform(-0.3, 0.3)) / 0.7
            vel = [np.stack([-rel[:, 1], rel[:, 0]], 1) * w]
        r = run_in_world(fac, kw, o, params=over, init_vel=vel)[0]
        r["snapshot"] = int(i)
        out.append(r)
    return out


if __name__ == "__main__":
    import multiprocessing as mp
    fac, a, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    spread = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    kw = json.load(open(sys.argv[6])) if len(sys.argv) > 6 else {}
    idx = list(range(a, a + n))
    nw = int(os.environ.get("CW_NW", "6"))
    chunks = [(fac, idx[k::nw], spread, kw) for k in range(nw) if idx[k::nw]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        res = [r for part in pool.map(_work, chunks) for r in part]
    res.sort(key=lambda r: r["snapshot"])
    s = sum(r["success"] for r in res) / len(res)
    moving = sum((not r["success"]) and r["final_speed"] > 0.02 for r in res)
    json.dump(dict(factory=fac, kwargs=kw, spread=spread, success_rate=s, n=len(res), moving_at_limit=moving,
                   episodes=res, wall_s=time.time() - t0, steps=int(sum(r["n_steps"] for r in res))),
              open(out, "w"), indent=1)
    print("success", s, "moving_at_limit", moving, "n", len(res), "wall", time.time() - t0)
