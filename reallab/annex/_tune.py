"""Lab-side tuning runs on smoke seeds 3000..3049 (true simulator unless physics overrides are given).

usage: python -m reallab.annex._tune search <goal> <family> <n_seeds> <budget> [k=v physics] [vmax] [T]
Prints per-seed results and the success fraction; writes runs/annex/_lab/tune/<tag>.json.
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np


def _one(args):
    seed, goal, family, budget, ov, vmax, T = args
    from reallab.annex import _internal
    from reallab.annex._plan import Family, search
    from reallab.annex._sim import Sim
    phys = _internal.physical()
    ov = dict(ov)
    ov_robust = ov.pop("robust_k", 0)
    kw = {k: ov.pop(k) for k in ("w_min", "pocket_r", "v_rest", "pocket_w", "no_post") if k in ov}
    phys.update(ov)
    sim = Sim(phys, vmax=vmax, **kw)
    ep = sim.setup(seed)
    t0 = time.time()
    fam = Family(family, slow=vmax < 1.0)
    r = search(sim, fam, goal, budget=budget, T=T, rng=np.random.default_rng(seed), extra_after_success=60,
               robust_k=int(ov_robust))
    if r["success"]:                     # landing-point sensitivity: each parameter scaled by U(0.95, 1.05)
        rng = np.random.default_rng([seed, 5])
        base = np.array(r["tip_final"])
        moves, oks = [], []
        for _ in range(10):
            q = np.clip(r["params"] * rng.uniform(0.95, 1.05, len(r["params"])), fam.lo, fam.hi)
            rr = sim.rollout(fam.make(q, ep), T, stop_when=())
            moves.append(float(np.linalg.norm(rr["tip_final"] - base)))
            oks.append(rr[goal + "_t"] is not None)
        r["sens_median_m"] = float(np.median(moves))
        r["sens_p90_m"] = float(np.percentile(moves, 90))
        r["sens_success"] = float(np.mean(oks))
    r["params"] = None if r["params"] is None else [round(float(x), 4) for x in r["params"]]
    return dict(seed=seed, dist=ep["dist"], bearing=ep["bearing"], wall=time.time() - t0, **r)


def parse(s):
    out = {}
    for kv in filter(None, s.split(",")):
        k, v = kv.split("=")
        out[k] = float(v)
    return out


if __name__ == "__main__":
    goal, family, ns, budget = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    ov = parse(sys.argv[6]) if len(sys.argv) > 6 else {}
    vmax = float(sys.argv[7]) if len(sys.argv) > 7 else 4.0
    T = int(sys.argv[8]) if len(sys.argv) > 8 else 200
    with mp.get_context("spawn").Pool(min(12, ns)) as pool:
        res = pool.map(_one, [(3000 + i, goal, family, budget, ov, vmax, T) for i in range(ns)])
    for r in res:
        print(r["seed"], "succ", r["success"], "t", r["t_done"], "used", r["used"], "w", round(r["final_w"], 2),
              "tip", round(r["tip_dist"], 3), "dist", round(r["dist"], 2), "bear", round(r["bearing"], 2),
              "wall", round(r["wall"]), "sens", round(r.get("sens_median_m", -1), 3), round(r.get("sens_p90_m", -1), 3),
              r.get("sens_success"))
    sr = float(np.mean([r["success"] for r in res]))
    tag = f"{goal}_{family}_{budget}_{sys.argv[6] if len(sys.argv) > 6 else 'real'}_{vmax}"
    sens = [r["sens_median_m"] for r in res if "sens_median_m" in r]
    print("SUCCESS", goal, family, sr, "sens median-of-medians", np.median(sens) if sens else None,
          "mean succ under 5% perturbation", np.mean([r["sens_success"] for r in res if "sens_success" in r]) if sens else None)
    Path("runs/annex/_lab/tune").mkdir(parents=True, exist_ok=True)
    Path(f"runs/annex/_lab/tune/{tag.replace('=', '').replace(',', '_')}.json").write_text(json.dumps(res, indent=1))
