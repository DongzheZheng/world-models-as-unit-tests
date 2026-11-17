"""Lab-side tuning harness (internal).  python -m reallab.station._tune <factory> <json kwargs> <seed0> <n> [nodev] [dev overrides]"""
import json
import os
import sys
import time

import numpy as np


def classify(r):
    if r["success"]:
        return "success"
    if r["overload"]:
        return "overload"
    if r["sides_ok"] < 3:
        return "wrong_side"
    if r["final_dist"] > 0.02:
        return "not_at_target"
    return "not_at_rest"


def summary(res):
    c = {}
    for r in res:
        k = classify(r)
        c[k] = c.get(k, 0) + 1
    return dict(n=len(res), success=float(np.mean([r["success"] for r in res])), breakdown=c,
                mean_max_load=float(np.mean([r["max_load"] for r in res])),
                mean_steps=float(np.mean([r["n_steps"] for r in res])))


if __name__ == "__main__":
    fac, kw, s0, n = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    mode = sys.argv[5] if len(sys.argv) > 5 else "dev"
    if mode == "nodev":
        os.environ["CAWM_STATION_NODEV"] = "1"
    if len(sys.argv) > 6:
        os.environ["CAWM_STATION_DEV"] = sys.argv[6]
    from reallab.station._reference import lab_evaluate
    t0 = time.time()
    logd = os.environ.get("TUNE_LOGS")
    res = lab_evaluate(fac, kw, list(range(s0, s0 + n)), purpose="smoke", tag="tune", n_workers=24, log_dir=logd)
    print(mode, sys.argv[6] if len(sys.argv) > 6 else "", fac.split(":")[-1], kw, json.dumps(summary(res)),
          f"{time.time() - t0:.0f}s", flush=True)
    for r in res:
        extra = ""
        if logd and r["overload"]:
            d = np.load(f"{logd}/ep_{r['seed']}.npz")
            ld = d["load"]
            k = int(np.argmax(ld > 2.0))
            dist = np.linalg.norm(d["tool_pos"] - d["target"], axis=1)
            extra = f" load>2 at step {k} (dist {dist[k]:.3f}), end dist {dist[-1]:.3f}, load ramp {np.round(ld[max(0,k-1):k+4],1).tolist()}"
        print("  ", r["seed"], classify(r), r["n_steps"], round(r["max_load"], 2), round(r["final_dist"], 3), r["sides_ok"], extra)
