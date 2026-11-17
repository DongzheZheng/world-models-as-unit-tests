"""Lab-side reference numbers for station B (purpose=eval, seeds 1000-1049, station B rules) ->
runs/stationb/_lab/baselines.json, step logs in runs/stationb/_lab/logs/<name>/.

    python -m reallab.stationb._baselines ['<uniform kwargs json>' '<governor kwargs json>']
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("runs/stationb/_lab/baselines.json")
SEEDS = list(range(1000, 1050))


def outcome(r):
    if r["success"]:
        return "success"
    if r["overload"]:
        return "overload"
    if r["sides_ok"] < 3:
        return "wrong_side"
    return "time_out"


def main(uni_kw, gov_kw):
    from reallab.stationb import api
    from reallab.stationb._reference import lab_evaluate
    refs = {
        "random": ("reallab.stationb.parallel:random_policy", {"seed": 0}),
        "noop": ("reallab.stationb.parallel:noop_policy", {}),
        "naive_router": ("reallab.station._reference:naive_router", {}),
        "uniform": ("reallab.stationb._reference:uniform", uni_kw),
        "governor": ("reallab.stationb._reference:governor", gov_kw),
    }
    cur = {"time_limit": api.TIME_LIMIT, "rules": "station B (load NaN in the loop, takt time)", "seeds": [1000, 1049]}
    for name, (fac, kw) in refs.items():
        t0 = time.time()
        res = lab_evaluate(fac, kw, SEEDS, purpose="eval", tag=f"lab_ref_{name}", n_workers=12,
                           log_dir=f"runs/stationb/_lab/logs/{name}")
        br = {}
        for r in res:
            br[outcome(r)] = br.get(outcome(r), 0) + 1
        cur[name] = dict(factory=fac, kwargs=kw, n=len(res), success=float(np.mean([r["success"] for r in res])),
                         breakdown=br, mean_steps=float(np.mean([r["n_steps"] for r in res])),
                         mean_max_load=float(np.mean([r["max_load"] for r in res])),
                         privileged=(name == "governor"), wall_s=round(time.time() - t0, 1),
                         per_seed=[dict(seed=r["seed"], outcome=outcome(r), steps=r["n_steps"],
                                        max_load=round(r["max_load"], 3)) for r in res])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(cur, indent=1))
        print(name, cur[name]["success"], br, round(cur[name]["mean_steps"], 1), flush=True)


# cells chosen by reallab.stationb._takt at T_B = 170 (seeds 3000-3049): best uniform cell, fastest governor cell
UNIFORM_BEST = {"clear": 0.06, "speed": 0.1}
GOVERNOR_BEST = {"back_at": 9.0, "clear": 0.05, "hi": 8, "lo": 3, "vmax": 0.3}

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(json.loads(sys.argv[1]), json.loads(sys.argv[2]))
    else:
        main(UNIFORM_BEST, GOVERNOR_BEST)
