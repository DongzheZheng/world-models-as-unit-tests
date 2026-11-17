"""Reference numbers on the real rig (purpose=eval, seeds 1000-1049) -> runs/pusht/baselines.json.

    python -m reallab.pusht.baselines random noop            # trivial baselines
    python -m reallab.pusht.baselines true_sim_mpc [n]       # privileged planner, hidden deviations on (n episodes)
    python -m reallab.pusht.baselines nominal_mpc [n]        # planner whose model is the datasheet object
    python -m reallab.pusht.baselines true_sim_mpc_nodev [n] # deviations switched off in the real rig
    python -m reallab.pusht.baselines L_random | L_true_sim_mpc [n] | L_nominal_mpc [n] | L_true_sim_mpc_nodev [n]
                                                             # same on the L block -> baselines.json["L"]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path("runs/pusht/baselines.json")
MPC = "reallab.pusht._planner:true_sim_mpc"
KW = dict(K=192, H=14, iters=2)
KW_L = dict(KW, w_ang=4.0)    # retuned cost for the L: extra heading term


def stats(res):
    mc = np.array([r["max_coverage"] for r in res])
    return dict(n=len(res), success=float(np.mean([r["success"] for r in res])), mean_max_coverage=float(mc.mean()),
                mean_final_coverage=float(np.mean([r["final_coverage"] for r in res])),
                mean_steps=float(np.mean([r["n_steps"] for r in res])), mean_return=float(np.mean([r["ret"] for r in res])),
                seeds=[r["seed"] for r in res], successes=[bool(r["success"]) for r in res], max_coverages=mc.tolist())


def main(argv):
    from reallab.pusht.parallel import evaluate
    which = argv[0]
    obj = "T"
    if which.startswith("L_"):
        obj, which = "L", which[2:]
    n = int(argv[1]) if len(argv) > 1 else 50
    seeds = list(range(1000, 1000 + n))
    if which in ("random", "noop"):
        res = evaluate(f"reallab.pusht.parallel:{which}_policy", {"seed": 0} if which == "random" else {}, seeds,
                       purpose="eval", tag=f"baseline_{which}", n_workers=16, object=obj)
    elif which in ("true_sim_mpc", "nominal_mpc", "true_sim_mpc_nodev"):
        if which == "true_sim_mpc_nodev":
            os.environ["CAWM_PUSHT_NODEV"] = "1"
        kw = dict(KW_L if obj == "L" else KW, nominal=which != "true_sim_mpc", object=obj)
        res = evaluate(MPC, kw, seeds, purpose="eval", tag=which, n_workers=16, log_dir=f"runs/pusht/logs/{obj}_{which}" if obj == "L" else f"runs/pusht/logs/{which}",
                       record_first=1 if which == "true_sim_mpc" else 0, video_dir="runs/pusht/gifs", object=obj)
        if which == "true_sim_mpc":
            g = Path("runs/pusht/gifs/ep_1000.gif")
            if g.exists():
                g.replace("runs/pusht/gifs/true_sim_mpc%s.gif" % ("_L" if obj == "L" else ""))
    else:
        raise SystemExit(__doc__)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    entry = dict(stats(res), planner=(KW_L if obj == "L" else KW) if "mpc" in which else None)
    (cur.setdefault("L", {}) if obj == "L" else cur)[which] = entry
    OUT.write_text(json.dumps(cur, indent=1))
    print(obj, which, {k: v for k, v in entry.items() if k in ("n", "success", "mean_max_coverage", "mean_steps")}, flush=True)


if __name__ == "__main__":
    if sys.argv[1] in ("random", "noop"):
        for w in sys.argv[1:]:
            main([w])
    else:
        main(sys.argv[1:])
