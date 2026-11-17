"""Orchestrator audit of a rope-task (annex) controller on the real lab, on seeds no pilot has seen.

usage (from the repository root):
    python -m experiments.audit.annex_audit <factory> <label> <first_seed> <n> [workers]
e.g. python -m experiments.audit.annex_audit pilots.aspire_annex.policy_v6:make_policy cap_v6 5000 50
     python -m experiments.audit.annex_audit experiments.annex.code_wm.throw:make_final code_wm_official 5000 50
     python -m experiments.audit.annex_audit experiments.annex.code_wm.throw_v2:make_robust code_wm_ext 5000 25
     python -m experiments.audit.annex_audit experiments.annex.code_wm.throw_v2:make_robust code_wm_ext 5025 25
Writes runs/annex/audit/<label>_<first_seed>.json (step logs under runs/annex/audit/logs_<label>_<first_seed>/).
experiments.audit.annex_collect gathers these files into runs/probe/annex_audit.json.
"""
import json
import sys
import time
from pathlib import Path

OUT = Path("runs/annex/audit")

if __name__ == "__main__":
    import numpy as np
    from reallab.annex.parallel import evaluate
    fac, label, s0, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = evaluate(fac, {}, seeds=list(range(s0, s0 + n)), purpose="eval", pilot="", tag=f"audit_{label}",
                   n_workers=nw, log_dir=str(OUT / f"logs_{label}_{s0}"))
    td = np.array([e["tip_dist"] for e in eps])
    r = dict(factory=fac, label=label, first_seed=s0, n=n, success=float(np.mean([bool(e["success"]) for e in eps])),
             n_success=int(sum(bool(e["success"]) for e in eps)), tip_median=float(np.median(td)), tip_mean=float(td.mean()),
             within_0p2=int((td <= 0.2).sum()), wall_s=time.time() - t0, episodes=eps)
    json.dump(r, open(OUT / f"{label}_{s0}.json", "w"), indent=1, default=float)
    print(label, s0, "success", r["n_success"], "/", n, "tip median", round(r["tip_median"], 3), "wall", round(r["wall_s"]))
