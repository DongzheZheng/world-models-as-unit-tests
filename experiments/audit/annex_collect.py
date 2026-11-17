"""Collect the orchestrator's rope-task (annex) audits into runs/probe/annex_audit.json and copy the screening reports.

Reads runs/annex/audit/{cap_v6,cap_v19,code_wm_official}_5000.json, code_wm_ext_5000.json and code_wm_ext_5025.json
(written by annex_audit.py) and runs/annex/audit/screen/report_{1000,5000}.json (written by annex_screen.py report).
usage (from the repository root): python -m experiments.audit.annex_collect
"""
import json
import shutil
from pathlib import Path

import numpy as np

A = Path("runs/annex/audit")


def load(name):
    return json.load(open(A / name))


if __name__ == "__main__":
    out = {}
    for lab in ("cap_v6", "cap_v19", "code_wm_official"):
        r = load(f"{lab}_5000.json")
        out[lab] = r["success"]
        out[lab + "_detail"] = dict(n=r["n"], n_success=r["n_success"], tip_median=r["tip_median"],
                                    within_0p2=r["within_0p2"], wall_s=r["wall_s"])
    ra, rb = load("code_wm_ext_5000.json"), load("code_wm_ext_5025.json")
    eps = ra["episodes"] + rb["episodes"]
    td = np.array([e["tip_dist"] for e in eps])
    ok = [bool(e["success"]) for e in eps]
    out["code_wm_ext"] = float(np.mean(ok))
    out["code_wm_ext_detail"] = dict(n=len(eps), n_success=int(sum(ok)), tip_median=float(np.median(td)),
                                     within_0p2=int((td <= 0.2).sum()), wall_s=ra["wall_s"] + rb["wall_s"],
                                     halves=[ra["n_success"], rb["n_success"]],
                                     steps_mean=float(np.mean([e["n_steps"] for e in eps])),
                                     fail_seeds=[int(e["seed"]) for e in eps if not e["success"]])
    out["seeds"] = "5000-5049"
    Path("runs/probe").mkdir(exist_ok=True)
    json.dump(out, open("runs/probe/annex_audit.json", "w"), indent=1)
    for s in (1000, 5000):
        src = A / "screen" / f"report_{s}.json"
        if src.exists():
            shutil.copy(src, f"runs/probe/annex_screen_{s}.json")
    print(json.dumps(out, indent=1))
