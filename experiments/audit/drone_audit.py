"""Independent audit of the drone pilots' final controllers on seeds none of them has ever seen (5000-5049).

Run by the orchestrator, not by a pilot.  Both missions, both LLM paradigms, exactly the programs and settings
each pilot used for its official evaluation.  Writes runs/drone/audit.json and keeps step logs for GIFs.
usage: python experiments/audit/drone_audit.py
"""
import json
import statistics
from pathlib import Path

from reallab.drone.parallel import evaluate

SEEDS = list(range(5000, 5050))
CODE_WM = "experiments.drone.code_wm.controller:make_pilot"
CASES = [
    ("aspire_style", "A", "pilots.aspire_drone.policy_v5:make_policy", {}),
    ("code_wm", "A", CODE_WM, json.loads(Path("runs/drone/code_wm/cfg/eval_code_wm.json").read_text())["pilot_kwargs"]),
    ("aspire_style", "B", "pilots.aspire_drone.policy_v6:make_policy", {}),
    ("code_wm", "B", CODE_WM, json.loads(Path("runs/drone/code_wm/cfg/eval_code_wm_B.json").read_text())["pilot_kwargs"]),
]

if __name__ == "__main__":
    out = {}
    for name, mission, factory, kw in CASES:
        rows = evaluate(factory, kw, "site", seeds=SEEDS, purpose="eval", tag=f"audit_{name}_{mission}", n_workers=16,
                        log_dir=f"runs/drone/audit_logs/{name}_{mission}", pilot="audit", mission=mission,
                        command_mode="ctbr")
        ok = [r for r in rows if r["success"]]
        key = f"{name}_{mission}"
        out[key] = dict(n=len(rows), success=len(ok) / len(rows),
                        time_success_s=statistics.fmean(r["time_s"] for r in ok) if ok else None,
                        collision=sum(bool(r["collision"]) for r in rows), timeout=sum(bool(r["timeout"]) for r in rows),
                        out_of_bounds=sum(bool(r["out_of_bounds"]) for r in rows),
                        min_range_seen=min(r["min_range_seen"] for r in rows))
        print(key, out[key], flush=True)
        Path("runs/drone/audit.json").write_text(json.dumps(out, indent=1))
