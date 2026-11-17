"""Independent audit of the "code-as-policy inside a code world" programs on unseen seeds (5000+).
Run by the orchestrator.  usage: python experiments/audit/hybrid_audit.py drone|pusht   -> runs/probe/hybrid_audit.json

Everything runs under the __main__ guard: the labs' evaluate() starts worker processes with the "spawn" method,
which re-imports this file in every worker.
"""
import json
import statistics
import sys
from pathlib import Path

OUT = Path("runs/probe/hybrid_audit.json")


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    what = sys.argv[1]
    if what == "drone":
        from reallab.drone.parallel import evaluate
        ver = sys.argv[2] if len(sys.argv) > 2 else "15"
        rows = evaluate(f"pilots.aspire_drone.policy_v{ver}:make_policy", {}, "site", seeds=list(range(5000, 5100)),
                        purpose="eval", tag=f"audit_cap_in_world_v{ver}", n_workers=16,
                        log_dir=f"runs/drone/audit_logs/cap_in_world_v{ver}", pilot="audit", mission="C", command_mode="ctbr")
        ok = [r for r in rows if r["success"]]
        out[f"drone_C/cap_in_code_world_v{ver}"] = dict(
            n=len(rows), success=len(ok) / len(rows), time_success_s=statistics.fmean(r["time_s"] for r in ok) if ok else None,
            collision=sum(bool(r["collision"]) for r in rows), timeout=sum(bool(r["timeout"]) for r in rows),
            out_of_bounds=sum(bool(r["out_of_bounds"]) for r in rows))
    else:
        from reallab.pusht.parallel import evaluate
        ver, obj = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "L")
        eps = evaluate(f"pilots.aspire_pusht.policy_v{ver}:make_policy", {}, seeds=list(range(5000, 5050)), purpose="eval",
                       pilot="audit", tag=f"audit_cap_in_world_v{ver}", n_workers=12, object=obj)
        out[f"push_{obj}/cap_in_code_world_v{ver}"] = dict(
            n=len(eps), success=statistics.fmean(float(e["success"]) for e in eps),
            max_coverage=statistics.fmean(e["max_coverage"] for e in eps), n_steps=statistics.fmean(e["n_steps"] for e in eps))
    out = {**(json.loads(OUT.read_text()) if OUT.exists() else {}), **out}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
