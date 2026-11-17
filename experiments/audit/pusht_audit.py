"""Independent audit of Push-T controllers on seeds no pilot has ever seen (5000-5049).

Run by the orchestrator, not by a pilot.  usage: python experiments/audit/pusht_audit.py name=factory[:json_kwargs_file] ...
e.g.   python experiments/audit/pusht_audit.py aspire_v5=pilots.aspire_pusht.policy_v5:make_policy
Appends to runs/pusht/audit.json.
"""
import json
import statistics
import sys
from pathlib import Path

from reallab.pusht.parallel import evaluate

SEEDS = list(range(5000, 5050))
OUT = Path("runs/pusht/audit.json")

if __name__ == "__main__":
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for arg in sys.argv[1:]:
        name, spec = arg.split("=", 1)
        factory, _, kwfile = spec.partition("@")
        kw = json.loads(Path(kwfile).read_text()) if kwfile else {}
        eps = evaluate(factory, kw, seeds=SEEDS, purpose="eval", pilot="audit", tag=f"audit_{name}", n_workers=10)
        out[name] = dict(n=len(eps), success=statistics.fmean(float(e["success"]) for e in eps),
                         max_coverage=statistics.fmean(e["max_coverage"] for e in eps),
                         n_steps=statistics.fmean(e["n_steps"] for e in eps))
        print(name, out[name], flush=True)
        OUT.write_text(json.dumps(out, indent=1))
