"""Independent audit of the Doom pilots' final controllers on seeds none of them has ever seen.

Run by the orchestrator, not by a pilot.  Seeds 5000-5049 (the pilots' official evaluation used 1000-1099).
Writes runs/doom/audit.json.  usage: python experiments/audit/doom_audit.py [n_workers]
"""
import json
import statistics
import sys
from pathlib import Path

from reallab.doom.parallel import evaluate

SEEDS = list(range(5000, 5050))
PILOTS = {
    "aspire_style": ("pilots.aspire_doom.policy_v5:make_policy", {}),
    "code_wm": ("experiments.doom.code_wm.planner:planner", dict(horizon=24, margin=2.0, w_center=0.3)),
}

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    out = {}
    for name, (factory, kw) in PILOTS.items():
        eps = evaluate(factory, kw, scenario="take_cover", seeds=SEEDS, purpose="eval", pilot="audit",
                       tag=f"audit_{name}", n_workers=workers)
        rets = [e["ret"] for e in eps]
        out[name] = dict(n=len(rets), mean=statistics.fmean(rets), median=statistics.median(rets),
                         sem=statistics.pstdev(rets) / len(rets) ** 0.5, min=min(rets), max=max(rets),
                         frac_ge_750=sum(r >= 750 for r in rets) / len(rets), rets=rets)
        print(name, {k: v for k, v in out[name].items() if k != "rets"}, flush=True)
        Path("runs/doom/audit.json").write_text(json.dumps(out, indent=1))
