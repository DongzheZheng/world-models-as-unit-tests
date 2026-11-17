"""Lab-side: refine the failure split in runs/stationb/_lab/baselines.json from the step logs (internal).
time_out = time limit hit with the tool not at the target; wrong_side = ended at the target (or overloaded/ended)
with a post on the wrong side."""
import json
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    p = Path("runs/stationb/_lab/baselines.json")
    b = json.loads(p.read_text())
    for name in ("random", "noop", "naive_router", "uniform", "governor"):
        br = {}
        for e in b[name]["per_seed"]:
            d = np.load(f"runs/stationb/_lab/logs/{name}/ep_{e['seed']}.npz")
            dist = float(np.linalg.norm(d["tool_pos"][-1] - d["target"][-1]))
            ok = int(np.sum(d["sides_now"][-1] == d["post_sides"][-1]))
            if e["outcome"] in ("success", "overload"):
                o = e["outcome"]
            elif dist > 0.02:
                o = "time_out_not_at_target" if ok == 3 else "time_out_route_unfinished"
            else:
                o = "wrong_side_at_target" if ok < 3 else "time_out_not_at_rest"
            e["outcome_detail"] = o
            br[o] = br.get(o, 0) + 1
        b[name]["breakdown_detail"] = br
        print(name, b[name]["success"], br)
    p.write_text(json.dumps(b, indent=1))
