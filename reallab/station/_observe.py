"""Lab-side: produce the observation set (naive router, real installation, seeds 900-909) and two GIFs of it."""
import json
from pathlib import Path

if __name__ == "__main__":
    from reallab.station import _reference
    from reallab.station._render import render_life
    res = _reference.run_observation(range(900, 910))
    idx = json.loads(Path("runs/station/observation/index.json").read_text())
    print(json.dumps(idx, indent=1), flush=True)
    eps = idx["episodes"]
    succ = [e for e in eps if e["outcome"] == "success"]
    fail = [e for e in eps if e["outcome"] != "success"]
    for tag, lst in (("success", succ), ("failure", fail)):
        if lst:
            s = lst[0]["seed"]
            out = render_life(f"runs/station/observation/episode_{s}.npz", f"runs/station/_lab/gifs/observation_{s}_{tag}.gif")
            print("gif", out, flush=True)
