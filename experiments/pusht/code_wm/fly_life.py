"""Fly ONE calibration life and render it.   python experiments/pusht/code_wm/fly_life.py <spec.json>
spec: {"pilot": "code_wm", "seed": k, "factory": "experiments.pusht.code_wm.probe:probe_policy", "kwargs": {...}}"""
import json
import sys
from pathlib import Path

import numpy as np

from reallab.pusht import parallel, render


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    pilot = spec.get("pilot", "code_wm")
    res = parallel.evaluate(spec["factory"], spec.get("kwargs", {}), seeds=[spec["seed"]], purpose="calibration",
                            pilot=pilot, tag=spec.get("tag", "life"), object=spec.get("object", "T"))
    print(json.dumps(res))
    obj = spec.get("object", "T")
    bb = Path("runs/pusht/blackbox" if obj == "T" else "runs/pusht/blackbox_L") / pilot
    lives = sorted(bb.glob("life_*.npz"), key=lambda p: p.stat().st_mtime)
    last = lives[-1]
    d = np.load(last)
    print("life file", last, "seed", int(d["seed"]), "steps", int(d["n_steps"]), "max_cov", float(d["max_coverage"]))
    out = Path("runs/pusht/code_wm/gifs") / (f"{pilot}_{last.stem}.gif" if obj == "T" else f"L_{pilot}_{last.stem}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    render.render_life(str(last), str(out))
    print("gif", out)


if __name__ == "__main__":
    main()
