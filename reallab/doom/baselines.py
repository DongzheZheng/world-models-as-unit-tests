"""Run the trivial random / no-op baselines (purpose=eval) and write runs/doom/baselines.json."""
import json
from pathlib import Path

import numpy as np

from reallab.doom.parallel import evaluate

SCENS = ["take_cover", "defend_the_center", "health_gathering", "predict_position", "basic"]
POLS = {"random": "reallab.doom.parallel:random_policy", "noop": "reallab.doom.parallel:noop_policy"}


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), std=float(x.std()), median=float(np.median(x)), min=float(x.min()),
                max=float(x.max()))


def main():
    seeds = list(range(1000, 1100))
    out = {}
    for sc in SCENS:
        for name, fac in POLS.items():
            kw = {"seed": 0} if name == "random" else {}
            res = evaluate(fac, kw, sc, seeds, purpose="eval", tag=f"baseline_{name}", n_workers=32)
            out.setdefault(sc, {})[name] = dict(
                n=len(res), ret=stats([r["ret"] for r in res]), length_tics=stats([r["length_tics"] for r in res]),
                frac_terminated=float(np.mean([r["terminated"] for r in res])),
                frac_truncated=float(np.mean([r["truncated"] for r in res])),
                mean_kills=float(np.mean([r["player"]["killcount"] for r in res])))
            print(sc, name, out[sc][name]["ret"]["mean"], out[sc][name]["length_tics"]["mean"], flush=True)
    Path("runs/doom").mkdir(parents=True, exist_ok=True)
    Path("runs/doom/baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
