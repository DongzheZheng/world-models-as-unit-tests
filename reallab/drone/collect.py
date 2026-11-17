"""CLI: run policies / open-loop action sequences on the real drone and log every step.

  python -m reallab.drone.collect --task port --purpose calibration --seeds 0:10 --actions seq.npy --out runs/drone/cal_x
  python -m reallab.drone.collect --task waypoints --policy ":random_policy" --seeds 0,1,2 --out runs/drone/cal_y
"""
from __future__ import annotations

import argparse
import json

from .parallel import evaluate


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if ":" in part:
            a, b = part.split(":")
            out += list(range(int(a), int(b)))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["waypoints", "port"])
    ap.add_argument("--purpose", default="calibration")
    ap.add_argument("--tag", default="collect")
    ap.add_argument("--seeds", required=True, help="e.g. 0:10 or 0,5,7")
    ap.add_argument("--policy", default=":hover_policy")
    ap.add_argument("--policy-kwargs", default="{}")
    ap.add_argument("--actions", default=None, help=".npy (T,4): open-loop replay; the episode is capped at T steps unless --no-cap")
    ap.add_argument("--no-cap", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None, help="log directory (gzipped JSONL per episode)")
    ap.add_argument("--difficulty", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args(argv)
    kw = {}
    if a.difficulty:
        kw["difficulty"] = a.difficulty
    max_steps = a.max_steps
    factory, pkw = a.policy, json.loads(a.policy_kwargs)
    if a.actions:
        import numpy as np
        factory, pkw = ":replay_policy", {"path": a.actions}
        if not a.no_cap and max_steps is None:
            max_steps = len(np.load(a.actions))
    if max_steps:
        kw["max_steps"] = max_steps
    rows = evaluate(factory, pkw, a.task, parse_seeds(a.seeds), purpose=a.purpose, tag=a.tag,
                    n_workers=a.workers, log_dir=a.out, **kw)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
