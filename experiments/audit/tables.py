"""Orchestrator's read-only check of recorded results (no real-world interaction).

usage: python experiments/audit/tables.py            -> one line per recorded controller, all demos
       python experiments/audit/tables.py ppo-check  -> sanity-check the PPO flights that finish mission A in ~3 s
"""
import glob
import json
import os
import sys

import numpy as np


def table():
    for p in sorted(glob.glob("runs/*/results/*.json")):
        r = json.load(open(p))
        e = r["eval"]
        extra = {k: (None if v["mean"] is None else round(v["mean"], 3)) for k, v in r.get("other", {}).items()}
        print(f"{r['demo']:6s} {r['scenario']:11s} {r['controller']:26s} {r['family']:15s} {r['metric']:8s} "
              f"mean {e['mean']:10.3f} sem {e['sem']:8.3f} n {e['n']:3d}  cal {r['real_steps_calibration']:6d} "
              f"train {r['real_steps_training']:8d}  {extra}")


def ppo_check():
    files = [f for f in sorted(glob.glob("runs/drone/baselines_rl/evlogs/**/*", recursive=True)) if os.path.isfile(f)]
    print("log files:", len(files), [os.path.relpath(f, "runs/drone/baselines_rl/evlogs") for f in files[:4]])
    from reallab.drone.render import load_log
    shown = 0
    for f in files:
        if "14811" not in f:
            continue
        try:
            log = load_log(f)
        except Exception as ex:                      # not an episode log
            continue
        pos, t = np.asarray(log["pos"]), np.asarray(log["t"]).reshape(-1)
        v = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
        print(os.path.basename(f), "outcome", log.get("outcome"), "steps", len(pos), "t_end", round(float(t[-1]), 2),
              "x range", round(float(pos[:, 0].min()), 1), round(float(pos[:, 0].max()), 1),
              "|y| max", round(float(np.abs(pos[:, 1]).max()), 2), "z range", round(float(pos[:, 2].min()), 2),
              round(float(pos[:, 2].max()), 2), "speed mean/max", round(float(v.mean()), 1), round(float(v.max()), 1),
              "relay_index end", np.asarray(log["relay_index"]).reshape(-1)[-1] if "relay_index" in log else None)
        shown += 1
        if shown >= 4:
            break


if __name__ == "__main__":
    ppo_check() if len(sys.argv) > 1 else table()
