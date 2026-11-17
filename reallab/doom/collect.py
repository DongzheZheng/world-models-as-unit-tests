"""CLI: run a policy for N episodes on one scenario and log every step (calibration data collection).

  python -m reallab.doom.collect --scenario take_cover --episodes 20 --policy reallab.doom.parallel:random_policy \
      --seed0 0 --purpose calibration --pilot <pilot> --tag <tag> --out runs/doom/calib/<name>

Writes <out>/ep_<seed>.jsonl.gz (format: see reallab.doom.parallel docstring) and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reallab.doom.parallel import evaluate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--policy", default="reallab.doom.parallel:random_policy")
    ap.add_argument("--policy-kwargs", default="{}", help="JSON dict passed to the factory")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--purpose", default="calibration")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pilot", default=None, help="required for purpose=calibration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--record-first", type=int, default=0)
    a = ap.parse_args(argv)
    seeds = list(range(a.seed0, a.seed0 + a.episodes))
    res = evaluate(a.policy, json.loads(a.policy_kwargs), a.scenario, seeds, purpose=a.purpose, tag=a.tag,
                   n_workers=a.workers, frame_skip=a.frame_skip, log_dir=a.out, record_first=a.record_first, pilot=a.pilot)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "summary.json").write_text(json.dumps(res, indent=1))
    print(f"{a.scenario}: {len(res)} episodes, {sum(r['n_steps'] for r in res)} agent steps, "
          f"mean return {sum(r['ret'] for r in res) / len(res):.3f}")


if __name__ == "__main__":
    main()
