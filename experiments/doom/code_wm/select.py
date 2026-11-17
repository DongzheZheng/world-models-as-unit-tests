"""Choose the final controller by CODE-WORLD score only (same world seeds for every candidate).
usage: python experiments/doom/code_wm/select.py  -> runs/doom/code_wm/selection.json"""
import json
import sys
import time
from pathlib import Path

import torch

from cawm.ppo import ActorCritic
from experiments.doom.code_wm.planner import plan_act
from experiments.doom.code_wm.train import world_eval

cands = {}
for tag in ("v1", "v2"):
    p = Path(f"runs/doom/code_wm/policy_{tag}.pt")
    if p.exists():
        pi = ActorCritic.load(torch.load(p, map_location="cuda"), "cuda")
        cands[f"ppo_{tag}"] = (lambda pi: lambda o, s, g: pi.act(o, deterministic=True))(pi)
for h, m, w in [(32, 2.0, 0.3), (24, 2.0, 0.3), (32, 4.0, 0.3)]:
    cands[f"planner_h{h}_m{m}_w{w}"] = (lambda h, m, w: lambda o, s, g: plan_act(o, h, m, w))(h, m, w)
from experiments.doom.code_wm.train import ref_policies
out = {k: world_eval(f, n=4096, max_steps=6000, seed=999) for k, f in ref_policies().items()}
print(out, flush=True)
for k, f in cands.items():
    t0 = time.time()
    out[k] = world_eval(f, n=512, max_steps=6000, seed=999)
    out[k]["eval_wall_s"] = time.time() - t0
    print(k, out[k], flush=True)
best = max(cands, key=lambda k: out[k]["mean"])
Path(f"runs/doom/code_wm/selection_{sys.argv[1] if len(sys.argv) > 1 else 'x'}.json").write_text(json.dumps(dict(best=best, candidates=out), indent=1))
print("BEST", best)
