"""Phase 2, step 1: reproduce the missed-relay deadlock of the phase-1 controller IN THE CODE WORLD, then check the fix.
The deadlock was first noticed on evaluation seed 1038 (mission A); nothing from that episode is used here.
   python experiments/drone/code_wm/repro_deadlock.py  -> runs/drone/code_wm/phase2_repro.json"""
import json

import numpy as np
import torch

from experiments.drone.code_wm.imagine import load_world, randomise, rollout, summarise

cfg = json.load(open("runs/drone/code_wm/cfg/tune_A123.json"))
tj = json.load(open("runs/drone/code_wm/tune_A123.json"))
belief = load_world("runs/drone/code_wm/fit_l123.json")
world = dict(belief); world.update(cfg["world_override"])
unc = {k: (v[0], tuple(v[1]) if isinstance(v[1], list) else v[1]) for k, v in cfg["unc"].items()}
out = {}
for fix in (0.0, 1.0):
    res = []
    for seed in (777, 778):        # 2 x 2048 randomised imagined lives (seed 777 = the phase-1 selection set)
        g = torch.Generator(device="cuda").manual_seed(seed)
        pw = randomise(world, 2048, "cuda", g, unc)
        pc = dict(tj["controller"]); pc["return_fix"] = fix
        r = rollout(pc, pw, belief, B=2048, seed=seed)
        res.append(r)
    c = np.concatenate([r["cause"] for r in res]); t = np.concatenate([r["t"] for r in res])
    k = np.concatenate([r["k"] for r in res]); fp = np.concatenate([r["final_pos"] for r in res])
    to = (c == 4) | (c == 0)
    stuck = to & (k < 3)
    out[f"return_fix={int(fix)}"] = dict(
        n=int(len(c)), **summarise(dict(cause=c, t=t)),
        timeouts=int(to.sum()), timeouts_relays_reached=np.bincount(k[to], minlength=4).tolist(),
        timeout_final_x_examples=fp[to][:8, 0].round(2).tolist(),
        timeout_final_y_examples=fp[to][:8, 1].round(2).tolist())
    print(json.dumps(out[f"return_fix={int(fix)}"]), flush=True)
json.dump(out, open("runs/drone/code_wm/phase2_repro.json", "w"), indent=1)
