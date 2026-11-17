"""Open-loop fidelity of the code world on logged lives.
usage: python fidelity.py <params.json> <out.json> <life npz>...   (lives should NOT be the ones the params were fitted to)
For every start reading t (every 4 steps, t>=2) replay the logged commands and report mean node error and p1 error
after 5, 20, 60, 150 steps; plus whole-life replay from the reset reading: final p1 distance real vs world."""
import json
import sys

import numpy as np
import torch

import importlib
import os

W = importlib.import_module("worlds.annex." + os.environ.get("CW_WORLD", "annex_world"))

KEYS = ("points", "tool_pos", "tool_vel", "post", "zone", "target", "t")

if __name__ == "__main__":
    prm = json.load(open(sys.argv[1]))["params"]
    out = dict(params_file=sys.argv[1], horizons={}, whole_life={})
    dev = "cuda"
    res = {h: dict(node=[], p1=[]) for h in (5, 20, 60, 150)}
    for path in sys.argv[3:]:
        L = dict(np.load(path))
        T = len(L["actions"])
        starts = list(range(2, T - 5, 4))
        obs = [{k: L[k][t] for k in KEYS} for t in starts]
        st = W.init_from_obs(obs, [{"points": L["points"][t - 1]} for t in starts], prm, dev,
                             prev2_obs_batch=[{"points": L["points"][t - 2]} for t in starts])
        Hmax = min(150, T - 2)
        A = np.stack([np.concatenate([L["actions"][t:t + Hmax], np.repeat(L["actions"][-1:], max(0, t + Hmax - T), 0)])[:Hmax] for t in starts])
        P, _, _ = W.rollout_points(st, torch.tensor(A, dtype=torch.float32, device=dev))
        P = P.cpu().numpy()
        for h in res:
            for j, t in enumerate(starts):
                if t + h <= T and h <= Hmax:
                    e = np.linalg.norm(P[j, h - 1] - L["points"][t + h, :, :2], axis=1)
                    res[h]["node"].append(float(e.mean()))
                    res[h]["p1"].append(float(e[0]))
        # whole life from the reset reading (zero initial velocity: a single reading carries none)
        st0 = W.init_from_obs([{k: L[k][0] for k in KEYS}], None, prm, dev)
        P0, _, _ = W.rollout_points(st0, torch.tensor(L["actions"][None], dtype=torch.float32, device=dev))
        # and from reading 2 with velocity (fair start)
        st2 = W.init_from_obs([{k: L[k][2] for k in KEYS}], [{"points": L["points"][1]}], prm, dev,
                              prev2_obs_batch=[{"points": L["points"][0]}])
        P2, _, _ = W.rollout_points(st2, torch.tensor(L["actions"][None, 2:], dtype=torch.float32, device=dev))
        out["whole_life"][path] = dict(
            real_final_p1=L["points"][-1, 0, :2].tolist(), world_final_p1_from_reset=P0[0, -1, 0].tolist(),
            dist_from_reset=float(np.linalg.norm(P0[0, -1, 0].cpu().numpy() - L["points"][-1, 0, :2])),
            dist_from_reading2_with_velocity=float(np.linalg.norm(P2[0, -1, 0].cpu().numpy() - L["points"][-1, 0, :2])),
            real_success=bool(L["success"]), n_steps=int(T))
    for h, r in res.items():
        if r["node"]:
            out["horizons"][str(h)] = dict(n=len(r["node"]), node_mean=float(np.mean(r["node"])), node_median=float(np.median(r["node"])),
                                           p1_mean=float(np.mean(r["p1"])), p1_median=float(np.median(r["p1"])), p1_p90=float(np.quantile(r["p1"], 0.9)))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out, indent=1))
