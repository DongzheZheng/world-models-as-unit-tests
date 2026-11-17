"""World sanity: snapshot 0, give the rod a 1 m/s spin, hold the tool, print max node speed over time."""
import json
import sys

import numpy as np
import torch

from worlds.annex import annex_world as W
from experiments.annex.code_wm.screen import snapshot_obs, world_params

if __name__ == "__main__":
    prm = world_params(json.load(open(sys.argv[1]))["params"] if len(sys.argv) > 1 else None)
    o = snapshot_obs([0])[0]
    st = W.init_from_obs([o], None, prm, "cuda")
    rel = st["pos"] - st["pos"][:, -1:]
    st["vel"] = torch.stack([-rel[..., 1], rel[..., 0]], -1) * 1.5
    st["vel"][:, -1] = 0
    a = torch.tensor(o["tool_pos"][None], dtype=torch.float32, device="cuda")
    for k in range(236):
        st, r, d = W.step(st, a)
        if k % 10 == 0:
            sp = st["vel"].norm(dim=-1)[0]
            print(k, f"max {float(sp.max()):.3f} tip {float(sp[0]):.3f} done {bool(d[0])}")
        if bool(d[0]):
            print("done at", k, "success", bool(st["success"][0]))
            break
