# Replay logged commands in the code world, restarting from the logged state every K steps; print predicted vs real.
#   python experiments/station/code_wm/replay.py FIT.json LOG.npz [K] [every]
import sys, json
import numpy as np
import torch
from worlds.station import station_world as W


def restart_state(E, t0, par, dev="cuda"):
    pts = E["points"][:, :, :2]
    x0 = torch.tensor(pts[t0], dtype=torch.float32, device=dev)[None]
    v0 = (x0 - torch.tensor(pts[max(t0 - 1, 0)], dtype=torch.float32, device=dev)[None]) / W.DT
    v0[:, -1] = torch.tensor(E["tool_vel"][t0], dtype=torch.float32, device=dev) / par.get("vread", 1.0)
    ref = E["actions"][t0 - 1] if t0 >= 1 else E["tool_pos"][0]
    st = W.init_from_obs(dict(points=E["points"][t0][None], tool_pos=E["tool_pos"][t0][None], tool_vel=E["tool_vel"][t0][None],
                              posts=E["posts"][0][None], post_sides=E["post_sides"][0][None], target=E["target"][0][None],
                              load=np.array([E["load"][t0]])), par=par, device=dev)
    st["v"] = v0; st["ref"] = torch.tensor(ref, dtype=torch.float32, device=dev)[None]
    st["t"] = torch.full((1,), t0, dtype=torch.long, device=dev)
    return st


if __name__ == "__main__":
    par = json.load(open(sys.argv[1]))["par"]
    E = np.load(sys.argv[2])
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    T = len(E["actions"])
    A = torch.tensor(E["actions"], dtype=torch.float32, device="cuda")
    st = None
    for t in range(T):
        if t % K == 0:
            st = restart_state(E, t, par)
        st, _, _ = W.step(st, A[t:t + 1])
        st["done"][:] = False
        if (t + 1) % every == 0:
            x = st["x"][0].cpu().numpy(); real = E["points"][t + 1][:, :2]
            still = E["points"][(t // K) * K][:, :2]
            print(f"t={t+1:3d} load real {E['load'][t+1]:6.2f} pred {st['load'].item():6.2f}  pos err {np.linalg.norm(x-real,axis=1).mean()*1000:5.1f} mm "
                  f"(still-baseline {np.linalg.norm(still-real,axis=1).mean()*1000:5.1f})  tool err {np.linalg.norm(x[-1]-real[-1])*1000:5.1f} mm "
                  f"vread real {np.linalg.norm(E['tool_vel'][t+1]):.3f} pred {(st['par']['vread']*st['v'][0,-1]).norm().item():.3f}")
