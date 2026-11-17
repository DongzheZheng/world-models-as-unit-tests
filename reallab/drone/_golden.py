"""Lab-only: golden trajectories for the regression test (missions A/B must stay bit-for-bit)."""
import os, sys
os.environ["CAWM_LEDGER"] = "/tmp/golden_ledger.jsonl"
import numpy as np
from reallab.drone.api import make


def traj(mission, mode, seed, n=60):
    env = make("site", "smoke", "golden", mission=mission, command_mode=mode)
    obs, _ = env.reset(seed=seed)
    out = [np.concatenate([np.ravel(v) for v in obs.values()])]
    for i in range(n):
        if mode == "ctbr":
            a = np.array([0.2 * np.sin(0.3 * i), 0.1 * np.cos(0.2 * i), 0.05, 0.31 + 0.02 * np.sin(0.1 * i)])
        else:
            a = np.array([3.0 + 0.1 * i, 0.3 * np.sin(0.2 * i), 0.0, 2.5])
        obs, r, te, tr, info = env.step(a)
        out.append(np.concatenate([np.ravel(v) for v in obs.values()]))
        if te or tr:
            break
    env.close()
    return np.array(out)


if __name__ == "__main__":
    res = {f"{m}_{mode}_{s}": traj(m, mode, s) for m, mode, s in [("A", "ctbr", 5), ("A", "position", 6), ("B", "ctbr", 7)]}
    np.savez_compressed(sys.argv[1], **res)
    print({k: v.shape for k, v in res.items()})
