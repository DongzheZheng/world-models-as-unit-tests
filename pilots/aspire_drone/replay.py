"""Open-loop replay of LOGGED observations through a new program: what it would have commanded at each logged state.
No dynamics are integrated -- the states are the real logged ones.  The program's internal command history is
overwritten with the logged commands so its observers see what was really applied.
   python pilots/aspire_drone/replay.py v2 runs/drone/blackbox/aspire_style/life_1.npz [every]"""
import importlib
import sys

import numpy as np

ver, path = sys.argv[1], sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 3
pol = importlib.import_module(f"pilots.aspire_drone.policy_{ver}").make_policy()
pol.reset()
z = np.load(path)
keys = [k for k in ("ang_vel", "quat", "lin_vel", "pos", "goal", "ranges", "t", "relay_index", "relays", "movers", "n_movers") if k in z.files]
A = z["actions"]
print("  t     x      y   | logged cmd               | new cmd                  | dist_y  r_hat_y")
for i in range(len(A)):
    obs = {k: z[k][i] for k in keys}
    a = pol.act(obs)
    if hasattr(pol, "hist") and pol.hist:
        h = list(pol.hist[-1]); h[2] = float(A[i, 3]); pol.hist[-1] = tuple(h)
    if i % every == 0:
        dd = pol.dbg
        print(f" {i/30:4.2f} {obs['pos'][0]:6.2f} {obs['pos'][1]:5.2f} | {np.array2string(A[i], precision=2, floatmode='fixed')}"
              f" | {np.array2string(a, precision=2, floatmode='fixed')} | {dd.get('dist', [0,0,0])[1]:6.2f} {dd.get('r_hat', [0,0,0])[1]:6.2f}")
