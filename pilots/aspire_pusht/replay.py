"""Replay logged observations through a program to see what it commands (no dynamics; observations are the log's).
python pilots/aspire_pusht/replay.py <version> <npz>"""
import sys, importlib
import numpy as np
if __name__ == "__main__":
    m = importlib.import_module(f"pilots.aspire_pusht.policy_v{sys.argv[1]}")
    d = np.load(sys.argv[2]); pol = m.make_policy(); pol.reset()
    T = len(d["actions"]); dev = []
    for t in range(T):
        obs = dict(agent_pos=d["agent_pos"][t], agent_vel=d["agent_vel"][t], block_pose=d["block_pose"][t],
                   goal_pose=d["goal_pose"][t], coverage=float(d["coverage"][t]), t=t)
        a = pol.act(obs); dev.append(np.linalg.norm(a - d["actions"][t]))
    tr = pol.trace; segs = []
    for t, (mode, i, s) in enumerate(tr):
        key = (mode, i)
        if not segs or segs[-1][0] != key: segs.append([key, t, 1, s])
        else: segs[-1][2] += 1
    for (mode, i), t0, n, s in segs: print(f"t={t0:3d} {mode:8s} contact={i:3d} n={n:3d} score={s:.2f} cov={d['coverage'][t0]:.3f}")
    print("max |action - logged| = %.3g" % max(dev))
