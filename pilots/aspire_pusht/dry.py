"""Offline crash test on synthetic observations (no env, no dynamics)."""
import sys, importlib
import numpy as np
if __name__ == "__main__":
    m = importlib.import_module(f"pilots.aspire_pusht.policy_v{sys.argv[1]}")
    pol = m.make_policy(); pol.reset()
    rng = np.random.default_rng(0)
    for k in range(200):
        obs = dict(agent_pos=rng.uniform(30, 480, 2), agent_vel=np.zeros(2),
                   block_pose=np.r_[rng.uniform(120, 390, 2), rng.uniform(0, 2*np.pi)],
                   goal_pose=np.array([256, 256, np.pi/4]), coverage=0.1, t=k)
        a = pol.act(obs); assert a.shape == (2,) and np.all(np.isfinite(a))
    print("ok", pol.trace[-3:])
