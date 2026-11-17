import hashlib, numpy as np
from reallab.pusht import api
def trajectory_hash(seed=1234, n=80, rng_seed=0):
    env = api.make("smoke"); rng = np.random.default_rng(rng_seed)
    o = env.reset(seed); h = hashlib.sha256(); steps = 0
    def add(o, r=0.0):
        for k in ("agent_pos", "agent_vel", "block_pose", "goal_pose"): h.update(np.asarray(o[k], np.float64).tobytes())
        h.update(np.float64(o["coverage"]).tobytes()); h.update(np.float64(r).tobytes())
    add(o)
    for _ in range(n):
        a = rng.uniform(100, 420, 2) if steps % 7 else o["block_pose"][:2] + rng.uniform(-30, 30, 2)
        o, r, te, tr, info = env.step(a); steps += 1; add(o, r)
        if te or tr: break
    env.close(); return h.hexdigest(), o["block_pose"].tolist(), o["coverage"]
if __name__ == "__main__": print(trajectory_hash())
