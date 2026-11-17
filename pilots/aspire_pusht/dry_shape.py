"""Offline checks of a shape-generic program: contact geometry clearance and crash test on synthetic observations."""
import sys, importlib
import numpy as np
import shapely.geometry as sg
if __name__ == "__main__":
    m = importlib.import_module(f"pilots.aspire_pusht.policy_v{sys.argv[1]}")
    shape = sys.argv[2]
    goal = {"T": [256, 256, np.pi/4], "L": [300, 200, np.pi/2]}[shape]
    pol = m.make_policy(); pol.bind({"goal_pose": goal, "object": shape} if shape == "L" else {"goal_pose": goal, "block": {"shape": "T"}})
    body = pol.poly(np.zeros(3))
    touch = pol.P - pol.N * m.R_PUSHER
    dmin = min(body.distance(sg.Point(t)) for t in touch)
    print(shape, "n contacts", len(pol.P), "min clearance of contact pusher centres %.2f (should be ~15)" % dmin,
          "centroid", np.round(body.centroid.coords[0], 1), "max vertex radius %.1f" % max(np.linalg.norm(np.array(body.exterior.coords) - body.centroid.coords[0], axis=1)) if body.geom_type == "Polygon" else "")
    pol.reset(); rng = np.random.default_rng(0)
    for k in range(200):
        obs = dict(agent_pos=rng.uniform(30, 480, 2), agent_vel=np.zeros(2),
                   block_pose=np.r_[rng.uniform(120, 390, 2), rng.uniform(0, 2*np.pi)],
                   goal_pose=np.array(goal), coverage=0.1, t=k)
        a = pol.act(obs); assert a.shape == (2,) and np.all(np.isfinite(a))
    print("ok")
