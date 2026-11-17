"""Offline checks of policy_v6 without any dynamics:
  1. mission-A regression: replay logged lives 2-5 through v5 and v6, compare the commands at the logged states
  2. route planner on mission B nominal + jittered relays: corners, leg clearance
  3. static guidance snapshots along the planned B route (what v6 would command at hand-placed states)"""
import numpy as np

from pilots.aspire_drone import policy_v5, policy_v6

keys = ("ang_vel", "quat", "lin_vel", "pos", "goal", "ranges", "t", "relay_index", "relays")
for L in (2, 3, 4, 5):
    z = np.load(f"runs/drone/blackbox/aspire_style/life_{L}.npz")
    pols = [policy_v5.make_policy(), policy_v6.make_policy()]
    for p in pols:
        p.reset()
    diffs = []
    for i in range(len(z["actions"])):
        obs = {k: z[k][i] for k in keys}
        a5, a6 = [p.act(obs) for p in pols]
        for p in pols:
            h = list(p.hist[-1]); h[2] = float(z["actions"][i, 3]); p.hist[-1] = tuple(h)
        diffs.append(np.abs(a5 - a6).max())
    print(f"life {L}: max |v6 - v5| command diff {max(diffs):.4f}, steps with diff > 0.01: {sum(d > 0.01 for d in diffs)}")

B = np.array([[17, 0, 2.5], [23.75, 6, 2.5], [57, 0, 2.5]])
start = np.array([3.0, 0, 2])
wps = policy_v6.plan_route(start, B)
print("nominal B route:", [(np.round(w[0], 2).tolist(), w[1]) for w in wps])
rng = np.random.default_rng(0)
mins, ncorner, lengths = [], [], []
for _ in range(500):
    r = B + rng.uniform(-0.5, 0.5, B.shape)
    s = start + np.r_[rng.uniform(-1, 1, 2), 0]
    w = policy_v6.plan_route(s, r)
    ncorner.append(sum(k == "corner" for _, k, _ in w))
    pts = [x[0] for x in w]
    lengths.append(sum(np.linalg.norm(b - a) for a, b in zip(pts, pts[1:])))
    m = 9.0
    for a, b in zip(pts, pts[1:]):
        for s_ in np.linspace(0, 1, 200):
            q = a + s_ * (b - a)
            m = min(m, policy_v6.box_clearance(q[0], q[1])[0])
    mins.append(m)
print("jittered B: corners per route", sorted(set(ncorner)), "min leg clearance %.2f m" % min(mins),
      "route length %.1f-%.1f m" % (min(lengths), max(lengths)))
wA = policy_v6.plan_route(start, [[14, 0, 2.5], [28, 0, 2.5], [57, 0, 2.5]])
print("mission A route:", [w[1] for w in wA])

pol = policy_v6.make_policy()
for name, pos, vel, ri in [("canyon 20,0 cruising", [20, 0.1, 2.5], [6, 0, 0], 1),
                           ("near corner 23.0", [23.0, 0.3, 2.5], [2.5, 0, 0], 1),
                           ("aisle y=2 going up", [23.75, 2.0, 2.5], [0, 2.5, 0], 1),
                           ("aisle y=4.5 going up", [23.75, 4.5, 2.5], [0, 2.0, 0], 1),
                           ("aisle y=4 coming down", [23.75, 4.0, 2.5], [0, -2.5, 0], 2),
                           ("aisle y=1 coming down", [23.75, 1.0, 2.5], [0, -2.0, 0], 2)]:
    pol.reset()
    obs = dict(pos=np.array(pos, float), lin_vel=np.array(vel, float), quat=np.array([0, 0, 0, 1.0]),
               ang_vel=np.zeros(3), t=np.array([5.0]), relay_index=np.array([float(ri)]), relays=B, goal=B[ri])
    pol.wps = policy_v6.plan_route(start, B)
    a = pol.act(obs)
    print(f"{name:24s} wp {pol.dbg['wi']} vref {pol.dbg['vref']:5.2f} a_cmd {np.round(pol.dbg['a'], 1)} "
          f"jet {np.round(pol.dbg['dist'], 1)} cmd {np.round(a, 2)}")
