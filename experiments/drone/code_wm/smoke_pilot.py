"""Offline smoke test of the real-world Pilot wrapper on synthetic observations (no environment)."""
import json, sys
import numpy as np
from experiments.drone.code_wm.controller import Pilot
p = Pilot(**json.load(open(sys.argv[1])))
rel = np.array([[17, 0, 2.5], [23.9, 6.2, 2.5], [57, 0, 2.5]])
for pos, ri in (([3, 0, 2], 0), ([20, 0, 2.5], 1), ([23.9, 0.3, 2.5], 1), ([23.9, 5.0, 2.5], 1), ([23.9, 4.0, 2.5], 2)):
    obs = dict(pos=np.array(pos, float), lin_vel=np.zeros(3), quat=np.array([0, 0, 0, 1.]), relays=rel,
               relay_index=np.array([ri]), t=np.array([1.0]))
    a = p.act(obs)
    print(pos, ri, "segment", int(p.k[0]), "action", a.round(3))
