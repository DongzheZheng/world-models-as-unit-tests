"""Sanity test of the mission-C code world: movers move like life 1, collisions register, avoidance runs batched."""
import json
import numpy as np
import torch
from experiments.drone.code_wm.controller import DEFAULT
from experiments.drone.code_wm.imagine import load_world, rollout, summarise
from worlds.drone import site_world as W

belief = load_world("runs/drone/code_wm/fit_all.json")
tj = json.load(open("runs/drone/code_wm/tune_A123.json"))
route = ["R1", [20, 0, 4.8], [26, 0, 4.8], "R2", [31, 0, 4.8], [50, 0, 4.8], "R3"]
for name, pc, rt in (("phase1_A_controller_no_avoid", dict(tj["controller"], return_fix=1.0), None),
                     ("lane4.8_no_avoid", dict(tj["controller"], return_fix=1.0), route),
                     ("lane4.8_avoid", dict(tj["controller"], return_fix=1.0, av_k=12.0, av_R=2.0, av_T=1.5, av_up=0.5), route)):
    r = rollout(pc, belief, belief, B=1024, seed=1, time_limit=W.TIME_LIMIT_C, movers={}, route=rt, record=True)
    print(name, summarise(r))
print("mover sample traj env0 (every 1.5 s):")
for i in range(0, r["mtraj"].shape[1], 45):
    print(i / 30, r["mtraj"][0, i].round(1).tolist()[:4])
