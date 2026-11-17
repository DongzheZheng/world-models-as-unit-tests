"""Closed-loop check: fly the life-1 controller (datasheet gains, no compensation) in a code world and report outcomes."""
import sys
import numpy as np
from experiments.drone.code_wm.controller import DEFAULT
from experiments.drone.code_wm.imagine import load_world, rollout, summarise

for wpath in sys.argv[1:]:
    pw = load_world(None if wpath == "datasheet" else wpath)
    pc = dict(DEFAULT)
    r = rollout(pc, pw, pw, B=256, id_time=2.0, wind_ff=False, record=True)
    print(wpath, summarise(r), "mean end x", r["final_pos"][:, 0].mean().round(2), "end pos[0]", r["final_pos"][0].round(2), "t", r["t"][0])
