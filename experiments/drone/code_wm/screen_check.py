"""Harness check: code_wm's own evaluated controllers through screen.run_in_world (should match the GPU predictions)."""
import json, time
from experiments.drone.code_wm.screen import run_in_world, summarise
t0 = time.time()
print("own A", json.dumps(summarise(run_in_world("experiments.drone.code_wm.controller:make_pilot",
      {"tuned": "runs/drone/code_wm/tune_A123.json", "belief": "runs/drone/code_wm/fit_l123.json"}, "A", 100, 11))), round(time.time() - t0))
t0 = time.time()
print("own C", json.dumps(summarise(run_in_world("experiments.drone.code_wm.controller:make_pilot",
      json.load(open("runs/drone/code_wm/cfg/C_final.json")), "C", 100, 12))), round(time.time() - t0))
t0 = time.time()
print("v1 A", json.dumps(summarise(run_in_world("pilots.aspire_drone.policy_v1:make_policy", {}, "A", 20, 13))), round(time.time() - t0))
