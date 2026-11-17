"""Phase 2: stress-test mission-B candidates in the code world beyond the training randomisation (selection in imagination).
   python experiments/drone/code_wm/stress_B.py -> runs/drone/code_wm/phase2_stress.json"""
import json

import torch

from experiments.drone.code_wm.imagine import load_world, randomise, rollout, summarise

cfg = json.load(open("runs/drone/code_wm/cfg/tune_B.json"))
tj = json.load(open("runs/drone/code_wm/tune_B.json"))
belief = load_world(cfg["world"])
base = dict(belief); base.update(cfg["world_override"])
unc = {k: (v[0], tuple(v[1]) if isinstance(v[1], list) else v[1]) for k, v in cfg["unc"].items()}
mission = tuple(tuple(r) for r in cfg["mission"])
cands = {n: dict(tj["controller"], **tj["finals"][n]["params"]) for n in ("best", "mean")}
cands["best_tilt1.2"] = dict(cands["best"], max_tilt=1.2)
cands["best_no_fix"] = dict(cands["best"], return_fix=0.0)
scen = {
    "training_randomisation": {},
    "nominal_no_randomisation": None,
    "aisle_jet_x2.2_inside (GY=-0.3)": {"JET_N_GY": -0.3},
    "aisle_jet_weak_inside (GY=+0.15)": {"JET_N_GY": 0.15},
    "jets_x1.5": {"JET_N_AY": belief["JET_N_AY"] * 1.5, "JET_S_AY": belief["JET_S_AY"] * 1.5},
    "thrust_gain_-8pct": {"THRUST_GAIN": belief["THRUST_GAIN"] * 0.92},
    "delay_2_steps": {"CMD_DELAY": 2.0},
    "gusts_x2": {"GUST_STD": 2.0},
}
out = {}
for sn, ov in scen.items():
    out[sn] = {}
    for cn, pc in cands.items():
        g = torch.Generator(device="cuda").manual_seed(4242)
        if ov is None:
            pw = dict(base)
        else:
            pw = randomise(base, 1024, "cuda", g, unc)
            pw.update(ov)
        r = rollout(pc, pw, belief, B=1024, seed=4242, mission=mission, time_limit=cfg["time_limit"], route=cfg["route"])
        out[sn][cn] = summarise(r)
    print(sn, json.dumps({c: (round(v["success"], 3), v["t_success"] and round(v["t_success"], 2)) for c, v in out[sn].items()}), flush=True)
json.dump(out, open("runs/drone/code_wm/phase2_stress.json", "w"), indent=1)
