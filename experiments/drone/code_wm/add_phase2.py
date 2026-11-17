"""Add the phase-2 (mission B) block to runs/drone/code_wm/summary.json."""
import json
import numpy as np

O = "runs/drone/code_wm/"
s = json.load(open(O + "summary.json"))
tj = json.load(open(O + "tune_B.json"))
rows = json.load(open(O + "eval_code_wm_B_rows.json"))
rep = json.load(open(O + "phase2_repro.json"))
st = json.load(open(O + "phase2_stress.json"))
ts = [r["time_s"] for r in rows if r["success"]]
ends = {k: sum(1 for r in rows if r.get(k)) for k in ("success", "collision", "out_of_bounds", "timeout")}
f = tj["finals"][tj["pick"]]
s["phase2_mission_B"] = dict(
    new_real_data=0, real_steps_calibration_recorded=1654, world="fit_all.json (lives 1-5), mechanisms of worlds/drone/site_world.py",
    route=json.load(open(O + "cfg/eval_code_wm_B.json"))["pilot_kwargs"]["route"],
    changes=["route mode with aisle entry/exit points aligned to the jittered R2",
             "missed-relay deadlock fix (first noticed on mission-A eval seed 1038; reproduced and validated in the code world only)",
             "aisle-interior jet randomised (JET_N_GY in [-0.2, 0.1]) since never flown deeper than y ~3.1; tilt limit range raised to 1.4 rad"],
    deadlock_reproduction_in_code_world=rep,
    synthesis=dict(wm_steps=tj["wm_steps"], wall_s=round(tj["wall_s"], 1), wm_steps_per_s=round(tj["wm_steps_per_s"]),
                   picked=tj["pick"], controller=tj["controller"], stress_test=st),
    code_world_prediction=dict(success=f["success"], t_success_s=f["t_success"],
                               end_causes={k: f[k] for k in ("success", "collision", "oob", "timeout")}, imagined_lives=2048),
    real_eval=dict(episodes=len(rows), seeds=[rows[0]["seed"], rows[-1]["seed"]],
                   success_rate=sum(bool(r["success"]) for r in rows) / len(rows),
                   mean_time_success_s=float(np.mean(ts)) if ts else None, end_causes=ends,
                   mean_relays_reached=float(np.mean([r["relays_reached"] for r in rows]))),
    gif="runs/drone/code_wm/gifs/eval_code_wm_B_seed1000.gif",
    record="runs/drone/results/site_B__code_wm.json")
json.dump(s, open(O + "summary.json", "w"), indent=1)
print(json.dumps({k: s["phase2_mission_B"][k] for k in ("code_world_prediction", "real_eval")}, indent=1))
