# Completes runs/station/code_wm/summary.json after the official evaluation (keeps the pre-eval prediction fields).
import json, os, time
import numpy as np
from experiments.station.code_wm.official_eval import KW, REAL_CAL_STEPS, WM_STEPS, WM_WALL

OUT = "runs/station/code_wm"
S = json.load(open(f"{OUT}/summary.json"))
E = json.load(open(f"{OUT}/eval_result.json"))["episodes"]
n = len(E); succ = sum(e["success"] for e in E); ov = sum(e["overload"] for e in E)
wrong = sum((not e["success"]) and (not e["overload"]) and e["sides_ok"] < 3 for e in E)
late = n - succ - ov - wrong
S.update(
 completed=time.strftime("%Y-%m-%d %H:%M:%S"),
 lives_used=3,
 lives=[dict(life=1, seed=18, purpose="scripted probe: speed steps 0.02-0.20 m/s while dragging, lateral drag, slack 0.3 m/s push, near-taut pull",
             outcome="no success intended; 600 steps, no overload, max load 9.97 N (guard stopped the taut pull)", steps=600),
        dict(life=2, seed=3, purpose="end-to-end test of the code-world controller", outcome="success at step 378, max load 4.63 N (world: success 388, 5.35 N)", steps=378),
        dict(life=3, seed=12, purpose="end-to-end test on the code world's hardest snapshot (predicted 9.7 N peak)",
             outcome="success at step 308, max load 4.13 N (world: success 328, 9.71 N)", steps=308)],
 real_steps_total=REAL_CAL_STEPS,
 station_world_bytes=os.path.getsize("worlds/station/station_world.py"),
 controller_type="hand-written waypoint router (pegs on required side at clearance, detours), load-governed speed, snag pause, taut back-off; "
                 "structure written and debugged in the code world, settings selected there (no runtime planning)",
 controller_settings=dict(KW, v_max=0.10, v_min=0.015, lag_pause=0.02, v_end=0.04, end_dist=0.10),
 runtime_compute_per_step="< 0.1 ms (path planned once at the first step; no model at run time)",
 code_world_steps=WM_STEPS, code_world_wall_s=WM_WALL, code_world_steps_per_s=round(WM_STEPS / WM_WALL),
 world_fit=dict(method="CEM over 14 parameters, 256 candidates x 120 segments in one GPU batch, 80 iterations", wall_s=30),
 screening_harness_check=dict(snapshots_200=S["prediction_basis"]["nominal_world_200_snapshots"]["success"], observation_situations_10=1.0,
                              note="experiments/station/code_wm/screen.py run_in_world with this controller (tune.py)"),
 real_success_rate=succ / n,
 real_eval=dict(episodes=n, success=succ, overload=ov, wrong_side=wrong, not_at_target_in_time=late,
                mean_steps=float(np.mean([e["n_steps"] for e in E])), mean_max_load=float(np.mean([e["max_load"] for e in E])),
                max_max_load=float(np.max([e["max_load"] for e in E]))),
 extended_run="not needed (official success >= 50%)",
)
json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
