"""Add the phase-3 (mission C, other aircraft) block to runs/drone/code_wm/summary.json."""
import json
import os
import numpy as np
from worlds.drone import site_world as W

O = "runs/drone/code_wm/"
s = json.load(open(O + "summary.json"))
purposes = {1: "observe: hover at the start for the whole 19.4 s window; the instrumentation records every mover's true track (baseline behaviour, no interaction)",
            2: "first mission-C controller (altitude lane z 4.8 + reactive avoidance) synthesised with a mover model from life 1; revealed the pursuing 'guard' and shuttles running through the side aisles",
            3: "re-synthesised controller (guards modelled); revealed 3-D pursuit and one dodge, faster shuttles",
            4: "controller re-synthesised with 3-D pursuit guards and the best of 3 route shapes; revealed slower wanderers and more y-shuttles",
            5: "end-to-end test of exactly the evaluated controller (prior updated with life 4); no new behaviour type; closest pass 0.67 m"}
lives = []
for k in range(1, 6):
    z = np.load(f"runs/drone/blackbox/code_wm_C/life_{k}.npz")
    lives.append(dict(life=k, seed=int(z["seed"]), real_steps=int(z["n_steps"]), end_cause=str(z["end_cause"]),
                      time_s=round(int(z["n_steps"]) / 30, 3), relays_reached=int(z["relays_reached"]), purpose=purposes[k],
                      gif=f"runs/drone/code_wm/gifs/code_wm_C_life_{k}.gif"))
mv = json.load(open(O + "movers_C_all.json"))
census = []
for L in mv:
    kinds = {}
    for m in L["movers"]:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    census.append(dict(life=L["life"].split("/")[-1], n_movers=len(L["movers"]), kinds_by_track_shape=kinds,
                       speeds=sorted(sp for m in L["movers"] for sp in m["speeds"]),
                       z_range=[min(m["z_level"][0] for m in L["movers"]), max(m["z_level"][1] for m in L["movers"])]))
tj = json.load(open(O + "tune_C5.json"))
f = tj["finals"][tj["pick"]]
rows = json.load(open(O + "eval_code_wm_C_rows.json"))
ts = [r["time_s"] for r in rows if r["success"]]
ends = {k: sum(1 for r in rows if r.get(k)) for k in ("success", "collision", "out_of_bounds", "timeout")}
runs = {}
for n in ["C1", "C2", "C3a", "C3b", "C3c", "C4", "C5"]:
    d = json.load(open(O + f"tune_{n}.json")); fp = d["finals"][d["pick"]]
    runs[n] = dict(wm_steps=d["wm_steps"], wall_s=round(d["wall_s"], 1), route=d["cfg"].get("route"),
                   code_world_success=fp["success"], t_success=fp["t_success"])
s["phase3_mission_C"] = dict(
    lives_used=5, pilot="code_wm_C", real_steps_per_life=[l["real_steps"] for l in lives],
    real_steps_total=sum(l["real_steps"] for l in lives), lives=lives,
    other_aircraft_model=dict(
        where="worlds/drone/site_world.py (sample_movers, movers_step, mover_hit, observe_movers)",
        mechanisms={
            "shuttle_x": "back and forth along x at fixed (y, z), constant speed, short dwell at the ends (seen in all 5 lives)",
            "shuttle_y": "back and forth across the canyon at fixed (x, z); some run deep into a side aisle (to |y| ~6.5 at x ~23-24.6 and ~39-40; lives 2-5)",
            "wanderer": "hops at 3-6 m/s to random points of its own canyon box, dwells 0.3-7.5 s (lives 1, 2, 4, 5)",
            "guard": "an x-shuttle that, once we are within ~1.5-3.5 m, leaves its lane and flies toward us at 1-2 m/s (3-D, climbing to our altitude) until we are 5-10 m away, then returns (life 2 mover 4, life 3 mover 2; 2 of 8 close passes)"},
        fitted_ranges=W.MOVER_PRIOR, mover_radius=W.MOVER_R,
        per_life_census=census,
        not_observed_or_not_modelled=[
            "whether wanderers ever pursue: their hops sometimes pointed at us (cos 0.97-0.99), sometimes away; modelled as a random CHASE probability 0-0.3 per hop",
            "the 'dodge' reaction (life 3 mover 0 moved away at 3.5 m/s for 0.8 s) is not modelled",
            "how high a pursuing guard can climb (seen up to z 4.2, capped at 5.5 in the world) and its exact trigger rule (distance vs alignment)",
            "line of sight in the observation is not modelled (inside the canyon all movers within 15 m were visible in the logs)",
            "the mover population is a random draw per episode; 5 lives give 39 mover tracks, so rarer behaviours may be missing"]),
    controller_type="phase-1/2 cascade with missed-relay fix, waypoint route at an altitude lane z 4.8 with steep dips to R1 and R2 (route shape chosen among 3 in the code world), plus reactive avoidance: constant-velocity closest-approach repulsion on the observed movers with an upward bias; 12 gains by cross-entropy search over imagined lives with randomised aircraft, wind and mover populations",
    synthesis=dict(final_run="tune_C5", wm_steps_final=tj["wm_steps"], wall_s_final=round(tj["wall_s"], 1),
                   wm_steps_all_mission_C=int(sum(r["wm_steps"] for r in runs.values())),
                   wall_s_all_mission_C=round(sum(r["wall_s"] for r in runs.values()), 1), runs=runs, controller=tj["controller"]),
    code_world_prediction=dict(success=f["success"], t_success_s=f["t_success"], imagined_lives=4096,
                               end_causes={k: f[k] for k in ("success", "collision", "oob", "timeout")}),
    real_eval=dict(episodes=len(rows), seeds=[rows[0]["seed"], rows[-1]["seed"]],
                   success_rate=sum(bool(r["success"]) for r in rows) / len(rows),
                   mean_time_success_s=float(np.mean(ts)), end_causes=ends,
                   mean_relays_reached=float(np.mean([r["relays_reached"] for r in rows])),
                   failures="seed 1059 (t 7.7 s, 1 relay) and 1088 (t 5.7 s, 2 relays): both mover contacts at x ~30 just after the relay-2 dip, the place where the code world also concentrates its collisions"),
    gif_first_eval="runs/drone/code_wm/gifs/eval_code_wm_C_seed1000.gif",
    record="runs/drone/results/site_C__code_wm.json", site_world_py_bytes=os.path.getsize("worlds/drone/site_world.py"))
json.dump(s, open(O + "summary.json", "w"), indent=1, default=float)
print(json.dumps({k: s["phase3_mission_C"][k] for k in ("real_steps_total", "code_world_prediction", "real_eval")}, indent=1, default=float))
