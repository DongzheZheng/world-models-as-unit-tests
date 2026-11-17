"""Model-based RL under 5 lives: life 1 = stabiliser-with-excitation exploration, lives 2-5 = MPPI on the current learned
ensemble, refit on all of the pilot's own black-box logs after every life.
--purpose oracle_train --pilot neural_wm_dry is a code dry run (keeps its logs in memory, consumes no lives)."""
import argparse, json, os, time
import numpy as np, torch
from reallab.drone.api import make
from experiments.drone.baselines import wm
from experiments.drone.baselines.explore import Explorer

p = argparse.ArgumentParser()
p.add_argument("--purpose", default="calibration")
p.add_argument("--pilot", default="neural_wm")
p.add_argument("--out", default="runs/drone/baselines_rl/neural_wm")
p.add_argument("--seed0", type=int, default=0)
p.add_argument("--threads", type=int, default=4)
a = p.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
logs, summary, t0, real_steps = [], [], time.time(), 0
model, planner = None, None

for life in range(1, 6):
    env = make("site", purpose=a.purpose, tag=f"neural_wm_life{life}", pilot=a.pilot, mission="A", command_mode="ctbr")
    obs, _ = env.reset(seed=a.seed0 + life - 1)
    pol = Explorer(seed=a.seed0 + life) if life == 1 else planner
    if life > 1:
        pol.reset()
    mem = {k: [obs[k]] for k in ("pos", "lin_vel", "quat", "ang_vel")}; acts = []
    tl = time.time()
    while True:
        act = pol.act(obs) if life == 1 else pol.plan(obs)
        obs, r, te, tr, info = env.step(act)
        acts.append(act)
        for k in mem: mem[k].append(obs[k])
        if te or tr:
            break
    env.close()
    real_steps += len(acts)
    summary.append(dict(life=life, policy="explorer" if life == 1 else "mppi", steps=len(acts), relays=int(info.get("relays_reached", 0)),
                        success=bool(info.get("success")), collision=bool(info.get("collision")), time_s=info.get("time_s"),
                        plan_s_per_step=(time.time() - tl) / len(acts)))
    print(summary[-1], flush=True)
    if a.purpose == "calibration":
        logs = [wm.load_npz(f"runs/drone/blackbox/{a.pilot}/life_{i}.npz") for i in range(1, life + 1)]   # ONLY the black-box logs
    else:
        logs.append({**{k: np.stack(v) for k, v in mem.items()}, "actions": np.stack(acts)})
    X, Y, L = wm.transitions(logs)
    model, fi = wm.fit(X, Y, seed=life)
    summary[-1]["fit"] = fi
    wm.save_model(model, f"{a.out}/model_after_life{life}.pt")
    if life < 5:
        planner = wm.Planner(model, seed=life)

wall = time.time() - t0
wm.save_model(model, f"{a.out}/model_final.pt")
json.dump(dict(lives=summary, real_steps=real_steps, train_wall_s=wall), open(f"{a.out}/lives.json", "w"), indent=1)
print("done", real_steps, wall)
