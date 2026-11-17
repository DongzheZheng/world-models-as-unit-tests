"""Fidelity of the code world on a life it was NOT fitted to.
usage: python experiments/doom/code_wm/fidelity.py <test_life> <fitted_on e.g. "1"> <policy: sticky|path.pt> [out_key]
Appends/updates runs/doom/code_wm/fidelity.json[out_key].
"""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from worlds.fps import take_cover_world as W

life, fitted_on, pol = int(sys.argv[1]), sys.argv[2], sys.argv[3]
key = sys.argv[4] if len(sys.argv) > 4 else f"life_{life}"
recs = [json.loads(l) for l in gzip.open(f"runs/doom/blackbox/take_cover/code_wm/life_{life}.jsonl.gz", "rt")]
S = [dict(obs=recs[1]["reset"]["obs"], gt=recs[1]["reset"]["info"]["gt_objects"], action=None)] + \
    [dict(obs=r["obs"], gt=r["info"]["gt_objects"], action=r["action"], term=r["terminated"]) for r in recs[2:]]
real_ret = sum(r["reward"] for r in recs[2:])
out = dict(test_life=life, fitted_on=fitted_on, policy=pol, real_return=real_ret, real_steps=len(S) - 1)
dev = "cuda"


def player_only_state(y, v, tic):
    s = W.init_state(1, dev)
    s.update(py=torch.tensor([y], device=dev), vy=torch.tensor([v], device=dev),
             tic=torch.tensor([tic], device=dev), imp_on=torch.zeros_like(s["imp_on"]),
             n_imp=torch.full((1,), W.N_IMP_MAX, device=dev), rate=torch.zeros(1, device=dev),
             wake_rate=torch.zeros(1, device=dev))
    return s


# 1) player motion, open loop over k steps with the logged actions
n_live = len(S) - 1 if not S[-1].get("term") else len(S) - 2   # last obs is post-death
errs = {k: [] for k in (1, 5, 10, 25)}
for i in range(0, min(n_live, 600)):   # first 600 steps suffice for motion error
    p = S[i]["obs"]["player"]
    s = player_only_state(p["y"], p["vy"], S[i]["obs"]["tic"])
    for k in range(1, 26):
        if i + k > n_live:
            break
        s, _, _ = W.step(s, torch.tensor([S[i + k]["action"]], device=dev))
        if k in errs:
            errs[k].append(abs(float(s["py"][0]) - S[i + k]["obs"]["player"]["y"]))
out["player_abs_err_y"] = {f"{k}_steps": dict(mean=float(np.mean(v)), max=float(np.max(v)), n=len(v))
                           for k, v in errs.items() if v}

# 2) fireball trajectories: straight line at BALL_SPEED from first sighting
first, traj = {}, {}
for s_ in S:
    for g in s_["gt"]:
        if "Ball" in g["name"] and math.hypot(g["vx"], g["vy"]) > 1:
            traj.setdefault(g["id"], []).append((s_["obs"]["tic"], g["x"], g["y"], g["vx"], g["vy"]))
berr = {4: [], 20: [], 40: []}
speed_err = []
for bid, tr in traj.items():
    t0, x0, y0, vx, vy = tr[0]
    sp = math.hypot(vx, vy)
    speed_err.append(abs(sp - W.BALL_SPEED))
    for t, x, y, *_ in tr[1:]:
        dt = t - t0
        e = math.hypot(x0 + vx / sp * W.BALL_SPEED * dt - x, y0 + vy / sp * W.BALL_SPEED * dt - y)
        for k in berr:
            if dt == k:
                berr[k].append(e)
out["ball_pos_err"] = {f"{k}_tics": dict(mean=float(np.mean(v)) if v else None, n=len(v)) for k, v in berr.items()}
out["ball_speed_abs_err_max"] = float(max(speed_err)) if speed_err else None

# 3) visibility rule
ok = n = 0
for s_ in S:
    py = s_["obs"]["player"]["y"]
    for g in s_["gt"]:
        r = W.IMP_R if g["name"] == "DoomImp" else W.BALL_R
        pred = bool(W.visible(torch.tensor(g["x"] - W.PX), torch.tensor(g["y"] - py), r))
        ok += int(pred == bool(g["visible"]))
        n += 1
out["visibility_rule_acc"] = ok / max(n, 1)

# 4) shots vs the world's hazard, given this life's imp exposure
births, end = {}, S[-1]["obs"]["tic"]
for s_ in S:
    for g in s_["gt"]:
        if g["name"] == "DoomImp":
            births.setdefault(g["id"], s_["obs"]["tic"])
expo = sum(max(0, end - b - W.WAKE_DELAY) for b in births.values())
out["shots"] = dict(observed=len(traj), world_expected=W.FIRE_RATE * expo,
                    world_expected_range=[W.FIRE_RATE * W.FIRE_RATE_SCALE[0] * expo, W.FIRE_RATE * W.FIRE_RATE_SCALE[1] * expo])
sp_t = sorted(set(b for b in births.values() if b > 1))
attempts = [t for t in range(W.SPAWN_FIRST, end + 1, W.SPAWN_PERIOD)]
out["imp_spawns"] = dict(observed_tics=sp_t, on_world_schedule=all((t - W.SPAWN_FIRST) % W.SPAWN_PERIOD == 0 for t in sp_t),
                         attempts=len(attempts), successes=len(sp_t), n_initial=sum(1 for b in births.values() if b <= 1))

# 5) survival of the flown policy as predicted by the code world
N = 1024
g = torch.Generator(device=dev).manual_seed(7)
s = W.init_state(N, dev, g)
if pol == "sticky":   # sticky random: hold a uniform action for U{2..12} steps (as policies.sticky_random)
    a = torch.zeros(N, dtype=torch.long, device=dev)
    left = torch.zeros(N, dtype=torch.long, device=dev)
    def act(o):
        global a, left
        new = left <= 0
        a = torch.where(new, torch.randint(3, (N,), device=dev, generator=g), a)
        left = torch.where(new, torch.randint(2, 13, (N,), device=dev, generator=g), left) - 1
        return a
elif pol.startswith("planner"):   # planner:h,m,w
    from experiments.doom.code_wm.planner import plan_act
    h, m, w = pol.split(":")[1].split(",")
    act = lambda o: plan_act(o, int(h), float(m), float(w))
else:
    from cawm.ppo import ActorCritic
    pi = ActorCritic.load(torch.load(pol, map_location=dev), dev)
    act = lambda o: pi.act(o, deterministic=True)
ret = torch.zeros(N, device=dev)
with torch.no_grad():
    for t in range(2500):
        s, r, d = W.step(s, act(W.observe(s)), g)
        ret += r
        if not s["alive"].any():
            break
ret = ret.cpu().numpy()
out["survival"] = dict(real=real_ret, world_mean=float(ret.mean()), world_median=float(np.median(ret)),
                       world_p10=float(np.percentile(ret, 10)), world_p90=float(np.percentile(ret, 90)),
                       real_percentile_in_world=float((ret < real_ret).mean() * 100))
p = Path("runs/doom/code_wm/fidelity.json")
allf = json.loads(p.read_text()) if p.exists() else {}
allf[key] = out
p.write_text(json.dumps(allf, indent=1))
print(json.dumps(out, indent=1))
