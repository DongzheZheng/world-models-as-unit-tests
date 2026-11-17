"""Fit the code world's numeric parameters to the black-box lives.
usage: python experiments/doom/code_wm/fit.py [lives e.g. 1,2]  -> runs/doom/code_wm/fit.json (+ printout)
Measurements (all from gt_objects / player readings of the logs):
  player: per-tic model v+=A*dir (from tic CONTROL_START); y+=v; clamp; v*=F   -> least-squares A, F over steps
  balls : speed |v|, spawn point, aim (does the velocity point at the player's position at fire time?)
  imps  : spawn tics and y, firing events per imp (ball back-projected to its source), exposure -> hazard
  hits  : ball-player offsets at the death step; visibility rule |dy| < dx + r vs the 'visible' flag
"""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

lives = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else \
    sorted(int(p.name.split("_")[1].split(".")[0]) for p in Path("runs/doom/blackbox/take_cover/code_wm").glob("life_*.jsonl.gz"))
PX = -416.0


def load(k):
    recs = [json.loads(l) for l in gzip.open(f"runs/doom/blackbox/take_cover/code_wm/life_{k}.jsonl.gz", "rt")]
    steps = [dict(obs=recs[1]["reset"]["obs"], gt=recs[1]["reset"]["info"]["gt_objects"], action=None)]
    for r in recs[2:]:
        steps.append(dict(obs=r["obs"], gt=r["info"]["gt_objects"], action=r["action"], term=r["terminated"]))
    return recs[0]["meta"], steps


def sim_player(y, v, a_dir, tic0, A, F, ctrl=8, lo=16.0, hi=752.0, n=4):
    for t in range(tic0, tic0 + n):
        if t >= ctrl:
            v += A * a_dir
        y += v
        if y < lo or y > hi:
            y, v = min(max(y, lo), hi), 0.0
        v *= F
    return y, v


out = dict(lives=lives, per_life={})
trans, shots, exposure, spawns, ball_speed, aim_err, vis_ok, vis_n, start = [], [], 0.0, [], [], [], 0, 0, []
hits, deaths, intervals = [], [], []
for k in lives:
    meta, S = load(k)
    p0 = S[0]["obs"]["player"]
    start.append((p0["x"], p0["y"], p0["angle"]))
    T_end = S[-1]["obs"]["tic"]
    for i in range(1, len(S)):
        if S[i].get("term"):
            break
        a, b = S[i - 1]["obs"], S[i]["obs"]
        d = {0: 0, 1: 1, 2: -1}[S[i]["action"]]
        trans.append((a["player"]["y"], a["player"]["vy"], d, a["tic"], b["player"]["y"], b["player"]["vy"]))
    seen, imp_birth, imp_pos, last_shot = {}, {}, {}, {}
    ptrack = {s["obs"]["tic"]: s["obs"]["player"]["y"] for s in S}
    for s in S:
        tic = s["obs"]["tic"]
        py = s["obs"]["player"]["y"]
        for g in s["gt"]:
            if g["name"] == "DoomImp" and g["id"] not in imp_birth:
                imp_birth[g["id"]] = tic
                imp_pos[g["id"]] = (g["x"], g["y"])
            r = 20.0 if g["name"] == "DoomImp" else 6.0
            dx, dy = g["x"] - PX, g["y"] - py
            pred = dx > 0 and abs(dy) < dx + r
            vis_ok += int(pred == bool(g["visible"]))
            vis_n += 1
            if "Ball" in g["name"] and g["id"] not in seen and math.hypot(g["vx"], g["vy"]) > 1:
                seen[g["id"]] = (tic, g)
    for bid, (tic, g) in sorted(seen.items()):
        sp = math.hypot(g["vx"], g["vy"])
        ball_speed.append(sp)
        ktics = (g["x"] - 64.0) / -g["vx"]
        src_y = g["y"] - g["vy"] * ktics
        fire_tic = tic - ktics
        alive = [i for i in imp_pos if imp_birth[i] <= fire_tic + 2] or list(imp_pos)
        iid = min(alive, key=lambda i: abs(imp_pos[i][1] - src_y))
        tt = min(ptrack, key=lambda x: abs(x - fire_tic))
        aim_y = imp_pos[iid][1] + g["vy"] / g["vx"] * (PX - imp_pos[iid][0])
        aim_err.append(aim_y - ptrack[tt])
        shots.append(dict(life=k, imp=iid, fire_tic=round(fire_tic, 1), since_spawn=round(fire_tic - imp_birth[iid], 1),
                          src_y_err=round(src_y - imp_pos[iid][1], 1), first_seen_x=g["x"], ktics=round(ktics, 2)))
        if iid in last_shot:
            intervals.append(fire_tic - last_shot[iid])
        last_shot[iid] = fire_tic
    for iid, b in imp_birth.items():
        exposure += T_end - b
        spawns.append(dict(life=k, imp=iid, tic=b, y=imp_pos[iid][1], x=imp_pos[iid][0]))
    last = S[-2]
    pyl = S[-1]["obs"]["player"]["y"]
    for g in last["gt"]:
        if "Ball" in g["name"] and math.hypot(g["vx"], g["vy"]) > 1:
            for j in range(1, 5):
                bx, by = g["x"] + g["vx"] * j, g["y"] + g["vy"] * j
                if abs(bx - PX) < 40 and abs(by - pyl) < 60:
                    hits.append(dict(life=k, tics_after=j, dx=round(bx - PX, 1), dy=round(by - pyl, 1)))
    fp = S[-1]["obs"]["player"]
    deaths.append(dict(life=k, tic=T_end, damage=fp["damage_taken"], hits=fp["hits_taken"], health=fp["health"]))
    out["per_life"][k] = dict(seed=meta["seed"], steps=len(S) - 1, end_tic=T_end, n_imps=len(imp_birth), n_shots=len(seen))

trans = trans[:600]   # player fit on the first 600 transitions (enough; keeps the grid search fast)
best = None
for A in np.arange(0.70, 0.80, 0.0025):
    for F in np.arange(0.89, 0.92, 0.0005):
        err = 0.0
        for y, v, d, tic, y2, v2 in trans:
            yy, vv = sim_player(y, v, d, tic, A, F)
            err += (yy - y2) ** 2 + 10 * (vv - v2) ** 2
        if best is None or err < best[0]:
            best = (err, A, F)
err, A, F = best
res = [sim_player(y, v, d, tic, A, F)[0] - y2 for y, v, d, tic, y2, v2 in trans]
out["player"] = dict(A=round(A, 4), F=round(F, 4), y_rmse_1step=float(np.sqrt(np.mean(np.square(res)))), n=len(trans),
                     v_terminal=round(A * F / (1 - F), 3), start=start)
out["balls"] = dict(speed_mean=float(np.mean(ball_speed)), speed_std=float(np.std(ball_speed)),
                    aim_err_mean=float(np.mean(aim_err)), aim_err_std=float(np.std(aim_err)), n=len(ball_speed))
out["imps"] = dict(spawns=spawns, n_initial=[sum(1 for s in spawns if s["life"] == k and s["tic"] <= 1) for k in lives],
                   xs=sorted(set(s["x"] for s in spawns)), ys=sorted(s["y"] for s in spawns))
out["shots"] = shots
firsts = {}
for s in shots:
    key = (s["life"], s["imp"])
    firsts[key] = min(firsts.get(key, 1e9), s["since_spawn"])
out["hazard"] = dict(n_shots=len(shots), imp_tics=exposure, per_tic=len(shots) / max(exposure, 1),
                     same_imp_intervals=sorted(round(x, 1) for x in intervals),
                     first_shot_since_spawn=sorted(firsts.values()))
out["hits"], out["deaths"] = hits, deaths
out["visibility_rule_acc"] = vis_ok / max(vis_n, 1)
Path("runs/doom/code_wm").mkdir(parents=True, exist_ok=True)
Path("runs/doom/code_wm/fit.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("shots", "imps")}, indent=1))
print("imp spawns:", [(s["life"], s["tic"], round(s["y"])) for s in spawns])
print("shots:", [(s["life"], s["imp"], s["fire_tic"], s["since_spawn"], s["src_y_err"]) for s in shots])
