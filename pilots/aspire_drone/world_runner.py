"""Run my programs inside the code world (experiments.drone.code_wm.screen setup, unchanged world and randomisation),
with per-episode failure details: end cause, time, relays, final position, distance to nearest wall / aircraft and
that aircraft's kind (0 shuttle_x, 1 shuttle_y, 2 wanderer, 3 guard) and whether a guard was engaged.
   python pilots/aspire_drone/world_runner.py 'v14;v15@{\"look_t\": 1.0}' 400 7000 [out.json]      (batches of 50 in parallel workers)"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import sys
import time

import numpy as np


def run_batch(args):
    ver, n, seed0 = args
    import torch
    torch.set_num_threads(1)
    from experiments.drone.code_wm.imagine import load_world, randomise
    from experiments.drone.code_wm.screen import SETUP, CAUSE
    from worlds.drone import site_world as W
    st = SETUP["C"]
    cfg = json.load(open(st["cfg"]))
    base = dict(load_world(st["world"])); base.update(cfg["world_override"])
    unc = {k: (v[0], tuple(v[1]) if isinstance(v[1], list) else v[1]) for k, v in cfg["unc"].items()}
    B = n
    g = torch.Generator().manual_seed(seed0)
    pw = randomise(base, B, "cpu", g, unc)
    start, relays = W.sample_start(B, "cpu", g, st["mission"])
    s = W.init_state(B, relays, start, "cpu", pw)
    s.update(W.sample_movers(B, "cpu", g, cfg.get("movers", {})))
    mod, _, kwj = ver.partition("@")
    make = getattr(importlib.import_module(f"pilots.aspire_drone.policy_{mod}"), "make_policy")
    kw = json.loads(kwj) if kwj else {}
    pols = [make(**kw) for _ in range(B)]
    for p in pols:
        p.reset()
    ranges = np.full((6, 16), 12.0)
    n_steps = int(round(st["time_limit"] * 30))
    steps_used = 0
    min_md = np.full(B, 99.0)
    trace = [[] for _ in range(B)]
    for _ in range(n_steps):
        o = W.observe(s, pw, g)
        on = {k: v.double().cpu().numpy() for k, v in o.items() if torch.is_tensor(v)}
        done = s["done"].cpu().numpy()
        acts = np.zeros((B, 4))
        for i in range(B):
            if done[i]:
                acts[i] = (0, 0, 0, 0.3)
                continue
            steps_used += 1
            k = int(on["relay_index"][i])
            ob = dict(pos=on["pos"][i], lin_vel=on["lin_vel"][i], quat=on["quat"][i], ang_vel=on["ang_vel"][i],
                      t=np.array([on["t"][i]]), relay_index=np.array([float(k)]), relays=on["relays"][i],
                      goal=on["relays"][i][k], ranges=ranges.copy(), movers=on["movers"][i],
                      n_movers=np.array([float(on["n_movers"][i])]))
            a = np.asarray(pols[i].act(ob), dtype=np.float64).reshape(4)
            acts[i] = a if np.all(np.isfinite(a)) else (0, 0, 0, 0)
            dbg = getattr(pols[i], "dbg", {})
            near = on["movers"][i][0] if on["n_movers"][i] > 0 else np.zeros(7)
            trace[i].append([round(float(on["t"][i]), 2)] + [round(float(x), 2) for x in on["pos"][i]] +
                            [round(float(x), 2) for x in on["lin_vel"][i]] + [round(float(x), 2) for x in near[:6]] +
                            [round(float(x), 1) for x in np.r_[dbg.get("v_av", np.zeros(3)), dbg.get("a_av", np.zeros(3)), dbg.get("a", np.zeros(3))]])
        s, _, d = W.step(s, torch.as_tensor(acts, dtype=torch.float32), pw, st["time_limit"], g)
        md = ((s["mp"] - s["pos"][:, None]).norm(dim=-1) + 1e3 * (~s["m_active"]).float()).amin(-1).numpy()
        live = ~done
        min_md[live] = np.minimum(min_md[live], md[live])
        if bool(d.all()):
            break
    c, t, kk = s["cause"].numpy(), s["t"].numpy(), s["k"].numpy()
    P = s["pos"].numpy()
    wall = W.sdf_boxes(s["pos"]).numpy()
    dm = (s["mp"] - s["pos"][:, None]).norm(dim=-1) + 1e3 * (~s["m_active"]).float()
    j = dm.argmin(-1)
    ar = np.arange(B)
    kind = s["m_kind"].numpy()[ar, j.numpy()]
    eng = s["m_eng"].numpy()[ar, j.numpy()]
    rows = []
    for i in range(B):
        rows.append(dict(ver=ver, seed0=seed0, ep=i, success=bool(c[i] == 1), cause=CAUSE[int(c[i])], time_s=float(t[i]),
                         relays=int(min(kk[i], 3)), pos=[round(float(x), 2) for x in P[i]], wall=round(float(wall[i]), 2),
                         mover_d=round(float(dm[i, j[i]]), 2), mover_kind=int(kind[i]), guard_engaged=bool(eng[i]),
                         min_mover_d=round(float(min_md[i]), 2), trace=trace[i][-90:] if c[i] != 1 else None))
    return rows, steps_used


def summarise(rows):
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    out = dict(n=n, success=len(ok) / n, time_success_s=float(np.mean([r["time_s"] for r in ok])) if ok else None)
    for k in ("collision", "out_of_bounds", "timeout"):
        out[k] = sum(r["cause"] == k for r in rows) / n
    col = [r for r in rows if r["cause"] == "collision"]
    out["coll_wall"] = sum(r["wall"] < 0.1 or r["pos"][2] < 0.1 for r in col) / n
    out["coll_mover"] = sum(r["mover_d"] < 0.45 for r in col) / n
    out["coll_mover_by_kind"] = {str(k): sum(r["mover_d"] < 0.45 and r["mover_kind"] == k for r in col) for k in range(4)}
    out["coll_guard_engaged"] = sum(r["mover_d"] < 0.45 and r["guard_engaged"] for r in col)
    return out


if __name__ == "__main__":
    vers = sys.argv[1].split(";")
    n, seed0 = int(sys.argv[2]), int(sys.argv[3])
    outp = sys.argv[4] if len(sys.argv) > 4 else None
    bs = 50
    jobs = [(v, bs, seed0 + b) for v in vers for b in range(n // bs)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(14, len(jobs))) as pool:
        res = pool.map(run_batch, jobs, chunksize=1)
    wall = time.time() - t0
    allrows, steps = {}, 0
    for rows, su in res:
        steps += su
        for r in rows:
            allrows.setdefault(r["ver"], []).append(r)
    summ = {v: summarise(allrows[v]) for v in vers}
    for v in vers:
        print(v, json.dumps(summ[v]))
    print(json.dumps(dict(imagined_episodes=n * len(vers), imagined_steps=steps, wall_s=round(wall, 1))))
    if outp:
        json.dump(dict(summary=summ, rows=allrows, imagined_steps=steps, wall_s=wall), open(outp, "w"))
