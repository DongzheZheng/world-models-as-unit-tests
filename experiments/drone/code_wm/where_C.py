"""Where/how does a mission-C controller fail in the code world?  (collision location, mover kind hit, relay stage)"""
import json, sys
import numpy as np
import torch
from experiments.drone.code_wm.imagine import load_world, randomise
from experiments.drone.code_wm.controller import control, route_nodes, advance
from worlds.drone import site_world as W

cfg = json.load(open(sys.argv[1])); tj = json.load(open(sys.argv[2]))
belief = load_world(cfg["world"]); base = dict(belief); base.update(cfg["world_override"])
unc = {k: (v[0], tuple(v[1]) if isinstance(v[1], list) else v[1]) for k, v in cfg["unc"].items()}
B, dev = 2048, "cuda"
g = torch.Generator(device=dev).manual_seed(99)
pw = randomise(base, B, dev, g, unc)
start, relays = W.sample_start(B, dev, g)
s = W.init_state(B, relays, start, dev, pw); s.update(W.sample_movers(B, dev, g, cfg["movers"]))
nodes, rid = route_nodes(start, relays, cfg["route"], return_relay_ids=True)
kr = torch.zeros(B, dtype=torch.long, device=dev); integ = torch.zeros(B, 3, device=dev)
pc = dict(tj["controller"])
hitk = torch.full((B,), -1, device=dev, dtype=torch.long)
for i in range(int(round(cfg["time_limit"] * 30))):
    o = W.observe(s, pw, g)
    kr = advance(o["pos"], nodes, kr, 0.8, rid, o["relay_index"])
    a, integ = control(pc, o["pos"], o["lin_vel"], o["quat"], nodes, kr, integ, wind_est=W.wind_field(o["pos"], belief), movers=o["movers"])
    prev_done = s["done"].clone()
    s, r, done = W.step(s, a, pw, cfg["time_limit"], g)
    new = done & ~prev_done & (s["cause"] == 2)
    if new.any():
        d = (s["mp"] - s["pos"][:, None]).norm(dim=-1) + (~s["m_active"]) * 99
        mk = torch.gather(s["m_kind"], 1, d.argmin(-1, keepdim=True))[:, 0]
        near = d.min(-1).values < W.MOVER_R + W.DRONE_R + 0.05
        hitk = torch.where(new, torch.where(near, mk, torch.full_like(mk, 9)), hitk)
    if done.all(): break
c = s["cause"].cpu().numpy(); p = s["pos"].cpu().numpy(); hk = hitk.cpu().numpy(); k = s["k"].cpu().numpy()
print("success", (c == 1).mean(), "coll", (c == 2).mean())
col = c == 2
print("collisions by kind (0 sx,1 sy,2 wand,3 guard,9 wall/ground):", {int(u): int((hk[col] == u).sum()) for u in np.unique(hk[col])})
print("collision relay stage:", np.bincount(k[col], minlength=3).tolist())
print("collision x hist (5 m bins from 0):", np.histogram(p[col, 0], bins=np.arange(0, 65, 5))[0].tolist())
print("collision z hist (1 m bins):", np.histogram(p[col, 2], bins=np.arange(0, 7, 1))[0].tolist())
