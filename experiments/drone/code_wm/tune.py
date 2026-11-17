"""Controller synthesis inside the code world: cross-entropy search over the controller's gains, each candidate scored
on M imagined lives with randomised world parameters (domain randomisation over what the logs leave uncertain).

  python experiments/drone/code_wm/tune.py <cfg.json>
cfg: {"world": fitted json, "unc": {...}, "N": 48, "M": 32, "rounds": 3, "out": path, "mission": [[x,y,z],...],
      "time_limit": 17.0, "final_M": 1024}
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from experiments.drone.code_wm.controller import DEFAULT
from experiments.drone.code_wm.imagine import compensation, load_world, randomise, rollout, summarise

SPACE = dict(speed=(3.5, 8.0), kp=(1.0, 8.0), kv=(1.5, 7.0), ki=(0.0, 3.0), k_att=(4.0, 14.0), max_tilt=(0.6, 1.15),
             corner_speed=(1.5, 6.0), corner_dist=(1.5, 6.0), av_k=(0.0, 30.0), av_R=(0.8, 4.0), av_T=(0.3, 2.5),
             av_up=(0.0, 1.5))


def main():
    cfg = json.load(open(sys.argv[1]))
    dev = "cuda"
    belief = load_world(cfg["world"])
    belief.update(cfg.get("belief_override", {}))
    unc = {k: (v[0], tuple(v[1]) if isinstance(v[1], list) else v[1]) for k, v in cfg["unc"].items()}
    world_nom = dict(belief)
    world_nom.update(cfg.get("world_override", {}))
    mission = tuple(tuple(r) for r in cfg.get("mission", [[14, 0, 2.5], [28, 0, 2.5], [57, 0, 2.5]]))
    tl = cfg.get("time_limit", 17.0)
    N, M, rounds = cfg.get("N", 48), cfg.get("M", 32), cfg.get("rounds", 3)
    space = dict(SPACE); space.update({k: tuple(v) for k, v in cfg.get("space", {}).items()})
    keys = [k for k in space if k in cfg.get("search", list(space))]
    lo = np.array([space[k][0] for k in keys]); hi = np.array([space[k][1] for k in keys])
    mu = np.array([cfg.get("init", {}).get(k, DEFAULT[k]) for k in keys], dtype=float)
    sd = (hi - lo) / 4
    rng = np.random.default_rng(0)
    comp = compensation(belief)
    t0 = time.time()
    wm_steps = 0
    best = None
    hist = []
    for rd in range(rounds):
        cand = np.clip(mu + sd * rng.standard_normal((N, len(keys))), lo, hi)
        cand[0] = mu
        B = N * M
        g = torch.Generator(device=dev).manual_seed(1000 + rd)
        prm_w = randomise(world_nom, B, dev, g, unc)
        prm_c = dict(DEFAULT); prm_c.update(comp); prm_c["wind_ff"] = 1.0; prm_c.update(cfg.get("fixed", {}))
        for j, k in enumerate(keys):
            prm_c[k] = torch.tensor(cand[:, j], dtype=torch.float32, device=dev).repeat_interleave(M)[:, None]
        r = rollout(prm_c, prm_w, belief, B=B, device=dev, seed=rd, mission=mission, time_limit=tl, route=cfg.get("route"), r_adv=cfg.get("r_adv", 0.8), movers=cfg.get("movers"))
        wm_steps += r["wm_steps"]
        c = r["cause"].reshape(N, M); t = r["t"].reshape(N, M)
        succ = (c == 1).mean(1)
        tm = np.where(c == 1, t, tl).mean(1)
        score = succ - cfg.get("time_weight", 0.01) * tm
        order = np.argsort(-score)
        elite = cand[order[: max(4, N // 6)]]
        mu, sd = elite.mean(0), np.maximum(elite.std(0), (hi - lo) * 0.03)
        hist.append(dict(round=rd, best_succ=float(succ[order[0]]), best=dict(zip(keys, cand[order[0]].round(3).tolist())),
                         mean_succ=float(succ.mean()), elapsed=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
    # final: evaluate the elite mean and the best candidate of the last round on fresh imagined lives
    finals = {}
    for name, vec in (("mean", mu), ("best", cand[order[0]])):
        FM = cfg.get("final_M", 1024)
        g = torch.Generator(device=dev).manual_seed(777)
        prm_w = randomise(world_nom, FM, dev, g, unc)
        prm_c = dict(DEFAULT); prm_c.update(comp); prm_c["wind_ff"] = 1.0; prm_c.update(cfg.get("fixed", {}))
        prm_c.update({k: float(v) for k, v in zip(keys, vec)})
        r = rollout(prm_c, prm_w, belief, B=FM, device=dev, seed=777, mission=mission, time_limit=tl, route=cfg.get("route"), r_adv=cfg.get("r_adv", 0.8), movers=cfg.get("movers"))
        wm_steps += r["wm_steps"]
        finals[name] = dict(params={k: float(v) for k, v in zip(keys, vec)}, **summarise(r))
        print(name, json.dumps(finals[name]), flush=True)
    pick = max(finals, key=lambda n: (finals[n]["success"], -(finals[n]["t_success"] or 99)))
    out = dict(cfg=cfg, history=hist, finals=finals, pick=pick,
               controller=dict(DEFAULT, **comp, wind_ff=1.0, **cfg.get("fixed", {}), **finals[pick]["params"]),
               wm_steps=int(wm_steps), wall_s=time.time() - t0)
    out["wm_steps_per_s"] = out["wm_steps"] / out["wall_s"]
    json.dump(out, open(cfg["out"], "w"), indent=1)
    print("picked", pick, json.dumps(out["controller"]), "wm_steps", wm_steps, "wall", round(out["wall_s"]))


if __name__ == "__main__":
    main()
