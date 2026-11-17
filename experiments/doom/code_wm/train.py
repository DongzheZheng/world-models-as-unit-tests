"""Train a controller inside the code world with PPO; evaluate candidates in the code world only.
usage: python experiments/doom/code_wm/train.py <tag> <total_steps> [num_envs] [seed]
writes runs/doom/code_wm/policy_<tag>.pt and runs/doom/code_wm/train_<tag>.json
"""
import json
import sys
import time
from pathlib import Path

import torch

from cawm.ppo import PPOConfig, train_ppo
from worlds.fps import take_cover_world as W

OUT = Path("runs/doom/code_wm")


@torch.no_grad()
def world_eval(policy_fn, n=4096, max_steps=2500, seed=123, dev="cuda"):
    """Mean/sem/median of tics survived in the code world (return = tics alive, like the lab's)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    s = W.init_state(n, dev, g)
    ret = torch.zeros(n, device=dev)
    for t in range(max_steps):
        a = policy_fn(W.observe(s), s, g)
        s, r, done = W.step(s, a, g)
        ret += r
        if not s["alive"].any():
            break
    ret = ret + 0.0  # lab return ~ tics alive - 1 (start tic) ; keep as tics survived
    return dict(mean=float(ret.mean()), sem=float(ret.std() / n ** 0.5), median=float(ret.median()),
                frac_capped=float(s["alive"].float().mean()), n=n, max_tics=max_steps * 4)


def ref_policies():
    def noop(o, s, g):
        return torch.zeros(o.shape[0], dtype=torch.long, device=o.device)

    def rnd(o, s, g):
        return torch.randint(3, (o.shape[0],), device=o.device, generator=g)
    return dict(noop=noop, random=rnd)


if __name__ == "__main__":
    tag, total = sys.argv[1], float(sys.argv[2])
    ne = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    OUT.mkdir(parents=True, exist_ok=True)
    env = W.TakeCoverEnv(ne, "cuda", max_steps=1500, seed=seed)
    cfg = PPOConfig(total_steps=int(total), rollout_len=64, epochs=4, minibatches=8, gamma=0.995, lam=0.95,
                    lr=3e-4, ent_coef=0.005, hidden=(256, 256), seed=seed, log_every=20)
    t0 = time.time()
    pi, hist = train_ppo(env, cfg)
    wall = time.time() - t0
    torch.save(pi.export(), OUT / f"policy_{tag}.pt")
    res = dict(tag=tag, wm_steps=hist[-1]["steps"], train_wall_s=wall, sps=hist[-1]["sps"], history=hist,
               cfg=dict(total=total, num_envs=ne, seed=seed, gamma=cfg.gamma, ent=cfg.ent_coef))
    res["world_eval"] = {"ppo_det": world_eval(lambda o, s, g: pi.act(o, deterministic=True)),
                         "ppo_stoch": world_eval(lambda o, s, g: pi.act(o, deterministic=False))}
    for k, f in ref_policies().items():
        res["world_eval"][k] = world_eval(f)
    (OUT / f"train_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["world_eval"], indent=1), "wall", round(wall), "steps", res["wm_steps"])
