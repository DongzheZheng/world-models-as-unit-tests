"""Train PPO in the real env. Phases: `five` (exactly 5 lives, sequential) and `main` (vector PPO, checkpoints by lives)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cawm.ppo import ActorCritic, PPOConfig, train_ppo
from experiments.doom.rl_baseline.ppo_real import N_ACT, OBS_DIM, RealVecEnv

OUT = Path("runs/doom/rl_baseline")
CK = OUT / "ckpt"
GAMMA, LAM = 0.99, 0.95


def save(policy, name, **meta):
    CK.mkdir(parents=True, exist_ok=True)
    torch.save(dict(policy=policy.export(), meta=meta), CK / f"{name}.pt")
    with open(OUT / "ckpt_index.jsonl", "a") as f:
        f.write(json.dumps(dict(name=name, **meta)) + "\n")


# --------------------------------------------------------------------------------- 5 lives
def gae(rew, val):
    """one finished episode: terminal at the end, no bootstrap"""
    T = len(rew)
    adv = np.zeros(T, np.float32)
    g = 0.0
    for t in reversed(range(T)):
        nv = 0.0 if t == T - 1 else val[t + 1]
        d = rew[t] + GAMMA * nv - val[t]
        g = d + GAMMA * LAM * g * (1.0 if t < T - 1 else 0.0)
        adv[t] = g
    return adv


def ppo_update(policy, opt, obs, act, adv, ret, logp_old, epochs, mbs, clip=0.2, ent=0.01):
    n = len(obs)
    mb = max(1, n // mbs)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            if len(idx) < 2:
                continue
            d = policy.dist(obs[idx])
            lp = policy.logp(d, act[idx])
            ratio = (lp - logp_old[idx]).exp()
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            vl = 0.5 * (policy.value(obs[idx]) - ret[idx]).pow(2).mean()
            loss = pg + 0.5 * vl - ent * policy.entropy(d).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()


def five(args):
    """Sequential PPO with exactly 5 real lives. Variant 'reuse': after the 5 lives, further PPO iterations on the
    pooled 5-episode buffer (no new data). Variant 'noreuse': per-episode updates only."""
    torch.manual_seed(args.seed)
    env = RealVecEnv(1, device="cpu", tag="ppo_five", auto_reset=False)
    policy = ActorCritic(OBS_DIM, None, N_ACT, (128, 128), True)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    eps, t0, steps = [], time.time(), 0
    for life in range(5):
        obs = env.reset()
        O, A, R = [], [], []
        while True:
            with torch.no_grad():
                policy.norm.update(obs)
                a = policy.dist(obs).sample()
            O.append(obs[0]); A.append(int(a[0]))
            obs, r, done, _ = env.step(a)
            R.append(float(r[0])); steps += 1
            if bool(done[0]):
                break
        eps.append((torch.stack(O), torch.tensor(A), np.array(R, np.float32)))
        o, a_, r_ = eps[-1]
        with torch.no_grad():
            v = policy.value(o).numpy()
            lp = policy.logp(policy.dist(o), a_)
        adv = torch.as_tensor(gae(r_, v))
        ppo_update(policy, opt, o, a_, adv, adv + torch.as_tensor(v), lp, epochs=10, mbs=2)
        print(f"life {life + 1} len {len(r_)} steps (return {len(r_) * 4} tics)", flush=True)
    wall = time.time() - t0
    assert env.lives == 5
    save(policy, "ppo_real_5lives_noreuse", lives=5, steps=steps, wall_s=wall)
    for it in range(args.reuse_iters):
        O = torch.cat([e[0] for e in eps]); A = torch.cat([e[1] for e in eps])
        with torch.no_grad():
            V = [policy.value(e[0]).numpy() for e in eps]
            LP = torch.cat([policy.logp(policy.dist(e[0]), e[1]) for e in eps])
        ADV = torch.as_tensor(np.concatenate([gae(e[2], v) for e, v in zip(eps, V)]))
        RET = ADV + torch.as_tensor(np.concatenate(V))
        ppo_update(policy, opt, O, A, ADV, RET, LP, epochs=4, mbs=2)
    save(policy, "ppo_real_5lives", lives=5, steps=steps, wall_s=time.time() - t0)
    env.close()


# --------------------------------------------------------------------------------- main
def main(args):
    targets = [int(x) for x in args.targets.split(",")]
    cfg = PPOConfig(total_steps=int(args.steps), rollout_len=args.T, epochs=4, minibatches=4, gamma=GAMMA, lam=LAM,
                    lr=args.lr, ent_coef=args.ent, hidden=(128, 128), log_every=1, seed=args.seed)
    env = RealVecEnv(args.envs, device="cuda", tag="ppo_main")
    t0 = time.time()
    nxt = [0]
    curve = []

    def cb(policy, rec):
        wall = time.time() - t0
        while nxt[0] < len(targets) and env.lives >= targets[nxt[0]]:
            save(policy, f"ppo_real_{targets[nxt[0]]}lives", lives=env.lives, target=targets[nxt[0]],
                 steps=env.steps, wall_s=wall, ep_ret_train=rec["ep_ret"], ep_len_train=rec["ep_len"])
            print(f"CKPT target {targets[nxt[0]]} lives={env.lives} steps={env.steps} wall={wall:.0f}", flush=True)
            nxt[0] += 1
        if wall > args.wall:
            print("wall-clock limit", flush=True)
            return False

    cnt = [0]

    def lg(m):
        cnt[0] += 1
        if cnt[0] % 50 == 1:
            print(m, f"lives {env.lives}", flush=True)

    policy, hist = train_ppo(env, cfg, log=lg, callback=cb)
    wall = time.time() - t0
    save(policy, f"ppo_real_{env.lives}lives" if args.name == "main" else f"ppo_real_final_{args.name}", lives=env.lives, steps=env.steps, wall_s=wall)
    json.dump(hist, open(OUT / f"train_history_{args.name}.json", "w"))
    print(f"DONE lives={env.lives} steps={env.steps} wall={wall:.0f} sps={env.steps / wall:.0f}", flush=True)
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--steps", type=float, default=5e6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--wall", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse_iters", type=int, default=20)
    ap.add_argument("--name", default="main")
    ap.add_argument("--targets", default="20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    five(a) if a.phase == "five" else main(a)
