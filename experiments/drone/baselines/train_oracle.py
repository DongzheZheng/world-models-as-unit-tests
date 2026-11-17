"""Unlimited-reference PPO trained directly in the (metered) real site, purpose=oracle_train."""
import argparse, json, os, time
import torch
from cawm.ppo import PPOConfig, train_ppo
from experiments.drone.baselines.common import RealVecEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_envs", type=int, default=20)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--steps", type=float, default=30e6)
    p.add_argument("--wall_s", type=float, default=2400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--log_std", type=float, default=-1.0)
    p.add_argument("--out", default="runs/drone/baselines_rl/oracle")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    THRESH = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    manifest, nxt = [], [0]
    t0 = time.time()
    env = RealVecEnv(a.n_envs, device="cuda", seed=a.seed)
    cfg = PPOConfig(total_steps=int(a.steps), rollout_len=a.rollout, epochs=4, minibatches=8, gamma=a.gamma, lam=0.95,
                    lr=a.lr, ent_coef=a.ent, init_log_std=a.log_std, log_every=20, seed=a.seed)
    last = dict(ep=0, ok=0, rec=0)

    def cb(policy, rec):
        now = time.time() - t0
        # training-time success rate of the stochastic policy since the previous checkpoint
        while nxt[0] < len(THRESH) and env.episodes >= THRESH[nxt[0]]:
            nxt[0] += 1
            fresh = env.recent[last["rec"]:]
            ok = sum(1 for r in fresh if r[0])
            ck = f"{a.out}/ckpt_ep{env.episodes}.pt"
            torch.save(policy.export(), ck)
            manifest.append(dict(ckpt=ck, real_episodes=env.episodes, real_steps=env.steps, train_wall_s=round(now, 1),
                                 train_success_recent=ok / max(1, len(fresh)), n_recent=len(fresh)))
            last["rec"] = len(env.recent)
            json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
            print(f"[ckpt] episodes {env.episodes} steps {env.steps} wall {now:.0f}s train-succ {ok}/{len(fresh)}", flush=True)
        if rec["iter"] % 20 == 0:
            fresh = env.recent[-300:]
            print(f"   eps {env.episodes} succ_train_recent {sum(r[0] for r in fresh)}/{len(fresh)} "
                  f"relays {sum(r[2] for r in fresh)/max(1,len(fresh)):.2f}", flush=True)
        return now < a.wall_s

    policy, hist = train_ppo(env, cfg, callback=cb)
    now = time.time() - t0
    torch.save(policy.export(), f"{a.out}/final.pt")
    manifest.append(dict(ckpt=f"{a.out}/final.pt", real_episodes=env.episodes, real_steps=env.steps, train_wall_s=round(now, 1),
                         train_success_recent=sum(r[0] for r in env.recent[-300:]) / max(1, len(env.recent[-300:])), n_recent=300, final=True))
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    json.dump(hist, open(f"{a.out}/history.json", "w"))
    print("done", env.episodes, env.steps, now, flush=True)
    env.close()


if __name__ == '__main__':
    main()
