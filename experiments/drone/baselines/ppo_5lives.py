"""Model-free PPO that gets exactly 5 real episodes (lives) on mission A.

After every life the policy is updated with PPO (clipped objective, GAE recomputed with the current critic) using all
real transitions gathered so far.  No hyper-parameter is selected from outcomes of the lives or of the eval seeds.
--purpose oracle_train is only a code dry-run switch (does not consume the pilot's lives)."""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from cawm.ppo import ActorCritic
from reallab.drone.api import make
from experiments.drone.baselines.common import encode, action_to_cmd, OBS_DIM, ACT_DIM, REW_SCALE

p = argparse.ArgumentParser()
p.add_argument("--purpose", default="calibration")
p.add_argument("--out", default="runs/drone/baselines_rl/ppo5")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--log_std", type=float, default=-1.0)
p.add_argument("--gamma", type=float, default=0.995)
p.add_argument("--seed0", type=int, default=0)
a = p.parse_args()
torch.manual_seed(0); np.random.seed(0)
os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(2)
net = ActorCritic(OBS_DIM, ACT_DIM, None, (256, 256), True, a.log_std)
opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
GAMMA, LAM, CLIP = a.gamma, 0.95, 0.2
eps_data = []                  # per life: dict(obs (T+1,D), act (T,A), logp (T,), rew (T,), term bool, timeout bool)
t0 = time.time(); real_steps = 0; summary = []

def update():
    # concatenated data; GAE per episode with the current critic
    O, A, LP, ADV, RET = [], [], [], [], []
    with torch.no_grad():
        for e in eps_data:
            o = torch.as_tensor(e["obs"]); v = net.value(o).numpy()
            r = e["rew"]; T = len(r)
            nv = 0.0 if (e["term"]) else float(v[T])        # terminal (crash/success): no bootstrap
            adv = np.zeros(T); g = 0.0
            for t in reversed(range(T)):
                nxt = nv if t == T - 1 else v[t + 1]
                d = r[t] + GAMMA * nxt - v[t]
                g = d + GAMMA * LAM * g
                adv[t] = g
            O.append(e["obs"][:T]); A.append(e["act"]); LP.append(e["logp"]); ADV.append(adv); RET.append(adv + v[:T])
    O = torch.as_tensor(np.concatenate(O)); A = torch.as_tensor(np.concatenate(A), dtype=torch.float32)
    LP = torch.as_tensor(np.concatenate(LP), dtype=torch.float32); ADV = torch.as_tensor(np.concatenate(ADV), dtype=torch.float32)
    RET = torch.as_tensor(np.concatenate(RET), dtype=torch.float32)
    net.norm.update(O)
    n = len(O)
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        for idx in torch.tensor_split(perm, 4):
            d = net.dist(O[idx]); lp = net.logp(d, A[idx]); ratio = (lp - LP[idx]).exp()
            ad = ADV[idx]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
            pg = -torch.min(ratio * ad, ratio.clamp(1 - CLIP, 1 + CLIP) * ad).mean()
            vl = 0.5 * (net.value(O[idx]) - RET[idx]).pow(2).mean()
            loss = pg + 0.5 * vl
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    return n

for life in range(5):
    env = make("site", purpose=a.purpose, tag=f"ppo_real_5lives_life{life + 1}", pilot="ppo_real", mission="A", command_mode="ctbr")
    obs, info = env.reset(seed=a.seed0 + life)
    prev = np.zeros(4); E = [encode(obs, prev)]; ACT, LP, RW = [], [], []
    while True:
        with torch.no_grad():
            x = torch.as_tensor(E[-1])[None]
            d = net.dist(x); act = d.sample(); lp = net.logp(d, act)
        act = act[0].numpy(); ACT.append(act); LP.append(float(lp))
        obs, r, te, tr, info = env.step(action_to_cmd(act))
        prev = np.clip(act, -1, 1); E.append(encode(obs, prev)); RW.append(r * REW_SCALE)
        if te or tr:
            break
    env.close()
    real_steps += len(RW)
    eps_data.append(dict(obs=np.stack(E), act=np.stack(ACT), logp=np.array(LP), rew=np.array(RW),
                         term=not bool(info.get("timeout", False))))
    n = update()
    summary.append(dict(life=life + 1, seed=a.seed0 + life, steps=len(RW), relays=int(info.get("relays_reached", 0)),
                        success=bool(info.get("success")), collision=bool(info.get("collision")), ret=float(np.sum(RW) / REW_SCALE),
                        train_samples=n))
    print(summary[-1], flush=True)

wall = time.time() - t0
torch.save(net.export(), f"{a.out}/ppo_5lives.pt")
json.dump(dict(lives=summary, real_steps=real_steps, train_wall_s=wall, epochs=a.epochs, lr=a.lr, log_std=a.log_std),
          open(f"{a.out}/lives.json", "w"), indent=1)
print("done", real_steps, wall)
