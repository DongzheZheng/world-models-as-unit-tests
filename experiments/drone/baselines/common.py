"""Shared pieces for the drone RL baselines: obs encoder, action map, real-env vector env, policy factory."""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

HOVER_T = 0.30          # datasheet hover command
T_SPAN = 0.20           # thrust residual range around hover: T in [0.10, 0.50]
RATE_SCALE = np.array([np.pi, np.pi, np.pi / 2])
TIME_LIMIT = 17.0
OBS_DIM = 46
ACT_DIM = 4
REW_SCALE = 0.1


def action_to_cmd(a) -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    return np.array([*(a[:3] * RATE_SCALE), np.clip(HOVER_T + T_SPAN * a[3], 0.0, 0.8)])


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def encode(obs: dict, prev_a) -> np.ndarray:
    R = _rot(obs["quat"])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])           # world -> heading frame
    pos, goal = obs["pos"], obs["goal"]
    k = int(round(float(obs["relay_index"][0])))
    relays = obs["relays"]
    nxt = relays[k + 1] if k + 1 < len(relays) else goal
    rg = Rz @ (goal - pos) / 20.0
    rn = Rz @ (nxt - pos) / 20.0 if k + 1 < len(relays) else np.zeros(3)
    rng = np.asarray(obs["ranges"])
    hz = rng.min(0) / 12.0                                         # (16,) min over elevation per azimuth
    down, up = rng[:2].min() / 12.0, rng[4:].min() / 12.0
    onehot = np.zeros(3); onehot[min(k, 2)] = 1.0
    t_rem = (TIME_LIMIT - float(obs["t"][0])) / TIME_LIMIT
    v = Rz @ obs["lin_vel"] / 5.0
    p = np.array([pos[0] / 30.0 - 1.0, pos[1] / 3.0, pos[2] - 2.0])
    return np.concatenate([R[2], np.asarray(obs["ang_vel"]) / np.pi, v, rg, rn, p, [c, s], hz, [down, up],
                           [t_rem], onehot, np.asarray(prev_a, dtype=np.float64)]).astype(np.float32)


# ------------------------------------------------------------------------------------------ eval policy
class RLPolicy:
    """Deterministic policy from a ppo.ActorCritic export, used by reallab.drone.parallel.evaluate."""

    def __init__(self, ckpt: str, deterministic: bool = True):
        import torch
        from cawm.ppo import ActorCritic
        torch.set_num_threads(1)
        self.torch = torch
        self.net = ActorCritic.load(torch.load(ckpt, map_location="cpu", weights_only=False))
        self.det = deterministic
        self.prev = np.zeros(4)

    def reset(self):
        self.prev = np.zeros(4)

    def act(self, obs):
        with self.torch.no_grad():
            x = self.torch.as_tensor(encode(obs, self.prev))[None]
            a = self.net.act(x, deterministic=self.det)[0].numpy()
        self.prev = np.clip(a, -1, 1)
        return action_to_cmd(a)


def rl_policy(**kw):
    return RLPolicy(**kw)


# ------------------------------------------------------------------------------------------ vector env
def _worker(conn, idx, kw):
    from reallab.drone.api import make
    env = make("site", purpose=kw["purpose"], tag=kw["tag"], pilot=kw["pilot"], mission="A", command_mode="ctbr")
    rng = np.random.default_rng(kw["seed"] * 1000 + idx)
    base = 1_000_000 + idx * 20_000_000

    def reset():
        seed = int(base + rng.integers(0, 19_000_000))
        o, _ = env.reset(seed=seed)
        return o

    obs = reset(); prev = np.zeros(4)
    conn.send(encode(obs, prev))
    ep_steps = 0
    while True:
        cmd, a = conn.recv()
        if cmd == "close":
            env.close(); conn.close(); return
        o, r, te, tr, info = env.step(action_to_cmd(a))
        prev = np.clip(a, -1, 1); ep_steps += 1
        done = bool(te or tr)
        enc = encode(o, prev)
        out = dict(r=float(r), done=done, enc=enc, timeout=False, success=False, info=None)
        if done:
            out.update(timeout=bool(info.get("timeout", False)), success=bool(info.get("success", False)),
                       info=dict(relays=int(info.get("relays_reached", 0)), time_s=float(info.get("time_s", 0.0)),
                                 collision=bool(info.get("collision")), n=ep_steps), term_enc=enc)
            obs = reset(); prev = np.zeros(4); ep_steps = 0
            out["enc"] = encode(obs, prev)
        conn.send(out)


class RealVecEnv:
    """Step-synchronous vector env of REAL (metered) drone processes with the batched-torch interface of cawm.ppo."""
    act_dim, obs_dim = ACT_DIM, OBS_DIM
    n_actions = None

    def __init__(self, n: int, device="cuda", purpose="oracle_train", pilot="ppo_real", tag="ppo_real", seed=0):
        import torch
        self.torch, self.num_envs, self.device = torch, n, device
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, i, dict(purpose=purpose, pilot=pilot, tag=tag, seed=seed)), daemon=True)
            p.start(); self.conns.append(a); self.procs.append(p)
        self.episodes = 0
        self.steps = 0
        self.succ = 0
        self.recent = []                # (success, time_s, relays) of finished episodes
        self._first = [c.recv() for c in self.conns]

    def reset(self):
        return self.torch.as_tensor(np.stack(self._first), device=self.device)

    def step(self, a):
        torch = self.torch
        an = a.detach().cpu().numpy().astype(np.float64)
        for c, x in zip(self.conns, an):
            c.send(("step", x))
        res = [c.recv() for c in self.conns]
        self.steps += self.num_envs
        obs = torch.as_tensor(np.stack([r["enc"] for r in res]), device=self.device)
        rew = torch.as_tensor([r["r"] * REW_SCALE for r in res], dtype=torch.float32, device=self.device)
        done = torch.as_tensor([r["done"] for r in res], device=self.device)
        info = {}
        if done.any():
            self.episodes += int(done.sum())
            tout = torch.as_tensor([r["timeout"] for r in res], device=self.device)
            tobs = torch.as_tensor(np.stack([r.get("term_enc", r["enc"]) for r in res]), device=self.device)
            info = dict(time_out=tout, terminal_obs=tobs)
            for r in res:
                if r["done"]:
                    self.succ += int(r["success"])
                    self.recent.append((r["success"], r["info"]["time_s"], r["info"]["relays"]))
        return obs, rew, done, info

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
