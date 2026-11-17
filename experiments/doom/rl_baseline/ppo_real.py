"""Model-free PPO trained directly in the real VizDoom lab (take_cover). Encoder, vector env, policy factory."""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

SCEN = "take_cover"
N_ACT = 3
K_PROJ, K_MON = 6, 5
PROJ_F, MON_F, PLAYER_F = 7, 2, 3
FRAME_DIM = PLAYER_F + N_ACT + K_MON * MON_F + K_PROJ * PROJ_F      # 58
OBS_DIM = 2 * FRAME_DIM                                            # current + previous frame
Y0 = 384.0


class Encoder:
    """Stateful obs-dict -> vector. Player-frame features; stacks the current and the previous frame."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None
        self.last_a = 0

    def frame(self, obs):
        p = obs["player"]
        a = math.radians(p["angle"])
        c, s = math.cos(a), math.sin(a)
        f = np.zeros(FRAME_DIM, np.float32)
        f[0] = (p["y"] - Y0) / 384.0
        f[1] = (-p["vx"] * s + p["vy"] * c) / 5.0
        f[2] = max(p["health"], 0.0) / 100.0
        f[PLAYER_F + self.last_a] = 1.0
        projs, mons = [], []
        for o in obs["objects"]:
            dx, dy = o["x"] - p["x"], o["y"] - p["y"]
            fw, lt = dx * c + dy * s, -dx * s + dy * c
            nm = o["name"]
            if nm.endswith("Ball"):
                vf, vl = o["vx"] * c + o["vy"] * s, -o["vx"] * s + o["vy"] * c
                if fw > 0 and vf < -1e-3:
                    ttc = -fw / vf
                    key, icp = ttc, lt + vl * ttc
                    tt = min(ttc / 4.0, 50.0) / 50.0
                else:
                    key, icp, tt = 1e4 + abs(fw), 0.0, 1.0
                projs.append((key, [1.0, fw / 500.0, lt / 400.0, vf / 10.0, vl / 10.0, tt,
                                    max(-3.0, min(3.0, icp / 200.0))]))
            elif nm == "DoomImp":
                mons.append((lt, [1.0, lt / 400.0]))
        projs.sort(key=lambda t: t[0])
        mons.sort(key=lambda t: t[0])
        o0 = PLAYER_F + N_ACT
        for i, (_, v) in enumerate(mons[:K_MON]):
            f[o0 + i * MON_F:o0 + (i + 1) * MON_F] = v
        o1 = o0 + K_MON * MON_F
        for i, (_, v) in enumerate(projs[:K_PROJ]):
            f[o1 + i * PROJ_F:o1 + (i + 1) * PROJ_F] = v
        return f

    def __call__(self, obs):
        f = self.frame(obs)
        out = np.concatenate([f, f if self.prev is None else self.prev])
        self.prev = f
        return out

    def note_action(self, a):
        self.last_a = int(a)


# ------------------------------------------------------------------------------------ policy for lab.evaluate
class CkptPolicy:
    def __init__(self, ckpt: str):
        import torch
        torch.set_num_threads(1)
        from cawm.ppo import ActorCritic
        self.torch = torch
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net = ActorCritic.load(blob["policy"])
        self.enc = Encoder()

    def bind(self, scenario, action_names):
        assert len(action_names) == N_ACT

    def reset(self):
        self.enc.reset()

    @staticmethod
    def _noop():
        pass

    def act(self, obs):
        x = self.torch.as_tensor(self.enc(obs))[None]
        with self.torch.no_grad():
            a = int(self.net.act(x, deterministic=True)[0])
        self.enc.note_action(a)
        return a


def make_policy(ckpt: str):
    return CkptPolicy(ckpt)


# ------------------------------------------------------------------------------------ vector env
def _worker(wid, conn, tag, seed0):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from reallab.doom import api
    env = api.make(SCEN, purpose="oracle_train", pilot="ppo_real", tag=tag, seed=seed0 + wid)
    enc = Encoder()
    counter = [0]

    def reset():
        counter[0] += 1
        enc.reset()
        return enc(env.reset(seed0 + wid * 1_000_000 + counter[0]))

    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                conn.send(reset())
            elif cmd == "step":
                enc.note_action(arg)
                o, r, te, tr, _ = env.step(arg)
                done = bool(te or tr)
                x = enc(o) if not done else None
                conn.send((x, r, done))
            elif cmd == "close":
                break
    finally:
        env.close()
        conn.close()


class RealVecEnv:
    """Step-synchronous torch vector env over real VizDoom workers. auto_reset=True: finished envs restart inside
    step() (a reset consumes one real life). `lives` = number of resets issued so far."""

    n_actions = N_ACT
    obs_dim = OBS_DIM

    def __init__(self, num_envs, device="cuda", tag="ppo", seed0=50_000_000, auto_reset=True, reward_scale=0.25):
        import torch
        self.torch, self.num_envs, self.device = torch, num_envs, device
        self.auto_reset, self.rs = auto_reset, reward_scale
        self.lives, self.steps = 0, 0
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(num_envs):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(w, b, tag, seed0), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def _t(self, arr):
        return self.torch.as_tensor(np.asarray(arr), dtype=self.torch.float32, device=self.device)

    def reset(self, idx=None):
        idx = range(self.num_envs) if idx is None else idx
        for i in idx:
            self.conns[i].send(("reset", None))
        self.lives += len(list(idx))
        return self._t([self.conns[i].recv() for i in idx])

    def step(self, action):
        acts = action.tolist() if hasattr(action, "tolist") else list(action)
        for c, a in zip(self.conns, acts):
            c.send(("step", int(a)))
        res = [c.recv() for c in self.conns]
        self.steps += self.num_envs
        done = np.array([r[2] for r in res])
        rew = np.array([r[1] for r in res], np.float32) * self.rs
        obs = [r[0] for r in res]
        if self.auto_reset:
            di = np.nonzero(done)[0]
            for i in di:
                self.conns[i].send(("reset", None))
            self.lives += len(di)
            for i in di:
                obs[i] = self.conns[i].recv()
        else:
            for i in np.nonzero(done)[0]:
                obs[i] = np.zeros(OBS_DIM, np.float32)
        return self._t(obs), self._t(rew), self.torch.as_tensor(done, device=self.device), {}

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=20)
