"""code_wm controller v2: receding-horizon CEM over a THROW primitive, evaluated by long rollouts in the code world
under domain randomisation of the uncertain world parameters.  Lab policy protocol.

Primitive (times relative to the planning step):  go to wind-up point Wp, at t1 go to throw point Tp (the reference
then moves at up to 4 m/s), at t2 go to final point Fp and hold there.  8 numbers.
Cost of a candidate = mean over M imagined worlds of [ -1 + 0.002*t_success  if the world's success rule fires within
the rollout, else  |p1 - target| + 0.3*max node speed  at the end of the rollout ].
Plan at t=2 (velocities from 3 readings), then re-plan every `every` steps from the live reading (warm-started).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from worlds.annex import annex_world as W

LO = torch.tensor(W.ZONE[0])
HI = torch.tensor(W.ZONE[1])
DR_KEYS = ("mu", "k_bend", "c_bend", "aniso", "tip_mass", "lat_mass", "k_stretch", "c_stretch", "mu_post")


class ThrowCEM:
    def __init__(self, params_path="runs/annex/code_wm/world_params.json", n_cand=192, M=4, iters0=5, iters=2,
                 every=20, H=150, dr=0.10, elite=16, device="cuda", seed=0, param_override=None, vcmd=None,
                 f_max=1.0):
        self.vcmd, self.f_max = vcmd, f_max
        self.cmd0 = None
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        prm = dict(W.PARAMS)
        if params_path and os.path.exists(params_path):
            prm.update(json.load(open(params_path))["params"])
        if param_override:
            prm.update(param_override)
        self.prm = prm
        self.n_cand, self.M, self.iters0, self.iters, self.every, self.H = n_cand, M, iters0, iters, every, H
        self.dr, self.elite = dr, elite
        self.base_seed = seed
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        self.lo, self.hi = LO.to(self.dev).float(), HI.to(self.dev).float()
        self.plan_times = []
        # randomised world parameter sets (first = nominal)
        g = torch.Generator().manual_seed(1234 + seed)
        sets = []
        for m in range(M):
            p = {}
            for k, v in prm.items():
                f = 1.0 if (m == 0 or k not in DR_KEYS) else float(1 + dr * (2 * torch.rand(1, generator=g) - 1))
                p[k] = v * f
            sets.append(p)
        self.sets = sets

    def bind(self, task_card):
        self.card = task_card

    def set_seed(self, s):
        self.gen.manual_seed(int(self.base_seed) * 100003 + int(s))

    def reset(self):
        self.hist = []
        self.rethrow = 1
        self.psucc = 0.0
        self.theta = None      # current plan (8,) with absolute times
        self.t_plan = -1

    # ---------------------------------------------------------------- primitive
    def _commands(self, th, t0, T, tool):
        """th (B,8) [Wx,Wy,Tx,Ty,Fx,Fy,t1,t2] (t1,t2 absolute step indices); returns (B,T,2) commands from step t0."""
        B = th.shape[0]
        ts = torch.arange(t0, t0 + T, device=self.dev).float().unsqueeze(0)
        Wp, Tp, Fp = th[:, 0:2], th[:, 2:4], th[:, 4:6]
        t1, t2 = th[:, 6:7], th[:, 7:8]
        A = torch.where((ts < t1).unsqueeze(-1), Wp.unsqueeze(1), torch.where((ts < t2).unsqueeze(-1), Tp.unsqueeze(1), Fp.unsqueeze(1)))
        A = torch.maximum(torch.minimum(A, self.hi), self.lo)
        if self.vcmd is None:
            return A
        # rate-limit the commanded point (caps tool speed; life 2: an uncapped 0.4 m stroke threw the rod 0.27 m
        # into the air, a regime the planar world does not model)
        c = self.cmd0.reshape(1, 2).expand(B, 2)
        out = []
        vs = self.vcmd * W.DT
        for k in range(T):
            d = A[:, k] - c
            n = d.norm(dim=-1, keepdim=True)
            c = c + d * torch.clamp(vs / n.clamp_min(1e-9), max=1.0)
            out.append(c)
        return torch.stack(out, 1)

    def _sample(self, n, mean, std, t0):
        x = mean + std * torch.randn(n, 8, device=self.dev, generator=self.gen)
        x[:, 0:6] = torch.maximum(torch.minimum(x[:, 0:6], self.hi.repeat(3)), self.lo.repeat(3))
        # one throw only: the hold point stays within f_max of the throw point
        off = x[:, 4:6] - x[:, 2:4]
        x[:, 4:6] = x[:, 2:4] + off * torch.clamp(self.f_max / off.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
        if float(mean[6]) < t0:          # phase already started in reality: its switch time is history
            x[:, 6] = mean[6]
        else:
            x[:, 6] = torch.clamp(x[:, 6], min=t0)
        if float(mean[7]) < t0:
            x[:, 7] = mean[7]
        else:
            x[:, 7] = torch.maximum(torch.clamp(x[:, 7], min=t0), x[:, 6] + 2)
        return x

    @torch.no_grad()
    def _evaluate(self, th, t0):
        n = th.shape[0]
        M = self.M
        o = self.hist[-1]
        B = n * M
        prm = {k: torch.tensor([s[k] for s in self.sets], device=self.dev).repeat_interleave(n) for k in self.prm}
        st = W.init_from_obs([o] * B, [self.hist[-2]] * B, prm, self.dev,
                             prev2_obs_batch=[self.hist[-3]] * B if len(self.hist) >= 3 else None)
        st["t"] = torch.full((B,), int(o["t"]), device=self.dev)
        T = min(self.H, W.TIME_LIMIT - int(o["t"]))
        A = self._commands(th, t0, T, None).repeat(M, 1, 1)
        tsucc = torch.full((B,), float("inf"), device=self.dev)
        for k in range(T):
            st, r, d = W.step(st, A[:, k])
            newly = st["success"] & torch.isinf(tsucc)
            tsucc = torch.where(newly, torch.full_like(tsucc, float(k)), tsucc)
            if bool(st["done"].all()):
                break
        dist = (st["pos"][:, 0] - st["target"]).norm(dim=-1)
        sp = st["vel"].norm(dim=-1).amax(-1)
        c = torch.where(st["success"], -1.0 + 0.002 * tsucc, dist + 0.3 * sp)
        c = torch.nan_to_num(c, nan=10.0)
        return c.reshape(M, n).mean(0), st["success"].float().reshape(M, n).mean(0)

    def _plan(self, t0):
        tool = torch.tensor(self.hist[-1]["tool_pos"], device=self.dev).float()
        tg = torch.tensor(self.hist[-1]["target"], device=self.dev).float()
        if self.theta is None:
            # broad prior: wind-up anywhere, throw towards the target side of the zone, hold near the throw point
            mean = torch.tensor([0.40, 0.0, 0.50, 0.0, 0.45, 0.0, t0 + 10.0, t0 + 25.0], device=self.dev)
            std = torch.tensor([0.10, 0.20, 0.10, 0.20, 0.10, 0.20, 8.0, 10.0], device=self.dev)
            iters = self.iters0
        else:
            mean = self.theta.clone()
            std = torch.tensor([0.02, 0.03, 0.02, 0.03, 0.03, 0.04, 2.0, 2.0], device=self.dev)
            iters = self.iters
        best, bestc, bests = self.theta, float("inf"), 0.0
        for it in range(iters):
            th = self._sample(self.n_cand, mean, std, t0)
            if best is not None:
                th[0] = best
            if self.theta is not None and it == 0:
                th[1] = self.theta
                th[2, :] = torch.cat([tool, tool, tool, torch.tensor([t0, t0 + 2.0], device=self.dev)])   # hold still
            c, s = self._evaluate(th, t0)
            idx = torch.argsort(c)[: self.elite]
            if float(c[idx[0]]) < bestc:
                bestc, best, bests = float(c[idx[0]]), th[idx[0]].clone(), float(s[idx[0]])
            mean = th[idx].mean(0)
            std = th[idx].std(0) * 0.8 + torch.tensor([0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.5, 0.5], device=self.dev)
        self.theta, self.cost, self.psucc = best, bestc, bests

    def act(self, obs):
        self.hist.append({k: (np.array(v, dtype=np.float64) if not np.isscalar(v) else v) for k, v in obs.items()})
        self.hist = self.hist[-3:]
        t = int(obs["t"])
        if len(self.hist) < 3:
            self.cmd0 = torch.tensor(obs["tool_pos"], device=self.dev).float()
            return np.array(obs["tool_pos"], dtype=np.float64)
        if self.theta is None or (t - self.t_plan) >= self.every:
            # second throw: the first throw is over, and no imagined world predicts success any more (life 4: the
            # tip rebounded towards the tool after a throw that overshot sideways)
            if (self.rethrow and self.theta is not None and t > float(self.theta[7]) + 20 and t < 160
                    and self.psucc == 0.0):
                self.theta = None
                self.rethrow -= 1
            t0 = time.time()
            self._plan(t)
            self.t_plan = t
            self.plan_times.append(time.time() - t0)
        a = self._commands(self.theta.unsqueeze(0), t, 1, None)[0, 0]
        self.cmd0 = a.clone()
        return a.cpu().numpy().astype(np.float64)


def make(**kw):
    return ThrowCEM(**kw)


FINAL = dict(n_cand=128, M=4, iters0=5, iters=1, every=8, H=120, dr=0.25, vcmd=3.0, f_max=0.06)


def make_final(**kw):
    """The controller selected in the code world (settings in FINAL; kwargs override)."""
    return ThrowCEM(**{**FINAL, **kw})
