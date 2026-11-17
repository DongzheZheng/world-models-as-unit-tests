"""code_wm controller: sampling MPC (MPPI + fling primitives) with the code world as the model, re-planned from the
live readings every `replan` steps.  Lab policy protocol: reset(), act(obs)->(2,), bind(task_card), set_seed(s).

Planning state: node positions from the reading, node velocities from a 2nd-order backward difference of the last
three readings (first step: hold still to observe motion).  Cost (per sample, over the horizon H):
  w_track * mean_t(w(t) * |p1 - target|) + w_term * |p1(H) - target| + w_vel * mean node speed at H
  + w_rest * (max node speed at H > rest threshold)  -  success bonus if the world's success rule fires.
Samples: Gaussian perturbations of the warm-started mean (knots every `knot` steps, linear interpolation) plus
`n_prim` fling primitives (wind-up point -> throw point -> hold point, random timings).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from worlds.annex import annex_world as W

LO = np.array(W.ZONE[0])
HI = np.array(W.ZONE[1])


class CodeWMController:
    def __init__(self, params_path="runs/annex/code_wm/world_params.json", H=50, knot=5, K=192, n_prim=64, replan=2,
                 sigma=0.06, lam=0.02, w_track=1.0, w_term=3.0, w_vel=0.5, w_rest=0.3, iters=1, device="cuda",
                 param_override=None, seed=0):
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        prm = dict(W.PARAMS)
        if params_path and os.path.exists(params_path):
            prm.update(json.load(open(params_path))["params"])
        if param_override:
            prm.update(param_override)
        self.prm = prm
        self.H, self.knot, self.K, self.n_prim, self.replan = H, knot, K, n_prim, replan
        self.sigma, self.lam = sigma, lam
        self.w = dict(track=w_track, term=w_term, vel=w_vel, rest=w_rest)
        self.iters = iters
        self.nk = H // knot + 1
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        self.base_seed = seed
        self.plan_time = []

    # ---------------- protocol
    def bind(self, task_card):
        self.card = task_card

    def set_seed(self, s):
        self.gen.manual_seed(int(self.base_seed) * 100003 + int(s))

    def reset(self):
        self.hist = []
        self.mean = None          # (nk,2) knot means
        self.queue = []

    def act(self, obs):
        self.hist.append({k: (np.array(v, dtype=np.float64) if not np.isscalar(v) else v) for k, v in obs.items()})
        self.hist = self.hist[-3:]
        if len(self.hist) < 2:
            return np.array(obs["tool_pos"], dtype=np.float64)       # observe one step to get velocities
        if not self.queue:
            t0 = time.time()
            self.queue = list(self._plan())
            self.plan_time.append(time.time() - t0)
        return self.queue.pop(0)

    # ---------------- planning
    def _state(self, B):
        o = self.hist[-1]
        prev = [self.hist[-2]] * B
        prev2 = [self.hist[-3]] * B if len(self.hist) >= 3 else None
        return W.init_from_obs([o] * B, prev, self.prm, self.dev, prev2_obs_batch=prev2)

    def _interp(self, knots):
        # knots (B,nk,2) -> actions (B,H,2)
        B = knots.shape[0]
        s = torch.arange(self.H, device=self.dev, dtype=torch.float32) / self.knot
        i0 = s.floor().long().clamp(max=self.nk - 2)
        f = (s - i0).unsqueeze(-1)
        return knots[:, i0] * (1 - f) + knots[:, i0 + 1] * f

    def _prims(self, n, tool, target):
        """Fling primitives: wind-up (away from target) -> throw point (towards target) -> hold point."""
        g = self.gen
        lo = torch.tensor(LO, device=self.dev, dtype=torch.float32)
        hi = torch.tensor(HI, device=self.dev, dtype=torch.float32)
        r = lambda *s: torch.rand(*s, device=self.dev, generator=g)
        A = torch.empty(n, self.H, 2, device=self.dev)
        wind = lo + (hi - lo) * r(n, 2)
        throw = lo + (hi - lo) * r(n, 2)
        hold = throw + 0.08 * (r(n, 2) - 0.5)
        t1 = (r(n) * 15).long()
        t2 = t1 + 3 + (r(n) * 15).long()
        ts = torch.arange(self.H, device=self.dev).unsqueeze(0)
        A[:] = wind.unsqueeze(1)
        A = torch.where((ts < t1.unsqueeze(1)).unsqueeze(-1), tool.unsqueeze(0).unsqueeze(0).expand(n, self.H, 2) * 0 + wind.unsqueeze(1), A)
        A = torch.where((ts >= t1.unsqueeze(1)).unsqueeze(-1) & (ts < t2.unsqueeze(1)).unsqueeze(-1), wind.unsqueeze(1).expand(n, self.H, 2), A)
        A = torch.where((ts >= t2.unsqueeze(1)).unsqueeze(-1), throw.unsqueeze(1).expand(n, self.H, 2), A)
        late = (ts >= (t2 + 12).unsqueeze(1)).unsqueeze(-1)
        A = torch.where(late, hold.unsqueeze(1).expand(n, self.H, 2), A)
        return torch.maximum(torch.minimum(A, hi), lo)

    @torch.no_grad()
    def _cost(self, A):
        B = A.shape[0]
        st = self._state(B)
        tg = st["target"]
        c = torch.zeros(B, device=self.dev)
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        t_now = int(self.hist[-1]["t"])
        Hs = min(self.H, W.TIME_LIMIT - t_now)
        nrest = max(1, Hs // 4)
        for k in range(Hs):
            prev_tool = st["tool"]
            st, r, d = W.step(st, A[:, k])
            dist = (st["pos"][:, 0] - tg).norm(dim=-1)
            wt = (k + 1) / Hs
            c = c + self.w["track"] * wt * dist * (~done) / Hs
            if k >= Hs - nrest:   # dense rest signal over the last quarter of the horizon
                mv = (st["pos"] - st["prev_pos"]).norm(dim=-1).amax(-1) / W.REST_STEP
                c = c + self.w["rest"] * torch.clamp(mv - 1, 0, 3) * (~st["success"]) / nrest
            if k > 0:
                c = c + 0.02 * (A[:, k] - A[:, k - 1]).norm(dim=-1)
            done = done | d
        dist = (st["pos"][:, 0] - tg).norm(dim=-1)
        sp = st["vel"].norm(dim=-1)
        c = c + self.w["term"] * dist * (~st["success"]) + self.w["vel"] * sp.mean(-1) * (~st["success"])
        c = c - 2.0 * st["success"].float()
        c = torch.nan_to_num(c, nan=1e3)
        return c

    def _plan(self):
        o = self.hist[-1]
        tool = torch.tensor(o["tool_pos"], device=self.dev, dtype=torch.float32)
        target = torch.tensor(o["target"], device=self.dev, dtype=torch.float32)
        lo = torch.tensor(LO, device=self.dev, dtype=torch.float32)
        hi = torch.tensor(HI, device=self.dev, dtype=torch.float32)
        if self.mean is None:
            self.mean = tool.unsqueeze(0).repeat(self.nk, 1)
        else:   # shift warm start by `replan` steps
            sh = self.replan / self.knot
            s = torch.arange(self.nk, device=self.dev, dtype=torch.float32) + sh
            i0 = s.floor().long().clamp(max=self.nk - 2)
            f = (s - i0).clamp(max=1).unsqueeze(-1)
            self.mean = self.mean[i0] * (1 - f) + self.mean[(i0 + 1).clamp(max=self.nk - 1)] * f
        best_A, best_c = None, None
        for it in range(self.iters):
            Kg = self.K - (self.n_prim if it == 0 else 0)
            eps = torch.randn(Kg, self.nk, 2, device=self.dev, generator=self.gen) * self.sigma
            eps[0] = 0
            knots = torch.maximum(torch.minimum(self.mean.unsqueeze(0) + eps, hi), lo)
            A = self._interp(knots)
            hold = tool.reshape(1, 1, 2).expand(1, self.H, 2)
            A = torch.cat([A, hold], 0)
            if it == 0 and self.n_prim:
                A = torch.cat([A, self._prims(self.n_prim, tool, target)], 0)
            c = self._cost(A)
            i = int(torch.argmin(c))
            if best_c is None or c[i] < best_c:
                best_c, best_A = float(c[i]), A[i].clone()
            w = torch.softmax(-(c - c.min()) / self.lam, 0)
            Am = (w.reshape(-1, 1, 1) * A).sum(0)
            # refit knot means from the weighted action sequence (sample at knot times)
            idx = torch.clamp(torch.arange(self.nk, device=self.dev) * self.knot, max=self.H - 1)
            self.mean = Am[idx]
        # execute the best sample's first actions (more robust than the weighted mean for multimodal flings)
        self.last_cost = best_c
        self.mean = best_A[torch.clamp(torch.arange(self.nk, device=self.dev) * self.knot, max=self.H - 1)]
        return [best_A[k].cpu().numpy().astype(np.float64) for k in range(self.replan)]


def make(**kw):
    return CodeWMController(**kw)
