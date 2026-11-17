"""Push-primitive sampling MPC with the code world as the model (all planning happens inside the code world).

A primitive = transit to a pre-contact point (around the block or straight), push along a direction for a distance at
a speed, retreat.  Each replanning samples N primitives (contact point on the T boundary, push-angle offset from the
inward normal, distance, speed, transit mode), rolls ALL of them forward in the code world (batched on the GPU) incl.
a few settle steps, refines the best by CEM, and executes the winner open loop until its retreat ends, then replans.
Score: -1000 + t if the world predicts success during the primitive, else pose cost at its end + time_w * its length.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from worlds.pusht import pusht_world as W

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# object geometry is read from the world module at call time (W.set_object switches T / L)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def pose_cost(q, ang_w):
    """Distance between block and goal: T-centroid distance [u] + ang_w * |angle error| [rad]."""
    g = torch.tensor(W.GOAL, device=q.device)
    C = torch.tensor(W.CENTROID, device=q.device)
    c = W.to_world(q, C.expand(q.shape[0], 1, 2))[:, 0]
    gc = W.to_world(g[None], C[None, None])[0, 0]
    return (c - gc).norm(dim=-1) + ang_w * wrap(q[:, 2] - g[2]).abs()


def wall_cost(q, margin):
    """How far [u] the block's outline comes inside `margin` of the walls (0 when clear). The wall mechanism (M5) is
    the least trustworthy part of the world (L life 2: the real block penetrated the wall by up to 25 u and was later
    ejected), so the planner is told to keep the block away from walls."""
    V = torch.tensor(W.VERTS, dtype=q.dtype, device=q.device)
    cw = W.to_world(q, V.expand(q.shape[0], -1, -1))
    clear = torch.minimum(cw - W.WALL_LO, W.WALL_HI - cw).amin(dim=(1, 2))
    return (margin - clear).clamp(min=0)


def boundary_point(s):
    """s in [0,1) -> (body point on T perimeter, outward normal)."""
    V = torch.tensor(W.VERTS, dtype=torch.float32, device=s.device)
    E = torch.roll(V, -1, 0) - V
    lens = E.norm(dim=-1)
    cum = torch.cat([torch.zeros(1, device=s.device), lens.cumsum(0)])
    x = s * cum[-1]
    i = (torch.searchsorted(cum, x, right=True) - 1).clamp(0, len(V) - 1)
    f = ((x - cum[i]) / lens[i])[:, None]
    pt = V[i] + f * E[i]
    d = E[i] / lens[i][:, None]
    nrm = torch.stack([d[:, 1], -d[:, 0]], -1)          # CCW polygon -> outward normal = (dy, -dx)
    return pt, nrm


class Planner:
    def __init__(self, n_samples=1536, cem_iters=2, n_elite=48, R=150.0, transit_step=20.0, gap=8.0,
                 settle=4, retreat=3, time_w=0.08, ang_w=60.0, max_push_steps=40, max_transit=40, params=None,
                 seed=0, mix_straight=0.5, succ_thr=0.95, object="T", wall_w=0.0, wall_margin=25.0, chain_k=0, chain_n=64):
        W.set_object(object)
        self.__dict__.update(dict(n=n_samples, iters=cem_iters, ne=n_elite, R=R, ts=transit_step, gap=gap,
                                  settle=settle, retreat=retreat, time_w=time_w, ang_w=ang_w, mps=max_push_steps,
                                  mtr=max_transit, params=params, mix_straight=mix_straight,
                                  succ_thr=succ_thr, wall_w=wall_w, wall_margin=wall_margin,
                                  chain_k=chain_k, chain_n=chain_n))
        self.g = torch.Generator(device=DEV).manual_seed(seed)
        self.wm_steps = 0                      # code-world agent steps simulated while planning

    # -------- primitive -> action sequence (batched)
    def sequences(self, p, q, z):
        """p (M,2) pusher, q (M,3) block, z (M,6) primitive params in [0,1]: s, angle, log-dist, speed, mode, gap.
        Returns actions (M,T,2), exec_len (M,), score_idx (M,)."""
        M = p.shape[0]
        dev = p.device
        s, ang, ld, sp, mode = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4]
        pt, nrm = boundary_point(s % 1.0)
        a = (ang - 0.5) * math.radians(70)
        ca, sa = torch.cos(a), torch.sin(a)
        din = -torch.stack([ca * nrm[:, 0] - sa * nrm[:, 1], sa * nrm[:, 0] + ca * nrm[:, 1]], -1)   # body
        L = torch.exp(math.log(1.5) + ld * (math.log(90.0) - math.log(1.5)))
        speed = 1.5 + sp * 8.5
        cw = W.to_world(q, pt[:, None])[:, 0]
        c, s_ = torch.cos(q[:, 2]), torch.sin(q[:, 2])
        dw = torch.stack([c * din[:, 0] - s_ * din[:, 1], s_ * din[:, 0] + c * din[:, 1]], -1)
        pre = cw - dw * (W.R_PUSHER + self.gap)
        # transit polyline: [p, out, arc(12), pre] (around) or [p ... p, pre] (straight)
        cen = W.to_world(q, torch.tensor(W.CENTROID, device=dev).expand(M, 1, 2))[:, 0]
        rel0, rel1 = p - cen, pre - cen
        a0, a1 = torch.atan2(rel0[:, 1], rel0[:, 0]), torch.atan2(rel1[:, 1], rel1[:, 0])
        r0 = rel0.norm(dim=-1).clamp(min=self.R)
        da = wrap(a1 - a0)
        K = 12
        fr = torch.linspace(0, 1, K, device=dev)
        arc_a = a0[:, None] + da[:, None] * fr[None]
        arc_r = r0[:, None] + (self.R - r0[:, None]) * fr[None]
        arc = cen[:, None] + arc_r[..., None] * torch.stack([torch.cos(arc_a), torch.sin(arc_a)], -1)
        arc = arc.clamp(22.0, 490.0)
        around = torch.cat([p[:, None], arc, pre[:, None]], 1)                    # (M,K+2,2)
        straight = p[:, None] + (pre - p)[:, None] * torch.linspace(0, 1, K + 2, device=dev)[None, :, None]
        use_st = (mode < self.mix_straight)[:, None, None]
        poly = torch.where(use_st, straight, around)
        seg = (poly[:, 1:] - poly[:, :-1]).norm(dim=-1)
        cum = torch.cat([torch.zeros(M, 1, device=dev), seg.cumsum(1)], 1)
        total = cum[:, -1]
        n_tr = torch.ceil(total / self.ts).clamp(min=1, max=self.mtr).long()
        n_push = torch.ceil((self.gap + L) / speed).clamp(min=1, max=self.mps).long()
        T = int((n_tr + 1 + n_push + self.retreat + self.settle).max())
        k = torch.arange(T, device=dev)[None].expand(M, T)
        # transit position at arc length min((k+1)*step_len, total), step_len = total / n_tr
        sl = (total / n_tr)[:, None] * (k + 1).float()
        sl = torch.minimum(sl, total[:, None])
        j = (torch.searchsorted(cum, sl.contiguous(), right=True) - 1).clamp(0, poly.shape[1] - 2)
        c0 = torch.gather(cum, 1, j)
        sj = torch.gather(seg, 1, j).clamp(min=1e-6)
        P0 = torch.gather(poly, 1, j[..., None].expand(-1, -1, 2))
        P1 = torch.gather(poly, 1, (j + 1)[..., None].expand(-1, -1, 2))
        tr = P0 + (P1 - P0) * ((sl - c0) / sj).clamp(0, 1)[..., None]
        kp = (k - n_tr[:, None] - 1).clamp(min=0)                                 # push step index
        push_d = torch.minimum((kp + 1).float() * speed[:, None], (self.gap + L)[:, None])
        push = pre[:, None] + dw[:, None] * push_d[..., None]
        end = pre + dw * (self.gap + L)[:, None]
        kr = (k - n_tr[:, None] - 1 - n_push[:, None]).clamp(min=0)
        back = end[:, None] - dw[:, None] * (torch.minimum(kr + 1, torch.full_like(kr, self.retreat)).float() * 8.0)[..., None]
        A = torch.where((k < n_tr[:, None])[..., None], tr,
                        torch.where((k == n_tr[:, None])[..., None], pre[:, None].expand(-1, T, -1),
                                    torch.where((kr == 0)[..., None] & (k < n_tr[:, None] + 1 + n_push[:, None])[..., None],
                                                push, back)))
        exec_len = n_tr + 1 + n_push + self.retreat
        return A.clamp(1.0, 511.0), exec_len, exec_len + self.settle - 1

    @torch.no_grad()
    def score(self, state, z, steps_left):
        """state: dict with batch M (already repeated per candidate)."""
        A, exec_len, sidx = self.sequences(state["p"], state["q"], z)
        M, T = A.shape[:2]
        s = state
        cost_at = torch.zeros(M, device=A.device)
        s_end = {k: v.clone() for k, v in s.items()}
        succ_t = torch.full((M,), 10_000, device=A.device, dtype=torch.long)
        self.wm_steps += M * T
        for t in range(T):
            s = W.physics(s, A[:, t], self.params)
            cov = W.coverage(s["q"])
            hit = (cov > self.succ_thr) & (succ_t == 10_000) & (t < exec_len) & (t < steps_left)
            succ_t = torch.where(hit, torch.full_like(succ_t, t), succ_t)
            cost_at = torch.where(sidx == t, pose_cost(s["q"], self.ang_w) + self.wall_w * wall_cost(s["q"], self.wall_margin),
                                  cost_at)
            at = sidx == t
            for k in ("p", "pv", "q", "v", "w"):
                s_end[k] = torch.where(at.reshape(-1, *([1] * (s[k].dim() - 1))), s[k], s_end[k])
        sc = cost_at + self.time_w * exec_len.float()
        sc = torch.where(exec_len > steps_left, sc + 50.0, sc)
        sc = torch.where(succ_t < 10_000, -1000.0 + succ_t.float(), sc)
        self.last_end = s_end
        return sc, A, exec_len

    @torch.no_grad()
    def plan(self, state, steps_left):
        """state: batch B. Returns per-env (actions (T,2) numpy, exec_len, predicted score)."""
        B = state["p"].shape[0]
        N = self.n
        rep = lambda st, n: {k: v.repeat_interleave(n, 0) for k, v in st.items()}
        z = torch.rand(B, N, 6, device=DEV, generator=self.g)
        sl = torch.as_tensor(steps_left, device=DEV).reshape(-1).expand(B).repeat_interleave(N)
        for it in range(self.iters + 1):
            sc, A, el = self.score(rep(state, N), z.reshape(B * N, 6), sl)
            sc = sc.reshape(B, N)
            if it == self.iters:
                break
            order = sc.argsort(-1)[:, :self.ne]
            el_z = torch.gather(z, 1, order[..., None].expand(-1, -1, 6))
            sd = 0.08 / (it + 1)
            pick = torch.randint(self.ne, (B, N), device=DEV, generator=self.g)
            nz = torch.gather(el_z, 1, pick[..., None].expand(-1, -1, 6)) + sd * torch.randn(B, N, 6, device=DEV, generator=self.g)
            nz[..., 0] = nz[..., 0] % 1.0
            nz = nz.clamp(0, 1)
            nz[:, :self.ne] = el_z                                       # keep the elites
            z = nz
        if self.chain_k:
            # two-primitive lookahead: from the end state of each of the K best first primitives, sample chain_n
            # second primitives; a first primitive is worth min(own score, best follow-up score + its own time)
            K, N2 = self.chain_k, self.chain_n
            topk = sc.argsort(-1)[:, :K]                                           # (B,K)
            flat = (topk + torch.arange(B, device=DEV)[:, None] * N).reshape(-1)
            end = {k: v[flat] for k, v in self.last_end.items()}
            end["t"] = torch.zeros(B * K, dtype=torch.long, device=DEV)
            el1 = el.reshape(B * N)[flat]
            sl2 = (sl.reshape(B, N)[:, :K].reshape(-1) - el1).clamp(min=1)
            z2 = torch.rand(B * K, N2, 6, device=DEV, generator=self.g)
            sc2, _, _ = self.score(rep(end, N2), z2.reshape(-1, 6), sl2.repeat_interleave(N2))
            sc2 = sc2.reshape(B, K, N2).min(-1).values
            first = torch.gather(sc, 1, topk)
            succ_first = first < -500
            sc2 = torch.where(sc2 < -500, sc2 + el1.reshape(B, K).float(), sc2 + self.time_w * el1.reshape(B, K).float())
            chained = torch.where(succ_first, first, torch.minimum(first, sc2))
            sc = torch.full_like(sc, float("inf")).scatter(1, topk, chained)
        best = sc.argmin(-1)
        A = A.reshape(B, N, *A.shape[1:])
        el = el.reshape(B, N)
        out = []
        for b in range(B):
            i = best[b]
            out.append((A[b, i].cpu().numpy(), int(el[b, i]), float(sc[b, i])))
        return out


def obs_state(obs, prev_pose):
    """World state from an observation; block twist from the previous observed pose (finite difference)."""
    p = torch.tensor(np.asarray(obs["agent_pos"])[None], dtype=torch.float32, device=DEV)
    pv = torch.tensor(np.asarray(obs["agent_vel"])[None], dtype=torch.float32, device=DEV)
    q = torch.tensor(np.asarray(obs["block_pose"])[None], dtype=torch.float32, device=DEV)
    v = torch.zeros(1, 2, device=DEV); w = torch.zeros(1, device=DEV)
    if prev_pose is not None:
        qm = torch.tensor(np.asarray(prev_pose)[None], dtype=torch.float32, device=DEV)
        prm = W.PARAMS
        comb = torch.tensor([[[prm["com_u"], prm["com_v"]]]], device=DEV)
        v = (W.to_world(q, comb)[:, 0] - W.to_world(qm, comb)[:, 0]) / W.DT
        w = wrap(q[:, 2] - qm[:, 2]) / W.DT
    return W.make_state(p, pv, q, v, w)


class MPCPolicy:
    """Real-world policy (protocol of reallab.pusht.parallel): replans with the Planner in the code world."""

    def __init__(self, **planner_kw):
        self.kw = planner_kw
        self.planner = None

    def set_seed(self, s):
        self.seed = int(s)

    def reset(self):
        if self.planner is None:
            self.planner = Planner(**self.kw)
        self.planner.g.manual_seed(getattr(self, "seed", 0))
        self.queue, self.prev, self.t = [], None, 0

    def act(self, obs):
        if not self.queue:
            st = obs_state(obs, self.prev)
            A, el, sc = self.planner.plan(st, 300 - self.t)[0]
            self.queue = list(A[:el])
        self.prev = np.asarray(obs["block_pose"]).copy()
        self.t += 1
        return self.queue.pop(0).astype(np.float64)


def mpc_policy(**kw):
    return MPCPolicy(**kw)
