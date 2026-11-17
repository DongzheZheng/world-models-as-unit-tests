"""Learned neural dynamics model (ensemble) + MPPI planner for the drone site.  Trained only on the pilot's own black-box
logs (obs + commands).  Public site_map() geometry is used for collision checks; dynamics are learned."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cawm.mpc import MPPI
from reallab.drone.api import site_map
from experiments.drone.baselines.common import _rot, action_to_cmd, RATE_SCALE, HOVER_T, T_SPAN

FD = 18          # state features: pos(3) vel_world(3) R flat(9) ang_vel(3)
AD = 4


def feat(obs: dict) -> np.ndarray:
    R = _rot(obs["quat"])
    return np.concatenate([obs["pos"], obs["lin_vel"], R.reshape(-1), obs["ang_vel"]]).astype(np.float32)


def cmd_to_an(cmd) -> np.ndarray:
    cmd = np.asarray(cmd, dtype=np.float64)
    return np.clip(np.concatenate([cmd[..., :3] / RATE_SCALE, ((cmd[..., 3:] - HOVER_T) / T_SPAN)], -1), -1, 1).astype(np.float32)


# ------------------------------------------------------------------------------------------------ data
def transitions(logs: list[dict], drop_last_if_crash: bool = True):
    """logs: dicts with obs fields (T+1, ...) and 'actions' (T, 4) in command units.  Returns X (N, FD+2AD), Y (N, FD), life id."""
    X, Y, L = [], [], []
    for li, z in enumerate(logs):
        T = len(z["actions"])
        crash = str(z.get("end_cause", "")) == "collision" or bool(z.get("crashed", False))
        F = np.stack([feat({k: z[k][t] for k in ("pos", "lin_vel", "quat", "ang_vel")}) for t in range(T + 1)])
        A = cmd_to_an(z["actions"])
        PA = np.concatenate([np.zeros((1, AD), np.float32), A[:-1]])
        Xi, Yi = np.concatenate([F[:-1], A, PA], 1), F[1:] - F[:-1]
        if drop_last_if_crash and crash and T > 1:      # the impact step is a contact discontinuity; collisions come from the geometry
            Xi, Yi = Xi[:-1], Yi[:-1]
        X.append(Xi); Y.append(Yi); L.append(np.full(len(Xi), li))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(L)


def load_npz(path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in ("pos", "lin_vel", "quat", "ang_vel", "actions")}
    d["end_cause"] = str(z["end_cause"])
    return d


# ------------------------------------------------------------------------------------------------ model
class Ensemble(nn.Module):
    def __init__(self, k=5, hidden=128, din=FD + 2 * AD, dout=FD):
        super().__init__()
        self.k = k
        def mk():
            return nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dout))
        self.nets = nn.ModuleList([mk() for _ in range(k)])
        self.register_buffer("xm", torch.zeros(din)); self.register_buffer("xs", torch.ones(din))
        self.register_buffer("ym", torch.zeros(dout)); self.register_buffer("ys", torch.ones(dout))
        self.register_buffer("xtrain", torch.zeros(1, din - 3))   # normalised training inputs (support set for the novelty penalty)
        self.register_buffer("d_ref", torch.tensor(1.0))

    def member(self, i, x):                       # x raw -> raw delta
        return self.nets[i]((x - self.xm) / self.xs) * self.ys + self.ym

    def forward_all(self, x):
        return torch.stack([self.member(i, x) for i in range(self.k)])

    def sample(self, x, with_unc=False):          # one random member per row
        if with_unc:
            allp = self.forward_all(x)             # (K, B, FD)
            m = torch.randint(self.k, (x.shape[0],), device=x.device)
            out = allp[m, torch.arange(x.shape[0], device=x.device)]
            unc = (allp.std(0) / self.ys).pow(2).mean(-1).sqrt()
            return out, unc
        m = torch.randint(self.k, (x.shape[0],), device=x.device)
        out = torch.zeros(x.shape[0], FD, device=x.device)
        for i in range(self.k):
            idx = (m == i).nonzero(as_tuple=True)[0]
            if len(idx):
                out[idx] = self.member(i, x[idx])
        return out


def fit(X, Y, k=5, hidden=128, epochs=400, lr=2e-3, wd=1e-3, val_frac=0.1, seed=0, log=print, patience=60):
    """Bootstrap ensemble, full-batch AdamW, early stopping per member on a held-out random 10% of the transitions."""
    g = torch.Generator().manual_seed(seed)
    X, Y = torch.as_tensor(X), torch.as_tensor(Y)
    n = len(X)
    perm = torch.randperm(n, generator=g)
    nv = max(1, int(n * val_frac))
    vi, ti = perm[:nv], perm[nv:]
    m = Ensemble(k, hidden)
    m.xm.copy_(X[ti].mean(0)); m.xs.copy_(X[ti].std(0) + 1e-3)
    m.ym.copy_(Y[ti].mean(0)); m.ys.copy_(Y[ti].std(0) + 1e-4)
    Xn, Yn = (X - m.xm) / m.xs, (Y - m.ym) / m.ys
    vals = []
    for i in range(k):
        torch.manual_seed(seed * 100 + i)
        boot = ti[torch.randint(len(ti), (len(ti),), generator=g)]
        net, opt = m.nets[i], torch.optim.AdamW(m.nets[i].parameters(), lr=lr, weight_decay=wd)
        best, bs, bad = 1e9, None, 0
        for ep in range(epochs):
            loss = (net(Xn[boot]) - Yn[boot]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = float((net(Xn[vi]) - Yn[vi]).pow(2).mean())
            if v < best - 1e-5:
                best, bs, bad = v, {a: b.clone() for a, b in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad > patience:
                    break
        net.load_state_dict(bs); vals.append(best)
    with torch.no_grad():                       # support set + reference distance: 95th pct of leave-one-out nearest-neighbour dist
        Z = Xn[:, 3:]
        D = torch.cdist(Z, Z); D.fill_diagonal_(1e9)
        m.xtrain = Z.clone()
        m.d_ref = torch.quantile(D.min(1).values, 0.95)
    log(f"[wm] fit n={n} (val {nv}) member val-mse(normalised) {np.round(vals, 4).tolist()}")
    return m.eval(), dict(n=n, val_mse=vals)


# ------------------------------------------------------------------------------------------------ planner
class Planner:
    """MPPI over the learned ensemble with the native dense reward and known-geometry collision checks."""

    def __init__(self, model: Ensemble, horizon=30, n_samples=384, iters=2, sigma=0.4, temp=0.5, relay_r=0.9, seed=0,
                 margin=0.20, device="cpu", unc_pen=0.0, amax=1.0, tilt_max=0.5, nov_pen=0.5):
        sm = site_map()
        self.model, self.dev = model.to(device).eval(), device
        self.boxes_lo = torch.tensor([[b["center"][i] - b["half_extents"][i] - margin for i in range(3)] for b in sm["boxes"]], device=device)
        self.boxes_hi = torch.tensor([[b["center"][i] + b["half_extents"][i] + margin for i in range(3)] for b in sm["boxes"]], device=device)
        self.relay_r, self.unc_pen, self.nov_pen = relay_r, unc_pen, nov_pen
        self.cos_tilt = None if tilt_max is None else float(np.cos(tilt_max))
        torch.manual_seed(seed)
        self.mppi = MPPI(self.step_fn, [-amax, -amax, -amax, -1], [amax, amax, amax, 1], horizon, n_samples, sigma, temperature=temp, gamma=1.0, iters=iters)
        self.pa = np.zeros(4, np.float32)

    def reset(self):
        self.mppi.reset(); self.pa = np.zeros(4, np.float32)

    @torch.no_grad()
    def step_fn(self, s, a):
        f, pa, k, rl = s["f"], s["pa"], s["k"], s["relays"]
        x = torch.cat([f, a, pa], -1)
        if self.unc_pen > 0:
            dl, unc = self.model.sample(x, True)
        else:
            dl, unc = self.model.sample(x), 0.0
        nf = f + dl
        nov = 0.0
        if self.nov_pen > 0:                     # distance of (state w/o position, action) to the nearest training sample
            z = (x[:, 3:] - self.model.xm[3:]) / self.model.xs[3:]
            nov = self.nov_pen * (torch.cdist(z, self.model.xtrain).min(1).values - self.model.d_ref).clamp(min=0)
        B = f.shape[0]
        ar = torch.arange(B, device=f.device)
        tgt = rl[ar, k.clamp(max=2)]
        d_prev = (f[:, :3] - tgt).norm(dim=-1)
        d = (nf[:, :3] - tgt).norm(dim=-1)
        r = (d_prev - d) - 0.01 - self.unc_pen * unc - nov
        hit = (d < self.relay_r) & (k < 3)
        r = r + 20.0 * hit
        nk = k + hit.long()
        p = nf[:, :3]
        inside = ((p[:, None] > self.boxes_lo) & (p[:, None] < self.boxes_hi)).all(-1).any(-1)
        bad = inside | (p[:, 2] < 0.10) | (p[:, 2] > 6.0) | (p[:, 0] < -2) | (p[:, 0] > 62) | (p[:, 1].abs() > 14)
        if self.cos_tilt is not None:
            bad = bad | (nf[:, 14] < self.cos_tilt)
        win = nk >= 3
        r = r + 50.0 * (win & ~bad) - 50.0 * bad
        done = bad | win
        s["f"], s["pa"], s["k"] = nf, a, nk
        return s, r, done

    def plan(self, obs: dict) -> np.ndarray:
        k = int(round(float(obs["relay_index"][0])))
        st = dict(f=torch.as_tensor(feat(obs))[None], pa=torch.as_tensor(self.pa)[None], k=torch.tensor([k]),
                  relays=torch.as_tensor(np.asarray(obs["relays"], np.float32))[None])
        a = self.mppi.plan(st)[0].numpy()
        self.pa = a.astype(np.float32)
        return action_to_cmd(a)


class WMPolicy:
    """evaluate()-compatible wrapper: loads a saved ensemble blob and plans with MPPI on one CPU thread."""

    def __init__(self, ckpt: str, **kw):
        torch.set_num_threads(1)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        m = Ensemble(blob["k"], blob["hidden"]); m.xtrain = torch.zeros_like(blob["state"]["xtrain"]); m.load_state_dict(blob["state"])
        self.pl = Planner(m.eval(), **kw)

    def reset(self, seed=None):
        self.pl.reset()
        if seed is not None:
            torch.manual_seed(int(seed))

    def act(self, obs):
        return self.pl.plan(obs)


def wm_policy(**kw):
    return WMPolicy(**kw)


def save_model(m: Ensemble, path):
    torch.save(dict(k=m.k, hidden=m.nets[0][0].out_features, state=m.state_dict()), path)
