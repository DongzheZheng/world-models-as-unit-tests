"""Hand-written candidate controller, written and tuned ONLY inside the code world (never against reality).

It acts on the shared observation features (worlds.fps.take_cover_world.encode), so it runs unchanged on code-world
observations (batched) and on real observations (via encode_obs_dict).  From the features it reconstructs the known
fireballs (time-to-arrival, current offset, vy -> full linear trajectory, |v| = BALL_SPEED), then rolls the player's
code-world dynamics forward for a small set of action plans ("a for 1 step, then c held") and picks the plan whose
first predicted contact (inflated by `margin`) is latest; ties broken by a pull towards the room centre.
usage (tuning grid in the code world): python experiments/doom/code_wm/planner.py
"""
import itertools
import json
import sys
from pathlib import Path

import torch

from worlds.fps import take_cover_world as W

PLANS = [(a, c, k) for a in range(3) for c in range(3) for k in (1, 3)]   # a for k steps, then c


def plan_act(obs, horizon=48, margin=6.0, w_center=0.3):
    B, dev = obs.shape[0], obs.device
    py0 = obs[:, 0] * 368.0 + W.PY0
    vy0 = obs[:, 1] * 7.27
    f = obs[:, 3:3 + 5 * W.K_BALLS].reshape(B, W.K_BALLS, 5)
    valid = f[..., 0] > 0.5
    tau = f[..., 1] * 40.0
    dy_now = f[..., 3] * 256.0 + py0[:, None]     # absolute ball y now
    bvy = f[..., 4] * 10.0
    bvx = -(W.BALL_SPEED ** 2 - bvy ** 2).clamp(min=1.0).sqrt()
    bx0 = W.PX - bvx * tau
    dirs = torch.tensor([0.0, 1.0, -1.0], device=dev)
    P = len(PLANS)
    a_seq = torch.tensor([[a if t < k * W.TICS_PER_STEP else c for t in range(horizon)] for a, c, k in PLANS], device=dev)
    d = dirs[a_seq]                                   # (P, H)
    py = py0[:, None].expand(B, P).clone()
    vy = vy0[:, None].expand(B, P).clone()
    first_hit = torch.full((B, P), float(horizon), device=dev)
    lim = W.HIT_HALF + margin
    for t in range(horizon):
        vy = vy + W.ACCEL * d[None, :, t]
        py = py + vy
        wall = (py < W.Y_LO) | (py > W.Y_HI)
        py = py.clamp(W.Y_LO, W.Y_HI)
        vy = torch.where(wall, torch.zeros_like(vy), vy) * W.FRICTION
        bx = bx0 + bvx * (t + 1)
        by = dy_now + bvy * (t + 1)
        inb = valid & (bx > W.BALL_X_MIN)
        hit = (inb[:, None] & ((bx - W.PX).abs()[:, None] < lim) &
               ((by[:, None] - py[..., None]).abs() < lim)).any(-1)
        first_hit = torch.where(hit & (first_hit >= horizon), torch.full_like(first_hit, float(t)), first_hit)
    score = first_hit - w_center * (py - W.PY0).abs() / 100.0
    best = score.argmax(1)
    return torch.tensor([p[0] for p in PLANS], device=dev)[best]


class Planner:
    def __init__(self, horizon=48, margin=6.0, w_center=0.3):
        self.kw = dict(horizon=horizon, margin=margin, w_center=w_center)
        self.mem = None

    def reset(self):
        self.mem = None

    def act(self, obs):
        o, self.mem = W.encode_obs_dict(obs, self.mem)
        return int(plan_act(torch.as_tensor(o)[None], **self.kw)[0])


def planner(**kw):
    return Planner(**kw)


if __name__ == "__main__":
    from experiments.doom.code_wm.train import world_eval
    res = []
    for h, m, w in itertools.product((32, 48, 64), (2.0, 6.0, 12.0), (0.0, 0.3, 1.0)):
        r = world_eval(lambda o, s, g: plan_act(o, h, m, w), n=2048)
        res.append(dict(horizon=h, margin=m, w_center=w, **r))
        print(res[-1], flush=True)
    best = max(res, key=lambda r: r["mean"])
    Path("runs/doom/code_wm/planner_tuning.json").write_text(json.dumps(dict(best=best, grid=res), indent=1))
    print("BEST", best)
