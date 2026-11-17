"""Tests for the shared core: geometric queries and PPO (run on the compute server)."""
import math

import torch

from cawm import geom
from cawm.ppo import PPOConfig, train_ppo

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _rays(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(1, n, 3, generator=g) * 4 - 2
    d = torch.randn(1, n, 3, generator=g)
    return o, d / d.norm(dim=-1, keepdim=True)


def test_ray_aabb_known_distance():
    o = torch.tensor([[[0.0, 0.0, 1.0]]])
    d = torch.tensor([[[1.0, 0.0, 0.0]]])
    c = torch.tensor([[[5.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    h = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])          # second slot is padding
    dist, idx = geom.ray_aabb(o, d, c, h, 50.0)
    assert abs(float(dist) - 4.0) < 1e-5 and int(idx) == 0
    dist, idx = geom.ray_aabb(o, -d, c, h, 50.0)
    assert float(dist) == 50.0 and int(idx) == -1


def test_ray_hits_agree_with_sdf_by_sphere_tracing():
    """Independent cross-check: marching a ray with the signed-distance function must find the same hit."""
    torch.manual_seed(1)
    K = 6
    c = torch.rand(1, K, 3) * 10 - 5
    h = torch.rand(1, K, 3) * 1.5 + 0.2
    cc = torch.rand(1, K, 3) * 10 - 5
    r = torch.rand(1, K) * 0.8 + 0.2
    hh = torch.full((1, K), 50.0)                                     # tall columns: caps never matter
    o, d = _rays(400, seed=2)
    o = o * 0.1 + torch.tensor([20.0, 0.0, 0.0])                      # start well outside everything
    d = -d.abs() * torch.tensor([1.0, 0.0, 0.0]) + d * torch.tensor([0.0, 0.3, 0.3])
    d = d / d.norm(dim=-1, keepdim=True)
    da, _ = geom.ray_aabb(o, d, c, h, 60.0)
    dc, _ = geom.ray_vcylinder(o, d, cc, r, hh, 60.0)
    ref = torch.minimum(da, dc)
    t = torch.zeros(1, 400)
    for _ in range(3000):
        p = o + t[..., None] * d
        sd = torch.minimum(geom.sdf_aabb(p, c, h), geom.sdf_vcylinder(p, cc, r, hh))
        t = (t + sd.clamp(min=0)).clamp(max=60.0)
    assert float((t - ref).abs().max()) < 2e-2


def test_sdf_gradient_points_away_from_box():
    p = torch.tensor([[[3.0, 0.2, 0.1]]], requires_grad=True)
    c = torch.zeros(1, 1, 3)
    h = torch.ones(1, 1, 3)
    sd = geom.sdf_aabb(p, c, h)
    sd.sum().backward()
    assert abs(float(sd) - 2.0) < 1e-6 and float(p.grad[0, 0, 0]) > 0.99


def test_ray_segments_and_circles_2d():
    o = torch.zeros(1, 2, 2)
    d = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    a = torch.tensor([[[3.0, -1.0], [0.0, 0.0]]])
    b = torch.tensor([[[3.0, 1.0], [0.0, 0.0]]])                      # second segment is padding
    dist, idx = geom.ray_segments_2d(o, d, a, b, 20.0)
    assert abs(float(dist[0, 0]) - 3.0) < 1e-6 and int(idx[0, 0]) == 0 and int(idx[0, 1]) == -1
    dist, _ = geom.ray_circles_2d(o, d, torch.tensor([[[0.0, 5.0]]]), torch.tensor([[1.0]]), 20.0)
    assert float(dist[0, 0]) == 20.0 and abs(float(dist[0, 1]) - 4.0) < 1e-6


def test_fan_and_ground():
    az = torch.linspace(-math.pi / 3, math.pi / 3, 16)
    el = torch.linspace(-0.4, 0.4, 6)
    dirs = geom.yaw_rotate(geom.fan_dirs(az, el), torch.tensor([math.pi / 2]))
    assert dirs.shape == (1, 96, 3) and float((dirs.norm(dim=-1) - 1).abs().max()) < 1e-6
    assert float(dirs[0, :, 1].min()) > 0                              # yawed 90 deg: every ray points toward +y
    o = torch.tensor([[[0.0, 0.0, 2.0]]]).expand(1, 96, 3)
    g = geom.ray_ground(o, dirs, 0.0, 12.0)
    down = dirs[0, :, 2] < -1e-3
    assert torch.allclose(g[0][down], (2.0 / -dirs[0, :, 2][down]).clamp(max=12.0), atol=1e-5)
    assert bool((g[0][~down] == 12.0).all())


class _Reach:
    """Toy batched world: drive a point to the origin.  Used to check PPO end to end."""

    def __init__(self, n, discrete, device):
        self.num_envs, self.obs_dim, self.device = n, 2, device
        self.n_actions = 5 if discrete else None
        self.act_dim = None if discrete else 2
        self._moves = torch.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float32, device=device)

    def _spawn(self, k):
        return torch.rand(k, 2, device=self.device) * 4 - 2

    def reset(self):
        self.p = self._spawn(self.num_envs)
        self.t = torch.zeros(self.num_envs, device=self.device)
        return self.p.clone()

    def step(self, a):
        v = self._moves[a] if self.n_actions else a.clamp(-1, 1)
        self.p = self.p + 0.1 * v
        self.t += 1
        dist = self.p.norm(dim=-1)
        reward = -dist * 0.1
        time_out = self.t >= 50
        info = {"time_out": time_out, "terminal_obs": self.p.clone()}
        if time_out.any():
            self.p[time_out] = self._spawn(int(time_out.sum()))
            self.t[time_out] = 0
        return self.p.clone(), reward, time_out, info


def _final_dist(policy, env):
    obs = env.reset()
    for _ in range(49):
        obs, *_ = env.step(policy.act(obs, deterministic=True))
    return float(obs.norm(dim=-1).mean())


def test_ppo_learns_continuous_and_discrete():
    for discrete in (False, True):
        env = _Reach(2048, discrete, DEV)
        policy, hist = train_ppo(env, PPOConfig(total_steps=3_000_000, rollout_len=25, log_every=5), log=lambda s: None)
        d = _final_dist(policy, _Reach(512, discrete, DEV))
        assert d < 0.3, f"discrete={discrete}: mean final distance {d:.3f} (random start is ~1.5)"
        assert hist[-1]["sps"] > 0
