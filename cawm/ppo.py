"""PPO for torch-native batched code worlds.

A code world steps thousands of environments at once as tensors on the GPU, so there is no
Python-level vectorisation and no CPU<->GPU copying: the rollout loop below is the whole pipeline.

Environment protocol (duck-typed):
    env.num_envs, env.obs_dim, env.device
    env.act_dim (continuous, actions clipped by the env)   OR   env.n_actions (discrete)
    env.reset() -> obs (B, obs_dim)
    env.step(action) -> obs, reward (B,), done (B,), info
        Finished environments must be auto-reset inside step(); obs is then the first observation
        of the new episode.  Optional info keys:
          "time_out"      (B,) bool   episode ended by a time limit (not a true terminal state)
          "terminal_obs"  (B, obs_dim) observation before the auto-reset (valid where done)
        When both are present the value of the truncated state is bootstrapped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    total_steps: int = 5_000_000        # environment steps summed over all envs
    rollout_len: int = 64
    epochs: int = 4
    minibatches: int = 8
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    anneal_lr: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: tuple = (256, 256)
    norm_obs: bool = True
    init_log_std: float = -0.5
    target_kl: float | None = None
    log_every: int = 10                 # iterations
    seed: int = 0
    extra: dict = field(default_factory=dict)


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + b
        delta = m - self.mean
        self.mean += delta * b / tot
        self.var.copy_((self.var * self.count + v * b + delta ** 2 * self.count * b / tot) / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var.sqrt() + 1e-8)).clamp(-self.clip, self.clip)


def mlp(sizes, out_gain: float) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, out_gain if last else 2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if not last:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """MLP actor-critic.  Discrete (Categorical) if n_actions is given, else diagonal Gaussian."""

    def __init__(self, obs_dim: int, act_dim: int | None = None, n_actions: int | None = None,
                 hidden=(256, 256), norm_obs: bool = True, init_log_std: float = -0.5):
        super().__init__()
        assert (act_dim is None) != (n_actions is None), "give exactly one of act_dim / n_actions"
        self.obs_dim, self.act_dim, self.n_actions = obs_dim, act_dim, n_actions
        self.hidden, self.norm_obs = tuple(hidden), norm_obs
        self.norm = RunningNorm(obs_dim) if norm_obs else nn.Identity()
        out = n_actions if n_actions is not None else act_dim
        self.pi = mlp([obs_dim, *hidden, out], 0.01)
        self.v = mlp([obs_dim, *hidden, 1], 1.0)
        if n_actions is None:
            self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs: torch.Tensor):
        out = self.pi(self.norm(obs))
        if self.n_actions is not None:
            return torch.distributions.Categorical(logits=out)
        return torch.distributions.Normal(out, self.log_std.exp().expand_as(out))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(self.norm(obs)).squeeze(-1)

    def logp(self, dist, action: torch.Tensor) -> torch.Tensor:
        lp = dist.log_prob(action)
        return lp if self.n_actions is not None else lp.sum(-1)

    def entropy(self, dist) -> torch.Tensor:
        e = dist.entropy()
        return e if self.n_actions is not None else e.sum(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        d = self.dist(obs)
        if deterministic:
            return d.probs.argmax(-1) if self.n_actions is not None else d.mean
        return d.sample()

    # ---- persistence (so a policy trained in the code world can be loaded by a real-world worker)
    def export(self) -> dict:
        return dict(obs_dim=self.obs_dim, act_dim=self.act_dim, n_actions=self.n_actions,
                    hidden=self.hidden, norm_obs=self.norm_obs,
                    state={k: v.detach().cpu() for k, v in self.state_dict().items()})

    @staticmethod
    def load(blob: dict, device="cpu") -> "ActorCritic":
        m = ActorCritic(blob["obs_dim"], blob["act_dim"], blob["n_actions"], blob["hidden"], blob["norm_obs"])
        m.load_state_dict(blob["state"])
        return m.to(device).eval()


def train_ppo(env, cfg: PPOConfig, log=print, callback=None, policy: ActorCritic | None = None):
    """Train a policy in a batched code world.  Returns (policy, history list of dicts)."""
    torch.manual_seed(cfg.seed)
    dev, B, T = env.device, env.num_envs, cfg.rollout_len
    discrete = hasattr(env, "n_actions") and env.n_actions is not None
    if policy is None:
        policy = ActorCritic(env.obs_dim, None if discrete else env.act_dim, env.n_actions if discrete else None,
                             cfg.hidden, cfg.norm_obs, cfg.init_log_std).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    iters = max(1, int(cfg.total_steps) // (B * T))

    obs = env.reset()
    ep_ret = torch.zeros(B, device=dev)
    ep_len = torch.zeros(B, device=dev)
    done_rets, done_lens = [], []
    history, t0, steps = [], time.time(), 0

    obs_buf = torch.zeros(T, B, env.obs_dim, device=dev)
    act_buf = torch.zeros((T, B) if discrete else (T, B, env.act_dim), device=dev,
                          dtype=torch.long if discrete else torch.float32)
    logp_buf, rew_buf, done_buf, val_buf = (torch.zeros(T, B, device=dev) for _ in range(4))

    for it in range(iters):
        if cfg.anneal_lr:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1 - it / iters)
        # ---------------------------------------------------------------- rollout
        with torch.no_grad():
            for t in range(T):
                if cfg.norm_obs:
                    policy.norm.update(obs)
                d = policy.dist(obs)
                a = d.sample()
                obs_buf[t], act_buf[t] = obs, a
                logp_buf[t], val_buf[t] = policy.logp(d, a), policy.value(obs)
                obs, r, done, info = env.step(a)
                r = r.clone()
                if "time_out" in info and "terminal_obs" in info:
                    to = info["time_out"] & done
                    if to.any():
                        r[to] += cfg.gamma * policy.value(info["terminal_obs"][to])
                rew_buf[t], done_buf[t] = r, done.float()
                ep_ret += r
                ep_len += 1
                if done.any():
                    done_rets.append(ep_ret[done].clone())
                    done_lens.append(ep_len[done].clone())
                    ep_ret[done] = 0
                    ep_len[done] = 0
            last_val = policy.value(obs)
            # ------------------------------------------------------------ GAE
            adv = torch.zeros(T, B, device=dev)
            gae = torch.zeros(B, device=dev)
            for t in reversed(range(T)):
                nv = last_val if t == T - 1 else val_buf[t + 1]
                nonterm = 1.0 - done_buf[t]
                delta = rew_buf[t] + cfg.gamma * nv * nonterm - val_buf[t]
                gae = delta + cfg.gamma * cfg.lam * nonterm * gae
                adv[t] = gae
            ret = adv + val_buf
        steps += T * B
        # ---------------------------------------------------------------- update
        fo = obs_buf.reshape(T * B, -1)
        fa = act_buf.reshape(T * B) if discrete else act_buf.reshape(T * B, -1)
        flp, fadv, fret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        n = T * B
        mb = n // cfg.minibatches
        kl = 0.0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(cfg.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                d = policy.dist(fo[idx])
                lp = policy.logp(d, fa[idx])
                ratio = (lp - flp[idx]).exp()
                a_ = fadv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = -torch.min(ratio * a_, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_).mean()
                vl = 0.5 * (policy.value(fo[idx]) - fret[idx]).pow(2).mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * policy.entropy(d).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt.step()
                kl = float((flp[idx] - lp).mean())
            if cfg.target_kl is not None and kl > cfg.target_kl:
                break
        # ---------------------------------------------------------------- logging
        if (it + 1) % cfg.log_every == 0 or it == iters - 1:
            if done_rets:
                rr, ll = torch.cat(done_rets), torch.cat(done_lens)
                mr, ml, ne = float(rr.mean()), float(ll.mean()), int(rr.numel())
            else:
                mr, ml, ne = float("nan"), float("nan"), 0
            done_rets, done_lens = [], []
            el = time.time() - t0
            rec = dict(iter=it + 1, steps=steps, wall_s=round(el, 1), sps=int(steps / max(el, 1e-9)),
                       ep_ret=mr, ep_len=ml, episodes=ne, kl=kl)
            history.append(rec)
            log(f"[ppo] it {it + 1:5d}/{iters} steps {steps:.3g} sps {rec['sps']:,} "
                f"ep_ret {mr:9.3f} ep_len {ml:7.1f} kl {kl:.4f} wall {el:6.1f}s")
            if callback is not None and callback(policy, rec) is False:
                break
    return policy, history
