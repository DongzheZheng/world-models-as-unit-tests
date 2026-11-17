"""Sampling-based MPC that uses a code world as its predictive model.

A code world is *functional*: its state is a dict of tensors with a leading batch dimension and
    step_fn(state, action) -> (next_state, reward (B,), done (B,))
never mutates its input.  That makes planning trivial: fork the current state N x M times
(N candidate action sequences, M samples of the world's own randomness), roll all of them forward
on the GPU, and act on the best first action.  Nothing here is specific to a demo.
"""
from __future__ import annotations

import torch


def repeat_state(state: dict, n: int) -> dict:
    """Fork every environment n times: tensors (B, ...) -> (B*n, ...), copies of env i are contiguous."""
    return {k: (v.repeat_interleave(n, dim=0) if torch.is_tensor(v) else v) for k, v in state.items()}


def discrete_sequences(n_actions: int, batch: int, n_samples: int, horizon: int, hold: int = 1,
                       device="cpu", generator=None, include_constant: bool = True) -> torch.Tensor:
    """Random discrete action sequences (batch, n_samples, horizon), each action held `hold` steps.

    With include_constant the first n_actions candidates are the constant sequences ("always a"),
    which gives the planner a sane fallback for every action.
    """
    chunks = -(-horizon // hold)
    seq = torch.randint(n_actions, (batch, n_samples, chunks), device=device, generator=generator)
    seq = seq.repeat_interleave(hold, dim=2)[:, :, :horizon]
    if include_constant and n_samples >= n_actions:
        seq[:, :n_actions] = torch.arange(n_actions, device=device)[None, :, None]
    return seq


@torch.no_grad()
def evaluate_sequences(step_fn, state: dict, actions: torch.Tensor, gamma: float = 1.0, n_noise: int = 1,
                       terminal_value=None) -> torch.Tensor:
    """Score candidate action sequences by rolling the code world forward.

    state: dict of (B, ...) tensors.  actions: (B, N, H) long or (B, N, H, A) float.
    Returns scores (B, N): discounted return averaged over n_noise replicas of the world's randomness.
    Reward stops accumulating after an environment reports done.
    terminal_value(state) -> (B*N*M,) may add a value estimate for states still alive at the horizon.
    """
    B, N, H = actions.shape[:3]
    M = n_noise
    s = repeat_state(state, N * M)
    a = actions.repeat_interleave(M, dim=1).reshape(B * N * M, H, *actions.shape[3:])
    ret = torch.zeros(B * N * M, device=actions.device)
    alive = torch.ones(B * N * M, device=actions.device)
    disc = 1.0
    for t in range(H):
        s, r, done = step_fn(s, a[:, t])
        ret += disc * alive * r
        alive = alive * (1.0 - done.float())
        disc *= gamma
    if terminal_value is not None:
        ret += disc * alive * terminal_value(s)
    return ret.reshape(B, N, M).mean(-1)


@torch.no_grad()
def plan_discrete(step_fn, state: dict, n_actions: int, horizon: int, n_samples: int, gamma: float = 1.0,
                  n_noise: int = 1, hold: int = 1, terminal_value=None, generator=None):
    """Random-shooting MPC for discrete actions.  Returns (first_action (B,), best_score (B,), scores (B,N))."""
    B = next(v for v in state.values() if torch.is_tensor(v)).shape[0]
    dev = next(v for v in state.values() if torch.is_tensor(v)).device
    seqs = discrete_sequences(n_actions, B, n_samples, horizon, hold, dev, generator)
    scores = evaluate_sequences(step_fn, state, seqs, gamma, n_noise, terminal_value)
    best = scores.argmax(-1)
    return seqs[torch.arange(B, device=dev), best, 0], scores.max(-1).values, scores


class MPPI:
    """Model-predictive path integral control for continuous actions, with warm start.

    plan(state) refines a nominal action sequence (B, H, A) kept between calls and returns the first action.
    """

    def __init__(self, step_fn, act_low, act_high, horizon: int, n_samples: int, sigma, temperature: float = 1.0,
                 gamma: float = 1.0, iters: int = 1, n_noise: int = 1, terminal_value=None, init_action=None):
        self.step_fn, self.h, self.n = step_fn, horizon, n_samples
        self.low, self.high = torch.as_tensor(act_low).float(), torch.as_tensor(act_high).float()
        self.sigma = torch.as_tensor(sigma).float().expand_as(self.low).clone()
        self.temp, self.gamma, self.iters, self.n_noise = temperature, gamma, iters, n_noise
        self.terminal_value = terminal_value
        self.init_action = None if init_action is None else torch.as_tensor(init_action).float()
        self.nominal = None

    def reset(self) -> None:
        self.nominal = None

    @torch.no_grad()
    def plan(self, state: dict) -> torch.Tensor:
        ref = next(v for v in state.values() if torch.is_tensor(v))
        B, dev = ref.shape[0], ref.device
        low, high, sigma = self.low.to(dev), self.high.to(dev), self.sigma.to(dev)
        if self.nominal is None or self.nominal.shape[0] != B:
            init = (low + high) / 2 if self.init_action is None else self.init_action.to(dev)
            self.nominal = init.expand(B, self.h, -1).clone()
        for _ in range(self.iters):
            eps = torch.randn(B, self.n, self.h, low.numel(), device=dev) * sigma
            eps[:, 0] = 0                                               # always keep the nominal plan as a candidate
            cand = (self.nominal[:, None] + eps).clamp(low, high)
            scores = evaluate_sequences(self.step_fn, state, cand, self.gamma, self.n_noise, self.terminal_value)
            w = torch.softmax((scores - scores.max(-1, keepdim=True).values) / self.temp, dim=-1)
            self.nominal = (w[:, :, None, None] * cand).sum(1)
        action = self.nominal[:, 0].clone()
        self.nominal = torch.cat([self.nominal[:, 1:], self.nominal[:, -1:]], dim=1)   # shift for the next call
        return action
