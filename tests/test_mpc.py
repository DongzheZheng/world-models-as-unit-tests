"""Tests for the sampling-based MPC utilities and the results writer (toy functional world)."""
import json

import torch

from cawm import mpc, results

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOVES = torch.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float32)


def step_discrete(state, a):
    p = state["p"] + 0.1 * MOVES.to(a.device)[a]
    return {"p": p}, -p.norm(dim=-1), torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)


def step_continuous(state, a):
    v = state["v"] + 0.2 * a                      # double integrator: needs look-ahead to brake in time
    p = state["p"] + 0.1 * v
    return {"p": p, "v": v}, -(p.norm(dim=-1) + 0.1 * v.norm(dim=-1)), torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)


def test_repeat_state_keeps_env_copies_contiguous():
    s = mpc.repeat_state({"p": torch.arange(3.0)[:, None], "name": "x"}, 4)
    assert s["p"][:, 0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] and s["name"] == "x"


def test_plan_discrete_reaches_origin():
    g = torch.Generator(device=DEV).manual_seed(0)
    state = {"p": torch.tensor([[1.5, -1.0], [-2.0, 0.7]], device=DEV)}
    for _ in range(60):
        a, _, _ = mpc.plan_discrete(step_discrete, state, n_actions=5, horizon=8, n_samples=256, hold=2, generator=g)
        state, _, _ = step_discrete(state, a)
    assert float(state["p"].norm(dim=-1).max()) < 0.2


def test_mppi_stabilises_double_integrator():
    torch.manual_seed(0)
    planner = mpc.MPPI(step_continuous, [-1, -1], [1, 1], horizon=20, n_samples=512, sigma=0.5, temperature=0.05, iters=2)
    state = {"p": torch.tensor([[2.0, -1.5]], device=DEV), "v": torch.zeros(1, 2, device=DEV)}
    for _ in range(120):
        state, _, _ = step_continuous(state, planner.plan(state))
    assert float(state["p"].norm()) < 0.15 and float(state["v"].norm()) < 0.3


def test_results_roundtrip(tmp_path):
    eps = [dict(seed=1000 + i, ret=float(i), success=i % 2 == 0) for i in range(10)]
    results.record("toy", "s", "c1", "code_as_wm", "code_world", eps, "ret", real_steps_calibration=123,
                   other_metrics=("success",), runs=tmp_path)
    recs = results.collect(tmp_path)
    assert len(recs) == 1 and recs[0]["eval"]["mean"] == 4.5 and recs[0]["other"]["success"]["mean"] == 0.5
    assert "c1" in results.table(recs)
    json.dumps(recs)
