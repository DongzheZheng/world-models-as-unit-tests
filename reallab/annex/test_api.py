"""Plumbing tests for the annex lab (purpose='smoke'; temporary ledger and black-box directories)."""
import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest

from cawm import ledger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OBS_KEYS = {"points", "tool_pos", "tool_vel", "post", "zone", "target", "t"}
LOG_SCALARS = {"seed", "life", "pilot", "purpose", "success", "final_tip_dist", "n_steps",
               "terminated", "truncated", "control_hz"}
INFO_KEYS = {"success", "at_rest", "tip_dist", "n_steps"}


@pytest.fixture(autouse=True)
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAWM_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("CAWM_BLACKBOX_DIR", str(tmp_path / "blackbox"))
    monkeypatch.setattr(ledger, "DEFAULT_PATH", tmp_path / "ledger.jsonl")
    return tmp_path


def _api():
    from reallab.annex import api
    return api


def _run(env, seed, n=15, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    obs = env.reset(seed)
    traj = [obs]
    for _ in range(n):
        obs, r, te, tr, info = env.step(obs["tool_pos"] + rng.uniform(-0.05, 0.05, 2))
        traj.append(obs)
        if te or tr:
            break
    return traj


def test_obs_schema_and_card(tmp_env):
    api = _api()
    env = api.make("smoke")
    obs = env.reset(seed=7)
    assert set(obs) == OBS_KEYS
    assert obs["points"].shape == (20, 3) and obs["tool_pos"].shape == (2,) and obs["tool_vel"].shape == (2,)
    assert obs["post"].shape == (2,) and obs["zone"].shape == (2, 2)
    assert obs["target"].shape == (2,) and obs["t"] == 0
    assert np.allclose(obs["points"][-1, :2], obs["tool_pos"], atol=5e-3)          # point 20 moves with the tool
    arc = np.linalg.norm(np.diff(obs["points"], axis=0), axis=1).sum()
    assert 0.65 < arc <= 0.701
    assert env.control_hz == 50.0 and np.allclose(env.action_low, [0.30, -0.2]) and np.allclose(env.action_high, [0.55, 0.2])
    z = obs["zone"]
    assert not (z[0, 0] <= obs["post"][0] <= z[1, 0] and z[0, 1] <= obs["post"][1] <= z[1, 1])
    obs, r, te, tr, info = env.step(np.array([5.0, 5.0]))            # clipped to the zone
    assert obs["t"] == 1 and not te and not tr and isinstance(r, float)
    assert set(info) == INFO_KEYS
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 0.0]))
    card = api.task_card()
    assert card["time_limit_steps"] == api.TIME_LIMIT and card["success"]["target_radius"] == api.TARGET_RADIUS
    assert "load" not in json.dumps(card) and "force_reading" not in json.dumps(card)
    env.close()


def test_tool_stays_in_zone_and_noop_times_out(tmp_env):
    api = _api()
    env = api.make("smoke")
    obs = env.reset(seed=3)
    done = False
    while not done:
        obs, r, te, tr, info = env.step(np.array([9.0, -9.0]))
        assert np.all(obs["tool_pos"] >= api.ZONE[0] - 0.01) and np.all(obs["tool_pos"] <= api.ZONE[1] + 0.01)
        done = te or tr
    assert tr and not info["success"] and obs["t"] == api.TIME_LIMIT
    env.close()


def test_determinism(tmp_env):
    api = _api()
    a, b = api.make("smoke"), api.make("smoke")
    ta, tb = _run(a, 11), _run(b, 11)
    for oa, ob in zip(ta, tb):
        for k in oa:
            assert np.array_equal(oa[k], ob[k])
    tc = _run(b, 12)
    assert not np.array_equal(ta[0]["post"], tc[0]["post"])
    a.close(); b.close()


def test_rules_hand_made():
    api = _api()
    pts = np.zeros((20, 3))
    pts[:, :2] = np.stack([np.linspace(0.9, 0.5, 20), np.full(20, 0.0)], 1)
    obs = dict(points=pts, tool_pos=np.array([0.5, 0.0]), post=np.array([0.75, 0.1]), target=np.array([0.95, 0.0]))
    assert api.goal_met(obs, obs)                                       # at rest, far end within 0.10 m
    assert not api.goal_met(dict(obs, target=np.array([1.01, 0.0])), dict(obs, target=np.array([1.01, 0.0])))
    moved = dict(obs, points=pts + np.array([0.001, 0.0, 0.0]))
    assert not api.goal_met(moved, obs)                                  # 0.05 m/s > rest speed
    slow = dict(obs, points=pts + np.array([0.0003, 0.0, 0.0]))
    assert api.goal_met(slow, obs)                                       # 0.015 m/s <= rest speed
    out = dict(obs, tool_pos=np.array([0.2, 0.0]))
    assert not api.goal_met(out, out)


def test_lab_rules_match_api(tmp_env):
    """The lab's fast simulator must reproduce the public episode exactly (readings and success step)."""
    api = _api()
    from reallab.annex._sim import Sim
    sim = Sim()
    ep = sim.setup(5)
    env = api.make("smoke")
    obs = env.reset(5)
    assert np.array_equal(obs["points"], sim.pts0) and np.array_equal(obs["post"], ep["post"])
    rng = np.random.default_rng(3)
    acts = rng.uniform(api.ZONE[0], api.ZONE[1], (30, 2))
    r = sim.rollout(acts, 30, stop_when=(), record=True)
    for k in range(30):
        obs, *_ = env.step(acts[k])
        assert np.array_equal(obs["points"], r["rec"][k][0])
    env.close()


def test_lives_cap_blackbox_and_ledger(tmp_env):
    api = _api()
    with pytest.raises(ValueError):
        api.make("calibration")
    with pytest.raises(ValueError):
        api.make("observation")
    env = api.make("calibration", pilot="tester")
    for k in range(5):
        _run(env, k, n=10)
    with pytest.raises(ledger.BudgetExceeded):
        env.reset(seed=5)
    env.close()
    for k in range(1, 6):
        assert (tmp_env / "blackbox" / "tester" / f"life_{k}.npz").exists()
    d = np.load(tmp_env / "blackbox" / "tester" / "life_2.npz")
    assert d["points"].shape == (11, 20, 3) and d["actions"].shape == (10, 2) and int(d["life"]) == 2
    assert int(d["seed"]) == 1 and str(d["pilot"]) == "tester" and d["rewards"].shape == (10,)
    assert set(d.files) == OBS_KEYS | LOG_SCALARS | {"actions", "rewards"}
    t = ledger.totals(tmp_env / "ledger.jsonl")[("annex", "annex:tester", "calibration")]
    assert t == {"steps": 50, "episodes": 5}
    env2 = api.make("calibration", pilot="tester")
    with pytest.raises(ledger.BudgetExceeded):
        env2.reset(seed=1)
    env2.close()


def test_abandoned_episode_counts(tmp_env):
    api = _api()
    env = api.make("calibration", pilot="p")
    o = env.reset(seed=1); env.step(o["tool_pos"])
    env.reset(seed=2)
    env.close()
    t = ledger.totals(tmp_env / "ledger.jsonl")[("annex", "annex:p", "calibration")]
    assert t == {"steps": 1, "episodes": 2}
    assert (tmp_env / "blackbox" / "p" / "life_1.npz").exists()


def test_seed_ranges(tmp_env):
    api = _api()
    ev = api.make("eval")
    with pytest.raises(ValueError):
        ev.reset(seed=5)
    ev.reset(seed=1000); ev.reset()
    ev.close()
    cal = api.make("calibration", pilot="s")
    with pytest.raises(ValueError):
        cal.reset(seed=1000)
    cal.reset(seed=999)
    cal.close()


def test_evaluate_four_workers_and_no_lab_hooks(tmp_env):
    import inspect
    from reallab.annex import parallel
    from reallab.annex.parallel import evaluate
    assert set(inspect.signature(evaluate).parameters) == {"policy_factory", "policy_kwargs", "seeds", "purpose",
                                                           "pilot", "tag", "n_workers", "log_dir"}
    seeds = [1000, 1001, 1002, 1003]
    res = evaluate(":random_policy", {"seed": 1}, seeds, purpose="smoke", n_workers=4, log_dir=str(tmp_env / "logs"))
    assert [r["seed"] for r in res] == seeds
    for r in res:
        assert set(r) == {"seed", "ret", "n_steps", "success", "tip_dist"}
        assert r["n_steps"] >= 1
    d = np.load(tmp_env / "logs" / "ep_1003.npz")
    assert OBS_KEYS <= set(d.files)
    assert ledger.totals(tmp_env / "ledger.jsonl")[("annex", "annex", "smoke")]["episodes"] == 4
    res2 = evaluate(":random_policy", {"seed": 1}, seeds[:2], purpose="smoke", n_workers=2)
    assert [r["ret"] for r in res2] == [r["ret"] for r in res[:2]]
    assert parallel.MAX_WORKERS == 12


def test_lab_gif_and_render_log(tmp_env):
    api = _api()
    from reallab.annex import _render
    from PIL import Image
    env = api.make("calibration", pilot="r")
    _run(env, 4, n=30)
    env.close()
    out = _render.render_log(str(tmp_env / "blackbox" / "r" / "life_1.npz"), str(tmp_env / "life.gif"), "a title")
    im = Image.open(out)
    assert im.n_frames >= 15 and im.size[0] >= 480
    from reallab.annex.parallel import evaluate
    evaluate(":noop_policy", None, [1000], purpose="smoke", n_workers=1, log_dir=str(tmp_env / "logs"))
    out = _render.render_log(str(tmp_env / "logs" / "ep_1000.npz"), str(tmp_env / "ep.gif"), "eval step log")
    assert Image.open(out).n_frames >= 20


# ------------------------------------------------------------------------------------------------ disclosure
_WORDS = base64.b64decode(
    b"Y2FibGUgcm9wZSB3aXJlIGNvcmQgc3RyaW5nIGNoYWluIHRldGhlciBkZWZvcm1hYmxlIGZsZXhpYmxlIGVsYXN0aWMgc29mdCBzdGlmZm5lc3Mg"
    b"YmVuZGluZyBmcmljdGlvbiBjYXBzdGFuIHBlZyBjb25uZWN0b3IgY2xpcCBhbmNob3IgZ3JpcHBlciBtdWpvY28gcGx1Z2luIHJvdXRpbmcgaGFy"
    b"bmVzcyBjb21wb3NpdGUgZGF0YXNoZWV0").decode().split() + base64.b64decode(
    b"d2hpcCBsYXNzbyBmbGluZyBjYXN0IHRocm93IGh1cmwgdG9zcyBsYXNoIGZsZXggcm9kIHlhcm4gaG9zZSB0ZXRoZXJiYWxsIHN0YWtlIHBvbGUg"
    b"c2xpbmcgbm9vc2Uga25vdCBzbmFrZSB3ZWlnaHQgbWFzcyBkYW1waW5nIGRyYWcgaW5lcnRpYSBwZW5kdWx1bQ==").decode().split()
assert len(_WORDS) == 51
_RX = re.compile(r"(?i)\b(" + "|".join(_WORDS) + ")")


def _hits(text):
    return sorted(set(m.group(0).lower() for m in _RX.finditer(text)))


def test_no_revealing_words_in_public_files_and_names():
    for f in ("api.py", "parallel.py", "__init__.py"):
        assert _hits((HERE / f).read_text()) == [], f
    for p in HERE.rglob("*"):
        if "__pycache__" not in p.parts:
            assert not any(w in p.name.lower() for w in _WORDS), p.name
    obs_dir = ROOT / "runs" / "annex" / "observation"
    if obs_dir.exists():
        for p in obs_dir.rglob("*"):
            assert not any(w in p.name.lower() for w in _WORDS), p.name
            if p.suffix == ".npz":
                d = np.load(p)
                assert _hits(" ".join(d.files)) == []
    from reallab.annex import api
    assert _hits(json.dumps(api.task_card())) == []
    env = api.make("smoke")
    o = env.reset(1)
    assert _hits(" ".join(o)) == []
    env.close()


def test_the_scan_works():
    assert _hits("the Gripper holds a cable") == ["cable", "gripper"] and _hits("a recorded scene") == []
    assert _hits("a Whip and a lasso") == ["lasso", "whip"]
