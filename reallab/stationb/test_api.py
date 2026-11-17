"""Plumbing tests for station B (purpose='smoke'; temporary ledger and black-box directory)."""
import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest

from cawm import ledger

HERE = Path(__file__).resolve().parent
OBS_KEYS = {"points", "tool_pos", "tool_vel", "posts", "post_sides", "sides_now", "target", "load", "t"}
LOG_SCALARS = {"seed", "life", "pilot", "purpose", "success", "sides_ok", "final_dist", "max_load", "overload",
               "n_steps", "terminated", "truncated", "control_hz", "time_limit"}


@pytest.fixture(autouse=True)
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAWM_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("CAWM_BLACKBOX_DIR", str(tmp_path / "blackbox"))
    monkeypatch.setattr(ledger, "DEFAULT_PATH", tmp_path / "ledger.jsonl")
    return tmp_path


def _api():
    from reallab.stationb import api
    return api


def _run(env, seed, n=15, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    obs = env.reset(seed)
    traj = [obs]
    for _ in range(n):
        obs, r, te, tr, info = env.step(obs["tool_pos"] + rng.uniform(-0.03, 0.03, 2))
        traj.append(obs)
        if te or tr:
            break
    return traj


def _eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def test_schema_no_load_in_loop_and_reward(tmp_env):
    api = _api()
    env = api.make("smoke")
    obs = env.reset(seed=7)
    assert set(obs) == OBS_KEYS and np.isnan(obs["load"]) and obs["t"] == 0
    assert obs["points"].shape == (20, 3) and obs["posts"].shape == (3, 2)
    prev = obs
    for k in range(20):
        obs, r, te, tr, info = env.step(prev["tool_pos"] + np.array([0.01, 0.0]))
        assert np.isnan(obs["load"])
        assert set(info) == {"success", "sides_ok", "final_dist", "n_steps"}       # nothing load-derived
        d0 = np.linalg.norm(prev["tool_pos"] - prev["target"]); d1 = np.linalg.norm(obs["tool_pos"] - obs["target"])
        n0 = np.sum(prev["sides_now"] == prev["post_sides"]); n1 = np.sum(obs["sides_now"] == obs["post_sides"])
        assert r == pytest.approx(10 * (d0 - d1) + (n1 - n0), abs=1e-12)
        prev = obs
    card = api.task_card()
    assert card["time_limit_steps"] == api.TIME_LIMIT and card["load_limit"] == 10.0
    env.close()


def test_terminal_info_has_load_values(tmp_env):
    api = _api()
    env = api.RealStationB("smoke", "", "", None, _lab=True, _time_limit=5)
    obs = env.reset(3)
    for _ in range(5):
        obs, r, te, tr, info = env.step(obs["tool_pos"])
    assert tr and set(info) == {"success", "sides_ok", "final_dist", "n_steps", "overload", "max_load"}
    assert np.isfinite(info["max_load"]) and info["overload"] is False
    env.close()
    plain = api.RealStationB("smoke", "", "", None, _time_limit=5)      # the override is lab-only
    assert plain._limit == api.TIME_LIMIT
    plain.close()


def test_same_situation_as_station_and_determinism(tmp_env):
    api = _api()
    from reallab.station import api as station_api
    a, b, s = api.make("smoke"), api.make("smoke"), station_api.make("smoke")
    ta, tb = _run(a, 11), _run(b, 11)
    for oa, ob in zip(ta, tb):
        for k in oa:
            assert _eq(oa[k], ob[k])
    ts = _run(s, 11)
    for oa, os_ in zip(ta, ts):
        for k in oa:
            if k != "load":
                assert _eq(oa[k], os_[k]), k
    a.close(); b.close(); s.close()


def test_lives_cap_and_blackbox_with_true_load(tmp_env):
    api = _api()
    with pytest.raises(ValueError):
        api.make("calibration")
    env = api.make("calibration", pilot="tester")
    for k in range(5):
        _run(env, k, n=10)
    with pytest.raises(ledger.BudgetExceeded):
        env.reset(seed=5)
    env.close()
    d = np.load(tmp_env / "blackbox" / "tester" / "life_2.npz")
    assert set(d.files) == OBS_KEYS | LOG_SCALARS | {"actions", "rewards"}
    assert d["points"].shape == (11, 20, 3) and d["load"].shape == (11,) and np.all(np.isfinite(d["load"]))
    assert float(d["max_load"]) == pytest.approx(d["load"].max()) and int(d["life"]) == 2
    t = ledger.totals(tmp_env / "ledger.jsonl")[("stationb", "stationb:tester", "calibration")]
    assert t == {"steps": 50, "episodes": 5}


def test_seed_ranges_and_observation_refused(tmp_env):
    api = _api()
    from reallab.stationb import parallel
    for p in ("observation",):
        with pytest.raises(ValueError):
            api.make(p)
        with pytest.raises(ValueError):
            parallel.evaluate(":noop_policy", None, [900], purpose=p)
    ev = api.make("eval")
    with pytest.raises(ValueError):
        ev.reset(seed=5)
    ev.reset(seed=1000)
    ev.close()
    cal = api.make("calibration", pilot="s")
    with pytest.raises(ValueError):
        cal.reset(seed=1000)
    cal.close()


def test_evaluate_four_workers_logs_true_load(tmp_env):
    from reallab.stationb.parallel import evaluate
    seeds = [1000, 1001, 1002, 1003]
    res = evaluate(":random_policy", {"seed": 1}, seeds, purpose="smoke", n_workers=4, log_dir=str(tmp_env / "logs"))
    assert [r["seed"] for r in res] == seeds
    for r in res:
        assert set(r) == {"seed", "ret", "n_steps", "success", "sides_ok", "final_dist", "max_load", "overload"}
        d = np.load(tmp_env / "logs" / f"ep_{r['seed']}.npz")
        assert d["load"].shape == (r["n_steps"] + 1,) and np.all(np.isfinite(d["load"]))
        assert d["load"].max() == pytest.approx(r["max_load"])
    assert ledger.totals(tmp_env / "ledger.jsonl")[("stationb", "stationb", "smoke")]["episodes"] == 4
    res2 = evaluate(":random_policy", {"seed": 1}, seeds[:2], purpose="smoke", n_workers=2)
    assert [r["ret"] for r in res2] == [r["ret"] for r in res[:2]]


# ------------------------------------------------------------------------------------------------ disclosure
_WORDS = base64.b64decode(
    b"Y2FibGUgcm9wZSB3aXJlIGNvcmQgc3RyaW5nIGNoYWluIHRldGhlciBkZWZvcm1hYmxlIGZsZXhpYmxlIGVsYXN0aWMgc29mdCBzdGlmZm5lc3Mg"
    b"YmVuZGluZyBmcmljdGlvbiBjYXBzdGFuIHBlZyBjb25uZWN0b3IgY2xpcCBhbmNob3IgZ3JpcHBlciBtdWpvY28gcGx1Z2luIHJvdXRpbmcgaGFy"
    b"bmVzcyBjb21wb3NpdGUgZGF0YXNoZWV0").decode().split()
_RX = re.compile(r"(?i)\b(" + "|".join(_WORDS) + ")")


def _hits(text):
    return sorted(set(m.group(0).lower() for m in _RX.finditer(text)))


def test_no_revealing_words_in_public_files_and_names():
    for f in ("api.py", "parallel.py", "__init__.py"):
        assert _hits((HERE / f).read_text()) == [], f
    for p in HERE.rglob("*"):
        if "__pycache__" not in p.parts:
            assert not any(w in p.name.lower() for w in _WORDS), p.name
    from reallab.stationb import api
    assert _hits(json.dumps(api.task_card())) == []
    assert _hits("the Gripper holds a cable") == ["cable", "gripper"]
