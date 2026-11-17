"""Plumbing tests for the station lab (purpose='smoke'; temporary ledger, black-box and observation directories)."""
import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest

from cawm import ledger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OBS_KEYS = {"points", "tool_pos", "tool_vel", "posts", "post_sides", "sides_now", "target", "load", "t"}
LOG_SCALARS = {"seed", "life", "pilot", "purpose", "success", "sides_ok", "final_dist", "max_load", "overload",
               "n_steps", "terminated", "truncated", "control_hz"}


@pytest.fixture(autouse=True)
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAWM_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("CAWM_BLACKBOX_DIR", str(tmp_path / "blackbox"))
    monkeypatch.setenv("CAWM_OBSERVATION_DIR", str(tmp_path / "observation"))
    monkeypatch.setattr(ledger, "DEFAULT_PATH", tmp_path / "ledger.jsonl")
    return tmp_path


def _api():
    from reallab.station import api
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


def test_obs_schema_and_card(tmp_env):
    api = _api()
    env = api.make("smoke")
    obs = env.reset(seed=7)
    assert set(obs) == OBS_KEYS
    assert obs["points"].shape == (20, 3) and obs["tool_pos"].shape == (2,) and obs["tool_vel"].shape == (2,)
    assert obs["posts"].shape == (3, 2) and obs["post_sides"].shape == (3,) and set(obs["post_sides"]) <= {-1, 1}
    assert obs["sides_now"].shape == (3,) and obs["target"].shape == (2,) and obs["t"] == 0
    assert np.allclose(obs["points"][-1, :2], obs["tool_pos"], atol=5e-3)          # point 20 moves with the tool
    assert np.array_equal(obs["sides_now"], api.sides(obs["points"], obs["posts"]))
    assert env.control_hz == 20.0 and np.allclose(env.action_low, [0, -0.4]) and np.allclose(env.action_high, [1.2, 0.4])
    obs, r, te, tr, info = env.step(obs["tool_pos"])
    assert obs["t"] == 1 and not te and not tr and isinstance(r, float)
    assert set(info) == {"success", "sides_ok", "final_dist", "max_load", "overload", "n_steps"}
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 0.0]))
    card = api.task_card()
    assert card["time_limit_steps"] == 600 and card["load_limit"] == 10.0 and card["success"]["tolerance"] == 0.02
    env.close()


def test_determinism(tmp_env):
    api = _api()
    a, b = api.make("smoke"), api.make("smoke")
    ta, tb = _run(a, 11), _run(b, 11)
    for oa, ob in zip(ta, tb):
        for k in oa:
            assert np.array_equal(oa[k], ob[k])
    tc = _run(b, 12)
    assert not np.array_equal(ta[0]["posts"], tc[0]["posts"])
    a.close(); b.close()


def test_side_rule_hand_made():
    api = _api()
    post = np.array([[0.5, 0.0]])
    x = np.linspace(0.0, 1.0, 20)
    below = np.stack([x, np.full(20, -0.1)], 1)      # travelling +x below the post: post on the left -> +1
    above = np.stack([x, np.full(20, 0.1)], 1)
    assert api.sides(below, post)[0] == 1 and api.sides(above, post)[0] == -1
    assert api.winding(below, post)[0] == pytest.approx(np.pi - 2 * np.arctan(0.1 / 0.5), abs=1e-9)
    short = np.stack([np.linspace(0.0, 0.3, 20), np.full(20, -0.1)], 1)   # never gets past the post
    assert api.sides(short, post)[0] == 0
    th = np.linspace(np.pi, np.pi + 2 * np.pi + 0.5, 20)                  # a loop and a half counter-clockwise
    loop = 0.5 + 0.1 * np.stack([np.cos(th), np.sin(th)], 1) - np.array([0.0, 0.5])
    assert api.winding(loop, post)[0] == pytest.approx(2 * np.pi + 0.5, abs=1e-9)
    assert api.sides(loop[::-1], post)[0] == -1
    # three posts at once, 3-d points accepted
    pts = np.concatenate([below, np.zeros((20, 1))], 1)
    assert list(api.sides(pts, np.array([[0.5, 0.0], [0.5, -0.3], [2.0, 0.0]]))) == [1, -1, 0]


def test_lives_cap_blackbox_and_ledger(tmp_env):
    api = _api()
    with pytest.raises(ValueError):
        api.make("calibration")
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
    t = ledger.totals(tmp_env / "ledger.jsonl")[("station", "station:tester", "calibration")]
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
    t = ledger.totals(tmp_env / "ledger.jsonl")[("station", "station:p", "calibration")]
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


def test_observation_is_lab_only_and_logged(tmp_env):
    api = _api()
    from reallab.station import parallel
    with pytest.raises(ValueError):
        api.make("observation")
    with pytest.raises(ValueError):
        api.make("observation", pilot="sneaky")
    with pytest.raises(ValueError):
        parallel.evaluate(":noop_policy", None, [900], purpose="observation")
    from reallab.station import _reference
    env = api.RealStation("observation", "", "test", None, _lab=True)
    with pytest.raises(ValueError):
        env.reset(seed=899)
    with pytest.raises(ValueError):
        env.reset(seed=1000)
    env.close()
    res = _reference.run_observation([900, 901], max_steps_test=12)
    assert len(res) == 2
    t = ledger.totals(tmp_env / "ledger.jsonl")[("station", "station", "observation")]
    assert t["episodes"] == 2 and t["steps"] == sum(r["n_steps"] for r in res)
    d = np.load(tmp_env / "observation" / "episode_900.npz")
    assert set(d.files) == OBS_KEYS | LOG_SCALARS | {"actions", "rewards"}
    assert str(d["purpose"]) == "observation" and str(d["pilot"]) == "" and int(d["life"]) == 0
    idx = json.loads((tmp_env / "observation" / "index.json").read_text())
    assert [e["seed"] for e in idx["episodes"]] == [900, 901]
    assert set(idx["episodes"][0]) == {"seed", "outcome", "steps", "max_load"}


def test_evaluate_four_workers(tmp_env):
    from reallab.station.parallel import evaluate
    seeds = [1000, 1001, 1002, 1003]
    res = evaluate(":random_policy", {"seed": 1}, seeds, purpose="smoke", n_workers=4, log_dir=str(tmp_env / "logs"))
    assert [r["seed"] for r in res] == seeds
    for r in res:
        assert set(r) == {"seed", "ret", "n_steps", "success", "sides_ok", "final_dist", "max_load", "overload"}
        assert r["n_steps"] >= 1
    d = np.load(tmp_env / "logs" / "ep_1003.npz")
    assert OBS_KEYS <= set(d.files)
    assert ledger.totals(tmp_env / "ledger.jsonl")[("station", "station", "smoke")]["episodes"] == 4
    res2 = evaluate(":random_policy", {"seed": 1}, seeds[:2], purpose="smoke", n_workers=2)
    assert [r["ret"] for r in res2] == [r["ret"] for r in res[:2]]


def test_lab_gif_and_recording_does_not_change_trajectory(tmp_env):
    api = _api()
    a = api.make("smoke")
    b = api.RealStation("smoke", "", "", None, _lab=True)
    b._rec_path, b._rec_title = str(tmp_env / "rec.gif"), "test {seed}"
    ta, tb = _run(a, 21, n=40), _run(b, 21, n=40)
    for oa, ob in zip(ta, tb):
        for k in oa:
            assert np.array_equal(oa[k], ob[k])
    b.close(); a.close()
    from PIL import Image
    im = Image.open(tmp_env / "rec.gif")
    assert im.n_frames > 30 and im.size[0] >= 480 and (tmp_env / "rec.gif").stat().st_size < 6_500_000
    env = api.make("calibration", pilot="r")
    _run(env, 4, n=12)
    env.close()
    from reallab.station._render import render_life
    out = render_life(str(tmp_env / "blackbox" / "r" / "life_1.npz"), str(tmp_env / "life.gif"))
    assert Image.open(out).n_frames >= 13


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
    obs_dir = ROOT / "runs" / "station" / "observation"
    if obs_dir.exists():
        for p in obs_dir.rglob("*"):
            assert not any(w in p.name.lower() for w in _WORDS), p.name
            if p.suffix == ".npz":
                d = np.load(p)
                assert _hits(" ".join(d.files)) == []
                assert _hits(" ".join(str(d[k]) for k in d.files if d[k].dtype.kind == "U")) == []
            if p.suffix == ".json":
                assert _hits(p.read_text()) == []
    from reallab.station import api
    assert _hits(json.dumps(api.task_card())) == []


def test_the_scan_works():
    assert _hits("the Gripper holds a cable") == ["cable", "gripper"] and _hits("a recorded scene") == []
