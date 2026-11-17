"""Plumbing tests for the Push-T lab (purpose='smoke'; temporary ledger and black-box directory)."""
import numpy as np
import pytest

from cawm import ledger


@pytest.fixture(autouse=True)
def tmp_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAWM_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("CAWM_BLACKBOX_DIR", str(tmp_path / "blackbox"))
    monkeypatch.setattr(ledger, "DEFAULT_PATH", tmp_path / "ledger.jsonl")
    return tmp_path


def _api():
    from reallab.pusht import api
    return api


def _run(env, seed, n=15, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    obs = env.reset(seed)
    traj = [obs]
    for _ in range(n):
        obs, r, te, tr, info = env.step(rng.uniform(150, 360, 2))
        traj.append(obs)
        if te or tr:
            break
    return traj


def test_obs_schema_and_conventions(tmp_env):
    api = _api()
    env = api.make("smoke")
    obs = env.reset(seed=7)
    assert set(obs) == {"agent_pos", "agent_vel", "block_pose", "goal_pose", "coverage", "t"}
    assert obs["agent_pos"].shape == (2,) and obs["agent_vel"].shape == (2,)
    assert obs["block_pose"].shape == (3,) and 0 <= obs["block_pose"][2] < 2 * np.pi
    assert np.allclose(obs["goal_pose"], [256, 256, np.pi / 4]) and obs["t"] == 0
    assert 0.0 <= obs["coverage"] <= 1.0
    assert env.control_hz == 10.0 and np.all(env.action_low == 0) and np.all(env.action_high == 512)
    obs, r, te, tr, info = env.step(np.array([300.0, 300.0]))
    assert obs["t"] == 1 and 0.0 <= r <= 1.0 and not te and not tr
    assert set(info) == {"success", "max_coverage", "final_coverage", "n_steps"}
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 0.0]))
    card = api.task_card()
    assert card["time_limit_steps"] == 300 and card["goal_pose"][2] == pytest.approx(np.pi / 4)
    env.close()


def test_time_limit_and_no_step_after_end(tmp_env):
    api = _api()
    env = api.make("smoke")
    obs = env.reset(seed=3)
    for i in range(300):
        obs, r, te, tr, info = env.step(obs["agent_pos"])
    assert tr and not te and info["n_steps"] == 300
    with pytest.raises(RuntimeError):
        env.step(obs["agent_pos"])
    env.close()


def test_determinism(tmp_env):
    api = _api()
    a, b = api.make("smoke"), api.make("smoke")
    ta, tb = _run(a, 11), _run(b, 11)
    for oa, ob in zip(ta, tb):
        for k in oa:
            assert np.array_equal(oa[k], ob[k])
    tc = _run(b, 12)
    assert not np.array_equal(ta[0]["block_pose"], tc[0]["block_pose"])
    a.close(); b.close()


def test_lives_cap_blackbox_and_ledger(tmp_env):
    api = _api()
    with pytest.raises(ValueError):
        api.make("calibration")                       # pilot required
    env = api.make("calibration", pilot="tester")
    for k in range(5):
        _run(env, k, n=10)
    with pytest.raises(ledger.BudgetExceeded):
        env.reset(seed=5)
    env.close()
    for k in range(1, 6):
        f = tmp_env / "blackbox" / "tester" / f"life_{k}.npz"
        assert f.exists()
    d = np.load(tmp_env / "blackbox" / "tester" / "life_2.npz")
    assert d["block_pose"].shape == (11, 3) and d["actions"].shape == (10, 2) and int(d["life"]) == 2
    assert int(d["seed"]) == 1 and str(d["pilot"]) == "tester" and d["rewards"].shape == (10,)
    t = ledger.totals(tmp_env / "ledger.jsonl")[("pusht", "pusht:tester", "calibration")]
    assert t == {"steps": 50, "episodes": 5}
    # a new env object for the same pilot still sees the used-up lives
    env2 = api.make("calibration", pilot="tester")
    with pytest.raises(ledger.BudgetExceeded):
        env2.reset(seed=1)
    env2.close()


def test_abandoned_episode_counts(tmp_env):
    api = _api()
    env = api.make("calibration", pilot="p")
    env.reset(seed=1); env.step(np.array([200.0, 200.0]))
    env.reset(seed=2)
    env.close()
    t = ledger.totals(tmp_env / "ledger.jsonl")[("pusht", "pusht:p", "calibration")]
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


def test_evaluate_four_workers_and_gif(tmp_env):
    from reallab.pusht.parallel import evaluate
    seeds = [1000, 1001, 1002, 1003]
    res = evaluate(":random_policy", {"seed": 1}, seeds, purpose="smoke", n_workers=4, record_first=1,
                   log_dir=str(tmp_env / "logs"), video_dir=str(tmp_env / "vid"))
    assert [r["seed"] for r in res] == seeds
    for r in res:
        assert set(r) == {"seed", "ret", "n_steps", "success", "max_coverage", "final_coverage"}
        assert r["n_steps"] >= 1
    gif = tmp_env / "vid" / "ep_1000.gif"
    assert gif.exists() and 10_000 < gif.stat().st_size < 6_500_000
    from PIL import Image
    im = Image.open(gif)
    assert im.n_frames > 5 and min(im.size) >= 384
    assert (tmp_env / "logs" / "ep_1003.npz").exists()
    t = ledger.totals(tmp_env / "ledger.jsonl")[("pusht", "pusht", "smoke")]
    assert t["episodes"] == 4
    # same policy/seed gives the same result in another call (cross-process determinism)
    res2 = evaluate(":random_policy", {"seed": 1}, seeds[:2], purpose="smoke", n_workers=2)
    assert [r["ret"] for r in res2] == [r["ret"] for r in res[:2]]


def test_recording_does_not_change_trajectory(tmp_env):
    api = _api()
    a = api.make("smoke")
    b = api.make("smoke", record_path=str(tmp_env / "rec.gif"))
    ta, tb = _run(a, 21, n=40), _run(b, 21, n=40)
    for oa, ob in zip(ta, tb):
        for k in oa:
            assert np.array_equal(oa[k], ob[k])
    b.close()
    assert (tmp_env / "rec.gif").exists()
    a.close()


def test_render_life(tmp_env):
    api = _api()
    env = api.make("calibration", pilot="r")
    _run(env, 4, n=12)
    env.close()
    from reallab.pusht.render import render_life
    out = render_life(str(tmp_env / "blackbox" / "r" / "life_1.npz"), str(tmp_env / "life.gif"))
    from PIL import Image
    assert Image.open(out).n_frames >= 13


# ------------------------------------------------------------------------------------------------ T regression
def _trajectory_hash(seed=1234, n=80, rng_seed=0):
    import hashlib
    api = _api()
    env = api.make("smoke")
    rng = np.random.default_rng(rng_seed)
    o = env.reset(seed)
    h = hashlib.sha256()
    steps = 0

    def add(o, r=0.0):
        for k in ("agent_pos", "agent_vel", "block_pose", "goal_pose"):
            h.update(np.asarray(o[k], np.float64).tobytes())
        h.update(np.float64(o["coverage"]).tobytes())
        h.update(np.float64(r).tobytes())

    add(o)
    for _ in range(n):
        a = rng.uniform(100, 420, 2) if steps % 7 else o["block_pose"][:2] + rng.uniform(-30, 30, 2)
        o, r, te, tr, info = env.step(a)
        steps += 1
        add(o, r)
        if te or tr:
            break
    env.close()
    return h.hexdigest(), o["block_pose"].tolist(), o["coverage"]


def test_T_trajectory_unchanged_by_L_extension():
    """Golden values recorded on the server before the L block was added."""
    digest, pose, cov = _trajectory_hash()
    assert digest == "2a7f927a81d9bb700b618e2027d5d94d765ca47c9b215b34f9148e0ec0e7870f"
    assert pose == [434.6005621070493, 6.935600353564638, 6.282587357452458] and cov == 0.0
    assert _api().task_card()["goal_pose"] == [256.0, 256.0, float(np.pi / 4)]


# ------------------------------------------------------------------------------------------------ the L block
def test_L_schema_card_and_geometry(tmp_env):
    api = _api()
    env = api.make("smoke", object="L")
    obs = env.reset(seed=7)
    assert set(obs) == {"agent_pos", "agent_vel", "block_pose", "goal_pose", "coverage", "t"}
    assert np.allclose(obs["goal_pose"], [300, 200, np.pi / 2]) and obs["t"] == 0
    card = api.task_card(object="L")
    assert card["block"]["shape"] == "L" and card["goal_pose"][:2] == [300.0, 200.0]
    assert card["calibration_lives"] == 2 and api.task_card()["block"]["shape"] == "T"
    # coverage is 1 when the block sits exactly at the goal pose, and the canonical function agrees with the sim
    assert api.coverage_of((300.0, 200.0, np.pi / 2), "L") == pytest.approx(1.0)
    assert api.coverage_of((300.0, 200.0, np.pi / 2), "T") < 0.5
    sim = env._sim
    sim.block.angle = np.pi / 2
    sim.block.position = (300.0, 200.0)
    assert sim._get_coverage() == pytest.approx(1.0, abs=1e-9)
    # body-frame check of the datasheet: the sim's block vertices are the documented rectangles
    verts = sorted(tuple(map(tuple, np.round([list(v) for v in s.get_vertices()], 6))) for s in sim.block.shapes)
    assert sorted(max(v[0] for v in s) - min(v[0] for v in s) for s in verts) == [40.0, 60.0]
    obs, r, te, tr, info = env.step(np.array([300.0, 300.0]))
    assert obs["t"] == 1 and 0.0 <= r <= 1.0
    env.close()


def test_L_determinism_and_difference_from_T(tmp_env):
    api = _api()
    a, b, t = api.make("smoke", object="L"), api.make("smoke", object="L"), api.make("smoke")
    ta, tb, tt = _run(a, 11), _run(b, 11), _run(t, 11)
    for oa, ob in zip(ta, tb):
        for k in oa:
            assert np.array_equal(oa[k], ob[k])
    assert not np.array_equal(ta[-1]["goal_pose"], tt[-1]["goal_pose"])
    a.close(); b.close(); t.close()


def test_L_lives_cap_2_blackbox_and_ledger(tmp_env):
    api = _api()
    env = api.make("calibration", pilot="lp", object="L")
    for k in range(2):
        _run(env, k, n=10)
    with pytest.raises(ledger.BudgetExceeded):
        env.reset(seed=2)
    env.close()
    for k in (1, 2):
        assert (tmp_env / "blackbox_L" / "lp" / f"life_{k}.npz").exists()
    assert not (tmp_env / "blackbox" / "lp").exists()
    d = np.load(tmp_env / "blackbox_L" / "lp" / "life_2.npz")
    assert d["block_pose"].shape == (11, 3) and int(d["life"]) == 2 and str(d["object"]) == "L"
    t = ledger.totals(tmp_env / "ledger.jsonl")[("pusht", "pusht_L:lp", "calibration")]
    assert t == {"steps": 20, "episodes": 2}
    # the T lives of the same pilot are independent (5 lives, other ledger key)
    envT = api.make("calibration", pilot="lp")
    envT.reset(seed=1); envT.close()
    assert ledger.totals(tmp_env / "ledger.jsonl")[("pusht", "pusht:lp", "calibration")]["episodes"] == 1
    ev = api.make("eval", object="L")
    ev.reset(seed=1000); ev.step(np.array([200.0, 200.0])); ev.close()
    assert ledger.totals(tmp_env / "ledger.jsonl")[("pusht", "pusht_L", "eval")]["episodes"] == 1
    with pytest.raises(ValueError):
        api.make("smoke", object="Z")


def test_L_evaluate_gif_and_render_life(tmp_env):
    from PIL import Image
    from reallab.pusht.parallel import evaluate
    from reallab.pusht.render import render_life
    res = evaluate(":random_policy", {"seed": 1}, [1000, 1001], purpose="smoke", n_workers=2, record_first=1,
                   video_dir=str(tmp_env / "vid"), object="L")
    assert len(res) == 2 and (tmp_env / "vid" / "ep_1000.gif").exists()
    assert ledger.totals(tmp_env / "ledger.jsonl")[("pusht", "pusht_L", "smoke")]["episodes"] == 2
    api = _api()
    env = api.make("calibration", pilot="rl", object="L")
    _run(env, 4, n=12)
    env.close()
    out = render_life(str(tmp_env / "blackbox_L" / "rl" / "life_1.npz"), str(tmp_env / "life.gif"))
    im = Image.open(out)
    assert im.n_frames >= 13 and min(im.size) >= 384
