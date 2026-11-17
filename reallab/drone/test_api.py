import json
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="drone_ledger_")
os.environ["CAWM_LEDGER"] = os.path.join(_TMP, "ledger.jsonl")   # must precede the cawm.ledger import

import numpy as np
import pytest

from cawm.ledger import BudgetExceeded, Meter
from reallab.drone import _course
from reallab.drone._lab import make_lab
from reallab.drone.api import CAL_CAP_STEPS, make, site_map
from reallab.drone.parallel import evaluate

LEDGER = os.environ["CAWM_LEDGER"]
HOVER = np.array([0.0, 0.0, 0.0, 0.30])


def ledger_sum(tag, path=LEDGER):
    s = e = 0
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r["tag"] == tag:
                s += r["steps"]; e += r["episodes"]
    return s, e


def run_until_end(env, act, seed, limit=3000):
    obs, info = env.reset(seed=seed)
    n, ret = 0, 0.0
    while n < limit:
        obs, r, te, tr, info = env.step(act(obs) if callable(act) else act)
        n += 1; ret += r
        if te or tr:
            return obs, info, n, ret, te, tr
    raise AssertionError("episode did not end")


def test_schema_waypoints():
    env = make("waypoints", "smoke", "t_schema_wp")
    assert env.task == "waypoints" and env.control_hz == 30.0
    assert env.action_low.shape == env.action_high.shape == (4,)
    np.testing.assert_allclose(env.action_low, [-np.pi] * 3 + [0.0]); np.testing.assert_allclose(env.action_high, [np.pi] * 3 + [0.8])
    obs, info = env.reset(seed=1)
    shapes = dict(ang_vel=(3,), quat=(4,), lin_vel=(3,), lin_pos=(3,), prev_action=(4,), aux=(4,), target_deltas=(4, 3))
    assert set(obs) == set(shapes)
    for k, s in shapes.items():
        assert obs[k].shape == s and obs[k].dtype == np.float64, k
    obs, r, te, tr, info = env.step(HOVER)
    np.testing.assert_allclose(obs["prev_action"], HOVER)
    obs, info, n, ret, te, tr = run_until_end(env, HOVER, 2)
    for k in ["targets_reached", "num_targets", "success", "crashed", "out_of_bounds", "timeout"]:
        assert k in info
    assert info["num_targets"] == 4 and info["success"] is False
    env.close()


def test_schema_port():
    env = make("port", "smoke", "t_schema_port")
    obs, info = env.reset(seed=1)
    shapes = dict(ang_vel=(3,), quat=(4,), lin_vel=(3,), pos=(3,), goal=(3,), ranges=(6, 16), t=(1,))
    assert set(obs) == set(shapes)
    for k, s in shapes.items():
        assert obs[k].shape == s and obs[k].dtype == np.float64, k
    assert 56 <= obs["goal"][0] <= 58 and abs(obs["goal"][1]) <= 6 and 2 <= obs["goal"][2] <= 4
    assert abs(obs["pos"][0] - 3) < 0.5 and abs(obs["pos"][1]) <= 3.5 and abs(obs["pos"][2] - 2) < 0.5
    assert obs["ranges"].max() <= 12.0 and obs["ranges"].min() > 0
    obs, info, n, ret, te, tr = run_until_end(env, HOVER * [1, 1, 1, 0.0], 3)   # zero thrust -> falls and hits the ground
    assert te and info["collision"] and not info["success"]
    for k in ["success", "collision", "out_of_bounds", "timeout", "time_s", "path_length_m", "mean_speed", "min_range_seen"]:
        assert k in info
    env.close()


@pytest.mark.parametrize("task,kw", [("waypoints", {}), ("port", dict(realism=False)), ("port", dict(realism=True))])
def test_determinism(task, kw):
    def traj(seed):
        env = make(task, "smoke", "t_det", **kw)
        obs, _ = env.reset(seed=seed)
        out = []
        for i in range(40):
            obs, *_ = env.step(np.array([0.1 * np.sin(i), 0.05, 0.0, 0.31]))
            out.append(np.concatenate([np.ravel(v) for v in obs.values()]))
        env.close()
        return np.array(out)
    a, b, c = traj(5), traj(5), traj(6)
    np.testing.assert_array_equal(a, b)
    assert np.abs(a - c).max() > 0


def test_range_sanity_wall():
    wall = dict(kind="container", center=[10.5, 0.0, 3.0], half_extents=[0.5, 10.0, 3.0])
    course = dict(boxes=[wall], cylinders=[], start=[3.0, 0.0, 2.0], goal=[57.0, 0.0, 3.0], difficulty="x", attempt=0)
    env = make_lab("port", "smoke", "t_range", realism=False, _course=course)
    obs, _ = env.reset(seed=0)
    R = obs["ranges"]
    pos = obs["pos"]
    el = np.deg2rad(-25 + 10 * np.arange(6)); az = np.deg2rad(-60 + 8 * np.arange(16))
    checked = 0
    for i in range(6):
        for j in range(16):
            d = np.array([np.cos(el[i]) * np.cos(az[j]), np.cos(el[i]) * np.sin(az[j]), np.sin(el[i])])
            t = (10.0 - pos[0]) / d[0]
            hit = pos + t * d
            if t <= 12.0 and abs(hit[1]) < 9.5 and 0.1 < hit[2] < 5.9:
                assert abs(R[i, j] - t) < 0.02, (i, j, R[i, j], t)
                checked += 1
    assert checked > 40
    # straight ahead (az = +-4 deg, el = +-5 deg) is about 7 m
    assert abs(R[3, 7] - 7.0 / (np.cos(np.deg2rad(5)) * np.cos(np.deg2rad(4)))) < 0.02
    # rays that miss everything read exactly the max range; downward rays hit the ground
    assert np.all(R[R > 11.99] == 12.0)
    assert R[0, 0] < 12.0 and abs(R[0, 8] - (2.0 / np.sin(np.deg2rad(25)))) > -1  # sanity: finite
    env.close()


def test_range_yaw_stabilised():
    """Ranges are expressed in the heading frame: same wall, drone yawed by +30 deg gives a shifted pattern."""
    wall = dict(kind="container", center=[10.5, 0.0, 3.0], half_extents=[0.5, 10.0, 3.0])
    course = dict(boxes=[wall], cylinders=[], start=[3.0, 0.0, 2.0], goal=[57.0, 0.0, 3.0], difficulty="x", attempt=0)
    env = make_lab("port", "smoke", "t_range2", realism=False, _course=course)
    obs, _ = env.reset(seed=0)
    for _ in range(14):                                     # yaw left with a constant yaw-rate command
        obs, *_ = env.step(np.array([0.0, 0.0, 1.0, 0.2988]))
    q = obs["quat"]; yaw = np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    assert 0.3 < yaw < 0.7
    pos = obs["pos"]; R = obs["ranges"]
    el = np.deg2rad(-25 + 10 * np.arange(6)); az = np.deg2rad(-60 + 8 * np.arange(16))
    i, j = 3, 7
    d = np.array([np.cos(el[i]) * np.cos(az[j] + yaw), np.cos(el[i]) * np.sin(az[j] + yaw), np.sin(el[i])])
    t = (10.0 - pos[0]) / d[0]
    assert abs(R[i, j] - t) < 0.05
    env.close()


def test_meter_counts():
    env = make("port", "smoke", "t_meter")
    env.reset(seed=1)
    for _ in range(7):
        env.step(HOVER)
    env.reset(seed=2)                    # abandons episode 1 -> counts as an episode
    for _ in range(5):
        env.step(HOVER)
    env.close()                          # abandons episode 2
    assert ledger_sum("t_meter") == (12, 2)
    env = make("port", "smoke", "t_meter2")
    run_until_end(env, HOVER * [1, 1, 1, 0.0], 3)
    s, e = ledger_sum("t_meter2")
    env.close()
    assert e == 1 and s > 0 and ledger_sum("t_meter2") == (s, 1)


def test_calibration_cap(tmp_path):
    led = tmp_path / "cap.jsonl"
    led.write_text(json.dumps(dict(t=0, demo="drone", scenario="waypoints", purpose="calibration", tag="old",
                                   steps=CAL_CAP_STEPS - 10, episodes=3, pid=0)) + "\n")
    env = make("waypoints", "calibration", "t_cap")
    env.meter.path = led
    env.meter._spent = None
    env.reset(seed=1)
    with pytest.raises(BudgetExceeded):
        for _ in range(50):
            _, _, te, tr, _ = env.step(HOVER)
            assert not (te or tr)
    assert sum(json.loads(l)["steps"] for l in led.read_text().splitlines()) == CAL_CAP_STEPS
    # the cap is per task: port is untouched
    env2 = make("port", "calibration", "t_cap2"); env2.meter.path = led; env2.meter._spent = None
    env2.reset(seed=1); env2.step(HOVER); env2.close()


def test_course_info_visibility():
    for purpose, seed in [("calibration", 3), ("eval", 1003), ("smoke", 3)]:
        env = make("port", purpose, "t_course")
        obs, info = env.reset(seed=seed)
        if purpose == "calibration":
            c = info["course"]
            assert set(c) == {"bounds", "boxes", "cylinders"} and len(c["boxes"]) > 0
            assert set(c["boxes"][0]) == {"kind", "center", "half_extents"}
            assert set(c["cylinders"][0]) == {"kind", "center", "radius", "height"} if c["cylinders"] else True
        else:
            assert "course" not in info
        env.close()
    env = make("waypoints", "calibration", "t_course"); _, info = env.reset(seed=1); assert "course" not in info; env.close()


def test_seed_enforcement():
    for purpose, ok, bad in [("calibration", 999, 1000), ("eval", 1000, 999), ("oracle_train", 1_000_000, 1000)]:
        for task in ["port", "waypoints"]:
            env = make(task, purpose, "t_seed")
            env.reset(seed=ok)
            with pytest.raises(ValueError):
                env.reset(seed=bad)
            with pytest.raises(ValueError):
                env.reset(seed=-1)
            env.close()
    with pytest.raises(ValueError):
        make("port", "eval", "t_seed", seed=5)
    env = make("port", "calibration", "t_seed")          # random default seeds fall in range
    for _ in range(3):
        _, info = env.reset(); assert 0 <= info["seed"] < 1000
    env.close()
    with pytest.raises(ValueError):
        make("port", "calibration", "t_seed", realism=False)


def test_evaluate_four_workers():
    seeds = [1000, 1001, 1002, 1003]
    with tempfile.TemporaryDirectory() as logd:
        rows = evaluate(":hover_policy", {}, "port", seeds, purpose="smoke", tag="t_eval4", n_workers=4, log_dir=logd,
                        difficulty="easy")
        assert [r["seed"] for r in rows] == seeds
        for r in rows:
            for k in ["ret", "n_steps", "success", "collision", "out_of_bounds", "timeout", "time_s", "path_length_m",
                      "mean_speed", "min_range_seen"]:
                assert k in r
        assert len(os.listdir(logd)) == 4
        import gzip
        lines = [json.loads(l) for l in gzip.open(os.path.join(logd, f"port_smoke_{seeds[0]}.jsonl.gz"), "rt")]
        assert lines[0]["type"] == "reset" and lines[-1]["type"] == "end" and len(lines) == rows[0]["n_steps"] + 2
    s, e = ledger_sum("t_eval4")
    assert e == 4 and s == sum(r["n_steps"] for r in rows)
    rows2 = evaluate(":random_policy", {"seed": 1}, "waypoints", [1, 2], purpose="smoke", tag="t_eval4b", n_workers=2)
    assert len(rows2) == 2 and "targets_reached" in rows2[0]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_courses_solvable(difficulty):
    blocked = 0
    for seed in range(20):
        c = _course.generate(seed, difficulty)
        assert _course.solvable(c)
        for b in c["boxes"]:
            lo = np.array(b["center"]) - np.array(b["half_extents"])
            hi = np.array(b["center"]) + np.array(b["half_extents"])
            assert lo[2] >= -1e-9 and lo[0] > 7.0 and hi[0] < 53.0 and abs(lo[1]) < 12.0 and abs(hi[1]) < 12.0
        assert 3 <= c["start"][0] <= 3 and abs(c["start"][1]) <= 3 and c["start"][2] == 2.0
        assert 56 <= c["goal"][0] <= 58 and abs(c["goal"][1]) <= 6 and 2 <= c["goal"][2] <= 4
        blocked += _course.straight_line_blocked(c)
        # deterministic
        assert _course.generate(seed, difficulty)["boxes"] == c["boxes"]
    print(difficulty, "straight line blocked in", blocked, "/ 20")


# ----------------------------------------------------------------------------------------------- site
SITE_KEYS = dict(ang_vel=(3,), quat=(4,), lin_vel=(3,), pos=(3,), goal=(3,), ranges=(6, 16), t=(1,),
                 relay_index=(1,), relays=(3, 3))


@pytest.mark.parametrize("mode", ["ctbr", "position"])
def test_site_schema_and_modes(mode):
    env = make_lab("site", "smoke", "t_site_schema", command_mode=mode, anomalies=False)
    obs, info = env.reset(seed=1)
    assert info["command_mode"] == mode and info["mission"] == "A"
    assert set(obs) == set(SITE_KEYS)
    for k, sh in SITE_KEYS.items():
        assert obs[k].shape == sh and obs[k].dtype == np.float64, k
    np.testing.assert_allclose(obs["goal"], obs["relays"][0])
    assert abs(obs["pos"][0] - 3) < 1.2 and abs(obs["pos"][1]) < 1.2
    if mode == "position":
        assert env.action_high[3] == 6.0 and env.action_low[3] == 0.0
        act = lambda o: np.array([o["pos"][0] + 2.0, 0.0, 0.0, 2.0])
    else:
        assert env.action_high[3] == 0.8
        act = lambda o: np.array([0.0, 0.0, 0.0, 0.30])
    for _ in range(60):
        obs, r, te, tr, info = env.step(act(obs))
        assert not (te or tr)
    if mode == "position":
        assert obs["pos"][0] > 4.5
    env.close()


def test_site_mission_b_and_hidden_switches():
    env = make("site", "eval", "t_siteB", mission="B", pilot="x")
    obs, _ = env.reset(seed=1001)
    assert abs(obs["relays"][1][1] - 6.0) < 1.0
    env.close()
    for kw in [dict(anomalies=False), dict(flight_mode=7), dict(max_steps=5)]:
        with pytest.raises(TypeError):
            make("site", "eval", "t", **kw)
    with pytest.raises(ValueError):
        make_lab("site", "eval", "t", anomalies=False)             # lab switch is smoke-only
    with pytest.raises(ValueError):
        make("site", "calibration", "t", pilot="p", mission="B")
    with pytest.raises(ValueError):
        make("site", "calibration", "t")                           # pilot is required
    assert "course" not in site_map() and "B" not in site_map()["missions"]


def test_site_lives_and_blackbox(tmp_path, monkeypatch):
    monkeypatch.setenv("CAWM_BLACKBOX_DIR", str(tmp_path / "bb"))
    led = tmp_path / "lives.jsonl"
    def new(pilot, mode="ctbr"):
        e = make("site", "calibration", "t_lives", pilot=pilot, command_mode=mode)
        e.meter.path = led; return e
    env = new("pilotA")
    for k in range(1, 6):
        env.reset(seed=k)
        for _ in range(4):
            env.step(np.array([0.0, 0.0, 0.0, 0.30]))
    with pytest.raises(BudgetExceeded):
        env.reset(seed=6)
    env.close()
    env2 = new("pilotA")
    with pytest.raises(BudgetExceeded):                            # lives are ledger-wide per pilot
        env2.reset(seed=7)
    env3 = new("pilotB", "position")                               # another pilot has its own five lives
    env3.reset(seed=1)
    for _ in range(3):
        env3.step(np.array([3.0, 0.0, 0.0, 2.0]))
    env3.close()
    with pytest.raises(ValueError):
        make("site", "calibration", "t", pilot="p").reset(seed=1000)
    files = sorted(os.listdir(tmp_path / "bb" / "pilotA"))
    assert files == [f"life_{k}.npz" for k in range(1, 6)]
    z = np.load(tmp_path / "bb" / "pilotA" / "life_2.npz")
    assert str(z["end_cause"]) == "abandoned" and int(z["n_steps"]) == 4 and str(z["command_mode"]) == "ctbr"
    assert z["pos"].shape == (5, 3) and z["ranges"].shape == (5, 6, 16) and z["actions"].shape == (4, 4)
    assert not (set(z.files) & {"course", "truth", "wind", "anomalies"})
    zb = np.load(tmp_path / "bb" / "pilotB" / "life_1.npz")
    assert str(zb["command_mode"]) == "position" and int(zb["life"]) == 1


def _ray_box(o, d, b):
    c, h = np.array(b["center"]), np.array(b["half_extents"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (c - h - o) / d, (c + h - o) / d
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    t0, t1_ = np.nanmax(np.where(np.isnan(tmin), -np.inf, tmin)), np.nanmin(np.where(np.isnan(tmax), np.inf, tmax))
    inside_slab = np.all((o >= c - h) & (o <= c + h) | (d != 0))
    return t0 if (t0 <= t1_ and t1_ > 0 and t0 > 0 and inside_slab) else np.inf


def test_site_map_matches_range_sensor():
    sm = site_map()
    env = make_lab("site", "smoke", "t_map", anomalies=False)
    obs, _ = env.reset(seed=1)
    R, pos = obs["ranges"], obs["pos"]
    el = np.deg2rad(-25 + 10 * np.arange(6)); az = np.deg2rad(-60 + 8 * np.arange(16))
    bad = hits = 0
    for i in range(6):
        for j in range(16):
            d = np.array([np.cos(el[i]) * np.cos(az[j]), np.cos(el[i]) * np.sin(az[j]), np.sin(el[i])])
            t = min([_ray_box(pos, d, b) for b in sm["boxes"]] + [(pos[2] / -d[2]) if d[2] < 0 else np.inf, 12.0])
            if t < 11.5:
                hits += 1
                bad += abs(R[i, j] - t) > 0.1 + 0.03 * t
    assert hits >= 20 and bad <= 3, (hits, bad)
    env.close()


def test_site_determinism():
    def traj():
        env = make("site", "smoke", "t_site_det")
        obs, _ = env.reset(seed=5); out = []
        for i in range(60):
            obs, r, te, tr, _ = env.step(np.array([0.1 * np.sin(i), 0.02, 0.0, 0.31]))
            out.append(np.concatenate([np.ravel(v) for v in obs.values()]))
            if te or tr:
                break
        env.close(); return np.array(out)
    np.testing.assert_array_equal(traj(), traj())
    env = make("site", "smoke", "t_site_det"); a, _ = env.reset(seed=5); b, _ = env.reset(seed=6)
    assert np.abs(a["pos"] - b["pos"]).max() > 0; env.close()


def test_site_hides_truth():
    import reallab.drone.api as api
    src = open(api.__file__).read().lower()
    for word in ["rotor_eff", "jet", "turbulence", "anomal"]:
        assert word not in src, word


def test_site_time_limits_and_stock_autopilot_too_slow():
    assert site_map()["time_limit_s"] == 17.0
    env = make_lab("site", "smoke", "t_tl", command_mode="position", anomalies=False)
    obs, _ = env.reset(seed=1)
    n = 0
    while True:
        obs, r, te, tr, info = env.step(np.array([obs["goal"][0], obs["goal"][1], 0.0, obs["goal"][2]]))   # fly at full speed to the current relay
        n += 1
        if te or tr:
            break
    assert tr and info["timeout"] and not info["success"] and n == 510 and info["time_limit_s"] == 17.0
    assert info["relays_reached"] == 2
    envB = make("site", "smoke", "t_tl", mission="B"); envB.reset(seed=1)
    for _ in range(726):
        _, _, te, tr, info = envB.step(np.array([0.0, 0.0, 0.0, 0.30]))
        if te or tr:
            break
    assert info["time_limit_s"] == 24.2


def test_privileged_ctbr_completes_inside_window():
    from reallab.drone._ctbr_pilot import informed_policy
    wins = 0
    for mission, seeds in (("A", (1, 2, 3, 4)), ("B", (1, 2))):
        for sd in seeds:
            env = make("site", "smoke", "t_priv", mission=mission, command_mode="ctbr")
            pol = informed_policy(); obs, _ = env.reset(seed=sd); pol.reset()
            while True:
                obs, r, te, tr, info = env.step(pol.act(obs))
                if te or tr:
                    break
            wins += bool(info["success"])
            assert info["time_s"] < info["time_limit_s"] or not info["success"]
    assert wins >= 5


# ----------------------------------------------------------------------------------------------- mission C and regression
def _traj_ab(mission, mode, seed, n=60):
    env = make("site", "smoke", "golden", mission=mission, command_mode=mode)
    obs, _ = env.reset(seed=seed)
    out = [np.concatenate([np.ravel(v) for v in obs.values()])]
    for i in range(n):
        if mode == "ctbr":
            a = np.array([0.2 * np.sin(0.3 * i), 0.1 * np.cos(0.2 * i), 0.05, 0.31 + 0.02 * np.sin(0.1 * i)])
        else:
            a = np.array([3.0 + 0.1 * i, 0.3 * np.sin(0.2 * i), 0.0, 2.5])
        obs, r, te, tr, info = env.step(a)
        out.append(np.concatenate([np.ravel(v) for v in obs.values()]))
        if te or tr:
            break
    env.close()
    return np.array(out)


@pytest.mark.parametrize("key,args", [("A_ctbr_5", ("A", "ctbr", 5)), ("A_position_6", ("A", "position", 6)),
                                       ("B_ctbr_7", ("B", "ctbr", 7))])
def test_missions_A_B_unchanged_bit_for_bit(key, args):
    gold = np.load(os.path.join(os.path.dirname(__file__), "_golden_AB.npz"))[key]
    new = _traj_ab(*args)
    assert new.shape == gold.shape
    np.testing.assert_array_equal(new, gold)


def test_mission_C_schema_determinism_and_obs_rules():
    def traj(seed):
        env = make_lab("site", "smoke", "t_C", mission="C", command_mode="position", anomalies=False)
        obs, info = env.reset(seed=seed)
        assert info["mission"] == "C"
        out, seen = [], 0
        for i in range(120):
            assert obs["movers"].shape == (8, 7) and obs["n_movers"].shape == (1,)
            rows = obs["movers"][obs["movers"][:, 6] > 0]
            assert len(rows) == int(obs["n_movers"][0])
            if len(rows):
                d = np.linalg.norm(rows[:, :3] - obs["pos"], axis=1)
                assert np.all(d < 15.5) and np.all(np.diff(d) >= -0.2)         # nearest first (noise tolerance)
                assert np.allclose(rows[:, 6], 0.35) and np.all(obs["movers"][len(rows):] == 0)
                seen += 1
            out.append(np.concatenate([np.ravel(v) for v in obs.values()]))
            obs, r, te, tr, info = env.step(np.array([obs["pos"][0] + 2.0, 0.0, 0.0, 2.5]))
            if te or tr:
                break
        env.close()
        return np.array(out), seen
    (a, s), (b, _), (c, _) = traj(7), traj(7), traj(8)
    np.testing.assert_array_equal(a, b)
    assert a.shape != c.shape or np.abs(a - c).max() > 0
    assert set(np.load(os.path.join(os.path.dirname(__file__), "_golden_AB.npz")).files)    # golden file present
    # A and B observations do not carry the extra fields
    for m in "AB":
        e = make("site", "smoke", "t", mission=m); o, _ = e.reset(seed=1); assert "movers" not in o and "n_movers" not in o; e.close()


def test_mission_C_lives_blackbox_and_collision(tmp_path, monkeypatch):
    monkeypatch.setenv("CAWM_BLACKBOX_DIR", str(tmp_path / "bb"))
    led = tmp_path / "c.jsonl"
    env = make("site", "calibration", "t_lifeC", pilot="pc", mission="C"); env.meter.path = led
    for k in range(1, 6):
        env.reset(seed=k)
        for _ in range(5):
            env.step(np.array([0.0, 0.0, 0.0, 0.30]))
    with pytest.raises(BudgetExceeded):
        env.reset(seed=6)
    env.close()
    z = np.load(tmp_path / "bb" / "pc" / "life_3.npz")
    assert str(z["mission"]) == "C" and z["movers_true_pos"].shape[0] == 6 and z["movers_true_pos"].shape[2] == 3
    assert z["movers_true_vel"].shape == z["movers_true_pos"].shape and int(z["movers_true_count"][0]) in (7, 8)
    assert not (set(z.files) & {"kind", "react", "mode", "speed", "dash", "axes", "trig", "state", "params"})
    with pytest.raises(ValueError):
        make("site", "calibration", "t", pilot="pc", mission="B")
    # a mover placed on the drone's path ends the episode as a collision
    envm = make_lab("site", "smoke", "t_hit", mission="C", command_mode="position", anomalies=False)
    envm.reset(seed=3)
    m = envm._inner.movers
    for s_ in m.m:
        s_["vel"][:] = 0; s_["pause"] = 1e9
    hit = False
    p0 = np.array(envm._inner.drone.state[3])
    m.m[0]["pos"] = np.array([12.0, p0[1], 2.0])                                          # park mover 0 on the flight path
    for _ in range(300):
        _, _, te, tr, info = envm.step(np.array([30.0, p0[1], 0.0, 2.0]))
        if te:
            hit = info["collision"]; break
    assert hit


def test_mission_C_time_limit_and_stock_too_slow():
    assert site_map()["time_limit_s"] == 17.0
    env = make_lab("site", "smoke", "t_C2", mission="C", command_mode="position", anomalies=False)
    env.reset(seed=1)
    n = 0
    while True:
        _, _, te, tr, info = env.step(np.array([3.0, 0.0, 0.0, 2.0])); n += 1
        if te or tr:
            break
    assert info["time_limit_s"] == 19.4 and (info["timeout"] or info["collision"])


def test_no_hidden_wording_in_public_files():
    import reallab.drone.api as api, reallab.drone.render as render
    for mod, words in [(api, ["rotor_eff", "jet", "turbulence", "anomal", "patrol", "reactive", "dart"]),
                       (render, ["rotor", "jet", "turbulence", "anomal", "wind", "damage", "hidden", "patrol", "reactive", "dart"])]:
        src = open(mod.__file__).read().lower()
        for w in words:
            assert w not in src, (mod.__name__, w)


def test_privileged_avoider_completes_mission_C():
    from reallab.drone._ctbr_pilot import avoid_policy
    wins = 0
    for sd in range(1, 7):
        env = make("site", "smoke", "t_avoid", mission="C", command_mode="ctbr")
        pol = avoid_policy(); obs, _ = env.reset(seed=sd); pol.reset()
        while True:
            obs, r, te, tr, info = env.step(pol.act(obs))
            if te or tr:
                break
        wins += bool(info["success"])
    assert wins >= 3
