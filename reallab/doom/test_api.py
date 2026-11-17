import gzip
import json
import os

import numpy as np
import pytest

from cawm import ledger
from reallab.doom import api, parallel

SC = sorted(api.SCENARIOS)
PLAYER_KEYS = {"x", "y", "z", "angle", "pitch", "vx", "vy", "vz", "health", "armor", "selected_weapon", "ammo",
               "attack_ready", "killcount", "itemcount", "dead", "on_ground", "damage_taken", "hits_taken",
               "damage_dealt", "hits_dealt"}
OBJ_KEYS = {"id", "name", "x", "y", "z", "angle", "vx", "vy", "vz", "sx", "sy", "sw", "sh"}


@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    p = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("CAWM_LEDGER", str(p))
    monkeypatch.setenv("CAWM_BLACKBOX_DIR", str(tmp_path / "blackbox"))
    return p


def _check_obs(o):
    assert set(o) == {"tic", "player", "objects", "depth_scan"}
    assert isinstance(o["tic"], int)
    assert set(o["player"]) == PLAYER_KEYS
    assert o["depth_scan"].shape == (64,) and o["depth_scan"].dtype == np.float32
    for d in o["objects"]:
        assert set(d) == OBJ_KEYS and d["name"] != "DoomPlayer"


@pytest.mark.parametrize("sc", SC)
def test_obs_schema(sc):
    e = api.make(sc, "smoke", seed=0)
    assert e.action_names[0] == "noop" and len(e.action_names) >= 3
    o = e.reset(1)
    _check_obs(o)
    rng = np.random.default_rng(0)
    for _ in range(30):
        o, r, te, tr, info = e.step(int(rng.integers(len(e.action_names))))
        _check_obs(o)
        assert isinstance(r, float) and "gt_objects" not in info
        if te or tr:
            break
    e.close()


@pytest.mark.parametrize("sc", SC)
def test_determinism(sc):
    def run():
        e = api.make(sc, "smoke")
        o = e.reset(11)
        rng = np.random.default_rng(5)
        out = [(o["tic"], o["player"], o["objects"], o["depth_scan"].tolist())]
        for _ in range(60):
            o, r, te, tr, _ = e.step(int(rng.integers(len(e.action_names))))
            out.append((r, o["tic"], o["player"], o["objects"], o["depth_scan"].tolist()))
            if te or tr:
                break
        e.close()
        return json.dumps(out, sort_keys=True)
    assert run() == run()


def test_meter_counts(tmp_ledger):
    e = api.make("take_cover", "smoke", tag="t")
    e.reset(1)
    n = 0
    for _ in range(5):
        e.step(0)
        n += 1
    e.reset(2)          # abandons episode 1 mid-way -> counted
    e.step(0)
    n += 1
    e.close()           # abandons episode 2 -> counted
    t = ledger.totals(tmp_ledger)[("doom", "take_cover", "smoke")]
    assert t["steps"] == n == 6 and t["episodes"] == 2


def test_meter_counts_finished_episode(tmp_ledger):
    e = api.make("basic", "smoke")
    e.reset(1)
    n = 0
    while True:
        _, _, te, tr, _ = e.step(0)
        n += 1
        if te or tr:
            break
    e.close()
    t = ledger.totals(tmp_ledger)[("doom", "basic", "smoke")]
    assert t["steps"] == n and t["episodes"] == 1
    assert tr and not te   # no-op in basic: scenario timeout


def test_calibration_cap(tmp_ledger, monkeypatch):
    monkeypatch.setattr(api, "CAL_CAP_STEPS", 12)
    e = api.make("take_cover", "calibration", pilot="p")
    e.reset(1)
    with pytest.raises(ledger.BudgetExceeded):
        for _ in range(100):
            _, _, te, tr, _ = e.step(0)
            if te or tr:
                e.reset(2)
    assert ledger.totals(tmp_ledger)[("doom", "take_cover:p", "calibration")]["steps"] == 12


def test_gt_objects_only_in_calibration():
    e = api.make("health_gathering", "calibration", pilot="p")
    o = e.reset(1)
    _, _, _, _, info = e.step(0)
    assert len(info["gt_objects"]) >= len(o["objects"])
    assert set(info["gt_objects"][0]) == OBJ_KEYS | {"visible"}
    e.close()
    for p in ("eval", "smoke", "oracle_train"):
        e = api.make("health_gathering", p)
        e.reset(1)
        _, _, _, _, info = e.step(0)
        assert "gt_objects" not in info
        with pytest.raises(PermissionError):
            e.last_gt_objects
        e.close()


def test_floor_plan():
    fp = api.floor_plan("health_gathering")
    assert len(fp["walls"]) >= 4 and fp["bounds"]["xmax"] > fp["bounds"]["xmin"]
    assert {"floor", "ceiling", "lines"} <= set(fp["sectors"][0])


def test_evaluate_four_workers(tmp_path, tmp_ledger):
    seeds = [1, 2, 3, 4, 5, 6]
    res = parallel.evaluate("reallab.doom.parallel:random_policy", {"seed": 0}, "take_cover", seeds,
                            purpose="smoke", n_workers=4, log_dir=str(tmp_path / "logs"))
    assert [r["seed"] for r in res] == seeds
    assert all({"ret", "length_tics", "n_steps", "terminated", "truncated", "player"} <= set(r) for r in res)
    t = ledger.totals(tmp_ledger)[("doom", "take_cover", "smoke")]
    assert t["steps"] == sum(r["n_steps"] for r in res) and t["episodes"] == 6
    with gzip.open(tmp_path / "logs" / "ep_1.jsonl.gz", "rt") as f:
        lines = [json.loads(l) for l in f]
    assert "meta" in lines[0] and "reset" in lines[1] and len(lines) == 2 + res[0]["n_steps"]
    # reproducible regardless of worker assignment
    res2 = parallel.evaluate("reallab.doom.parallel:random_policy", {"seed": 0}, "take_cover", seeds,
                             purpose="smoke", n_workers=2)
    assert [r["ret"] for r in res] == [r["ret"] for r in res2]


def test_evaluate_calibration_logs_gt(tmp_path):
    res = parallel.evaluate("reallab.doom.parallel:noop_policy", {}, "basic", [1], purpose="calibration", pilot="p",
                            n_workers=8, log_dir=str(tmp_path))
    with gzip.open(tmp_path / "ep_1.jsonl.gz", "rt") as f:
        lines = [json.loads(l) for l in f]
    assert "gt_objects" in lines[1]["reset"]["info"] and "gt_objects" in lines[2]["info"]
    assert res[0]["truncated"]


def test_five_lives_per_pilot_and_blackbox(tmp_path, tmp_ledger):
    with pytest.raises(ValueError):
        api.make("basic", "calibration")
    e = api.make("basic", "calibration", pilot="A")
    for k in range(1, 6):
        e.reset(k)
        for _ in range(3):
            e.step(0)
    with pytest.raises(ledger.BudgetExceeded):
        e.reset(6)
    e.close()
    assert ledger.totals(tmp_ledger)[("doom", "basic:A", "calibration")]["episodes"] == 5
    e2 = api.make("basic", "calibration", pilot="A")   # fresh env: same exhausted budget
    with pytest.raises(ledger.BudgetExceeded):
        e2.reset(7)
    e4 = api.make("basic", "calibration", pilot="B")   # another pilot has its own lives
    e4.reset(1)
    e4.close()
    for k in range(1, 6):
        f = tmp_path / "blackbox" / "basic" / "A" / f"life_{k}.jsonl.gz"
        with gzip.open(f, "rt") as fh:
            lines = [json.loads(l) for l in fh]
        assert lines[0]["meta"]["life"] == k and len(lines) == 2 + 3
        assert "gt_objects" in lines[1]["reset"]["info"] and "gt_objects" in lines[2]["info"]
    assert (tmp_path / "blackbox" / "basic" / "B" / "life_1.jsonl.gz").exists()
    # non-calibration: plain scenario key, pilot goes into the tag, no cap
    for _ in range(7):
        e3 = api.make("basic", "smoke", pilot="A")
        e3.reset(1)
        e3.close()
    assert ledger.totals(tmp_ledger)[("doom", "basic", "smoke")]["episodes"] == 7


def test_gif_and_mp4_recording(tmp_path):
    for fmt in ("gif", "mp4"):
        res = parallel.evaluate("reallab.doom.parallel:random_policy", {"seed": 0}, "take_cover", [3],
                                purpose="smoke", n_workers=1, record_first=1, record_format=fmt,
                                video_dir=str(tmp_path))
        f = tmp_path / f"ep_3.{fmt}"
        assert f.exists() and 0 < f.stat().st_size <= 6_000_000
        if fmt == "gif":
            from PIL import Image
            im = Image.open(f)
            assert im.n_frames >= res[0]["n_steps"]
    # recording does not change the trajectory
    plain = parallel.evaluate("reallab.doom.parallel:random_policy", {"seed": 0}, "take_cover", [3],
                              purpose="smoke", n_workers=1)
    assert plain[0]["ret"] == res[0]["ret"] and plain[0]["n_steps"] == res[0]["n_steps"]
