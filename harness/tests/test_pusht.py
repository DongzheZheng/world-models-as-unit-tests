"""The pushing example: the study's recorded numbers, reproduced through the generic `screen` code path.

Recorded:  runs/probe/screen_pusht.json   success of policy_v1..v8 in the world model, 200 episodes, seeds 50000..50199
           runs/probe/curves.json         success of the same programs in the environment, 50 episodes, seeds 6000..6049
"""
from __future__ import annotations

import importlib.util

import pytest

import harness_support as hs
from harness import cli

CONFIG = hs.HARNESS / "examples" / "pusht" / "screen.toml"
PROGRAMS = hs.ROOT / "pilots" / "aspire_pusht"

pytestmark = pytest.mark.skipif(any(importlib.util.find_spec(m) is None for m in ("torch", "pymunk", "shapely")),
                                reason="needs the repository's requirements (torch, pymunk, shapely)")


def _recorded():
    model = hs.read_json(hs.ROOT / "runs" / "probe" / "screen_pusht.json")["T"]
    real = hs.read_json(hs.ROOT / "runs" / "probe" / "curves.json")
    return model, {f"v{k}": real[f"push_T/aspire_v{k}"] for k in range(1, 9)}


def test_adapters_reproduce_recorded_real_episodes_one_by_one(tmp_path):
    """Quick: two programs, eight recorded real episodes each, and a few model episodes."""
    out = tmp_path / "out"
    code = cli.main(["screen", "--config", str(CONFIG), "--out", str(out), "--workers", "8", "--model-episodes", "8",
                     "--real-seeds", "6000-6007", "--policy", str(PROGRAMS / "policy_v4.py"),
                     "--policy", str(PROGRAMS / "policy_v5.py"), "--name", "v4", "--name", "v5"])
    assert code == 0
    _, real = _recorded()
    episodes = hs.read_json(out / "screen_episodes.json")
    for name in ("v4", "v5"):
        got = {str(e["seed"]): e["success"] for e in episodes[name]["real"]}
        assert got == {s: real[name]["per_seed"][s] for s in got}, name
        assert all(e["flag"] is None for e in episodes[name]["real"] + episodes[name]["model"])
        assert [e["seed"] for e in episodes[name]["model"]] == list(range(50000, 50008))
    rows = {r["name"]: r for r in hs.read_json(out / "screen.json")["policies"]}
    assert rows["v4"]["model"]["format_problems"] == []           # the adapter speaks the environment's format


@pytest.mark.slow
def test_screen_reproduces_the_recorded_numbers_of_all_eight_programs(tmp_path):
    out = tmp_path / "out"
    assert cli.main(["screen", "--config", str(CONFIG), "--out", str(out), "--workers", "8"]) == 0
    data = hs.read_json(out / "screen.json")
    assert data["model_seeds"] == [50000, 50199] and data["real_seeds"] == [6000, 6049]
    model, real = _recorded()
    rows = {r["name"]: r for r in data["policies"]}
    assert list(rows) == [f"v{k}" for k in range(1, 9)]
    table = []
    for name, row in rows.items():
        assert row["model"]["episodes"] == model[name]["n"] == 200 and row["real"]["episodes"] == real[name]["n"] == 50
        assert row["model"]["flagged"] == 0 and row["real"]["flagged"] == 0
        table.append((name, row["model"]["success_rate"], model[name]["success"], row["real"]["success_rate"],
                      real[name]["success"]))
    print("\nprogram  model: reproduced recorded   environment: reproduced recorded")
    for name, m, m_rec, r, r_rec in table:
        print(f"{name:7s}        {m:10.3f} {m_rec:8.3f}                {r:10.3f} {r_rec:8.3f}")
    for name, m, m_rec, r, r_rec in table:
        assert m == pytest.approx(m_rec, abs=1e-12) and r == pytest.approx(r_rec, abs=1e-12), name
    episodes = hs.read_json(out / "screen_episodes.json")
    for name in rows:                                           # the real episodes agree one by one with the record
        got = {str(e["seed"]): e["success"] for e in episodes[name]["real"]}
        assert got == real[name]["per_seed"], name
    assert (out / "screen.svg").is_file() and data["summary"]["programs"] == 8
