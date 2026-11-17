"""`screen`: several programs in one world model and in the environment; table, JSON and scatter plot."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

import harness_support as hs
from harness import cli, config, report, screen


def test_screen_of_the_toy_example_from_its_config(tmp_path, capsys):
    out = tmp_path / "out"
    code = cli.main(["screen", "--config", str(hs.TOY / "run.toml"), "--out", str(out), "--model-episodes", "100",
                     "--real-seeds", "1000-1059", "--workers", "4"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "| reference |" in printed and "| poor |" in printed
    data = hs.read_json(out / "screen.json")
    assert data["environment"] == "puck table" and data["world"] == str(hs.REFERENCE / "world.py")
    assert data["model_seeds"] == [100000, 100099] and data["real_seeds"] == [1000, 1059]
    rows = {r["name"]: r for r in data["policies"]}
    assert list(rows) == ["reference", "poor"] and rows["poor"]["file"] == str(hs.REFERENCE / "policy_poor.py")
    for r in rows.values():
        assert r["model"]["episodes"] == 100 and r["real"]["episodes"] == 60
        assert r["model"]["flagged"] == r["real"]["flagged"] == 0 and r["paired"] is None
    # the world model ranks the two programs as the environment does
    assert rows["reference"]["model"]["success_rate"] > rows["poor"]["model"]["success_rate"] + 0.2
    assert rows["reference"]["real"]["success_rate"] > rows["poor"]["real"]["success_rate"] + 0.2
    assert data["summary"]["programs"] == 2 and data["summary"]["pearson_r"] is None
    expected = sum(abs(r["model"]["success_rate"] - r["real"]["success_rate"]) for r in rows.values()) / 2
    assert data["summary"]["mean_abs_difference"] == pytest.approx(expected)
    episodes = hs.read_json(out / "screen_episodes.json")
    assert len(episodes["poor"]["model"]) == 100 and len(episodes["poor"]["real"]) == 60

    svg = ET.parse(out / "screen.svg").getroot()
    assert svg.tag == "{http://www.w3.org/2000/svg}svg"
    assert len(svg.findall("{http://www.w3.org/2000/svg}circle")) == 2
    labels = [t.text for t in svg.iter("{http://www.w3.org/2000/svg}text")]
    assert "reference" in labels and "poor" in labels and "real success rate (environment)" in labels
    table = (out / "screen.md").read_text()
    assert "| program | predicted (world model) |" in table and "| mean absolute difference |" in table
    assert not (hs.TOY / "runs" / "toy" / "screen").exists()         # --out was respected


def test_screen_with_files_from_the_command_line_paired_and_a_broken_program(make_run, tmp_path, capsys):
    cfg = make_run(init=False)
    broken = tmp_path / "broken.py"
    broken.write_text("def make_policy():\n    raise RuntimeError('not today')\n")
    crashing = tmp_path / "crashing.py"
    crashing.write_text(hs.PLAIN_POLICY.replace("return np.clip(", "raise ValueError('mid-air')\n        return np.clip("))
    out = tmp_path / "out"
    code = cli.main(["screen", "--config", cfg["config_file"], "--out", str(out), "--world", str(hs.REFERENCE / "world.py"),
                     "--policy", str(hs.REFERENCE / "policy.py"), "--policy", str(broken), "--policy", str(crashing),
                     "--name", "good", "--name", "broken", "--name", "crashing", "--model-episodes", "10",
                     "--real-seeds", "1000-1007", "--paired"])
    assert code == 0
    rows = {r["name"]: r for r in hs.read_json(out / "screen.json")["policies"]}
    assert rows["good"]["real"]["episodes"] == 8 and rows["good"]["real"]["flagged"] == 0
    assert rows["good"]["paired"]["episodes"] == 8 and 0.0 <= rows["good"]["paired"]["agreement"] <= 1.0
    assert rows["broken"]["real"]["flags"] == {"load_error": 8} and "not today" in rows["broken"]["real"]["error"]
    assert rows["broken"]["model"]["flags"] == {"load_error": 10} and rows["broken"]["model"]["error_in"] == "policy"
    assert rows["broken"]["paired"] is None
    assert rows["crashing"]["real"]["flags"] == {"policy_error": 8} and rows["crashing"]["model"]["flags"] == {"policy_error": 10}
    table = (out / "screen.md").read_text()
    assert "not today" in table and "mid-air" in table and "| predicted from real resets |" in table
    assert "outputs:" in capsys.readouterr().out


def test_screen_world_only_environment_only_and_mistakes(make_run, tmp_path, capsys):
    cfg = make_run(init=False)
    policy = str(hs.REFERENCE / "policy_poor.py")
    out = screen.screen(cfg, world=hs.REFERENCE / "world.py", policies=[policy], out_dir=tmp_path / "a",
                        model_episodes=5, skip_real=True)
    assert out["policies"][0]["real"] is None and out["policies"][0]["model"]["episodes"] == 5
    assert out["policies"][0]["name"] == "policy_poor" and out["files"]["plot"] is None
    assert not (tmp_path / "a" / "screen.svg").exists()
    out = screen.screen(cfg, policies=[policy, policy], out_dir=tmp_path / "b", real_seeds=[1000, 1500])
    assert [r["name"] for r in out["policies"]] == ["policy_poor", "policy_poor_"]
    assert out["policies"][0]["model"] is None and out["policies"][0]["real"]["episodes"] == 2
    assert cli.main(["screen", "--config", cfg["config_file"], "--out", str(tmp_path / "c")]) == 2
    assert "no policy files" in capsys.readouterr().err
    assert cli.main(["screen", "--config", cfg["config_file"], "--policy", str(tmp_path / "missing.py")]) == 2
    assert "no file" in capsys.readouterr().err
    assert not config.is_initialised(cfg)                 # screen needs no run directory


def test_scatter_and_statistics():
    assert report.pearson([0.0, 0.5, 1.0], [0.1, 0.5, 0.9]) == pytest.approx(1.0)
    assert report.pearson([0.0, 1.0], [0.0, 1.0]) is None and report.pearson([0.5, 0.5, 0.5], [0.1, 0.2, 0.3]) is None
    assert report.mean_abs_error([0.2, 0.8], [0.3, 0.6]) == pytest.approx(0.15)
    svg = report.scatter_svg([("a<b", 0.25, 0.75), ("c", 1.0, 0.0)], "x & y", "real", "note")
    root = ET.fromstring(svg)
    circles = root.findall("{http://www.w3.org/2000/svg}circle")
    assert [(float(c.get("cx")), float(c.get("cy"))) for c in circles] == [(170.0, 130.0), (470.0, 430.0)]
    assert "a&lt;b" in svg and "x &amp; y" in svg
    crowded = ET.fromstring(report.scatter_svg([("v1", 0.9, 0.9), ("v2", 0.9, 0.9), ("v3", 0.9, 0.9)], "x", "y"))
    heights = [float(t.get("y")) for t in crowded.iter("{http://www.w3.org/2000/svg}text") if t.text in ("v1", "v2", "v3")]
    assert len(set(heights)) == 3                         # labels of coinciding points are moved apart
    assert {c.get("cx") for c in crowded.findall("{http://www.w3.org/2000/svg}circle")} == {"430.0"}
