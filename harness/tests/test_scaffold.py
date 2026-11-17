"""`init`: what goes into the workspace, what stays out, and the checks of the config."""
from __future__ import annotations

import ast
import re

import pytest

import harness_support as hs
from harness import budget, config, scaffold

EXPECTED = {"BRIEF.md", "MANUAL.md", "task_card.json", "lab.py"}


def _workspace_files(cfg):
    ws = config.paths(cfg).workspace
    return {str(f.relative_to(ws)) for f in ws.rglob("*") if f.is_file()}


def test_init_creates_workspace_and_state(make_run):
    cfg = make_run()
    p = config.paths(cfg)
    assert _workspace_files(cfg) == EXPECTED
    assert all((p.workspace / d).is_dir() for d in ("trials", "notes", ".lab"))
    assert (p.workspace / "lab.py").read_bytes() == (hs.HARNESS / "client" / "lab.py").read_bytes()
    manual = (p.workspace / "MANUAL.md").read_text()
    assert manual.startswith("PUCK TABLE") and "SUCCESS" in manual
    assert hs.read_json(p.workspace / "task_card.json")["max_steps"] == 80
    for f in (p.config_json, p.config_toml, p.environment, p.budget):
        assert f.is_file()
    assert budget.read(p.budget) == {"trial_budget": 3, "trials": [], "hypothesis_at_first_trial": None}
    assert hs.read_json(p.environment)["max_steps"] == 80


def test_workspace_holds_no_environment_source_and_no_harness_state(make_run):
    cfg = make_run()
    p = config.paths(cfg)
    assert p.state not in p.workspace.parents and p.workspace not in p.state.parents
    hidden = ["toy_env", "_GAIN", "_FRICTION", "_DRIFT", "_NOISE", "hidden physics", "0.35, -0.20", "held_out_seed_range",
              str(hs.TOY)]
    for f in p.workspace.rglob("*"):
        if f.is_file():
            assert f.suffix not in (".toml",) and f.name not in ("config.json", "budget.json", "environment.json")
            text = f.read_text(errors="replace")
            for needle in hidden:
                assert needle not in text, f"{f.name} mentions {needle!r}"
    # the only Python file is the client, and it does not import the environment
    assert [f.name for f in p.workspace.rglob("*.py")] == ["lab.py"]
    tree = ast.parse((p.workspace / "lab.py").read_text())
    imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imports |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imports <= {"__future__", "base64", "hashlib", "hmac", "json", "os", "socket", "struct", "pathlib", "numpy"}


@pytest.mark.parametrize("role", ["world_model", "cap", "cap_with_tests"])
def test_every_brief_template_renders(make_run, role):
    extra = {"workspace": {"files": {"world.py": str(hs.REFERENCE / "world.py")}}} if role == "cap_with_tests" else {}
    cfg = make_run({"agent": {"role": role, "timeout_s": 3600, "python": "python3"}, **extra})
    p = config.paths(cfg)
    brief = (p.workspace / "BRIEF.md").read_text()
    assert "{{" not in brief and "}}" not in brief
    assert '"puck table"' in brief and "**3 real trials**" in brief and "from 0 to 99" in brief
    assert "at most 80 steps" in brief and "45 minutes" in brief and "python3 -c" in brief
    assert "4 held-out situations" in brief and "20 seconds" in brief
    assert ("world.py" in _workspace_files(cfg)) == (role == "cap_with_tests")
    if role == "cap":
        assert "prediction.json" not in brief and "simulator or forward model" in brief
    else:
        assert "prediction.json" in brief and "6 episodes" in brief
    assert not re.search(r"[^\x00-\x7f]", brief)            # plain ASCII, whatever the template


def test_custom_brief_and_unknown_placeholder(make_run, tmp_path):
    good = tmp_path / "brief.md"
    good.write_text("Env {{environment_name}}, {{ trial_budget }} trials, seeds {{seed_lo}}-{{seed_hi}}. JSON: {\"a\": {\"b\": 1}}\n")
    cfg = make_run({"agent": {"brief": str(good)}})
    assert (config.paths(cfg).workspace / "BRIEF.md").read_text() == \
        "Env puck table, 3 trials, seeds 0-99. JSON: {\"a\": {\"b\": 1}}\n"
    bad = tmp_path / "bad.md"
    bad.write_text("{{no_such_thing}}")
    with pytest.raises(scaffold.ScaffoldError) as err:
        make_run({"agent": {"brief": str(bad)}})
    assert "no_such_thing" in str(err.value)


def test_extra_workspace_files_are_copied(make_run):
    cfg = make_run({"workspace": {"files": {"given/notes.json": str(hs.REFERENCE / "notes" / "hypothesis.json")}}})
    assert "given/notes.json" in _workspace_files(cfg)


def test_init_refuses_an_existing_run_or_a_used_workspace(make_run, tmp_path):
    cfg = make_run()
    with pytest.raises(scaffold.ScaffoldError):
        scaffold.init(cfg)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "something.txt").write_text("x")
    with pytest.raises(scaffold.ScaffoldError):
        make_run({"run": {"workspace": str(other)}})


def test_workspace_can_live_away_from_the_state(make_run, tmp_path):
    cfg = make_run({"run": {"workspace": str(tmp_path / "far" / "ws")}})
    p = config.paths(cfg)
    assert p.workspace == tmp_path / "far" / "ws" and (p.workspace / "BRIEF.md").is_file()
    assert p.state.parent != p.workspace.parent
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        assert lab.status()["trials_left"] == 3


@pytest.mark.parametrize("overrides, message", [
    ({"budget": {"seed_range": [0, 1000]}}, "overlap"),
    ({"budget": {"held_out_episodes": 5000}}, "held_out_episodes"),
    ({"budget": {"trials": -1}}, "trials"),
    ({"budget": {"seed_range": [5, 1]}}, "seed_range"),
    ({"budget": {"typo": 1}}, "typo"),
    ({"nonsense": {"a": 1}}, "nonsense"),
    ({"environment": {"factory": "no_colon"}}, "factory"),
    ({"agent": {"kind": "telepathy"}}, "kind"),
    ({"agent": {"role": "wizard"}}, "role"),
    ({"agent": {"kind": "command"}}, "command"),
    ({"agent": {"kind": "replay"}}, "script"),
    ({"agent": {"role": "cap_with_tests"}}, "world.py"),
    ({"workspace": {"files": {"../escape.py": "x"}}}, "inside the workspace"),
    ({"verify": {"workers": 0}}, "workers"),
    ({"server": {"transport": "carrier pigeon"}}, "transport"),
])
def test_config_mistakes_are_reported(make_run, overrides, message):
    with pytest.raises(config.ConfigError) as err:
        make_run(overrides, init=False)
    assert message in str(err.value)


def test_defaults_and_paths_are_resolved(make_run):
    cfg = make_run(init=False)
    assert cfg["verify"]["model_seed_first"] == 100000 and cfg["verify"]["paired"] is True
    assert cfg["agent"]["timeout_s"] == 7200 and cfg["agent"]["time_allowance_min"] == 90
    assert cfg["server"]["transport"] == "auto" and cfg["budget"]["held_out_episodes"] == 4
    assert config.held_out_seeds(cfg) == [1000, 1001, 1002, 1003]
    assert config.model_seeds(cfg) == list(range(100000, 100006))
    assert cfg["environment"]["cwd"] == str(config.paths(cfg).run.parent)
    assert config.brief_template(cfg).name == "world_model.md"
