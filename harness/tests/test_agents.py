"""The agent adapters end to end: `run` with a stub command, with the replay script, and with an agent that hangs."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

import harness_support as hs
from harness import agents, budget, cli, config

STUB = Path(__file__).with_name("stub_agent.py")


def _run(cfg) -> int:
    return cli.main(["run", "--config", cfg["config_file"], "--run-dir", cfg["run"]["dir"]])


@pytest.mark.parametrize("name, program", [("claude_code.toml", "claude"), ("codex.toml", "codex")])
def test_agent_example_snippets_are_valid_config_sections(make_run, name, program):
    """The two examples are not executed (no tokens); they must at least load and expand."""
    text = (hs.HARNESS / "examples" / "agents" / name).read_text()
    assert "NOT EXECUTED" in text.splitlines()[0]
    section = tomllib.loads(text)
    assert list(section) == ["agent"]
    cfg = make_run({"agent": section["agent"]})
    argv = agents.build_command(cfg)
    p = config.paths(cfg)
    assert argv[0] == program and all("{" not in a or a.startswith('{"') for a in argv)
    joined = " ".join(argv)
    assert "BRIEF.md" in joined and str(p.workspace) in joined
    assert any("{state_dir}" in a for a in section["agent"]["command"]) == (str(p.state) in joined)


def test_serve_keeps_the_server_up_until_interrupted(make_run):
    cfg = make_run(init=False)
    p = config.paths(cfg)
    proc = subprocess.Popen([sys.executable, "-m", "harness", "serve", "--config", cfg["config_file"],
                             "--run-dir", cfg["run"]["dir"]], cwd=hs.ROOT, stdout=subprocess.PIPE, text=True)
    try:
        assert "environment server is up" in proc.stdout.readline()
        with hs.lab_client(cfg) as lab:
            lab.reset(3)
            lab.step([0.0, 0.0])
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0 and "real trials used: 1 of 3" in out
    assert not (p.workspace / ".lab" / "connection.json").exists()
    assert budget.read(p.budget)["trials"][0]["outcome"] == "abandoned"


def test_command_adapter_end_to_end_with_a_stub_agent(make_run, capsys):
    command = ["{python}", str(STUB), "{brief_file}", "{workspace}", str(hs.REFERENCE), "{brief_text}"]
    cfg = make_run({"agent": {"kind": "command", "command": command, "stdin": "brief", "timeout_s": 120,
                              "env": {"STUB_MARKER": "from-config"}},
                    "budget": {"held_out_episodes": 20}, "verify": {"model_episodes": 40}}, init=False)
    assert _run(cfg) == 0
    p = config.paths(cfg)
    out = capsys.readouterr().out
    assert "environment server is up" in out and "agent finished: exit code 0" in out and "| real trials used | 2 of 3 |" in out

    args = hs.read_json(p.workspace / "agent_args.json")
    assert args == {"cwd": str(p.workspace), "brief_file": str(p.workspace / "BRIEF.md"), "workspace": str(p.workspace),
                    "brief_text_is_the_brief": True, "stdin_is_the_brief": True, "budget_at_start": 3,
                    "trials_left_at_end": 1, "marker": "from-config"}
    agent = hs.read_json(p.agent)
    assert agent["exit_code"] == 0 and not agent["timed_out"] and agent["error"] is None
    assert agent["command"] == command                    # stored with its placeholders, not with the brief text
    assert "stub agent done" in Path(agent["stdout"]).read_text()
    assert p.logs in Path(agent["stdout"]).parents        # the logs are outside the workspace

    rep = hs.read_json(p.report_json)
    assert rep["trials"] == {"used": 2, "budget": 3, "hypothesis_at_first_trial": True}
    assert rep["agent"]["exit_code"] == 0
    assert [rep["deliverables"][f]["status"] for f in ("policy.py", "world.py", "prediction.json", "notes/hypothesis.json",
                                                       "notes/diagnosis.json")] == ["present"] * 4 + ["missing"]
    assert rep["predicted_by_agent"] == 0.91
    for part, n in (("model", 40), ("real", 20), ("model_from_held_out_starts", 20)):
        assert rep[part]["episodes"] == n and rep[part]["flagged"] == 0 and rep[part]["error"] is None
        assert 0.75 <= rep[part]["success_rate"] <= 1.0
        assert rep[part]["successes"] == round(rep[part]["success_rate"] * n)
    assert rep["real"]["seeds"] == [1000, 1019] and rep["model"]["seeds"] == [100000, 100039]
    assert rep["model"]["format_problems"] == []
    assert 0.5 <= rep["model_from_held_out_starts"]["agreement"] <= 1.0
    assert [t["seed"] for t in budget.read(p.budget)["trials"]] == [1, 2]
    assert sorted(f.name for f in (p.workspace / "trials").iterdir()) == ["trial_1.npz", "trial_2.npz"]
    assert not (p.workspace / ".lab" / "connection.json").exists()        # the server was stopped before verifying
    assert len(hs.read_json(p.verify / "real_episodes.json")) == 20
    assert "| policy.py | present | yes |" in p.report_md.read_text()

    assert _run(cfg) == 0                                 # a second run continues with the same budget
    assert hs.read_json(p.report_json)["trials"]["used"] == 3
    assert "exit code 1" in capsys.readouterr().out        # the stub ran into BudgetExceeded on its second trial


def test_replay_adapter_end_to_end_from_the_command_line(tmp_path):
    t0 = time.time()
    done = subprocess.run([sys.executable, "-m", "harness", "run", "--config", str(hs.TOY / "run.toml"),
                           "--run-dir", str(tmp_path / "toy")], cwd=hs.ROOT, capture_output=True, text=True, timeout=300)
    wall = time.time() - t0
    assert done.returncode == 0, done.stdout + done.stderr
    assert wall < 60, f"the toy run took {wall:.1f} s"
    rep = hs.read_json(tmp_path / "toy" / "report.json")
    assert rep["run"] == {"name": "toy", "role": "world_model", "environment": "puck table",
                          "config": str(hs.TOY / "run.toml")}
    assert rep["trials"] == {"used": 3, "budget": 5, "hypothesis_at_first_trial": True}
    assert rep["agent"]["kind"] == "replay" and rep["agent"]["exit_code"] == 0
    assert all(d["status"] == "present" for d in rep["deliverables"].values())
    assert rep["predicted_by_agent"] == 0.91
    assert rep["model"]["episodes"] == 200 and rep["real"]["episodes"] == 50
    assert rep["model"]["flagged"] == rep["real"]["flagged"] == 0
    assert 0.8 <= rep["model"]["success_rate"] <= 1.0 and 0.8 <= rep["real"]["success_rate"] <= 1.0
    assert abs(rep["model"]["success_rate"] - rep["real"]["success_rate"]) < 0.15
    assert rep["model_from_held_out_starts"]["episodes"] == 50
    counter = budget.read(tmp_path / "toy" / "state" / "budget.json")
    assert [(t["seed"], t["outcome"], t["n_steps"]) for t in counter["trials"]] == \
        [(11, "finished", 80), (12, "finished", 80), (13, "finished", 80)]
    snapshot = hs.read_json(tmp_path / "toy" / "state" / "hypothesis_at_first_trial.json")
    assert snapshot == hs.read_json(hs.REFERENCE / "notes" / "hypothesis.json")
    table = (tmp_path / "toy" / "report.md").read_text()
    assert "| real trials used | 3 of 5 |" in table and "| predicted by the agent (prediction.json) | 0.910 |" in table

    status = subprocess.run([sys.executable, "-m", "harness", "status", "--run-dir", str(tmp_path / "toy")],
                            cwd=hs.ROOT, capture_output=True, text=True, timeout=60)
    assert status.returncode == 0 and '"trials_used": 3' in status.stdout and '"server_running": false' in status.stdout


def test_replay_from_a_saved_record(make_run, tmp_path):
    first = make_run({"budget": {"trials": 1}})
    with hs.workspace_server(first), hs.lab_client(first) as lab:
        lab.reset(9)
        for k in range(80):
            lab.step([0.01 * k, -0.5])
    record = config.paths(first).trials / "trial_1.npz"
    script = tmp_path / "script.json"
    script.write_text('{"steps": [{"trial": {"record": "%s"}}, {"trial": {"seed": 4, "actions": [[0, 0], {"repeat": 3, "action": [1, 1]}]}},'
                      ' {"copy": {"policy.py": "%s"}}]}' % (record, hs.REFERENCE / "policy_poor.py"))
    cfg = make_run({"agent": {"kind": "replay", "script": str(script), "role": "cap"}})
    assert _run(cfg) == 0
    p = config.paths(cfg)
    counter = budget.read(p.budget)
    assert [(t["seed"], t["n_steps"], t["outcome"]) for t in counter["trials"]] == [(9, 80, "finished"), (4, 4, "abandoned")]
    assert (p.trials / "trial_1.npz").read_bytes() != b""
    with hs.lab_client(cfg) as lab:                       # the re-flown trial equals the recorded one
        import numpy as np
        again, original = lab.load_trial(1), np.load(record)
        assert np.array_equal(again["pos"], original["pos"]) and np.array_equal(again["actions"], original["actions"])
    rep = hs.read_json(p.report_json)
    assert rep["deliverables"]["policy.py"]["status"] == "present"
    assert rep["deliverables"]["world.py"] == {"status": "missing", "required": False, "detail": None}
    assert rep["model"] is None and rep["real"]["episodes"] == 4 and rep["predicted_by_agent"] is None


def test_agent_that_hangs_is_stopped_with_everything_it_started(make_run, capsys):
    code = ("import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
            "open('child.pid', 'w').write(str(child.pid))\n"
            "import lab; lab.reset(1); lab.step([0.0, 0.0])\n"
            "print('working', flush=True)\n"
            "time.sleep(300)\n")
    cfg = make_run({"agent": {"kind": "command", "command": [sys.executable, "-c", code], "timeout_s": 3}}, init=False)
    t0 = time.time()
    assert _run(cfg) == 0
    assert time.time() - t0 < 60
    p = config.paths(cfg)
    agent = hs.read_json(p.agent)
    assert agent["timed_out"] and agent["exit_code"] != 0 and 3 <= agent["wall_s"] < 30
    pid = int((p.workspace / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert "stopped at the time limit" in capsys.readouterr().out
    counter = budget.read(p.budget)                       # the trial it had started is counted and recorded
    assert [(t["outcome"], t["n_steps"]) for t in counter["trials"]] == [("abandoned", 1)]
    rep = hs.read_json(p.report_json)
    assert rep["agent"]["timed_out"] and rep["trials"]["used"] == 1
    assert rep["trials"]["hypothesis_at_first_trial"] is False
    assert rep["deliverables"]["policy.py"]["status"] == "missing"
    assert rep["real"] is None and rep["model"] is None and rep["predicted_by_agent"] is None


def test_agent_command_that_does_not_exist_or_fails(make_run, capsys):
    cfg = make_run({"agent": {"kind": "command", "command": ["/no/such/agent", "{brief_file}"]}}, init=False)
    assert _run(cfg) == 0
    agent = hs.read_json(config.paths(cfg).agent)
    assert agent["exit_code"] is None and "could not be started" in agent["error"]
    cfg = make_run({"agent": {"kind": "command", "command": [sys.executable, "-c", "import sys; sys.exit(7)"]}}, init=False)
    assert _run(cfg) == 0
    assert hs.read_json(config.paths(cfg).agent)["exit_code"] == 7
    assert "exit code 7" in capsys.readouterr().out


def test_kind_none_only_verifies_what_is_in_the_workspace(make_run):
    cfg = make_run(files=hs.reference_files())
    assert _run(cfg) == 0
    p = config.paths(cfg)
    assert hs.read_json(p.agent)["kind"] == "none"
    rep = hs.read_json(p.report_json)
    assert rep["trials"]["used"] == 0 and rep["real"]["episodes"] == 4 and rep["model"]["episodes"] == 6
