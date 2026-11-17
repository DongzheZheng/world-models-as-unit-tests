"""The environment server seen from the workspace (lab.py) and from the harness (control channel)."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import harness_support as hs
from harness import budget, config, evaluate, scaffold


def _episode(lab, seed, action=(0.2, -0.1)):
    obs = lab.reset(seed)
    done, info, n = False, {}, 0
    while not done:
        obs, done, info = lab.step(list(action))
        n += 1
    return obs, info, n


def test_round_trip_and_trial_record(make_run):
    cfg = make_run()
    p = config.paths(cfg)
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        status = lab.status()
        assert status == {"max_steps": 80, "episode_running": False, "steps_in_episode": 0, "trial_budget": 3,
                          "trials_used": 0, "trials_left": 3, "seed_range": [0, 99]}
        assert lab.task_card() == hs.read_json(p.workspace / "task_card.json")

        obs = lab.reset(5)
        assert set(obs) == {"pos", "vel", "target", "t"}
        assert isinstance(obs["pos"], np.ndarray) and obs["pos"].shape == (2,) and obs["t"] == 0
        obs2, done, info = lab.step(np.array([1.0, 0.0]))
        assert done is False and obs2["t"] == 1 and set(info) == {"distance", "speed"}
        assert lab.status()["episode_running"] and lab.status()["steps_in_episode"] == 1
        while not done:
            obs2, done, info = lab.step([1.0, 0.0])
        assert obs2["t"] == 80 and isinstance(info["success"], bool)
        assert lab.status()["trials_used"] == 1 and not lab.status()["episode_running"]

        same = lab.reset(5)                               # the same seed gives the same start
        assert np.array_equal(same["pos"], obs["pos"]) and np.array_equal(same["target"], obs["target"])

        record = lab.load_trial(1)
        assert record["pos"].shape == (81, 2) and record["vel"].shape == (81, 2) and record["t"].shape == (81,)
        assert record["actions"].shape == (80, 2) and np.array_equal(record["actions"][0], [1.0, 0.0])
        assert np.array_equal(record["pos"][0], obs["pos"]) and np.array_equal(record["pos"][80], obs2["pos"])
        assert int(record["seed"]) == 5 and int(record["trial"]) == 1 and int(record["n_steps"]) == 80
        assert bool(record["finished"]) and bool(record["success"]) == info["success"]
        assert json.loads(str(record["info_json"]))["success"] == info["success"]
        assert [f.name for f in lab.trial_files()] == ["trial_1.npz"]
    # the harness keeps its own copy outside the workspace, and the counter has the trial
    assert (p.trials / "trial_1.npz").read_bytes() == (p.workspace / "trials" / "trial_1.npz").read_bytes()
    counter = budget.read(p.budget)
    assert [(t["trial"], t["seed"], t["outcome"], t["n_steps"]) for t in counter["trials"]] == \
        [(1, 5, "finished", 80), (2, 5, "abandoned", 0)]
    assert counter["trials"][0]["record"] == "trial_1.npz" and counter["trials"][1]["record"] == "trial_2.npz"


def test_run_trial_and_env_object(make_run):
    cfg = make_run()

    class Push:
        def reset(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            return [0.0, 0.0]

    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        policy = Push()
        result = lab.run_trial(policy, 3)
        assert result["n_steps"] == 80 and policy.calls == 80 and result["seed"] == 3
        assert result["success"] == result["info"]["success"]
        env = lab.Env()
        assert env.max_steps == 80
        obs = env.reset(4)
        obs, done, info = env.step([0.0, 0.0])
        assert obs["t"] == 1 and not done


def test_budget_is_enforced_and_abandoned_episodes_count(make_run):
    cfg = make_run({"budget": {"trials": 2}})
    p = config.paths(cfg)
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        lab.reset(1)
        lab.step([0.0, 0.0])
        lab.reset(2)                                      # abandons trial 1 after one step
        for _ in range(3):
            lab.step([0.0, 0.0])
        with pytest.raises(lab.BudgetExceeded) as err:
            lab.reset(3)
        assert "2" in str(err.value)
        status = lab.status()
        assert status["trials_used"] == 2 and status["trials_left"] == 0
        assert status["episode_running"] and status["steps_in_episode"] == 3      # the refused reset changed nothing
        lab.step([0.0, 0.0])
        with pytest.raises(lab.BudgetExceeded):
            lab.reset(4)
        first = lab.load_trial(1)
        assert not bool(first["finished"]) and int(first["n_steps"]) == 1 and first["pos"].shape == (2, 2)
    counter = budget.read(p.budget)
    assert [(t["outcome"], t["n_steps"]) for t in counter["trials"]] == [("abandoned", 1), ("abandoned", 4)]
    assert sorted(f.name for f in (p.workspace / "trials").iterdir()) == ["trial_1.npz", "trial_2.npz"]


def test_budget_survives_a_restart_of_the_server(make_run):
    cfg = make_run({"budget": {"trials": 2}})
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        _episode(lab, 1)
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        assert lab.status()["trials_used"] == 1
        lab.reset(2)
        with pytest.raises(lab.BudgetExceeded):
            lab.reset(3)
    assert len(budget.read(config.paths(cfg).budget)["trials"]) == 2


@pytest.mark.parametrize("seed", [100, 1000, 1001, 1999, -1, 10 ** 12, 1.0, "3", None, True, [1]])
def test_seeds_outside_the_range_are_refused_and_cost_nothing(make_run, seed):
    cfg = make_run()
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        with pytest.raises(lab.SeedNotAllowed):
            lab.reset(seed)
        status = lab.status()
        assert status["trials_used"] == 0 and not status["episode_running"]
    assert budget.read(config.paths(cfg).budget)["trials"] == []
    assert list((config.paths(cfg).workspace / "trials").iterdir()) == []


def test_only_the_control_channel_reaches_held_out_seeds(make_run):
    cfg = make_run()
    p = config.paths(cfg)
    with hs.workspace_server(cfg) as server, hs.lab_client(cfg) as lab:
        with pytest.raises(lab.SeedNotAllowed):
            lab.reset(1000)
        reply = server.call("reset", seed=1000)           # the harness's channel: an inherited socket
        assert reply["obs"]["pos"].shape == (2,) and "trial" not in reply
        reply = server.call("step", action=[0.0, 0.0])
        assert reply["obs"]["t"] == 1
        with pytest.raises(lab.NoActiveTrial):            # the workspace cannot step the harness's episode
            lab.step([0.0, 0.0])
        assert lab.status()["trials_used"] == 0
    assert budget.read(p.budget)["trials"] == [] and list((p.workspace / "trials").iterdir()) == []
    # nothing that gives access to the control channel is stored anywhere in the run directory
    names = [f.name for f in p.run.rglob("*") if f.is_file()]
    assert "connection.json" not in names and not any(n.endswith(".sock") for n in names)


def test_step_needs_a_running_episode(make_run):
    cfg = make_run()
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        with pytest.raises(lab.NoActiveTrial):
            lab.step([0.0, 0.0])
        _episode(lab, 1)
        with pytest.raises(lab.NoActiveTrial):
            lab.step([0.0, 0.0])
        assert lab.status()["trials_used"] == 1


def test_a_rejected_action_gives_a_message_without_internals_and_keeps_the_episode(make_run):
    cfg = make_run()
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        lab.reset(1)
        for bad in ([1.0, 2.0, 3.0], [float("nan"), 0.0], "push", None, {"fx": 1.0}):
            with pytest.raises(lab.LabError) as err:
                lab.step(bad)
            text = str(err.value)
            assert not isinstance(err.value, lab.NoActiveTrial)
            assert "Traceback" not in text and "toy_env" not in text and "/" not in text and ".py" not in text
        obs, done, info = lab.step([0.0, 0.0])            # the episode is still open, no step was taken
        assert obs["t"] == 1
        with pytest.raises(lab.LabError):
            lab._call("no_such_request")
        assert lab.status()["steps_in_episode"] == 1


def test_a_closed_connection_abandons_its_episode_and_saves_the_record(make_run):
    cfg = make_run()
    p = config.paths(cfg)
    with hs.workspace_server(cfg):
        code = "import lab; lab.reset(7); lab.step([1.0, 1.0]); lab.step([1.0, 1.0])"
        done = subprocess.run([sys.executable, "-c", code], cwd=p.workspace, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        with hs.lab_client(cfg) as lab:
            for _ in range(100):                          # the server notices the closed connection at once
                if not lab.status()["episode_running"]:
                    break
            assert not lab.status()["episode_running"]
            record = lab.load_trial(1)
            assert int(record["n_steps"]) == 2 and not bool(record["finished"]) and int(record["seed"]) == 7


def test_connection_file_exists_only_while_the_server_runs_and_a_wrong_key_is_refused(make_run):
    cfg = make_run()
    connection = config.paths(cfg).workspace / ".lab" / "connection.json"
    assert not connection.exists()
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        info = hs.read_json(connection)
        assert info["family"] in ("unix", "tcp") and len(info["key"]) == 64
        assert connection.stat().st_mode & 0o077 == 0
        assert lab.status()["trials_used"] == 0
        lab.close()
        connection.write_text(json.dumps({**info, "key": "00" * 32}))
        with pytest.raises(lab.NotConnected):
            lab.status()
        connection.write_text(json.dumps(info))
        assert lab.status()["trials_used"] == 0
    assert not connection.exists()
    with hs.lab_client(cfg) as lab:
        with pytest.raises(lab.NotConnected):
            lab.status()


@pytest.mark.parametrize("transport", ["unix", "tcp"])
def test_transports_also_with_a_long_workspace_path(tmp_path, transport):
    deep = tmp_path / ("d" * 60) / ("e e" * 20)           # longer than a socket path may be, and with spaces
    cfg = hs.new_run(deep, {"server": {"transport": transport}})
    assert len(str(config.paths(cfg).workspace)) > 120
    with hs.workspace_server(cfg) as server, hs.lab_client(cfg) as lab:
        assert server.info["endpoint"]["family"] == transport
        _, info, n = _episode(lab, 2)
        assert n == 80 and lab.status()["trials_used"] == 1
        code = "import lab; print(lab.status()['trials_used'])"       # and from a script started in the workspace
        out = subprocess.run([sys.executable, "-c", code], cwd=config.paths(cfg).workspace, capture_output=True,
                             text=True, timeout=60)
        assert out.stdout.strip() == "1", out.stderr


def test_environment_that_breaks_the_contract_is_reported_at_init(tmp_path):
    (tmp_path / "bad_env.py").write_text(
        "class Env:\n    manual = 'm'\n    def task_card(self): return {}\n    def reset(self, seed): return {}\n"
        "    def step(self, a): return {}, True, {}\n\ndef make(): return Env()\n\ndef boom(): raise RuntimeError('no robot')\n")
    env = {"factory": "bad_env:make", "paths": [str(tmp_path)]}
    with pytest.raises(scaffold.ScaffoldError) as err:
        hs.new_run(tmp_path / "a", {"environment": env})
    assert "max_steps" in str(err.value)
    with pytest.raises(scaffold.ScaffoldError) as err:
        hs.new_run(tmp_path / "b", {"environment": {**env, "factory": "bad_env:boom"}})
    assert "no robot" in str(err.value)
    assert not (tmp_path / "a" / "run" / "state" / "config.json").exists()        # left uninitialised


def test_eval_server_reports_an_environment_that_cannot_start(make_run, tmp_path):
    cfg = make_run()
    p = config.paths(cfg)
    broken = hs.read_json(p.config_json)
    broken["environment"]["factory"] = "toy_env:does_not_exist"
    bad = tmp_path / "broken.json"
    bad.write_text(json.dumps(broken))
    with pytest.raises(evaluate.ServerError) as err:
        evaluate.ServerProcess(bad, "eval", tmp_path / "server.log", 30.0)
    assert "does_not_exist" in str(err.value)


def test_one_server_per_run_directory(make_run):
    cfg = make_run()
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        lab.reset(1)
        with pytest.raises(evaluate.ServerError) as err:
            with hs.workspace_server(cfg):
                pass
        assert "already running" in str(err.value)
        assert lab.step([0.0, 0.0])[0]["t"] == 1          # the first server and its episode are untouched
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:      # and after it stopped, a new one may start
        assert lab.status()["trials_used"] == 1


def test_step_limit_is_enforced_and_unstackable_records_are_saved_as_json(tmp_path):
    """An environment that never ends its episodes and whose observations change shape."""
    (tmp_path / "odd_env.py").write_text(
        "import numpy as np\n\nclass Env:\n    manual = 'An odd environment.'\n    max_steps = 5\n"
        "    def task_card(self): return {'steps': 5}\n"
        "    def reset(self, seed):\n        self.t = 0\n        return {'items': np.zeros(1)}\n"
        "    def step(self, action):\n        self.t += 1\n        return {'items': np.full(self.t + 1, float(action))}, False, {'note': 'never done'}\n"
        "\ndef make(): return Env()\n")
    cfg = hs.new_run(tmp_path / "odd", {"environment": {"factory": "odd_env:make", "paths": [str(tmp_path)]}})
    with hs.workspace_server(cfg), hs.lab_client(cfg) as lab:
        assert lab.status()["max_steps"] == 5 and lab.task_card() == {"steps": 5}
        lab.reset(0)
        for k in range(5):
            obs, done, info = lab.step(k)
            assert done == (k == 4)
        assert info == {"note": "never done", "success": False, "ended_by_step_limit": True}
        with pytest.raises(lab.NoActiveTrial):
            lab.step(0)
        assert [f.name for f in lab.trial_files()] == ["trial_1.json"]
        record = lab.load_trial(1)
        assert record["n_steps"] == 5 and record["finished"] is True and record["success"] is False
        assert [o["items"].shape for o in record["observations"]] == [(1,), (2,), (3,), (4,), (5,), (6,)]
        assert record["actions"] == [0, 1, 2, 3, 4] and record["observations"][3]["items"].tolist() == [2.0] * 4
    counter = budget.read(config.paths(cfg).budget)
    assert counter["trials"][0]["record"] == "trial_1.json" and counter["trials"][0]["outcome"] == "finished"
