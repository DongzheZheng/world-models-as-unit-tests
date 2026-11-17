"""`verify`: the report for good deliverables, and every way a deliverable can be broken.

Broken deliverables must end up in the report; none of them may stop the harness.
"""
from __future__ import annotations

import json
import time

import pytest

import harness_support as hs
from harness import cli, config, verify

GOOD_WORLD = (hs.REFERENCE / "world.py").read_text()
PLAIN = hs.PLAIN_POLICY                                  # a policy that needs no other file


def _verify(make_run, files, overrides=None):
    cfg = make_run(overrides, files=files)
    return cfg, verify.verify(cfg)


def _policy(body: str, extra: str = "") -> str:
    """A policy file whose act() is `body` (with self.seed, obs available)."""
    return ("import numpy as np\nimport os, sys, time\n" + extra + "\n\nclass P:\n"
            "    def set_seed(self, seed):\n        self.seed = seed\n"
            "    def reset(self):\n        pass\n"
            "    def act(self, obs):\n" + "".join(f"        {line}\n" for line in body.splitlines()) +
            "\ndef make_policy():\n    return P()\n")


def _world(step_body: str = "", reset_body: str = "", extra: str = "") -> str:
    """The reference world model with World.step and/or World.reset replaced."""
    text = GOOD_WORLD + "\n\n_Base = World\n\n" + extra + "\n\nclass World(_Base):\n    pass\n"
    if step_body:
        text += "    def step(self, action):\n" + "".join(f"        {line}\n" for line in step_body.splitlines())
    if reset_body:
        text += "    def reset(self, seed):\n" + "".join(f"        {line}\n" for line in reset_body.splitlines())
    return text


# ------------------------------------------------------------------------------------------------------------------
def test_report_for_good_deliverables(make_run):
    cfg, rep = _verify(make_run, {**hs.reference_files(), "notes/diagnosis.json": "{}", "notes/fidelity.json": "not json"})
    p = config.paths(cfg)
    assert rep == hs.read_json(p.report_json)
    assert set(rep) == {"harness_version", "run", "trials", "agent", "deliverables", "predicted_by_agent", "model",
                        "model_from_held_out_starts", "real", "files", "wall_s"}
    assert rep["trials"] == {"used": 0, "budget": 3, "hypothesis_at_first_trial": None} and rep["agent"] is None
    d = rep["deliverables"]
    assert d["policy.py"] == {"status": "present", "required": True, "detail": None}
    assert d["world.py"]["status"] == "present" and d["prediction.json"]["status"] == "present"
    assert d["notes/hypothesis.json"]["status"] == "missing" and d["notes/diagnosis.json"]["status"] == "present"
    assert d["notes/fidelity.json"] == {"status": "broken", "required": False, "detail": "not readable as JSON"}
    assert rep["predicted_by_agent"] == 0.91
    for part, n, seeds in (("model", 6, [100000, 100005]), ("real", 4, [1000, 1003]),
                           ("model_from_held_out_starts", 4, [1000, 1003])):
        s = rep[part]
        assert (s["episodes"], s["seeds"], s["flagged"], s["flags"], s["error"]) == (n, seeds, 0, {}, None)
        assert s["success_rate"] == s["successes"] / n
    episodes = hs.read_json(p.verify / "real_episodes.json")
    assert [e["seed"] for e in episodes] == [1000, 1001, 1002, 1003]
    assert all(set(e) == {"seed", "success", "n_steps", "flag", "error", "wall_s"} and e["n_steps"] == 80 for e in episodes)
    assert len(hs.read_json(p.verify / "model_episodes.json")) == 6
    table = p.report_md.read_text()
    assert "| real: policy.py in the environment, held-out |" in table and "| notes/fidelity.json | broken | no |" in table
    assert not list(p.workspace.rglob("__pycache__"))     # verifying leaves nothing in the workspace
    assert verify.verify(cfg)["real"]["successes"] == rep["real"]["successes"]       # and can be repeated


def test_verify_options_and_command_line(make_run, capsys):
    cfg = make_run(files=hs.reference_files())
    rep = verify.verify(cfg, model_episodes=3, skip_real=True)
    assert rep["real"] is None and rep["model"]["episodes"] == 3 and rep["model_from_held_out_starts"] is None
    rep = verify.verify(cfg, skip_model=True, workers=1)
    assert rep["model"] is None and rep["real"]["episodes"] == 4
    assert cli.main(["verify", "--run-dir", cfg["run"]["dir"], "--model-episodes", "2", "--workers", "3"]) == 0
    assert "| predicted: policy.py in world.py |" in capsys.readouterr().out
    assert cli.main(["verify", "--run-dir", str(config.paths(cfg).run / "nowhere")]) == 2
    assert "not an initialised run directory" in capsys.readouterr().err


def test_cap_role_needs_no_world_model(make_run):
    cfg, rep = _verify(make_run, {"policy.py": PLAIN}, {"agent": {"role": "cap"}})
    assert rep["deliverables"]["world.py"] == {"status": "missing", "required": False, "detail": None}
    assert rep["deliverables"]["prediction.json"]["required"] is False
    assert rep["model"] is None and rep["predicted_by_agent"] is None and rep["real"]["episodes"] == 4


def test_cap_with_tests_role_is_scored_in_the_world_model_it_was_given(make_run):
    given = {"workspace": {"files": {"world.py": str(hs.REFERENCE / "world.py")}}}
    cfg, rep = _verify(make_run, {"policy.py": PLAIN, "prediction.json": '{"predicted_success_rate": 0.4}'},
                       {"agent": {"role": "cap_with_tests"}, **given})
    assert rep["run"]["role"] == "cap_with_tests" and rep["predicted_by_agent"] == 0.4
    assert rep["deliverables"]["world.py"] == {"status": "present", "required": True, "detail": None}
    assert rep["model"]["episodes"] == 6 and rep["real"]["episodes"] == 4
    assert rep["model_from_held_out_starts"]["episodes"] == 4


def test_missing_policy(make_run):
    cfg, rep = _verify(make_run, hs.reference_files("world.py", "prediction.json"))
    assert rep["deliverables"]["policy.py"] == {"status": "missing", "required": True, "detail": None}
    assert rep["deliverables"]["world.py"]["status"] == "present"
    assert rep["real"] is None and rep["model"] is None and rep["model_from_held_out_starts"] is None
    assert "| policy.py | missing | yes |" in config.paths(cfg).report_md.read_text()


@pytest.mark.parametrize("source, expected", [
    ("import a_module_that_does_not_exist\n", "ModuleNotFoundError"),
    ("def make_policy(:\n", "SyntaxError"),
    ("x = 1\n", "has no function make_policy()"),
    ("def make_policy():\n    raise RuntimeError('cannot build')\n", "RuntimeError: cannot build"),
    ("def make_policy():\n    return object()\n", "without a method reset()"),
    ("class P:\n    def reset(self): pass\ndef make_policy():\n    return P()\n", "without a method act()"),
    ("def make_policy(required_argument):\n    return None\n", "TypeError"),
    ("import sys\nsys.exit(3)\n", "SystemExit"),
    ("import os\nos._exit(4)\n", "exit code 4"),
])
def test_policy_that_cannot_be_loaded_is_reported_as_broken(make_run, source, expected):
    cfg, rep = _verify(make_run, {**hs.reference_files("world.py", "prediction.json"), "policy.py": source})
    entry = rep["deliverables"]["policy.py"]
    assert entry["status"] == "broken" and expected in entry["detail"], entry
    assert rep["real"] is None and rep["model"] is None
    assert rep["deliverables"]["world.py"]["status"] == "present"
    assert "| policy.py | broken | yes |" in config.paths(cfg).report_md.read_text()


@pytest.mark.parametrize("source, expected", [
    ("import a_module_that_does_not_exist\n", "ModuleNotFoundError"),
    ("World = 3\n", "has no class World"),
    ("class World:\n    def __init__(self):\n        raise ValueError('bad parameters')\n", "ValueError: bad parameters"),
    ("class World:\n    def reset(self, seed):\n        return {}\n", "World has no method step()"),
])
def test_world_that_cannot_be_loaded_is_reported_and_the_real_evaluation_still_runs(make_run, source, expected):
    cfg, rep = _verify(make_run, {"policy.py": PLAIN, "world.py": source, "prediction.json": '{"predicted_success_rate": 0.5}'})
    entry = rep["deliverables"]["world.py"]
    assert entry["status"] == "broken" and expected in entry["detail"], entry
    assert rep["model"] is None and rep["model_from_held_out_starts"] is None
    assert rep["real"]["episodes"] == 4 and rep["real"]["flagged"] == 0 and rep["predicted_by_agent"] == 0.5


def test_program_that_raises_in_some_episodes_counts_as_failure_there_and_is_flagged(make_run):
    body = "if self.seed % 2 == 1 and obs['t'] == 5:\n    raise RuntimeError('boom at step five')\nreturn np.zeros(2)"
    cfg, rep = _verify(make_run, {**hs.reference_files("world.py"), "policy.py": _policy(body)})
    p = config.paths(cfg)
    assert rep["deliverables"]["policy.py"]["status"] == "present"
    for part, file in (("real", "real_episodes.json"), ("model", "model_episodes.json")):
        episodes = hs.read_json(p.verify / file)
        odd = [e for e in episodes if e["seed"] % 2 == 1]
        even = [e for e in episodes if e["seed"] % 2 == 0]
        assert len(odd) == len(even) == len(episodes) // 2
        for e in odd:
            assert e["success"] is False and e["flag"] == "policy_error" and e["n_steps"] == 5
            assert "RuntimeError: boom at step five" in e["error"] and "policy.py" in e["error"]
        assert all(e["flag"] is None and e["n_steps"] == 80 for e in even)        # the other episodes are untouched
        assert rep[part]["flags"] == {"policy_error": len(odd)} and rep[part]["flagged"] == len(odd)
        assert "boom at step five" in rep[part]["first_error"]
    assert "policy_error 2" in p.report_md.read_text()


@pytest.mark.parametrize("body, real_flag, model_flag, text", [
    ("return np.zeros(3)", "action_rejected", "world_error", "shape (2,)"),
    ("return 'left'", "action_rejected", "world_error", "ValueError"),
    ("return None", "action_rejected", "world_error", "ValueError"),
    ("return object()", "policy_error", "world_error", ""),
    ("return {'force': np.zeros(2)}", "action_rejected", "world_error", ""),
    ("sys.exit(0)", "policy_error", "policy_error", "SystemExit"),
    ("return obs['no_such_key']", "policy_error", "policy_error", "KeyError"),
])
def test_wrong_return_values_of_a_program_are_flagged(make_run, body, real_flag, model_flag, text):
    cfg, rep = _verify(make_run, {**hs.reference_files("world.py"), "policy.py": _policy(body)})
    assert rep["real"]["flags"] == {real_flag: 4} and rep["real"]["successes"] == 0
    assert rep["model"]["flags"] == {model_flag: 6} and rep["model"]["successes"] == 0
    assert text in rep["real"]["first_error"]


def test_program_that_kills_its_process_is_flagged_and_the_next_episode_runs(make_run):
    body = "if self.seed in (1001, 100001) and obs['t'] == 2:\n    os._exit(9)\nreturn np.zeros(2)"
    cfg, rep = _verify(make_run, {**hs.reference_files("world.py"), "policy.py": _policy(body)}, {"verify": {"workers": 1}})
    p = config.paths(cfg)
    real = {e["seed"]: e for e in hs.read_json(p.verify / "real_episodes.json")}
    assert real[1001]["flag"] == "policy_error" and "exit code 9" in real[1001]["error"]
    assert [real[s]["flag"] for s in (1000, 1002, 1003)] == [None, None, None]
    model = {e["seed"]: e for e in hs.read_json(p.verify / "model_episodes.json")}
    assert model[100001]["flag"] == "process_died" and "exit code 9" in model[100001]["error"]
    assert all(model[s]["flag"] is None for s in model if s != 100001)


def test_program_that_hangs_times_out_and_the_rest_is_not_run(make_run):
    body = "if obs['t'] == 3:\n    time.sleep(120)\nreturn np.zeros(2)"
    t0 = time.time()
    cfg, rep = _verify(make_run, {**hs.reference_files("world.py"), "policy.py": _policy(body)},
                       {"verify": {"episode_timeout_s": 2, "workers": 1, "paired": False},
                        "budget": {"held_out_episodes": 5}})
    assert time.time() - t0 < 60
    assert rep["real"]["flags"] == {"timeout": 2, "not_run": 3} and rep["real"]["successes"] == 0
    assert rep["model"]["flags"] == {"timeout": 2, "not_run": 4} and rep["model_from_held_out_starts"] is None
    assert "longer than 2 s" in rep["real"]["first_error"]
    assert "timeout 2" in config.paths(cfg).report_md.read_text()


def test_file_that_hangs_while_loading_is_reported_as_broken(make_run):
    t0 = time.time()
    first, slow = "from __future__ import annotations", "from __future__ import annotations\nimport time\ntime.sleep(120)"
    cfg, rep = _verify(make_run, {"policy.py": PLAIN.replace(first, slow), "world.py": GOOD_WORLD.replace(first, slow)},
                       {"verify": {"load_timeout_s": 2}})
    assert time.time() - t0 < 30
    for name in ("policy.py", "world.py"):
        assert rep["deliverables"][name]["status"] == "broken" and "timeout" in rep["deliverables"][name]["detail"]
    assert rep["real"] is None and rep["model"] is None


@pytest.mark.parametrize("step_body, reset_body, flag, text", [
    ("return self._obs(), True", "", "world_contract", "must return (obs, done, info), got 2 values"),
    ("return None", "", "world_contract", "got NoneType"),
    ("obs, done, info = super().step(action)\nreturn [1.0, 2.0], done, info", "", "world_contract", "dict of observations"),
    ("obs, done, info = super().step(action)\ninfo.pop('success', None)\nreturn obs, done, info", "", "world_contract",
     'info["success"]'),
    ("obs, done, info = super().step(action)\nreturn obs, False, info", "", "no_end", "within 80 steps"),
    ("raise ZeroDivisionError('model blew up')", "", "world_error", "ZeroDivisionError: model blew up"),
    ("", "return super().reset(seed), {}", "world_contract", "World.reset() must return a dict"),
    ("", "raise KeyError('seed table')", "world_error", "KeyError"),
])
def test_world_with_wrong_return_shapes_or_errors_is_flagged(make_run, step_body, reset_body, flag, text):
    cfg, rep = _verify(make_run, {"policy.py": PLAIN, "world.py": _world(step_body, reset_body)})
    assert rep["deliverables"]["world.py"]["status"] == "present"        # it loads; it fails when it is used
    assert rep["model"]["flags"] == {flag: 6} and rep["model"]["successes"] == 0 and rep["model"]["success_rate"] == 0.0
    assert text in rep["model"]["first_error"]
    assert rep["real"]["flagged"] == 0 and rep["real"]["episodes"] == 4  # the real evaluation is unaffected
    assert f"{flag} 6" in config.paths(cfg).report_md.read_text()


def test_world_whose_observations_differ_from_the_environment_is_reported(make_run):
    step = ("obs, done, info = super().step(action)\nobs.pop('vel')\nobs['extra'] = 1.0\n"
            "obs['pos'] = np.append(obs['pos'], 0.0)\nreturn obs, done, info")
    reset = "obs = super().reset(seed)\nobs.pop('vel')\nobs['extra'] = 1.0\nobs['pos'] = np.append(obs['pos'], 0.0)\nreturn obs"
    policy = _policy("return np.zeros(2)")
    cfg, rep = _verify(make_run, {"policy.py": policy, "world.py": _world(step, reset)})
    problems = rep["model"]["format_problems"]
    assert any("missing in the world model: vel" in x for x in problems)
    assert any("only in the world model: extra" in x for x in problems)
    assert any("'pos'" in x and "(2,)" in x and "(3,)" in x for x in problems)
    assert rep["model"]["episodes"] == 6 and rep["model"]["flagged"] == 0
    assert "world.py format" in config.paths(cfg).report_md.read_text()


def test_world_that_hangs_times_out(make_run):
    step = "if self._t == 3:\n    time.sleep(120)\nreturn super().step(action)"
    t0 = time.time()
    cfg, rep = _verify(make_run, {"policy.py": PLAIN, "world.py": _world(step, extra="import time")},
                       {"verify": {"episode_timeout_s": 2, "workers": 1, "paired": False}})
    assert time.time() - t0 < 60
    assert rep["model"]["flags"] == {"timeout": 2, "not_run": 4}
    assert rep["real"]["flagged"] == 0


@pytest.mark.parametrize("content, problem", [
    ("{not json", "not readable as JSON"),
    ('{"success": 0.9}', "predicted_success_rate"),
    ('{"predicted_success_rate": "high"}', "predicted_success_rate"),
    ('{"predicted_success_rate": 91}', "not in [0, 1]"),
    ('{"predicted_success_rate": true}', "predicted_success_rate"),
    ("[0.9]", "predicted_success_rate"),
])
def test_broken_prediction_file(make_run, content, problem):
    cfg = make_run(files={**hs.reference_files("world.py", "policy.py"), "prediction.json": content})
    rep = verify.verify(cfg, skip_real=True, skip_model=True)
    entry = rep["deliverables"]["prediction.json"]
    assert entry["status"] == "broken" and problem in entry["detail"] and entry["required"] is True
    assert rep["predicted_by_agent"] is None


def test_program_cannot_spend_real_trials_or_reach_the_environment_while_verified(make_run):
    """During verification the workspace's lab.py has no server to talk to, and the program only ever sees observations."""
    body = ("import lab\ntry:\n    lab.reset(1)\n    open('reached.txt', 'w').write('yes')\n"
            "except lab.LabError as ex:\n    open('refused.txt', 'w').write(type(ex).__name__)\nreturn np.zeros(2)")
    cfg, rep = _verify(make_run, {"policy.py": _policy(body)}, {"agent": {"role": "cap"}})
    ws = config.paths(cfg).workspace
    assert (ws / "refused.txt").read_text() == "NotConnected" and not (ws / "reached.txt").exists()
    assert rep["trials"]["used"] == 0 and rep["real"]["flagged"] == 0


def test_environment_that_fails_during_verification_is_reported(make_run, capsys):
    cfg = make_run(files=hs.reference_files())
    p = config.paths(cfg)
    stored = hs.read_json(p.config_json)
    stored["environment"]["factory"] = "toy_env:gone_missing"          # the environment no longer starts
    p.config_json.write_text(json.dumps(stored))
    assert cli.main(["verify", "--run-dir", cfg["run"]["dir"]]) == 1
    rep = hs.read_json(p.report_json)
    assert rep["real"]["flags"] == {"environment_error": 4} and rep["real"]["successes"] == 0
    assert "could not be started" in rep["real"]["error"] and "gone_missing" in rep["real"]["error"]
    assert rep["model"]["episodes"] == 6 and rep["model"]["flagged"] == 0          # the model part still runs
    assert "error (environment)" in capsys.readouterr().out
