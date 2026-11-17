# Harness

A small harness for trying the procedure of this repository on your own environment. You bring a simulator or robot interface in Python and an API key for a coding agent. The harness gives the agent a workspace, a budget of real trials, and a brief that describes the task. When the agent finishes, the harness runs its delivered controller in the delivered world model and on held-out seeds in the environment, and writes a report with the predicted and real success rates side by side.

**Status.** No model has been run through this harness by us. It was tested without spending tokens: a stub agent script, the replay adapter, the toy example below and the Push-T reproduction. The two agent example configs (Claude Code, Codex CLI) were written from each tool's documentation and were not executed, and whether lab.py reaches the environment server from inside each tool's sandbox has not been tested. Tested on Linux with Python 3.12. macOS is untested and Windows is not supported. How well an agent does on a new environment through this harness is unknown.

## Try it without tokens

The toy example runs the whole procedure on a small environment (a puck on a table with hidden gain, friction, drift and noise) using a replay agent that spends no tokens.

```bash
python -m harness run --config harness/examples/toy/run.toml
```

It finishes in about a second and writes `harness/examples/toy/runs/toy/report.md`. The world model and policies in `harness/examples/toy/reference/` are hand-written stand-ins for an agent's output, and the replay script flies 3 of the 5 available trials. The report:

> Real trials used: 3 of 5. Predicted by the agent: 0.910. Predicted by re-running policy.py in world.py: 0.910 (182/200). In world.py from held-out reset observations: 0.920 (46/50). Real, held-out: 0.960 (48/50). Same outcome in world.py and in the environment per held-out seed: 0.880.

## What it does

One run creates a workspace that contains only a brief (`BRIEF.md`), the environment's manual (`MANUAL.md`), the public task facts (`task_card.json`) and a thin client (`lab.py`). The environment itself runs in a separate process, so its source is never in the workspace. The agent gets a budget of real trials (default 5) on an allowed seed range. It leaves a policy (`policy.py`), a world model with the environment's interface (`world.py`) and the success rate it predicts (`prediction.json`). The harness then runs the policy in the delivered world model and on held-out seeds in the environment, and writes `report.md` with predicted against real.

## The three roles

Each role has a brief (`harness/briefs/`) that the harness gives to the agent as `BRIEF.md`.

| Role | Brief | What the agent does | Deliverables |
|------|-------|---------------------|-------------|
| `world_model` | `world_model.md` | The world-model agent of the article. Writes a world model, obtains and tests a controller inside it. | `world.py`, `policy.py`, `prediction.json` |
| `cap` | `cap.md` | A Code-as-Policies agent that can only test with real trials. | `policy.py` |
| `cap_with_tests` | `cap_with_tests.md` | A CaP agent that is given a world model written earlier and uses it as its tests. | `policy.py`, `prediction.json` |

## What you write

### The environment

A Python object with the interface described in `harness/contract.py`:

| Attribute / method | Description |
|-------------------|-------------|
| `manual` | `str`. The text the agent is given as `MANUAL.md`: interface, units, frames, the exact success rule. Leave out whatever the agent is supposed to find out by experiment. |
| `max_steps` | `int`. The longest an episode can be. The harness ends an episode that reaches it. |
| `task_card()` | Returns a JSON-serialisable `dict` of public facts about the task. |
| `reset(seed)` | Start an episode. `seed` is an integer from one of the two configured ranges. Returns `obs`, a dict whose values are numbers or numpy arrays (same keys and shapes on every step). |
| `step(action)` | Apply one action. Returns `(obs, done, info)`. `done` is `True` on the last step, and then `info["success"]` (bool) must say whether the task was achieved. Raise `ValueError` for an unacceptable action. |
| `close()` | Optional. Called when the process stops. |

The same seed with the same actions should give the same episode if the system allows it. One instance serves one episode at a time. Held-out evaluation creates one instance per worker process, so a system that exists only once (a robot) needs `[verify] workers = 1`.

The run config names the environment as a factory, `"package.module:function"`. The harness calls it in a process of its own and passes the keyword arguments of the config's `[environment] kwargs` table. See `harness/examples/toy/toy_env.py` for a working example.

### The config file

A TOML file. All keys with their defaults (from `harness/config.py`):

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `[run]` | `name` | config file name | Shown in the brief and in reports |
| | `dir` | `runs/<name>` | The run directory |
| | `workspace` | `<dir>/workspace` | The agent's workspace |
| `[environment]` | `factory` | (required) | `"package.module:function"` returning an object that follows the contract |
| | `paths` | `[]` | Extra import paths for the environment process only |
| | `kwargs` | `{}` | Keyword arguments passed to the factory |
| | `cwd` | `"."` | Working directory of the environment process |
| | `name` | run name | What the brief calls the environment |
| `[budget]` | `trials` | `5` | Real trials the agent may spend |
| | `seed_range` | `[0, 999]` | Seeds the agent may use (inclusive) |
| | `held_out_seed_range` | `[1000, 1999]` | Seeds reserved for held-out evaluation |
| | `held_out_episodes` | `50` | How many held-out seeds `verify` uses |
| `[agent]` | `kind` | `"command"` | `"command"` runs `command`. `"replay"` follows a script with no tokens. `"none"` does nothing. |
| | `role` | `"world_model"` | Brief template: `"world_model"`, `"cap"` or `"cap_with_tests"` |
| | `brief` | `""` | Path to your own brief template (replaces the one chosen by `role`) |
| | `command` | `[]` | Argument list. Placeholders: `{workspace}`, `{brief_file}`, `{brief_text}`, `{python}`, `{state_dir}` |
| | `stdin` | `"none"` | `"brief"` feeds the brief text to the command's standard input |
| | `timeout_s` | `7200` | The command is killed after this many seconds |
| | `time_allowance_min` | 3/4 of timeout | What the brief tells the agent |
| | `env` | `{}` | Extra environment variables for the command |
| | `script` | `""` | `kind = "replay"` only: the JSON script |
| | `python` | harness interpreter | The interpreter the brief tells the agent to use |
| `[workspace.files]` | | `{}` | Extra files copied into the workspace at init: `"path in workspace" = "source"`. Role `"cap_with_tests"` needs `"world.py"` here. |
| `[verify]` | `model_episodes` | `200` | Episodes of the delivered policy in the delivered world model |
| | `model_seed_first` | `100000` | First seed of the model episodes |
| | `paired` | `true` | If `World` has `reset_from()`: also start the model from the held-out resets |
| | `workers` | `4` | Processes for episodes |
| | `load_timeout_s` | `120` | Importing a delivered file and constructing its object |
| | `episode_timeout_s` | `120` | One episode, wall clock |
| `[screen]` | `world` | `""` | The world model file |
| | `policies` | `[]` | Policy files |
| | `names` | `[]` | Optional names, one per policy |
| | `paired` | `false` | Also start from real reset observations |
| | `out` | `<run dir>/screen` | Output directory |
| `[server]` | `transport` | `"auto"` | How the workspace reaches the environment: `"unix"`, `"tcp"` or `"auto"` (unix where available) |
| | `start_timeout_s` | `120` | How long the environment may take to start |

Paths are relative to the config file unless they are absolute.

## What the agent can call

The workspace contains `lab.py`, a client of the environment that runs in another process. Its public interface (from the docstring):

```python
import lab

lab.status()                 # free: trials used and left, the seeds you may use, the step limit
lab.task_card()              # free: the public facts of the task, as a dict (same as task_card.json)

obs = lab.reset(seed)        # starts one real trial and returns the first observation (a dict)
obs, done, info = lab.step(action)
...                          # until done is True; info["success"] says how the trial ended

result = lab.run_trial(policy, seed)   # the loop above for a policy object with reset() and act(obs)
log = lab.load_trial(k)               # the saved record of the k-th real trial, as a dict of arrays

env = lab.Env()              # the same two calls as methods, so code written against env also runs
                             # against an object with the same two methods (such as a world model)
```

Exceptions: `lab.BudgetExceeded` (all trials used up), `lab.SeedNotAllowed` (seed outside the range, costs nothing), `lab.NoActiveTrial` (`step()` without a running episode), `lab.NotConnected` (the environment process is not running).

Every real trial is saved when it ends or is abandoned: `trials/trial_<k>.npz`.

## Commands

All commands are run from the repository root.

```bash
python -m harness init    --config run.toml
```

Create the run directory and the agent's workspace (`BRIEF.md`, `MANUAL.md`, `task_card.json`, `lab.py`, `trials/`, `notes/`).

```bash
python -m harness serve   --config run.toml
```

Keep the environment server up until Ctrl-C. Use this when you start the agent yourself (for example interactively in the workspace), then run `verify`.

```bash
python -m harness run     --config run.toml
```

The whole procedure: `init` if needed, start the environment server, run the agent, stop the server, verify. Add `--skip-verify` to stop after the agent finishes.

```bash
python -m harness verify  --run-dir DIR
```

Score the delivered policy in the delivered world model (predicted success) and on the held-out seeds in the environment (real success). Write `report.json` and `report.md`. Delivered code runs in subprocesses with time limits.

```bash
python -m harness screen  --config run.toml
```

Score any list of policy files in a world model and in the environment. Write `screen.json`, `screen.md` and `screen.svg` (a scatter plot). Useful for evaluating a set of program versions against a single world model.

```bash
python -m harness status  --run-dir DIR
```

Print the budget counter and which deliverables are in the workspace, as JSON.

## The Push-T reproduction

The `screen` command can reproduce the article's predicted-against-real comparison for the pushing task through the harness's generic code path.

```bash
python -m harness screen --config harness/examples/pusht/screen.toml --workers 8
```

It runs the CaP agent's eight Push-T programs through the repository's pushing environment and the agent-written world model, behind two small hand-written adapters. It reproduces the recorded numbers exactly, for example v4: 0.980 in the world model and 0.960 in the environment, v5: 0.200 and 0.160 (Pearson r = 0.985 over the eight programs). It takes about two and a half minutes on 8 workers.

## Configuring the agent

The harness does not call any model itself. It starts whatever agent command you configure in the run config (`[agent] kind = "command"`). The command runs in the workspace directory. Example configs for Claude Code and for the OpenAI Codex CLI are in `harness/examples/agents/`. Both were written from the tools' documentation and were not executed.

A second kind, `"replay"`, spends no tokens. It re-flies recorded trials and copies recorded deliverables. This is what the tests and examples use.

## What is enforced

The environment is only imported in the server process. Its source is never copied into the workspace.

The budget is counted on the server at each `reset()` and survives restarts. The trial after the last one fails with `lab.BudgetExceeded`. Seeds outside the allowed range are refused at no cost with `lab.SeedNotAllowed`. Held-out seeds can only be requested by the harness through a channel the workspace never sees.

Every trial is recorded, including abandoned ones. The harness records whether `notes/hypothesis.json` existed when the first trial started.

Delivered code runs in subprocesses with timeouts (configurable via `[verify] load_timeout_s` and `episode_timeout_s`). It is never imported into the harness's own process or into the process that holds the environment.

## What is not enforced

This is not a sandbox. An agent tool that can read outside its working directory can read the harness's state and the environment's source. The agent tool's own sandbox or a container is needed for real isolation. Delivered code runs as the same user, with timeouts only. Some role rules are instructions only (for example, the rule that a CaP agent builds no forward model).

## Tests

```bash
python -m pytest harness/tests -q
```

124 tests, about three and a half minutes. For a faster check:

```bash
python -m pytest harness/tests -q -m "not slow"
```

123 tests in under a minute.

## Requirements

Python 3.11 or later. POSIX (inherited sockets, process groups). The core needs only the standard library and numpy. Tested on Linux with Python 3.12.
