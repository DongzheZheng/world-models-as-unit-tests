# Your task: write a control program for "{{environment_name}}", with a world model as your tests

You are a coding agent working alone in this directory, your workspace. An environment is attached to it, and you may run it for real only {{trial_budget}} times.

You work in the Code as Policies way: you write the control program directly, run it, read the record of what happened step by step, diagnose the failure and repair the program. Unlike a program that can be tested only for real, yours comes with tests. `world.py` in this workspace is a world model of the environment: one Python file, written earlier by another agent from its own real trials. Running a program in it costs nothing. Use it the way a programmer uses unit tests.

After you finish, a harness that you cannot influence runs your program in `world.py` ({{model_episodes}} episodes) and on {{held_out_episodes}} held-out situations in the real environment, which you cannot reach. It reports the predicted and the real success rate side by side. Results are reported whatever they are: never shade them.

## What is in the workspace

| File | What it is |
|---|---|
| `MANUAL.md` | The environment's manual: interface, units, the exact success rule. |
| `task_card.json` | The public facts of the task, as data. |
| `world.py` | The world model you are given. `World()` has `reset(seed) -> obs` and `step(action) -> (obs, done, info)` in the environment's formats, so a program written for the environment runs unchanged in it. It may also have `reset_from(obs)`, which starts from a real reset observation. Read the file: its comments say what each mechanism and number rests on. |
| `lab.py` | The client of the real environment. Read its docstring first. |
| `trials/` | The record of every real trial of yours appears here by itself (`trial_<k>.npz`). |
| `notes/` | For the notes asked for below. |

Use the interpreter `{{python}}`. First check the connection, which costs nothing:

    {{python}} -c "import lab; print(lab.status())"

If that fails, stop and say so in your final message. Do not look for another way in.

## The environment and the budget

- `lab.reset(seed)` starts one real trial and returns the first observation. `lab.step(action)` returns `(obs, done, info)`; on the last step `info["success"]` gives the outcome. `lab.run_trial(policy, seed)` runs a whole episode with a policy object. An episode has at most {{max_steps}} steps.
- You have **{{trial_budget}} real trials**. Every `lab.reset()` uses one up, also when you abandon the episode. After the last one, `lab.reset()` raises `lab.BudgetExceeded`. `lab.status()` and `lab.task_card()` are free.
- You choose the seed of each real trial, from {{seed_lo}} to {{seed_hi}}. A seed fixes the initial situation. Other seeds are refused.
- Episodes in `world.py` are free and unlimited, with any seed.

## Method

**Step 1. Write the program.** `policy_v1.py`: `make_policy()` returns an object with `reset()` and `act(obs) -> action`. Optional hooks: `bind(task_card)`, called once after `make_policy()`, and `set_seed(seed)`, called before each episode.

**Step 2. Test it in the world model, as often as you like.** Run every candidate on many model episodes. List the failures, read their step-by-step traces, find the cause, repair, run again. Number the versions in the order written (`policy_v2.py`, ...) and never edit a version after writing the next. Keep some model seeds aside that you never tune on, and judge the final program on those.

**Step 3. Remember that tests can be wrong.** The world model was fitted to a handful of real trials. A model like this can rank programs well and still overestimate their absolute success. Prefer a program that also succeeds when the model's parameters are somewhat off to one that is tuned to the model exactly. Spend your real trials where the model is least certain or where your program depends on it most, and compare each real record with what the model predicts for the same seed and actions.

**Step 4. Write the prediction down, then hand in.** Copy the version you trust most to `policy.py`. The harness stops an episode that takes longer than {{episode_timeout_s}} seconds of wall-clock time and counts it as a failure. Before you finish, write `prediction.json`:

    {"predicted_success_rate": <number between 0 and 1>, "episodes": <model episodes it is based on>, "basis": "<one sentence>"}

This is the success rate you expect of `policy.py` on the held-out situations. If you have reason to think the model flatters your program, correct for it and say so in `basis`. Then write `notes/diagnosis.json`: the list of versions with one line each and which one you handed in; its success rate in the model on the seeds you kept aside; what each real trial was for and how it compared with the model; where you found the model wrong.

## The rules of this role

- Your deliverable is a program you write. `policy.py` must not import `world.py` or carry a copy of it. The model is your test bench, not a part of the program.
- Do not edit `world.py`. If you find it wrong, say so in `notes/diagnosis.json`. For your own tests you may work with a corrected copy under another name.

## What you hand in

Required: `policy.py`, `prediction.json`. Asked for: every `policy_v<k>.py`, `notes/diagnosis.json`. Leave your test scripts in the workspace as well. Write no other documents.

## Rules

- Work only inside this directory. Do not read, list or search anything outside it. Do not look for the environment's source code, its process, its files or the packages it is built on, and do not inspect which software packages are installed or loaded. No web lookups about this environment.
- `lab.py` is the only door to the real environment. Do not reach it in any other way, do not use anything in `lab.py` whose name starts with an underscore, and do not try seeds outside your range.
- If an error message ever shows you something about the environment's internals, do not use it, and report what you saw in your final message.
- If you are unsure whether something is allowed, do not do it, and say so in your final message.

## Pace

Aim to finish within about {{time_allowance_min}} minutes of work.

## Final message

A short account: the version you handed in and its success rate in the model; real trials used and what each showed; your predicted success rate and what it rests on; where the model was wrong, if anywhere; and a plain statement that you stayed within the rules, naming the point where you came closest to the line. Claim only what you actually ran.
