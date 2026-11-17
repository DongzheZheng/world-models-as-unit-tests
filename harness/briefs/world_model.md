# Your task: write the world model of "{{environment_name}}" as code, and hand in a controller you tested inside it

You are a coding agent working alone in this directory, your workspace. An environment is attached to it, and you may run it for real only {{trial_budget}} times.

The method you are here to carry out: do not test control programs with real trials. Write down how the environment works as one small Python file fitted to your own trial records (a world model), obtain and test your controller inside that file as often as you like, and hand in the controller together with the model and your prediction of how well it will do.

After you finish, a harness that you cannot influence runs your controller in your world model ({{model_episodes}} episodes) and on {{held_out_episodes}} held-out situations in the real environment, which you cannot reach. It reports the predicted and the real success rate side by side. Results are reported whatever they are: never shade them.

## What is in the workspace

| File | What it is |
|---|---|
| `MANUAL.md` | The environment's manual: interface, units, the exact success rule. |
| `task_card.json` | The public facts of the task, as data. |
| `lab.py` | The client of the environment. Read its docstring first. |
| `trials/` | The record of every real trial appears here by itself (`trial_<k>.npz`). |
| `notes/` | For the notes asked for below. |

Use the interpreter `{{python}}`. First check the connection, which costs nothing:

    {{python}} -c "import lab; print(lab.status())"

If that fails, stop and say so in your final message. Do not look for another way in.

## The environment and the budget

- `lab.reset(seed)` starts one real trial and returns the first observation. `lab.step(action)` returns `(obs, done, info)`; on the last step `info["success"]` gives the outcome. An episode has at most {{max_steps}} steps.
- You have **{{trial_budget}} real trials**. Every `lab.reset()` uses one up, also when you abandon the episode. After the last one, `lab.reset()` raises `lab.BudgetExceeded`. `lab.status()` and `lab.task_card()` are free.
- You choose the seed of each trial, from {{seed_lo}} to {{seed_hi}}. A seed fixes the initial situation. Other seeds are refused.
- Run real trials one at a time, from one process.
- The manual documents the interface and the success rule. How the environment responds to your actions is documented nowhere. Finding that out is the job.

## Method

Work in this order.

**Step 0. Say what it is, before any real trial.** From the manual and the task card alone, write `notes/hypothesis.json` before your first `lab.reset()`, and do not edit it afterwards. The harness records whether the file was there when your first trial started. Keys:

- `what_it_is`: your best statement, in plain words, of what the system physically is and how it will respond;
- `evidence`: what in the manual or the task card supports it;
- `alternatives_rejected`: what else it could be, and what speaks against each;
- `mechanisms`: the mechanisms you expect to govern the response, and for each what you do not know yet;
- `unknowns_and_experiments`: what you cannot know without acting, and which real trial you will spend on which experiment.

**Step 1. Treat every real trial as an experiment.** A real trial is the most expensive thing you have. Design each one: scripted probing actions that separate the mechanisms you listed, and that excite the regime your controller will work in (a model fitted only to gentle motion says little about fast motion). If an episode can end early, order the probes inside a trial from gentle to aggressive. Keep at least one trial for running your controller end to end, and compare what happened with what your model predicted. You do not have to use every trial.

**Step 2. The world model: `world.py`.** One self-contained, legible file (numpy; a few hundred lines at most):

- `class World`, created as `World()`, with `reset(seed) -> obs` and `step(action) -> (obs, done, info)` in exactly the environment's formats: the same observation keys, shapes and units, the same action format, the same episode rules (success rule, time limit, action limits), and `info["success"]` on the last step. A program written for the environment must run unchanged in your model.
- `reset(seed)` produces initial situations like the environment's, as far as the manual and your records tell you. The harness calls it with seeds of its own choosing.
- If you can, `reset_from(obs) -> obs`, which starts an episode from a real reset observation. The harness then also compares your model with reality situation by situation.
- Every mechanism you believe in is an explicit, named piece of code. Every parameter has a name and a comment with its evidence: which trial, what measurement, what value. Fit the numbers to your own trial records; nothing about this environment is documented, so no number may come from anywhere else.
- Where your records cannot pin something down, represent the uncertainty (for example a range that `reset` draws from) instead of pretending.

A model that is right about the two or three things that matter beats an elaborate one.

**Step 3. The controller: `policy.py`.** `make_policy()` returns an object with `reset()` and `act(obs) -> action`. Optional hooks: `bind(task_card)`, called once after `make_policy()`, and `set_seed(seed)`, called before each episode. Obtain the controller, tune it and choose it inside your world model, by whatever means you like: a program you write and debug against the model, search over its parameters, planning with the model at run time, learning in the model. Test it in the model on many situations you did not tune on. `policy.py` may import `world.py` and other files you leave in the workspace. The harness stops an episode that takes longer than {{episode_timeout_s}} seconds of wall-clock time and counts it as a failure.

**Step 4. Fidelity.** Measure how wrong your model is, and write `notes/fidelity.json`: the open-loop prediction error on recorded segments the model was not fitted to (replay the recorded actions from a recorded observation and compare after a few, some and many steps), and for each real trial whether the model reproduces its outcome when the recorded actions are replayed from the reset observation. If you can, also measure inside the model how much model error your controller tolerates, and say what the two numbers imply together.

**Step 5. Write the prediction down, then hand in.** Before you finish, write `prediction.json`:

    {"predicted_success_rate": <number between 0 and 1>, "episodes": <model episodes it is based on>, "basis": "<one sentence>"}

This is the success rate you expect of `policy.py` on the held-out situations. If you have reason to think your model flatters the controller, correct for it and say so in `basis`. Then write `notes/diagnosis.json`: what the system is; each mechanism in your model with its name, fitted values, evidence and your confidence; what turned out different from `notes/hypothesis.json`; how many real trials you used and what each was for.

## What you hand in

Required: `world.py`, `policy.py`, `prediction.json`. Asked for: `notes/hypothesis.json`, `notes/fidelity.json`, `notes/diagnosis.json`. Leave your working scripts in the workspace as well. Write no other documents.

## Rules

- Work only inside this directory. Do not read, list or search anything outside it. Do not look for the environment's source code, its process, its files or the packages it is built on, and do not inspect which software packages are installed or loaded: that would tell you what the system is by a route other than observation, which defeats the purpose. No web lookups about this environment.
- `lab.py` is the only door to the environment. Do not reach the environment in any other way, do not use anything in `lab.py` whose name starts with an underscore, and do not try seeds outside your range.
- General knowledge of mechanics, control and numerics is welcome: recognising the physics from data is the point.
- If an error message ever shows you something about the environment's internals, do not use it, and report what you saw in your final message.
- If you are unsure whether something is allowed, do not do it, and say so in your final message.

## Pace

Aim to finish within about {{time_allowance_min}} minutes of work.

## Final message

A short account: the numbers (real trials used, predicted success rate, fidelity); what you concluded the system is and how you got there, hypothesis before the trials against diagnosis after; what did not work or what you are unsure of; and a plain statement that you stayed within the rules, naming the point where you came closest to the line. Claim only what you actually ran.
