# Your task: write a control program for "{{environment_name}}" and test it with real trials

You are a coding agent working alone in this directory, your workspace. An environment is attached to it, and you may run it for real only {{trial_budget}} times.

You work in the Code as Policies way: you write the control program directly, run it in the environment, read the record of what happened step by step, diagnose the failure, repair the program, and keep what you learned as reusable skills. A program can be tested only by running it for real. Play this as well as it can be played.

After you finish, a harness that you cannot influence runs your program on {{held_out_episodes}} held-out situations in the real environment, which you cannot reach, and reports its success rate. Results are reported whatever they are: never shade them.

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

- `lab.reset(seed)` starts one real trial and returns the first observation. `lab.step(action)` returns `(obs, done, info)`; on the last step `info["success"]` gives the outcome. `lab.run_trial(policy, seed)` runs a whole episode with a policy object. An episode has at most {{max_steps}} steps.
- You have **{{trial_budget}} real trials**. Every `lab.reset()` uses one up, also when you abandon the episode. After the last one, `lab.reset()` raises `lab.BudgetExceeded`. `lab.status()` and `lab.task_card()` are free.
- You choose the seed of each trial, from {{seed_lo}} to {{seed_hi}}. A seed fixes the initial situation. Other seeds are refused.
- Run real trials one at a time, from one process.
- The manual documents the interface and the success rule. How the environment responds to your actions is documented nowhere.

## The rule that defines this role

- A program can only be tested by executing it for real, and every execution costs a real trial.
- You may analyse the manual and your own trial records offline as much as you like: statistics, plots, fits, replaying recorded observations through a new program to see what it would have commanded. Your program may contain any reasoning and any closed-form rule you can justify from your records, including feedback on the live observations.
- You may not build or use a simulator or forward model of the environment to roll out, score, search, optimise or train a controller, neither offline nor inside the program at run time: no planning over predicted observations, no reinforcement learning, no parameter search against simulated episodes. If you catch yourself writing a function that advances the system's state under hypothetical actions and using it to compare candidates, stop. That is a different method. A one-shot closed-form estimate inside a hand-written rule ("an action of this size moves it by about k times ...") is on the allowed side.

## Method

**Step 0. Say what it is, before any real trial.** From the manual and the task card alone, write `notes/hypothesis.json` before your first `lab.reset()`, and do not edit it afterwards. The harness records whether the file was there when your first trial started. Keys: `what_it_is` (your best plain-words statement of what the system physically is and how it will respond), `evidence` (what in the manual or the task card supports it), `unknowns` (what you cannot know without acting, and how you will use your real trials to find out).

**Step 1. Write, run, diagnose, repair.** Write the program as `policy_v1.py`: `make_policy()` returns an object with `reset()` and `act(obs) -> action`. Optional hooks: `bind(task_card)`, called once after `make_policy()`, and `set_seed(seed)`, called before each episode. Run it for real, then read the record in `trials/`: where did it go wrong, at which step, why. Repair it as the next version, `policy_v2.py`, and so on. Number the versions in the order written, keep every version you seriously considered running, and never edit a version after writing the next.

**Step 2. Keep the skills.** After each trial, add what you learned to `notes/skills.json`, a list of skills, each with: `failure_signature` (how the problem shows in a record), `when_to_apply`, `repair` (the strategy), `code_sketch`.

**Step 3. Hand in.** Copy the version you trust most to `policy.py`. The harness stops an episode that takes longer than {{episode_timeout_s}} seconds of wall-clock time and counts it as a failure. Then write `notes/diagnosis.json`: for each real trial the program version, the seed, the outcome, what the record showed and what you changed; the list of versions with one line each and which one you handed in; what you conclude the system is.

## What you hand in

Required: `policy.py`. Asked for: every `policy_v<k>.py`, `notes/hypothesis.json`, `notes/skills.json`, `notes/diagnosis.json`. Leave your analysis scripts in the workspace as well. Write no other documents.

## Rules

- Work only inside this directory. Do not read, list or search anything outside it. Do not look for the environment's source code, its process, its files or the packages it is built on, and do not inspect which software packages are installed or loaded: that would tell you what the system is by a route other than observation, which defeats the purpose. No web lookups about this environment.
- `lab.py` is the only door to the environment. Do not reach the environment in any other way, do not use anything in `lab.py` whose name starts with an underscore, and do not try seeds outside your range.
- General knowledge of mechanics and control is welcome: recognise the physics from your records.
- If an error message ever shows you something about the environment's internals, do not use it, and report what you saw in your final message.
- If you are unsure whether something is allowed, do not do it, and say so in your final message.

## Pace

Aim to finish within about {{time_allowance_min}} minutes of work.

## Final message

A short account: real trials used and what each showed; which version you handed in; what you concluded about how the system behaves; the skills in brief; what worked and what did not; and a plain statement that you stayed within the rule of this role and the access rules, naming the point where you came closest to the line. Claim only what you actually ran.
