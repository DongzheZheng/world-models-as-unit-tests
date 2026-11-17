Recorded results from [World Models as Unit Tests](https://dongzhezheng.github.io/world-models-as-unit-tests/), an early exploration in which coding agents write a task's world model as code and use it to verify control programs before deployment. We tried this across eight tasks in four simulators (Doom, a quadrotor, planar pushing, cable and rope manipulation), with each agent capped at five real trials per task. The [repository](https://github.com/DongzheZheng/world-models-as-unit-tests) holds the environments, every program and world model the agents wrote, and a script (`reproduce.py`) that re-runs any frozen controller. An [interactive Space](https://huggingface.co/spaces/DenzelZheng/world-models-as-unit-tests) lets you browse the pushing runs episode by episode.

## Subsets

The dataset has four subsets (JSON Lines files). A fifth component, `raw/runs/`, is a copy of the repository's `runs/` directory with the original result files and real-trial logs (model checkpoint files are left out and are in the repository).

### `results` (default)

46 rows, one per recorded held-out evaluation of a controller.

| Column | Description |
|--------|-------------|
| `task` | One of doom, drone_A, drone_B, drone_C, push_T, push_L, cable, rope |
| `agent` | The agent or baseline that produced this controller |
| `extended_run` | Whether this is a run with more than the five-trial budget |
| `real_trials` | Total real trials the controller's deployment drew on |
| `real_trials_training` | Real trials used to write the program or train the policy |
| `real_trials_calibration` | Real trials used to fit a world model |
| `metric` | Success rate (a fraction from 0 to 1), or survival in game ticks for Doom |
| `value` | The metric value |
| `n_episodes` | Number of held-out episodes evaluated |
| `first_seed`, `last_seed` | Seed range of the held-out episodes |
| `value_std`, `value_sem`, `value_median`, `value_min`, `value_max` | Distribution statistics |
| `real_steps` | Environment steps in the real trials |
| `real_steps_calibration` | Environment steps used to fit a world model |
| `real_steps_training` | Environment steps used to write the program or train the policy |
| `world_model_steps` | Simulated steps in the world model |
| `train_wall_s` | Wall-clock time for training or writing |
| `trained_in` | Environment or world model |
| `record_name` | The original name in the repository |
| `program` | v5 etc. when the controller is a numbered CaP program |
| `controller_file` | Path to the controller source file (joins `code.path`) |
| `episodes_file` | Path to the per-episode record |
| `reproduce_task` | The task name `reproduce.py` takes |
| `recorded_at` | Timestamp (local time, no time zone) when the record was written |
| `real_trials_source` | Where the trial count was derived from |
| `source_file` | The original result file in the repository |

### `programs`

34 rows. The CaP agent's program versions used in the article's comparison of success predicted in the world model with success in the environment (five tasks: drone A, drone C, Push-T, Push-L, Doom). Over these 34 rows the correlation between predicted and real is r = 0.974 and the mean absolute error is 5.2 percentage points.

| Column | Description |
|--------|-------------|
| `task` | One of doom, drone_A, drone_C, push_T, push_L |
| `agent` | The agent that wrote the program |
| `program` | Version label (v1, v2, ...) |
| `record_name` | The original name in the repository |
| `written_after_real_trials` | How many real trials the agent had used before writing this version |
| `submitted` | Whether this is the version the agent handed in |
| `metric` | Success rate, or for Doom the share of episodes surviving 750 game ticks |
| `predicted` | Success rate in the world model |
| `predicted_n` | Number of world-model episodes |
| `real` | Success rate in the environment |
| `real_n` | Number of environment episodes |
| `program_file` | Path to the program source file (joins `code.path`) |
| `predicted_source`, `real_source` | Source files for the two numbers |

### `episodes`

10,350 rows, one per recorded episode where the repository has per-episode records. 1,850 held-out episodes in the environment, 5,450 probe episodes in the environment, 3,050 probe episodes in a world model.

| Column | Description |
|--------|-------------|
| `task` | Task identifier |
| `agent` | The agent that produced the controller |
| `record_name` | The original name in the repository |
| `program` | Version label when applicable |
| `run_in` | Environment or world model |
| `evaluation` | Held-out or probe |
| `seed` | Episode seed |
| `success` | Whether the episode succeeded (empty for Doom) |
| `episode_return` | Episode return as recorded. Filled for every held-out row and for Doom probe rows. Empty for probe rows of other tasks. For Doom it is the number of game ticks survived |
| `n_steps` | Number of steps (where recorded) |
| `source_file` | The original per-episode file |

"Held-out" episodes use seeds from 1000 that no agent had access to. "Probe" episodes are additional runs made after the study for the predicted-against-real comparisons. The `seed` column gives the seed of each episode.

### `code`

317 rows, one per text file the agents wrote, with the full text. 6 world models (249 to 412 lines, 14 to 26 KB), 62 CaP programs, 177 scripts, 65 data files, 7 notes.

| Column | Description |
|--------|-------------|
| `task` | The environment family: doom, drone, push, cable, rope |
| `agent` | The agent that wrote the file |
| `kind` | One of: world model, program, script, data, notes |
| `path` | File path in the repository (joins `results.controller_file` and `programs.program_file`) |
| `n_lines` | Number of lines |
| `n_bytes` | File size in bytes |
| `text` | The full file text |

## Loading

```python
from datasets import load_dataset
programs = load_dataset("DenzelZheng/world-models-as-unit-tests", "programs", split="train")
```

Replace `"programs"` with `"results"`, `"episodes"` or `"code"` for the other subsets. Omitting the subset name loads `results` (the default).

## Glossary

The `code` texts and the raw files use the repository's working names.

| Working name | Meaning |
|-------------|---------|
| lives | Real trials |
| pilot | Agent |
| lab | Environment |
| code world | World model |
| official | Held-out evaluation |
| screening | Model-based evaluation |
| aspire_style | The CaP agent (modelled on ASPIRE, five-trial budget) |
| code_wm | The world-model agent |
| cap_in_code_world | The CaP agent using the world model as its tests |
| aspire_blind | The CaP agent without observation logs |
| Names ending in _extended | Runs with more real trials |

## Caveats

The environments are simulators that stand in for real systems. Each agent ran once per task. The results are not peer-reviewed.

`results` holds the controllers that have a result record in the repository. Reference controllers named in the article (the oracle planner, the stock autopilot, the random policy and similar) and the re-test runs are only in `raw/`.

`real_trials` is derived from the trial logs: the number of logged real trials whose steps add up to `real_steps`. For the CaP agent with the world model as its tests it is the sum of two agents' trials, the CaP agent's own (`real_trials_training`) and those the world model was fitted on (`real_trials_calibration`). On drone_C the record counts the five mission-C trials the world model was refitted on, but that world model had first been built from five trials on mission A, which the record does not count. The article's table therefore gives 20+10 where this table gives 20+5.

For drone_B the real trials are mission-A trials. No trial was flown on mission B.

The PPO baselines' record names give round numbers. The actual number of real episodes is in `real_trials` (for example the record named `ppo_real_100lives` used 102).

Doom has no success flag. Its value is the survival time in game ticks.

## How it was produced

Claude Fable 5.2 assigned tasks and checked isolation rules. Claude Opus 5.5 ran both agents (the world-model agent and the CaP agent). Claude Sonnet 5.5 built the environments (cable environment: Claude Opus 5.5). The `code` table is therefore code written by LLM agents.

## Licence

MIT.

## Citation

```bibtex
@misc{cawm2026,
  title       = {World Models as Unit Tests},
  author      = {Denzel Zheng},
  institution = {Princeton University},
  year        = {2026},
  note        = {Early research note. Simulators stand in for
                 reality; one run per agent per task; results
                 are not peer-reviewed.},
  howpublished = {\url{https://github.com/DongzheZheng/world-models-as-unit-tests}}
}
```
