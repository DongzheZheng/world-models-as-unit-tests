#!/usr/bin/env python
"""Build the dataset folder from the files recorded in this repository.

    python dataset/build_dataset.py <out_dir>        run from the repository root; standard library and numpy

    <out_dir>/README.md             dataset card: the YAML header, then dataset/card_body.md when that file exists
    <out_dir>/LICENSE               the repository's licence
    <out_dir>/data/results.jsonl    one row per recorded held-out evaluation (runs/*/results/*.json)
    <out_dir>/data/programs.jsonl   the programs of the predicted-against-real comparison (experiments/audit/fig_judge.py)
    <out_dir>/data/episodes.jsonl   one row per recorded episode, in the environment or in a world model
    <out_dir>/data/code.jsonl       one row per file the agents wrote
    <out_dir>/raw/runs/             runs/ as it is in the repository, without the model checkpoints (*.pt)

Nothing is re-run.  Every value is read from a file of the repository; a value that no file holds is null.
The tables use the public names of the article.  The names used inside the repository stay in the columns
record_name and reproduce_task and in the paths.  dataset/check_dataset.py verifies the result.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reproduce  # noqa: E402   REGISTRY and TASKS: tasks, controllers, factories and record files

# --------------------------------------------------------------------------------------------------------------------
# Public names
# --------------------------------------------------------------------------------------------------------------------
TASK_NAME = {"doom": "doom", "drone_A": "drone_A", "drone_B": "drone_B", "drone_C": "drone_C",   # reproduce.py -> public
             "pusht_T": "push_T", "pusht_L": "push_L", "cable": "cable", "rope": "rope"}
TASK_ORDER = list(TASK_NAME.values())
ENVIRONMENT, WORLD_MODEL = "environment", "world model"
HELD_OUT, PROBE = "held-out", "probe"
SUCCESS_RATE = "success rate"
SURVIVAL = "survival in game ticks"
SURVIVAL_SHARE = "share of episodes surviving 750 game ticks"
CAP_AGENT = "CaP agent"                                             # CaP = Code as Policies
WM_AGENT = "world-model agent"
CAP_WITH_WM = "CaP agent with the world model as its tests"
CAP_NO_LOGS = "CaP agent without observation logs"
PPO = "baseline: PPO trained in the environment"
PPO_NO_REUSE = "baseline: PPO trained in the environment, no data reuse"
NEURAL_WM = "baseline: neural world model with MPPI"
OPEN_LOOP = "baseline: open-loop router"


def agent_label(record_name):
    """Public label of a controller name used in the records (README.md "Record names"; reproduce.py `what`)."""
    if record_name in ("code_wm", "code_wm_extended"):
        return WM_AGENT
    if record_name in ("aspire_style", "aspire_style_extended"):
        return CAP_AGENT
    if record_name == "cap_in_code_world":
        return CAP_WITH_WM
    if record_name == "aspire_blind":
        return CAP_NO_LOGS
    if re.fullmatch(r"ppo_real_\d+lives", record_name):
        return PPO
    if re.fullmatch(r"ppo_real_\d+lives_noreuse", record_name):
        return PPO_NO_REUSE
    if re.fullmatch(r"neural_wm_\d+lives", record_name):
        return NEURAL_WM
    raise ValueError(f"no public label for record name {record_name!r}")


# --------------------------------------------------------------------------------------------------------------------
# Tables: column -> type.  Every row has exactly these columns, in this order; a value is of that type or null.
# The types are those the datasets library reads from the file: string, int (int64), float (float64), bool, and
# timestamp (a string "YYYY-MM-DD hh:mm:ss" in the file, read as timestamp[s]).
# --------------------------------------------------------------------------------------------------------------------
RESULTS = {
    "task": "string",                    # doom, drone_A, drone_B, drone_C, push_T, push_L, cable, rope
    "agent": "string",                   # public label of who produced the controller
    "extended_run": "bool",              # run continued with more real trials than the first submission
    "real_trials": "int",                # real_trials_training + real_trials_calibration
    "real_trials_training": "int",       # logged real trials whose steps add up to real_steps_training
    "real_trials_calibration": "int",    # logged real trials whose steps add up to real_steps_calibration
    "metric": "string",                  # success rate (fraction of episodes), or survival in game ticks (mean)
    "value": "float",                    # mean of the metric over the evaluation episodes
    "n_episodes": "int",
    "first_seed": "int",
    "last_seed": "int",
    "value_std": "float",
    "value_sem": "float",
    "value_median": "float",
    "value_min": "float",
    "value_max": "float",
    "real_steps": "int",                 # real_steps_calibration + real_steps_training
    "real_steps_calibration": "int",     # environment steps used to fit a world model
    "real_steps_training": "int",        # environment steps used to write the program or train the policy
    "world_model_steps": "int",          # simulated steps inside the world model
    "train_wall_s": "float",             # wall-clock seconds of training, search or tuning, as recorded
    "trained_in": "string",              # environment, or world model
    "record_name": "string",             # controller name in the repository's records
    "program": "string",                 # v1, v2, ... when the controller is a numbered program of a CaP agent
    "controller_file": "string",         # file that defines the evaluated controller (reproduce.py REGISTRY)
    "episodes_file": "string",           # per-episode record of this evaluation, when the repository has one
    "reproduce_task": "string",          # task name for `python reproduce.py real <task> <record_name>`
    "recorded_at": "timestamp",          # written in the record: local time of the machine, no time zone
    "real_trials_source": "string",      # trial-log directories (training logs first), or an index of checkpoints
    "source_file": "string",             # the record
}
PROGRAMS = {
    "task": "string",                    # drone_A, drone_C, push_T, push_L, doom
    "agent": "string",
    "program": "string",                 # v1, v2, ... in the order written
    "record_name": "string",             # name of the program in runs/probe/curves.json
    "written_after_real_trials": "int",  # real trials used before this program existed
    "submitted": "bool",                 # the program fig_judge.py marks as submitted for the held-out evaluation
    "metric": "string",                  # success rate; for doom the share of episodes surviving 750 game ticks
    "predicted": "float",                # in the world model
    "predicted_n": "int",                # simulated episodes behind `predicted`
    "real": "float",                     # in the environment, seeds no agent had seen
    "real_n": "int",                     # environment episodes behind `real`
    "program_file": "string",
    "predicted_source": "string",
    "real_source": "string",
}
EPISODES = {
    "task": "string",
    "agent": "string",
    "record_name": "string",             # name of the controller or program in the source file
    "program": "string",                 # v1, v2, ... when the row is a numbered program of a CaP agent
    "run_in": "string",                  # environment, or world model
    "evaluation": "string",              # held-out (the recorded evaluation of results), or probe (run afterwards)
    "seed": "int",                       # environment seed; in a world model, the seed whose reset reading starts it
    "success": "bool",                   # null for doom, which has no success flag
    "episode_return": "float",           # episode return where recorded; for doom the game ticks survived
    "n_steps": "int",                    # control steps where recorded
    "source_file": "string",
}
CODE = {
    "task": "string",                    # doom, drone (A, B, C), push (T, L), cable, rope
    "agent": "string",
    "kind": "string",                    # world model, program, script, data, notes
    "path": "string",
    "n_lines": "int",
    "n_bytes": "int",
    "text": "string",
}
PY = {"string": str, "int": int, "float": float, "bool": bool, "timestamp": str}
TIMESTAMP = "%Y-%m-%d %H:%M:%S"     # a timestamp is written as this string; the Arrow JSON reader types it timestamp[s]


def write_table(path, schema, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    plain = {col: None for col, kind in schema.items() if kind == "string"}    # True: has a value that is no timestamp
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            if set(row) != set(schema):
                raise ValueError(f"{path.name}: columns {sorted(row)} differ from the schema")
            row = {col: row[col] for col in schema}
            for col, kind in schema.items():
                if row[col] is None:
                    continue
                if type(row[col]) is not PY[kind] or kind == "timestamp" and not is_timestamp(row[col]):
                    raise TypeError(f"{path.name}: {col}={row[col]!r} is not {kind}")
                if kind == "string":
                    plain[col] = plain[col] or not is_timestamp(row[col])
            line = json.dumps(row, ensure_ascii=False, allow_nan=False)
            f.write(line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + "\n")
    if False in plain.values():
        raise TypeError(f"{path.name}: string columns that would be read as timestamps: {[c for c, p in plain.items() if p is False]}")
    return len(rows)


def is_timestamp(value):
    if not isinstance(value, str):
        return False
    try:
        return datetime.strptime(value, TIMESTAMP).strftime(TIMESTAMP) == value
    except ValueError:
        return False


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def as_int(x):
    if int(x) != x:
        raise ValueError(f"{x!r} is not a whole number")
    return int(x)


def rel_if_file(rel):
    return rel if rel and (ROOT / rel).is_file() else None


def factory_file(factory):
    """'package.module:function' -> 'package/module.py'."""
    return factory.split(":")[0].replace(".", "/") + ".py"


def program_of(factory):
    m = re.search(r"\.policy_(v\d+):", factory)
    return m.group(1) if m else None


# --------------------------------------------------------------------------------------------------------------------
# Real trials.  The records hold real steps, not a number of real trials.  The trial logs are in runs/: one file per
# real trial, in the order flown.  A count of real trials is the number of leading logs whose steps add up to the
# record's real steps (some agents flew further trials after their evaluation), 0 for 0 steps, and null when the logs
# give no such number.  It is taken separately for the two step counts of a record: real_trials_training for
# real_steps_training, real_trials_calibration for real_steps_calibration; real_trials is their sum.
# A list below holds the logs of one agent, which stand behind the one step count of its record that is not zero.
# The two records of the CaP agent with the world model as its tests carry the steps of two agents' trials and no
# new trial: those of the CaP agent behind the program (training) and those the world model was fitted on
# (calibration).
# --------------------------------------------------------------------------------------------------------------------
TRIAL_LOGS = {   # (public task, record name) -> directories of trial logs, in order
    ("drone_C", "cap_in_code_world"): dict(
        training=["runs/drone/blackbox/aspire_style_C", "runs/drone/blackbox/aspire_style_C_x1",
                  "runs/drone/blackbox/aspire_style_C_x2", "runs/drone/blackbox/aspire_style_C_x3"],
        calibration=["runs/drone/blackbox/code_wm_C"]),
    ("push_L", "cap_in_code_world"): dict(
        training=["runs/pusht/blackbox_L/aspire_style", "runs/pusht/blackbox_L/aspire_style_x1",
                  "runs/pusht/blackbox_L/aspire_style_x2", "runs/pusht/blackbox_L/aspire_style_x3"],
        calibration=["runs/pusht/blackbox_L/code_wm"]),
    ("doom", "aspire_style"): ["runs/doom/blackbox/take_cover/aspire_style"],
    ("doom", "code_wm"): ["runs/doom/blackbox/take_cover/code_wm"],
    ("drone_A", "aspire_style"): ["runs/drone/blackbox/aspire_style"],
    ("drone_A", "code_wm"): ["runs/drone/blackbox/code_wm"],
    ("drone_A", "ppo_real_5lives"): ["runs/drone/blackbox/ppo_real"],
    ("drone_A", "neural_wm_5lives"): ["runs/drone/blackbox/neural_wm"],
    ("drone_B", "aspire_style"): ["runs/drone/blackbox/aspire_style"],        # mission B: no trial on it, those of mission A
    ("drone_B", "code_wm"): ["runs/drone/blackbox/code_wm"],
    ("drone_C", "aspire_style"): ["runs/drone/blackbox/aspire_style_C"],
    ("drone_C", "aspire_style_extended"): ["runs/drone/blackbox/aspire_style_C", "runs/drone/blackbox/aspire_style_C_x1",
                                           "runs/drone/blackbox/aspire_style_C_x2", "runs/drone/blackbox/aspire_style_C_x3"],
    ("drone_C", "code_wm"): ["runs/drone/blackbox/code_wm_C"],
    ("push_T", "aspire_style"): ["runs/pusht/blackbox/aspire_style"],
    ("push_T", "aspire_style_extended"): ["runs/pusht/blackbox/aspire_style", "runs/pusht/blackbox/aspire_style_x1",
                                          "runs/pusht/blackbox/aspire_style_x2"],
    ("push_T", "code_wm"): ["runs/pusht/blackbox/code_wm"],
    ("push_L", "aspire_style"): ["runs/pusht/blackbox_L/aspire_style"],
    ("push_L", "aspire_style_extended"): ["runs/pusht/blackbox_L/aspire_style", "runs/pusht/blackbox_L/aspire_style_x1",
                                          "runs/pusht/blackbox_L/aspire_style_x2", "runs/pusht/blackbox_L/aspire_style_x3"],
    ("push_L", "code_wm"): ["runs/pusht/blackbox_L/code_wm"],
    ("cable", "aspire_style"): ["runs/station/blackbox/aspire_style"],
    ("cable", "aspire_blind"): ["runs/station/blackbox/aspire_blind"],
    ("cable", "code_wm"): ["runs/station/blackbox/code_wm"],
    ("rope", "aspire_style"): ["runs/annex/blackbox/aspire_style"],
    ("rope", "aspire_style_extended"): ["runs/annex/blackbox/aspire_style", "runs/annex/blackbox/aspire_style_x1",
                                        "runs/annex/blackbox/aspire_style_x2", "runs/annex/blackbox/aspire_style_x3"],
    ("rope", "code_wm"): ["runs/annex/blackbox/code_wm"],
    ("rope", "code_wm_extended"): ["runs/annex/blackbox/code_wm", "runs/annex/blackbox/code_wm_x1",
                                   "runs/annex/blackbox/code_wm_x2", "runs/annex/blackbox/code_wm_x3"],
}
DOOM_PPO_INDEX = "runs/doom/rl_baseline/eval_done.json"          # checkpoint name -> real episodes and steps
DRONE_PPO_INDEX = "runs/drone/baselines_rl/ppo_curve.json"       # checkpoint file -> real episodes and steps


def trial_steps(path):
    """Environment steps in the log of one real trial."""
    if path.name.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            return int(z["n_steps"])
    with gzip.open(path, "rt", encoding="utf-8") as f:           # .jsonl.gz: one line per step after the header lines
        return sum("action" in json.loads(line) for line in f)


def trial_logs(dirs):
    out = []
    for d in dirs:
        out += sorted((ROOT / d).glob("life_*"), key=lambda p: int(re.search(r"life_(\d+)", p.name).group(1)))
    return out


def count_trials(dirs, steps):
    """Number of leading trial logs in `dirs` whose steps add up to `steps`: 0 for 0 steps, None when there is none."""
    if steps == 0:
        return 0
    total = 0
    for k, p in enumerate(trial_logs(dirs), 1):
        total += trial_steps(p)
        if total == steps:
            return k
        if total > steps:
            break
    return None


def real_trials(task, name, entry, training_steps, calibration_steps):
    """(trials behind the training steps, trials behind the calibration steps, where they were read).
    A count is None when the logs do not give it; the source lists the training logs first."""
    logs = TRIAL_LOGS.get((task, name))
    if isinstance(logs, list):
        if training_steps and calibration_steps:
            raise ValueError(f"{task}/{name}: one list of trial logs for two step counts")
        logs = dict(training=logs if training_steps else [], calibration=logs if calibration_steps else [])
    if logs:
        return (count_trials(logs["training"], training_steps), count_trials(logs["calibration"], calibration_steps),
                ";".join(logs["training"] + logs["calibration"]) or None)
    n_calibration = 0 if calibration_steps == 0 else None
    if task == "doom" and name.startswith("ppo_real_"):     # PPO checkpoints: the index of the training run
        e = load(DOOM_PPO_INDEX).get(name)
        if e and e["real_steps"] == training_steps:
            return as_int(e["real_episodes"]), n_calibration, DOOM_PPO_INDEX
    if task == "drone_A" and name.startswith("ppo_real_") and entry:
        for c in load(DRONE_PPO_INDEX)["curve"]:
            if c["ckpt"] == entry["kwargs"].get("ckpt") and c["real_steps"] == training_steps:
                return as_int(c["real_episodes"]), n_calibration, DRONE_PPO_INDEX
    return (0 if training_steps == 0 else None), n_calibration, None


# --------------------------------------------------------------------------------------------------------------------
# results: one row per runs/*/results/*.json
# --------------------------------------------------------------------------------------------------------------------
def build_results():
    task_of = {(demo, scen): task for task, (_, demo, scen, _) in reproduce.TASKS.items()}
    registry = {(e["task"], e["controller"]): (i, e) for i, e in enumerate(reproduce.REGISTRY)}
    keyed = []
    for p in sorted(ROOT.glob("runs/*/results/*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        name = rec["controller"]
        if p.name != f'{rec["scenario"]}__{name}.json':
            raise ValueError(f"{p}: file name and record disagree")
        rtask = task_of[(rec["demo"], rec["scenario"])]
        task = TASK_NAME[rtask]
        order, entry = registry.get((rtask, name), (len(registry), None))
        ev, seeds = rec["eval"], rec["seeds"]
        if len(seeds) != ev["n"]:
            raise ValueError(f"{p}: {len(seeds)} seeds for n={ev['n']}")
        steps_training, steps_calibration = as_int(rec["real_steps_training"]), as_int(rec["real_steps_calibration"])
        n_training, n_calibration, trials_src = real_trials(task, name, entry, steps_training, steps_calibration)
        keyed.append((TASK_ORDER.index(task), order, name, {
            "task": task,
            "agent": agent_label(name),
            "extended_run": name.endswith("_extended"),
            "record_name": name,
            "program": program_of(entry["factory"]) if entry else None,
            "metric": {"success": SUCCESS_RATE, "ret": SURVIVAL}[rec["metric"]],
            "value": float(ev["mean"]),
            "n_episodes": as_int(ev["n"]),
            "value_std": float(ev["std"]),
            "value_sem": float(ev["sem"]),
            "value_median": float(ev["median"]),
            "value_min": float(ev["min"]),
            "value_max": float(ev["max"]),
            "first_seed": as_int(min(seeds)),
            "last_seed": as_int(max(seeds)),
            "real_trials": None if None in (n_training, n_calibration) else n_training + n_calibration,
            "real_trials_training": n_training,
            "real_trials_calibration": n_calibration,
            "real_steps": steps_training + steps_calibration,
            "real_steps_calibration": steps_calibration,
            "real_steps_training": steps_training,
            "world_model_steps": as_int(rec["wm_steps"]),
            "train_wall_s": float(rec["train_wall_s"]),
            "trained_in": {"real": ENVIRONMENT, "code_world": WORLD_MODEL}[rec["trained_in"]],
            "controller_file": rel_if_file(factory_file(entry["factory"])) if entry else None,
            "episodes_file": rel_if_file(entry["episodes"]) if entry else None,
            "reproduce_task": rtask if entry else None,
            "recorded_at": rec["written"],
            "real_trials_source": trials_src,
            "source_file": p.relative_to(ROOT).as_posix(),
        }))
    keyed.sort(key=lambda k: k[:3])
    return [row for *_, row in keyed]


# --------------------------------------------------------------------------------------------------------------------
# programs: the comparison of experiments/audit/fig_judge.py.  This table repeats its PANELS (task, programs, the
# predicted value it reads, the program it marks as submitted); check_dataset.py compares it with that file.
# --------------------------------------------------------------------------------------------------------------------
CURVES = "runs/probe/curves.json"
JUDGE_PANELS = [   # task, key of the task in curves.json, programs, file and keys of the predicted values, their field
    dict(task="drone_A", curves="drone_A", versions=range(1, 6), file="runs/probe/screen_drone.json", keys=("A",),
         field="success", submitted=5, programs="pilots/aspire_drone"),
    dict(task="drone_C", curves="drone_C", versions=range(6, 15), file="runs/probe/screen_drone.json", keys=("C",),
         field="success", submitted=11, programs="pilots/aspire_drone"),
    dict(task="push_T", curves="push_T", versions=range(1, 9), file="runs/probe/screen_pusht.json", keys=("T",),
         field="success", submitted=5, programs="pilots/aspire_pusht"),
    dict(task="push_L", curves="push_L", versions=range(9, 16), file="runs/probe/screen_pusht.json", keys=("L",),
         field="success", submitted=10, programs="pilots/aspire_pusht"),
    dict(task="doom", curves="doom", versions=range(1, 6), file="runs/probe/screen_doom.json", keys=(),
         field="frac_ge_750", submitted=5, programs="pilots/aspire_doom"),
]


def build_programs():
    truth = load(CURVES)
    rows = []
    for panel in JUDGE_PANELS:
        predicted = load(panel["file"])
        for k in panel["keys"]:
            predicted = predicted[k]
        for v in panel["versions"]:
            name = f"aspire_v{v}"
            real, pred = truth[f'{panel["curves"]}/{name}'], predicted[f"v{v}"]
            rows.append({
                "task": panel["task"],
                "agent": CAP_AGENT,
                "program": f"v{v}",
                "record_name": name,
                "written_after_real_trials": as_int(real["lives_before"]),
                "submitted": v == panel["submitted"],
                "metric": SURVIVAL_SHARE if panel["task"] == "doom" else SUCCESS_RATE,
                "predicted": float(pred[panel["field"]]),
                "predicted_n": as_int(pred["n"]),
                "real": float(real["success"]),
                "real_n": as_int(real["n"]),
                "program_file": rel_if_file(f'{panel["programs"]}/policy_v{v}.py'),
                "predicted_source": panel["file"],
                "real_source": CURVES,
            })
    return rows


# --------------------------------------------------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------------------------------------------------
def episode(task, agent, name, program, where, evaluation, seed, success, ret, n_steps, source):
    return {"task": task, "agent": agent, "record_name": name, "program": program, "run_in": where,
            "evaluation": evaluation, "seed": as_int(seed),
            "success": None if success is None else bool(success),
            "episode_return": None if ret is None else float(ret),
            "n_steps": None if n_steps is None else as_int(n_steps),
            "source_file": source}


def recorded_episodes(rel):
    d = load(rel)
    if isinstance(d, dict):
        d = d.get("episodes", d.get("rows"))
    return d


def held_out_episodes(results):
    """Per-episode records of the held-out evaluations (reproduce.py REGISTRY, `episodes=`)."""
    rows = []
    for r in results:
        if not r["episodes_file"]:
            continue
        eps = sorted(recorded_episodes(r["episodes_file"]), key=lambda e: int(e["seed"]))
        for e in eps:
            rows.append(episode(r["task"], r["agent"], r["record_name"], r["program"], ENVIRONMENT, HELD_OUT, e["seed"],
                                e.get("success"), e["ret"], e["n_steps"], r["episodes_file"]))
    return rows


def program_probe_episodes(programs):
    """Environment episodes of the programs of the comparison (runs/probe/curves.json, `per_seed`).
    per_seed holds the success flag, and for doom the episode return."""
    truth = load(CURVES)
    key = {p["task"]: p["curves"] for p in JUDGE_PANELS}
    rows = []
    for p in programs:
        per_seed = truth[f'{key[p["task"]]}/{p["record_name"]}']["per_seed"]
        for seed in sorted(per_seed, key=int):
            x = per_seed[seed]
            doom = p["task"] == "doom"
            if isinstance(x, bool) == doom:
                raise TypeError(f'{p["task"]}/{p["record_name"]}: unexpected per_seed value {x!r}')
            rows.append(episode(p["task"], p["agent"], p["record_name"], p["program"], ENVIRONMENT, PROBE, int(seed),
                                None if doom else x, x if doom else None, None, CURVES))
    return rows


ROPE_PROBE = "runs/annex/audit/screen/{}_{}.json"       # real | world, first seed; episode i has seed first + i


def rope_probe_episodes():
    """Every rope program of the CaP agent, run in the environment and in the world model from the reset readings
    of the same seeds (experiments/audit/annex_screen.py)."""
    rows = []
    for first in (1000, 5000):
        files = {ENVIRONMENT: ROPE_PROBE.format("real", first), WORLD_MODEL: ROPE_PROBE.format("world", first)}
        data = {w: load(f) for w, f in files.items() if (ROOT / f).is_file()}
        names = sorted({v for d in data.values() for v in d}, key=lambda v: int(v[1:]))
        for v in names:
            for where, d in data.items():
                for i, e in enumerate(d.get(v, [])):
                    rows.append(episode("rope", CAP_AGENT, v, v, where, PROBE, first + i, e["success"], None,
                                        e["n_steps"], files[where]))
    return rows


CABLE_PROBE = "runs/probe/station_probe.json"           # {"real" | "world": {"<label>@<first seed>": {per_seed: [0|1]}}}


def cable_label(label):
    for pattern, agent in ((r"cap_(v\d+)", CAP_AGENT), (r"blind_(v\d+)", CAP_NO_LOGS)):
        m = re.fullmatch(pattern, label)
        if m:
            return agent, m.group(1)
    if label == "code_wm_ctrl":
        return WM_AGENT, None
    if label.startswith("open_loop_"):                  # the environment's reference router at a fixed speed and clearance
        return OPEN_LOOP, None
    raise ValueError(f"no public label for cable program {label!r}")


def cable_probe_episodes():
    """Cable programs run in the environment and in the world model from the reset readings of the same seeds
    (experiments/audit/station_probe.py)."""
    d = load(CABLE_PROBE)
    rows = []
    for where, part in ((ENVIRONMENT, "real"), (WORLD_MODEL, "world")):
        for key, rec in d[part].items():
            label, first = key.rsplit("@", 1)
            if int(first) != rec["first_seed"] or len(rec["per_seed"]) != rec["n"]:
                raise ValueError(f"{CABLE_PROBE}: {key} is inconsistent")
            agent, program = cable_label(label)
            for i, ok in enumerate(rec["per_seed"]):
                rows.append(episode("cable", agent, label, program, where, PROBE, int(first) + i, ok, None, None,
                                    CABLE_PROBE))
    return rows


# --------------------------------------------------------------------------------------------------------------------
# code: the files the agents wrote
# --------------------------------------------------------------------------------------------------------------------
WORLD_DIRS = {"fps": "doom", "drone": "drone", "pusht": "push", "station": "cable", "annex": "rope"}
PROGRAM_DIRS = {"aspire_doom": ("doom", CAP_AGENT), "aspire_drone": ("drone", CAP_AGENT),
                "aspire_pusht": ("push", CAP_AGENT), "aspire_station": ("cable", CAP_AGENT),
                "aspire_blind": ("cable", CAP_NO_LOGS), "aspire_annex": ("rope", CAP_AGENT)}
SCRIPT_DIRS = {"doom": "doom", "drone": "drone", "pusht": "push", "station": "cable", "annex": "rope"}
OTHER_KINDS = {".py": "script", ".json": "data", ".txt": "notes"}


def files_under(rel):
    return sorted((p for p in (ROOT / rel).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix not in (".pyc", ".pyo")),
                  key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.relative_to(ROOT).as_posix())])


def build_code():
    listed = []   # (path, task, agent, kind)
    for p in files_under("worlds"):
        parts = p.relative_to(ROOT).parts
        if len(parts) > 2:
            listed.append((p, WORLD_DIRS[parts[1]], WM_AGENT, "world model" if p.suffix == ".py" else OTHER_KINDS.get(p.suffix, "other")))
        elif p.stat().st_size:
            raise ValueError(f"unexpected file {p}")
    for p in files_under("pilots"):
        parts = p.relative_to(ROOT).parts
        if len(parts) > 2:
            task, agent = PROGRAM_DIRS[parts[1]]
            program = len(parts) == 3 and re.fullmatch(r"policy_v\d+\.py", p.name)
            listed.append((p, task, agent, "program" if program else OTHER_KINDS.get(p.suffix, "other")))
        elif p.stat().st_size:
            raise ValueError(f"unexpected file {p}")
    for d in sorted(ROOT.glob("experiments/*/code_wm")):
        for p in files_under(d.relative_to(ROOT).as_posix()):
            listed.append((p, SCRIPT_DIRS[d.parent.name], WM_AGENT, OTHER_KINDS.get(p.suffix, "other")))
    rows, skipped = [], []
    for p, task, agent, kind in listed:
        raw = p.read_bytes()
        rel = p.relative_to(ROOT).as_posix()
        if not raw and p.name == "__init__.py":
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is None or "\x00" in text:
            skipped.append(rel)                         # binary
            continue
        rows.append({"task": task, "agent": agent, "kind": kind, "path": rel,
                     "n_lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
                     "n_bytes": len(raw), "text": text})
    return rows, skipped


# --------------------------------------------------------------------------------------------------------------------
# dataset card, raw copy, main
# --------------------------------------------------------------------------------------------------------------------
RAW_NOT_RUN_RECORDS = ("reproduce", "ledger.jsonl*", "__pycache__")   # outputs of re-runs, not part of the records
RAW_LEFT_OUT = ("*.pt",)                                # model checkpoints: weights, not records (they stay in the repository)
CARD_BODY = ROOT / "dataset" / "card_body.md"           # text of the dataset card, written separately; only read here
CARD_HEADER = """---
license: mit
language:
- en
pretty_name: World Models as Unit Tests
tags:
- world-models
- robotics
- code-as-policies
- llm-agents
- agent-generated-code
- code-generation
- evaluation
- simulation
task_categories:
- robotics
- reinforcement-learning
size_categories:
- {size}
configs:
- config_name: results
  data_files: "data/results.jsonl"
  default: true
- config_name: programs
  data_files: "data/programs.jsonl"
- config_name: episodes
  data_files: "data/episodes.jsonl"
- config_name: code
  data_files: "data/code.jsonl"
---
"""


def size_category(n):
    for bound, label in ((10**3, "n<1K"), (10**4, "1K<n<10K"), (10**5, "10K<n<100K"), (10**6, "100K<n<1M")):
        if n < bound:
            return label
    return "1M<n<10M"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1]).expanduser().resolve()
    if out == ROOT or ROOT in out.parents and out.relative_to(ROOT).parts[:2] != ("dataset", "out"):
        sys.exit("the output folder must be outside the repository, or dataset/out")
    for sub in ("data", "raw"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    out.mkdir(parents=True, exist_ok=True)

    results = build_results()
    programs = build_programs()
    episodes = held_out_episodes(results) + program_probe_episodes(programs) + rope_probe_episodes() + cable_probe_episodes()
    code, skipped = build_code()
    counts = {
        "results": write_table(out / "data" / "results.jsonl", RESULTS, results),
        "programs": write_table(out / "data" / "programs.jsonl", PROGRAMS, programs),
        "episodes": write_table(out / "data" / "episodes.jsonl", EPISODES, episodes),
        "code": write_table(out / "data" / "code.jsonl", CODE, code),
    }
    shutil.copytree(ROOT / "runs", out / "raw" / "runs",
                    ignore=shutil.ignore_patterns(*RAW_NOT_RUN_RECORDS, *RAW_LEFT_OUT))
    for d in sorted((p for p in (out / "raw").rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):                            # a folder that held only left-out files
            d.rmdir()
    shutil.copyfile(ROOT / "LICENSE", out / "LICENSE")
    card = CARD_HEADER.format(size=size_category(sum(counts.values())))
    if CARD_BODY.is_file():                                 # header, blank line, body
        body = CARD_BODY.read_text(encoding="utf-8")
        card += "\n" + body + ("" if body.endswith("\n") else "\n")
    (out / "README.md").write_text(card, encoding="utf-8")

    for name, n in counts.items():
        print(f"{name:9s} {n:6d} rows   data/{name}.jsonl")
    n_raw = sum(p.is_file() for p in (out / "raw").rglob("*"))
    left_out = [p for pattern in RAW_LEFT_OUT for p in (ROOT / "runs").rglob(pattern) if p.is_file()]
    print(f"raw/runs  {n_raw:6d} files; left out: {len(left_out)} model checkpoints ({', '.join(RAW_LEFT_OUT)}), "
          f"{sum(p.stat().st_size for p in left_out):,} bytes")
    print("README.md  header" + (f" and body ({CARD_BODY.relative_to(ROOT).as_posix()})" if CARD_BODY.is_file() else " only"))
    if skipped:
        print("not in code.jsonl (binary):", ", ".join(skipped))
    print("written to", out)


if __name__ == "__main__":
    main()
