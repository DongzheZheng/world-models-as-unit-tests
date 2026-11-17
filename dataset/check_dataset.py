#!/usr/bin/env python
"""Verify a dataset folder written by dataset/build_dataset.py against the files of this repository.

    python dataset/check_dataset.py <out_dir> [--private-terms a,b,c] [--public-names d,e]      from the repository root

Prints one line per check ("ok" or "FAIL"), detail lines indented below it, and a final PASS or FAIL (exit status
0 or 1).  The checks read the repository on their own; they do not import build_dataset.py.  Standard library and
numpy; the load test runs only in an interpreter that has the datasets library.

--private-terms   further strings for the private-string scan (user names, hosts), given on the command line so
                  that no file holds them.  The scan always looks for home-directory and mount paths, e-mail
                  addresses and the user and host name of the machine it runs on.
--public-names    names the dataset states on purpose (author, account, institution): allowed in LICENSE and
                  README.md only, where their occurrences are listed.
"""
from __future__ import annotations

import argparse
import ast
import collections
import getpass
import gzip
import hashlib
import html
import io
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reproduce  # noqa: E402

TABLES = ("results", "programs", "episodes", "code")
TASK_NAME = {"pusht_T": "push_T", "pusht_L": "push_L"}          # reproduce.py task -> public task, where they differ
FAILED = []


def check(name, ok, detail=""):
    print(f'{"ok  " if ok else "FAIL"} {name}{": " + detail if detail else ""}', flush=True)
    if not ok:
        FAILED.append(name)
    return ok


def info(text):
    print("       " + text, flush=True)


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


# --------------------------------------------------------------------------------------------------------------------
# 1. the tables parse, with the same columns in every row and one type per column
# --------------------------------------------------------------------------------------------------------------------
TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")


def type_name(v):
    """Type of a JSON value as the datasets library reads a column of such values.  A column of date-and-time
    strings is read as a timestamp, so such a string is its own type here."""
    if type(v) is str and TIMESTAMP.fullmatch(v):
        return "timestamp"
    for t, name in ((bool, "bool"), (int, "int"), (float, "float"), (str, "string")):
        if type(v) is t:
            return name
    return "null" if v is None else "nested"


def read_table(path):
    rows = []
    with open(path, encoding="utf-8", newline="\n") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def check_tables(out):
    tables = {}
    for name in TABLES:
        path = out / "data" / f"{name}.jsonl"
        try:
            rows = read_table(path)
        except (OSError, ValueError) as e:
            check(f"table {name} parses", False, str(e))
            tables[name] = []
            continue
        cols = list(rows[0]) if rows else []
        same = all(list(r) == cols for r in rows)
        types = {c: collections.Counter(type_name(r.get(c)) for r in rows) for c in cols}
        bad = [c for c, t in types.items() if len(set(t) - {"null"}) != 1 or "nested" in t]
        check(f"table {name} parses", bool(rows) and same and not bad,
              f"{len(rows)} rows, {len(cols)} columns, same columns in every row: {same}, columns without a single type: {bad or 'none'}")
        info(", ".join(f'{c}:{next(iter(set(t) - {"null"}), "null")}' + (f' ({t["null"]} null)' if t["null"] else "")
                       for c, t in types.items()))
        tables[name] = rows
    return tables


# --------------------------------------------------------------------------------------------------------------------
# 2. results against the records behind reproduce.py's REGISTRY
# --------------------------------------------------------------------------------------------------------------------
ARTICLE_REAL_TRIALS = {   # column "Real trials" of the summary table in blog/body.html, rows with a single number
    ("drone_A", "ppo_real_5lives"): 5, ("drone_A", "neural_wm_5lives"): 5, ("drone_A", "aspire_style"): 5,
    ("drone_A", "code_wm"): 4,
    ("drone_C", "aspire_style"): 5, ("drone_C", "aspire_style_extended"): 20, ("drone_C", "code_wm"): 5,
    ("doom", "ppo_real_5lives"): 5, ("doom", "aspire_style"): 5, ("doom", "code_wm"): 4,
    ("push_T", "aspire_style"): 5, ("push_T", "aspire_style_extended"): 13, ("push_T", "code_wm"): 4,
    ("push_L", "aspire_style"): 2, ("push_L", "aspire_style_extended"): 8, ("push_L", "code_wm"): 2,
    ("cable", "aspire_style"): 5, ("cable", "aspire_blind"): 5, ("cable", "code_wm"): 3,
    ("rope", "aspire_style"): 5, ("rope", "aspire_style_extended"): 19, ("rope", "code_wm"): 4,
    ("rope", "code_wm_extended"): 20,
}


ARTICLE_TWO_AGENT_ROWS = [("drone_C", "cap_in_code_world"), ("push_L", "cap_in_code_world")]   # rows "CaP in world
# model" of that table, in its order; their cell is a sum of two terms with a footnote, so it is printed, not compared


def logged_steps(path):
    """Environment steps in the log of one real trial."""
    if path.name.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            return int(z["n_steps"])
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return sum("action" in json.loads(line) for line in f)


def trial_logs(dirs):
    logs = []
    for d in dirs:
        logs += sorted((ROOT / d).glob("life_*"), key=lambda p: int(re.search(r"life_(\d+)", p.name).group(1)))
    return logs


def leading_logs(dirs, steps):
    """Number of leading trial logs in `dirs` whose steps add up to `steps`: 0 for 0 steps, None when there is none."""
    if steps == 0:
        return 0 if not dirs else None      # no logs may be listed for a step count of zero
    total = 0
    for k, p in enumerate(trial_logs(dirs), 1):
        total += logged_steps(p)
        if total >= steps:
            return k if total == steps else None
    return None


def check_results(results):
    rows = {(r["task"], r["record_name"]): r for r in results}
    check("results: one row per task and record name", len(rows) == len(results), f"{len(results)} rows")
    files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob("runs/*/results/*.json"))
    check("results: one row per file runs/*/results/*.json", sorted(r["source_file"] for r in results) == files,
          f"{len(files)} files")
    mismatches, with_record = [], 0
    for e in reproduce.REGISTRY:
        rec = reproduce._record(e)
        if rec is None:
            continue
        with_record += 1
        row = rows.get((TASK_NAME.get(e["task"], e["task"]), e["controller"]))
        name = f'{e["task"]}/{e["controller"]}'
        if row is None:
            mismatches.append(f"{name}: no row")
        elif row["value"] != rec["eval"]["mean"] or row["n_episodes"] != rec["eval"]["n"]:
            mismatches.append(f'{name}: table {row["value"]} (n={row["n_episodes"]}), record {rec["eval"]["mean"]} (n={rec["eval"]["n"]})')
        elif (row["first_seed"], row["last_seed"]) != (rec["seeds"][0], rec["seeds"][-1]) or \
                rec["seeds"] != list(range(rec["seeds"][0], rec["seeds"][-1] + 1)):
            mismatches.append(f"{name}: seeds differ")
        elif (row["metric"] == "survival in game ticks") != (reproduce._metric(e) == "ret"):
            mismatches.append(f"{name}: metric differs")
    check("results: value and n equal the record for every REGISTRY entry with a record", not mismatches,
          f"{with_record} of {len(reproduce.REGISTRY)} REGISTRY entries have a record" +
          ("; mismatches: " + "; ".join(mismatches) if mismatches else ""))
    by_task = collections.Counter(r["task"] for r in results)
    info("rows per task: " + ", ".join(f"{t} {n}" for t, n in by_task.items()))

    # real trials: the two counts add up to the total, and each equals the number of leading trial logs whose steps
    # add up to its step count.  real_trials_source lists the training logs first, then the calibration logs.
    part_sum = [f'{r["task"]}/{r["record_name"]}' for r in results
                if r["real_trials"] != (None if None in (r["real_trials_training"], r["real_trials_calibration"])
                                        else r["real_trials_training"] + r["real_trials_calibration"])
                or r["real_steps"] != r["real_steps_training"] + r["real_steps_calibration"]]
    check("results: real_trials = real_trials_training + real_trials_calibration (and the same for real_steps)", not part_sum,
          ", ".join(part_sum) or f"{len(results)} rows")
    bad, n_logs, two_agents = [], 0, []
    for r in results:
        src, name = r["real_trials_source"], f'{r["task"]}/{r["record_name"]}'
        if src is None or src.endswith(".json"):
            continue
        dirs = src.split(";")
        n_logs += 1
        split = next((s for s in range(len(dirs), -1, -1)
                      if (leading_logs(dirs[:s], r["real_steps_training"]), leading_logs(dirs[s:], r["real_steps_calibration"]))
                      == (r["real_trials_training"], r["real_trials_calibration"])), None)
        if split is None or r["real_trials"] is None:
            bad.append(name)
            continue
        parts = [(part, dirs[a:b], r[f"real_trials_{part}"], r[f"real_steps_{part}"])
                 for part, a, b in (("training", 0, split), ("calibration", split, len(dirs))) if dirs[a:b]]
        for part, ds, n, steps in parts:
            if len(trial_logs(ds)) > n:
                info(f'{name}: {n} of the {len(trial_logs(ds))} trial logs in {";".join(ds)} precede the evaluation')
        if len(parts) == 2:
            two_agents.append(f"{name}: " + " + ".join(
                f'{n} trials = {steps} steps ({part}; {len(trial_logs(ds))} logs in {", ".join(d.rsplit("/", 1)[-1] for d in ds)})'
                for part, ds, n, steps in parts) + f' = {r["real_trials"]} trials, {r["real_steps"]} steps')
    check("results: each real-trial count equals the number of leading trial logs whose steps add up to its step count",
          not bad, ", ".join(bad) or f"{n_logs} rows counted from trial logs")
    for line in two_agents:
        info(line)
    index = {r["record_name"]: r for r in results if (r["real_trials_source"] or "").endswith(".json")}
    renamed = {k: r["real_trials"] for k, r in index.items() if int(re.search(r"_(\d+)lives", k).group(1)) != r["real_trials"]}
    info(f"{len(index)} rows counted from an index of checkpoints; name and count differ for: " +
         (", ".join(f"{k} = {v}" for k, v in renamed.items()) or "none"))
    rows_by_key = {(r["task"], r["record_name"]): r for r in results}
    differ = [f'{t}/{n}: table {rows_by_key[(t, n)]["real_trials"] if (t, n) in rows_by_key else None}, article {v}'
              for (t, n), v in ARTICLE_REAL_TRIALS.items()
              if (t, n) not in rows_by_key or rows_by_key[(t, n)]["real_trials"] != v]
    check("results: real_trials equals the count in the article's summary table (blog/body.html), rows with a single number",
          not differ, "; ".join(differ) or f"{len(ARTICLE_REAL_TRIALS)} rows of that table")
    cells = re.findall(r'<td>CaP in world model</td><td class="num">([^<]*)</td>', (ROOT / "blog/body.html").read_text(encoding="utf-8"))
    for key, cell in zip(ARTICLE_TWO_AGENT_ROWS, cells if len(cells) == len(ARTICLE_TWO_AGENT_ROWS) else [None] * 2):
        r = rows_by_key.get(key)
        info(f'{key[0]}/{key[1]} (not compared): table real_trials = {r["real_trials"]} '
             f'({r["real_trials_training"]} training + {r["real_trials_calibration"]} calibration); article cell: '
             + (repr(html.unescape(cell)) if cell is not None else "not found"))
    nulls = [f'{r["task"]}/{r["record_name"]}' for r in results
             if None in (r["real_trials"], r["real_trials_training"], r["real_trials_calibration"])]
    info(f"real-trial counts null in {len(nulls)} rows: " + (", ".join(nulls) or "none"))


# --------------------------------------------------------------------------------------------------------------------
# 3. programs against experiments/audit/fig_judge.py, and its two summary numbers
# --------------------------------------------------------------------------------------------------------------------
FIG_JUDGE = "experiments/audit/fig_judge.py"
JUDGE_INPUTS = ("curves.json", "screen_drone.json", "screen_pusht.json", "screen_doom.json")


def judge_panels():
    """PANELS of fig_judge.py, read from its source: task, program versions, submitted version, and its own
    expression for the predicted value (compiled here, so the values are read exactly as that script reads them)."""
    tree = ast.parse((ROOT / FIG_JUDGE).read_text(encoding="utf-8"))
    sd, sp, sm = (load(f"runs/probe/screen_{k}.json") for k in ("drone", "pusht", "doom"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PANELS" for t in node.targets):
            panels = []
            for call in node.value.elts:
                kw = {k.arg: k.value for k in call.keywords}
                pred = eval(compile(ast.Expression(body=kw["pred"]), FIG_JUDGE, "eval"), {"sd": sd, "sp": sp, "sm": sm})
                panels.append(dict(task=ast.literal_eval(kw["task"]), vs=ast.literal_eval(kw["vs"]),
                                   shipped=ast.literal_eval(kw["shipped"]), pred=pred))
            return panels
    raise ValueError("PANELS not found in " + FIG_JUDGE)


def pearson(a, b):     # as in fig_judge.py
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((u - ma) ** 2 for u in a)
    vb = sum((u - mb) ** 2 for u in b)
    return sum((u - ma) * (w - mb) for u, w in zip(a, b)) / (va * vb) ** 0.5 if va and vb else float("nan")


def judge_lines(programs):
    """The lines fig_judge.py prints, recomputed from the table with its own statements."""
    lines, allr, allp = [], [], []
    tasks = list(dict.fromkeys(p["task"] for p in programs))
    for task in tasks:
        rows = [p for p in programs if p["task"] == task]
        vs = [int(p["program"][1:]) for p in rows]
        real, pred = [p["real"] for p in rows], [p["predicted"] for p in rows]
        allr += real
        allp += pred
        err = [abs(a - b) for a, b in zip(real, pred)]
        best_pred = vs[max(range(len(pred)), key=lambda i: pred[i])]
        best_real = vs[max(range(len(real)), key=lambda i: real[i])]
        lines.append(f'{task:8s} n={len(real)} mean|err|={sum(err) / len(err) * 100:4.1f} pts  max|err|={max(err) * 100:4.1f}  r={pearson(real, pred):.3f}  '
                     f'world picks v{best_pred} (real {real[vs.index(best_pred)] * 100:.0f}%), truly best v{best_real} ({max(real) * 100:.0f}%)')
    err = [abs(a - b) for a, b in zip(allr, allp)]
    lines.append(f"ALL      n={len(allr)} mean|err|={sum(err) / len(err) * 100:.1f} pts  within 5 pts: {sum(e <= 0.05 for e in err)}  within 10 pts: {sum(e <= 0.10 for e in err)}  "
                 f"max|err|={max(err) * 100:.1f}  r={pearson(allr, allp):.3f}")
    return lines, pearson(allr, allp), sum(err) / len(err) * 100


def check_programs(programs, out):
    check("programs: 34 rows", len(programs) == 34, f"{len(programs)} rows")
    truth = load("runs/probe/curves.json")
    expected, panels = [], judge_panels()
    for p in panels:
        for v in p["vs"]:
            t = truth[f'{p["task"]}/aspire_v{v}']
            expected.append(dict(task=p["task"], program=f"v{v}", record_name=f"aspire_v{v}", submitted=v == p["shipped"],
                                 predicted=p["pred"](v), real=t["success"], real_n=t["n"],
                                 written_after_real_trials=t["lives_before"]))
    diff = [f'{e["task"]}/{e["program"]}' for e, r in zip(expected, programs) if any(r[k] != e[k] for k in e)]
    check("programs: tasks, programs, submitted marks, predicted and real values equal fig_judge.py and its source files",
          len(expected) == len(programs) and not diff,
          f"{len(panels)} panels, {len(expected)} programs in fig_judge.py" + ("; differing: " + ", ".join(diff) if diff else ""))
    n_diff = []
    for r in programs:
        src = load(r["predicted_source"])
        key = {"drone_A": ("A",), "drone_C": ("C",), "push_T": ("T",), "push_L": ("L",), "doom": ()}[r["task"]]
        for k in key:
            src = src[k]
        if src[r["program"]]["n"] != r["predicted_n"] or not (ROOT / r["program_file"]).is_file():
            n_diff.append(f'{r["task"]}/{r["program"]}')
    check("programs: predicted_n equals the source file and program_file exists", not n_diff, ", ".join(n_diff) or "all rows")

    lines, r, mae = judge_lines(programs)
    check("programs: summary numbers recomputed from the table as fig_judge.py computes them",
          f"{r:.3f}" == "0.974" and f"{mae:.1f}" == "5.2",
          f"Pearson r = {r:.3f}, mean absolute error = {mae:.1f} percentage points, n = {len(programs)} (article: r = 0.974, 5.2 points)")
    with tempfile.TemporaryDirectory() as tmp:      # fig_judge.py itself, on the copies in raw/, in a scratch folder
        probe = Path(tmp) / "runs" / "probe"
        probe.mkdir(parents=True)
        for name in JUDGE_INPUTS:
            shutil.copy(out / "raw" / "runs" / "probe" / name, probe / name)
        run = subprocess.run([sys.executable, str(ROOT / FIG_JUDGE)], cwd=tmp, capture_output=True, text=True)
    printed = run.stdout.splitlines()
    check("programs: fig_judge.py run on raw/runs/probe prints the same lines as the recomputation from the table",
          run.returncode == 0 and printed == lines, f"{len(printed)} lines" if printed == lines else (run.stderr.strip()[-300:] or "lines differ"))
    for line in printed:
        info(line)


# --------------------------------------------------------------------------------------------------------------------
# 4. episodes against programs, results and the source files
# --------------------------------------------------------------------------------------------------------------------
def check_episodes(episodes, programs, results):
    keys = collections.Counter((e["task"], e["record_name"], e["run_in"], e["evaluation"], e["seed"]) for e in episodes)
    check("episodes: one row per task, record name, run_in, evaluation and seed", max(keys.values(), default=0) == 1,
          f"{len(episodes)} rows")
    groups = collections.defaultdict(list)
    for e in episodes:
        groups[(e["evaluation"], e["run_in"], e["task"], e["record_name"])].append(e)
    info("rows by evaluation and run_in: " + ", ".join(
        f"{k[0]} / {k[1]} {n}" for k, n in collections.Counter((e["evaluation"], e["run_in"]) for e in episodes).items()))

    bad = []
    for p in programs:
        eps = groups.get(("probe", "environment", p["task"], p["record_name"]), [])
        if not eps:
            bad.append(f'{p["task"]}/{p["program"]}: no episodes')
            continue
        # doom has no success flag: fig_judge's value is the share of episodes with a return of at least 750
        # (experiments/audit/probe.py); the other tasks record success
        m = mean(e["episode_return"] >= 750 for e in eps) if p["task"] == "doom" else mean(e["success"] for e in eps)
        if m != p["real"] or len(eps) != p["real_n"]:
            bad.append(f'{p["task"]}/{p["program"]}: episodes {m} (n={len(eps)}), programs {p["real"]} (n={p["real_n"]})')
    check("episodes: for every program the mean of its environment episodes equals `real` in programs", not bad,
          "; ".join(bad) or f"{len(programs)} programs")

    bad, n_with, worst = [], 0, 0.0
    registry = {(TASK_NAME.get(e["task"], e["task"]), e["controller"]): e for e in reproduce.REGISTRY}
    for r in results:
        name = f'{r["task"]}/{r["record_name"]}'
        eps = groups.get(("held-out", "environment", r["task"], r["record_name"]), [])
        entry = registry.get((r["task"], r["record_name"]))
        recorded = reproduce._recorded_episodes(entry) if entry else None
        if not recorded:
            if eps:
                bad.append(f"{name}: episodes without a per-episode record")
            continue
        n_with += 1
        doom = r["metric"] == "survival in game ticks"
        field, col = ("ret", "episode_return") if doom else ("success", "success")
        m = mean(float(e[col]) for e in eps) if eps else float("nan")
        worst = max(worst, abs(m - r["value"])) if eps else worst
        if not eps or abs(m - r["value"]) > 1e-9 or len(eps) != r["n_episodes"]:
            bad.append(f'{name}: episodes {m} (n={len(eps)}), results {r["value"]} (n={r["n_episodes"]})')
        elif [e["seed"] for e in eps] != list(range(r["first_seed"], r["last_seed"] + 1)):
            bad.append(f"{name}: seeds differ from results")
        elif any(float(e[col]) != float(recorded[e["seed"]][field]) or e["n_steps"] != recorded[e["seed"]]["n_steps"]
                 or float(e["episode_return"]) != float(recorded[e["seed"]]["ret"]) for e in eps):
            bad.append(f"{name}: rows differ from the per-episode record")
    check("episodes: for every held-out record with per-episode data the mean equals `value` in results", not bad,
          "; ".join(bad) or f"{n_with} of {len(results)} records have per-episode data, largest difference {worst:.1e}")
    info("held-out records without per-episode data: " +
         (", ".join(f'{r["task"]}/{r["record_name"]}' for r in results if not r["episodes_file"]) or "none"))

    bad, n_runs = [], 0
    for first in (1000, 5000):      # rope: the report of the audit holds the success rate per program
        rep = load(f"runs/annex/audit/screen/report_{first}.json")
        for row in rep["rows"]:
            for where, col in (("environment", "real"), ("world model", "world")):
                eps = [e for e in groups.get(("probe", where, "rope", row["version"]), []) if first <= e["seed"] < first + 50]
                n_runs += 1
                if len(eps) != 50 or abs(mean(e["success"] for e in eps) - row[col]) > 1e-9:
                    bad.append(f'rope/{row["version"]}@{first} {where}')
    probe = load("runs/probe/station_probe.json")      # cable: each run records its success rate
    for where, part in (("environment", "real"), ("world model", "world")):
        for key, rec in probe[part].items():
            label, first = key.rsplit("@", 1)
            eps = [e for e in groups.get(("probe", where, "cable", label), []) if int(first) <= e["seed"] < int(first) + rec["n"]]
            n_runs += 1
            if len(eps) != rec["n"] or abs(mean(e["success"] for e in eps) - rec["success"]) > 1e-9:
                bad.append(f"cable/{key} {where}")
    known = sum(len(v) for k, v in groups.items() if k[0] == "held-out" or k[2] in ("rope", "cable")) + \
        sum(len(groups.get(("probe", "environment", p["task"], p["record_name"]), [])) for p in programs)
    check("episodes: rope and cable probe runs (environment and world model) equal the success rates in their source files",
          not bad and known == len(episodes), "; ".join(bad) or f"{n_runs} runs; every row of the table belongs to a checked group: {known == len(episodes)}")

    # the summaries of those two comparisons, recomputed from the table
    def rate(run_in, task, name, first, n):
        eps = sorted((e for e in groups.get(("probe", run_in, task, name), []) if first <= e["seed"] < first + n), key=lambda e: e["seed"])
        return [e["success"] for e in eps]
    same = True
    for first in (1000, 5000):      # as experiments/audit/annex_screen.py reports it
        rep = load(f"runs/annex/audit/screen/report_{first}.json")
        runs = [(rate("environment", "rope", row["version"], first, 50), rate("world model", "rope", row["version"], first, 50)) for row in rep["rows"]]
        mae = mean(abs(mean(a) - mean(b)) for a, b in runs) * 100
        agree = sum(x == y for a, b in runs for x, y in zip(a, b)) / sum(len(a) for a, _ in runs)
        same = same and abs(mae - rep["mae_points"]) < 1e-9 and abs(agree - rep["episode_agreement"]) < 1e-12
        info(f"rope, seeds {first}-{first + 49}: {len(runs)} programs, mean |real - predicted| = {mae:.1f} points, same outcome in {100 * agree:.1f}% of episodes")
    both = [(k, rec) for k, rec in probe["world"].items() if k in probe["real"]]      # as experiments/audit/station_probe.py reports it
    pairs = [(mean(rate("environment", "cable", k.rsplit("@", 1)[0], rec["first_seed"], rec["n"])),
              mean(rate("world model", "cable", k.rsplit("@", 1)[0], rec["first_seed"], rec["n"]))) for k, rec in both]
    same = same and pairs == [(probe["real"][k]["success"], rec["success"]) for k, rec in both]
    err = [abs(a - b) for a, b in pairs]
    info(f"cable: {len(pairs)} programs run in both, mean |real - predicted| = {100 * mean(err):.1f} points, "
         f"Pearson r = {pearson([a for a, _ in pairs], [b for _, b in pairs]):.3f}")
    check("episodes: the rope and cable comparisons recomputed from the table equal the audit reports", same)


# --------------------------------------------------------------------------------------------------------------------
# 5. code against the files of the repository
# --------------------------------------------------------------------------------------------------------------------
def agent_written_files():
    out = {}
    roots = [ROOT / "worlds", ROOT / "pilots"] + sorted(ROOT.glob("experiments/*/code_wm"))
    for root in roots:
        for p in root.rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
                continue
            raw = p.read_bytes()
            if not raw and p.name == "__init__.py":
                continue
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if b"\x00" not in raw:
                out[p.relative_to(ROOT).as_posix()] = raw
    return out


def check_code(code):
    files = agent_written_files()
    paths = [r["path"] for r in code]
    check("code: one row per agent-written text file (worlds/, pilots/, experiments/*/code_wm/; empty __init__.py and binary files left out)",
          sorted(paths) == sorted(files) and len(set(paths)) == len(paths), f"{len(code)} rows, {len(files)} files")
    n_lines = lambda raw: raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0)
    bad = [r["path"] for r in code if r["path"] not in files or r["text"].encode("utf-8") != files[r["path"]]
           or r["n_bytes"] != len(files[r["path"]]) or r["n_lines"] != n_lines(files[r["path"]])]
    check("code: text, n_bytes and n_lines of every row equal the file", not bad, ", ".join(bad[:10]) or f"{len(code)} rows")
    wrong = [r["path"] for r in code
             if (r["kind"] == "world model") != bool(re.fullmatch(r"worlds/[^/]+/[^/]+\.py", r["path"]))
             or (r["kind"] == "program") != bool(re.fullmatch(r"pilots/[^/]+/policy_v\d+\.py", r["path"]))]
    check("code: kind agrees with the path (world model: worlds/*/*.py, program: pilots/*/policy_v*.py)", not wrong,
          ", ".join(wrong[:10]) or "all rows")
    kinds = list(dict.fromkeys(r["kind"] for r in code))
    tasks = list(dict.fromkeys(r["task"] for r in code))
    count = collections.Counter((r["kind"], r["task"]) for r in code)
    info("rows per kind and task:")
    info(f'{"":12s}' + "".join(f"{t:>7s}" for t in tasks) + "  total")
    for k in kinds:
        info(f"{k:12s}" + "".join(f"{count[(k, t)]:7d}" for t in tasks) + f"{sum(count[(k, t)] for t in tasks):7d}")
    info(f'{"total":12s}' + "".join(f"{sum(count[(k, t)] for k in kinds):7d}" for t in tasks) + f"{len(code):7d}")
    info("rows per agent: " + ", ".join(f"{a} {n}" for a, n in collections.Counter(r["agent"] for r in code).items()))
    worlds = [r for r in code if r["kind"] == "world model"]
    info("world models:")
    for r in worlds:
        info(f'  {r["path"]:34s} {r["n_lines"]:4d} lines {r["n_bytes"]:6d} bytes   ({r["task"]})')
    m = re.search(r"\((\d+) to (\d+) lines, (\d+) to (\d+)\s*KB\)", (ROOT / "README.md").read_text(encoding="utf-8"))
    lines = (min(r["n_lines"] for r in worlds), max(r["n_lines"] for r in worlds))
    size = (min(r["n_bytes"] for r in worlds), max(r["n_bytes"] for r in worlds))
    kb = tuple(round(b / 1000) for b in size)
    check("code: world-model sizes agree with README.md", bool(m) and tuple(map(int, m.groups())) == lines + kb,
          f'README.md states {m.group(0) if m else "nothing found"}; the table has {lines[0]} to {lines[1]} lines and '
          f"{size[0]} to {size[1]} bytes ({kb[0]} to {kb[1]} KB, 1 KB = 1000 bytes)")


# --------------------------------------------------------------------------------------------------------------------
# 6. raw copy of runs/, dataset-card header, sizes
# --------------------------------------------------------------------------------------------------------------------
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


LEFT_OUT_OF_RAW = (".pt",)                              # model checkpoints: weights in pickle format, not records
PICKLE_SUFFIXES = (".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".bin", ".joblib")


def check_raw(out):
    skip = lambda rel: rel.parts[0] == "reproduce" or rel.name.startswith("ledger.jsonl") or "__pycache__" in rel.parts
    runs = {p.relative_to(ROOT / "runs").as_posix(): p for p in (ROOT / "runs").rglob("*")
            if p.is_file() and not skip(p.relative_to(ROOT / "runs"))}
    left_out = {f: p for f, p in runs.items() if p.suffix in LEFT_OUT_OF_RAW}
    src = set(runs) - set(left_out)
    dst = {p.relative_to(out / "raw" / "runs").as_posix() for p in (out / "raw" / "runs").rglob("*") if p.is_file()}
    differ = [f for f in sorted(src & dst) if digest(ROOT / "runs" / f) != digest(out / "raw" / "runs" / f)]
    check("raw/runs: the files of runs/ with the same content, model checkpoints (*.pt) left out", src == dst and not differ,
          f"{len(dst)} files; missing {len(src - dst)}, extra {len(dst - src)}, differing {len(differ)}; left out: "
          f"{len(left_out)} model checkpoints, {sum(p.stat().st_size for p in left_out.values()):,} bytes "
          f"({len(runs)} files, {sum(p.stat().st_size for p in runs.values()):,} bytes in runs/)")
    pickles = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.suffix in PICKLE_SUFFIXES]
    empty = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_dir() and not any(p.iterdir())]
    check("the folder holds no pickle-format file and no empty directory", not pickles and not empty,
          ", ".join(pickles + empty) or "none of " + ", ".join("*" + s for s in PICKLE_SUFFIXES))


def check_license(out):
    path = out / "LICENSE"
    check("LICENSE: the repository's licence file", path.is_file() and path.read_bytes() == (ROOT / "LICENSE").read_bytes(),
          f"{path.stat().st_size} bytes, first line: {path.read_text(encoding='utf-8').splitlines()[0]!r}" if path.is_file() else "missing")


def read_header(head):
    """The YAML subset the header uses, as a dict: `key: value`; `key:` followed by `- item` lines or by blocks of
    `- key: value` and `  key: value` lines.  None if a line is of another form."""
    def scalar(s):
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
            return s[1:-1]
        return {"true": True, "false": False}.get(s, s)
    data, key = {}, None
    for line in head.splitlines():
        if m := re.fullmatch(r"([A-Za-z_]+): (\S.*)", line):
            data[m[1]], key = scalar(m[2]), None
        elif m := re.fullmatch(r"([A-Za-z_]+):", line):
            key = m[1]
            data[key] = []
        elif key and (m := re.fullmatch(r"- ([A-Za-z_]+): (\S.*)", line)):
            data[key].append({m[1]: scalar(m[2])})
        elif key and data[key] and isinstance(data[key][-1], dict) and (m := re.fullmatch(r"  ([A-Za-z_]+): (\S.*)", line)):
            data[key][-1][m[1]] = scalar(m[2])
        elif key and (m := re.fullmatch(r"- (\S.*)", line)):
            data[key].append(scalar(m[1]))
        else:
            return None
    return data


def check_card(out, tables):
    text = (out / "README.md").read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?\n)---\n", text, re.S)
    head, rest = (m.group(1), text[m.end():]) if m else ("", text)
    data = read_header(head) if m else None
    try:
        import yaml
        parsed = yaml.safe_load(head) if m else None
        parser = "also parsed with PyYAML: " + ("same content" if parsed == data else "DIFFERENT content")
    except ImportError:
        parsed, parser = data, "PyYAML is not installed in this interpreter: read with the built-in reader only"
    except Exception as e:      # noqa: BLE001   (a YAML error)
        parsed, parser = None, f"PyYAML: {e!r}"
    total = sum(len(t) for t in tables.values())
    bounds = {"n<1K": (0, 10**3), "1K<n<10K": (10**3, 10**4), "10K<n<100K": (10**4, 10**5), "100K<n<1M": (10**5, 10**6)}
    configs = (data or {}).get("configs") or []
    size = ((data or {}).get("size_categories") or [None])[0]
    lo, hi = bounds.get(size, (1, 0))
    ok = (bool(data) and parsed == data and data.get("license") == "mit" and data.get("language") == ["en"]
          and data.get("pretty_name") == "World Models as Unit Tests" and lo <= total < hi
          and configs == [dict(config_name=n, data_files=f"data/{n}.jsonl", **({"default": True} if n == "results" else {}))
                          for n in TABLES]
          and all((out / c["data_files"]).is_file() for c in configs))
    check("README.md: starts with a valid YAML block (license, language, pretty_name, one config per table with an existing file, results as the default)",
          ok, f'{len(head.splitlines())} lines, keys {list(data or {})}, configs {[c.get("config_name") for c in configs]}, '
              f"size_categories {size} for {total} rows in all tables; {parser}")
    body = ROOT / "dataset" / "card_body.md"
    if body.is_file():
        b = body.read_text(encoding="utf-8")
        check("README.md: after the header, a blank line and dataset/card_body.md", rest == "\n" + b + ("" if b.endswith("\n") else "\n"),
              f"{len(rest.encode('utf-8'))} bytes after the header, card_body.md has {len(b.encode('utf-8'))} bytes")
    else:
        check("README.md: nothing after the header (no dataset/card_body.md)", rest == "", f"{len(rest)} characters after the header")


def check_sizes(out):
    files = [p for p in out.rglob("*") if p.is_file()]
    raw = [p for p in files if "raw" in p.relative_to(out).parts[:1]]
    info("sizes: " + ", ".join(f'data/{n}.jsonl {(out / "data" / f"{n}.jsonl").stat().st_size:,} bytes' for n in TABLES)
         + "".join(f", {n} {(out / n).stat().st_size:,} bytes" for n in ("README.md", "LICENSE") if (out / n).is_file()))
    info(f"sizes: raw/runs {sum(p.stat().st_size for p in raw):,} bytes in {len(raw)} files; "
         f"whole folder {sum(p.stat().st_size for p in files):,} bytes in {len(files)} files")


# --------------------------------------------------------------------------------------------------------------------
# 7. the tables load with the libraries behind the Dataset Viewer, where they are installed
# --------------------------------------------------------------------------------------------------------------------
ARROW_TYPE = {"string": "string", "int": "int64", "float": "float64", "bool": "bool", "timestamp": "timestamp[s]"}


def short(v, n=70):
    return v if not isinstance(v, str) or len(v) <= n else v[:n] + f"... ({len(v)} characters)"


def as_written(row):
    """A loaded row with its timestamps back in the form of the file."""
    return {c: v.strftime("%Y-%m-%d %H:%M:%S") if hasattr(v, "strftime") else v for c, v in row.items()}


def check_loading(out, tables):
    """Load every config from the folder with the datasets library (the reader behind the Dataset Viewer), when it
    is installed: one split, the row count of the file, the column types of the JSON Lines file, the same first row."""
    try:
        import datasets
    except ImportError:
        info("the datasets library is not installed in this interpreter: load_dataset test not run")
        return
    info(f"datasets {datasets.__version__}, pyarrow {__import__('pyarrow').__version__}")
    try:
        names = datasets.get_dataset_config_names(str(out))
        check("datasets finds the four configs in README.md", sorted(names) == sorted(TABLES), f"{names}")
    except Exception as e:      # noqa: BLE001
        check("datasets finds the four configs in README.md", False, repr(e)[:600])
    with tempfile.TemporaryDirectory() as cache:
        for name in TABLES:
            rows = tables[name]
            want = {c: ARROW_TYPE[next(t for t in (type_name(r[c]) for r in rows) if t != "null")] for c in rows[0]}
            try:
                loaded = datasets.load_dataset(str(out), name, cache_dir=cache)
                ds = loaded["train"]
                got = {c: getattr(f, "dtype", str(f)) for c, f in ds.features.items()}
                unexpected = {c: f"{got.get(c)} (expected {t})" for c, t in want.items() if got.get(c) != t}
                first = as_written(ds[0])
                same_rows = ds.num_rows == len(rows) and all(as_written(ds[i]) == rows[i] for i in range(ds.num_rows))
                nulls = {c: sum(v is None for v in ds[c]) for c in ds.column_names}
                same_nulls = nulls == {c: sum(r[c] is None for r in rows) for c in rows[0]}
                check(f"datasets.load_dataset(<folder>, {name!r})",
                      list(loaded) == ["train"] and list(got) == list(want) and not unexpected and same_rows and same_nulls,
                      f"splits {list(loaded)}, {ds.num_rows} rows, {len(got)} columns; unexpected types: {unexpected or 'none'}; "
                      f"every row equals the file: {same_rows}; nulls per column equal the file: {same_nulls}")
                info("features: " + ", ".join(f"{c}:{t}" for c, t in got.items()))
                info("first row: " + json.dumps({c: short(v) for c, v in first.items()}, ensure_ascii=False, default=str))
            except Exception as e:      # noqa: BLE001
                check(f"datasets.load_dataset(<folder>, {name!r})", False, repr(e)[:600])
        try:
            default = datasets.load_dataset(str(out), cache_dir=cache)["train"]
            check("datasets.load_dataset(<folder>) without a config name loads results", default.num_rows == len(tables["results"])
                  and default.column_names == list(tables["results"][0]), f"{default.num_rows} rows, {len(default.column_names)} columns")
        except Exception as e:      # noqa: BLE001
            check("datasets.load_dataset(<folder>) without a config name loads results", False, repr(e)[:600])


# --------------------------------------------------------------------------------------------------------------------
# 8. public names in column names and labels
# --------------------------------------------------------------------------------------------------------------------
INTERNAL_WORDS = r"\b(lives?|pilots?|labs?|code[ _]worlds?|official|screen(ing|ed)?|aspire\w*)\b"
LABEL_COLUMNS = {"results": ("task", "agent", "metric", "trained_in"), "programs": ("task", "agent", "metric"),
                 "episodes": ("task", "agent", "run_in", "evaluation"), "code": ("task", "agent", "kind")}


def check_names(tables):
    hits = []
    for name, rows in tables.items():
        words = list(rows[0]) if rows else []
        for col in LABEL_COLUMNS[name]:
            values = list(dict.fromkeys(r[col] for r in rows))
            info(f"{name}.{col}: " + " | ".join(map(str, values)))
            words += values
        hits += [f"{name}: {w}" for w in words if re.search(INTERNAL_WORDS, str(w), re.I)]
    check("column names and labels use the public names", not hits, "; ".join(hits) or "no repository-internal word found")


# --------------------------------------------------------------------------------------------------------------------
# 9. private strings in the whole folder (text files, and the content of .gz, .npz and other zip archives)
# --------------------------------------------------------------------------------------------------------------------
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".html", ".py", ".csv", ".yaml", ".yml"}
MACHINE_PATH = re.compile(rb"/(?:home|Users|root|scratch|mnt|media|nfs|srv|Volumes)/[A-Za-z0-9._-]+")
# an address, not the matrix product `a@b.T` or `a @ b.mean()` of the code files
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![A-Za-z0-9_(\[.])")


def payloads(path, undecoded=None):
    """(label, bytes, is text) for a file and, for archives, for each member.  An array of an .npz file that
    cannot be decoded is still scanned as bytes, and its name is added to `undecoded`."""
    raw = path.read_bytes()
    if path.suffix == ".gz":
        yield str(path), gzip.decompress(raw), True
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                data = z.read(member)
                yield f"{path}!{member}", member.encode() + b"\n" + data, False
                if member.endswith(".npy"):         # text arrays are stored as UTF-32: decode them
                    try:
                        a = np.load(io.BytesIO(data), allow_pickle=False)
                    except ValueError:
                        if undecoded is not None:
                            undecoded.append(f"{path.name}!{member}")
                        continue
                    if a.dtype.kind in "US":
                        yield f"{path}!{member}", " ".join(map(str, a.ravel().tolist())).encode("utf-8"), True
    else:
        yield str(path), raw, path.suffix in TEXT_SUFFIXES


STATED_IN = ("LICENSE", "README.md")                    # the files that state the author, the account and the institution


def term_pattern(terms):
    terms = sorted((t for t in dict.fromkeys(x.strip().lower() for x in terms) if len(t) >= 3), key=len, reverse=True)
    return (re.compile(b"|".join(re.escape(t.encode()) for t in terms), re.I) if terms else None), len(terms)


def check_private(out, private_terms, public_names):
    """What a pass means: none of the patterns below occurs, and a public name occurs only in LICENSE and README.md
    (those occurrences are listed, for review).  A name that is neither the user or host of this machine nor given
    on the command line is not looked for."""
    private_re, n_private = term_pattern(private_terms)
    public_re, n_public = term_pattern(public_names)
    hits, stated, n_files, n_payloads, undecoded = collections.Counter(), collections.Counter(), 0, collections.Counter(), []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        n_files += 1
        rel = path.relative_to(out).as_posix()
        for label, data, is_text in [(str(path) + " (file name)", rel.encode("utf-8"), True), *payloads(path, undecoded)]:
            where = rel + label[len(str(path)):]
            n_payloads["text" if is_text else "binary"] += 1
            context = lambda m: data[max(0, m.start() - 30):m.end() + 30].decode("latin-1").replace("\n", " ")
            for m in MACHINE_PATH.finditer(data):
                hits[(where, "home-directory or mount path", m.group(0).decode("latin-1"))] += 1
            if is_text:
                for m in EMAIL.finditer(data):
                    hits[(where, "e-mail address", m.group(0).decode("latin-1"))] += 1
            for m in private_re.finditer(data) if private_re else ():
                hits[(where, "private term", context(m))] += 1
            for m in public_re.finditer(data) if public_re else ():
                if rel in STATED_IN:
                    stated[(rel, context(m))] += 1
                else:
                    hits[(where, "public name outside LICENSE and README.md", context(m))] += 1
    check("private-string scan of the whole folder", not hits,
          f"{sum(hits.values())} hits in {len({h[0] for h in hits})} places; scanned {n_files} files as "
          f'{n_payloads["text"]} text and {n_payloads["binary"]} binary parts (file names, file contents, unpacked .gz, '
          f"members of .npz archives, text arrays decoded; {len(undecoded)} arrays not decoded) for home-directory and "
          f"mount paths, e-mail addresses (text parts only), {n_private} user, host and private terms, and {n_public} "
          f"public names; {sum(stated.values())} occurrences of public names in {' and '.join(sorted({s[0] for s in stated})) or 'no file'}"
          + (" (listed below, not counted as hits)" if stated else ""))
    for name in undecoded[:20]:
        info(f"not decoded, scanned as bytes: {name}")
    for (where, kind, text), n in list(hits.items())[:60]:
        info(f"{kind} in {where} ({n}x): {text!r}")
    if len(hits) > 60:
        info(f"... {len(hits) - 60} more")
    for (rel, text), n in stated.items():
        info(f"public name in {rel} ({n}x): {text!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--private-terms", default="", help="comma-separated strings that must occur nowhere in the folder")
    ap.add_argument("--public-names", default="", help="comma-separated names the dataset states on purpose (author, "
                    "account, institution): they may occur in LICENSE and README.md only, and are listed")
    a = ap.parse_args()
    out = Path(a.out_dir).expanduser().resolve()
    tables = check_tables(out)
    if all(tables.values()):
        check_results(tables["results"])
        check_programs(tables["programs"], out)
        check_episodes(tables["episodes"], tables["programs"], tables["results"])
        check_code(tables["code"])
        check_names(tables)
        check_loading(out, tables)
    check_raw(out)
    check_license(out)
    check_card(out, tables)
    host = socket.gethostname()
    check_private(out, [getpass.getuser(), Path.home().name, host, host.split(".")[0]] + a.private_terms.split(","),
                  a.public_names.split(","))
    check_sizes(out)
    print("PASS" if not FAILED else f"FAIL ({len(FAILED)} checks failed)")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
