#!/usr/bin/env python
"""Record every pushing program once, for the static variant of the page (space/build_static.py).

    python space/make_data.py [--workers 12] [--out space/data]

For each of the 17 programs (T: v1-v8, L: v9-v17), run by space/runner.py with every trajectory recorded:
  world model   200 episodes, seeds 50000-50199, in batches of 25 consecutive seeds: the seeds and batches of the
                recorded screening runs/probe/screen_pusht.json (experiments/pusht/code_wm/screen.py)
  environment   50 episodes, seeds 6000-6049, purpose "eval": the seeds and purpose of the recorded runs in
                runs/probe/curves.json (experiments/audit/probe.py).  These 850 episodes are metered in
                runs/ledger.jsonl under the tag "space_data".

Writes
  <out>/index.json         catalog: tasks, programs (tag, file, episodes n and successes k per side), geometry per
                           task and side, the default selection, seeds, step limit, the encoding parameters
  <out>/<task>_v<k>.json   one program; per side: seed0, n, k, success (one "1" or "0" per episode), steps (per
                           episode) and traj (one encoded string per episode)

Trajectory encoding.  Every step is kept; the only loss is quantisation.  A trajectory is the rows
(pusher x, pusher y, block x, block y, block theta) of space/runner.py, one for the initial state and one per step.
Positions become integers in units of UNIT u, theta an integer in units of 2 pi / TURN.  The first row is written
as it is and every later row as its difference from the row before (for theta the shorter way round, so decoders
add up and reduce modulo TURN).  Each integer v is mapped to 2 v (v >= 0) or -2 v - 1 (v < 0) and written in
base 32, least significant digit first, with 32 added to every digit but the last; a digit is one character of
ALPHABET.  decode() below is the reference decoder; the page has the same in JavaScript.
space/check_data.py checks the result against the records and against fresh runs.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from space import runner  # noqa: E402

WORLD = dict(seed0=50000, n=200, batch=25)
REAL = dict(seed0=6000, n=50, batch=25)
LEDGER_TAG = "space_data"
MAX_WORKERS = 12

UNIT = 0.25             # position resolution [u]: decoded positions are within UNIT / 2 of the recorded ones
TURN = 4096             # angle resolution 2 pi / TURN rad: decoded angles are within pi / TURN of the recorded ones
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def program_file(obj: str, version: int) -> str:
    return f"{obj}_v{version}.json"


def quantise(row) -> tuple:
    return (round(row[0] / UNIT), round(row[1] / UNIT), round(row[2] / UNIT), round(row[3] / UNIT),
            round(row[4] / (2 * math.pi) * TURN) % TURN)


def encode(rows) -> str:
    out, prev = [], (0, 0, 0, 0, 0)
    for row in rows:
        q = quantise(row)
        turn = (q[4] - prev[4] + TURN // 2) % TURN - TURN // 2
        for v in (q[0] - prev[0], q[1] - prev[1], q[2] - prev[2], q[3] - prev[3], turn):
            z = 2 * v if v >= 0 else -2 * v - 1
            while z >= 32:
                out.append(ALPHABET[32 + (z & 31)])
                z >>= 5
            out.append(ALPHABET[z])
        prev = q
    return "".join(out)


def decode(text: str, unit: float = UNIT, turn: int = TURN, alphabet: str = ALPHABET) -> list:
    """Encoded string -> rows [pusher x, pusher y, block x, block y, block theta] in u and rad."""
    index = {c: i for i, c in enumerate(alphabet)}
    rows, cur, z, shift, j = [], [0, 0, 0, 0, 0], 0, 0, 0
    for c in text:
        d = index[c]
        z |= (d & 31) << shift
        shift += 5
        if d & 32:
            continue
        cur[j] += -(z + 1) // 2 if z & 1 else z // 2
        z, shift, j = 0, 0, j + 1
        if j == 5:
            j = 0
            rows.append([cur[0] * unit, cur[1] * unit, cur[2] * unit, cur[3] * unit,
                         cur[4] % turn * 2 * math.pi / turn])
    if j or shift:
        raise ValueError("truncated trajectory")
    return rows


def jobs() -> list:
    out = []
    for obj, versions in runner.PROGRAMS.items():
        for v in versions:
            for side, spec in (("world", WORLD), ("real", REAL)):
                for s in range(spec["seed0"], spec["seed0"] + spec["n"], spec["batch"]):
                    out.append((side, obj, v, s, min(spec["batch"], spec["seed0"] + spec["n"] - s)))
    return out


def run_job(job):
    """One batch with all trajectories (worker process)."""
    side, obj, v, seed0, n = job
    if side == "world":
        res = runner.run_world(obj, v, seed0, n, record="all")
    else:
        res = runner.run_real(obj, v, seed0, n, record="all", tag=LEDGER_TAG)
    return job, res


def pool(workers: int):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    return mp.get_context("spawn").Pool(max(1, min(MAX_WORKERS, workers)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"worker processes (at most {MAX_WORKERS})")
    ap.add_argument("--out", default=str(ROOT / "space" / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    todo = jobs()
    parts = {}
    with pool(a.workers) as p:
        for job, res in p.imap_unordered(run_job, todo, chunksize=1):
            parts[job] = res
    print(f"{len(todo)} batches, {sum(j[4] for j in todo)} episodes, {min(MAX_WORKERS, a.workers)} workers, "
          f"{time.time() - t0:.0f} s", flush=True)

    programs, sizes = [], {}
    for obj, versions in runner.PROGRAMS.items():
        for v in versions:
            record = dict(task=obj, version=v)
            for side, spec in (("world", WORLD), ("real", REAL)):
                mine = [parts[j] for j in todo if j[:3] == (side, obj, v)]        # in seed order
                success = [s for r in mine for s in r["success"]]
                record[side] = dict(seed0=spec["seed0"], n=len(success), k=sum(success),
                                    success="".join("1" if s else "0" for s in success),
                                    steps=[s for r in mine for s in r["n_steps"]],
                                    traj=[encode(t) for r in mine for t in r["trajectories"]])
            name = program_file(obj, v)
            text = json.dumps(record, separators=(",", ":"))
            (out / name).write_text(text + "\n")
            sizes[name] = len(text) + 1
            programs.append(dict(task=obj, version=v, tag=runner.TAGS.get((obj, v)), file=name,
                                 world=dict(n=record["world"]["n"], k=record["world"]["k"]),
                                 real=dict(n=record["real"]["n"], k=record["real"]["k"])))
            print(f"{obj}/v{v:<2d} world {record['world']['k']:3d}/{record['world']['n']}  "
                  f"environment {record['real']['k']:2d}/{record['real']['n']}  {name} {sizes[name] / 1000:.0f} kB",
                  flush=True)
    from reallab.pusht import api
    time_limit = next(iter(parts.values()))["time_limit"]
    index = dict(default=runner.DEFAULT, tasks=list(runner.PROGRAMS), programs=programs,
                 geometry={obj: dict(world=runner.world_geometry(obj), real=runner.real_geometry(obj))
                           for obj in runner.PROGRAMS},
                 world=dict(seed0=WORLD["seed0"], n=WORLD["n"]), real=dict(seed0=REAL["seed0"], n=REAL["n"]),
                 time_limit=time_limit, steps_per_second=int(api.task_card()["control_hz"]),
                 encoding=dict(unit=UNIT, turn=TURN, alphabet=ALPHABET))
    text = json.dumps(index, indent=1)
    (out / "index.json").write_text(text + "\n")
    sizes["index.json"] = len(text) + 1
    print(f"wrote {len(sizes)} files to {out}: {sum(sizes.values()) / 1e6:.2f} MB in total, largest program file "
          f"{max(v for k, v in sizes.items() if k != 'index.json') / 1000:.0f} kB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
