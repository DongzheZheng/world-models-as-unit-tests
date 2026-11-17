#!/usr/bin/env python
"""Check space/data (written by space/make_data.py) against the records and against fresh runs.

    python space/check_data.py [--workers 12] [--data space/data]

For each of the 17 programs:
  records      v1-v15: the world-model successes and mean steps equal runs/probe/screen_pusht.json (200 episodes),
               and the environment outcomes equal runs/probe/curves.json seed by seed (6000-6049)
  fresh runs   every world-model batch is run again by experiments.pusht.code_wm.screen._run_chunk itself (same
               seeds, same batches of 25), and every batch of both sides by space/runner.py: success and steps of
               every episode must equal the data, and every decoded trajectory must be within unit / 2 (positions)
               and pi / turn (angle) of the unencoded one at every step
  consistency  a trajectory has steps + 1 rows; the coverage recomputed from its decoded final block pose is on
               the outcome's side of 0.95, within COVERAGE_MARGIN (the effect of the quantisation); the row before
               a success is not yet a success; a failure runs to the step limit
  sizes        each program file at most 700 kB, the data folder at most 12 MB
The fresh environment runs are metered in runs/ledger.jsonl (purpose "eval", tag "space_check").
Exit status 1 if anything differs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from space import make_data, runner  # noqa: E402

COVERAGE_MARGIN = 0.005
MAX_FILE_BYTES, MAX_TOTAL_BYTES = 700_000, 12_000_000


def check_batch(args) -> dict:
    """Run one batch again and compare it with its part of the program file (worker process)."""
    (side, obj, v, seed0, n), data_dir = args
    from reallab.pusht import api
    index = json.loads((Path(data_dir) / "index.json").read_text())
    enc = index["encoding"]
    rec = json.loads((Path(data_dir) / make_data.program_file(obj, v)).read_text())[side]
    lo = seed0 - rec["seed0"]
    success = [c == "1" for c in rec["success"][lo:lo + n]]
    steps, traj = rec["steps"][lo:lo + n], rec["traj"][lo:lo + n]
    out = dict(n=n, chunk=None, runner=0, rows=0, final=0, pos=0.0, ang=0.0)
    if side == "world":
        from experiments.pusht.code_wm import screen
        ref = screen._run_chunk((runner.factory(obj, v), {}, obj, list(range(seed0, seed0 + n))))
        out["chunk"] = sum(e["success"] == s and e["n_steps"] == t for e, s, t in zip(ref, success, steps))
        res = runner.run_world(obj, v, seed0, n, record="all")
    else:
        res = runner.run_real(obj, v, seed0, n, record="all", tag="space_check")
    for i in range(n):
        out["runner"] += res["success"][i] == success[i] and res["n_steps"][i] == steps[i]
        rows = make_data.decode(traj[i], enc["unit"], enc["turn"], enc["alphabet"])
        raw = res["trajectories"][i]
        if len(rows) != steps[i] + 1 or len(rows) != len(raw):
            continue
        out["rows"] += 1
        for a, b in zip(raw, rows):
            out["pos"] = max(out["pos"], abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]), abs(a[3] - b[3]))
            d = abs(a[4] - b[4])
            out["ang"] = max(out["ang"], min(d, 2 * math.pi - d))
        cover = api.coverage_of(tuple(rows[-1][2:]), obj)
        if success[i]:
            before = api.coverage_of(tuple(rows[-2][2:]), obj) if len(rows) > 1 else 0.0
            ok = cover > api.SUCCESS_COVERAGE - COVERAGE_MARGIN and before <= api.SUCCESS_COVERAGE + COVERAGE_MARGIN
        else:
            ok = cover <= api.SUCCESS_COVERAGE + COVERAGE_MARGIN and steps[i] == index["time_limit"]
        out["final"] += ok
    return (side, obj, v), out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=make_data.MAX_WORKERS)
    ap.add_argument("--data", default=str(ROOT / "space" / "data"))
    a = ap.parse_args()
    data = Path(a.data)
    t0 = time.time()
    index = json.loads((data / "index.json").read_text())
    enc = index["encoding"]
    pos_tol, ang_tol = enc["unit"] / 2 + 1e-9, math.pi / enc["turn"] + 1e-9
    screen_rec = json.loads((ROOT / "runs/probe/screen_pusht.json").read_text())
    curves = json.loads((ROOT / "runs/probe/curves.json").read_text())

    sums = {}
    with make_data.pool(a.workers) as p:
        for key, part in p.imap_unordered(check_batch, [(j, str(data)) for j in make_data.jobs()], chunksize=1):
            s = sums.setdefault(key, dict(n=0, chunk=0, runner=0, rows=0, final=0, pos=0.0, ang=0.0))
            for k in ("n", "runner", "rows", "final"):
                s[k] += part[k]
            s["chunk"] += part["chunk"] or 0
            s["pos"], s["ang"] = max(s["pos"], part["pos"]), max(s["ang"], part["ang"])
    print(f"fresh runs of {sum(s['n'] for s in sums.values())} episodes with {min(make_data.MAX_WORKERS, a.workers)} "
          f"workers: {time.time() - t0:.0f} s", flush=True)

    ok, total = True, 0
    listed = {(p["task"], p["version"]): p for p in index["programs"]}
    for obj, versions in runner.PROGRAMS.items():
        for v in versions:
            path = data / make_data.program_file(obj, v)
            rec, size = json.loads(path.read_text()), path.stat().st_size
            total += size
            w, r, sw, sr = rec["world"], rec["real"], sums[("world", obj, v)], sums[("real", obj, v)]
            good = all(side["n"] == len(side["success"]) == len(side["steps"]) == len(side["traj"])
                       and side["k"] == side["success"].count("1") for side in (w, r))
            good &= (w["seed0"], w["n"], r["seed0"], r["n"]) == (make_data.WORLD["seed0"], make_data.WORLD["n"],
                                                               make_data.REAL["seed0"], make_data.REAL["n"])
            entry = listed.get((obj, v), {})
            good &= entry.get("file") == path.name and entry.get("world") == dict(n=w["n"], k=w["k"]) \
                and entry.get("real") == dict(n=r["n"], k=r["k"]) and entry.get("tag") == runner.TAGS.get((obj, v))
            sw_rec, sr_rec = screen_rec[obj].get(f"v{v}"), curves.get(f"push_{obj}/aspire_v{v}")
            if sw_rec:
                mean = sum(w["steps"]) / w["n"]
                same = sw_rec["n"] == w["n"] and round(sw_rec["success"] * sw_rec["n"]) == w["k"] \
                    and abs(sw_rec["n_steps"] - mean) < 1e-6
                good &= same
                world_txt = (f"screen_pusht.json {round(sw_rec['success'] * sw_rec['n'])}/{sw_rec['n']}, mean steps "
                             f"{sw_rec['n_steps']:.2f} vs {mean:.2f} {'equal' if same else 'DIFFERENT'}")
            else:
                world_txt = "no earlier record"
            if sr_rec:
                seeds = range(r["seed0"], r["seed0"] + r["n"])
                same = sum(sr_rec["per_seed"].get(str(s)) == (c == "1") for s, c in zip(seeds, r["success"]))
                good &= same == r["n"] == sr_rec["n"]
                real_txt = f"curves.json identical seeds {same}/{r['n']}"
            else:
                real_txt = "no earlier record"
            n = w["n"] + r["n"]
            good &= sw["chunk"] == w["n"] and sw["runner"] + sr["runner"] == n and sw["rows"] + sr["rows"] == n
            good &= sw["final"] + sr["final"] == n
            pos, ang = max(sw["pos"], sr["pos"]), max(sw["ang"], sr["ang"])
            good &= pos <= pos_tol and ang <= ang_tol and size <= MAX_FILE_BYTES
            ok &= good
            print(f"{obj}/v{v:<2d} world {w['k']}/{w['n']}: {world_txt}, _run_chunk identical episodes "
                  f"{sw['chunk']}/{w['n']} | environment {r['k']}/{r['n']}: {real_txt} | trajectories: fresh run "
                  f"identical {sw['runner'] + sr['runner']}/{n}, max error {pos:.3f} u {ang:.5f} rad, final state "
                  f"consistent {sw['final'] + sr['final']}/{n} | {size / 1000:.0f} kB  [{'PASS' if good else 'FAIL'}]",
                  flush=True)
    total += (data / "index.json").stat().st_size
    extra = sorted(p.name for p in data.iterdir() if p.name != "index.json"
                   and p.name not in {q["file"] for q in index["programs"]})
    sized = total <= MAX_TOTAL_BYTES and not extra and len(index["programs"]) == 17
    ok &= sized
    print(f"tolerances: positions {pos_tol:.3f} u, angle {ang_tol:.5f} rad, coverage margin {COVERAGE_MARGIN}; "
          f"data folder {total / 1e6:.2f} MB in {len(index['programs']) + 1} files"
          f"{' (unexpected files: ' + ', '.join(extra) + ')' if extra else ''}  [{'PASS' if sized else 'FAIL'}]")
    print(f"check_data: {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.0f} s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
