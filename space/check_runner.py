#!/usr/bin/env python
"""Check that space/runner.py reports the same per-episode outcomes as the repository's own evaluations.

    python space/check_runner.py

world model   run_world on seeds 50000-50024 against experiments.pusht.code_wm.screen._run_chunk on the same seeds
              (T v4, T v5, L v10, L v17), and the same batch run as two halves (how the Space splits a batch over
              its two worker processes).
environment   run_real on seeds 1000-1049 against the recorded evaluations of Push-T v5 (pusht_T aspire_style) and
              Push-L v17 (pusht_L cap_in_code_world).  These 100 episodes are metered in runs/ledger.jsonl
              (purpose "eval", tag "space_check").
Also checks the recorded trajectory of each first episode: one row per step plus the initial state, and a final
block pose whose coverage is on the reported side of the success threshold.
Single process, about one minute.  Exit status 1 if anything differs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

WORLD = [("T", 4), ("T", 5), ("L", 10), ("L", 17)]
WORLD_SEED0, WORLD_N = 50000, 25
REAL = [("T", 5, "runs/pusht/aspire_style/eval_aspire_style/episodes.json"),
        ("L", 17, "runs/pusht/aspire_style/eval_L_cap_in_code_world/episodes.json")]
REAL_SEED0, REAL_N = 1000, 50


def _trajectory_ok(res):
    """The first episode's trajectory has n_steps + 1 rows and ends on the reported side of the threshold."""
    from reallab.pusht import api
    f = res["first"]
    rows = f["trajectory"]
    cov = api.coverage_of(tuple(rows[-1][2:]), object=res["task"])
    # rows are rounded (0.1 u, 0.001 rad), so allow 0.01 of coverage around the 0.95 threshold
    side = cov > 0.94 if f["success"] else cov < 0.96
    return len(rows) == f["n_steps"] + 1 and f["success"] == res["success"][0] and side, cov


def main():
    from experiments.pusht.code_wm import screen
    from space import runner
    ok = True
    for obj, v in WORLD:
        t0 = time.time()
        seeds = list(range(WORLD_SEED0, WORLD_SEED0 + WORLD_N))
        ref = screen._run_chunk((runner.factory(obj, v), {}, obj, seeds))
        res = runner.run_world(obj, v, WORLD_SEED0, WORLD_N)
        same = sum(a == e["success"] and b == e["n_steps"] for a, b, e in zip(res["success"], res["n_steps"], ref))
        h = WORLD_N // 2
        a = runner.run_world(obj, v, WORLD_SEED0, h)
        b = runner.run_world(obj, v, WORLD_SEED0 + h, WORLD_N - h, record=False)
        split = sum(x == y for x, y in zip(a["success"] + b["success"], res["success"]))
        tr, cov = _trajectory_ok(res)
        good = same == WORLD_N and split == WORLD_N and tr and len(res["success"]) == WORLD_N
        ok &= good
        print(f"world {obj}/v{v:<2d} seeds {seeds[0]}-{seeds[-1]}: {sum(res['success'])}/{WORLD_N} succeed, "
              f"_run_chunk {sum(e['success'] for e in ref)}/{WORLD_N}, identical episodes {same}/{WORLD_N}, "
              f"as two halves {split}/{WORLD_N}, first-episode trajectory {len(res['first']['trajectory'])} rows "
              f"(final coverage {cov:.3f}) {'ok' if tr else 'WRONG'}  [{'PASS' if good else 'FAIL'}, "
              f"{time.time() - t0:.0f} s]", flush=True)
    for obj, v, record in REAL:
        t0 = time.time()
        rec = {int(e["seed"]): e for e in json.loads((ROOT / record).read_text())}
        seeds = list(range(REAL_SEED0, REAL_SEED0 + REAL_N))
        res = runner.run_real(obj, v, REAL_SEED0, REAL_N, tag="space_check")
        same = sum(a == rec[s]["success"] and b == rec[s]["n_steps"]
                   for s, a, b in zip(seeds, res["success"], res["n_steps"]))
        tr, cov = _trajectory_ok(res)
        good = same == REAL_N and tr and len(res["success"]) == REAL_N
        ok &= good
        print(f"real  {obj}/v{v:<2d} seeds {seeds[0]}-{seeds[-1]}: {sum(res['success'])}/{REAL_N} succeed, "
              f"recorded {sum(rec[s]['success'] for s in seeds)}/{REAL_N}, identical episodes {same}/{REAL_N}, "
              f"first-episode trajectory {len(res['first']['trajectory'])} rows (final coverage {cov:.3f}) "
              f"{'ok' if tr else 'WRONG'}  [{'PASS' if good else 'FAIL'}, {time.time() - t0:.0f} s]", flush=True)
    print("check_runner:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
