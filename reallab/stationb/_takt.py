"""Lab-side: choose the takt time T_B (internal).

Every reference runs once per seed with a long horizon (H steps, lab override); because neither reference looks at the
clock, the outcome under a time limit T is: success iff it succeeded within T steps (an overload after T never
happens in the limited episode; an overload before T is a failure either way).
    python -m reallab.stationb._takt run      -> runs/stationb/_lab/takt_runs.json   (purpose smoke, seeds 3000-3049)
    python -m reallab.stationb._takt table    -> prints the success-vs-limit table and the T_B decision
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("runs/stationb/_lab/takt_runs.json")
SEEDS = list(range(3000, 3050))
H = 1200
GOV = [dict(clear=0.05, vmax=vm, lo=lo, hi=hi, back_at=9.0) for vm in (0.3, 0.2) for lo, hi in ((3, 8), (2, 6), (4, 9))]
GEO = [dict(clear=0.05, K=K, a=a, vmin=0.04, vmax=vm) for a in (0.5, 1.0) for K in (0.15, 0.3, 0.6) for vm in (0.2, 0.3)]
GEO += [dict(clear=0.06, K=K, a=a, vmin=vmin, vmax=0.3) for K in (0.15, 0.25) for a in (0.3, 0.6) for vmin in (0.07, 0.1)]
UNI = [dict(clear=c, speed=v) for c in (0.045, 0.05, 0.06, 0.08) for v in (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)]


def outcome(r):
    if r["success"]:
        return "success"
    if r["overload"]:
        return "overload"
    if r["sides_ok"] < 3:
        return "wrong_side"
    return "time_out"


def run():
    from reallab.stationb._reference import lab_evaluate
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    grids = (("governor", GOV), ("uniform", UNI), ("geometric", GEO))
    if len(sys.argv) > 2:
        grids = tuple(g for g in grids if g[0] in sys.argv[2:])
    for name, grid in grids:
        for kw in grid:
            key = f"{name} {json.dumps(kw, sort_keys=True)}"
            if key in cur:
                continue
            t0 = time.time()
            res = lab_evaluate(f"reallab.stationb._reference:{name}", kw, SEEDS, purpose="smoke", tag=f"takt_{name}",
                               n_workers=12, time_limit=H)
            cur[key] = [dict(seed=r["seed"], success=r["success"], n_steps=r["n_steps"], outcome=outcome(r),
                             max_load=round(r["max_load"], 3)) for r in res]
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(cur))
            s = np.mean([r["success"] for r in res])
            print(f"{key}: success@{H} {s:.2f}  {time.time() - t0:.0f}s", flush=True)


def succ_at(rows, T):
    return float(np.mean([r["success"] and r["n_steps"] <= T for r in rows]))


def table():
    cur = json.loads(OUT.read_text())
    gov = {k: v for k, v in cur.items() if k.startswith("governor")}
    uni = {k: v for k, v in cur.items() if k.startswith("uniform")}
    Ts = list(range(10, H + 1, 10))
    best_gov = {T: max((succ_at(v, T), k) for k, v in gov.items()) for T in Ts}
    best_uni = {T: max((succ_at(v, T), k) for k, v in uni.items()) for T in Ts}
    TB = next((T for T in Ts if best_gov[T][0] >= 0.9), None)
    show = sorted(set([150, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200] + ([TB] if TB else [])))
    print(f"{'T':>5} {'governor(best)':>15} {'uniform(best)':>14}  best uniform cell")
    for T in show:
        print(f"{T:>5} {best_gov[T][0]:>15.2f} {best_uni[T][0]:>14.2f}  {best_uni[T][1]}")
    print("T_B =", TB, "governor cell", best_gov[TB][1] if TB else None)
    if TB:
        print("uniform best at T_B:", best_uni[TB])
        print("uniform grid at T_B:")
        for k, v in sorted(uni.items()):
            print(f"   {k}: {succ_at(v, TB):.2f}")
        for k, v in gov.items():
            print("  gov", k, f"{succ_at(v, TB):.2f}", "@H", f"{succ_at(v, H):.2f}")
    return TB


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else table()
