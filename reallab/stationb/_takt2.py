"""Lab-side: T_B by the amended rule (geometric reference >= 85% on seeds 3000-3049) and its sensitivity (internal).
    python -m reallab.stationb._takt2 choose        -> table + T_B + chosen geometric cell
    python -m reallab.stationb._takt2 sensitivity   -> runs the scaled variants of the chosen cell, prints the table
"""
import json
import sys
from pathlib import Path

import numpy as np

from reallab.stationb._takt import H, OUT, SEEDS, outcome, succ_at

SENS = Path("runs/stationb/_lab/takt_sensitivity.json")
SPEED_PARAMS = ("K", "vmin", "vmax")


def cells(prefix):
    cur = json.loads(OUT.read_text())
    return {k: v for k, v in cur.items() if k.startswith(prefix)}


def choose(verbose=True):
    geo, gov, uni = cells("geometric"), cells("governor"), cells("uniform")
    Ts = list(range(10, H + 1, 10))
    best = {T: max((succ_at(v, T), k) for k, v in geo.items()) for T in Ts}
    TB = next((T for T in Ts if best[T][0] >= 0.85), None)
    if verbose:
        print(f"{'T':>5} {'geometric':>10} {'governor':>9} {'uniform':>8}   (best cell of each, tuning seeds 3000-3049)")
        for T in sorted(set([120, 150, 170, 200, 250, 300, 350, 400, 500, 600, 800] + ([TB] if TB else []))):
            g = best[T][0]
            gv = max(succ_at(v, T) for v in gov.values())
            u = max(succ_at(v, T) for v in uni.values())
            print(f"{T:>5} {g:>10.2f} {gv:>9.2f} {u:>8.2f}")
        print("T_B =", TB, "geometric cell:", best[TB][1] if TB else None)
        if TB:
            print("at T_B: uniform best", max((succ_at(v, TB), k) for k, v in uni.items()),
                  "governor best", max((succ_at(v, TB), k) for k, v in gov.items()))
    return TB, (json.loads(best[TB][1].split(" ", 1)[1]) if TB else None)


def sensitivity():
    from reallab.stationb._reference import lab_evaluate
    TB, kw = choose(verbose=False)
    variants = {"base": dict(kw)}
    for p in SPEED_PARAMS:
        for f in (0.8, 1.25):
            v = dict(kw)
            v[p] = kw[p] * f
            variants[f"{p} x{f}"] = v
    for f in (0.8, 1.25):
        variants[f"all x{f}"] = {**kw, **{p: kw[p] * f for p in SPEED_PARAMS}}
    cur = json.loads(SENS.read_text()) if SENS.exists() else {}
    for name, v in variants.items():
        key = json.dumps(v, sort_keys=True)
        if key not in cur:
            res = lab_evaluate("reallab.stationb._reference:geometric", v, SEEDS, purpose="smoke", tag="takt_sens",
                               n_workers=12, time_limit=H)
            cur[key] = [dict(seed=r["seed"], success=r["success"], n_steps=r["n_steps"], outcome=outcome(r)) for r in res]
            SENS.write_text(json.dumps(cur))
        rows = cur[key]
        br = {}
        for r in rows:
            o = r["outcome"] if (r["success"] is False or r["n_steps"] <= TB) else "time_out"
            if not r["success"] and r["n_steps"] > TB:
                o = "time_out"
            br[o] = br.get(o, 0) + 1
        print(f"{name:12s} success@T_B={TB}: {succ_at(rows, TB):.2f}   @H: {succ_at(rows, H):.2f}   {br}", flush=True)


if __name__ == "__main__":
    choose() if sys.argv[1] == "choose" else sensitivity()
