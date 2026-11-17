"""Orchestrator-side checks for the station demo.  Everything runs under the __main__ guard (spawned workers).

  real   <factory> <label> [first_seed] [n]   real episodes of a program on unseen seeds (default 6000, 50)
  world  <factory> <label> [first_seed] [n]   the same program scored inside the code world, zero real episodes,
                                              started from the reset readings of the same seeds (looking is free)
  versions <k1,k2,...> real|world             shorthand for the code-as-policy versions pilots.aspire_station.policy_v<k>
  report                                      table of real against predicted

The reset readings are taken with `python -m reallab.station._snapshots` (zero agent steps) on first use and cached in
runs/station/audit/layouts_<first>_<last>.npz (see experiments/audit/snapshots.py).
Results accumulate in runs/probe/station_probe.json  ({"real": {label: {...}}, "world": {label: {...}}}).
usage (from the repository root): python -m experiments.audit.station_probe <command> ...
"""
import json
import statistics
import sys
import time
from pathlib import Path

OUT = Path("runs/probe/station_probe.json")


def _load():
    return json.loads(OUT.read_text()) if OUT.exists() else {"real": {}, "world": {}}


def _save(kind, label, rec):
    d = _load()
    d.setdefault(kind, {})[label] = rec
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=1))


def _summ(eps):
    n = len(eps)
    over = sum(bool(e["overload"]) for e in eps)
    ok = sum(bool(e["success"]) for e in eps)
    wrong = sum((not e["success"]) and (not e["overload"]) and int(e["sides_ok"]) < 3 for e in eps)
    return dict(n=n, success=ok / n, overload=over / n, wrong_side=wrong / n,
                not_at_target=(n - ok - over - wrong) / n,
                max_load=statistics.fmean(float(e["max_load"]) for e in eps),
                n_steps=statistics.fmean(float(e["n_steps"]) for e in eps),
                per_seed=[int(bool(e["success"])) for e in eps])


def real(factory, label, first=6000, n=50, kw=None):
    from reallab.station.parallel import evaluate
    eps = evaluate(factory, kw or {}, seeds=list(range(first, first + n)), purpose="eval", pilot="audit", tag=f"probe_{label}",
                   n_workers=12, log_dir=f"runs/station/audit_logs/{label}_{first}")
    rec = dict(_summ(eps), first_seed=first, factory=factory, kwargs=kw or {})
    _save("real", f"{label}@{first}", rec)
    print("REAL ", label, first, {k: round(v, 3) for k, v in rec.items() if isinstance(v, float)}, flush=True)


def _init_obs(first, n):
    from experiments.audit.snapshots import init_obs_station
    return init_obs_station(first, n)


def world(factory, label, first=6000, n=50, kw=None):
    from experiments.station.code_wm.screen import run_in_world
    t0 = time.time()
    eps = run_in_world(factory, kw or {}, _init_obs(first, n))
    rec = dict(_summ(eps), first_seed=first, factory=factory, kwargs=kw or {}, wall_s=time.time() - t0)
    _save("world", f"{label}@{first}", rec)
    print("WORLD", label, first, {k: round(v, 3) for k, v in rec.items() if isinstance(v, float)}, flush=True)


def report():
    d = _load()
    labels = sorted(set(d["real"]) | set(d["world"]))
    rows = []
    for lb in labels:
        r, w = d["real"].get(lb), d["world"].get(lb)
        rows.append((lb, r and r["success"], w and w["success"]))
        agree = ""
        if r and w:
            agree = f'  same outcome on {sum(a == b for a, b in zip(r["per_seed"], w["per_seed"]))}/{len(r["per_seed"])} seeds'
        print(f'{lb:34s} real {("%5.1f%%" % (100 * r["success"])) if r else "   -  "}  (overload {("%4.0f%%" % (100 * r["overload"])) if r else "  - "})'
              f'   world {("%5.1f%%" % (100 * w["success"])) if w else "   -  "}{agree}')
    both = [(a, b) for _, a, b in rows if a is not None and b is not None]
    if len(both) >= 2:
        err = [abs(a - b) for a, b in both]
        ma, mb = statistics.fmean(a for a, _ in both), statistics.fmean(b for _, b in both)
        va, vb = sum((a - ma) ** 2 for a, _ in both), sum((b - mb) ** 2 for _, b in both)
        r = sum((a - ma) * (b - mb) for a, b in both) / (va * vb) ** 0.5 if va and vb else float("nan")
        print(f"n={len(both)}  mean|err|={100 * statistics.fmean(err):.1f} pts  max|err|={100 * max(err):.1f}  r={r:.3f}")


# A spread of programs for testing the code world as a judge: the pilots' programs plus the lab's simple
# open-loop router at several speeds and clearances (no load feedback), which spans failure to success.
NAIVE = "reallab.station._reference:naive_router"
GRID = ([(f"cap_v{k}", f"pilots.aspire_station.policy_v{k}:make_policy", {}) for k in (1, 2, 3)]
        + [("code_wm_ctrl", "experiments.station.code_wm.policy:make_policy", {"clear": 0.09, "load_soft": 3.0, "load_hard": 6.0})]
        + [(f"open_loop_c{int(c * 1000):03d}_v{int(v * 100):02d}", NAIVE, {"clearance": c, "speed": v})
           for c in (0.09, 0.065) for v in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "audit":          # the two submitted controllers with their official settings, unseen seeds 5000..5049
        real(GRID[3][1], "code_wm_ctrl", 5000, 50, GRID[3][2])
        if "cap_v3@5000" not in _load()["real"]:
            real(GRID[2][1], "cap_v3", 5000, 50, GRID[2][2])
        sys.exit(0)
    if cmd == "grid":
        fn = real if sys.argv[2] == "real" else world
        done = _load().get(sys.argv[2], {})
        for label, fac, kw in GRID:
            if f"{label}@6000" not in done:
                fn(fac, label, 6000, 50, kw)
        sys.exit(0)
    if cmd in ("real", "world"):
        fn = real if cmd == "real" else world
        fn(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 6000, int(sys.argv[5]) if len(sys.argv) > 5 else 50)
    elif cmd == "versions":
        fn = real if sys.argv[3] == "real" else world
        for k in sys.argv[2].split(","):
            fn(f"pilots.aspire_station.policy_v{k}:make_policy", f"cap_v{k}")
    elif cmd == "report":
        report()
