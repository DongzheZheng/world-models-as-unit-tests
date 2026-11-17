"""Probes run by the orchestrator to test one reading of the results before it goes into any write-up:

    "When episodes barely differ, one real flight tells you whether a controller works; when they differ a lot,
     one real flight tells you little -- and that is where debugging against reality stops converging."

  seedrange  Are calibration seeds (< 1000) drawn from the same distribution as evaluation seeds?  Final programs
             on seeds 0-99 (purpose="smoke") to compare with their evaluation (1000+) and audit (5000+) numbers.
             If low seeds were easier, the gap between test flights and evaluation would be a lab artefact.
  curves     Every program version the pilots left behind, evaluated on seeds no pilot has seen (6000+), indexed
             by the number of real lives consumed before that version existed: lives -> true success rate.

usage: python experiments/audit/probe.py seedrange|curves [task ...]     -> runs/probe/*.json
"""
import json
import statistics
import sys
import traceback
from pathlib import Path

OUT = Path("runs/probe")
CW = "experiments.drone.code_wm.controller:make_pilot"
CWD = "runs/drone/code_wm"


def _save(name, key, val):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = val
    p.write_text(json.dumps(d, indent=1))
    print(key, {k: v for k, v in val.items() if k != "per_seed"}, flush=True)


def drone(factory, kw, mission, seeds, purpose, tag):
    from reallab.drone.parallel import evaluate
    rows = evaluate(factory, kw, "site", seeds=seeds, purpose=purpose, tag=tag, n_workers=16, pilot="probe",
                    mission=mission, command_mode="ctbr")
    ok = [r for r in rows if r["success"]]
    return dict(n=len(rows), success=len(ok) / len(rows),
                time_success_s=statistics.fmean(r["time_s"] for r in ok) if ok else None,
                collision=sum(bool(r["collision"]) for r in rows), timeout=sum(bool(r["timeout"]) for r in rows),
                out_of_bounds=sum(bool(r["out_of_bounds"]) for r in rows),
                per_seed={int(r["seed"]): bool(r["success"]) for r in rows})


def push(factory, kw, obj, seeds, purpose, tag, workers=12):
    from reallab.pusht.parallel import evaluate
    eps = evaluate(factory, kw, seeds=seeds, purpose=purpose, pilot="probe", tag=tag, n_workers=workers, object=obj)
    return dict(n=len(eps), success=statistics.fmean(float(e["success"]) for e in eps),
                max_coverage=statistics.fmean(e["max_coverage"] for e in eps),
                n_steps=statistics.fmean(e["n_steps"] for e in eps),
                per_seed={int(e["seed"]): bool(e["success"]) for e in eps})


def doom(factory, kw, seeds, purpose, tag):
    from reallab.doom.parallel import evaluate
    eps = evaluate(factory, kw, scenario="take_cover", seeds=seeds, purpose=purpose, pilot="probe", tag=tag, n_workers=20)
    rets = [e["ret"] for e in eps]
    return dict(n=len(rets), mean=statistics.fmean(rets), median=statistics.median(rets),
                success=sum(r >= 750 for r in rets) / len(rets),          # "solved" bar of the scenario
                per_seed={int(e["seed"]): float(e["ret"]) for e in eps})


def kwc(tuned, belief, route=None):
    kw = dict(tuned=f"{CWD}/{tuned}.json", belief=f"{CWD}/{belief}.json")
    if route:
        kw["route"] = route
    return kw


LANE = ["R1", [20, 0, 4.8], [26, 0, 4.8], "R2", [31, 0, 4.8], [50, 0, 4.8], "R3"]
PUSH_PLANNER = json.loads(Path("runs/pusht/code_wm/eval_code_wm.json").read_text())["cfg"]["planner"]
_pl = Path("runs/pusht/code_wm/eval_L_code_wm.json")
PUSH_L_PLANNER = json.loads(_pl.read_text())["cfg"]["planner"] if _pl.exists() else None
ASP_D = "pilots.aspire_drone.policy_v{}:make_policy"
ASP_P = "pilots.aspire_pusht.policy_v{}:make_policy"
ASP_F = "pilots.aspire_doom.policy_v{}:make_policy"

# (key, lives consumed before this version existed, runner)
CURVES = {
    "drone_A": [(f"aspire_v{v}", l, lambda v=v: drone(ASP_D.format(v), {}, "A", range(6000, 6050), "eval", f"probe_A_v{v}"))
                for v, l in [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]] + [
        ("code_wm_after1", 1, lambda: drone(CW, kwc("tune_l2", "fit_l1"), "A", range(6000, 6050), "eval", "probe_A_cw1")),
        ("code_wm_after2", 2, lambda: drone(CW, kwc("tune_A12", "fit_l12"), "A", range(6000, 6050), "eval", "probe_A_cw2")),
        ("code_wm_after3", 3, lambda: drone(CW, kwc("tune_A123", "fit_l123"), "A", range(6000, 6050), "eval", "probe_A_cw3")),
    ],
    "drone_C": [(f"aspire_v{v}", l, lambda v=v: drone(ASP_D.format(v), {}, "C", range(6000, 6100), "eval", f"probe_C_v{v}"))
                for v, l in [(6, 0), (7, 0), (8, 1), (9, 2), (10, 3), (11, 5), (12, 10), (13, 15), (14, 17)]] + [
        ("code_wm_missionA_ctrl", 0, lambda: drone(CW, kwc("tune_A123", "fit_l123"), "C", range(6000, 6100), "eval", "probe_C_cwA")),
        ("code_wm_after1", 1, lambda: drone(CW, kwc("tune_C1", "fit_all", LANE), "C", range(6000, 6100), "eval", "probe_C_cw1")),
        ("code_wm_after2", 2, lambda: drone(CW, kwc("tune_C3a", "fit_all", LANE), "C", range(6000, 6100), "eval", "probe_C_cw2")),
        ("code_wm_after3", 3, lambda: drone(CW, kwc("tune_C4", "fit_all", LANE), "C", range(6000, 6100), "eval", "probe_C_cw3")),
        ("code_wm_after4", 4, lambda: drone(CW, kwc("tune_C5", "fit_all", LANE), "C", range(6000, 6100), "eval", "probe_C_cw4")),
    ],
    "push_T": [(f"aspire_v{v}", l, lambda v=v: push(ASP_P.format(v), {}, "T", range(6000, 6050), "eval", f"probe_T_v{v}"))
               for v, l in [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 7), (8, 10)]] + [
        ("code_wm_after2", 2, lambda: push("experiments.pusht.code_wm.controller:mpc_policy", PUSH_PLANNER, "T",
                                           range(6000, 6050), "eval", "probe_T_cw", workers=8)),
    ],
    "push_L": [(f"aspire_v{v}", l, lambda v=v: push(ASP_P.format(v), {}, "L", range(6000, 6050), "eval", f"probe_L_v{v}"))
               for v, l in [(9, 0), (10, 1), (11, 2), (12, 3), (13, 5), (14, 6), (15, 7)]],
    "doom": [(f"aspire_v{v}", l, lambda v=v: doom(ASP_F.format(v), {}, range(6000, 6050), "eval", f"probe_doom_v{v}"))
             for v, l in [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]],
    "push_L_cw": [
        ("push_L/code_wm_after2", 2, lambda: push("experiments.pusht.code_wm.controller:mpc_policy", PUSH_L_PLANNER, "L",
                                                  range(6000, 6050), "eval", "probe_L_cw", workers=8)),
        ("push_L_audit/code_wm", 2, lambda: push("experiments.pusht.code_wm.controller:mpc_policy", PUSH_L_PLANNER, "L",
                                                 range(5000, 5050), "eval", "audit_L_cw", workers=8)),
        ("push_L_audit/aspire_v10", 1, lambda: push(ASP_P.format(10), {}, "L", range(5000, 5050), "eval", "audit_L_v10")),
        ("push_L_audit/aspire_v15", 7, lambda: push(ASP_P.format(15), {}, "L", range(5000, 5050), "eval", "audit_L_v15")),
    ],
    # the code-world pilot's final planner (chosen in the code world after 3 lives; life 4 was a held-out test)
    "doom_cw": [("code_wm_after3", 3, lambda: doom("experiments.doom.code_wm.planner:planner",
                                                   dict(horizon=24, margin=2.0, w_center=0.3), range(6000, 6050), "eval", "probe_doom_cw"))],
}

LOW = range(0, 100)      # the range calibration lives are drawn from
SEEDRANGE = [
    ("push_T/aspire_v5", lambda: push(ASP_P.format(5), {}, "T", LOW, "smoke", "probe_low_T_v5")),
    ("push_T/aspire_v8", lambda: push(ASP_P.format(8), {}, "T", LOW, "smoke", "probe_low_T_v8")),
    ("push_T/aspire_v1", lambda: push(ASP_P.format(1), {}, "T", LOW, "smoke", "probe_low_T_v1")),
    ("drone_C/aspire_v11", lambda: drone(ASP_D.format(11), {}, "C", LOW, "smoke", "probe_low_C_v11")),
    ("drone_C/aspire_v14", lambda: drone(ASP_D.format(14), {}, "C", LOW, "smoke", "probe_low_C_v14")),
    ("drone_C/code_wm", lambda: drone(CW, kwc("tune_C5", "fit_all", LANE), "C", LOW, "smoke", "probe_low_C_cw")),
    ("drone_A/aspire_v5", lambda: drone(ASP_D.format(5), {}, "A", LOW, "smoke", "probe_low_A_v5")),
]

if __name__ == "__main__":
    what, tasks = sys.argv[1], sys.argv[2:]
    if what == "seedrange":
        for key, run in SEEDRANGE:
            try:
                _save("seedrange", key, run())
            except Exception:
                print("FAILED", key, traceback.format_exc().splitlines()[-1], flush=True)
    else:
        for task in tasks or list(CURVES):
            for key, lives, run in CURVES[task]:
                try:
                    _save("curves", key if "/" in key else f"{task}/{key}", dict(lives_before=lives, **run()))
                except Exception:
                    print("FAILED", task, key, traceback.format_exc().splitlines()[-1], flush=True)
