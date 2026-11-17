"""Lab-side validation of the site tuning (smoke flights; privileged controllers and the anomalies-off switch live here).
Step 1 measures T_ap (stock autopilot, anomalies off) and T_c (privileged CTBR, anomalies on) with a 60 s limit,
sets L = min(0.8 T_ap, 1.5 T_c) per mission; step 2 re-runs every condition under L."""
import json, os, sys
import numpy as np
from reallab.drone.parallel import evaluate

SEEDS = list(range(3000, 3100))
C = "reallab.drone._ctbr_pilot:"
PILOTS = {
    "stock_autopilot": (":site_nominal_policy", {}, "position"),
    "privileged_ctbr_informed": (C + "informed_policy", {}, "ctbr"),
    "privileged_ctbr_windblind": (C + "blind_policy", {}, "ctbr"),
}


def run(name, mission, anomalies):
    fac, pk, mode = PILOTS[name]
    kw = dict(mission=mission, command_mode=mode)
    if not anomalies:
        kw["anomalies"] = False
    rows = evaluate(fac, pk, "site", SEEDS, purpose="smoke", tag=f"validate:{name}", n_workers=32, **kw)
    ok = [x for x in rows if x["success"]]
    r = dict(success=float(np.mean([x["success"] for x in rows])), relays=float(np.mean([x["relays_reached"] for x in rows])),
             time_s_successes=float(np.mean([x["time_s"] for x in ok])) if ok else None, n=len(rows),
             collision=float(np.mean([x["collision"] for x in rows])), timeout=float(np.mean([x["timeout"] for x in rows])),
             out_of_bounds=float(np.mean([x["out_of_bounds"] for x in rows])))
    print(name, mission, "anomalies", anomalies, r, flush=True)
    return r


if __name__ == "__main__":
    out = {}
    os.environ["CAWM_SITE_TUNE"] = json.dumps({"LA": 60.0, "LB": 60.0})
    cal = {}
    for m in "AB":
        tap = run("stock_autopilot", m, False)["time_s_successes"]
        tc = run("privileged_ctbr_informed", m, True)["time_s_successes"]
        cal[m] = dict(T_ap=tap, T_c=tc, L=round(min(0.8 * tap, 1.5 * tc), 1))
    print("CALIBRATION", cal, flush=True)
    out["time_limit_calibration"] = cal
    os.environ["CAWM_SITE_TUNE"] = json.dumps({"LA": cal["A"]["L"], "LB": cal["B"]["L"]})
    for m in "AB":
        for name in PILOTS:
            for an in (True, False):
                out[f"{name}_mission{m}_anomalies_{'on' if an else 'off'}"] = run(name, m, an)
    json.dump(out, open("runs/drone/site_validation_lab_privileged.json", "w"), indent=1)
