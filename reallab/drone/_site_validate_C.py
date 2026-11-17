"""Lab-side validation of mission C (movers): T_c of the privileged avoider -> L_C = 1.5 T_c, then success rates under L_C."""
import json, os
import numpy as np
from reallab.drone.parallel import evaluate

SEEDS = list(range(3000, 3100))
CT = "reallab.drone._ctbr_pilot:"
PILOTS = {
    "stock_autopilot": (":site_nominal_policy", {}, "position"),
    "privileged_ctbr_avoider": (CT + "avoid_policy", {}, "ctbr"),
    "privileged_ctbr_missionA_controller_ignores_movers": (CT + "informed_policy", {}, "ctbr"),
}


def run(name):
    fac, pk, mode = PILOTS[name]
    rows = evaluate(fac, pk, "site", SEEDS, purpose="smoke", tag=f"validateC:{name}", n_workers=32, mission="C", command_mode=mode)
    ok = [x for x in rows if x["success"]]
    r = dict(success=float(np.mean([x["success"] for x in rows])), relays=float(np.mean([x["relays_reached"] for x in rows])),
             time_s_successes=float(np.mean([x["time_s"] for x in ok])) if ok else None, n=len(rows),
             collision=float(np.mean([x["collision"] for x in rows])), timeout=float(np.mean([x["timeout"] for x in rows])),
             out_of_bounds=float(np.mean([x["out_of_bounds"] for x in rows])), time_limit_s=rows[0]["time_limit_s"])
    print(name, r, flush=True)
    return r


if __name__ == "__main__":
    out = json.load(open("runs/drone/site_validation_lab_privileged.json"))
    os.environ["CAWM_SITE_TUNE"] = json.dumps({"LC": 60.0})
    tc = run("privileged_ctbr_avoider")["time_s_successes"]
    L = round(1.5 * tc, 1)
    print("CALIBRATION C: T_c", tc, "L_C", L, flush=True)
    os.environ["CAWM_SITE_TUNE"] = json.dumps({"LC": L})
    out["time_limit_calibration"]["C"] = dict(T_c=tc, L=L, rule="1.5 x mean completion time of the privileged avoider")
    for name in PILOTS:
        out[f"{name}_missionC_anomalies_on"] = run(name)
    json.dump(out, open("runs/drone/site_validation_lab_privileged.json", "w"), indent=1)
