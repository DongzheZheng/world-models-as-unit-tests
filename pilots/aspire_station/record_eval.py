"""Record the official evaluation (already run once by fly.py) with cawm.results and print summary numbers."""
import json
import numpy as np

if __name__ == "__main__":
    from cawm import results
    res = json.load(open("runs/station/aspire_style/eval_aspire_style_v3.json"))
    lives = [np.load(f"runs/station/blackbox/aspire_style/life_{k}.npz", allow_pickle=True) for k in range(1, 6)]
    steps = int(sum(int(d["n_steps"]) for d in lives))
    rec = results.record(demo="station", scenario="station", controller="aspire_style", family="code_as_policy",
                         trained_in="real", episodes=res, metric="success",
                         other_metrics=("overload", "sides_ok", "final_dist", "max_load", "n_steps"),
                         real_steps_calibration=0, real_steps_training=steps,
                         notes=("ASPIRE-style code-as-policy; program policy_v3 (topological taut path at 0.065 m from "
                                "posts + load-governed reference follower with back-off/stall recovery); 5 real lives "
                                f"({steps} steps), all successes; no simulator/forward model used."))
    ml = np.array([r["max_load"] for r in res]); ns = np.array([r["n_steps"] for r in res])
    print("real_steps_training", steps, "per-life", [int(d["n_steps"]) for d in lives])
    print("eval", rec["eval"])
    print("max_load mean/max", ml.mean().round(2), ml.max().round(2), "n_steps mean/max", ns.mean(), ns.max())
    print("other", json.dumps(rec["other"]))
