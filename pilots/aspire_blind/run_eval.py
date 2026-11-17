"""Official evaluation of the submitted program: 50 episodes, seeds 1000-1049, then record."""
import json
import sys

if __name__ == "__main__":
    from reallab.station import parallel
    from cawm import results
    mod = sys.argv[1]
    controller = sys.argv[2] if len(sys.argv) > 2 else "aspire_blind"
    train_steps = int(sys.argv[3])
    notes = sys.argv[4] if len(sys.argv) > 4 else ""
    eps = parallel.evaluate(f"pilots.aspire_blind.{mod}:make_policy", {}, seeds=list(range(1000, 1050)), purpose="eval",
                            pilot="aspire_blind", n_workers=12, log_dir="runs/station/aspire_blind/eval_logs")
    rec = results.record(demo="station", scenario="station", controller=controller, family="code_as_policy",
                         trained_in="real", episodes=eps, metric="success",
                         other_metrics=("overload", "sides_ok", "final_dist", "max_load", "n_steps"),
                         real_steps_calibration=0, real_steps_training=train_steps, notes=notes)
    with open(f"runs/station/aspire_blind/eval_{controller}.json", "w") as f:
        json.dump(eps, f, indent=1)
    print("success rate", rec["eval"]["mean"])
    for e in eps:
        if not e["success"]:
            print("FAIL", e)
    print("other", json.dumps(rec["other"]))
