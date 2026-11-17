# Step 4: official evaluation, ONCE.  Writes the prediction into summary.json first, then runs 50 real episodes.
#   python experiments/station/code_wm/official_eval.py predict   -> writes summary.json with the predicted rate
#   python experiments/station/code_wm/official_eval.py eval      -> runs seeds 1000..1049 and records the result
import sys, json, time, os

KW = {"clear": 0.09, "load_soft": 3.0, "load_hard": 6.0}
OUT = "runs/station/code_wm"
REAL_CAL_STEPS = 600 + 378 + 308          # lives 1-3 (life_1 probe, life_2 seed 3, life_3 seed 12)
WM_STEPS, WM_WALL = 648_000, 120.0       # code-world steps / seconds spent on controller tuning + screening (tune.py logs)

if __name__ == "__main__":
    if sys.argv[1] == "predict":
        from experiments.station.code_wm.screen import run_in_world, layouts_as_obs, summarize
        res = run_in_world("experiments.station.code_wm.policy:make_policy", KW, layouts_as_obs())
        res_r = run_in_world("experiments.station.code_wm.policy:make_policy", KW, layouts_as_obs(), randomize=0.3, seed=7)
        s0, s1 = summarize(res), summarize(res_r)
        S = dict(written_before_eval=time.strftime("%Y-%m-%d %H:%M:%S"), controller="RoutePolicy " + json.dumps(KW),
                 predicted_success_rate=round(min(s0["success"], s1["success"]), 3),
                 prediction_basis=dict(nominal_world_200_snapshots=s0, randomized_world_30pct_200_snapshots=s1))
        json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
        print(json.dumps(S, indent=1))
    elif sys.argv[1] == "eval":
        from reallab.station import parallel
        from cawm import results
        assert not os.path.exists(f"{OUT}/eval_result.json"), "official eval already run"
        t0 = time.time()
        eps = parallel.evaluate("experiments.station.code_wm.policy:make_policy", KW, seeds=list(range(1000, 1050)),
                                purpose="eval", pilot="code_wm", tag="official", n_workers=10,
                                log_dir=f"{OUT}/eval_logs")
        json.dump(dict(episodes=eps, wall_s=time.time() - t0), open(f"{OUT}/eval_result.json", "w"), indent=1)
        rec = results.record(demo="station", scenario="station", controller="code_wm", family="code_as_wm",
                             trained_in="code_world", episodes=eps, metric="success",
                             other_metrics=("overload", "sides_ok", "final_dist", "max_load", "n_steps"),
                             real_steps_calibration=REAL_CAL_STEPS, real_steps_training=0, wm_steps=WM_STEPS,
                             train_wall_s=WM_WALL,
                             notes="rope-routing code world (spring-damper chain, viscous drag, inertia, peg rubbing, PD servo) "
                                   "fitted by CEM to 10 observation logs + life_1 probe; waypoint router with load-governed "
                                   "speed, snag pause and taut back-off, tuned in the code world; 3 of 5 lives used.")
        n = len(eps); s = sum(e["success"] for e in eps); o = sum(e["overload"] for e in eps)
        print("success", s, "/", n, "overload", o, "wrong side", sum((not e["success"]) and (not e["overload"]) and e["sides_ok"] < 3 for e in eps))
        for e in eps:
            if not e["success"]:
                print(e)
