"""Official evaluation (run ONCE): 50 real episodes, seeds 1000-1049, 6 workers; records the result.
usage: python run_eval.py <controller_name> <log_dir> <real_steps_calibration> <wm_steps> <train_wall_s> <notes>"""
import json
import sys
import time

if __name__ == "__main__":
    from reallab.annex.parallel import evaluate
    from cawm.results import record
    name, log_dir = sys.argv[1], sys.argv[2]
    cal, wm, tw = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    notes = open(sys.argv[6]).read().strip()          # notes come from a text file (shell quoting)
    t0 = time.time()
    eps = evaluate("experiments.annex.code_wm.throw:make_final", {}, seeds=list(range(1000, 1050)), purpose="eval",
                   pilot="code_wm", n_workers=6, log_dir=log_dir)
    wall = time.time() - t0
    rec = record(demo="annex", scenario="annex", controller=name, family="code_as_wm", trained_in="code_world",
                 episodes=eps, metric="success", other_metrics=("tip_dist", "n_steps"), real_steps_calibration=cal,
                 real_steps_training=0, wm_steps=wm, train_wall_s=tw, notes=notes)
    json.dump(dict(episodes=eps, wall_s=wall), open(f"runs/annex/code_wm/eval_{name}.json", "w"), indent=1)
    print("success rate", sum(e["success"] for e in eps) / len(eps), "wall", wall)
