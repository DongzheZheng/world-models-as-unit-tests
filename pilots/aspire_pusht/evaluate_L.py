"""Evaluation on the L (once per controller): python pilots/aspire_pusht/evaluate_L.py <version> <controller> <pilot_glob> <notes_file>"""
import glob
import json
import sys

import numpy as np

if __name__ == "__main__":
    from reallab.pusht.parallel import evaluate
    from cawm.results import record
    ver, controller, pglob = sys.argv[1], sys.argv[2], sys.argv[3]
    notes = open(sys.argv[4]).read().strip()
    out = f"runs/pusht/aspire_style/eval_L_{controller}"
    eps = evaluate(f"pilots.aspire_pusht.policy_v{ver}:make_policy", {}, list(range(1000, 1050)), purpose="eval",
                   pilot="aspire_style", tag=f"L_{controller}", n_workers=12, log_dir=out, record_first=3,
                   video_dir=f"runs/pusht/aspire_style/gifs/L_{controller}_eval", object="L")
    lives = sorted(glob.glob(f"runs/pusht/blackbox_L/{pglob}/life_*.npz"))
    steps = int(sum(int(np.load(f)["n_steps"]) for f in lives))
    rec = record(demo="pusht", scenario="pusht_L", controller=controller, family="code_as_policy", trained_in="real",
                 episodes=eps, metric="success", other_metrics=("max_coverage", "final_coverage", "n_steps"),
                 real_steps_calibration=0, real_steps_training=steps,
                 notes=notes + f" ({len(lives)} L lives, {steps} real L steps)")
    json.dump(eps, open(f"{out}/episodes.json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("eval", "other", "real_steps_training")}, indent=1))
    print("lives", lives)
