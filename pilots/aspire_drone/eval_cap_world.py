"""Real evaluation (once) of the program chosen inside the code world: mission C, seeds 1000-1099, pilot cap_in_world.
   python pilots/aspire_drone/eval_cap_world.py v15 <wm_steps> <train_wall_s>"""
import json
import sys
from pathlib import Path


def main():
    from cawm.results import record
    from reallab.drone.parallel import evaluate
    from reallab.drone.render import render_episode_gif
    ver, wm_steps, wall = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    out = Path("runs/drone/aspire_style")
    log_dir = out / "eval_logs_cap_in_code_world_C"
    rows = evaluate(f"pilots.aspire_drone.policy_{ver}:make_policy", {}, "site", seeds=list(range(1000, 1100)),
                    purpose="eval", tag="cap_in_code_world_eval_C", n_workers=16, log_dir=str(log_dir),
                    pilot="cap_in_world", mission="C", command_mode="ctbr")
    for r in rows:
        r["time_success_s"] = r["time_s"] if r["success"] else None
    (out / "eval_cap_in_code_world_C.json").write_text(json.dumps(rows, indent=1, default=str))
    notes = (f"policy_{ver}: aspire_style v14 lineage iterated only inside the code_wm code world (worlds/drone/site_world.py "
             "via the screen harness setup): corridor look-ahead lateral barrier (pinch / aisle end faces), smaller "
             "keep-outs (0.6 m moving, 1.3 m hovering), vertical bias 3, along-track escape cap; chosen on ~22.6k imagined "
             "episodes (code-world success 94.8 % on 1000 fresh imagined episodes; v14 77.4 %, v11 51.1 %). No real flight "
             "before this evaluation. wm_steps excludes one 50-episode trace run (<= 29,100 steps).")
    rec = record(demo="drone", scenario="site_C", controller="cap_in_code_world", family="code_as_policy_in_code_world",
                 trained_in="code_world", episodes=rows, metric="success",
                 other_metrics=("time_success_s", "relays_reached", "collision", "out_of_bounds", "timeout"),
                 real_steps_calibration=1810, real_steps_training=6031, wm_steps=wm_steps, train_wall_s=wall, notes=notes)
    print(json.dumps({k: rec[k] for k in ("eval", "other")}))
    print({k: sum(bool(r[k]) for r in rows) for k in ("success", "collision", "out_of_bounds", "timeout")})
    render_episode_gif(str(log_dir / "site_eval_1000.jsonl.gz"), str(out / "cap_in_code_world_C_eval_seed1000.gif"),
                       title=f"cap_in_code_world ({ver}) - eval seed 1000, mission C")


if __name__ == "__main__":
    main()
