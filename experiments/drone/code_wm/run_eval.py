"""Official evaluation (run ONCE per controller): 50 real episodes, seeds 1000-1049, mission A, then record.
   python experiments/drone/code_wm/run_eval.py <cfg.json>
cfg: {"controller": "code_wm", "pilot_kwargs": {...}, "tune_json": path (wm_steps / wall), "real_steps_calibration": int,
      "notes": str, "extra_wm_steps": int, "extra_wall_s": float}"""
import json
import sys
from pathlib import Path


def main():
    from cawm.results import record
    from reallab.drone.parallel import evaluate
    from reallab.drone.render import render_episode_gif
    cfg = json.load(open(sys.argv[1]))
    name = cfg["controller"]
    mission = cfg.get("mission", "A")
    scenario = cfg.get("scenario", "site_A")
    tagname = cfg.get("tag", name)              # file tag (phase 2: "code_wm_B"); the recorded controller name stays `name`
    log_dir = f"runs/drone/code_wm/eval_logs/{tagname}"
    seeds = list(range(1000, 1000 + int(cfg.get("n_eval", 50))))
    rows = evaluate("experiments.drone.code_wm.controller:make_pilot", cfg["pilot_kwargs"], "site", seeds=seeds,
                    purpose="eval", tag=f"code_wm eval {tagname}", n_workers=16, log_dir=log_dir, pilot="code_wm",
                    mission=mission, command_mode="ctbr")
    for r in rows:
        r["time_success_s"] = r["time_s"] if r["success"] else None
    Path(f"runs/drone/code_wm/eval_{tagname}_rows.json").write_text(json.dumps(rows, indent=1, default=float))
    tj = json.load(open(cfg["tune_json"]))
    rec = record(demo="drone", scenario=scenario, controller=name, family="code_as_wm", trained_in="code_world",
                 episodes=rows, metric="success",
                 other_metrics=("time_success_s", "relays_reached", "collision", "out_of_bounds", "timeout"),
                 real_steps_calibration=cfg["real_steps_calibration"], real_steps_training=0,
                 wm_steps=int(tj["wm_steps"]) + int(cfg.get("extra_wm_steps", 0)),
                 train_wall_s=float(tj["wall_s"]) + float(cfg.get("extra_wall_s", 0.0)), notes=cfg["notes"])
    print(json.dumps(rec["eval"]), json.dumps(rec["other"]))
    first = sorted(Path(log_dir).glob("*_1000.jsonl.gz"))[0]
    render_episode_gif(str(first), f"runs/drone/code_wm/gifs/eval_{tagname}_seed1000.gif",
                       title=f"{name} - eval seed 1000 (mission {mission})")


if __name__ == "__main__":
    main()
