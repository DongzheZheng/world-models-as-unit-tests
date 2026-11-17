"""Official real evaluation (run ONCE per controller): seeds 1000-1049, purpose="eval", then cawm.results.record.
python experiments/pusht/code_wm/eval_real.py <cfg.json>
cfg: {"controller": "code_wm", "planner": {...}, "real_steps_calibration": int, "wm_steps": int, "train_wall_s": float,
      "notes": str, "n_workers": 8}"""
import json
import sys
import time
from pathlib import Path

from cawm import results
from reallab.pusht import parallel


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    name = cfg["controller"]
    obj = cfg.get("object", "T")
    scen = "pusht" if obj == "T" else "pusht_L"
    tagname = name if obj == "T" else f"L_{name}"
    out = Path("runs/pusht/code_wm")
    t0 = time.time()
    eps = parallel.evaluate("experiments.pusht.code_wm.controller:mpc_policy", cfg["planner"],
                            seeds=list(range(1000, 1050)), purpose="eval", pilot="code_wm", tag=tagname, object=obj,
                            n_workers=cfg.get("n_workers", 8), log_dir=str(out / f"eval_logs_{tagname}"),
                            record_first=3, video_dir=str(out / "gifs" / f"eval_{tagname}"))
    wall = time.time() - t0
    (out / f"eval_{tagname}.json").write_text(json.dumps(dict(episodes=eps, wall_s=wall, cfg=cfg), indent=1))
    rec = results.record(demo="pusht", scenario=scen, controller=name, family="code_as_wm", trained_in="code_world",
                         episodes=eps, metric="success", other_metrics=("max_coverage", "final_coverage", "n_steps"),
                         real_steps_calibration=cfg["real_steps_calibration"], real_steps_training=0,
                         wm_steps=cfg["wm_steps"], train_wall_s=cfg["train_wall_s"], notes=cfg["notes"])
    print(json.dumps(dict(success=rec["eval"]["mean"], max_cov=rec["other"]["max_coverage"]["mean"],
                          n_steps=rec["other"]["n_steps"]["mean"], wall_s=wall), indent=1))


if __name__ == "__main__":
    main()
