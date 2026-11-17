"""Lab-side: public reference on mission C (stock autopilot, eval seeds 1000-1099) + the privileged GIF."""
import json, os
import numpy as np
from reallab.drone.parallel import evaluate
from reallab.drone import render
from reallab.drone._baselines import stats

if __name__ == "__main__":
    bl = json.load(open("runs/drone/baselines.json"))
    d = "runs/drone/logs/site_nominal_autopilot_C"
    rows = evaluate(":site_nominal_policy", {}, "site", list(range(1000, 1100)), purpose="eval", tag="baseline:nominal_autopilot",
                    n_workers=32, log_dir=d, mission="C", command_mode="position", pilot="nominal_autopilot")
    ok = [r for r in rows if r["success"]]
    bl["site_missionC/nominal_autopilot"] = dict(
        n_episodes=len(rows), success_rate=len(ok) / len(rows),
        time_s_successes_mean=float(np.mean([r["time_s"] for r in ok])) if ok else None,
        end_causes={c: int(sum(bool(r[c]) for r in rows)) for c in ["success", "collision", "out_of_bounds", "timeout"]},
        time_limit_s=rows[0]["time_limit_s"], stats=stats(rows), rows=rows,
        note="stock PyFlyt position autopilot (public command_mode='position'), centreline carrot 2.5 m; eval seeds 1000-1099")
    json.dump(bl, open("runs/drone/baselines.json", "w"), indent=1)
    print({k: bl["site_missionC/nominal_autopilot"][k] for k in ["success_rate", "end_causes", "time_limit_s", "time_s_successes_mean"]}, flush=True)
    prows = evaluate("reallab.drone._ctbr_pilot:avoid_policy", {}, "site", list(range(1000, 1010)), purpose="smoke", tag="gif:privilegedC",
                     n_workers=10, log_dir="runs/drone/logs/site_privileged_C", mission="C", command_mode="ctbr")
    print([(r["seed"], r["success"], r["time_s"]) for r in prows])
    win = next(r["seed"] for r in prows if r["success"])
    g = "runs/drone/gifs"
    render.render_episode_gif(f"runs/drone/logs/site_privileged_C/site_smoke_{win}.jsonl.gz", f"{g}/privileged_C_REVEAL.gif",
                              title=f"privileged CTBR pilot, mission C, seed {win} (truth revealed)", reveal_truth=True)
    print(os.path.getsize(f"{g}/privileged_C_REVEAL.gif") / 1e6, "MB")
