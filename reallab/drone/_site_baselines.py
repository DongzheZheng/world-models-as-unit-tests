"""Lab-side: public reference at the site (eval flights of the stock position autopilot) + GIFs."""
import json, os
import numpy as np
from reallab.drone.parallel import evaluate
from reallab.drone import render
from reallab.drone._baselines import stats

SEEDS = list(range(1000, 1050))
C = "reallab.drone._ctbr_pilot:"

if __name__ == "__main__":
    bl = json.load(open("runs/drone/baselines.json"))
    for k in [k for k in bl if k.startswith("site")]:
        del bl[k]
    logs = {}
    for m in ["A", "B"]:
        d = f"runs/drone/logs/site_nominal_autopilot_{m}"
        rows = evaluate(":site_nominal_policy", {}, "site", SEEDS, purpose="eval", tag="baseline:nominal_autopilot",
                        n_workers=32, log_dir=d, mission=m, command_mode="position", pilot="nominal_autopilot")
        ok = [r for r in rows if r["success"]]
        bl[f"site_mission{m}/nominal_autopilot"] = dict(
            n_episodes=len(rows), success_rate=len(ok) / len(rows),
            time_s_successes_mean=float(np.mean([r["time_s"] for r in ok])) if ok else None,
            end_causes={c: int(sum(bool(r[c]) for r in rows)) for c in ["success", "collision", "out_of_bounds", "timeout"]},
            time_limit_s=rows[0]["time_limit_s"], stats=stats(rows), rows=rows,
            note="stock PyFlyt position autopilot (public command_mode='position') following the centreline with a 2.5 m carrot, anomalies as flown by every pilot; eval seeds 1000-1049")
        print(m, {k: bl[f"site_mission{m}/nominal_autopilot"][k] for k in ["success_rate", "time_s_successes_mean", "end_causes", "time_limit_s"]}, flush=True)
        logs[m] = (d, rows)
    json.dump(bl, open("runs/drone/baselines.json", "w"), indent=1)
    g = "runs/drone/gifs"
    for f in os.listdir(g) if os.path.isdir(g) else []:
        os.remove(os.path.join(g, f))
    d, rows = logs["A"]
    s0 = rows[0]["seed"]
    render.render_episode_gif(f"{d}/site_eval_{s0}.jsonl.gz", f"{g}/nominal_autopilot.gif",
                              title=f"stock position autopilot, mission A, eval seed {s0}")
    prows = evaluate(C + "informed_policy", {}, "site", list(range(1000, 1008)), purpose="smoke", tag="gif:privileged",
                     n_workers=8, log_dir="runs/drone/logs/site_privileged_A", mission="A", command_mode="ctbr")
    win = next(r["seed"] for r in prows if r["success"])
    render.render_episode_gif(f"runs/drone/logs/site_privileged_A/site_smoke_{win}.jsonl.gz", f"{g}/privileged_ctbr_REVEAL.gif",
                              title=f"privileged CTBR pilot, mission A, seed {win} (truth revealed)", reveal_truth=True)
    sel = [r["seed"] for r in rows][:5]
    logsl = [f"{d}/site_eval_{s}.jsonl.gz" for s in sel] + [f"runs/drone/logs/site_privileged_A/site_smoke_{r['seed']}.jsonl.gz" for r in prows[:5]]
    render.render_topdown_gif(logsl, f"{g}/stock_vs_privileged_topdown.gif",
                              labels=[f"stock autopilot {s}" for s in sel] + [f"privileged CTBR {r['seed']}" for r in prows[:5]],
                              title="mission A: stock position autopilot vs privileged CTBR (truth revealed)", reveal_truth=True)
    print("privileged smoke rows", [(r["seed"], r["success"], r["time_s"]) for r in prows])
    print("gifs", [(f, round(os.path.getsize(os.path.join(g, f)) / 1e6, 2)) for f in os.listdir(g)])
