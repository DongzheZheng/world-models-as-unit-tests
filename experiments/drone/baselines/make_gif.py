import json, sys
from reallab.drone.render import render_episode_gif
name, label = sys.argv[1], sys.argv[2]
rows = json.load(open(f"runs/drone/baselines_rl/evals/{name}.json"))["rows"]
best = sorted(rows, key=lambda r: (r["success"], r["relays_reached"], -r["time_s"] if r["success"] else r["time_s"]), reverse=True)[0]
src = f"runs/drone/baselines_rl/evlogs/{name}/site_eval_{best['seed']}.jsonl.gz"
out = f"runs/drone/baselines_rl/{name}.gif"
render_episode_gif(src, out, title=f"{label}  seed {best['seed']}  relays {best['relays_reached']}/3  success={best['success']}")
print(out, best["seed"], best["relays_reached"], best["success"])
