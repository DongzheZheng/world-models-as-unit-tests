"""Tally code-world steps and wall time spent obtaining the controllers (from logs/json) -> runs/doom/code_wm/cost.json"""
import ast
import json
import re
from pathlib import Path

O = Path("runs/doom/code_wm")
wm, wall, parts = 0, 0.0, {}
for t in ("smoke", "smoke2", "v1", "v2"):
    p = O / f"train_{t}.json"
    if p.exists():
        r = json.loads(p.read_text())
        ev = sum(v["n"] * v["mean"] / 4 for v in r["world_eval"].values())
        parts[f"ppo_{t}"] = dict(train_steps=r["wm_steps"], eval_steps=int(ev), wall_s=r["train_wall_s"])
        wm += r["wm_steps"] + ev
        wall += r["train_wall_s"]
grid = [ast.literal_eval(l) for l in Path("runs/logs/cw_plan.log").read_text().splitlines() if l.startswith("{")]
g_steps = sum(d["n"] * d["mean"] / 4 for d in grid)
parts[f"planner_grid(killed after {len(grid)} of 27 configs)"] = dict(eval_steps=int(g_steps), configs=len(grid))
wm += g_steps
for f in ("selection.json", "selection_v3world.json"):
    p = O / f
    if p.exists():
        c = json.loads(p.read_text())["candidates"]
        s = sum(v["n"] * v["mean"] / 4 for v in c.values())
        w = sum(v.get("eval_wall_s", 0) for v in c.values())
        parts[f] = dict(eval_steps=int(s), wall_s=w)
        wm += s
        wall += w
wall += 70.0 * len(grid)   # planner grid: ~70 s per config (log timing; not saved per entry)
out = dict(wm_steps_total=int(wm), train_wall_s_total=round(wall), cost_parts=parts,
           wall_note="PPO training + candidate evaluation in the code world; excludes real lives and my editing time")
(O / "cost.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
