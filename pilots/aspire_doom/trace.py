"""Read an execution trace (black-box life log) and print a diagnosis.
python pilots/aspire_doom/trace.py <path.jsonl.gz> [--steps] [--replay vK]"""
import gzip
import json
import math
import sys
from collections import Counter

path = sys.argv[1]
lines = [json.loads(l) for l in gzip.open(path, "rt")]
meta, reset, steps = lines[0]["meta"], lines[1]["reset"], lines[2:]
p0 = reset["obs"]["player"]
print("meta", meta)
print("start player", {k: p0[k] for k in ("x", "y", "angle", "health")})
print("n_steps", len(steps), "return", sum(s["reward"] for s in steps), "actions", Counter(s["action"] for s in steps))
names = Counter()
gtn = Counter()
for s in steps:
    for o in s["obs"]["objects"]:
        names[o["name"]] += 1
    for o in s["info"].get("gt_objects", []):
        gtn[(o["name"], o["visible"])] += 1
print("visible names", names)
print("gt names", gtn)
ys = [s["obs"]["player"]["y"] for s in steps]
xs = [s["obs"]["player"]["x"] for s in steps]
print("player x range", min(xs), max(xs), "y range", min(ys), max(ys))
vl = [abs(s["obs"]["player"]["vy"]) for s in steps]
print("max |vy|", max(vl))
# hits
prev = p0
obs_prev = reset["obs"]
for i, s in enumerate(steps):
    pl = s["obs"]["player"]
    if pl["hits_taken"] > prev["hits_taken"]:
        balls = [o for o in obs_prev["objects"] if math.hypot(o["vx"], o["vy"]) > 2]
        bs = sorted(balls, key=lambda o: math.hypot(o["x"] - prev["x"], o["y"] - prev["y"]))[:3]
        print(f"HIT step {i} tic {s['obs']['tic']} dmg {pl['damage_taken']-prev['damage_taken']} hp {pl['health']} "
              f"pos ({prev['x']:.0f},{prev['y']:.0f}) vy {prev['vy']:.1f} act {s['action']} near balls:",
              [(o["name"], round(o["x"]), round(o["y"]), round(o["vx"], 1), round(o["vy"], 1)) for o in bs])
    prev, obs_prev = pl, s["obs"]
if "--steps" in sys.argv:
    for i, s in enumerate(steps):
        pl = s["obs"]["player"]
        objs = [(o["name"][:6], round(o["x"]), round(o["y"]), round(o["vx"], 1), round(o["vy"], 1))
                for o in s["obs"]["objects"]]
        print(i, s["action"], round(pl["x"]), round(pl["y"]), round(pl["vy"], 1), pl["health"], objs)
