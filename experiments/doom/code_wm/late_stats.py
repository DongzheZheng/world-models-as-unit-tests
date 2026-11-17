"""Late-game statistics of one life: imps over time, throw rate per imp by period, damage events.
usage: python experiments/doom/code_wm/late_stats.py <k>"""
import gzip
import json
import math
import sys

k = int(sys.argv[1])
recs = [json.loads(l) for l in gzip.open(f"runs/doom/blackbox/take_cover/code_wm/life_{k}.jsonl.gz", "rt")]
S = [(recs[1]["reset"]["obs"], recs[1]["reset"]["info"]["gt_objects"])] + [(r["obs"], r["info"]["gt_objects"]) for r in recs[2:]]
births, balls, dmg, prev = {}, {}, [], (0, 100)
xs = set()
for o, gt in S:
    t = o["tic"]
    p = o["player"]
    xs.add(round(p["x"], 1))
    if p["hits_taken"] != prev[0]:
        dmg.append(dict(tic=t, hits=p["hits_taken"], damage=p["damage_taken"], health=p["health"], x=round(p["x"], 1)))
        prev = (p["hits_taken"], p["health"])
    for g in gt:
        if g["name"] == "DoomImp":
            births.setdefault(g["id"], (t, round(g["y"])))
        elif "Ball" in g["name"] and math.hypot(g["vx"], g["vy"]) > 1:
            balls.setdefault(g["id"], t)
end = S[-1][0]["tic"]
sp = sorted(b[0] for b in births.values())
per = []
for a in range(0, end, 2000):
    b = a + 2000
    expo = sum(max(0, min(b, end) - max(a, t0)) for t0, _ in births.values())
    n = sum(1 for t in balls.values() if a <= t < b)
    per.append(dict(tics=[a, b], imps_at_end=sum(1 for t0 in sp if t0 < b), shots=n, imp_tics=expo,
                    rate=n / max(expo, 1)))
off = [t for t in sp if t > 1 and (t - 1) % 60]
print(json.dumps(dict(end=end, n_imps=len(births), spawn_off_schedule=off[:20], max_imps=len(births),
                      spawn_tics_last=sp[-5:], per_period=per, damage_events=dmg, player_x_values=sorted(xs)[:10],
                      imp_ys_sample=sorted(y for _, y in births.values())[:60]), indent=0))
