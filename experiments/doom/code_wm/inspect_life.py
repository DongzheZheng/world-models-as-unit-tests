"""Dump a readable trace of a black-box life.  usage: python experiments/doom/code_wm/inspect_life.py <k> [every]"""
import gzip
import json
import sys
from collections import Counter

k = int(sys.argv[1])
every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
recs = [json.loads(l) for l in gzip.open(f"runs/doom/blackbox/take_cover/code_wm/life_{k}.jsonl.gz", "rt")]
print(recs[0])
r0 = recs[1]["reset"]
names = Counter()
for i, r in enumerate(recs[1:]):
    o = r["obs"] if "obs" in r else r["reset"]["obs"]
    gt = r["info"]["gt_objects"] if "info" in r else r0["info"]["gt_objects"]
    for g in gt:
        names[g["name"]] += 1
    if i % every and not r.get("terminated"):
        continue
    p = o["player"]
    s = f"{i:3d} t{o['tic']:4d} a{r.get('action', '-')} rew{r.get('reward', 0):3.0f} P({p['x']:7.1f},{p['y']:6.1f}) v({p['vx']:5.2f},{p['vy']:5.2f}) ang{p['angle']:.0f} hp{p['health']:.0f} hits{p['hits_taken']}"
    objs = " | ".join(f"{g['name'][:6]}#{g['id']}({g['x']:.0f},{g['y']:.0f},{g['z']:.0f} v{g['vx']:.1f},{g['vy']:.1f} {'V' if g['visible'] else '-'})" for g in gt)
    print(s, "::", objs)
print(names)
print("depth sample", r0["obs"]["depth_scan"][:64])
