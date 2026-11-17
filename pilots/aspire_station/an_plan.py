"""Run the path geometry on the 200 snapshots and the 10 observation situations: sides, clearance, taut length."""
import json
import numpy as np
from pilots.aspire_station import geom

OBS = "runs/station/observation"

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    L = np.load(f"{OBS}/layouts.npz")
    bad, tl, plen, wmin = [], [], [], []
    for i in range(200):
        pts, posts, sides, tgt, tool = L["points"][i], L["posts"][i], L["post_sides"][i], L["target"][i], L["tool_pos"][i]
        P = geom.plan_path(tool, posts, sides, tgt)
        poly = np.concatenate([pts[:, :2], P[1:]])
        w = geom.winding_of(poly, posts)
        s = np.where(w >= np.pi / 2, 1, np.where(w <= -np.pi / 2, -1, 0))
        ok = np.all(s == sides)
        t_len, _ = geom.taut_length(poly, posts)
        tl.append(t_len); plen.append(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        wmin.append(np.min(np.abs(w)))
        cl = geom.dist_to_posts(P, posts).min()
        if not ok or cl < 0.055:
            bad.append((i, ok, w.round(2), sides, round(cl, 3)))
    tl = np.array(tl)
    print("bad", len(bad)); [print(b) for b in bad[:20]]
    print("taut length min/med/max", tl.min(), np.median(tl), tl.max(), "n>0.95", (tl > 0.95).sum(), "n>0.90", (tl > 0.90).sum())
    print("path length min/med/max", min(plen), np.median(plen), max(plen))
    print("min |W| over posts: min/med", min(wmin), np.median(wmin))
    print("slack percentiles (0.993 - taut)", np.percentile(0.993 - tl, [0, 5, 25, 50, 75, 100]))
    idx = json.load(open(f"{OBS}/index.json"))
    for e in idx["episodes"]:
        d = np.load(f"{OBS}/episode_{e['seed']}.npz", allow_pickle=True)
        pts, posts, sides, tgt = d["points"][0], d["posts"][0], d["post_sides"][0], d["target"][0]
        P = geom.plan_path(d["tool_pos"][0], posts, sides, tgt)
        poly = np.concatenate([pts[:, :2], P[1:]])
        t_len, _ = geom.taut_length(poly, posts)
        # taut length of actual final rope
        fin, _ = geom.taut_length(d["points"][-1][:, :2], posts)
        print(e["seed"], e["outcome"], "planned taut", round(t_len, 3), "final rope taut", round(fin, 3),
              "sides", geom.winding_of(poly, posts).round(2), sides)
