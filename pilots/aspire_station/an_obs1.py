"""Offline analysis of the free observation material (no env, no processes)."""
import json
import numpy as np

OBS = "runs/station/observation"

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=150)
    L = np.load(f"{OBS}/layouts.npz")
    P = L["points"]; tool = L["tool_pos"]; posts = L["posts"]; tgt = L["target"]; load = L["load"]
    print("layouts keys", list(L.keys()))
    print("load at rest: min/mean/max", load.min(), load.mean(), load.max())
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    print("segment length stats (3d)", seg.min(), seg.mean(), seg.max(), "per-index mean", seg.mean(0))
    print("total polyline length: min/mean/max", seg.sum(1).min(), seg.sum(1).mean(), seg.sum(1).max())
    print("z stats per index mean", P[:, :, 2].mean(0))
    print("point0 xy min/max", P[:, 0, :2].min(0), P[:, 0, :2].max(0), "std", P[:, 0, :2].std(0))
    print("point19 vs tool max diff", np.abs(P[:, 19, :2] - tool).max())
    print("tool xy min/max", tool.min(0), tool.max(0))
    print("target min/max", tgt.min(0), tgt.max(0))
    print("posts x min/max", posts[..., 0].min(), posts[..., 0].max(), "y", posts[..., 1].min(), posts[..., 1].max())
    print("post sides counts", np.unique(L["post_sides"], return_counts=True))
    d01 = np.linalg.norm(P[:, 0, :2] - tool, axis=1)
    print("anchor-tool distance min/mean/max", d01.min(), d01.mean(), d01.max())
    for s in range(3):
        print("seed", s, "points\n", P[s], "\ntool", tool[s], "posts", posts[s], "sides", L["post_sides"][s], "target", tgt[s], "load", load[s])
    # sort posts by x?
    print("posts sorted by x?", np.all(np.diff(posts[..., 0], axis=1) > 0))
    print("post distance to anchor-tool line etc: post x per order mean", posts[..., 0].mean(0), posts[..., 1].mean(0))
    idx = json.load(open(f"{OBS}/index.json"))
    for e in idx["episodes"]:
        s = e["seed"]
        d = np.load(f"{OBS}/episode_{s}.npz", allow_pickle=True)
        pts = d["points"]; ld = d["load"]; tp = d["tool_pos"]; tv = d["tool_vel"]; a = d["actions"]
        sp = np.linalg.norm(tv, axis=1)
        L3 = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(1)
        print(f"--- ep {s} {e['outcome']} steps {len(a)} maxload {ld.max():.2f} speed max {sp.max():.3f} "
              f"anchor moved {np.abs(pts[:,0]-pts[0,0]).max():.4f} len0 {L3[0]:.3f} lenmax {L3.max():.3f} "
              f"sides_ok_end {d['sides_ok']} post_sides {d['post_sides'][0]} sides_now_end {d['sides_now'][-1]}")
        print("   posts", d["posts"][0].ravel(), "target", d["target"][0], "tool0", tp[0], "anchor", pts[0, 0])
