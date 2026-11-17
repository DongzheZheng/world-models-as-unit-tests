# Observation-material analysis (pure numpy, reads only runs/station/observation).
import numpy as np, json
D = "runs/station/observation/"
np.set_printoptions(precision=4, suppress=True, linewidth=160)

L = np.load(D + "layouts.npz")
P = L["points"]
print("layout keys", list(L.keys()))
seg = np.linalg.norm(np.diff(P, axis=1), axis=2)            # (200,19)
print("segment lengths mean per index", seg.mean(0))
print("segment len min/max", seg.min(), seg.max(), "total len mean/std", seg.sum(1).mean(), seg.sum(1).std())
print("z stats", P[..., 2].min(), P[..., 2].max(), P[..., 2].mean(0))
print("point0 x,y range", P[:, 0, :2].min(0), P[:, 0, :2].max(0), "std", P[:, 0, :2].std(0))
print("point19 vs tool", np.abs(P[:, 19, :2] - L["tool_pos"]).max())
print("tool range", L["tool_pos"].min(0), L["tool_pos"].max(0))
print("end-to-end dist", np.linalg.norm(P[:, 19, :2] - P[:, 0, :2], axis=1).min(), np.linalg.norm(P[:, 19, :2] - P[:, 0, :2], axis=1).max())
print("load at reset", L["load"].min(), L["load"].max(), L["load"].mean())
print("posts range", L["posts"].reshape(-1, 2).min(0), L["posts"].reshape(-1, 2).max(0))
print("target range", L["target"].min(0), L["target"].max(0))
print("post_sides values", np.unique(L["post_sides"], return_counts=True))
# chord/straightness
print("example points seed0\n", P[0])
print("posts0", L["posts"][0], "sides", L["post_sides"][0], "target", L["target"][0], "tool", L["tool_pos"][0])
# order of posts by x
print("posts x sorted?", np.mean(np.all(np.diff(L["posts"][:, :, 0], axis=1) > 0, axis=1)))
# initial sides
from math import pi
def winding(pts, posts):
    out = []
    for c in posts:
        a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]); da = (np.diff(a) + pi) % (2 * pi) - pi; out.append(da.sum())
    return np.array(out)
W = np.array([winding(P[i], L["posts"][i]) for i in range(200)])
print("initial winding range", W.min(), W.max())
# min distance points to posts at reset
dpp = np.linalg.norm(P[:, :, None, :2] - L["posts"][:, None], axis=3)
print("min point-post distance", dpp.min())

for s in range(900, 910):
    E = np.load(D + f"episode_{s}.npz")
    if s == 900:
        print("episode keys", list(E.keys()))
    pts = E["points"]; T = len(E["actions"])
    sg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
    tv = np.linalg.norm(E["tool_vel"], axis=1)
    cmdstep = np.linalg.norm(np.diff(E["actions"], axis=0), axis=1)
    p0move = np.abs(pts[:, 0] - pts[0, 0]).max()
    print(f"seed {s} T={T} succ={E['success']} over={E['overload']} sides={E['post_sides']} -> {E['sides_now'][-1]} "
          f"seglen min/max {sg.min():.4f}/{sg.max():.4f}  z max {pts[...,2].max():.4f} tool|v| max {tv.max():.3f} "
          f"p0 move {p0move:.4f} loadmax {E['load'].max():.2f} cmd step med {np.median(cmdstep):.4f}")
