"""Rank snapshot situations by geometric difficulty for a policy version's path (pure geometry)."""
import importlib
import sys
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    ver = sys.argv[1]
    mod = importlib.import_module(f"pilots.aspire_station.policy_v{ver}")
    L = np.load("runs/station/observation/layouts.npz")
    rows = []
    for i in range(200):
        posts, sides, tgt, tool = L["posts"][i], L["post_sides"][i], L["target"][i], L["tool_pos"][i]
        P = mod.plan_path(tool, posts, sides, tgt)
        seg = np.diff(P, axis=0)
        ang = np.arctan2(seg[:, 1], seg[:, 0])
        turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi).sum()
        dmin = np.min(np.linalg.norm(P[:, None] - posts[None], axis=2))
        gaps = [np.linalg.norm(posts[a] - posts[b]) for a in range(3) for b in range(a + 1, 3)]
        rows.append((i, np.linalg.norm(seg, axis=1).sum(), turn, dmin, np.abs(P[:, 1]).max(), min(gaps)))
        poly = np.concatenate([L["points"][i][:, :2], P[1:]])
        pts = poly
        W = []
        for c in posts:
            a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            W.append((((np.diff(a) + np.pi) % (2 * np.pi)) - np.pi).sum())
        W = np.array(W)
        s = np.where(W >= np.pi / 2, 1, np.where(W <= -np.pi / 2, -1, 0))
        if not np.all(s == sides):
            print("SIDES WRONG", i, W.round(2), sides)
    R = np.array(rows)
    print("path len pct", np.percentile(R[:, 1], [50, 90, 100]).round(3), "turn pct", np.percentile(R[:, 2], [50, 90, 100]).round(2))
    print("min post clearance overall", R[:, 3].min().round(3), "max |y|", R[:, 4].max().round(3), "min post gap", R[:, 5].min().round(3))
    score = R[:, 2] / np.median(R[:, 2]) + R[:, 1] / np.median(R[:, 1])
    for j in np.argsort(-score)[:8]:
        print("hard", R[j].round(3), "sides", L["post_sides"][int(R[j, 0])], "posts", L["posts"][int(R[j, 0])].ravel())
    for j in np.argsort(-R[:, 1])[:5]:
        print("long", R[j].round(3))
    for j in np.argsort(R[:, 5])[:5]:
        print("tight gap", R[j].round(3))
