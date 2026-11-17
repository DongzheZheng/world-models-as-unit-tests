"""Static checks of a policy version on snapshot situations: planned path length, clearance, first commands."""
import importlib
import sys
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    ver = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or [0]
    mod = importlib.import_module(f"pilots.aspire_station.policy_v{ver}")
    L = np.load("runs/station/observation/layouts.npz")
    for i in seeds:
        obs = dict(points=L["points"][i], tool_pos=L["tool_pos"][i], tool_vel=np.zeros(2), posts=L["posts"][i],
                   post_sides=L["post_sides"][i], target=L["target"][i], load=float(L["load"][i]), t=0)
        pol = mod.make_policy()
        pol.reset()
        c = pol.act(obs)
        P = pol.P
        dmin = np.min(np.linalg.norm(P[:, None] - L["posts"][i][None], axis=2))
        print(i, "path len", round(pol.s[-1], 3), "min post dist", round(dmin, 3), "first cmd", c, "tool", obs["tool_pos"],
              "posts", L["posts"][i].ravel(), "sides", L["post_sides"][i], "target", L["target"][i])
        print("   path every 20:", P[::20].ravel())
