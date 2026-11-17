import sys, numpy as np
from pilots.aspire_annex import policy_v3 as m
np.set_printoptions(precision=3, suppress=True, linewidth=200)
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    for i in map(int, sys.argv[1:]):
        tool = d["tool_pos"][i]; P = d["points"][i, :, :2]; c = P.mean(0) - tool
        p = m.plan(d["target"][i], d["post"][i], tool, np.arctan2(c[1], c[0]))
        print(i, "tool0", tool, "post", d["post"][i], "tgt", d["target"][i], "rope_dir", np.degrees(np.arctan2(c[1], c[0])))
        print("  ", {k: np.round(v, 3) for k, v in p.items()}, "u deg", np.degrees(np.arctan2(p["u"][1], p["u"][0])))
        print("   rope x", P[:, 0]); print("   rope y", P[:, 1])
