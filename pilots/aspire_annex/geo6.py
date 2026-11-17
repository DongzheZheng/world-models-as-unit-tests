import sys, numpy as np
from pilots.aspire_annex import policy_v6 as m
np.set_printoptions(precision=3, suppress=True, linewidth=200)
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    L = []
    for i in range(200):
        p = m.plan(d["target"][i], d["post"][i]); L.append(p["L"] if p else 0)
    L = np.array(L); print("noplan", (L == 0).sum(), "L pct", np.percentile(L, [10, 50, 90]))
    for i in map(int, sys.argv[1:]):
        p = m.plan(d["target"][i], d["post"][i]); print(i, "tool0", d["tool_pos"][i], "tgt", d["target"][i], {k: np.round(v, 3) for k, v in p.items()})
