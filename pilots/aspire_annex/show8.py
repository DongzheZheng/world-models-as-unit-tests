import sys, numpy as np
from pilots.aspire_annex import policy_v8 as m
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz"); ms = []
    for i in range(200):
        c = m.candidates(d["target"][i], d["post"][i]); 
        if not c: ms.append(np.nan); continue
        b = m.choose(c, m.near_dir(d["points"][i, :, :2])); ms.append(abs(b["mis"]))
    ms = np.array(ms); print("no cand", np.isnan(ms).sum(), "|mis| pct (at t0)", np.nanpercentile(ms, [10, 25, 50, 75, 90]))
    for i in map(int, sys.argv[1:]):
        c = m.candidates(d["target"][i], d["post"][i]); b = m.choose(c, m.near_dir(d["points"][i, :, :2]))
        print(i, "tool0", d["tool_pos"][i], "near", round(m.near_dir(d["points"][i, :, :2]), 1), "F", b["F"], "angd", b["angd"], "L", b["L"], "mis", round(b["mis"], 1))
    good = [i for i in range(200) if not np.isnan(ms[i]) and ms[i] < 8]
    print("seeds with |mis|<8:", good[:30])
