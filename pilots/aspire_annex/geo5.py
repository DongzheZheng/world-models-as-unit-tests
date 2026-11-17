import sys, numpy as np
from pilots.aspire_annex import policy_v5 as m
np.set_printoptions(precision=3, suppress=True, linewidth=200)
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    ok, Es, T = 0, [], []
    for i in range(0, 200, 4):
        p = m.plan(d["target"][i], d["post"][i], d["tool_pos"][i])
        if p is None: continue
        ok += 1; Es.append(p["E"]); T.append(p["send"] / 0.5 + p["E"] / 0.5)
    print("plans", ok, "/50; E pct", np.percentile(Es, [10, 50, 90]), "time s pct", np.percentile(T, [50, 90]))
    for i in map(int, sys.argv[1:]):
        p = m.plan(d["target"][i], d["post"][i], d["tool_pos"][i])
        print(i, "tool0", d["tool_pos"][i], "post", d["post"][i], "tgt", d["target"][i])
        if p: print("  X", p["X"], "u", p["u"], "A", p["A"], "E", round(p["E"], 3), "clear", round(p["clear"], 3), "rho", p["rho"], "Lb", p["Lb"], "side", p["side"], "sX", round(p["sX"], 3), "send", round(p["send"], 3)); print("  track every 20:", p["T"][::20])
