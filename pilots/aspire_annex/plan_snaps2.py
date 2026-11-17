import numpy as np
from pilots.aspire_annex import policy_v1 as m
np.set_printoptions(precision=3, suppress=True)
d = np.load("runs/annex/observation/layouts.npz")
for clear in (0.035, 0.045):
    for reach in (0.60, 0.66):
        S = []
        for i in range(200):
            p = m.plan(d["target"][i], d["post"][i], reach=reach, clear=clear, aim_off=0.05)
            S.append(p[4] if p else 0)
        S = np.array(S); print("clear", clear, "reach", reach, "noplan", (S == 0).sum(), "stroke mean", S.mean(), "frac>=0.15", (S >= 0.15).mean())
P = d["points"]; tool = d["tool_pos"]
ch = np.linalg.norm(P[:, 0, :2] - P[:, 19, :2], axis=1)
cands = []
for i in range(200):
    p = m.plan(d["target"][i], d["post"][i], reach=0.63, clear=0.035, aim_off=0.05)
    if p and abs(d["target"][i][1]) < 0.12:
        cands.append((i, p[4], ch[i], P[i, 0, 0] - tool[i, 0]))
for c in sorted(cands, key=lambda c: -c[1] - c[2])[:10]:
    print(c, m.plan(d["target"][c[0]], d["post"][c[0]], reach=0.63, clear=0.035, aim_off=0.05)[:4], "tool0", tool[c[0]])
