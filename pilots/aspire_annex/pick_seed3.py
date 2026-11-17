import numpy as np
from pilots.aspire_annex import policy_v3 as m
np.set_printoptions(precision=3, suppress=True)
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    rows = []
    for i in range(200):
        tool = d["tool_pos"][i]; P = d["points"][i, :, :2]; c = P.mean(0) - tool; r = np.arctan2(c[1], c[0])
        p = m.plan(d["target"][i], d["post"][i], tool, r)
        rows.append((i, np.degrees(p["mis"]), p["s"], p["slack"], np.linalg.norm(P[0] - P[19]), p["R"]))
    R = np.array(rows)
    print("pct mis", np.percentile(R[:, 1], [25, 50, 75]), "stroke", np.percentile(R[:, 2], [25, 50, 75]), "slack", np.percentile(R[:, 3], [25, 50, 75]))
    good = R[(R[:, 1] < 25) & (R[:, 2] > 0.2) & (R[:, 3] < 0.1)]
    print(len(good)); 
    for r in good[:15]: print(r)
