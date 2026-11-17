import numpy as np
from pilots.aspire_annex import policy_v2 as m
np.set_printoptions(precision=3, suppress=True)
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    rows = []
    for i in range(200):
        tool = d["tool_pos"][i]; P = d["points"][i, :, :2]; c = P.mean(0) - tool; r = np.arctan2(c[1], c[0])
        p = m.plan(d["target"][i], d["post"][i], r)
        rv = np.array([np.cos(r), np.sin(r)])
        slack = max(0.0, float((p["S"] - tool) @ rv))
        chord = np.linalg.norm(P[0] - P[19])
        rows.append((i, np.degrees(p["mis"]), p["s"], slack, chord))
    R = np.array(rows)
    print("all: mis<10deg", (R[:, 1] < 10).mean(), "slack median", np.median(R[:, 3]))
    ok = R[(R[:, 1] < 8) & (R[:, 2] >= 0.18)]
    for r in ok[np.argsort(ok[:, 3])][:10]: print(r)
