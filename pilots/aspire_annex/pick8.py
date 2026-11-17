import numpy as np
from pilots.aspire_annex import policy_v17 as m
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    rows = []
    for i in range(200):
        c = m.candidates(d["target"][i], d["post"][i])
        if not c: continue
        P = d["points"][i, :, :2]; b = m.choose(c, m.near_dir(P))
        rows.append((i, abs(b["mis"]), np.linalg.norm(b["F"] - d["tool_pos"][i]), b["L"], np.linalg.norm(P[0] - P[19]), b["angd"]))
    R = np.array(rows); R = R[(R[:, 1] < 10)]
    for r in R[np.argsort(R[:, 2])][:10]: print(np.round(r, 3))
