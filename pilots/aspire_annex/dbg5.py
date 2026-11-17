import numpy as np
from pilots.aspire_annex import policy_v5 as m
np.set_printoptions(precision=3, suppress=True, linewidth=200)
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    i = 3; tgt, post, tool0 = d["target"][i], d["post"][i], d["tool_pos"][i]
    Q = np.array([0.54, -0.1]); u = (tgt - Q) / np.linalg.norm(tgt - Q)
    lat = np.array([-u[1], u[0]])
    for side in (-1, 1):
        l = side * lat; L1 = 0.0
        while m.inzone(Q - (L1 + 0.01) * u, 0.02) and m.inzone(Q - (L1 + 0.01) * u + 2 * 0.09 * l, 0.02): L1 += 0.01
        print("side", side, "L1", L1)
        B = Q - L1 * u; Cc = B + 0.09 * l
        arc = [Cc + 0.09 * (np.cos(t) * (-l) + np.sin(t) * (-u)) for t in np.linspace(0, np.pi, 13)]
        W = [tool0, Q + 0.08 * l, Q, Q - 0.05 * u, B] + arc + [arc[-1] + 0.6 * u]
        P = m.resample(m.chaikin(W, 2)); s = m.arclen(P)
        iX = int(np.argmin(np.linalg.norm(P - (Q - 0.05 * u), axis=1)))
        print(" sX", s[iX], "total", s[-1], "arc end", arc[-1], "outside:", [p for p in P[s <= s[iX] + 0.7] if not m.inzone(p, -0.005)][:3])
