import numpy as np
from pilots.aspire_annex import policy_v5 as m
np.set_printoptions(precision=3, suppress=True, linewidth=200)
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    i = 3; tgt, post, tool0 = d["target"][i], d["post"][i], d["tool_pos"][i]
    c = m.plan_aims(tgt, post, tool0)
    Qs = sorted(set((round(z[1][0][0], 2), round(z[1][0][1], 2)) for z in c)); print("Q set", Qs[:5], "...", len(Qs))
    Q = np.array([0.53, -0.17]); A = tgt; u = (A - Q) / np.linalg.norm(A - Q); lat = np.array([-u[1], u[0]])
    print("u", u, "clear", m.seg_dist(post, Q - 0.05 * u, A))
    for rho in (0.17, 0.15):
        for side in (-1, 1):
            l = side * lat; L1 = 0.05
            while m.inzone(Q - (L1 + 0.01) * u, 0.012) and m.inzone(Q - (L1 + 0.01) * u + 2 * rho * l, 0.012): L1 += 0.01
            B = Q - L1 * u; Cc = B + rho * l
            arc = [Cc + rho * (np.cos(t) * (-l) + np.sin(t) * (-u)) for t in np.linspace(0, np.pi, 13)]
            W = [tool0, np.clip(Q + 0.08 * l + 0.03 * u, m.ZLO, m.ZHI), Q, Q - 0.05 * u, B] + arc + [arc[-1] + 0.6 * u]
            P = m.resample(m.chaikin(W, 2)); s = m.arclen(P); iX = int(np.argmin(np.linalg.norm(P - (Q - 0.05 * u), axis=1)))
            PP = P[s <= s[iX] + 0.7]; bad = [p for p in PP if not m.inzone(p, -0.004)]
            print("rho", rho, "side", side, "L1", L1, "B", B, "arc0", arc[0], "arcend", arc[-1], "nbad", len(bad), bad[:1], bad[-1:] if bad else "", "end", PP[-1])
