import sys, numpy as np
from pilots.aspire_annex import policy_v4 as m
np.set_printoptions(precision=3, suppress=True, linewidth=200)
if __name__ == "__main__":
    d = np.load("runs/annex/observation/layouts.npz")
    rows = []
    for i in range(200):
        tool = d["tool_pos"][i]; P = d["points"][i, :, :2]
        r = m.rope_dir(P, tool)
        p = m.plan(d["target"][i], d["post"][i], tool, r)
        if p is None: rows.append((i, np.nan, 0, 0, 0, 0)); continue
        rows.append((i, p["err"], p["Lperp"], np.degrees(p["rot"]), p["slack"], np.linalg.norm(P[0] - P[19])))
    R = np.array(rows)
    print("no plan", np.isnan(R[:, 1]).sum(), "err pct", np.nanpercentile(R[:, 1], [50, 90]), "Lperp pct", np.percentile(R[:, 2], [10, 50, 90]), "rot pct", np.percentile(R[:, 3], [10, 50, 90]), "slack", np.percentile(R[:, 4], [50, 90]))
    for i in map(int, sys.argv[1:]):
        tool = d["tool_pos"][i]; P = d["points"][i, :, :2]; r = m.rope_dir(P, tool)
        p = m.plan(d["target"][i], d["post"][i], tool, r)
        print(i, "tool0", tool, "r", np.degrees(r), "post", d["post"][i], "tgt", d["target"][i], "chord", np.linalg.norm(P[0]-P[19]))
        print("  ", {k: np.round(np.degrees(v) if k in ("rot", "th") else v, 3) for k, v in p.items()})
    good = R[(R[:, 2] > 0.25) & (R[:, 4] < 0.05)]
    print("good seeds:", good[:12])
