"""Print a compact digest of a logged life.  usage: python inspect_life.py <npz> [every]"""
import sys
import numpy as np

if __name__ == "__main__":
    L = np.load(sys.argv[1])
    ev = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    P, tp, tv, A = L["points"], L["tool_pos"], L["tool_vel"], L["actions"]
    print("seed", L["seed"], "n", L["n_steps"], "succ", L["success"], "final tip", L["final_tip_dist"], "target", L["target"][0], "post", L["post"][0])
    V = np.diff(P, axis=0) * 50
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    print("z range", P[..., 2].min(), P[..., 2].max(), "seg range", seg.min(), seg.max())
    print(" t   cmd            tool          toolvel        p1              |v|max  |v|p1  zmax   segmax")
    for t in range(0, len(A), ev):
        sp = np.linalg.norm(V[t, :, :2], axis=1)
        print(f"{t:3d}", A[t], tp[t + 1], tv[t + 1], P[t + 1, 0], f"{sp.max():.3f} {sp[0]:.3f} {P[t+1,:,2].max():.4f} {seg[t+1].max():.4f}")
    print("final points\n", P[-1])
