"""Energy / angular momentum digest of a life (per unit node mass), central differences."""
import sys
import numpy as np

if __name__ == "__main__":
    L = np.load(sys.argv[1])
    ev = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    P = L["points"][:, :, :2]
    tp, tv = L["tool_pos"], L["tool_vel"]
    V = (P[2:] - P[:-2]) * 25.0                         # central diff velocity at t=1..T-1
    Pm = P[1:-1]
    ke = 0.5 * (V[:, :19] ** 2).sum((1, 2))             # exclude the tool node
    rel = Pm[:, :19] - Pm[:, 19:20]
    vrel = V[:, :19] - V[:, 19:20]
    Lz = (rel[..., 0] * vrel[..., 1] - rel[..., 1] * vrel[..., 0]).sum(1)
    # curvature energy proxy: sum of squared turning angles
    d = np.diff(Pm, axis=1)
    ang = np.arctan2(d[..., 1], d[..., 0])
    turn = np.angle(np.exp(1j * np.diff(ang, axis=1)))
    bend = (turn ** 2).sum(1)
    tvd = np.diff(tp, axis=0) * 50
    print(" t   KE      Lz      bend   speedmax  tool_vel   d(tool)/dt")
    for i in range(0, len(V), ev):
        t = i + 1
        print(f"{t:3d} {ke[i]:.4f} {Lz[i]:+.4f} {bend[i]:.3f} {np.linalg.norm(V[i],axis=1).max():.3f}", tv[t], tvd[t - 1])
