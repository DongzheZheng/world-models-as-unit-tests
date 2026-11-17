"""Quick looks at life logs: end-segment orientation (clamped vs pinned), segment stretch, tool tracking."""
import sys
import numpy as np

if __name__ == "__main__":
    L = np.load(sys.argv[1])
    P = L["points"][:, :, :2]
    tp, A = L["tool_pos"], L["actions"]
    e = P[:, 19] - P[:, 18]
    ang = np.degrees(np.unwrap(np.arctan2(e[:, 1], e[:, 0])))
    e2 = P[:, 1] - P[:, 0]
    ang0 = np.degrees(np.unwrap(np.arctan2(e2[:, 1], e2[:, 0])))
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    d = np.diff(P[:, 1:], axis=1)
    a2 = np.arctan2(d[..., 1], d[..., 0])
    turn = np.degrees(np.angle(np.exp(1j * np.diff(np.arctan2(np.diff(P, axis=1)[..., 1], np.diff(P, axis=1)[..., 0]), axis=1))))
    for t in range(0, len(P), 4):
        print(f"{t:3d} lastseg {ang[t]:8.1f}  firstseg {ang0[t]:8.1f}  seg min/max {seg[t].min():.4f} {seg[t].max():.4f}  turn@18 {turn[t,17]:6.1f} turn@1 {turn[t,0]:6.1f} maxturn {np.abs(turn[t]).max():5.1f}")
