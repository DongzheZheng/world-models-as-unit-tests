"""Print rope shapes (xy of every point) and per-point speeds at chosen times: ap_shapes.py <npz> t1 t2 ..."""
import sys, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
if __name__ == "__main__":
    L = np.load(sys.argv[1]); P = L["points"]
    V = np.r_[np.zeros((1, 20, 3)), np.diff(P, axis=0) * 50]
    for t in map(int, sys.argv[2:]):
        print(f"--- t={t} tool {L['tool_pos'][t]}")
        print(" x ", P[t, :, 0]); print(" y ", P[t, :, 1]); print(" z ", P[t, :, 2])
        print(" vx", V[t, :, 0]); print(" vy", V[t, :, 1])
        seg = np.linalg.norm(np.diff(P[t], axis=0), axis=1); print(" seg", seg)
