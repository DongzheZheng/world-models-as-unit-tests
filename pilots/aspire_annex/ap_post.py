"""Point-to-post distances and chord per step: ap_post.py <npz> t0 t1"""
import sys, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
if __name__ == "__main__":
    L = np.load(sys.argv[1]); P = L["points"]; post = L["post"][0]
    for t in range(int(sys.argv[2]), int(sys.argv[3])):
        dp = np.linalg.norm(P[t, :, :2] - post, axis=1)
        print(t, "argmin pt", int(dp.argmin()), "min d", round(float(dp.min()), 3), "z of that pt", round(float(P[t, dp.argmin(), 2]), 3), "chord", round(float(np.linalg.norm(P[t, 0, :2] - P[t, 19, :2])), 3), "z", P[t, :, 2][::3])
