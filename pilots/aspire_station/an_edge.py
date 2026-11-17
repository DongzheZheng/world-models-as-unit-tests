"""Edge situations in the snapshots: start near the anchor, target near a post."""
import numpy as np

if __name__ == "__main__":
    L = np.load("runs/station/observation/layouts.npz")
    da = np.linalg.norm(L["tool_pos"] - np.array([0.05, 0.0]), axis=1)
    dt = np.min(np.linalg.norm(L["target"][:, None] - L["posts"], axis=2), axis=1)
    print("start nearest anchor:", [(int(i), round(float(da[i]), 3)) for i in np.argsort(da)[:4]])
    print("target nearest a post:", [(int(i), round(float(dt[i]), 3)) for i in np.argsort(dt)[:4]])
