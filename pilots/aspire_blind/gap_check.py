"""Static: largest arc-length gap between consecutive path samples (carrot stalls if gap > lead=0.03)."""
import importlib
import sys
import numpy as np

if __name__ == "__main__":
    mod = importlib.import_module(f"pilots.aspire_blind.{sys.argv[1]}")
    d = np.load("runs/station/observation/layouts.npz")
    gaps = []
    for i in range(200):
        pts, S, mode = mod.plan(d["tool_pos"][i], d["posts"][i], d["post_sides"][i], d["target"][i])
        gaps.append(np.max(np.diff(S)))
    gaps = np.array(gaps)
    print("max sample gap: max %.3f  n>0.03: %d  seeds %s" % (gaps.max(), (gaps > 0.03).sum(), np.where(gaps > 0.03)[0]))
