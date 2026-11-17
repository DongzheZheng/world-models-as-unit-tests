"""Distrust check: pusher speed while in contact, real L lives vs imagined traces."""
import glob, sys
import numpy as np
import shapely.geometry as sg
from pilots.aspire_pusht import policy_v17 as m

def speeds(files):
    pol = m.make_policy(); pol.bind({"goal_pose": [300, 200, np.pi / 2], "object": "L"}); out = []
    for f in files:
        d = np.load(f); P, Q = d["block_pose"], d["agent_pos"]
        for t in range(len(Q) - 1):
            if pol.poly(P[t]).distance(sg.Point(Q[t])) < 16.5:
                out.append(np.linalg.norm(Q[t + 1] - Q[t]))
    out = np.array(out); return len(out), np.percentile(out, [50, 90, 99]).round(1), out.max().round(1)

if __name__ == "__main__":
    print("real L lives", speeds(sorted(glob.glob("runs/pusht/blackbox_L/aspire_style*/life_*.npz"))))
    print("imagined v17", speeds(sorted(glob.glob(sys.argv[1] + "/cw_*.npz"))))
