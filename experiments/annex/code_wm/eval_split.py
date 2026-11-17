"""Failure split of an evaluation: success / at rest outside the target at the end / still moving at the time limit.
usage: python eval_split.py <log_dir>"""
import glob
import json
import sys

import numpy as np

if __name__ == "__main__":
    out = dict(success=0, rest_outside=0, moving_at_limit=0, n=0, tip_dist_failures=[], ever_at_rest_outside=0)
    lim = 0.02 / 50
    for p in sorted(glob.glob(sys.argv[1] + "/ep_*.npz")):
        L = np.load(p)
        P, tl = L["points"][:, :, :2], L["tool_pos"]
        d = np.linalg.norm(P[-1, 0] - L["target"][-1])
        rest = lambda t: (np.linalg.norm(P[t] - P[t - 1], axis=1).max() <= lim) and (np.linalg.norm(tl[t] - tl[t - 1]) <= lim)
        n = len(L["actions"])
        out["n"] += 1
        ever = any(rest(t) for t in range(1, n + 1))
        if n < 236 or (rest(n) and d <= 0.1):
            out["success"] += 1
        elif rest(n):
            out["rest_outside"] += 1
            out["tip_dist_failures"].append(float(d))
        else:
            out["moving_at_limit"] += 1
            out["tip_dist_failures"].append(float(d))
        out["ever_at_rest_outside"] += int(ever and n == 236)
    out["tip_dist_failures_median"] = float(np.median(out["tip_dist_failures"])) if out["tip_dist_failures"] else None
    print(json.dumps(out))
