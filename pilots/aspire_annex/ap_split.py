"""Outcome split of an eval log dir (reporting only): success / at rest outside target / still moving at time limit."""
import sys, json, numpy as np
from pathlib import Path
if __name__ == "__main__":
    out = {"success": 0, "at_rest_outside": 0, "moving_at_limit": 0}
    ahead = 0
    for f in sorted(Path(sys.argv[1]).glob("ep_*.npz")):
        L = np.load(f); P = L["points"][:, :, :2]; tool = L["tool_pos"]
        d = np.linalg.norm(P[-1, 0] - L["target"][-1])
        rest = np.linalg.norm(P[-1] - P[-2], axis=1).max() <= 0.0004 and np.linalg.norm(tool[-1] - tool[-2]) <= 0.0004
        if len(L["actions"]) < 236: out["success"] += 1
        elif rest: out["at_rest_outside"] += 1
        else: out["moving_at_limit"] += 1
        ahead += int(P[-1, 0, 0] > tool[-1, 0] + 0.2)
    out["tip_ahead_of_tool_by_0.2m_in_x"] = ahead
    print(json.dumps(out))
