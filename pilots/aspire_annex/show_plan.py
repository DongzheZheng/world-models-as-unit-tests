import sys, importlib, numpy as np
np.set_printoptions(precision=3, suppress=True)
if __name__ == "__main__":
    m = importlib.import_module("pilots.aspire_annex.policy_" + sys.argv[1])
    d = np.load("runs/annex/observation/layouts.npz")
    for i in map(int, sys.argv[2:]):
        tool = d["tool_pos"][i]; c = d["points"][i, :, :2].mean(0) - tool
        p = m.plan(d["target"][i], d["post"][i], np.arctan2(c[1], c[0]))
        print(i, "tool0", tool, "rope_dir deg", np.degrees(np.arctan2(c[1], c[0])), "post", d["post"][i], "tgt", d["target"][i])
        print("   ", {k: (np.round(v, 3) if not np.isscalar(v) else round(float(v), 3)) for k, v in p.items()}, "u deg", np.degrees(np.arctan2(p['u'][1], p['u'][0])))
        print("   tip0", d["points"][i, 0, :2])
