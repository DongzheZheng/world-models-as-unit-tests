import sys, importlib, numpy as np
np.set_printoptions(precision=3, suppress=True)
mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "pilots.aspire_annex.policy_v1")
d = np.load("runs/annex/observation/layouts.npz")
fails, S = 0, []
for i in range(200):
    p = mod.plan(d["target"][i], d["post"][i])
    if p is None:
        fails += 1; continue
    S.append((i, p[4], p[5]))
S = np.array(S)
print("no plan:", fails, "stroke min/mean", S[:, 1].min(), S[:, 1].mean(), "clear min", S[:, 2].min())
for i in [0, 1, 3, 5, 7, 9, 11]:
    print(i, "tool0", d["tool_pos"][i], "post", d["post"][i], "tgt", d["target"][i], "plan", [np.round(x, 3) for x in (mod.plan(d['target'][i], d['post'][i]) or [None])])
