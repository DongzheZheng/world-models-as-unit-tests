# Which kinematic quantity of the tracked points explains the load reading?  (observation logs only)
import numpy as np, sys, glob
np.set_printoptions(precision=4, suppress=True, linewidth=200)
files = sys.argv[1:] or sorted(glob.glob("runs/station/observation/episode_*.npz"))
rows = []
for f in files:
    E = np.load(f)
    pts, tp, ld, A = E["points"][..., :2], E["tool_pos"], E["load"], E["actions"]
    posts = E["posts"][0]
    T = len(A)
    for t in range(1, T + 1):
        v = (pts[t] - pts[t - 1]) / 0.05                 # mean node velocity over the step (20,2)
        vprev = (pts[t - 1] - pts[t - 2]) / 0.05 if t >= 2 else np.zeros_like(v)
        sp = np.linalg.norm(v, axis=1)
        rows.append(dict(
            load=ld[t], vec_sum=np.linalg.norm(v[1:].sum(0)), abs_sum=sp[1:].sum(), sq_sum=(sp[1:] ** 2).sum(),
            tool_sp=sp[19], acc=np.linalg.norm(v[1:] - vprev[1:], axis=1).sum() / 0.05,
            n_moving=(sp[1:] > 0.01).sum(), coul=np.linalg.norm((v[1:] / np.maximum(sp[1:, None], 1e-3) * (sp[1:, None] > 0.005)).sum(0)),
            dtool=np.linalg.norm(tp[t][None] - posts, axis=1).min(), e2e=np.linalg.norm(pts[t, 19] - pts[t, 0]),
            last=(t == T), f=f[-8:-4]))
keys = ["vec_sum", "abs_sum", "sq_sum", "tool_sp", "n_moving", "coul", "acc"]
L = np.array([r["load"] for r in rows]); M = np.array([[r[k] for k in keys] for r in rows])
keep = np.array([not r["last"] for r in rows])
print("N", keep.sum())
for i, k in enumerate(keys):
    x = M[keep, i]; c = np.corrcoef(x, L[keep])[0, 1]
    # fit load = a + b x
    X = np.c_[np.ones_like(x), x]; w, *_ = np.linalg.lstsq(X, L[keep], rcond=None)
    r = L[keep] - X @ w
    print(f"{k:10s} corr {c:.3f}  fit load={w[0]:.3f}+{w[1]:.4f}*x  rmse {np.sqrt((r**2).mean()):.3f}")
for combo in [["vec_sum", "acc"], ["abs_sum", "acc"], ["vec_sum", "sq_sum"], ["vec_sum", "abs_sum", "sq_sum", "acc"]]:
    idx = [keys.index(k) for k in combo]
    X = np.c_[np.ones(keep.sum()), M[keep][:, idx]]; w, *_ = np.linalg.lstsq(X, L[keep], rcond=None)
    r = L[keep] - X @ w
    print(combo, "w", w, "rmse", np.sqrt((r ** 2).mean()))
# no-intercept vec_sum
x = M[keep, 0]; b = (x @ L[keep]) / (x @ x); print("load = b*vec_sum, b=", b, "rmse", np.sqrt(((L[keep] - b * x) ** 2).mean()))
# binned
x = M[keep, 0]
for lo in np.arange(0, 3.0, 0.2):
    m = (x >= lo) & (x < lo + 0.2)
    if m.sum():
        print(f"vec_sum [{lo:.1f},{lo+0.2:.1f}) n={m.sum():4d} load mean {L[keep][m].mean():.3f} sd {L[keep][m].std():.3f}")
print("last steps:")
for r in rows:
    if r["last"]:
        print({k: (round(float(v), 3) if not isinstance(v, str) else v) for k, v in r.items()})
