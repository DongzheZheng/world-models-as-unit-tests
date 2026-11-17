# Misc observation checks: reset load, servo lag vs load, capstan (wrap) effect, rope speed profile.
import numpy as np, glob
np.set_printoptions(precision=4, suppress=True, linewidth=200)
L = np.load("runs/station/observation/layouts.npz")
P = L["points"][..., :2]
e2e = np.linalg.norm(P[:, 19] - P[:, 0], axis=1)
turn = np.abs(np.diff(np.unwrap(np.arctan2(*np.diff(P, axis=1).transpose(2, 0, 1)[::-1]), axis=1), axis=1)).sum(1)
for name, x in [("e2e", e2e), ("total turning", turn), ("tool x", L["tool_pos"][:, 0]), ("tool y", L["tool_pos"][:, 1])]:
    print("reset load corr with", name, np.corrcoef(x, L["load"])[0, 1])
print("reset load quantiles", np.quantile(L["load"], [0, .25, .5, .75, 1]))

def wrap_total(pts, posts, rad=0.06):
    # total turning angle of the polyline at nodes within `rad` of any post (proxy for wrap / capstan angle)
    d = np.diff(pts, axis=0); a = np.arctan2(d[:, 1], d[:, 0]); da = (np.diff(a) + np.pi) % (2 * np.pi) - np.pi
    near = np.linalg.norm(pts[1:-1, None] - posts[None], axis=2).min(1) < rad
    return np.abs(da[near]).sum()

X, Y = [], []
for f in sorted(glob.glob("runs/station/observation/episode_*.npz")):
    E = np.load(f); pts = E["points"][..., :2]; ld = E["load"]; tp = E["tool_pos"]; A = E["actions"]; posts = E["posts"][0]
    T = len(A)
    lag = np.linalg.norm(A - tp[1:], axis=1)
    for t in range(2, T):
        sp = np.linalg.norm(pts[t] - pts[t - 1], axis=1) / 0.05
        X.append([sp[1:].sum(), wrap_total(pts[t], posts), lag[t - 1], np.linalg.norm(E["tool_vel"][t])]); Y.append(ld[t])
    # speed profile along the rope during steady dragging (mid-episode)
    t = T // 2
    sp = np.linalg.norm(pts[t] - pts[t - 1], axis=1) / 0.05
    print(f[-8:-4], "node speeds at t=", t, sp.round(3), "load", ld[t].round(2))
X, Y = np.array(X), np.array(Y)
print("corr(lag, load)", np.corrcoef(X[:, 2], Y)[0, 1], "corr(lag, toolv)", np.corrcoef(X[:, 2], X[:, 3])[0, 1])
for w0 in [0, 1, 2, 3, 4]:
    m = (X[:, 1] >= w0) & (X[:, 1] < w0 + 1)
    if m.sum() > 10:
        b = (X[m, 0] @ Y[m]) / (X[m, 0] @ X[m, 0])
        print(f"wrap in [{w0},{w0+1}) n={m.sum()} load/abs_sum = {b:.3f}")
# capstan-style fit: load = c * abs_sum * exp(mu*wrap)
best = None
for mu in np.linspace(0, 0.6, 31):
    z = X[:, 0] * np.exp(mu * X[:, 1]); c = (z @ Y) / (z @ z); r = np.sqrt(((Y - c * z) ** 2).mean())
    if best is None or r < best[0]:
        best = (r, mu, c)
print("capstan fit rmse %.3f mu %.3f c %.3f" % best)
# lag regression lag = a + b*load
A_ = np.c_[np.ones(len(Y)), Y, X[:, 3]]; w, *_ = np.linalg.lstsq(A_, X[:, 2], rcond=None); print("lag = a + b*load + c*toolv:", w)
