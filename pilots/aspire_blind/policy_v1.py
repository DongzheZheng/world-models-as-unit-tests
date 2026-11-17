"""aspire_blind policy v1: x-monotone corridor path around the posts on the required sides,
carrot following at moderate speed with load-proportional slowdown and back-off, settle at target."""
import numpy as np

DT = 0.05


def corridor_path(start, posts, sides, target, R=0.075, dx=0.005, iters=400):
    """Taut x-monotone path from start to target: at each x the tool y must stay below post (side +1) or above
    post (side -1) by a circle of radius R.  Elastic-band smoothing + clipping (pure geometry)."""
    x0, x1 = float(start[0]), float(target[0])
    n = max(3, int(np.ceil((x1 - x0) / dx)) + 1)
    xs = np.linspace(x0, x1, n)
    U = np.full(n, 0.38)
    L = np.full(n, -0.38)
    for p, s in zip(posts, sides):
        h = R ** 2 - (xs - p[0]) ** 2
        m = h > 0
        off = np.sqrt(np.where(m, h, 0.0))
        if s > 0:
            U = np.where(m, np.minimum(U, p[1] - off), U)
        else:
            L = np.where(m, np.maximum(L, p[1] + off), L)
    bad = L > U
    mid = 0.5 * (L + U)
    L = np.where(bad, mid, L)
    U = np.where(bad, mid, U)
    ys = np.linspace(start[1], target[1], n)
    ys = np.clip(ys, L, U)
    ys[0], ys[-1] = start[1], target[1]
    for _ in range(iters):
        ys[1:-1] = 0.5 * (ys[:-2] + ys[2:])
        ys = np.clip(ys, L, U)
        ys[0], ys[-1] = start[1], target[1]
    pts = np.stack([xs, ys], 1)
    # resample by arc length
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    return pts, s


class Policy:
    def __init__(self, vmax=0.10, vmin=0.02, load_lo=3.0, load_hi=7.0, load_back=8.0, lead=0.03, R=0.075):
        self.vmax, self.vmin, self.load_lo, self.load_hi, self.load_back = vmax, vmin, load_lo, load_hi, load_back
        self.lead, self.R = lead, R
        self.reset()

    def reset(self):
        self.path = None
        self.s_cmd = 0.0

    def _point(self, s):
        s = float(np.clip(s, 0.0, self.S[-1]))
        i = int(np.clip(np.searchsorted(self.S, s) - 1, 0, len(self.S) - 2))
        f = (s - self.S[i]) / max(1e-12, self.S[i + 1] - self.S[i])
        return self.path[i] + f * (self.path[i + 1] - self.path[i])

    def _project(self, p):
        d = np.linalg.norm(self.path - p, axis=1)
        return float(self.S[int(np.argmin(d))])

    def act(self, obs):
        tool = np.asarray(obs["tool_pos"], float)
        if self.path is None:
            self.target = np.asarray(obs["target"], float)
            self.path, self.S = corridor_path(tool, np.asarray(obs["posts"], float), np.asarray(obs["post_sides"]),
                                              self.target, R=self.R)
            self.s_cmd = 0.0
        load = float(obs["load"])
        s_tool = self._project(tool)
        if load > self.load_back:
            self.s_cmd = max(0.0, min(self.s_cmd, s_tool) - 0.01)
            return self._point(self.s_cmd)
        frac = np.clip((self.load_hi - load) / (self.load_hi - self.load_lo), 0.0, 1.0)
        v = self.vmin + (self.vmax - self.vmin) * frac
        self.s_cmd = min(self.s_cmd + v * DT, s_tool + self.lead, self.S[-1])
        if self.s_cmd >= self.S[-1] - 1e-9:
            return self.target.copy()
        return self._point(self.s_cmd)


def make_policy(**kw):
    return Policy(**kw)
