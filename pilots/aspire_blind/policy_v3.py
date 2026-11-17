"""aspire_blind policy v3 = v2 + uniform arc-length resampling of the planned path (skill S4: the x-gridded corridor
path can have one long near-vertical segment; vertex-only projection then never advances and the carrot stalls).
Written after the official evaluation exposed the stall; NOT flown and NOT submitted."""
import heapq

import numpy as np

DT = 0.05


def corridor_path(start, posts, sides, target, R=0.075, dx=0.005, iters=400):
    """Taut x-monotone path; returns (pts, S, feasible)."""
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
    feasible = not bool(np.any(bad[1:-1]))
    mid = 0.5 * (L + U)
    L = np.where(bad, mid, L)
    U = np.where(bad, mid, U)
    ys = np.clip(np.linspace(start[1], target[1], n), L, U)
    ys[0], ys[-1] = start[1], target[1]
    for _ in range(iters):
        ys[1:-1] = 0.5 * (ys[:-2] + ys[2:])
        ys = np.clip(ys, L, U)
        ys[0], ys[-1] = start[1], target[1]
    pts = np.stack([xs, ys], 1)
    S = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return pts, S, feasible


def _seg_ok(a, b, posts, sides, rclr):
    ab = b - a
    L2 = max(1e-12, float(ab @ ab))
    for p, s in zip(posts, sides):
        t = np.clip(float((p - a) @ ab) / L2, 0.0, 1.0)
        if np.linalg.norm(a + t * ab - p) < rclr:
            return False
        if (a[0] - p[0]) * (b[0] - p[0]) < 0:          # spans the post's x: must cross on the allowed side
            y = a[1] + (p[0] - a[0]) / (b[0] - a[0]) * ab[1]
            if s * (y - p[1]) > 0:
                return False
    return True


def grid_path(start, posts, sides, target, R=0.07, h=0.01, rclr=0.062):
    """Shortest 8-connected grid path in the homotopy class 'below posts with side +1, above posts with side -1'
    (forbidden vertical half-lines), then string pulling.  Pure geometry."""
    xs = np.arange(0.0, 1.2 + 1e-9, h)
    ys = np.arange(-0.38, 0.38 + 1e-9, h)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    free = np.ones(X.shape, bool)
    for p, s in zip(posts, sides):
        free &= np.hypot(X - p[0], Y - p[1]) >= R
        free &= ~((np.abs(X - p[0]) <= 0.75 * h) & (s * (Y - p[1]) > 0))
    si = (int(round(start[0] / h)), int(round((start[1] + 0.38) / h)))
    ti = (int(round(target[0] / h)), int(round((target[1] + 0.38) / h)))
    free[si] = free[ti] = True
    nx, ny = free.shape
    dist = np.full(free.shape, np.inf)
    prev = {}
    dist[si] = 0.0
    pq = [(0.0, si)]
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142),
             (-1, -1, 1.4142)]
    while pq:
        dcur, u = heapq.heappop(pq)
        if u == ti:
            break
        if dcur > dist[u]:
            continue
        for di, dj, c in moves:
            v = (u[0] + di, u[1] + dj)
            if 0 <= v[0] < nx and 0 <= v[1] < ny and free[v]:
                nd = dcur + c
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[ti]):
        return None
    cells = [ti]
    while cells[-1] != si:
        cells.append(prev[cells[-1]])
    cells = cells[::-1]
    raw = np.array([[xs[i], ys[j]] for i, j in cells])
    raw[0], raw[-1] = start, target
    # string pulling
    out = [raw[0]]
    k = 0
    while k < len(raw) - 1:
        j = len(raw) - 1
        while j > k + 1 and not _seg_ok(raw[k], raw[j], posts, sides, rclr):
            j -= 1
        out.append(raw[j])
        k = j
    pts = np.array(out)
    # densify
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        m = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.005)))
        for q in range(1, m + 1):
            dense.append(a + (b - a) * q / m)
    dense = np.array(dense)
    S = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    return dense, S


def resample(pts, S, ds=0.004):
    n = max(2, int(np.ceil(S[-1] / ds)) + 1)
    s = np.linspace(0.0, S[-1], n)
    out = np.stack([np.interp(s, S, pts[:, 0]), np.interp(s, S, pts[:, 1])], 1)
    return out, s


def _plan(start, posts, sides, target):
    pts, S, ok = corridor_path(start, posts, sides, target)
    if ok:
        return pts, S, "corridor"
    for R in (0.07, 0.06, 0.052):
        g = grid_path(start, posts, sides, target, R=R, rclr=R - 0.008)
        if g is not None:
            return g[0], g[1], "grid%.3f" % R
    return pts, S, "corridor_fallback"


def plan(start, posts, sides, target):
    pts, S, mode = _plan(start, posts, sides, target)
    pts, S = resample(pts, S)
    return pts, S, mode


class Policy:
    def __init__(self, vmax=0.10, vmin=0.02, load_lo=2.5, load_hi=6.5, load_back=8.0, lead=0.03):
        self.vmax, self.vmin, self.load_lo, self.load_hi, self.load_back = vmax, vmin, load_lo, load_hi, load_back
        self.lead = lead
        self.reset()

    def reset(self):
        self.path = None
        self.s_cmd = 0.0
        self.s_tool = 0.0

    def _point(self, s):
        s = float(np.clip(s, 0.0, self.S[-1]))
        i = int(np.clip(np.searchsorted(self.S, s) - 1, 0, len(self.S) - 2))
        f = (s - self.S[i]) / max(1e-12, self.S[i + 1] - self.S[i])
        return self.path[i] + f * (self.path[i + 1] - self.path[i])

    def _project(self, p):
        m = (self.S >= self.s_tool - 0.05) & (self.S <= self.s_cmd + 0.05)
        idx = np.where(m)[0]
        d = np.linalg.norm(self.path[idx] - p, axis=1)
        return float(self.S[idx[int(np.argmin(d))]])

    def act(self, obs):
        tool = np.asarray(obs["tool_pos"], float)
        if self.path is None:
            self.target = np.asarray(obs["target"], float)
            self.path, self.S, self.mode = plan(tool, np.asarray(obs["posts"], float), np.asarray(obs["post_sides"]),
                                                self.target)
            self.s_cmd = self.s_tool = 0.0
        load = float(obs["load"])
        self.s_tool = self._project(tool)
        if load > self.load_back:
            self.s_cmd = max(0.0, min(self.s_cmd, self.s_tool) - 0.01)
            return self._point(self.s_cmd)
        frac = np.clip((self.load_hi - load) / (self.load_hi - self.load_lo), 0.0, 1.0)
        v = self.vmin + (self.vmax - self.vmin) * frac
        self.s_cmd = min(self.s_cmd + v * DT, self.s_tool + self.lead, self.S[-1])
        if self.s_cmd >= self.S[-1] - 1e-9:
            return self.target.copy()
        return self._point(self.s_cmd)


def make_policy(**kw):
    return Policy(**kw)
