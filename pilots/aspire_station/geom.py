"""Pure plane geometry used by the aspire_station programs and offline checks (no installation model)."""
import numpy as np

ANCHOR = np.array([0.05, 0.0])
POST_R = 0.03
TOOL_R = 0.015
ROPE_R = 0.004


def resample(P, ds):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(np.ceil(s[-1] / ds)) + 1)
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])], 1)


def push_out(P, posts, rmin, fixed_ends=True):
    P = P.copy()
    lo, hi = (1, len(P) - 1) if fixed_ends else (0, len(P))
    for c in posts:
        d = P[lo:hi] - c
        r = np.linalg.norm(d, axis=1)
        m = r < rmin
        if np.any(m):
            P[lo:hi][m] = c + d[m] / np.maximum(r[m, None], 1e-9) * rmin
    return P


def plan_path(tool0, posts, sides, target, clr=0.07, ds=0.005, smooth_iters=30):
    """Tool path from tool0 to target passing each post on the side that gives the required winding.
    Side +1 (counter-clockwise, post on the left of the travel direction) -> pass on the right of the post."""
    tool0, target = np.asarray(tool0, float), np.asarray(target, float)
    d = target - ANCHOR
    d = d / np.linalg.norm(d)
    left = np.array([-d[1], d[0]])
    order = np.argsort([(c - ANCHOR) @ d for c in posts])
    wps = [tool0]
    for j in order:
        c = posts[j]
        wps.append(c - sides[j] * left * clr)
    wps.append(target)
    P = resample(np.array(wps), ds)
    # smooth corners while staying clear of posts
    rmin = POST_R + TOOL_R + 0.012
    for _ in range(smooth_iters):
        Q = P.copy()
        Q[1:-1] = 0.5 * P[1:-1] + 0.25 * (P[:-2] + P[2:])
        P = push_out(Q, posts, rmin)
    P = push_out(P, posts, rmin)
    return resample(P, ds)


def winding_of(poly, posts):
    pts = np.asarray(poly, float)[:, :2]
    out = np.zeros(len(posts))
    for j, c in enumerate(np.asarray(posts, float)):
        a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        out[j] = (((np.diff(a) + np.pi) % (2 * np.pi)) - np.pi).sum()
    return out


def taut_length(poly, posts, r=POST_R + ROPE_R, iters=3000):
    """Length of the shortest curve homotopic to `poly` (fixed ends) avoiding discs of radius r (string pulling,
    coarse-to-fine)."""
    P = resample(poly, 0.04)
    for ds, it in ((0.04, iters), (0.015, iters // 3)):
        P = resample(P, ds)
        for _ in range(it):
            Q = P.copy()
            Q[1:-1] = 0.5 * (P[:-2] + P[2:])
            P = push_out(Q, posts, r)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), P


def dist_to_posts(P, posts):
    return np.min(np.linalg.norm(np.asarray(P)[:, None, :] - np.asarray(posts)[None], axis=2), axis=1)
