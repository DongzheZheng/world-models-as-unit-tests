"""Pure path-planning geometry (no dynamics): waypoints around the posts on the required sides."""
import numpy as np

POST_R = 0.03
TOOL_R = 0.015


def seg_point_dist(a, b, p):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(1e-12, np.dot(ab, ab)), 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - p))


def waypoints(start, posts, sides, target, c=0.07, approach=0.0):
    """Tool waypoints: pass post i at clearance c on the side that gives the required winding.
    side +1 (CCW, post on the left travelling anchor->tool, i.e. travelling +x) -> pass at y below the post."""
    order = np.argsort(posts[:, 0])
    wps = []
    for i in order:
        p, s = posts[i], sides[i]
        if approach > 0:
            wps.append(p + np.array([-approach, -s * c]))
        wps.append(p + np.array([0.0, -s * c]))
    wps.append(np.asarray(target, float))
    return np.array(wps)


def polyline_len(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def densify(pts, ds=0.01):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / ds)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def winding(pts, posts):
    out = np.zeros(len(posts))
    for j, c in enumerate(posts):
        a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        da = (np.diff(a) + np.pi) % (2 * np.pi) - np.pi
        out[j] = da.sum()
    return out


def min_clearance(path, posts):
    return min(seg_point_dist(a, b, p) for a, b in zip(path[:-1], path[1:]) for p in posts)
