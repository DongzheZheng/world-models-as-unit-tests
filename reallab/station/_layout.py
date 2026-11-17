"""Per-episode randomisation of the cable rig with a solvability filter.  Internal: pilots must not read this module.

An episode (seed) fixes: the initial cable shape (a smooth random curve in the strip x <= 0.32 m left of the pegs,
starting at the clamp), the three peg positions (x in [0.38, 0.72], sorted by x, pairwise >= 0.15 m apart), their
required sides (random +-1), and the clip (right of the last peg).  Accepted only if the taut ("rubber band") route
anchor -> pegs on their sides -> clip is at most TAUT_MAX * cable length, stays on the table, and gives every peg a
winding of the required sign with |W| >= SIDE_THRESHOLD + WIND_MARGIN.
"""
from __future__ import annotations

import os

import numpy as np

from reallab.station import _limits as S

TAUT_MIN, TAUT_MAX = 0.70, 0.85
if os.environ.get("CAWM_STATION_TAUT"):          # lab tuning only
    TAUT_MIN, TAUT_MAX = map(float, os.environ["CAWM_STATION_TAUT"].split(","))
WIND_MARGIN = 0.35
REGION = dict(x=(0.03, 0.32), y=(-0.36, 0.36))
R_EFF = S.PEG_RADIUS + S.CABLE_RADIUS


def winding(points: np.ndarray, pegs: np.ndarray) -> np.ndarray:
    """Signed winding angle (rad) of the polyline `points` (anchor -> connector) about each peg centre (public rule)."""
    pts = np.asarray(points, float)[:, :2]
    out = np.zeros(len(pegs))
    for j, p in enumerate(np.asarray(pegs, float)):
        a = np.arctan2(pts[:, 1] - p[1], pts[:, 0] - p[0])
        da = np.diff(a)
        da = (da + np.pi) % (2 * np.pi) - np.pi
        out[j] = da.sum()
    return out


def sides(points, pegs) -> np.ndarray:
    w = winding(points, pegs)
    return np.where(w >= S.SIDE_THRESHOLD, 1, np.where(w <= -S.SIDE_THRESHOLD, -1, 0)).astype(np.int64)


def resample(points: np.ndarray, n: int) -> np.ndarray:
    """n points evenly spaced in arc length along a polyline (the 20 markers)."""
    p = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(p.shape[1])], axis=1)


def initial_shape(rng, n_edges: int, max_tries: int = 20000) -> np.ndarray:
    L = S.CABLE_LENGTH
    h = L / n_edges
    s = np.arange(n_edges) * h
    for _ in range(max_tries):
        K = 3
        amp = rng.uniform(-14.0, 14.0, K) / np.arange(1, K + 1)
        ph = rng.uniform(0, 2 * np.pi, K)
        lam = rng.uniform(0.5, 1.3)
        kappa = rng.uniform(-3, 3) + sum(amp[k] * np.sin(2 * np.pi * (k + 1) * s / lam + ph[k]) for k in range(K))
        kappa = np.clip(kappa, -1 / 0.07, 1 / 0.07)
        kappa[0] = 0.0
        phi = np.cumsum(kappa * h) - kappa[0] * h
        phi[0] = 0.0                                  # the clamped first edge points along +x
        pts = np.zeros((n_edges + 1, 2))
        pts[0] = S.ANCHOR
        pts[1:] = S.ANCHOR + np.cumsum(h * np.stack([np.cos(phi), np.sin(phi)], 1), axis=0)
        if pts[:, 0].min() < REGION["x"][0] or pts[:, 0].max() > REGION["x"][1]:
            continue
        if pts[:, 1].min() < REGION["y"][0] or pts[:, 1].max() > REGION["y"][1]:
            continue
        D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        iu = np.triu_indices(len(pts), k=4)
        if D[iu].min() < 0.03:
            continue
        return pts
    raise RuntimeError("no initial shape found")


def _tangent(C1, r1, C2, r2):
    """Points T1, T2 of the line tangent to two signed circles (r > 0: circle on the left of travel; r = 0: point)."""
    D = C2 - C1
    dist = np.linalg.norm(D)
    k = (r2 - r1) / max(dist, 1e-12)
    if abs(k) > 1.0:
        return None
    al = np.arctan2(D[1], D[0])
    for sgn in (1.0, -1.0):
        ang = al + sgn * np.arccos(k)
        n = np.array([np.cos(ang), np.sin(ang)])
        d = np.array([n[1], -n[0]])
        T1, T2 = C1 - r1 * n, C2 - r2 * n
        if d @ (T2 - T1) > 0 and abs(n @ (T2 - T1)) < 1e-7:
            return T1, T2
    return None


def _build(start, end, circles):
    """Polyline (with sampled arcs) of the path start -> tangent/arc around each (C, r) -> end."""
    nodes = [(np.asarray(start, float), 0.0)] + list(circles) + [(np.asarray(end, float), 0.0)]
    segs = []
    for i in range(len(nodes) - 1):
        t = _tangent(nodes[i][0], nodes[i][1], nodes[i + 1][0], nodes[i + 1][1])
        if t is None:
            return None
        segs.append(t)
    pts = [segs[0][0]]
    for i in range(len(segs)):
        if i > 0:                                   # arc on circle i from segs[i-1][1] to segs[i][0]
            C, r = nodes[i]
            a0 = np.arctan2(*(segs[i - 1][1] - C)[::-1])
            a1 = np.arctan2(*(segs[i][0] - C)[::-1])
            sweep = (a1 - a0) % (2 * np.pi) if r > 0 else -((a0 - a1) % (2 * np.pi))
            if abs(sweep) > 1.6 * np.pi:
                return None
            for a in np.linspace(a0, a0 + sweep, max(2, int(abs(sweep) / 0.1)))[1:]:
                pts.append(C + abs(r) * np.array([np.cos(a), np.sin(a)]))
        pts.append(segs[i][1])
    return np.array(pts)


def _ok(poly, pegs, peg_sides, i, R=None):
    """Peg i lies at distance >= R_EFF (minus a hair) from `poly` and on the required side of it."""
    p = pegs[i]
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    q = a + t[:, None] * ab
    if np.linalg.norm(q - p, axis=1).min() < (R_EFF if R is None else R) - 1e-4:
        return False
    return bool(winding(poly, p[None])[0] * peg_sides[i] > 0)


def taut_route(start, pegs, peg_sides, end, R=None):
    """Shortest route start -> end passing each peg (in order) with the peg on the route's left (side +1) or right
    (side -1), at distance >= R_EFF from the peg centres.  Exact tangent/arc construction with an active set."""
    pegs = np.asarray(pegs, float)
    R = R_EFF if R is None else R
    active = [True] * len(pegs)
    poly = None
    for _ in range(8):
        circ = [(pegs[i], peg_sides[i] * R) for i in range(len(pegs)) if active[i]]
        poly = _build(start, end, circ)
        if poly is None:
            return None
        changed = False
        for i in range(len(pegs)):
            if not active[i] and not _ok(poly, pegs, peg_sides, i, R):
                active[i] = True
                changed = True
                break
        if changed:
            continue
        for i in range(len(pegs)):                  # drop a contact that the path does not need
            if active[i]:
                trial = list(active)
                trial[i] = False
                c2 = [(pegs[j], peg_sides[j] * R) for j in range(len(pegs)) if trial[j]]
                p2 = _build(start, end, c2)
                if p2 is not None and _ok(p2, pegs, peg_sides, i, R) and length(p2) < length(poly):
                    active = trial
                    changed = True
                    break
        if not changed:
            return poly
    return poly


def length(poly) -> float:
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def sample_shape(seed: int, n_edges: int, attempt: int = 0) -> np.ndarray:
    return initial_shape(np.random.default_rng([7321, int(seed)] + ([attempt] if attempt else [])), n_edges)


def sample_posts(seed: int, n_edges: int, relaxed: np.ndarray, max_tries: int = 20000, attempt: int = 0) -> dict | None:
    """Posts, sides and clip for the relaxed initial cable (vertex xy, anchor first)."""
    rng = np.random.default_rng([7321, int(seed), 1] + ([attempt] if attempt else []))
    relaxed = np.asarray(relaxed, float)[:, :2]
    h = S.CABLE_LENGTH / n_edges
    clamp_end = np.array(S.ANCHOR) + h * np.array(S.CLAMP_DIR)
    markers = relaxed[::max(1, n_edges // (S.N_MARKERS - 1))]
    for it in range(max_tries):
        pegs = np.stack([rng.uniform(0.38, 0.72, 3), rng.uniform(-0.25, 0.25, 3)], 1)
        pegs = pegs[np.argsort(pegs[:, 0])]
        D = np.linalg.norm(pegs[:, None] - pegs[None], axis=2)[np.triu_indices(3, 1)]
        if D.min() < 0.15:
            continue
        if np.min(np.linalg.norm(relaxed[:, None] - pegs[None], axis=2)) < 0.07:
            continue
        if np.any(sides(markers, pegs) != 0):
            continue
        side = rng.choice([-1, 1], 3)
        clip = np.array([pegs[2, 0] + rng.uniform(0.10, 0.35), rng.uniform(-0.30, 0.30)])
        if clip[0] > 1.05 or np.linalg.norm(pegs - clip, axis=1).min() < 0.10:
            continue
        route = taut_route(clamp_end, pegs, side, clip)
        if route is None:
            continue
        route = np.vstack([np.array(S.ANCHOR)[None], route])
        Lr = length(route)
        if not (TAUT_MIN * S.CABLE_LENGTH <= Lr <= TAUT_MAX * S.CABLE_LENGTH):
            continue
        if route[:, 1].min() < S.TABLE_LOW[1] + 0.03 or route[:, 1].max() > S.TABLE_HIGH[1] - 0.03:
            continue
        w = winding(route, pegs)
        if np.any(w * side < S.SIDE_THRESHOLD + WIND_MARGIN):
            continue
        return dict(pegs=pegs, sides=side.astype(np.int64), clip=clip, taut_length=Lr, route=route)
    return None


def setup(scene, seed: int) -> dict:
    """Place the initial shape, let it relax, then draw a solvable post layout for the relaxed shape (a few attempts
    with fresh shapes if the relaxed shape leaves no room)."""
    for attempt in range(8):
        scene.reset(sample_shape(seed, scene.n_edges, attempt))
        e = sample_posts(seed, scene.n_edges, scene.points(), attempt=attempt)
        if e is not None:
            scene.place(e["pegs"], e["clip"])
            e["attempt"] = attempt
            e["posts"], e["target"] = e["pegs"], e["clip"]
            return e
    raise RuntimeError(f"seed {seed}: no solvable configuration found")
