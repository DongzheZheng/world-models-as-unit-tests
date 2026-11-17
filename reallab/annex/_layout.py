"""Per-episode randomisation of the annex installation.  Internal: pilots must not read this module.

An episode (seed) fixes: the tool's start (uniform in the zone shrunk by 3 cm), the initial resting shape of the rope
(a smooth random curve from the tool, inside x in [0.02, 0.62], |y| <= 0.38, no near self-contact), the post (bearing
in POST_BEARING and distance in POST_DIST from the zone centre), the required side (+-1) and the Pocket target (on the
far side of the post as seen from the zone centre).  Rejected: rope closer than 0.10 m to the post or to the Pocket
target, or an initial |winding| >= pi/2.
"""
from __future__ import annotations

import numpy as np

from reallab.annex import _limits as S

REGION = dict(x=(0.02, 0.62), y=(-0.38, 0.38))


def winding(points, post) -> float:
    pts = np.asarray(points, float)[:, :2]
    c = np.asarray(post, float)
    a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    da = (np.diff(a) + np.pi) % (2 * np.pi) - np.pi
    return float(da.sum())


def initial_shape(rng, n_edges: int, max_tries: int = 20000) -> np.ndarray:
    """(n_edges+1, 2) vertex xy, vertex 0 = free tip, last vertex = at the tool."""
    L = S.LENGTH
    h = L / n_edges
    s = np.arange(n_edges) * h
    lo = np.array(S.ZONE_LOW) + 0.03
    hi = np.array(S.ZONE_HIGH) - 0.03
    for _ in range(max_tries):
        tool = rng.uniform(lo, hi)
        head = rng.uniform(0.5 * np.pi, 1.5 * np.pi)            # leaves the tool going backwards / sideways
        K = 3
        amp = rng.uniform(-12.0, 12.0, K) / np.arange(1, K + 1)
        ph = rng.uniform(0, 2 * np.pi, K)
        lam = rng.uniform(0.4, 1.2)
        kappa = rng.uniform(-4, 4) + sum(amp[k] * np.sin(2 * np.pi * (k + 1) * s / lam + ph[k]) for k in range(K))
        kappa = np.clip(kappa, -1 / 0.06, 1 / 0.06)
        phi = head + np.cumsum(kappa * h) - kappa[0] * h
        pts = np.zeros((n_edges + 1, 2))
        pts[0] = tool
        pts[1:] = tool + np.cumsum(h * np.stack([np.cos(phi), np.sin(phi)], 1), axis=0)
        if pts[:, 0].min() < REGION["x"][0] or pts[:, 0].max() > REGION["x"][1]:
            continue
        if pts[:, 1].min() < REGION["y"][0] or pts[:, 1].max() > REGION["y"][1]:
            continue
        D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
        iu = np.triu_indices(len(pts), k=4)
        if D[iu].min() < 0.03:
            continue
        return pts[::-1].copy()
    raise RuntimeError("no initial shape found")


def pocket_target(post) -> np.ndarray:
    c = np.asarray(post, float)
    u = c - np.array(S.ZONE_CENTRE)
    u = u / np.linalg.norm(u)
    return c + S.POCKET_BEHIND * u


def sample_post(seed: int, relaxed: np.ndarray, attempt: int = 0, max_tries: int = 5000):
    rng = np.random.default_rng([9157, int(seed), 1, attempt])
    pts = np.asarray(relaxed, float)[:, :2]
    for _ in range(max_tries):
        th = rng.uniform(*S.POST_BEARING)
        r = rng.uniform(*S.POST_DIST)
        post = np.array(S.ZONE_CENTRE) + r * np.array([np.cos(th), np.sin(th)])
        side = int(rng.choice([-1, 1]))
        if np.min(np.linalg.norm(pts - post, axis=1)) < 0.10:
            continue
        tgt = pocket_target(post)
        if np.min(np.linalg.norm(pts - tgt, axis=1)) < 0.10:
            continue
        if abs(winding(pts[::2], post)) >= np.pi / 2:
            continue
        return dict(post=post, side=side, target=tgt, bearing=float(th), dist=float(r))
    return None


def setup(scene, seed: int) -> dict:
    for attempt in range(8):
        rng = np.random.default_rng([9157, int(seed), attempt])
        scene.reset(initial_shape(rng, scene.n_edges))
        e = sample_post(seed, scene.vertices(), attempt=attempt)
        if e is not None:
            scene.place(e["post"], e["target"])
            e["attempt"] = attempt
            return e
    raise RuntimeError(f"seed {seed}: no configuration found")
