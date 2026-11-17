"""Batched geometric queries for code worlds: range sensing and clearance.

This is NOT a renderer.  A code world keeps its geometry as lists of primitives per environment
(axis-aligned boxes, vertical cylinders, spheres, a ground plane, 2-D wall segments) and asks two
kinds of questions for thousands of environments at once on the GPU:

  * ray_*      "how far along this ray is the first surface?"   -> range sensors, line of sight
  * sdf_*      "how far is this point from the nearest surface?" -> collision tests and
               differentiable clearance costs (gradients flow to the query points)

Shapes: B environments, R rays (or N points), K primitives per environment.  Primitives are padded
to K per environment; a padded slot is marked by a non-positive size and never hit.
"""
from __future__ import annotations

import math

import torch

_EPS = 1e-8


def _safe(d: torch.Tensor) -> torch.Tensor:
    """Replace near-zero direction components so 1/d stays finite (sign-preserving)."""
    return torch.where(d.abs() < _EPS, torch.where(d < 0, -_EPS, _EPS).to(d.dtype), d)


# ----------------------------------------------------------------------------- 3-D rays
def ray_aabb(o: torch.Tensor, d: torch.Tensor, center: torch.Tensor, half: torch.Tensor,
             max_range: float) -> tuple[torch.Tensor, torch.Tensor]:
    """First hit of rays with axis-aligned boxes.

    o, d: (B,R,3) origins and unit directions.  center, half: (B,K,3); half <= 0 marks padding.
    Returns (dist (B,R) clipped to max_range, idx (B,R) of the box hit or -1).  An origin inside
    a box reads 0.
    """
    inv = 1.0 / _safe(d)[:, :, None, :]
    lo = (center - half)[:, None] - o[:, :, None, :]
    hi = (center + half)[:, None] - o[:, :, None, :]
    t1, t2 = lo * inv, hi * inv
    tmin = torch.minimum(t1, t2).amax(-1)
    tmax = torch.maximum(t1, t2).amin(-1)
    active = (half > 0).all(-1)[:, None, :]
    hit = (tmax >= tmin.clamp(min=0)) & (tmax > 0) & active
    t = torch.where(hit, tmin.clamp(min=0), torch.full_like(tmin, math.inf))
    dist, idx = t.min(-1)
    miss = dist > max_range
    return dist.clamp(max=max_range), torch.where(miss, torch.full_like(idx, -1), idx)


def ray_vcylinder(o: torch.Tensor, d: torch.Tensor, center: torch.Tensor, radius: torch.Tensor,
                  half_h: torch.Tensor, max_range: float) -> tuple[torch.Tensor, torch.Tensor]:
    """First hit of rays with vertical (z-axis) cylinders; side wall only, caps ignored.

    center: (B,K,3) cylinder centre, radius, half_h: (B,K); radius <= 0 marks padding.
    """
    oc = o[:, :, None, :2] - center[:, None, :, :2]                     # (B,R,K,2)
    dxy = d[:, :, None, :2]
    a = (dxy * dxy).sum(-1).clamp(min=_EPS)
    b = (oc * dxy).sum(-1)
    c = (oc * oc).sum(-1) - radius[:, None] ** 2
    disc = b * b - a * c
    t = (-b - disc.clamp(min=0).sqrt()) / a
    t = torch.where(c <= 0, torch.zeros_like(t), t)                      # origin inside the column
    z = o[:, :, None, 2] + t * d[:, :, None, 2]
    ok = (disc >= 0) & (t >= 0) & ((z - center[:, None, :, 2]).abs() <= half_h[:, None]) & (radius[:, None] > 0)
    t = torch.where(ok, t, torch.full_like(t, math.inf))
    dist, idx = t.min(-1)
    miss = dist > max_range
    return dist.clamp(max=max_range), torch.where(miss, torch.full_like(idx, -1), idx)


def ray_sphere(o: torch.Tensor, d: torch.Tensor, center: torch.Tensor, radius: torch.Tensor,
               max_range: float) -> tuple[torch.Tensor, torch.Tensor]:
    """First hit of rays with spheres.  center (B,K,3), radius (B,K); radius <= 0 marks padding."""
    oc = o[:, :, None, :] - center[:, None]
    b = (oc * d[:, :, None, :]).sum(-1)
    c = (oc * oc).sum(-1) - radius[:, None] ** 2
    disc = b * b - c
    t = -b - disc.clamp(min=0).sqrt()
    t = torch.where(c <= 0, torch.zeros_like(t), t)
    ok = (disc >= 0) & (t >= 0) & (radius[:, None] > 0)
    t = torch.where(ok, t, torch.full_like(t, math.inf))
    dist, idx = t.min(-1)
    miss = dist > max_range
    return dist.clamp(max=max_range), torch.where(miss, torch.full_like(idx, -1), idx)


def ray_ground(o: torch.Tensor, d: torch.Tensor, z0: float, max_range: float) -> torch.Tensor:
    """Distance along rays to the horizontal plane z = z0 (hit from above only). Returns (B,R)."""
    dz = d[..., 2]
    t = (z0 - o[..., 2]) / torch.where(dz < -_EPS, dz, torch.full_like(dz, -_EPS))
    t = torch.where((dz < -_EPS) & (t >= 0), t, torch.full_like(t, math.inf))
    return t.clamp(max=max_range)


# ----------------------------------------------------------------------------- 2-D rays (floor-plan worlds)
def ray_segments_2d(o: torch.Tensor, d: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                    max_range: float) -> tuple[torch.Tensor, torch.Tensor]:
    """First hit of planar rays with wall segments.

    o, d: (B,R,2) origins / unit directions.  a, b: (B,S,2) segment endpoints (a == b marks padding).
    Returns (dist (B,R), idx (B,R) or -1).
    """
    e = (b - a)[:, None]                                                 # (B,1,S,2)
    w = a[:, None] - o[:, :, None, :]                                    # (B,R,S,2)
    dd = d[:, :, None, :]
    den = dd[..., 0] * e[..., 1] - dd[..., 1] * e[..., 0]
    den_ok = den.abs() > _EPS
    den_s = torch.where(den_ok, den, torch.ones_like(den))
    t = (w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]) / den_s          # along the ray
    u = (w[..., 0] * dd[..., 1] - w[..., 1] * dd[..., 0]) / den_s        # along the segment
    ok = den_ok & (t >= 0) & (u >= 0) & (u <= 1)
    t = torch.where(ok, t, torch.full_like(t, math.inf))
    dist, idx = t.min(-1)
    miss = dist > max_range
    return dist.clamp(max=max_range), torch.where(miss, torch.full_like(idx, -1), idx)


def ray_circles_2d(o: torch.Tensor, d: torch.Tensor, center: torch.Tensor, radius: torch.Tensor,
                   max_range: float) -> tuple[torch.Tensor, torch.Tensor]:
    """First hit of planar rays with circles.  center (B,K,2), radius (B,K); radius <= 0 = padding."""
    oc = o[:, :, None, :] - center[:, None]
    bq = (oc * d[:, :, None, :]).sum(-1)
    c = (oc * oc).sum(-1) - radius[:, None] ** 2
    disc = bq * bq - c
    t = -bq - disc.clamp(min=0).sqrt()
    t = torch.where(c <= 0, torch.zeros_like(t), t)
    ok = (disc >= 0) & (t >= 0) & (radius[:, None] > 0)
    t = torch.where(ok, t, torch.full_like(t, math.inf))
    dist, idx = t.min(-1)
    miss = dist > max_range
    return dist.clamp(max=max_range), torch.where(miss, torch.full_like(idx, -1), idx)


# ----------------------------------------------------------------------------- signed distances (differentiable in p)
def sdf_aabb(p: torch.Tensor, center: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """Signed distance from points p (B,N,3) to the union of boxes (B,K,3)+(B,K,3). Returns (B,N)."""
    q = (p[:, :, None, :] - center[:, None]).abs() - half[:, None]
    outside = q.clamp(min=0).norm(dim=-1)
    inside = q.amax(-1).clamp(max=0)
    sd = outside + inside
    sd = torch.where((half > 0).all(-1)[:, None, :], sd, torch.full_like(sd, math.inf))
    return sd.amin(-1)


def sdf_vcylinder(p: torch.Tensor, center: torch.Tensor, radius: torch.Tensor, half_h: torch.Tensor) -> torch.Tensor:
    """Signed distance from points p (B,N,3) to the union of capped vertical cylinders. Returns (B,N)."""
    rel = p[:, :, None, :] - center[:, None]
    dr = rel[..., :2].norm(dim=-1) - radius[:, None]
    dz = rel[..., 2].abs() - half_h[:, None]
    outside = torch.stack([dr.clamp(min=0), dz.clamp(min=0)], -1).norm(dim=-1)
    inside = torch.maximum(dr, dz).clamp(max=0)
    sd = outside + inside
    sd = torch.where(radius[:, None] > 0, sd, torch.full_like(sd, math.inf))
    return sd.amin(-1)


# ----------------------------------------------------------------------------- sensor layouts
def fan_dirs(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """Unit directions of a range-sensor fan in the sensor frame (+x forward, +y left, +z up).

    az: (A,) azimuths [rad], el: (E,) elevations [rad].  Returns (E*A, 3), elevation-major order.
    """
    e, a = torch.meshgrid(el, az, indexing="ij")
    return torch.stack([e.cos() * a.cos(), e.cos() * a.sin(), e.sin()], -1).reshape(-1, 3)


def yaw_rotate(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate sensor-frame vectors v (R,3) or (B,R,3) about +z by yaw (B,). Returns (B,R,3)."""
    if v.dim() == 2:
        v = v[None].expand(yaw.shape[0], -1, -1)
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)
