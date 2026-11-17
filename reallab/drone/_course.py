"""Internal: container-yard course generator + solvability checker (lab-side, not for the world-model author)."""
from __future__ import annotations

import numpy as np

YARD_X = (0.0, 60.0)
YARD_Y = (-12.0, 12.0)
CEILING = 6.0
C20, C40, CW, CH = 6.06, 12.19, 2.44, 2.59

DIFFICULTY = {
    #          aisle range (m)  slot occupancy  P(stack height 1,2,3)   legs poles
    "easy":   dict(aisle=(5.0, 7.0), occ=0.55, ph=(0.45, 0.35, 0.20), legs=2, poles=3),
    "medium": dict(aisle=(3.5, 6.0), occ=0.70, ph=(0.35, 0.35, 0.30), legs=3, poles=5),
    "hard":   dict(aisle=(3.0, 4.5), occ=0.85, ph=(0.25, 0.35, 0.40), legs=4, poles=6),
}
CLEARANCE = 0.8       # metres between drone centre and any surface used by the solvability check
GRID = 0.5            # metres
Z_MIN, Z_MAX = CLEARANCE, CEILING - CLEARANCE


def _box(cx, cy, hx, hy, h, kind, stack=0):
    return dict(kind=kind, center=[float(cx), float(cy), float(h / 2)], half_extents=[float(hx), float(hy), float(h / 2)],
                stack=int(stack))


def _sample_layout(rng: np.random.Generator, diff: str) -> dict:
    P = DIFFICULTY[diff]
    boxes: list[dict] = []
    x_cur = 8.0 + rng.uniform(0.0, 1.0)
    while True:
        orient = rng.choice(["x", "y"])
        k = int(rng.choice([1, 2], p=[0.6, 0.4]))
        m = int(rng.choice([1, 2], p=[0.65, 0.35]))
        Lc = float(rng.choice([C20, C40], p=[0.5, 0.5]))
        long_ext = k * Lc + (k - 1) * 0.4
        wide_ext = m * CW + (m - 1) * 0.4
        ex, ey = (long_ext, wide_ext) if orient == "x" else (wide_ext, long_ext)
        if x_cur + ex > 52.0:
            break
        aisle_y = rng.uniform(*P["aisle"])
        y_cur = -11.5 + rng.uniform(0.0, 3.0)
        while y_cur + ey <= 11.5:
            if rng.random() < P["occ"]:
                for i in range(k):
                    for j in range(m):
                        if orient == "x":
                            lx, ly = Lc, CW
                            cx = x_cur + i * (Lc + 0.4) + Lc / 2
                            cy = y_cur + j * (CW + 0.4) + CW / 2
                        else:
                            lx, ly = CW, Lc
                            cx = x_cur + j * (CW + 0.4) + CW / 2
                            cy = y_cur + i * (Lc + 0.4) + Lc / 2
                        h = int(rng.choice([1, 2, 3], p=P["ph"]))
                        boxes.append(_box(cx, cy, lx / 2, ly / 2, h * CH, "container", h))
            y_cur += ey + aisle_y
        x_cur += ex + rng.uniform(*P["aisle"])
    # furniture standing in the aisles
    cyl: list[dict] = []

    def free(px, py, r):
        for b in boxes:
            cx, cy, _ = b["center"]; hx, hy, _ = b["half_extents"]
            dx, dy = max(abs(px - cx) - hx, 0.0), max(abs(py - cy) - hy, 0.0)
            if np.hypot(dx, dy) < r + 0.6:
                return False
        return True

    placed: list[tuple[float, float, float]] = []

    def place(r):
        for _ in range(200):
            px, py = rng.uniform(9.0, 51.0), rng.uniform(-11.0, 11.0)
            if free(px, py, r) and all(np.hypot(px - qx, py - qy) > r + qr + 1.5 for qx, qy, qr in placed):
                placed.append((px, py, r))
                return px, py
        return None

    for _ in range(P["legs"]):
        hw = 0.3 + rng.uniform(-0.03, 0.03)
        pos = place(hw * 1.5)
        if pos is not None:
            boxes.append(_box(pos[0], pos[1], hw, hw, 9.0, "crane_leg"))
    for _ in range(P["poles"]):
        r = 0.15 + rng.uniform(-0.02, 0.02)
        pos = place(r)
        if pos is not None:
            cyl.append(dict(kind="pole", center=[float(pos[0]), float(pos[1]), 4.5], radius=float(r), height=9.0))
    return dict(boxes=boxes, cylinders=cyl)


def _free_mask(layout: dict, margin: float = CLEARANCE) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = (np.arange(int(round((YARD_X[1] - YARD_X[0]) / GRID))) + 0.5) * GRID + YARD_X[0]
    ys = (np.arange(int(round((YARD_Y[1] - YARD_Y[0]) / GRID))) + 0.5) * GRID + YARD_Y[0]
    zs = (np.arange(int(round(CEILING / GRID))) + 0.5) * GRID
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    free = (Z >= Z_MIN) & (Z <= Z_MAX)
    for b in layout["boxes"]:
        c, h = b["center"], b["half_extents"]
        hit = (np.abs(X - c[0]) < h[0] + margin) & (np.abs(Y - c[1]) < h[1] + margin) & (Z < c[2] + h[2] + margin)
        free &= ~hit
    for c in layout["cylinders"]:
        hit = (np.hypot(X - c["center"][0], Y - c["center"][1]) < c["radius"] + margin) & (Z < c["height"] + margin)
        free &= ~hit
    return free, xs, ys, zs


def _cell(v, lo):
    return int((v - lo) // GRID)


def solvable(course: dict) -> bool:
    """Coarse 3-D grid connectivity check (0.5 m cells, 0.8 m clearance, flight band 0.8..5.2 m)."""
    from scipy import ndimage
    free, xs, ys, zs = _free_mask(course)
    lab, _ = ndimage.label(free)
    s, g = course["start"], course["goal"]
    cs = (_cell(s[0], YARD_X[0]), _cell(s[1], YARD_Y[0]), _cell(s[2], 0.0))
    cg = (_cell(g[0], YARD_X[0]), _cell(g[1], YARD_Y[0]), _cell(g[2], 0.0))
    if lab[cs] == 0 or lab[cg] == 0:
        return False
    return bool(lab[cs] == lab[cg])


def straight_line_blocked(course: dict, margin: float = 0.3) -> bool:
    s, g = np.array(course["start"]), np.array(course["goal"])
    for t in np.linspace(0, 1, 400):
        p = s + t * (g - s)
        for b in course["boxes"]:
            c, h = b["center"], b["half_extents"]
            if all(abs(p[i] - c[i]) < h[i] + margin for i in range(3)):
                return True
        for c in course["cylinders"]:
            if np.hypot(p[0] - c["center"][0], p[1] - c["center"][1]) < c["radius"] + margin and p[2] < c["height"]:
                return True
    return False


def generate(seed: int, difficulty: str = "medium") -> dict:
    if difficulty not in DIFFICULTY:
        raise ValueError(f"difficulty must be one of {tuple(DIFFICULTY)}")
    for attempt in range(200):
        rng = np.random.default_rng([int(seed), {"easy": 1, "medium": 2, "hard": 3}[difficulty], attempt])
        layout = _sample_layout(rng, difficulty)
        start = [3.0, float(rng.uniform(-3.0, 3.0)), 2.0]
        goal = [float(rng.uniform(56.0, 58.0)), float(rng.uniform(-6.0, 6.0)), float(rng.uniform(2.0, 4.0))]
        course = dict(layout, start=start, goal=goal, difficulty=difficulty, attempt=attempt)
        if solvable(course):
            return course
    raise RuntimeError("could not generate a solvable course")


def public_view(course: dict) -> dict:
    """What the 'site survey' reveals: obstacle geometry only."""
    return dict(
        bounds=dict(x=list(YARD_X), y=list(YARD_Y), ceiling=CEILING),
        boxes=[dict(kind=b["kind"], center=list(b["center"]), half_extents=list(b["half_extents"])) for b in course["boxes"]],
        cylinders=[dict(kind=c["kind"], center=list(c["center"]), radius=c["radius"], height=c["height"]) for c in course["cylinders"]],
    )
