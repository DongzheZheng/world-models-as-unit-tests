"""aspire_style policy v1 (take_cover): reactive dodger written from prior knowledge + floor plan, before any life.

Lateral axis = player's left. Every visible object that is closing in along the heading is treated as a
projectile; its lateral offset when it reaches the player's plane is predicted in closed form (straight line,
constant velocity). Offsets within R of a predicted impact are blocked; the policy strafes towards the nearest
unblocked lateral offset inside the walls. With no threat it drifts back towards the middle of the wall.
"""
import math

NOOP, LEFT, RIGHT = 0, 1, 2


class Policy:
    def __init__(self, R=40.0, horizon=60.0, wall_margin=40.0, center_band=120.0, speed=None):
        self.R, self.horizon, self.wall_margin, self.center_band = R, horizon, wall_margin, center_band
        self.speed = speed  # lateral units/tic estimate (None -> learn online from own motion)
        self.walls = None

    def bind(self, scenario, action_names):
        from reallab.doom import api
        self.walls = api.floor_plan(scenario)["walls"]

    def reset(self):
        self.vmax = self.speed or 8.0

    def _ray(self, x, y, dx, dy):
        best = 1e9
        for w in self.walls or []:
            ex, ey = w["x2"] - w["x1"], w["y2"] - w["y1"]
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = ((w["x1"] - x) * ey - (w["y1"] - y) * ex) / den
            u = ((w["x1"] - x) * dy - (w["y1"] - y) * dx) / den
            if t > 0 and -1e-6 <= u <= 1 + 1e-6:
                best = min(best, t)
        return best

    def act(self, obs):
        p = obs["player"]
        a = math.radians(p["angle"])
        fx, fy = math.cos(a), math.sin(a)
        lx, ly = -fy, fx
        x, y = p["x"], p["y"]
        vl = p["vx"] * lx + p["vy"] * ly
        self.vmax = max(self.vmax, abs(vl))
        room_left = self._ray(x, y, lx, ly) - self.wall_margin
        room_right = -(self._ray(x, y, -lx, -ly) - self.wall_margin)
        blocked = []
        for o in obs.get("objects", []):
            rx, ry = o["x"] - x, o["y"] - y
            df = rx * fx + ry * fy
            close = -(o["vx"] * fx + o["vy"] * fy)
            if close < 2.0 or df < -10:
                continue
            t = max(df, 0.0) / close
            if t > self.horizon:
                continue
            lhit = rx * lx + ry * ly + (o["vx"] * lx + o["vy"] * ly) * t
            blocked.append((lhit - self.R, lhit + self.R, t))
        if not blocked or all(not (lo < 0 < hi) for lo, hi, _ in blocked):
            # safe now; move away from walls / back to middle
            mid = 0.5 * (room_left + room_right)
            if mid > self.center_band:
                return LEFT
            if mid < -self.center_band:
                return RIGHT
            return NOOP
        cands = [0.0] + [b[0] - 1 for b in blocked] + [b[1] + 1 for b in blocked]
        best, bestc = None, 1e18
        for c in cands:
            if c > room_left or c < room_right:
                continue
            if any(lo < c < hi for lo, hi, _ in blocked):
                continue
            # reachability: penalise targets needing more time than the earliest threat on the way allows
            tneed = abs(c) / max(self.vmax, 1.0)
            tmin = min((t for lo, hi, t in blocked if min(0, c) <= hi and lo <= max(0, c)), default=1e9)
            cost = abs(c) + (1e4 if tneed > tmin + 2 else 0.0)
            if cost < bestc:
                best, bestc = c, cost
        if best is None:
            best = room_left if room_left > -room_right else room_right
        return LEFT if best > 0 else RIGHT


def make_policy(**kw):
    return Policy(**kw)
