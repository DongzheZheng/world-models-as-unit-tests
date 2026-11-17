"""aspire_style policy v2 (take_cover). Repairs after life 1 (v1, 296 tics, killed by ONE fireball, 105 dmg):

S1 track-out-of-view: fireballs leave the 90 deg FOV ~200 units before they arrive; v1 forgot them. v2 keeps a
   track per object id (constant velocity, measured exact in the log) and extrapolates while unseen.
S2 drift-respects-threats: v1's "return to centre" ignored blocked intervals and walked into a ball. v2 picks
   one target lateral offset from a single cost (distance + weak centre preference) subject to safety.
S3 commit-with-momentum: strafing accelerates v_{k+1} ~ 0.64 v_k + 2.5 per step (terminal ~7 u/tic) and coasts
   ~9.5 x v units after release, measured in life 1; v2 steers on (target - coast) and only switches target
   with hysteresis, to stop left/right chattering.
"""
import math

NOOP, LEFT, RIGHT = 0, 1, 2


class Policy:
    def __init__(self, R=34.0, horizon=70.0, wall_margin=36.0, w_center=0.25, coast=9.5, vterm=6.5,
                 ramp=6.0, switch_gain=30.0):
        self.R, self.horizon, self.wall_margin, self.w_center = R, horizon, wall_margin, w_center
        self.coast, self.vterm, self.ramp, self.switch_gain = coast, vterm, ramp, switch_gain
        self.walls = None

    def bind(self, scenario, action_names):
        from reallab.doom import api
        self.walls = api.floor_plan(scenario)["walls"]

    def reset(self):
        self.tracks = {}
        self.target = None

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

    def _tau(self, d):
        """tics needed to shift laterally by d from (near) rest; closed form of the measured ramp."""
        return abs(d) / self.vterm + self.ramp

    def act(self, obs):
        p = obs["player"]
        tic = obs["tic"]
        a = math.radians(p["angle"])
        fx, fy = math.cos(a), math.sin(a)
        lx, ly = -fy, fx
        x, y = p["x"], p["y"]
        vl = p["vx"] * lx + p["vy"] * ly
        # --- S1: update projectile tracks
        for o in obs.get("objects", []):
            if math.hypot(o["vx"], o["vy"]) > 2.0:
                self.tracks[o["id"]] = (o["x"], o["y"], o["vx"], o["vy"], tic)
            elif o["id"] in self.tracks:  # exploded / stopped
                del self.tracks[o["id"]]
        room_left = self._ray(x, y, lx, ly) - self.wall_margin
        room_right = -(self._ray(x, y, -lx, -ly) - self.wall_margin)
        blocked = []
        for k, (ox, oy, ovx, ovy, t0) in list(self.tracks.items()):
            dt = tic - t0
            px, py = ox + ovx * dt, oy + ovy * dt
            rx, ry = px - x, py - y
            df = rx * fx + ry * fy
            close = -(ovx * fx + ovy * fy)
            if df < -30 or dt > 120 or close < 1.0:
                del self.tracks[k]
                continue
            t = max(df, 0.0) / close
            if t > self.horizon:
                continue
            lhit = rx * lx + ry * ly + (ovx * lx + ovy * ly) * t
            blocked.append((lhit - self.R, lhit + self.R, t))
        mid = 0.5 * (room_left + room_right)

        def feasible(c):
            if c > room_left + 1e-6 or c < room_right - 1e-6:
                return False
            a0, a1 = min(0.0, c), max(0.0, c)
            for lo, hi, t in blocked:
                if lo < c < hi:
                    return False
                if hi <= a0 or lo >= a1:
                    continue  # not on the way
                near, far = (lo, hi) if c > 0 else (hi, lo)
                enter = 0.0 if lo <= 0 <= hi else self._tau(near)
                exit_ = self._tau(far)
                if not (exit_ < t - 2 or enter > t + 4):
                    return False
            return True

        def cost(c):
            return abs(c) + self.w_center * abs(c - mid)

        cands = {0.0, max(min(mid, room_left), room_right), room_left, room_right}
        for lo, hi, _ in blocked:
            cands.add(lo - 2.0)
            cands.add(hi + 2.0)
        if self.target is not None:
            cands.add(self.target)
        feas = [c for c in cands if feasible(c)]
        if feas:
            best = min(feas, key=cost)
            if self.target is not None and self.target in feas and cost(self.target) <= cost(best) + self.switch_gain:
                best = self.target
        else:  # nothing provably safe: run for the side with more room / fewer imminent balls
            best = room_left if room_left > -room_right else room_right
        # keep target in world-lateral terms: store relative offset for next step
        err = best - self.coast * vl
        self.target = best - vl * 4.0  # offset expected after this step
        if not feas:
            self.target = None
        if err > 6:
            return LEFT
        if err < -6:
            return RIGHT
        return NOOP


def make_policy(**kw):
    return Policy(**kw)
