"""aspire_style policy v2 (take_cover). Repairs after life 1 (v1, 296 tics, killed by ONE fireball, 105 dmg):

S1 track-out-of-view: fireballs leave the 90 deg FOV ~200 units before they arrive; v1 forgot them. v2 keeps a
   track per object id (constant velocity, measured exact in the log) and extrapolates while unseen.
S2 drift-respects-threats: v1's "return to centre" ignored blocked intervals and walked into a ball. v2 picks
   one target lateral offset from a single cost (distance + weak centre preference) subject to safety.
S3 commit-with-momentum: strafing accelerates v_{k+1} ~ 0.64 v_k + 2.5 per step (terminal ~7 u/tic) and coasts
   ~9.5 x v units after release, measured in life 1; v2 steers on (target - coast) and only switches target
   with hysteresis, to stop left/right chattering.

v3 repairs after life 2 (v2, 638 tics, killed by one ball at y=684 near the +y wall):
S4 wall-margin-from-body: the only safe offset (+25) was rejected by a 36-unit wall margin, though the body
   radius is ~16; margin -> 20.
S5 least-bad-escape: with no provably safe target v2 ran for the "roomier" side, straight into a ball 2 tics
   away. v3 scores every candidate by the arrival time of the EARLIEST threat it conflicts with and takes the
   candidate that conflicts latest (most time to re-plan).
S6 centre-pull: v2 lingered near the wall (y 650-710) where escapes are one-sided; w_center 0.25 -> 0.5.

v4 repairs after life 3 (v3, 811 tics, 3 hits: 21+63+84 dmg; damage per hit is variable, 21..168 seen):
S7 swept-box-collision: all 3 hits were oblique balls (|v_lat/v_close| ~ 0.4-0.75) that v3 judged a near miss
   because it tested a single point at the player's plane. Collision happens while the ball's forward
   coordinate is within B = 22 (body 16 + ball 6, box overlap) of the player's; v4 blocks the lateral span the
   ball sweeps over that whole window, widened by B + margin, with entry/exit times.
S8 reversal-cost: reaching an offset against current lateral velocity costs ~0.8 tic per unit/tic (life-1..3
   strafe logs: v_{k+1} = 0.64 v_k + 2.5 per step); tau() now includes it.
"""
import math

NOOP, LEFT, RIGHT = 0, 1, 2


class Policy:
    def __init__(self, B=22.0, margin=7.0, horizon=70.0, wall_margin=20.0, w_center=0.5, coast=9.5, vterm=6.5,
                 ramp=6.0, switch_gain=30.0):
        self.B, self.margin, self.horizon, self.wall_margin, self.w_center = B, margin, horizon, wall_margin, w_center
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
        """tics needed to shift laterally by d; closed form of the measured ramp (+ reversal cost)."""
        rev = 0.8 * abs(self.vl) if self.vl * d < 0 else -0.5 * abs(self.vl)
        return max(0.0, abs(d) / self.vterm + self.ramp + rev)

    def act(self, obs):
        p = obs["player"]
        tic = obs["tic"]
        a = math.radians(p["angle"])
        fx, fy = math.cos(a), math.sin(a)
        lx, ly = -fy, fx
        x, y = p["x"], p["y"]
        vl = p["vx"] * lx + p["vy"] * ly
        self.vl = vl
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
            if df < -(self.B + 4) or dt > 120 or close < 1.0:
                del self.tracks[k]
                continue
            t_in = max(df - self.B, 0.0) / close
            t_out = (df + self.B) / close
            if t_in > self.horizon:
                continue
            l0 = rx * lx + ry * ly
            vlat = ovx * lx + ovy * ly
            la, lb = l0 + vlat * t_in, l0 + vlat * t_out
            w = self.B + self.margin
            blocked.append((min(la, lb) - w, max(la, lb) + w, t_in, t_out))
        mid = 0.5 * (room_left + room_right)

        def conflict(c):
            """earliest arrival time of a threat that target c conflicts with (inf = safe)."""
            if c > room_left + 1e-6 or c < room_right - 1e-6:
                return -1.0
            worst = math.inf
            a0, a1 = min(0.0, c), max(0.0, c)
            for lo, hi, t, t_out in blocked:
                if lo < c < hi:
                    worst = min(worst, t)
                    continue
                if hi <= a0 or lo >= a1:
                    continue  # not on the way
                near, far = (lo, hi) if c > 0 else (hi, lo)
                enter = 0.0 if lo <= 0 <= hi else self._tau(near)
                exit_ = self._tau(far)
                if not (exit_ < t - 1 or enter > t_out + 1):
                    worst = min(worst, t)
            return worst

        def feasible(c):
            return conflict(c) == math.inf

        def cost(c):
            return abs(c) + self.w_center * abs(c - mid)

        cands = {0.0, max(min(mid, room_left), room_right), room_left, room_right}
        for lo, hi, _, _ in blocked:
            cands.add(lo - 2.0)
            cands.add(hi + 2.0)
        if self.target is not None:
            cands.add(self.target)
        feas = [c for c in cands if feasible(c)]
        if feas:
            best = min(feas, key=cost)
            if self.target is not None and self.target in feas and cost(self.target) <= cost(best) + self.switch_gain:
                best = self.target
        else:  # S5: nothing provably safe -> candidate whose first conflict is latest
            grid = list(cands) + [room_right + k * (room_left - room_right) / 24.0 for k in range(25)]
            best = max(grid, key=lambda c: (conflict(c), -abs(c)))
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
