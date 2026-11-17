"""Internal (lab-only): the SITE task -- fixed canyon yard, hidden anomalies.  The world-model author must never read this."""
from __future__ import annotations

import numpy as np

import pybullet as pb

from . import _course
from ._port_env import CONTROL_HZ, PortEnv

CONTAINER_H = 3 * 2.59
TIME_S = 60.0

# ---------------------------------------------------------------- public geometry (same for every episode)
WALL_Y0, WALL_Y1 = 2.0, 12.0
NORTH_GAP = (22.0, 25.5)       # side aisle in the north wall (y > 0); the strong jet comes out of it
SOUTH_GAP = (38.0, 41.5)       # side aisle in the south wall
WALL_X = (10.0, 50.0)
PINCH = (31.0, 34.0, 0.4)      # x0, x1, protrusion of each wall (m): clear width 4 -> 3.2 m

MISSIONS = {
    "A": [(14.0, 0.0, 2.5), (28.0, 0.0, 2.5), (57.0, 0.0, 2.5)],
    "B": [(17.0, 0.0, 2.5), (23.75, 6.0, 2.5), (57.0, 0.0, 2.5)],
}
MISSIONS["C"] = MISSIONS["A"]     # same relays as A (stochastic movers on top)


def _slab(x0, x1, y0, y1, kind="container_stack"):
    return dict(kind=kind, center=[(x0 + x1) / 2, (y0 + y1) / 2, CONTAINER_H / 2],
                half_extents=[(x1 - x0) / 2, (y1 - y0) / 2, CONTAINER_H / 2])


def boxes():
    b = []
    nx = [(WALL_X[0], NORTH_GAP[0]), (NORTH_GAP[1], WALL_X[1])]
    sx = [(WALL_X[0], SOUTH_GAP[0]), (SOUTH_GAP[1], WALL_X[1])]
    for x0, x1 in nx:
        b.append(_slab(x0, x1, WALL_Y0, WALL_Y1))
    for x0, x1 in sx:
        b.append(_slab(x0, x1, -WALL_Y1, -WALL_Y0))
    p0, p1, d = PINCH
    b.append(_slab(p0, p1, WALL_Y0 - d, WALL_Y0))
    b.append(_slab(p0, p1, -WALL_Y0, -WALL_Y0 + d))
    return b


def site_map() -> dict:
    return dict(
        bounds=dict(x=[0.0, 60.0], y=[-12.0, 12.0], ceiling=6.0),
        boxes=[dict(kind=k["kind"], center=list(k["center"]), half_extents=list(k["half_extents"])) for k in boxes()],
        cylinders=[],
        start=[3.0, 0.0, 2.0],
        goal=[57.0, 0.0, 2.5],
        missions={m: [list(p) for p in v] for m, v in MISSIONS.items() if m == "A"},   # B is held out
        time_limit_s=TIME_LIMITS["A"],
    )


def site_map_lab() -> dict:
    d = site_map()
    d["missions"] = {m: [list(p) for p in v] for m, v in MISSIONS.items()}
    return d


# ---------------------------------------------------------------- hidden truth
ROTOR = 2                 # motor index (front-left: +x, +y)
ROTOR_EFF = 0.60          # thrust and torque coefficient scale
JET1 = dict(xc=23.75, sx=1.8, dir=(0.0, -1.0, 0.0), U=40.0, yc=1.0, sy=6.0)    # out of the north gap, blowing toward -y
JET2 = dict(xc=39.75, sx=1.8, dir=(0.0, 1.0, 0.0), U=30.0, yc=-1.0, sy=6.0)    # out of the south gap, blowing toward +y
WIND_COEF = 0.01          # N per (m/s) of relative air speed (same as port)
TURB_AMP = 0.25           # relative OU fluctuation of the jet speed
TURB_VEC = 0.8            # m/s OU vector turbulence std
TURB_TAU = 0.4


TIME_LIMITS = {"A": 17.0, "B": 24.2, "C": 19.4}       # s, per mission: L = min(0.8 T_ap, 1.5 T_c), see _site_validate
import json as _json, os as _os            # TUNING HOOK (lab-only)
_T = _json.loads(_os.environ.get("CAWM_SITE_TUNE", "{}"))
ROTOR_EFF = _T.get("eff", ROTOR_EFF)
JET1["U"] = _T.get("u1", JET1["U"]); JET2["U"] = _T.get("u2", JET2["U"])
TIME_LIMITS["A"] = _T.get("LA", TIME_LIMITS["A"]); TIME_LIMITS["B"] = _T.get("LB", TIME_LIMITS["B"]); TIME_LIMITS["C"] = _T.get("LC", TIME_LIMITS["C"])


def truth() -> dict:
    return dict(rotor=ROTOR, rotor_eff=ROTOR_EFF, jet1=JET1, jet2=JET2)


def _profile(j, p):
    fx = np.exp(-((p[0] - j["xc"]) / j["sx"]) ** 2)
    fy = np.exp(-((p[1] - j["yc"]) / j["sy"]) ** 2)
    fz = 1.0 - 0.6 / (1.0 + np.exp(-(p[2] - 4.5) / 0.5))
    return fx * fy * fz


SITE_CFG = dict(wind=False, param_var=False, range_noise=0.01, dropout=0.005, pos=0.02, vel=0.02, gyro=0.005,
                quat=0.001, latency=1)


class SiteEnv(PortEnv):
    CFG = SITE_CFG
    TIME_STEPS = int(TIME_S * CONTROL_HZ)
    RELAY_BONUS = 20.0
    HAS_EXTRA = True

    def __init__(self, mission: str = "A", lab: dict | None = None, purpose: str = "smoke", flight_mode: int = 0, **kw):
        anomalies = (lab or {}).get("anomalies", True)
        if anomalies is not True and purpose != "smoke":
            raise ValueError("lab option only available for purpose='smoke'")
        if mission not in MISSIONS:
            raise ValueError("mission must be 'A', 'B' or 'C'")
        self.mission, self.anomalies = mission, bool(anomalies)
        super().__init__(difficulty="medium", realism=True, flight_mode=flight_mode, **kw)
        self.time_limit_s = TIME_LIMITS[mission]
        self.max_steps = int(round(self.time_limit_s * CONTROL_HZ))
        if self.flight_mode == 7:
            self.action_low = np.array([-2.0, -14.0, -np.pi, 0.0])
            self.action_high = np.array([62.0, 14.0, np.pi, 6.0])

    def _make_course(self):
        rng = np.random.default_rng([self.seed, 99])
        j = rng.uniform(-1.0, 1.0, size=(4, 3)) * np.array([1.0, 1.0, 0.0])
        start = [3.0 + j[0, 0], 0.0 + j[0, 1], 2.0]
        relays = []
        for i, r in enumerate(MISSIONS[self.mission]):
            d = rng.uniform(-0.5, 0.5, 3) * np.array([1.0, 1.0, 0.4])
            relays.append([float(r[0] + d[0]), float(r[1] + d[1]), float(r[2] + d[2])])
        return dict(boxes=boxes(), cylinders=[], start=[float(v) for v in start], goal=relays[-1], relays=relays)

    def _after_build(self):
        self.ou_amp = np.zeros(2)
        self.ou_vec = np.zeros((2, 3))
        if self.anomalies:
            m = self.drone.motors
            m.thrust_coef = m.thrust_coef.copy()
            m.torque_coef = m.torque_coef.copy()
            m.thrust_coef[ROTOR] *= ROTOR_EFF
            m.torque_coef[ROTOR] *= ROTOR_EFF

    def _extra_force(self, pos, vel_world):
        if not self.anomalies:
            return None
        dt = 1.0 / 240.0
        a = np.exp(-dt / TURB_TAU)
        s = np.sqrt(1 - a * a)
        g = self.rng
        self.ou_amp = a * self.ou_amp + s * g.standard_normal(2)
        self.ou_vec = a * self.ou_vec + s * g.standard_normal((2, 3))
        f = np.zeros(3)
        for i, j in enumerate((JET1, JET2)):
            w = _profile(j, pos)
            if w < 1e-4:
                continue
            u = j["U"] * (1.0 + TURB_AMP * self.ou_amp[i]) * np.array(j["dir"]) + TURB_VEC * self.ou_vec[i]
            f += WIND_COEF * w * (u - vel_world)
        return f

    # ------------------------------------------------------------------ movers (mission C only)
    def _build_world(self, env):
        super()._build_world(env)
        self.movers = None
        if self.mission == "C":
            from . import _movers
            self.movers = _movers.Movers(self.seed, np.array(self.course["relays"]), np.array(self.course["start"]))
            col = env.createCollisionShape(pb.GEOM_SPHERE, radius=_movers.RADIUS)
            self.mover_ids = [env.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=p.tolist())
                              for p in self.movers.pos]
            self.mover_rng = np.random.default_rng([self.seed, 4243])

    def _pre_step(self):
        if self.movers is not None:
            self.movers.step(1.0 / CONTROL_HZ, np.array(self.drone.state[3]))

    def _pre_physics(self, k):
        if self.movers is not None:
            f = (k + 1) / 8.0
            for bid, a, b in zip(self.mover_ids, self.movers.prev_pos, self.movers.pos):
                self.env.resetBasePositionAndOrientation(bid, (a + (b - a) * f).tolist(), [0, 0, 0, 1])

    def mover_truth(self):
        return self.movers.pos.copy(), self.movers.vel.copy()

    def _extra_obs(self, obs):
        obs["relay_index"] = np.array([float(min(self.k, 2))])
        obs["relays"] = self.targets.copy()
        if self.movers is not None:
            from . import _movers
            p = np.array(self.drone.state[3])
            P, V = self.movers.pos, self.movers.vel
            d = np.linalg.norm(P - p, axis=1)
            idx = [i for i in np.argsort(d) if d[i] <= 15.0]
            if idx:
                res = self.env.rayTestBatch([p.tolist()] * len(idx), [P[i].tolist() for i in idx], numThreads=1)
                idx = [i for i, r in zip(idx, res) if r[0] == self.mover_ids[i]]
            rows = np.zeros((8, 7))
            g = self.mover_rng
            for j, i in enumerate(idx[:8]):
                rows[j, :3] = P[i] + 0.03 * g.standard_normal(3)
                rows[j, 3:6] = V[i] + 0.05 * g.standard_normal(3)
                rows[j, 6] = _movers.RADIUS
            obs["movers"] = rows
            obs["n_movers"] = np.array([float(min(len(idx), 8))])
        return obs

    def step(self, action):
        out = super().step(action)
        if out[4]:
            out[4]["time_limit_s"] = self.time_limit_s
        return out
