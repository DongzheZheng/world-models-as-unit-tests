"""Internal: the PORT course, built in PyBullet on top of PyFlyt's QuadX (lab-side only)."""
from __future__ import annotations

import contextlib
import os

import numpy as np
import pybullet as pb

from PyFlyt.core.aviary import Aviary

from . import _course

CONTROL_HZ = 30
PHYS_PER_STEP = 8            # 240 Hz physics / 30 Hz agent
MAX_STEPS = 40 * CONTROL_HZ  # 40 s
GOAL_RADIUS = 1.0
N_EL, N_AZ, MAX_RANGE = 6, 16, 12.0
RAY_OFFSET = 0.10            # rays start this far from the drone centre (outside its own collision box)
EL = np.deg2rad(np.linspace(-25.0, 25.0, N_EL))
AZ = np.deg2rad(np.linspace(-60.0, 60.0, N_AZ))
_EL, _AZ = np.meshgrid(EL, AZ, indexing="ij")           # (6,16): [elevation_idx, azimuth_idx]
_DIR_LOCAL = np.stack([np.cos(_EL) * np.cos(_AZ), np.cos(_EL) * np.sin(_AZ), np.sin(_EL)], -1).reshape(-1, 3)
HOVER_T = 0.2988              # lab-side knowledge: nominal hover throttle (measured: zero vertical drift, realism off)

# hidden-realism magnitudes (lab-side constants)
WIND_STEADY_MAX = 2.0        # m/s horizontal, uniform direction, magnitude U(0, 2)
WIND_STEADY_Z = 0.2          # m/s vertical std
GUST_SIGMA = (0.7, 0.7, 0.3) # m/s OU gust std (x, y, z)
GUST_TAU = 1.0               # s
PULSE_RATE = 0.15            # strong gust events per second
PULSE_MAG = (1.0, 2.5)       # m/s
PULSE_TAU = 0.6              # s
WIND_COEF = 0.01             # N per (m/s) of relative air velocity
MASS_VAR = 0.10              # uniform +-10 %
THRUST_VAR = 0.10            # uniform +-10 %
RANGE_NOISE = 0.02           # multiplicative sigma
RANGE_DROPOUT = 0.02         # per-ray per-step probability of reading max range
POS_NOISE, VEL_NOISE, GYRO_NOISE, QUAT_NOISE = 0.05, 0.05, 0.01, 0.002
LATENCY_STEPS = 1
PORT_CFG = dict(wind=True, param_var=True, range_noise=RANGE_NOISE, dropout=RANGE_DROPOUT, pos=POS_NOISE, vel=VEL_NOISE,
                gyro=GYRO_NOISE, quat=QUAT_NOISE, latency=LATENCY_STEPS)


@contextlib.contextmanager
def quiet():
    """Silence pybullet's C-level stdout chatter while connecting."""
    try:
        sys_fd = os.dup(1)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        os.dup2(sys_fd, 1)
        os.close(sys_fd)
        os.close(null)


def yaw_of(q):
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class PortEnv:
    CFG = PORT_CFG
    TIME_STEPS = MAX_STEPS
    RELAY_BONUS = 0.0
    HAS_EXTRA = False
    action_low = np.array([-np.pi, -np.pi, -np.pi, 0.0])
    action_high = np.array([np.pi, np.pi, np.pi, 0.8])

    def __init__(self, difficulty: str = "medium", realism: bool = True, flight_mode: int = 0,
                 max_steps: int | None = None, course: dict | None = None):
        if difficulty not in _course.DIFFICULTY:
            raise ValueError(f"difficulty must be one of {tuple(_course.DIFFICULTY)}")
        self.difficulty, self.realism, self.flight_mode = difficulty, bool(realism), int(flight_mode)
        self.max_steps = min(self.TIME_STEPS, int(max_steps)) if max_steps else self.TIME_STEPS
        self._course_override = course
        if self.flight_mode != 0:
            self.action_low = np.full(4, -np.inf)
            self.action_high = np.full(4, np.inf)
        self.env = None
        self.course = None

    # -------------------------------------------------------------- lifecycle
    def close(self):
        if self.env is not None:
            try:
                self.env.disconnect()
            except Exception:
                pass
            self.env = None

    def reset(self, seed: int):
        self.close()
        self.seed = int(seed)
        self.course = self._course_override or self._make_course()
        crng = np.random.default_rng([self.seed, 77])       # realism stream
        self.rng = crng
        s = self.course["start"]
        sim_rng = np.random.default_rng([self.seed, 5])
        with quiet():
            env = Aviary(start_pos=np.array([s]), start_orn=np.zeros((1, 3)), drone_type="quadx",
                         render=False, drone_options={}, np_random=sim_rng)
        self.env = env
        self._build_world(env)
        env.register_all_new_bodies()
        env.set_mode(self.flight_mode)
        self.drone = env.drones[0]
        # hidden realism: per-episode parameter variation + wind
        self.s_mass = self.s_thrust = 1.0
        if self.realism and self.CFG["param_var"]:
            self.s_mass = 1.0 + crng.uniform(-MASS_VAR, MASS_VAR)
            self.s_thrust = 1.0 + crng.uniform(-THRUST_VAR, THRUST_VAR)
            info = env.getDynamicsInfo(self.drone.Id, -1)
            env.changeDynamics(self.drone.Id, -1, mass=info[0] * self.s_mass,
                               localInertiaDiagonal=[i * self.s_mass for i in info[2]])
            self.drone.motors.thrust_coef = self.drone.motors.thrust_coef * self.s_thrust
            ang = crng.uniform(0, 2 * np.pi)
            mag = crng.uniform(0, WIND_STEADY_MAX)
            self.wind_steady = np.array([mag * np.cos(ang), mag * np.sin(ang), crng.normal(0, WIND_STEADY_Z)])
            self.gust = np.zeros(3)
            self.pulse = np.zeros(3)
        self._after_build()
        # start hovering: spin motors to the (true) hover throttle, zero velocity, no latency transient
        hover = HOVER_T * np.sqrt(self.s_mass / self.s_thrust)
        self.drone.motors.throttle[:] = hover
        if self.flight_mode == 0:
            self._hold = np.array([0.0, 0.0, 0.0, hover])
        else:
            self._hold = np.array([s[0], s[1], 0.0, s[2]])
        self.drone.setpoint = self._hold.copy()
        self._queue = [self._hold.copy() for _ in range(self.CFG['latency'] if self.realism else 0)]
        self.n = 0
        self.targets = np.array(self.course.get("relays", [self.course["goal"]]), dtype=np.float64)
        self.k = 0
        self.goal = self.targets[0].copy()
        self.path_length = 0.0
        self.min_range = MAX_RANGE
        self._prev_pos = np.array(self.drone.state[3])
        self._prev_dist = float(np.linalg.norm(self.goal - self._prev_pos))
        self.done = False
        obs = self._obs()
        self.min_range = float(obs["ranges"].min())
        info = {"seed": self.seed}
        return obs, info

    def _make_course(self):
        return _course.generate(self.seed, self.difficulty)

    def _after_build(self):
        pass

    def _extra_force(self, pos, vel_world):
        return None

    def _extra_obs(self, obs):
        return obs

    def _pre_step(self):
        pass

    def _pre_physics(self, k):
        pass

    def _build_world(self, env):
        for b in self.course["boxes"]:
            col = env.createCollisionShape(pb.GEOM_BOX, halfExtents=b["half_extents"])
            env.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=b["center"])
        for c in self.course["cylinders"]:
            col = env.createCollisionShape(pb.GEOM_CYLINDER, radius=c["radius"], height=c["height"])
            env.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=c["center"])

    # -------------------------------------------------------------- sensing
    def _ranges(self, pos, yaw) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        d = _DIR_LOCAL @ R.T
        frm = pos + RAY_OFFSET * d
        to = pos + MAX_RANGE * d
        res = self.env.rayTestBatch(frm.tolist(), to.tolist(), numThreads=1)
        frac = np.fromiter((r[2] for r in res), dtype=np.float64, count=len(res))
        hit = np.fromiter((r[0] >= 0 for r in res), dtype=bool, count=len(res))
        rng = np.where(hit, RAY_OFFSET + frac * (MAX_RANGE - RAY_OFFSET), MAX_RANGE)
        return rng.reshape(N_EL, N_AZ)

    def _true_state(self):
        st = self.drone.state                 # ang_vel(body), euler, lin_vel(body), pos
        quat = np.array(pb.getQuaternionFromEuler(st[1]))
        R = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
        return np.array(st[0]), quat, R @ np.array(st[2]), np.array(st[3])

    def _obs(self) -> dict:
        w, q, v, p = self._true_state()
        r = self._ranges(p, yaw_of(q))
        if self.realism:
            g = self.rng
            c = self.CFG
            r = np.minimum(r * (1.0 + c["range_noise"] * g.standard_normal(r.shape)), MAX_RANGE)
            r = np.where(g.random(r.shape) < c["dropout"], MAX_RANGE, r)
            p = p + c["pos"] * g.standard_normal(3)
            v = v + c["vel"] * g.standard_normal(3)
            w = w + c["gyro"] * g.standard_normal(3)
            q = q + c["quat"] * g.standard_normal(4)
            q = q / np.linalg.norm(q)
        return self._extra_obs(dict(ang_vel=w, quat=q, lin_vel=v, pos=p, goal=self.goal.copy(), ranges=r,
                                    t=np.array([self.n / CONTROL_HZ])))

    # -------------------------------------------------------------- dynamics
    def _wind_force(self, pos, vel_world):
        dt = 1.0 / 240.0
        a = np.exp(-dt / GUST_TAU)
        self.gust = a * self.gust + np.sqrt(1 - a * a) * np.array(GUST_SIGMA) * self.rng.standard_normal(3)
        self.pulse *= np.exp(-dt / PULSE_TAU)
        if self.rng.random() < PULSE_RATE * dt:
            th = self.rng.uniform(0, 2 * np.pi)
            self.pulse += self.rng.uniform(*PULSE_MAG) * np.array([np.cos(th), np.sin(th), 0.0])
        wind = self.wind_steady + self.gust + self.pulse
        return WIND_COEF * (wind - vel_world)

    def step(self, action):
        env, drone = self.env, self.drone
        a = np.asarray(action, dtype=np.float64)
        if self.realism:
            self._queue.append(a.copy())
            applied = self._queue.pop(0)
        else:
            applied = a
        drone.setpoint = applied
        collided = False
        self._pre_step()
        for _k in range(PHYS_PER_STEP):
            drone.update_control(env.physics_steps)
            drone.update_physics()
            pos = drone.state[3]
            f = None
            if self.realism and self.CFG["wind"]:
                vw = np.array(env.getBaseVelocity(drone.Id)[0])
                f = self._wind_force(pos, vw)
            if self.HAS_EXTRA:
                vw = np.array(env.getBaseVelocity(drone.Id)[0])
                x = self._extra_force(pos, vw)
                if x is not None:
                    f = x if f is None else f + x
            if f is not None:
                env.applyExternalForce(drone.Id, -1, f.tolist(), list(pos), pb.WORLD_FRAME)
            self._pre_physics(_k)
            env.stepSimulation()
            drone.update_state()
            env.physics_steps += 1
            if not collided and env.getContactPoints(bodyA=drone.Id):
                collided = True
        env.elapsed_time = env.physics_steps / env.physics_hz
        self.n += 1
        w, q, v, p = self._true_state()
        self.path_length += float(np.linalg.norm(p - self._prev_pos))
        self._prev_pos = p
        dist = float(np.linalg.norm(self.goal - p))
        oob = bool(p[0] < -2.0 or p[0] > 62.0 or p[1] < -14.0 or p[1] > 14.0 or p[2] > _course.CEILING)
        reward = (self._prev_dist - dist) - 0.01
        success = False
        if dist < GOAL_RADIUS and not collided:
            if self.k == len(self.targets) - 1:
                success = True
                self.k += 1
            else:
                self.k += 1
                self.goal = self.targets[self.k].copy()
                reward += self.RELAY_BONUS
                dist = float(np.linalg.norm(self.goal - p))
        timeout = bool(self.n >= self.max_steps and not (success or collided or oob))
        self._prev_dist = dist
        if success:
            reward += 50.0
        if collided or oob:
            reward -= 50.0
        obs = self._obs()
        self.min_range = min(self.min_range, float(obs["ranges"].min()))
        terminated = bool(success or collided or oob)
        truncated = timeout
        info: dict = {}
        if terminated or truncated:
            t = self.n / CONTROL_HZ
            info = dict(success=success, collision=bool(collided), out_of_bounds=bool(oob and not collided), timeout=timeout,
                        time_s=t, path_length_m=self.path_length, mean_speed=self.path_length / t,
                        min_range_seen=self.min_range)
            if len(self.targets) > 1:
                info["relays_reached"] = int(self.k)
            self.done = True
        return obs, float(reward), terminated, truncated, info
