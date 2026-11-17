"""Exploration policy for life 1 of the neural-WM pilot: a hand-written attitude/altitude stabiliser built only from the
datasheet (hover T=0.30, command semantics) plus random excitation (random speed/lateral set-points, thrust dither).
It never looks at the learned model and uses no knowledge of the hidden effects."""
import numpy as np
from experiments.drone.baselines.common import _rot


class Explorer:
    def __init__(self, seed=0, vmax=3.0, dither=0.07, rate_noise=0.5, kp=6.0, z_ref=2.0):
        self.rng = np.random.default_rng(seed)
        self.vmax, self.dither, self.kp, self.z_ref, self.rate_noise = vmax, dither, kp, z_ref, rate_noise
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.k = 0
        self.vdes = 1.0
        self.yref = 0.0
        self.tdit = 0.0
        self.idx = 0
        self.rn = np.zeros(3)

    def act(self, obs):
        if self.k % 30 == 0:                      # new random set-points every second
            self.vdes = self.rng.uniform(0.3, self.vmax)
            self.yref = self.rng.uniform(-1.0, 1.0)
            self.tdit = self.rng.uniform(-self.dither, self.dither)
        if self.k % 4 == 0:                      # state-independent dither (identifies the command->response maps)
            self.tdit = self.rng.uniform(-self.dither, self.dither)
            self.rn = self.rng.normal(0, self.rate_noise, 3)
        self.k += 1
        R = _rot(obs["quat"])
        up = R[2]
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pos, vel, w = obs["pos"], obs["lin_vel"], obs["ang_vel"]
        # desired tilt (forward accel / lateral accel), world frame (yaw ~ 0 assumed)
        th_ref = np.clip(0.10 * (self.vdes - vel[0]), -0.15, 0.30)          # nose down -> +x
        ph_ref = np.clip(0.10 * (pos[1] - self.yref) + 0.25 * vel[1], -0.2, 0.2)    # right side down -> -y
        ux_ref, uy_ref = -np.sin(th_ref), np.sin(ph_ref)
        p = -self.kp * (up[1] - uy_ref) - 0.3 * w[0]
        q = self.kp * (up[0] - ux_ref) - 0.3 * w[1]
        r = -1.0 * yaw - 0.2 * w[2]
        p, q, r = p + self.rn[0], q + self.rn[1], r + self.rn[2]
        T = 0.30 + 0.08 * (self.z_ref - pos[2]) - 0.06 * vel[2] + self.tdit
        T = T / max(up[2], 0.7)
        return np.array([np.clip(p, -np.pi, np.pi), np.clip(q, -np.pi, np.pi), np.clip(r, -np.pi, np.pi), np.clip(T, 0.05, 0.6)])


def explorer(**kw):
    return Explorer(**kw)
