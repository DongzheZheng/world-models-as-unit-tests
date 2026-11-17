"""Scripted probing policies for calibration lives (lab policy protocol). No lab import here."""
import numpy as np


class Life1Probe:
    """Life 1: system identification.
    phase A  t<10      hold                      -> quiescence
    phase B  t<70      slow drag in -y, 0.25 m/s -> friction at low speed, trailing behaviour
    phase C  t<85      hold                      -> does anything keep sliding after a slow drag?
    phase D  t<125     one large step command    -> servo lag at max accel, rope jerk, free sliding decel
    phase E  t<140     move back to x=0.30       -> wind-up
    phase F  t>=140    fast fling towards target, then hold -> sliding out, landing, coming to rest
    """

    def reset(self):
        self.t = 0
        self.start = None

    def act(self, obs):
        t = self.t
        self.t += 1
        if self.start is None:
            self.start = np.array(obs["tool_pos"], dtype=float)
        s = self.start
        if t < 10:
            return s.copy()
        if t < 70:
            y = max(s[1] - 0.25 * 0.02 * (t - 9), -0.15)
            return np.array([s[0], y])
        if t < 85:
            return np.array([s[0], max(s[1] - 0.25 * 0.02 * 60, -0.15)])
        if t < 125:
            return np.array([0.50, 0.15])
        if t < 140:
            return np.array([0.30, 0.10])
        tg = np.asarray(obs["target"])
        d = tg - np.array([0.30, 0.10])
        d /= np.linalg.norm(d)
        p = np.array([0.30, 0.10]) + 0.30 * d
        return np.clip(p, [0.30, -0.20], [0.55, 0.20])


def life1_probe():
    return Life1Probe()


class Replay:
    """Replays a fixed command sequence (np.ndarray (T,2)); holds the last command afterwards."""

    def __init__(self, path):
        self.A = np.load(path)

    def reset(self):
        self.t = 0

    def act(self, obs):
        a = self.A[min(self.t, len(self.A) - 1)]
        self.t += 1
        return np.array(a, dtype=float)


def replay(path):
    return Replay(path)
