"""Scripted stroke probes for the extended lives (lab policy protocol, no lab import).
Schedule file (JSON list): [[x, y, vcmd_m_per_s, n_steps], ...]; the commanded point moves towards (x, y) at most
vcmd m/s (vcmd >= 4 means a direct jump) and is held there for the rest of the segment's n_steps."""
import json

import numpy as np


class Script:
    def __init__(self, path):
        self.S = json.load(open(path))

    def reset(self):
        self.k, self.n, self.c = 0, 0, None

    def act(self, obs):
        if self.c is None:
            self.c = np.array(obs["tool_pos"], dtype=float)
        if self.k >= len(self.S):
            return self.c.copy()
        x, y, v, n = self.S[self.k]
        tgt = np.array([x, y], dtype=float)
        d = tgt - self.c
        L = np.linalg.norm(d)
        step = v * 0.02
        self.c = tgt if (v >= 4 or L <= step) else self.c + d / L * step
        self.n += 1
        if self.n >= n:
            self.k, self.n = self.k + 1, 0
        return self.c.copy()


def script(path):
    return Script(path)
