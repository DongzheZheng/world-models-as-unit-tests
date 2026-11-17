# Life 1: scripted probe (speed steps while dragging the rope straight out, lateral drag, slack fast move, taut pull).
# Run:  python experiments/station/code_wm/life1_probe.py --pick   (choose seed from snapshots, no life used)
#       python experiments/station/code_wm/life1_probe.py --fly SEED
import sys
import numpy as np

DT = 0.05


class ProbePolicy:
    """Phases: ('hold', n) | ('move', (dx, dy), speed, n, guard_load) | ('jump', (dx, dy), n_hold) | ('until_x', speed, x_max, guard)."""

    def __init__(self):
        self.phases = [("hold", 15)]
        self.phases += [("move", (1, 0), v, n, 8.0) for v, n in
                        [(0.02, 20), (0.05, 20), (0.02, 15), (0.10, 15), (0.02, 15), (0.15, 12), (0.02, 15)]]
        self.phases += [("hold", 10), ("move", (1, 0), 0.20, 10, 8.0), ("move", (1, 0), 0.02, 15, 8.0),
                        ("move", (1, 0), 0.05, 20, 8.0)]
        self.phases += [("move", (0, -1), 0.05, 20, 8.0), ("move", (0, 1), 0.05, 20, 8.0), ("hold", 10)]
        self.phases += [("jump", (-0.12, 0.0), 20), ("hold", 10)]
        self.phases += [("until_x", 0.06, 1.15, 9.0), ("hold", 400)]

    def reset(self):
        self.k, self.n, self.ref = 0, 0, None

    def act(self, obs):
        tp = np.asarray(obs["tool_pos"], float)
        if self.ref is None:
            self.ref = tp.copy()
        if self.k >= len(self.phases):
            return self.ref.copy()
        ph = self.phases[self.k]
        kind = ph[0]
        done = False
        if kind == "hold":
            done = self.n >= ph[1]
        elif kind == "move":
            _, d, v, n, guard = ph
            if obs["load"] > guard and v > 0.03:
                self.ref = tp.copy(); done = True
            else:
                self.ref = self.ref + np.array(d, float) * v * DT
                done = self.n + 1 >= n
        elif kind == "jump":
            if self.n == 0:
                self.ref = self.ref + np.array(ph[1], float)
            done = self.n >= ph[2]
        elif kind == "until_x":
            _, v, xmax, guard = ph
            if obs["load"] > guard or self.ref[0] >= xmax:
                self.ref = tp.copy(); done = True
            else:
                self.ref = self.ref + np.array([v * DT, 0.0])
        self.n += 1
        if done:
            self.k += 1; self.n = 0
        return np.clip(self.ref, [0.0, -0.4], [1.2, 0.4])


def make_probe():
    return ProbePolicy()


def pick():
    L = np.load("runs/station/observation/layouts.npz")
    best = []
    for i in range(200):
        y0 = L["tool_pos"][i, 1]
        gap = np.abs(L["posts"][i, :, 1] - y0).min()
        best.append((gap, i, y0, L["tool_pos"][i, 0]))
    best.sort(reverse=True)
    for b in best[:8]:
        print("gap %.3f seed %d tool y %.3f x %.3f" % b)


if __name__ == "__main__":
    if sys.argv[1] == "--pick":
        pick()
    elif sys.argv[1] == "--fly":
        from reallab.station import parallel
        seed = int(sys.argv[2])
        res = parallel.evaluate("experiments.station.code_wm.life1_probe:make_probe", {}, seeds=[seed],
                                purpose="calibration", pilot="code_wm", tag="life1_probe", n_workers=1)
        print(res)
