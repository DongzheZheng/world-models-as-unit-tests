"""Internal: reference numbers from the real stack -> runs/drone/baselines.json"""
import json
import os
import time

import numpy as np

from reallab.drone.parallel import evaluate

SEEDS = list(range(1000, 1100))


def stats(rows):
    keys = [k for k, v in rows[0].items() if k != "seed" and isinstance(v, (int, float, bool, np.bool_))]
    out = {}
    for k in keys:
        x = np.array([float(r[k]) for r in rows])
        out[k] = dict(mean=float(x.mean()), std=float(x.std()), median=float(np.median(x)))
    return out


def speed(task, **kw):
    """Single-process throughput: agent steps per second (step() only) and seconds per reset()."""
    from reallab.drone.api import make
    env = make(task, "smoke", "speed", **kw)
    env.reset(seed=0)                         # warm-up (numba compilation etc.)
    for _ in range(5):
        env.step(np.array([0.0, 0.0, 0.0, 0.30]))
    n, t_step, t_reset = 0, 0.0, 0.0
    ep = 1
    while n < 3000:
        t0 = time.perf_counter(); env.reset(seed=ep); t_reset += time.perf_counter() - t0; ep += 1
        while True:
            t0 = time.perf_counter()
            _, _, te, tr, _ = env.step(np.array([0.0, 0.0, 0.0, 0.30]))
            t_step += time.perf_counter() - t0
            n += 1
            if te or tr:
                break
    env.close()
    return dict(steps_per_s=n / t_step, reset_s=t_reset / (ep - 1), steps=n)


def main():
    res = {}
    jobs = [
        ("waypoints/pyflyt_pid_mode7", ":pid_waypoints_policy", {}, "waypoints", dict(flight_mode=7)),
        ("waypoints/random", ":random_policy", {"seed": 0}, "waypoints", {}),
        ("port_medium/pyflyt_pid_mode7_straight_line", ":pid_straight_policy", {}, "port", dict(flight_mode=7, difficulty="medium")),
        ("port_medium/random", ":random_policy", {"seed": 0}, "port", dict(difficulty="medium")),
        ("port_medium_realism_off/pyflyt_pid_mode7_straight_line", ":pid_straight_policy", {}, "port",
         dict(flight_mode=7, difficulty="medium", realism=False)),
    ]
    for name, fac, pk, task, kw in jobs:
        t = time.time()
        rows = evaluate(fac, pk, task, SEEDS, purpose="eval", tag="baseline:" + name, n_workers=32, **kw)
        res[name] = dict(n_episodes=len(rows), stats=stats(rows), wall_s=time.time() - t, rows=rows)
        print(name, {k: round(v["mean"], 3) for k, v in res[name]["stats"].items()}, flush=True)
    res["speed_steps_per_s_per_process"] = {
        "waypoints": speed("waypoints"),
        "port_realism_on": speed("port", realism=True),
        "port_realism_off": speed("port", realism=False),
    }
    print(res["speed_steps_per_s_per_process"])
    os.makedirs("runs/drone", exist_ok=True)
    json.dump(res, open("runs/drone/baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
