"""Lab-side reference controllers and lab-only runners.  Internal: pilots must not read this module.

naive_router   the straightforward program written from the manual: a waypoint path around each post on its
               required side at a fixed clearance, at a fixed speed, then to the target and hold.  (Formerly called
               "datasheet_router" in the brief.)  Uses only the observation and the task card.
informed       privileged reference (see _informed.py).
lab_evaluate   parallel.evaluate with lab options: GIF recording of the first N seeds, the observation purpose.
"""
from __future__ import annotations

import numpy as np

CLEARANCE = 0.09     # m from the post centre (post radius 0.03 + 6 cm)
SPEED = 0.15         # m/s along the waypoint path (half the tool's speed limit)


def route_waypoints(start, posts, sides, target, clearance=CLEARANCE):
    way = [np.asarray(start, float)]
    prev = way[0]
    for i, p in enumerate(posts):
        nxt = posts[i + 1] if i + 1 < len(posts) else np.asarray(target, float)
        d = nxt - prev
        d = d / (np.linalg.norm(d) + 1e-12)
        n = np.array([-d[1], d[0]])
        q = p - sides[i] * n * clearance            # side +1: the post must stay on the left -> pass on its right
        way += [q - d * clearance, q, q + d * clearance]
        prev = p
    way.append(np.asarray(target, float))
    return np.array(way)


class NaiveRouter:
    def __init__(self, clearance: float = CLEARANCE, speed: float = SPEED):
        self.clearance, self.speed = clearance, speed

    def bind(self, card):
        self.hz = card["control_hz"]

    def reset(self):
        self.path = None

    def act(self, obs):
        if self.path is None:
            self.path = route_waypoints(obs["tool_pos"], obs["posts"], obs["post_sides"], obs["target"], self.clearance)
            seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
            self.cum = np.concatenate([[0.0], np.cumsum(seg)])
            self.s = 0.0
        self.s = min(self.s + self.speed / 20.0, self.cum[-1])
        return np.array([np.interp(self.s, self.cum, self.path[:, k]) for k in range(2)])


def naive_router(clearance: float = CLEARANCE, speed: float = SPEED):
    return NaiveRouter(clearance, speed)


# ---------------------------------------------------------------- lab-only runners
def _lab_env(cfg):
    from reallab.station import api
    return api.RealStation(cfg["purpose"], cfg["pilot"], cfg["tag"], None, _lab=True)


def _rec_hook(env, cfg, idx, seed):
    if idx < cfg.get("record_first", 0):
        from pathlib import Path
        env._rec_path = str(Path(cfg["video_dir"]) / f"ep_{seed}.gif")
        env._rec_title = cfg.get("record_title", "seed {seed}")
    else:
        env._rec_title = None


def lab_evaluate(policy_factory, policy_kwargs, seeds, purpose="eval", pilot="", tag="", n_workers=16, log_dir=None,
                 record_first=0, video_dir="runs/station/_lab/gifs", record_title=None):
    from reallab.station.parallel import _run
    cfg = dict(policy_factory=policy_factory, policy_kwargs=policy_kwargs or {}, purpose=purpose, pilot=pilot, tag=tag,
               log_dir=log_dir, lab_env="reallab.station._reference:_lab_env",
               episode_hook="reallab.station._reference:_rec_hook", record_first=record_first, video_dir=video_dir,
               record_title=record_title or (policy_factory.split(":")[-1] + " seed {seed}"))
    return _run(cfg, seeds, n_workers)


# ---------------------------------------------------------------- observation set (lab only)
def outcome_of(d) -> str:
    if bool(d["success"]):
        return "success"
    if bool(d["overload"]):
        return "overload"
    if bool(d["truncated"]):
        return "time_limit"
    return "abandoned"


def write_observation_index(obs_dir) -> dict:
    import json
    from pathlib import Path
    eps = []
    for f in sorted(Path(obs_dir).glob("episode_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
        d = np.load(f)
        eps.append(dict(seed=int(d["seed"]), outcome=outcome_of(d), steps=int(d["n_steps"]),
                        max_load=round(float(d["max_load"]), 3)))
    idx = dict(controller="an existing program that moves the tool along a waypoint path around the posts at a "
                          "fixed clearance and a fixed speed", episodes=eps)
    (Path(obs_dir) / "index.json").write_text(json.dumps(idx, indent=1))
    return idx


def run_observation(seeds=range(900, 910), max_steps_test=None):
    """Run the naive router on the real installation (deviations as configured, i.e. ON unless the lab switched them
    off) for the observation seeds; logs go to runs/station/observation (or $CAWM_OBSERVATION_DIR)."""
    from reallab.station import api
    env = api.RealStation("observation", "", "naive_router", None, _lab=True)
    pol = naive_router()
    pol.bind(api.task_card())
    res = []
    for seed in seeds:
        pol.reset()
        obs = env.reset(seed)
        done, info = False, {}
        while not done:
            obs, r, te, tr, info = env.step(pol.act(obs))
            done = te or tr or (max_steps_test is not None and obs["t"] >= max_steps_test)
        res.append(dict(seed=int(seed), **info))
    env.close()
    write_observation_index(api._observation_dir())
    return res


# ---------------------------------------------------------------- privileged scripted reference
class Informed:
    """Scripted controller designed with knowledge of the hidden properties (a stiff cable that needs tension to
    conform to the posts and slides easily on the table, so every centimetre of cable spent in a wide loop is missing
    at the end, and a fast pull runs out of slack with a snatch).  At run time it uses only the public observation:
      * the tool follows the shortest ("taut") route from its start around the posts on their required sides to the
        target, kept `margin` m outside the posts (tight wraps, no wasted loops);
      * travel at `speed`, the last `approach` m at `slow`;
      * load guard: while load > `guard` N it stops advancing and backs off along the route at `back` m/s;
      * it aims `short` m before the target along the final direction and creeps back up to `max_short` (still inside
        the 0.02 m tolerance) while the load stays above `hold` N at the end.
    """

    def __init__(self, margin=0.012, speed=0.05, slow=0.03, guard=8.0, hold=9.0, back=0.05, short=0.006,
                 max_short=0.016, approach=0.12, mode="taut"):
        self.mode = mode
        self.margin, self.speed, self.slow, self.guard, self.hold, self.back = margin, speed, slow, guard, hold, back
        self.short, self.max_short, self.approach = short, max_short, approach

    def bind(self, card):
        self.post_r = card["posts"]["radius"]

    def reset(self):
        self.path = None

    def act(self, obs):
        from reallab.station import _layout
        if self.path is None:
            R = getattr(self, "post_r", 0.03) + 0.004 + self.margin
            way = None
            if self.mode == "taut":
                way = _layout.taut_route(obs["tool_pos"], obs["posts"], obs["post_sides"], obs["target"], R=R)
            if way is None:
                way = route_waypoints(obs["tool_pos"], obs["posts"], obs["post_sides"], obs["target"], R + 0.02)
            way = np.asarray(way, float)
            d = way[-1] - way[-2]
            self.dir = d / (np.linalg.norm(d) + 1e-12)
            way = np.vstack([way[:-1], way[-1] - self.dir * self.short])
            self.path = way
            seg = np.linalg.norm(np.diff(way, axis=0), axis=1)
            self.cum = np.concatenate([[0.0], np.cumsum(seg)])
            self.s, self.extra = 0.0, 0.0
        end = self.cum[-1]
        load = float(obs["load"])
        at_end = self.s >= end - 1e-9
        if at_end:
            if load > self.hold:
                self.extra = min(self.extra + 0.0005, self.max_short - self.short)
        elif load > self.guard:
            self.s = max(0.0, self.s - self.back / 20.0)
        else:
            v = self.slow if self.s > end - self.approach else self.speed
            self.s = min(self.s + v / 20.0, end)
        p = np.array([np.interp(self.s, self.cum, self.path[:, k]) for k in range(2)])
        if self.s >= end - 1e-9:
            p = p - self.dir * self.extra
        return p


def informed(**kw):
    return Informed(**kw)
