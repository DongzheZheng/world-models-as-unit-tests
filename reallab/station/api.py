"""REAL-WORLD INSTRUMENT MANUAL: "Station lab" (black-box real world)
=================================================================

An installation on a table.  You get a metered black box: send 2-d position commands to a tool, receive structured
sensor readings.  There are no pixels.  Nothing is documented here about what the installation is made of or how it
responds: find out by watching it (observation logs, below) and from your own lives.

USAGE
-----
    from reallab.station import api
    env = api.make("eval", tag="my_run", seed=0)          # purpose, [pilot], [tag], [seed]
    obs = env.reset(seed=1234)                            # same seed + same commands => identical trajectory
    obs, reward, terminated, truncated, info = env.step(np.array([x, y]))
    env.close()
    api.task_card()                                       # public task facts, free of charge, not metered
    env.action_low, env.action_high, env.control_hz       # [0, -0.4], [1.2, 0.4], 20.0
    api.winding(points, posts), api.sides(points, posts)  # the side rule below, as code

PURPOSES, LIVES, OBSERVATION
----------------------------
"calibration": LIVES PER PILOT.  make("calibration", pilot="<name>") is required; each pilot has CAL_CAP_EPISODES = 5
  episodes in total; the 6th reset() of that pilot raises cawm.ledger.BudgetExceeded.  Every reset() uses up a life,
  even if the previous episode was abandoned.  One life at a time (parallel.evaluate forces one worker for
  calibration).  Calibration seeds must be < 1000.  Every calibration episode is saved automatically as
  runs/station/blackbox/<pilot>/life_<k>.npz, k = 1..5 (format below).
"observation": episodes run by the lab, never by a pilot.  Pilots cannot fly observation episodes themselves
  (make("observation", ...) raises ValueError).  The lab has run an existing program -- one that moves the tool along a
  waypoint path around the posts at a fixed clearance and a fixed speed -- on the real installation for 10 episodes,
  seeds 900..909.  Their logs are runs/station/observation/episode_<seed>.npz (same format as the black-box logs) with
  a summary in runs/station/observation/index.json (seed, outcome, steps, max load).  Every pilot may study them
  freely, before and between its own five lives; they cost nothing.  Observation seeds lie in [900, 1000).
SCENE SNAPSHOTS: looking is free.  runs/station/observation/layouts.npz holds the reset reading (no command was ever
  sent) of 200 situations, seeds 0..199: arrays seeds (200,), points (200,20,3), tool_pos (200,2), posts (200,3,2),
  post_sides (200,3), target (200,2), load (200,).  reset(seed) with the same seed reproduces the same situation.
  Every pilot may study them freely; they cost nothing.
"eval": final evaluation only; seeds >= 1000.
"smoke": plumbing tests (any seed).  "oracle_train": privileged references (seeds < 1000).
For purposes other than calibration `pilot` is optional and only prefixed to the ledger tag.  Every agent step and
episode is counted in the ledger.  Stepping after an episode ended raises RuntimeError.
The seed fixes the whole initial situation (post positions, required sides, target, initial positions of the tracked
points); reset(None) draws a seed from the RNG given to make(seed=...), inside the purpose's seed range.

UNITS AND FRAMES
----------------
Lengths in metres, forces in newtons, velocities in m/s.  The table frame: +x to the right, +y away from the operator,
+z up; the table surface is z = 0 and spans x in [0, 1.2], y in [-0.4, 0.4].  Time: one agent step = 0.05 s
(control_hz = 20).  Time limit: 600 agent steps (30 s).

OBSERVATION (dict of numpy arrays / numbers)
--------------------------------------------
obs["points"]     (20, 3) float64  positions (x, y, z) of twenty tracked points, always in the same order; point 20
                                   (index 19) is the one that moves with the tool
obs["tool_pos"]   (2,)  float64    tool position (x, y)
obs["tool_vel"]   (2,)  float64    tool velocity (vx, vy)
obs["posts"]      (3, 2) float64   centres of the three fixed vertical posts (they move only between episodes)
obs["post_sides"] (3,)  int64      required side of each post, -1 or +1 (see SUCCESS RULE)
obs["sides_now"]  (3,)  int64      api.sides(points, posts) for the current reading: -1, +1, or 0 (undefined)
obs["target"]     (2,)  float64    where the tool must come to rest (changes between episodes)
obs["load"]       float            a force reading of the installation (N), averaged over the last control period
                                   (0.05 s); its limit is task_card()["load_limit"]
obs["t"]          int              agent steps taken in this episode (0 right after reset)

ACTION
------
np.ndarray (2,): the position (x, y) the tool should move to.  It is held for the whole 0.05 s step.  The tool is
servoed to a motion reference that travels from its current value towards the command at 0.30 m/s at most; the tool
follows that reference with a position servo of limited force, so under load it can lag behind the reference (and
then catch up).  A command outside the table [0, 1.2] x [-0.4, 0.4] is replaced by the nearest point of it;
non-finite values raise ValueError.  The tool is a ball of radius 0.015 m; it cannot pass through the posts (posts:
vertical cylinders of radius 0.03 m, height 0.08 m).

SUCCESS RULE (exact)
--------------------
Winding: for post c, let a_i be the angle of (p_i - c) for the tracked points p_1..p_20 (x, y only, in order from
point 1 to point 20), and W = sum over i = 1..19 of wrap(a_{i+1} - a_i), wrap to (-pi, pi].  W > 0 means the polyline
goes counter-clockwise around the post (the post is on the left of the polyline when travelling from point 1 to
point 20).  side = +1 if W >= pi/2, -1 if W <= -pi/2, 0 otherwise.  An episode succeeds (terminated, success) at the
first step at which ALL of:
  * |tool_pos - target| <= 0.02 m and |tool_vel| <= 0.02 m/s (the tool is at rest at the target), and
  * sides(points, posts) == post_sides for all three posts,
hold, within the time limit.  If the load reading exceeds the load limit (10 N) at any step the episode ends at once
(terminated) as a failure ("overload").  At the 600th step without either: truncated.

REWARD (per step, documented, simple)
--------------------------------------
reward = 10 * (d_prev - d)                       d = |tool_pos - target|
       + (n_ok - n_ok_prev)                      n_ok = number of posts with sides_now == post_sides
       - 0.1 * max(0, load / load_limit - 0.5)
       + 10 on success, - 10 on overload.

STEP RETURN
-----------
terminated  success or overload;  truncated  time limit reached.  Exactly one is True on the last step.
info        {"success", "sides_ok", "final_dist", "max_load", "overload", "n_steps"} (running values; final at the end).
reset() returns obs only (t = 0).

BLACK-BOX LOG FORMAT (calibration lives and observation episodes; numpy.load)
----------------------------------------------------------------------------
Arrays with leading dimension T+1 (index 0 = the reset reading): points, tool_pos, tool_vel, posts, post_sides,
sides_now, target, load, t.  Leading dimension T: actions (as passed to step(), after limiting to the table), rewards.
Scalars: seed, life (0 for observation episodes), pilot ("" for observation), purpose, success, sides_ok, final_dist,
max_load, overload, n_steps, terminated, truncated, control_hz.  Written when the episode ends or is abandoned.

Field note: the people who run this installation say it 'fights back' when you hurry. Nobody has written down why.

OTHER
-----
reallab.station.parallel.evaluate(...) runs a policy over many seeds in worker processes.
DETERMINISM: the same reset(seed) followed by the same commands reproduces every reading bit-for-bit across env
instances and processes (verified on one machine, see test_api.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cawm import ledger

CAL_CAP_EPISODES = 5
TIME_LIMIT = 600
SEED_SPLIT = 1000
OBS_SEEDS = (900, 1000)
LOAD_LIMIT = 10.0
TOL = 0.02
REST_SPEED = 0.02
N_POINTS = 20
_DEMO = "station"


def task_card() -> dict:
    """Unmetered public facts (interface only)."""
    return dict(
        table=dict(low=[0.0, -0.4], high=[1.2, 0.4], surface_z=0.0),
        control_hz=20.0, time_limit_steps=TIME_LIMIT,
        action=dict(meaning="(x, y) position the tool should move to", low=[0.0, -0.4], high=[1.2, 0.4],
                    tool_speed_limit=0.30, tool_radius=0.015),
        posts=dict(count=3, radius=0.03, height=0.08, note="vertical cylinders; positions change between episodes"),
        points=dict(count=N_POINTS, note="tracked points in fixed order; point 20 (index 19) moves with the tool"),
        success=dict(tolerance=TOL, rest_speed=REST_SPEED,
                     rule="|tool_pos - target| <= tolerance and |tool_vel| <= rest_speed and sides_now == post_sides"),
        side_rule=("W = sum_i wrap(angle(p_{i+1} - c) - angle(p_i - c)) over the tracked points 1..20 (x, y), wrap to "
                   "(-pi, pi]; side = +1 if W >= pi/2, -1 if W <= -pi/2, else 0; +1 = counter-clockwise (post on the "
                   "left when travelling from point 1 to point 20)"),
        side_threshold=float(np.pi / 2),
        load_limit=LOAD_LIMIT, load_note="mean over each control period; exceeding the limit ends the episode (failure)",
        reward="10*(d_prev-d) + (n_ok-n_ok_prev) - 0.1*max(0, load/limit-0.5) + 10*success - 10*overload",
        calibration_lives=CAL_CAP_EPISODES,
        observation=dict(seeds=[900, 909], logs="runs/station/observation/episode_<seed>.npz",
                         index="runs/station/observation/index.json",
                         scene_snapshots="runs/station/observation/layouts.npz (reset readings, seeds 0..199)"),
    )


def winding(points, posts) -> np.ndarray:
    """Signed winding angle W (rad) of the polyline through `points` (in order) about each post centre."""
    pts = np.asarray(points, float)[:, :2]
    out = np.zeros(len(posts))
    for j, c in enumerate(np.asarray(posts, float)):
        a = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        da = (np.diff(a) + np.pi) % (2 * np.pi) - np.pi
        out[j] = da.sum()
    return out


def sides(points, posts) -> np.ndarray:
    w = winding(points, posts)
    th = np.pi / 2
    return np.where(w >= th, 1, np.where(w <= -th, -1, 0)).astype(np.int64)


_LOG_KEYS = ("points", "tool_pos", "tool_vel", "posts", "post_sides", "sides_now", "target", "load", "t")


def _base_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "station"


def _blackbox_dir(pilot: str) -> Path:
    return Path(os.environ.get("CAWM_BLACKBOX_DIR") or _base_dir() / "blackbox") / pilot


def _observation_dir() -> Path:
    return Path(os.environ.get("CAWM_OBSERVATION_DIR") or _base_dir() / "observation")


class RealStation:
    """A metered, black-box episode interface to the installation.  Create with make()."""

    def __init__(self, purpose, pilot, tag, seed, ledger_path=None, _lab=False):
        from reallab.station import _internal, _scene
        if purpose not in ledger.PURPOSES:
            raise ValueError(f"purpose must be one of {ledger.PURPOSES}")
        if purpose == "observation" and not _lab:
            raise ValueError("observation episodes are run by the lab only; pilots study their logs "
                             "(runs/station/observation)")
        self._calib = purpose == "calibration"
        if self._calib and not pilot:
            raise ValueError("purpose='calibration' requires pilot='<name>' (lives are per pilot)")
        self.pilot, self.purpose = pilot or "", purpose
        if pilot and not self._calib:
            tag = f"{pilot}/{tag}" if tag else pilot
        self.tag = tag
        self._ledger_key = f"{_DEMO}:{pilot}" if self._calib else _DEMO
        self._ledger_path = ledger_path or os.environ.get("CAWM_LEDGER", ledger.default_path())
        self.meter = ledger.Meter(_DEMO, self._ledger_key, purpose, tag,
                                  cap_episodes=CAL_CAP_EPISODES if self._calib else None, path=self._ledger_path)
        self._rng = np.random.default_rng(seed)
        self._scene = _scene.Scene(_internal.physical())
        self.action_low = np.array([0.0, -0.4])
        self.action_high = np.array([1.2, 0.4])
        self.control_hz = 20.0
        self._rec = None            # lab-side recorder (reallab.station._render), never part of the interface
        self._rec_title = None
        self._rec_path = None
        self._active = False
        self._closed = False
        self._bb = None
        self._seed = None
        self._ep = None

    # ---- seeds ---------------------------------------------------------------------------------------
    def _check_seed(self, seed: int) -> int:
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be >= 0")
        if self.purpose == "eval" and seed < SEED_SPLIT:
            raise ValueError(f"purpose='eval' requires seed >= {SEED_SPLIT} (got {seed})")
        if self.purpose in ("calibration", "oracle_train") and seed >= SEED_SPLIT:
            raise ValueError(f"purpose='{self.purpose}' requires seed < {SEED_SPLIT} (got {seed})")
        if self.purpose == "observation" and not (OBS_SEEDS[0] <= seed < OBS_SEEDS[1]):
            raise ValueError(f"purpose='observation' requires {OBS_SEEDS[0]} <= seed < {OBS_SEEDS[1]} (got {seed})")
        return seed

    def _draw_seed(self) -> int:
        if self.purpose == "eval":
            return int(self._rng.integers(SEED_SPLIT, 2**31 - 1))
        if self.purpose in ("calibration", "oracle_train"):
            return int(self._rng.integers(0, SEED_SPLIT))
        if self.purpose == "observation":
            return int(self._rng.integers(*OBS_SEEDS))
        return int(self._rng.integers(0, 2**31 - 1))

    # ---- readings --------------------------------------------------------------------------------------
    def _obs(self, load):
        sc, ep = self._scene, self._ep
        pts = sc.tracked_points(N_POINTS)
        return dict(points=pts, tool_pos=sc.tool_pos(), tool_vel=sc.tool_vel(), posts=ep["posts"].copy(),
                    post_sides=ep["sides"].copy(), sides_now=sides(pts, ep["posts"]), target=ep["target"].copy(),
                    load=float(load), t=self._t)

    # ---- logs ------------------------------------------------------------------------------------------
    def _bb_flush(self):
        if self._bb is None:
            return
        L, self._bb = self._bb, None
        obs = L["obs"]
        T = len(L["actions"])
        out = {k: np.array([o[k] for o in obs]) for k in _LOG_KEYS}
        out["actions"] = np.array(L["actions"], dtype=np.float64).reshape(T, 2)
        out["rewards"] = np.array(L["rewards"], dtype=np.float64)
        out.update(seed=self._seed, life=L["life"], pilot=self.pilot, purpose=self.purpose, success=self._success,
                   sides_ok=int(np.sum(obs[-1]["sides_now"] == obs[-1]["post_sides"])),
                   final_dist=float(np.linalg.norm(obs[-1]["tool_pos"] - obs[-1]["target"])),
                   max_load=self._max_load, overload=self._overload, n_steps=T, terminated=self._terminated,
                   truncated=self._truncated, control_hz=self.control_hz)
        L["path"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(L["path"], **out)

    def _rec_frame(self, obs):
        if self._rec is not None:
            self._rec.add(self._scene, obs)

    def _rec_close(self, outcome=None):
        if self._rec is not None:
            self._rec.close(outcome)
            self._rec = None

    # ---- episode interface -----------------------------------------------------------------------------
    def _abandon(self):
        if self._active:
            self.meter.episode()
        self._active = False
        self._bb_flush()
        self._rec_close("abandoned")

    def reset(self, seed: int | None = None) -> dict:
        from reallab.station import _layout
        if self._closed:
            raise RuntimeError("env is closed")
        seed = self._draw_seed() if seed is None else self._check_seed(seed)
        self._abandon()
        self.meter.start_episode()   # raises BudgetExceeded when the lives cap is used up
        self._seed = seed
        e = _layout.setup(self._scene, seed)
        self._ep = dict(posts=e["posts"], sides=e["sides"], target=e["target"])
        self._active = True
        self._t, self._ret = 0, 0.0
        self._success = self._terminated = self._truncated = self._overload = False
        obs = self._obs(self._scene.load_now())
        self._max_load = obs["load"]
        self._last = obs
        if self._calib:
            k = ledger.totals(self._ledger_path).get((_DEMO, self._ledger_key, "calibration"), {"episodes": 1})["episodes"]
            self._bb = dict(path=_blackbox_dir(self.pilot) / f"life_{int(k)}.npz", life=int(k), obs=[obs], actions=[],
                            rewards=[])
        elif self.purpose == "observation":
            self._bb = dict(path=_observation_dir() / f"episode_{seed}.npz", life=0, obs=[obs], actions=[], rewards=[])
        if self._rec_title is not None:
            from reallab.station import _render
            self._rec = _render.Recorder(self._rec_path, self._rec_title.format(seed=seed))
        self._rec_frame(obs)
        return obs

    def step(self, action):
        if self._closed:
            raise RuntimeError("env is closed")
        if not self._active:
            raise RuntimeError("episode is over (or reset() not called); call reset()")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise ValueError("action must be a finite array of shape (2,)")
        a = np.minimum(np.maximum(a, self.action_low), self.action_high)
        self.meter.step()
        load = self._scene.step(a)
        self._t += 1
        obs = self._obs(load)
        prev = self._last
        d_prev = float(np.linalg.norm(prev["tool_pos"] - prev["target"]))
        d = float(np.linalg.norm(obs["tool_pos"] - obs["target"]))
        n_ok_prev = int(np.sum(prev["sides_now"] == prev["post_sides"]))
        n_ok = int(np.sum(obs["sides_now"] == obs["post_sides"]))
        self._max_load = max(self._max_load, load)
        overload = load > LOAD_LIMIT
        success = (not overload) and d <= TOL and float(np.linalg.norm(obs["tool_vel"])) <= REST_SPEED and n_ok == 3
        reward = 10.0 * (d_prev - d) + (n_ok - n_ok_prev) - 0.1 * max(0.0, load / LOAD_LIMIT - 0.5)
        reward += 10.0 if success else 0.0
        reward -= 10.0 if overload else 0.0
        terminated = bool(success or overload)
        truncated = (not terminated) and self._t >= TIME_LIMIT
        self._success, self._overload = bool(success), bool(overload)
        self._terminated, self._truncated = terminated, truncated
        self._ret += reward
        self._last = obs
        info = dict(success=bool(success), sides_ok=n_ok, final_dist=d, max_load=float(self._max_load),
                    overload=bool(overload), n_steps=self._t)
        if self._bb is not None:
            self._bb["obs"].append(obs)
            self._bb["actions"].append(a)
            self._bb["rewards"].append(reward)
        self._rec_frame(obs)
        if terminated or truncated:
            self._active = False
            self.meter.episode()
            self._bb_flush()
            self._rec_close("SUCCESS" if success else ("OVERLOAD" if overload else
                            f"TIME OUT - dist {d:.3f} m, sides {n_ok}/3"))
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._abandon()
        self._closed = True
        self.meter.flush()


def make(purpose: str, pilot: str = "", tag: str = "", seed: int | None = None, ledger_path=None) -> RealStation:
    """Create a metered real installation.  See the module docstring.  purpose="observation" is lab-only."""
    if purpose == "observation":
        raise ValueError("observation episodes are run by the lab only; pilots study their logs "
                         "(runs/station/observation)")
    return RealStation(purpose, pilot, tag, seed, ledger_path)
