"""REAL-WORLD INSTRUMENT MANUAL: "Annex lab" (black-box real world)
=================================================================

An installation on a table.  You get a metered black box: send 2-d position commands to a tool, receive structured
sensor readings.  There are no pixels and there is no force reading.  Nothing is documented here about what the
installation is made of or how it responds: find out from the scene snapshots (below) and from your own lives.

USAGE
-----
    from reallab.annex import api
    env = api.make("eval", tag="my_run", seed=0)          # purpose, [pilot], [tag], [seed]
    obs = env.reset(seed=1234)                            # same seed + same commands => identical trajectory
    obs, reward, terminated, truncated, info = env.step(np.array([x, y]))
    env.close()
    api.task_card()                                       # public task facts, free of charge, not metered
    env.action_low, env.action_high, env.control_hz       # the zone corners [0.30, -0.20], [0.55, 0.20]; 50.0
    api.at_rest(prev_obs, obs), api.goal_met(obs, prev_obs)   # the rules below, as code

PURPOSES, LIVES
---------------
"calibration": LIVES PER PILOT.  make("calibration", pilot="<name>") is required; each pilot has CAL_CAP_EPISODES = 5
  episodes in total; the 6th reset() of that pilot raises cawm.ledger.BudgetExceeded.  Every reset() uses up a life,
  even if the previous episode was abandoned.  One life at a time (parallel.evaluate forces one worker for
  calibration).  Calibration seeds must be < 1000.  Every calibration episode is saved automatically as
  runs/annex/blackbox/<pilot>/life_<k>.npz, k = 1..5 (format below).  A life is an episode with exactly the rules
  and the time limit of an evaluation episode.
"eval": final evaluation only; seeds >= 1000.
"smoke": plumbing tests (any seed).  "oracle_train": privileged references (seeds < 1000).
There is no "observation" purpose for this installation: no recordings of it in motion exist for anyone.
For purposes other than calibration `pilot` is optional and only prefixed to the ledger tag.  Every agent step and
episode is counted in the ledger.  Stepping after an episode ended raises RuntimeError.
The seed fixes the whole initial situation (the tool's start, the initial positions of the tracked points, the post
and the target); reset(None) draws a seed from the RNG given to make(seed=...), inside the
purpose's seed range.
SCENE SNAPSHOTS: looking is free.  runs/annex/observation/layouts.npz holds the reset reading (no command was ever
  sent) of 200 situations, seeds 0..199: arrays seeds (200,), points (200,20,3), tool_pos (200,2), post (200,2),
  zone (200,2,2), target (200,2).  reset(seed) with the same seed reproduces the same situation.
  Every pilot may study them freely; they cost nothing.

UNITS AND FRAMES
----------------
Lengths in metres, velocities in m/s, angles in radians.  The table frame: +x to the right, +y away from the operator,
+z up; the table surface is z = 0 and extends well beyond every place mentioned here.  Time: one agent step = 0.02 s
(control_hz = 50).  Time limit: TIME_LIMIT = 236 agent steps (4.72 s).

OBSERVATION (dict of numpy arrays / numbers)
--------------------------------------------
obs["points"]      (20, 3) float64  positions (x, y, z) of twenty tracked points, always in the same order; point 20
                                    (index 19) is the one that moves with the tool, point 1 (index 0) is the far end
obs["tool_pos"]    (2,)  float64    tool position (x, y)
obs["tool_vel"]    (2,)  float64    tool velocity (vx, vy)
obs["post"]        (2,)  float64    centre of the fixed vertical post (it moves only between episodes)
obs["zone"]        (2, 2) float64   tool zone [[x_low, y_low], [x_high, y_high]] (the same in every episode)
obs["target"]      (2,)  float64    centre of the target disc (changes between episodes)
obs["t"]           int              agent steps taken in this episode (0 right after reset)

ACTION
------
np.ndarray (2,): the position (x, y) the tool should move to.  It is held for the whole 0.02 s step.  A command
outside the tool zone [0.30, 0.55] x [-0.20, 0.20] is replaced by the nearest point of the zone; non-finite values
raise ValueError.  The tool is servoed to a motion reference that travels from its current value towards the
command with speed at most 4.0 m/s and acceleration at most 60 m/s^2 (it brakes so as to stop at the command); the
tool follows that reference with a position servo whose force is limited to 60 N per axis, so it can lag behind the
reference.  The tool collides with nothing.  The post is a vertical cylinder of radius 0.03 m and height 0.08 m.  The
post and the target lie outside the zone, beyond the tool's reach.  Point 20 stays with the tool; the rest arc
length from point 1 to point 20 is 0.70 m (the twenty points are equally spaced along it).

SUCCESS RULE (exact)
--------------------
At rest: a reading is at rest when, compared with the previous reading (0.02 s earlier), every tracked point (x, y)
and the tool have each moved by at most REST_SPEED / control_hz = 0.02 / 50 m (speed <= 0.02 m/s).
An episode succeeds (terminated, success) at the first step t >= 1 at which ALL of:
  * the reading is at rest,
  * |p_1 - target| <= TARGET_RADIUS = 0.10 m (x, y; point 1 inside the target disc),
  * the tool is inside the zone (within 0.01 m),
hold, within the time limit.  The tracked points may touch the post and pass it on either side.  At the TIME_LIMIT-th step without success: truncated (failure).  Nothing else ends an
episode.

REWARD (per step, documented, simple)
--------------------------------------
reward = (d_prev - d) + 10 on success,     d = |p_1 - target| (x, y).

STEP RETURN
-----------
terminated  success;  truncated  time limit reached.  Exactly one is True on the last step.
info        {"success", "at_rest", "tip_dist", "n_steps"} (running values; final at the end);
            tip_dist = |p_1 - target| (x, y).
reset() returns obs only (t = 0).

BLACK-BOX LOG FORMAT (calibration lives; numpy.load)
----------------------------------------------------
Arrays with leading dimension T+1 (index 0 = the reset reading): points, tool_pos, tool_vel, post, zone, target, t.  Leading dimension T: actions (as passed to step(), after limiting to the zone), rewards.
Scalars: seed, life, pilot, purpose, success, final_tip_dist, n_steps, terminated, truncated, control_hz.  Written when the episode ends or is abandoned.

OTHER
-----
reallab.annex.parallel.evaluate(...) runs a policy over many seeds in worker processes.
DETERMINISM: the same reset(seed) followed by the same commands reproduces every reading bit-for-bit across env
instances and processes (verified on one machine, see test_api.py).

Field note: the people who run this installation say you get one go at it, and that doing it slowly gets you nowhere. Nobody has written down why. They also say that a small change in how you do it changes where it ends up.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cawm import ledger

CAL_CAP_EPISODES = 5
TIME_LIMIT = 236
SEED_SPLIT = 1000
N_POINTS = 20
CONTROL_HZ = 50.0
ZONE = np.array([[0.30, -0.20], [0.55, 0.20]])
ZONE_TOL = 0.01
REST_SPEED = 0.02
TARGET_RADIUS = 0.10
_DEMO = "annex"
_PURPOSES = ("calibration", "eval", "smoke", "oracle_train")


def task_card() -> dict:
    """Unmetered public facts (interface only)."""
    return dict(
        control_hz=CONTROL_HZ, step_seconds=1.0 / CONTROL_HZ, time_limit_steps=TIME_LIMIT,
        table=dict(surface_z=0.0, note="extends well beyond the zone, the post and the target"),
        zone=dict(low=ZONE[0].tolist(), high=ZONE[1].tolist(), note="a command outside the zone is replaced by its nearest point in the zone"),
        action=dict(meaning="(x, y) position the tool should move to", low=ZONE[0].tolist(), high=ZONE[1].tolist(),
                    reference_speed_limit=4.0, reference_accel_limit=60.0, servo_force_limit_per_axis=60.0,
                    note="the tool collides with nothing"),
        post=dict(count=1, radius=0.03, height=0.08, note="vertical cylinder outside the zone; moves between episodes"),
        points=dict(count=N_POINTS, rest_arc_length=0.70,
                    note="tracked points in fixed order, equally spaced along the rest arc length; point 20 "
                         "(index 19) moves with the tool, point 1 (index 0) is the far end"),
        success=dict(target_radius=TARGET_RADIUS, rest_speed=REST_SPEED, zone_tol=ZONE_TOL,
                     rule="at rest (every point and the tool moved <= rest_speed/control_hz since the previous "
                          "reading) and |p_1 - target| <= target_radius and the tool inside the zone (within "
                          "zone_tol)"),
        reward="(d_prev - d) + 10*success, d = |p_1 - target|",
        calibration_lives=CAL_CAP_EPISODES,
        scene_snapshots="runs/annex/observation/layouts.npz (reset readings, seeds 0..199)",
    )


def at_rest(prev_obs, obs) -> bool:
    lim = REST_SPEED / CONTROL_HZ
    dp = np.linalg.norm(np.asarray(obs["points"])[:, :2] - np.asarray(prev_obs["points"])[:, :2], axis=1).max()
    dt = np.linalg.norm(np.asarray(obs["tool_pos"]) - np.asarray(prev_obs["tool_pos"]))
    return bool(dp <= lim and dt <= lim)


def goal_met(obs, prev_obs) -> bool:
    """The success rule for one reading (given the previous reading)."""
    tool = np.asarray(obs["tool_pos"])
    inside = bool(np.all(tool >= ZONE[0] - ZONE_TOL) and np.all(tool <= ZONE[1] + ZONE_TOL))
    d = float(np.linalg.norm(np.asarray(obs["points"])[0, :2] - np.asarray(obs["target"])))
    return at_rest(prev_obs, obs) and d <= TARGET_RADIUS and inside


_LOG_KEYS = ("points", "tool_pos", "tool_vel", "post", "zone", "target", "t")


def _base_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "annex"


def _blackbox_dir(pilot: str) -> Path:
    return Path(os.environ.get("CAWM_BLACKBOX_DIR") or _base_dir() / "blackbox") / pilot


class RealAnnex:
    """A metered, black-box episode interface to the installation.  Create with make()."""

    def __init__(self, purpose, pilot, tag, seed, ledger_path=None, _q=None):
        from reallab.annex import _internal, _scene
        if purpose not in _PURPOSES:
            raise ValueError(f"purpose must be one of {_PURPOSES}")
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
        q = dict(_q or {})
        self._scene = _scene.Scene(q.get("p") or _internal.physical(), **({"vmax": q["v"]} if "v" in q else {}))
        self.action_low = ZONE[0].copy()
        self.action_high = ZONE[1].copy()
        self.control_hz = CONTROL_HZ
        self._hook = None           # lab-side frame hook (reallab.annex._render), never part of the interface
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
        return seed

    def _draw_seed(self) -> int:
        if self.purpose == "eval":
            return int(self._rng.integers(SEED_SPLIT, 2**31 - 1))
        if self.purpose in ("calibration", "oracle_train"):
            return int(self._rng.integers(0, SEED_SPLIT))
        return int(self._rng.integers(0, 2**31 - 1))

    # ---- readings --------------------------------------------------------------------------------------
    def _obs(self):
        sc, ep = self._scene, self._ep
        pts = sc.tracked_points()
        return dict(points=pts, tool_pos=sc.tool_pos(), tool_vel=sc.tool_vel(), post=ep["post"].copy(),
                    zone=ZONE.copy(), target=ep["target"].copy(), t=self._t)

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
        last = obs[-1]
        out.update(seed=self._seed, life=L["life"], pilot=self.pilot, purpose=self.purpose, success=self._success,
                   final_tip_dist=float(np.linalg.norm(last["points"][0, :2] - last["target"])), n_steps=T,
                   terminated=self._terminated, truncated=self._truncated, control_hz=self.control_hz)
        L["path"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(L["path"], **out)

    def _frame(self, obs):
        if self._hook is not None:
            self._hook.add(obs)

    def _frames_end(self, outcome=None):
        if self._hook is not None:
            self._hook.close(outcome)
            self._hook = None

    # ---- episode interface -----------------------------------------------------------------------------
    def _abandon(self):
        if self._active:
            self.meter.episode()
        self._active = False
        self._bb_flush()
        self._frames_end("abandoned")

    def reset(self, seed: int | None = None) -> dict:
        from reallab.annex import _layout
        if self._closed:
            raise RuntimeError("env is closed")
        seed = self._draw_seed() if seed is None else self._check_seed(seed)
        self._abandon()
        self.meter.start_episode()   # raises BudgetExceeded when the lives cap is used up
        self._seed = seed
        e = _layout.setup(self._scene, seed)
        self._ep = dict(post=e["post"], target=e["target"])
        self._active = True
        self._t = 0
        self._success = self._terminated = self._truncated = False
        obs = self._obs()
        self._last = obs
        if self._calib:
            k = ledger.totals(self._ledger_path).get((_DEMO, self._ledger_key, "calibration"), {"episodes": 1})["episodes"]
            self._bb = dict(path=_blackbox_dir(self.pilot) / f"life_{int(k)}.npz", life=int(k), obs=[obs], actions=[],
                            rewards=[])
        if callable(getattr(self, "_hook_factory", None)):
            self._hook = self._hook_factory(seed)
        self._frame(obs)
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
        self._scene.step(a)
        self._t += 1
        obs = self._obs()
        prev = self._last
        d_prev = float(np.linalg.norm(prev["points"][0, :2] - prev["target"]))
        d = float(np.linalg.norm(obs["points"][0, :2] - obs["target"]))
        rest = at_rest(prev, obs)
        success = goal_met(obs, prev)
        reward = (d_prev - d) + (10.0 if success else 0.0)
        terminated = bool(success)
        truncated = (not terminated) and self._t >= TIME_LIMIT
        self._success, self._terminated, self._truncated = bool(success), terminated, truncated
        self._last = obs
        info = dict(success=bool(success), at_rest=rest, tip_dist=d, n_steps=self._t)
        if self._bb is not None:
            self._bb["obs"].append(obs)
            self._bb["actions"].append(a)
            self._bb["rewards"].append(reward)
        self._frame(obs)
        if terminated or truncated:
            self._active = False
            self.meter.episode()
            self._bb_flush()
            self._frames_end("SUCCESS" if success else f"TIME OUT - distance {d:.3f} m")
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._abandon()
        self._closed = True
        self.meter.flush()


def make(purpose: str, pilot: str = "", tag: str = "", seed: int | None = None, ledger_path=None) -> RealAnnex:
    """Create a metered real installation.  See the module docstring."""
    return RealAnnex(purpose, pilot, tag, seed, ledger_path)
