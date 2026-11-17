"""REAL-WORLD INSTRUMENT MANUAL: "Push-T lab" (black-box real world for the manipulation demo)
==========================================================================================

The real world is a planar pushing rig: a round pusher on a flat table moves a T-shaped block, without grasping it,
towards a marked goal pose. You get a metered black box: send 2-d pusher targets, receive structured state readings.
There are no pixels. Nothing about how the block responds to being pushed is documented here: learn it from data.

USAGE
-----
    from reallab.pusht import api
    env = api.make("eval", tag="my_run", seed=0)          # purpose, [pilot], [tag], [seed], [record_path]
    obs = env.reset(seed=1234)                            # same seed + same actions => identical trajectory
    obs, reward, terminated, truncated, info = env.step(np.array([x, y]))
    env.close()
    api.task_card()                                       # public task facts, free of charge, not metered
    env.action_low, env.action_high, env.control_hz       # [0, 0], [512, 512], 10.0

Purposes: "calibration" (LIVES PER PILOT: make("calibration", pilot="<name>") is required; each pilot has
CAL_CAP_EPISODES = 5 episodes in total; the 6th reset() of that pilot raises cawm.ledger.BudgetExceeded. Every reset()
uses up a life, even if the previous episode was abandoned. Fly calibration lives one at a time (evaluate() forces
one worker for calibration). Calibration seeds must be < 1000). Every calibration episode is saved automatically as
runs/pusht/blackbox/<pilot>/life_<k>.npz, k = 1..5 (format below). "eval" (final evaluation only; seeds >= 1000).
"oracle_train" (privileged RL reference, costed; seeds < 1000), "smoke" (plumbing tests, any seed). For purposes other
than calibration `pilot` is optional and is only prefixed to the ledger tag. Every agent step and episode is counted
in the ledger. Stepping after an episode ended raises RuntimeError.
The seed fixes the initial pusher position and the initial block pose (reset(None) draws a seed from the RNG given
to make(seed=...), inside the purpose's seed range).

UNITS, FRAMES, ANGLES (verified empirically)
--------------------------------------------
* Lengths in table units (u). The table frame is a 2-D Cartesian frame, +x to the right and +y up as drawn in the
  GIFs. The workspace is the square [0, 512] x [0, 512]; the pusher target must lie in it (actions are clipped to it).
  Inside it a closed wall (rectangle with corners (5, 5) and (506, 506), see task_card) encloses the block.
* Time: one agent step = 0.1 s of rig time (control_hz = 10). The time limit is 300 agent steps (30 s).
* Block pose (x, y, theta) is the pose of the block's body frame (its drawing frame, see the datasheet) in the table
  frame: a body-frame point (u, v) is at (x, y) + R(theta) (u, v), with R(theta) the counter-clockwise rotation
  (theta > 0 turns the block counter-clockwise, as seen in the GIFs). theta is reported in [0, 2 pi); the goal
  pose is therefore (256, 256, pi/4).
* Velocity of the pusher in u/s in the table frame.

DATASHEET (nominal, as published by the task designers)
-------------------------------------------------------
Pusher: a disc of radius 15 u. Its position (the point that is observed) is the disc's centre.
Block: a flat "T" made of two rectangles, in the block's body frame (u along the block's x-axis, v along its y-axis):
    bar:  u in [-60, 60], v in [0, 30]          (120 x 30)
    stem: u in [-15, 15], v in [30, 120]        ( 30 x 90)
i.e. the stem is centred on the bar and points along +v; the block body-frame origin is the middle of the bar's
bottom edge. Lengths are exact; nothing is published about the block's mass or its mass distribution.
Goal: the T drawn at pose (256, 256, pi/4) (light green in the GIFs).
Field note: operators report that this particular block 'does not push the way the drawing suggests'. Nobody has characterised why.

OBSERVATION (dict of numpy arrays / numbers; state only)
--------------------------------------------------------
obs["agent_pos"]   (2,) float64   pusher centre (x, y) in u
obs["agent_vel"]   (2,) float64   pusher velocity (vx, vy) in u/s
obs["block_pose"]  (3,) float64   (x, y, theta) of the block's body frame, theta in [0, 2 pi)
obs["goal_pose"]   (3,) float64   (256, 256, pi/4), constant
obs["coverage"]    float          area(block ∩ goal T) / area(goal T), in [0, 1]
obs["t"]           int            number of agent steps taken in this episode (0 right after reset)

ACTION
------
np.ndarray (2,): the target position (x, y) in u that the pusher is commanded to go to during this agent step
(the pusher is controlled in position; the target is held for the whole 0.1 s step). Clipped to [0, 512]^2;
non-finite values raise ValueError.

STEP RETURN
-----------
reward      float, native: clip(coverage / 0.95, 0, 1) of the state reached after the step (not a sum).
terminated  True when coverage > 0.95 (success; the episode ends at once).
truncated   True at the 300th step if not terminated (time limit). Exactly one of them is True on the last step.
info        {"success", "max_coverage", "final_coverage", "n_steps"} (running values; final on the last step).
reset() returns obs only (t = 0).

BLACK-BOX LOG (calibration), runs/pusht/blackbox/<pilot>/life_<k>.npz  (numpy.load)
-----------------------------------------------------------------------------------
Arrays with a leading dimension T+1 (index 0 = the reset observation): agent_pos, agent_vel, block_pose, goal_pose,
coverage, t. Arrays with leading dimension T: actions (the arrays passed to step(), after clipping), rewards.
Scalars: seed, life, pilot, success, max_coverage, final_coverage, n_steps, terminated, truncated, control_hz.
The file is written when the episode ends or is abandoned (next reset() or close()).

SECOND OBJECT: THE L BLOCK  (make(..., object="L"); default object="T"; task_card(object="L"))
----------------------------------------------------------------------------------------------
Everything above applies unchanged (observation, action, step return, units, frames, angle convention, 300-step
limit, success rule coverage > 0.95, reward) except for the block, its goal, the lives and the logs:
Block: a flat "L" made of two rectangles, in the block's body frame (u, v as for the T):
    leg:  u in [0, 40],   v in [0, 110]         ( 40 x 110)
    foot: u in [40, 100], v in [0, 40]          ( 60 x 40)
i.e. the leg points along +v and the foot extends along +u from the leg's bottom; the body-frame origin is the
bottom-left outer corner of the L. Lengths are exact; nothing is published about mass or mass distribution.
Goal: the L drawn at pose (300, 200, pi/2) (light green in the GIFs); obs["goal_pose"] is that pose, and coverage is
area(block ∩ goal L) / area(goal L). The pusher is the same disc of radius 15 u.
Field note: this block came from the same batch as the T.
Lives on the L: make("calibration", pilot="<name>", object="L") has CAL_CAP_EPISODES_L = 2 episodes per pilot
(ledger key pusht_L:<pilot>), calibration seeds < 1000, one life at a time; the 3rd reset() raises
cawm.ledger.BudgetExceeded. Black-box logs: runs/pusht/blackbox_L/<pilot>/life_<k>.npz, k = 1..2 (same format as
above, plus a scalar "object" = "L"). Evaluation (seeds >= 1000) is counted under scenario pusht_L.
reallab.pusht.parallel.evaluate(..., object="L") runs a policy on the L.

OTHER
-----
make(..., record_path="x.gif"): records the env's own rendering (one frame per agent step, text overlay) for human
inspection; it never changes the dynamics. reallab.pusht.render.render_life(npz, gif) re-renders a saved life.
reallab.pusht.parallel.evaluate(...) runs a policy over many seeds in worker processes.

DETERMINISM (verified)
----------------------
The same reset(seed) followed by the same action sequence reproduced observations bit-for-bit across separate env
instances and across processes (see test_api.py); recording does not change them. Verified on one machine only.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cawm import ledger

CAL_CAP_EPISODES = 5     # lives per pilot for purpose="calibration" (T)
CAL_CAP_EPISODES_L = 2   # lives per pilot on the L
TIME_LIMIT = 300
SUCCESS_COVERAGE = 0.95
SEED_SPLIT = 1000        # calibration / oracle_train seeds < 1000; eval seeds >= 1000


def task_card(object: str = "T") -> dict:
    """Unmetered public facts about the task (the datasheet of the manual, as data)."""
    if object == "L":
        return dict(
            object="L",
            workspace=dict(low=[0.0, 0.0], high=[512.0, 512.0]),
            walls=dict(shape="rectangle", corners=[[5.0, 5.0], [506.0, 506.0]], half_thickness=2.0),
            goal_pose=[300.0, 200.0, float(np.pi / 2)],
            pusher=dict(shape="disc", radius=15.0),
            block=dict(shape="L", leg=dict(u=[0.0, 40.0], v=[0.0, 110.0]), foot=dict(u=[40.0, 100.0], v=[0.0, 40.0]),
                       pose_convention="(x, y, theta) of the body frame (origin at the L's outer corner); "
                                       "theta counter-clockwise, in [0, 2 pi)"),
            control_hz=10.0,
            success=dict(rule="coverage > 0.95", coverage="area(block ∩ goal) / area(goal)"),
            time_limit_steps=TIME_LIMIT,
            reward="clip(coverage / 0.95, 0, 1) after each step",
            calibration_lives=CAL_CAP_EPISODES_L,
        )
    if object != "T":
        raise ValueError("object must be 'T' or 'L'")
    return dict(
        workspace=dict(low=[0.0, 0.0], high=[512.0, 512.0]),
        walls=dict(shape="rectangle", corners=[[5.0, 5.0], [506.0, 506.0]], half_thickness=2.0),
        goal_pose=[256.0, 256.0, float(np.pi / 4)],
        pusher=dict(shape="disc", radius=15.0),
        block=dict(shape="T", bar=dict(u=[-60.0, 60.0], v=[0.0, 30.0]), stem=dict(u=[-15.0, 15.0], v=[30.0, 120.0]),
                   pose_convention="(x, y, theta) of the body frame; theta counter-clockwise, in [0, 2 pi)"),
        control_hz=10.0,
        success=dict(rule="coverage > 0.95", coverage="area(block ∩ goal) / area(goal)"),
        time_limit_steps=TIME_LIMIT,
        reward="clip(coverage / 0.95, 0, 1) after each step",
        calibration_lives=CAL_CAP_EPISODES,
    )


_BAR = np.array([(-60, 0), (60, 0), (60, 30), (-60, 30)], float)
_STEM = np.array([(-15, 30), (15, 30), (15, 120), (-15, 120)], float)


from reallab.pusht._geometry import L_FOOT as _LF, L_LEG as _LL  # noqa: E402

_LEG, _FOOT = np.array(_LL, float), np.array(_LF, float)
_RECTS = {"T": (_BAR, _STEM), "L": (_LEG, _FOOT)}
_GOAL_POSE = {"T": (256.0, 256.0, np.pi / 4), "L": (300.0, 200.0, np.pi / 2)}


def _t_geom(x, y, theta, object="T"):
    import shapely.geometry as sg
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return sg.MultiPolygon([sg.Polygon(v @ R.T + (x, y)) for v in _RECTS[object]])


_GOAL_GEOM = {}


def coverage_of(pose, object="T") -> float:
    """area(block at `pose` ∩ goal block) / area(goal block), computed from the pose alone (canonical)."""
    if object not in _GOAL_GEOM:
        _GOAL_GEOM[object] = _t_geom(*_GOAL_POSE[object], object)
    g = _GOAL_GEOM[object]
    return float(_t_geom(*pose, object).intersection(g).area / g.area)


def _blackbox_dir(pilot: str, object: str = "T") -> Path:
    base = Path(os.environ.get("CAWM_BLACKBOX_DIR") or Path(__file__).resolve().parents[2] / "runs" / "pusht" / "blackbox")
    if object == "L":
        base = base.with_name(base.name + "_L")
    return base / pilot


class RealPushT:
    """A metered, black-box episode interface to the Push-T rig. Create with make()."""

    def __init__(self, purpose, pilot, tag, seed, record_path, ledger_path, record_title=None, object="T"):
        from reallab.pusht import _sim
        if object not in ("T", "L"):
            raise ValueError("object must be 'T' or 'L'")
        self.object = object
        if purpose not in ledger.PURPOSES:
            raise ValueError(f"purpose must be one of {ledger.PURPOSES}")
        self._calib = purpose == "calibration"
        if self._calib and not pilot:
            raise ValueError("purpose='calibration' requires pilot='<name>' (lives are per pilot)")
        self.pilot, self.purpose = pilot or "", purpose
        if pilot and not self._calib:
            tag = f"{pilot}/{tag}" if tag else pilot
        self.tag = tag
        base = "pusht" if object == "T" else "pusht_L"
        self._cap = CAL_CAP_EPISODES if object == "T" else CAL_CAP_EPISODES_L
        self._ledger_key = f"{base}:{pilot}" if self._calib else base
        self._ledger_path = ledger_path or os.environ.get("CAWM_LEDGER", ledger.DEFAULT_PATH)
        self.meter = ledger.Meter("pusht", self._ledger_key, purpose, tag,
                                  cap_episodes=self._cap if self._calib else None, path=self._ledger_path)
        self._rng = np.random.default_rng(seed)
        self._sim = _sim.SimPushT(object=object)
        self.action_low = np.zeros(2)
        self.action_high = np.full(2, 512.0)
        self.control_hz = float(self._sim.control_hz)
        self.record_path, self.record_title = record_path, record_title
        self._rec = None
        self._active = False
        self._closed = False
        self._bb = None
        self._seed = None

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

    # ---- observation ----------------------------------------------------------------------------------
    def _obs(self):
        s = self._sim
        pose = np.array([*s.block.position, s.block.angle % (2 * np.pi)], dtype=np.float64)
        cov = coverage_of((*s.block.position, s.block.angle), self.object)
        return dict(agent_pos=np.array(s.agent.position, dtype=np.float64),
                    agent_vel=np.array(s.agent.velocity, dtype=np.float64),
                    block_pose=pose, goal_pose=np.array(s.goal_pose, dtype=np.float64),
                    coverage=cov, t=self._t)

    # ---- recording -------------------------------------------------------------------------------------
    def _start_rec(self):
        if self.record_path:
            from reallab.pusht.render import Recorder
            self._rec = Recorder(self.record_path, self.record_title or f"push-{self.object} [{self.purpose}] seed {self._seed}")

    def _rec_frame(self, obs):
        if self._rec is not None:
            self._rec.add(self._sim.render(), obs["t"], obs["coverage"])

    def stop_recording(self, outcome: str | None = None):
        if self._rec is not None:
            self._rec.close(outcome)
            self._rec = None

    # ---- black-box log ---------------------------------------------------------------------------------
    def _bb_flush(self):
        if self._bb is None:
            return
        L, self._bb = self._bb, None
        obs = L["obs"]
        T = len(L["actions"])
        out = {k: np.array([o[k] for o in obs]) for k in ("agent_pos", "agent_vel", "block_pose", "goal_pose",
                                                          "coverage", "t")}
        out["actions"] = np.array(L["actions"], dtype=np.float64).reshape(T, 2)
        out["rewards"] = np.array(L["rewards"], dtype=np.float64)
        out.update(seed=self._seed, life=L["life"], pilot=self.pilot, success=self._success,
                   max_coverage=self._max_cov, final_coverage=obs[-1]["coverage"], n_steps=T,
                   terminated=self._terminated, truncated=self._truncated, control_hz=self.control_hz)
        if self.object == "L":
            out["object"] = "L"
        L["dir"].mkdir(parents=True, exist_ok=True)
        np.savez_compressed(L["dir"] / f"life_{L['life']}.npz", **out)

    # ---- episode interface -----------------------------------------------------------------------------
    def _abandon(self):
        if self._active:
            self.meter.episode()
        self._active = False
        self._bb_flush()
        self.stop_recording()

    def reset(self, seed: int | None = None) -> dict:
        if self._closed:
            raise RuntimeError("env is closed")
        seed = self._draw_seed() if seed is None else self._check_seed(seed)
        self._abandon()
        self.meter.start_episode()   # raises BudgetExceeded when the lives cap is used up
        self._seed = seed
        self._sim.reset(seed=seed)
        self._active = True
        self._t, self._ret = 0, 0.0
        self._success = self._terminated = self._truncated = False
        obs = self._obs()
        self._max_cov = obs["coverage"]
        if self._calib:
            k = ledger.totals(self._ledger_path).get(("pusht", self._ledger_key, "calibration"), {"episodes": 1})["episodes"]
            self._bb = dict(dir=_blackbox_dir(self.pilot, self.object), life=int(k), obs=[obs], actions=[], rewards=[])
        self._start_rec()
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
        a = np.clip(a, self.action_low, self.action_high)
        self.meter.step()
        self._sim.step(a)
        self._t += 1
        obs = self._obs()
        cov = obs["coverage"]
        reward = float(np.clip(cov / SUCCESS_COVERAGE, 0.0, 1.0))
        self._max_cov = max(self._max_cov, cov)
        terminated = bool(cov > SUCCESS_COVERAGE)
        truncated = (not terminated) and self._t >= TIME_LIMIT
        self._success, self._terminated, self._truncated = terminated, terminated, truncated
        self._ret += reward
        info = dict(success=terminated, max_coverage=float(self._max_cov), final_coverage=cov, n_steps=self._t)
        if self._bb is not None:
            self._bb["obs"].append(obs)
            self._bb["actions"].append(a)
            self._bb["rewards"].append(reward)
        self._rec_frame(obs)
        if terminated or truncated:
            self._active = False
            self.meter.episode()
            self._bb_flush()
            self.stop_recording("SUCCESS" if terminated else f"timeout - max coverage {self._max_cov:.2f}")
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self._abandon()
        self._closed = True
        self.meter.flush()
        self._sim.close()


def make(purpose: str, pilot: str = "", tag: str = "", seed: int | None = None, record_path: str | None = None,
         ledger_path=None, record_title: str | None = None, object: str = "T") -> RealPushT:
    """Create a metered real Push-T rig. See the module docstring."""
    return RealPushT(purpose, pilot, tag, seed, record_path, ledger_path, record_title, object)
