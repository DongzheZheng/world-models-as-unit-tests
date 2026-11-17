"""
REAL DRONE -- INSTRUMENT MANUAL
================================

You are looking at the only interface to a real (physically simulated) quadrotor.  It is a black box: you may
issue commands and read sensors, nothing else.  Every interaction is metered and logged.  Hidden effects of
real physics exist and are NOT described here; discover them by experiment.

Quick start
-----------
    from reallab.drone.api import make
    env = make("waypoints" | "port", purpose="calibration" | "eval", tag="my_experiment", seed=None)
    obs, info = env.reset(seed=3)
    obs, reward, terminated, truncated, info = env.step(action)     # action: np.ndarray, shape (4,)
    env.close()

make(task, purpose, tag="", seed=None, **kw) -> RealDrone
    task     "waypoints", "port" or "site".
    purpose  "calibration": flights you may use to write / fit your world model.  Seeds must be in [0, 1000).
             Hard budget ("waypoints", "port"): CAL_CAP_STEPS = 30_000 agent steps in total PER TASK (all calls, all
             processes, all time).  Hard budget ("site"): FIVE episodes ("lives") PER PILOT, ever; the 6th reset()
             of a pilot raises cawm.ledger.BudgetExceeded.  Exceeding any budget raises BudgetExceeded and nothing is
             executed.
             "eval": final evaluation.  Seeds must be >= 1000.  Never use eval flights to change anything.
    tag      free-text label written to the ledger (use it to name your experiment).
    seed     optional default seed.  If given, successive reset() calls without an explicit seed use seed, seed+1, ...
             If omitted, reset() without seed draws a random seed from the range allowed for the purpose.
    kw       task options: "port" accepts difficulty in {"easy","medium","hard"} (default "medium"),
             realism (bool, default True), max_steps (int, optional cap on episode length in agent steps).
             "waypoints" accepts max_steps only.
             "site" accepts mission ("A" default; "B" is available for purpose="eval" only; "C": same relays as "A", with other aircraft in the yard), command_mode
             ("ctbr" default, or "position"; fixed for the episode) and pilot (a name; REQUIRED for
             purpose="calibration": the lives are counted per pilot).
RealDrone attributes: task (str), action_low / action_high (np.ndarray (4,), command bounds),
control_hz (float, 30.0).  Methods: reset(seed=None) -> (obs, info), step(action) ->
(obs, reward, terminated, truncated, info), close().  An episode that is abandoned by calling reset() or close()
before it ended still counts as one (metered) episode.  Calling step() after an episode has ended is an error.

Command interface (both tasks, identical)
-----------------------------------------
The 4-vector command is   a = (vp, vq, vr, T)   and is held constant for one agent step (1/30 s).
  vp, vq, vr  desired body angular rates [rad/s] about the body x, y, z axes (right-hand rule).  The drone's
              body frame is the usual quadrotor ENU body frame used by PyFlyt: x forward, y left, z up.
              Bounds: [-pi, pi] each.
  T           collective thrust command, normalised throttle in [0, 1] (the bounds above allow up to 0.8; values
              outside [0, 1] are clipped by the flight controller).  See the datasheet below.
Commands are NOT clipped by the interface: they are handed to the flight controller as given.
The flight controller is PyFlyt QuadX flight mode 0 (rate + thrust).  Physics runs at 240 Hz, the flight
controller at 120 Hz, the agent at 30 Hz (one agent step = 8 physics steps).

Datasheet -- HEALTHY airframe (as published by the manufacturer; a Crazyflie-class nano quadrotor, "cf2x")
-----------------------------------------------------------------------------------------------------------
  mass                              0.027 kg
  inertia (body, diagonal)          Ixx = Iyy = 1.4e-5 kg m^2, Izz = 2.17e-5 kg m^2
  rotor positions (body frame)      (+0.028, -0.028), (-0.028, +0.028), (+0.028, +0.028), (-0.028, -0.028) m,
                                    motors numbered 0..3 in that order (x forward, y left); motors 0,1 spin
                                    one way and 2,3 the other
  hover command (healthy unit)      T = 0.30  (zero body-rate command, level attitude)
  thrust command -> thrust          thrust grows roughly with T squared; healthy unit: T = 0.5 gives thrust/weight
                                    about 2.8, T = 0.8 (upper command bound) about 7.2, T = 1.0 (clipping limit)
                                    about 11.2.  Motor response time constant 10 ms.
  body-rate command limits          |vp|, |vq|, |vr| <= pi rad/s; the inner loop tracks a commanded rate to about
                                    90 % within one agent step and settles at about 88-92 % of the command
  control rate                      agent 30 Hz; flight controller 120 Hz; physics 240 Hz
  command semantics (ctbr)          a = (vp, vq, vr, T), body frame x forward / y left / z up, right-hand rule:
                                    vp > 0 rolls the right side down (the drone then accelerates toward -y),
                                    vq > 0 pitches the nose down (accelerates toward +x when heading +x),
                                    vr > 0 yaws counter-clockwise seen from above; T > 0 pushes along body +z.
  position autopilot (command_mode="position", site task only)
                                    PyFlyt flight mode 7, the manufacturer's own cascaded position controller,
                                    designed for the healthy airframe.  a = (x, y, yaw, z): WORLD-frame position
                                    set-point x, y [m], desired heading yaw [rad] (counter-clockwise from +x,
                                    z-axis up), and altitude set-point z [m].  Cascade: position error -> velocity
                                    command (gain 1 1/s, horizontal speed limited to 2 m/s, vertical speed to
                                    1 m/s) -> attitude command (tilt limited to about 0.4 rad) -> body rates ->
                                    motors.  The set-point is not clipped by the interface; the nominal bounds
                                    reported in action_low / action_high are the yard limits, yaw in [-pi, pi]
                                    and z in [0, 6].  Observations are identical in both command modes.

--------------------------------------------------------------------------------------------------------------
TASK "waypoints"  --  the formal benchmark, unmodified
--------------------------------------------------------------------------------------------------------------
This is Gymnasium env  PyFlyt/QuadX-Waypoints-v4  with its default physics and default task settings:
4 waypoints, dense reward, 10 s time limit, spherical flight dome of radius 5 m centred on the world origin,
waypoint reached when the drone is within 0.2 m of it; waypoints must be reached in order.  The drone starts
at (0, 0, 1) with zero attitude; the env then runs 10 internal flight-controller steps before the first
observation is returned, so the first observation is NOT exactly the start pose (the drone has begun to move).
Waypoints are random per episode (seeded).

Observation: dict of float64 numpy arrays (PyFlyt's own semantics)
    ang_vel      (3,)   body-frame angular velocity [rad/s]
    quat         (4,)   attitude quaternion (x, y, z, w) rotating body -> world
    lin_vel      (3,)   BODY-frame linear velocity [m/s]
    lin_pos      (3,)   WORLD-frame position [m]; z is up
    prev_action  (4,)   the last command passed to step() (zeros right after reset)
    aux          (4,)   motor throttle states of the four motors (normalised, 0..1)
    target_deltas (n,3) BODY-frame vectors from the drone to each REMAINING waypoint, in the order they must be
                        reached; n shrinks as waypoints are reached (n = 4 at the start).
Reward: the env's native dense reward (-0.1 per sim step and shaping terms, +100 when a waypoint is reached,
        -100 on crash or leaving the dome).  Per agent step the native reward is the sum over the 4 internal
        sub-steps.  Not modified in any way.
terminated: True on ground contact or leaving the dome.   truncated: True on timeout, and also when the last
waypoint has been reached (success is signalled through info["success"], not through "terminated").
info at episode end (all episodes) contains, besides PyFlyt's own keys:
    targets_reached (int), num_targets (int, 4), success (bool, all reached), crashed (bool, ground contact),
    out_of_bounds (bool), timeout (bool), time_s (float).

--------------------------------------------------------------------------------------------------------------
TASK "port"
--------------------------------------------------------------------------------------------------------------
PORT COURSE. A flat container yard 60 m long (x from 0 to 60) and 24 m wide (y from -12 to 12) with a flight
ceiling of 6 m. The drone starts hovering near x=3, y within +-3 m, z=2 m, facing +x. The goal is a point near
x=57 with y within +-6 m and z between 2 and 4 m. Between x=8 and x=52 the yard holds shipping containers (20 ft:
6.06 x 2.44 x 2.59 m; 40 ft: 12.19 x 2.44 x 2.59 m) stacked 1 to 3 high and arranged in blocks/rows with their
long axis along x or y, separated by aisles roughly 3-7 m wide. There are also a few gantry-crane legs (square
columns about 0.6 m wide, taller than the ceiling) and light poles (radius about 0.15 m, taller than the
ceiling) standing in the aisles. Stacks 3 high are taller than the ceiling and must be flown around; a
collision-free route from start to goal always exists below the ceiling, but the straight line is usually
blocked. Success = reaching within 1.0 m of the goal without touching anything, inside the 40 s time limit.

Layouts are random per episode (seeded); the seed selects the whole course.  The course is generated for the
chosen difficulty ("easy" < "medium" < "hard": denser, narrower aisles).  World frame: x along the yard, y across,
z up, origin at the yard corner/centreline as stated above, ground at z = 0.  The drone starts at rest in a
hover-like state with zero tilt and yaw 0 (facing +x).  Time starts at t = 0.

Observation: dict of float64 numpy arrays
    ang_vel (3,)     body-frame angular velocity [rad/s]   (gyro; body axes as in the command section)
    quat    (4,)     attitude quaternion (x, y, z, w) rotating body -> world
    lin_vel (3,)     WORLD-frame linear velocity [m/s]
    pos     (3,)     WORLD-frame position [m]
    goal    (3,)     WORLD-frame goal point [m]
    ranges  (6, 16)  range sensor array [m], see below
    t       (1,)     time since the start of the episode [s]  (= agent steps / 30)
Range sensor: 96 rays cast from the drone centre (ray tests against obstacles and the ground; no images).  The
rays are STABILISED: they are defined in the heading-aligned, roll/pitch-compensated frame at the drone position,
i.e. the frame obtained by rotating the world frame about z by the drone's current yaw (heading).  Layout:
    ranges[i, j]  with i = elevation index 0..5, j = azimuth index 0..15
    elevation_i = -25 + 10*i degrees  (-25, -15, -5, 5, 15, 25; positive = up)
    azimuth_j   = -60 + 8*j degrees   (-60, -52, ..., -4, 4, ..., 60; positive = counter-clockwise seen from above,
                                       i.e. toward the drone's left when the heading is +x; 0 = straight ahead)
    ray direction in the heading frame = (cos el * cos az, cos el * sin az, sin el); heading frame -> world is a
    rotation by yaw about z.  Return value = distance along the ray from the drone centre to the first hit (any
    obstacle or the ground), and exactly 12.0 (maximum range) if nothing is hit within 12 m.
Reward (dense):  r = (d_prev - d) - 0.01  per agent step, where d = Euclidean distance (true position) to the
goal; plus +50 on success; minus 50 on collision or leaving the allowed region.
Termination:  terminated = success (true position within 1.0 m of the goal) or collision (any contact between
the drone and an obstacle or the ground) or out of bounds (outside the yard footprint by more than 2 m, i.e.
x < -2, x > 62, |y| > 14, or above the 6 m ceiling).   truncated = 40 s (1200 agent steps) timeout.
info at episode end: success, collision, out_of_bounds, timeout (bools), time_s, path_length_m (travelled path of
the true position), mean_speed (path_length_m / time_s), min_range_seen (smallest value that appeared in the
returned ranges during the episode).
info at reset: {"seed": int}; additionally, ONLY for purpose="calibration", info["course"] = site survey of that
episode's course:
    {"bounds": {"x": [0, 60], "y": [-12, 12], "ceiling": 6.0},
     "boxes":     [{"kind": "container" | "crane_leg", "center": [x, y, z], "half_extents": [hx, hy, hz]}, ...],
     "cylinders": [{"kind": "pole", "center": [x, y, z], "radius": r, "height": h}, ...]}
    Boxes are axis-aligned (center is the geometric centre of the box; a stack of containers is one box).
    Cylinders are vertical (center is the geometric centre, so z = height / 2 for objects standing on the ground).
Hidden realism: the port task is not a clean simulation.  Disturbances, sensor imperfections, actuation delays and
unit-to-unit variations exist (seeded per episode and not disclosed here).  make("port", ..., realism=False)
switches them all off -- this option is available only for purpose="smoke" (lab-internal) and "eval" ablations; it
is rejected for purpose="calibration".

--------------------------------------------------------------------------------------------------------------
TASK "site"
--------------------------------------------------------------------------------------------------------------
One fixed site, one fixed aircraft.  Yard 60 m x 24 m (x in [0, 60], y in [-12, 12], ground z = 0), flight ceiling
6 m.  Start near (3, 0, 2), facing +x.  A "canyon" runs along x between two walls of 3-high container stacks
(7.8 m tall, so they cannot be overflown): about 40 m long (x from 10 to 50), about 4 m clear width, pinched to
about 3.2 m somewhere, with two side aisles cutting through the walls (one in each).  The canyon is the only
route to the far end.  site_map() (api module function, unmetered) returns the static geometry:
    {"bounds": {"x": [0, 60], "y": [-12, 12], "ceiling": 6.0},
     "boxes": [{"kind", "center": [x, y, z], "half_extents": [hx, hy, hz]}, ...],   # axis-aligned, as in "port"
     "cylinders": [], "start": [3, 0, 2], "goal": [57, 0, 2.5],
     "missions": {"A": [relay1, relay2, relay3]}, "time_limit_s": 17.0}   # time limit of mission A
MISSION.  Three uplink relays must be reached in order: the drone must come within 1.0 m of relay 1, then relay 2,
then relay 3 ("success").  Mission "A" is published in site_map() and is the one used for calibration and for
final evaluation; mission "B" (same site, same aircraft, different relay positions) is held out and is available
for purpose="eval" only.  Each episode jitters the start by about +-1 m and the relays by about +-0.5 m, and reseeds
the sensor noise; nothing else changes between episodes.
Termination: success; collision (any contact with a wall or the ground); out of bounds (outside the yard
footprint by more than 2 m, or above the ceiling); timeout.  The time limit is per mission: mission A 17.0 s
(510 agent steps), mission B 24.2 s (726 agent steps); it is reported as info["time_limit_s"] at the end of an
episode and, for mission A, as site_map()["time_limit_s"].
TIME WINDOW.  The built-in position autopilot (command_mode="position") limits its horizontal speed to 2 m/s
(see the datasheet).  At that speed the relay route of mission A (about 54 m) takes at least 27 s and mission B
(about 62 m plus the detour) at least 32 s, even in still air: it is therefore TOO SLOW for the time limits and
cannot complete any of the three missions inside the window, whatever path it is given.  The window requires flying
substantially faster than the autopilot's speed limit, i.e. with command_mode="ctbr".
MISSION C.  Same relays as mission A, available for calibration (five lives per pilot) and for eval.  Other aircraft
operate in the yard on mission C. Their behaviour is not documented.  Touching one (each is a sphere of radius about
0.35 m) ends the episode as "collision".  The time limit of mission C is 19.4 s (582 agent steps).  Two extra
observation fields exist on mission C only:
    movers    (8, 7)  one row per other aircraft that is currently within 15 m of the drone AND in line of sight,
                      nearest first: position (3, world frame), velocity (3, world frame), radius; unused rows are
                      zero
    n_movers  (1,)    number of non-zero rows
The range sensor also sees the other aircraft.  For calibration lives on mission C the black-box file additionally
stores, as instrumentation, the true positions and velocities of all other aircraft at every step
(movers_true_pos, movers_true_vel: (steps+1, 8, 3), movers_true_count).
Observation (both command modes): ang_vel (3), quat (4), lin_vel (3), pos (3), ranges (6, 16), t (1) exactly as for
"port" (same frames, same ray layout, same maximum range 12 m), plus
    goal         (3,)    the CURRENT target relay (world frame)
    relay_index  (1,)    index 0..2 of the current target relay
    relays       (3, 3)  all three relays of this episode's mission
Reward: (d_prev - d) - 0.01 per step with d the distance to the current relay, +20 for each relay reached,
+50 on completing the mission, -50 on collision or leaving the allowed region.  info at reset: seed, mission,
command_mode.  info at episode end: success, collision, out_of_bounds, timeout, time_s, path_length_m, mean_speed,
min_range_seen, relays_reached, time_limit_s.
Lives (purpose="calibration"): five episodes per pilot, ever; the pilot name must be given to make().  Every
calibration episode is automatically written to runs/drone/blackbox/<pilot>/life_<k>.npz (k = 1..5): all
observation fields for every step (including the initial one), the commands, the rewards, the end cause, the
mission and the command mode used.  Sensors only.
Hidden realism: sensors are slightly noisy and the command is applied with a small delay.

Field note: this unit has an open maintenance flag after a hard landing, and site crews report 'odd air' in the yard. Nobody has characterised either.

Batch evaluation and logging
----------------------------
    from reallab.drone.parallel import evaluate
    rows = evaluate("package.module:factory", {"kw": 1}, "port", seeds=[1000, 1001], purpose="eval", tag="x",
                    n_workers=32, log_dir=None, difficulty="medium")
`factory(**policy_kwargs)` must return an object with reset() -> None and act(obs: dict) -> np.ndarray (4,);
it is imported and instantiated inside each worker process.  One episode per seed; each returned dict has
seed, ret, n_steps plus the end-of-episode info metrics.  With log_dir, one gzipped JSONL file per episode is
written (header line with seed/reset info incl. info["course"] when calibrating, then one line per step with
obs, action, reward, terminated, truncated, and a final line with the end-of-episode info).
For task "site" pass pilot=<name> (and mission=, command_mode=) as keyword arguments; calibration lives are flown
one at a time (n_workers is forced to 1).
CLI:  python -m reallab.drone.collect --task port --purpose calibration --seeds 0:10 --policy-file actions.npy
      --out runs/drone/cal_x  (replays an open-loop (T,4) action sequence from a .npy file and logs the flights;
      or --policy package.module:factory --policy-kwargs '{"a": 1}').
Built-in policies:  ":hover_policy" (a crude constant command), ":random_policy" (uniform random command).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from cawm.ledger import Meter, totals

CAL_CAP_STEPS = 30_000
SITE_LIVES = 5
_TASKS = ("waypoints", "port", "site")
_SEED_RANGES = {            # purpose -> [lo, hi)
    "calibration": (0, 1000),
    "eval": (1000, 1_000_000),
    "oracle_train": (1_000_000, 2_000_000_000),
    "smoke": (0, 2_000_000_000),
}
_PUBLIC_KW = {"waypoints": {"max_steps"}, "port": {"difficulty", "realism", "max_steps"},
              "site": {"mission", "pilot", "command_mode"}}


def site_map() -> dict:
    """Unmetered public survey of the site (see the manual): static geometry, nominal start / goal, missions available
    for practice."""
    from ._site import site_map as _sm
    return _sm()


class _CappedMeter(Meter):
    """Meter that publishes in-flight steps to the ledger every few steps and re-reads the global total, so that the
    calibration cap also holds approximately across parallel worker processes."""
    FLUSH_EVERY = 25

    def step(self, n: int = 1) -> None:
        super().step(n)
        if self.cap_steps is not None and self._steps >= self.FLUSH_EVERY:
            self.flush()
            self._spent = None


def _check_seed(seed, purpose):
    lo, hi = _SEED_RANGES[purpose]
    if not (isinstance(seed, (int, np.integer)) and lo <= seed < hi):
        raise ValueError(f"purpose={purpose!r} requires an integer seed in [{lo}, {hi}); got {seed!r}")
    return int(seed)


def _blackbox_root() -> Path:
    return Path(os.environ.get("CAWM_BLACKBOX_DIR", Path(__file__).resolve().parents[2] / "runs" / "drone" / "blackbox"))


class RealDrone:
    def __init__(self, task: str, purpose: str, tag: str = "", seed: int | None = None, _lab: dict | None = None, **kw):
        if task not in _TASKS:
            raise ValueError(f"task must be one of {_TASKS}")
        if purpose not in _SEED_RANGES:
            raise ValueError(f"purpose must be one of {tuple(_SEED_RANGES)}")
        lab = dict(_lab or {})
        bad = set(kw) - _PUBLIC_KW[task]
        if bad:
            raise TypeError(f"unknown options for task {task!r}: {sorted(bad)}")
        self.task, self.purpose, self.tag = task, purpose, tag
        self.control_hz = 30.0
        flight_mode = lab.get("flight_mode", 0)
        self.pilot = None
        self.mission = None
        self.command_mode = "ctbr"
        scenario = task
        cap_episodes = None
        if flight_mode != 0 and purpose == "calibration":
            raise ValueError("unsupported option for purpose='calibration'")
        if task == "port":
            realism = kw.get("realism", True)
            if realism is not True and purpose == "calibration":
                raise ValueError("realism=False is not available for purpose='calibration'")
            if lab.get("_course") is not None and purpose != "smoke":
                raise ValueError("custom courses are lab-internal (purpose='smoke' only)")
            from ._port_env import PortEnv
            self._inner = PortEnv(difficulty=kw.get("difficulty", "medium"), realism=realism, flight_mode=flight_mode,
                                  max_steps=kw.get("max_steps"), course=lab.get("_course"))
        elif task == "waypoints":
            from ._waypoints_env import WaypointsEnv
            self._inner = WaypointsEnv(flight_mode=flight_mode, max_steps=kw.get("max_steps"))
        else:
            from ._site import SiteEnv
            self.mission = kw.get("mission", "A")
            self.command_mode = kw.get("command_mode", "ctbr")
            self.pilot = kw.get("pilot")
            if self.command_mode not in ("ctbr", "position"):
                raise ValueError("command_mode must be 'ctbr' or 'position'")
            if self.mission not in ("A", "B", "C"):
                raise ValueError("mission must be 'A', 'B' or 'C'")
            if purpose == "calibration":
                if self.mission == "B":
                    raise ValueError("mission 'B' is not available for purpose='calibration'")
                if not (isinstance(self.pilot, str) and re.fullmatch(r"[A-Za-z0-9_.\-]+", self.pilot)):
                    raise ValueError("purpose='calibration' at the site requires pilot=<name> (letters, digits, _.-)")
                scenario = f"site:{self.pilot}"
                cap_episodes = SITE_LIVES
            elif self.pilot:
                tag = f"{tag}|pilot={self.pilot}"
                self.tag = tag
            fm = 7 if self.command_mode == "position" else 0
            self._inner = SiteEnv(mission=self.mission, lab=lab, purpose=purpose, flight_mode=fm)
        self.action_low = np.array(self._inner.action_low, dtype=np.float64)
        self.action_high = np.array(self._inner.action_high, dtype=np.float64)
        self.meter = _CappedMeter("drone", scenario, purpose, tag,
                                  cap_steps=CAL_CAP_STEPS if (purpose == "calibration" and task != "site") else None,
                                  cap_episodes=cap_episodes)
        self._default_seed = None if seed is None else _check_seed(seed, purpose)
        self._episodes_started = 0
        self._active = False
        self._closed = False
        self._rng = np.random.default_rng()
        self._bb = None
        self._life = None

    # ------------------------------------------------------------------ black box (site calibration only)
    def _bb_start(self, obs, seed):
        self._life = totals(self.meter.path).get(self.meter._key, {"episodes": 0})["episodes"]
        self._bb = dict(obs=[obs], act=[], rew=[], seed=seed, mv=[])
        self._bb_movers()

    def _bb_movers(self):
        if self.mission == "C":
            P, V = self._inner.mover_truth()
            n = len(P)
            self._bb["mv"].append((np.vstack([P, np.zeros((8 - n, 3))]), np.vstack([V, np.zeros((8 - n, 3))]), n))

    def render_extra(self):
        """Rendering helper for logs (not part of the policy interface): true positions of the other aircraft."""
        if self.task == "site" and self.mission == "C":
            return self._inner.mover_truth()[0]
        return None

    def _bb_save(self, cause, info):
        if self._bb is None:
            return
        bb, self._bb = self._bb, None
        obs = bb["obs"]
        out = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        d = _blackbox_root() / self.pilot
        d.mkdir(parents=True, exist_ok=True)
        if bb["mv"]:
            out["movers_true_pos"] = np.stack([m[0] for m in bb["mv"]])
            out["movers_true_vel"] = np.stack([m[1] for m in bb["mv"]])
            out["movers_true_count"] = np.array([m[2] for m in bb["mv"]])
        np.savez_compressed(d / f"life_{self._life}.npz", **out, actions=np.array(bb["act"]).reshape(-1, 4),
                            rewards=np.array(bb["rew"]), end_cause=cause, seed=bb["seed"], pilot=self.pilot,
                            life=self._life, mission=self.mission, command_mode=self.command_mode,
                            relays_reached=int(info.get("relays_reached", 0)), n_steps=len(bb["act"]),
                            control_hz=self.control_hz)

    @staticmethod
    def _cause(info):
        for k in ("success", "collision", "out_of_bounds", "timeout"):
            if info.get(k):
                return k
        return "unknown"

    # ------------------------------------------------------------------ helpers
    def _abandon(self):
        if self._active:
            self._active = False
            self._bb_save("abandoned", {"relays_reached": getattr(self._inner, "k", 0)})
            self.meter.episode()

    def reset(self, seed: int | None = None):
        if self._closed:
            raise RuntimeError("environment is closed")
        self._abandon()
        if seed is None:
            if self._default_seed is not None:
                seed = self._default_seed + self._episodes_started
            else:
                lo, hi = _SEED_RANGES[self.purpose]
                seed = int(self._rng.integers(lo, min(hi, lo + 1_000_000 if self.purpose != "calibration" else hi)))
        seed = _check_seed(seed, self.purpose)
        self.meter.start_episode()              # raises BudgetExceeded when the lives are used up
        self._episodes_started += 1
        obs, info = self._inner.reset(seed)
        info = dict(info)
        if self.task == "port" and self.purpose == "calibration":
            from . import _course
            info["course"] = _course.public_view(self._inner.course)
        if self.task == "site":
            info["mission"] = self.mission
            info["command_mode"] = self.command_mode
            if self.purpose == "calibration":
                self._bb_start(obs, seed)
        self._active = True
        return obs, info

    def step(self, action):
        if self._closed:
            raise RuntimeError("environment is closed")
        if not self._active:
            raise RuntimeError("step() called with no running episode; call reset()")
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (4,) or not np.all(np.isfinite(a)):
            raise ValueError("action must be a finite array of shape (4,)")
        self.meter.step()
        obs, r, te, tr, info = self._inner.step(a)
        if self._bb is not None:
            self._bb["obs"].append(obs); self._bb["act"].append(a); self._bb["rew"].append(r)
            self._bb_movers()
        if te or tr:
            self._active = False
            self._bb_save(self._cause(info), info)
            self.meter.episode()
        return obs, r, te, tr, info

    def close(self):
        if self._closed:
            return
        self._abandon()
        self._closed = True
        self._inner.close()
        self.meter.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make(task: str, purpose: str, tag: str = "", seed: int | None = None, **kw) -> RealDrone:
    return RealDrone(task, purpose, tag, seed, **kw)
