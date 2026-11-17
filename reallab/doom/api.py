"""REAL-WORLD INSTRUMENT MANUAL: "Doom lab" (black-box real world for the FPS demo)
=====================================================================================

The real world is a first-person-shooter game engine. You get a metered black box: send discrete actions,
receive structured sensor readings. There are no pixels. Nothing else about the engine is documented here;
anything not in this manual (how monsters decide, how things move, damage, ...) is to be learned from data.

USAGE
-----
    from reallab.doom import api
    env = api.make("take_cover", purpose="eval", tag="my_run", seed=0, frame_skip=4)
    obs = env.reset(seed=123)                       # same seed + same actions => identical trajectory
    obs, reward, terminated, truncated, info = env.step(action_id)   # action_id indexes env.action_names
    env.close()
    api.floor_plan("take_cover")                    # static site survey, free of charge, not metered
    api.SCENARIOS                                   # name -> spec (config + button list)

Purposes: "calibration" (LIVES PER PILOT: make(..., purpose="calibration", pilot="<name>") is required; each
(scenario, pilot) has CAL_CAP_EPISODES = 5 episodes in total and at most CAL_CAP_STEPS = 20000 agent steps;
the 6th reset() of that pilot raises cawm.ledger.BudgetExceeded, a step beyond the step cap
raises it *before* being executed. Every reset() uses up a life, even if the previous episode was abandoned. The
lives cap is enforced atomically in the ledger, but fly calibration lives one at a time (evaluate() forces
one worker for calibration). For other purposes `pilot` is optional and is only prefixed to the ledger tag. Every calibration
episode is saved automatically by the env as runs/doom/blackbox/<scenario>/<pilot>/life_<k>.jsonl.gz, k = 1..5, in the
step-log format of reallab.doom.parallel including gt_objects. Only this purpose receives ground-truth object
lists), "eval" (final evaluation only), "oracle_train", "smoke" (plumbing tests).
Every agent step and episode is counted in the ledger. An episode that is abandoned by reset()/close()
before it finished still counts as an episode. Stepping after an episode ended raises RuntimeError.

TIME AND UNITS
--------------
* Engine time is in tics (nominally 35 tics per second of game time; the simulation is not wall-clock).
* One agent step = `frame_skip` tics (default 4) with the same action held; reward is the sum over those tics.
* Length is in map units; angles in degrees; velocities in map units per tic.
* obs["tic"] is the engine's episode tic counter *including the scenario start-up tics* (after reset() it equals
  the scenario's start offset: 1 for take_cover/health_gathering/deadly_corridor, 10 for defend_the_*, 14 for
  basic, 16 for predict_position). It then grows by frame_skip per step. On the final (terminal) observation
  the engine sometimes reports no tic; then tic = previous tic + frame_skip (an upper bound, off by < frame_skip).
  Scenario timeouts below are counted from that start offset, i.e. a 300-tic timeout ends at tic offset + 300.

COORDINATES AND ANGLES (verified empirically)
---------------------------------------------
* Right-handed 2-D map plane (x, y), z is height (up). Positions are of the actor's base (z = 0 for the agent
  throughout all my tests, i.e. the floors are flat; eye height is not reported).
* angle: degrees in [0, 360). Angle 0 faces +x; angle increases counter-clockwise seen from above (towards +y).
  TURN_LEFT increases angle, TURN_RIGHT decreases it. MOVE_FORWARD moves along (cos angle, sin angle);
  MOVE_LEFT strafes towards the player's left (at angle 0 that is +y), MOVE_RIGHT towards the right.
* Object `angle` uses the same convention (facing / direction of travel for projectiles).
* Velocities are per tic in the same axes (verified: position change over a step / frame_skip matches the velocity
  reading to within the acceleration over the step).

OBSERVATION (all scenarios; dict)
---------------------------------
obs["tic"]        int, see above.
obs["player"]     dict, all measured on the agent's own body:
    x, y, z          position
    angle            heading, degrees [0, 360)
    pitch            always 0 (no look up/down actions)
    vx, vy, vz       velocity, units/tic
    health           hit points (100 at start, except health_gathering where it is already 92 at the first
                     observation; can be <= 0 at death)
    armor            armor points
    selected_weapon  weapon slot number (-1 if the agent holds no weapon)
    ammo             ammo of the selected weapon (-1 if there is no weapon / no ammo type)
    attack_ready     bool, weapon ready to fire again
    killcount        kills since episode start (int)
    itemcount        pickups since episode start (int)
    dead             bool
    on_ground        bool
    damage_taken     cumulative damage received since episode start
    hits_taken       cumulative number of damaging hits received
    damage_dealt     cumulative damage dealt since episode start
    hits_dealt       cumulative number of hits dealt
obs["objects"]    list (sorted by id) of objects *currently visible on screen* (any visible pixel; the agent itself
                  is excluded). Each is a dict:
    id (int, stable within an episode), name (str class name), x, y, z, angle, vx, vy, vz   (world frame, as
    above) and the screen bounding box sx, sy, sw, sh = left, top, width, height in pixels of the 320 x 240
    screen. The reading includes enemies, projectiles, pickups and cosmetic effects (blood, bullet puffs, smoke).
    Some objects exist but never become visible (e.g. invisible waypoint markers in predict_position).
obs["depth_scan"] float32 array, length 64: a planar range sensor, see below.

Screen geometry: 320 x 240 pixels, horizontal field of view 90 degrees, perspective projection. Pixel column c
(0 = left edge) looks along bearing b (relative to the heading, positive = to the left) with
tan(b) = (159.5 - c) / 160. (Verified against object bearings: predicted vs. bbox-centre columns agree to ~2 px.)

depth_scan: row 120 (horizontal centre line) of the engine's depth buffer, averaged over blocks of 5 columns:
element j covers columns 5j..5j+4 (j = 0 is the leftmost, i.e. the most positive bearing), centred on column 5j+2.
Raw engine values are 8-bit integers (0..255, so quantised), larger = farther; they are NOT normalised or
converted. What I established by regressing against wall distances computed from floor_plan() (health_gathering
room, ~2000 samples, columns across the whole field of view): value ~= 0.140 * z - 1.2, residual std 0.3 counts,
where z is the distance to the hit point measured *along the viewing direction* (z = euclidean distance * cos(b),
NOT the euclidean range along the ray: a flat wall seen head-on gives a constant reading across the scan).
Equivalently z ~= (value + 1.2) / 0.14 map units, i.e. one count ~ 7 units. Other objects (monsters, projectiles)
occlude walls and are present in the scan. The largest value observed in my tests was 114 (z ~ 820 units).
I do not know whether the mapping saturates at larger z or is exactly linear; treat it as measured, not specified.

Terminal observation: when an episode ends, the player dict is still valid (final state, dead flag etc.), but
objects = [] and depth_scan = zeros (nothing is rendered after the end).

STEP RETURN
-----------
reward      float, the engine's native reward summed over the frame_skip tics.
terminated  the game ended the episode by itself (agent died, or the scenario goal was reached, e.g. the
            target was killed). truncated: the scenario's tic timeout (see below) was hit while the agent was
            alive. Exactly one of them is True on the last step of an episode.
info        {"timeout": bool}. For purpose == "calibration" additionally
            info["gt_objects"]: list of ALL objects in the map (visible or not; the agent excluded), same fields as
            obs["objects"] plus "visible" (bool); sx..sh are None when not visible. (It is [] on the terminal
            step.) The reset observation's ground truth is env.last_gt_objects (calibration only). These keys
            never exist for any other purpose.

ACTIONS (index = position in env.action_names; 0 is always "noop"; a name "A+B" holds both buttons)
------------------------------------------------------------------------------------------------
take_cover         0 noop, 1 MOVE_LEFT, 2 MOVE_RIGHT
defend_the_center  0 noop, 1 TURN_LEFT, 2 TURN_RIGHT, 3 ATTACK, 4 TURN_LEFT+ATTACK, 5 TURN_RIGHT+ATTACK
defend_the_line    same as defend_the_center
health_gathering   0 noop, 1 TURN_LEFT, 2 TURN_RIGHT, 3 MOVE_FORWARD, 4 MOVE_FORWARD+TURN_LEFT,
                   5 MOVE_FORWARD+TURN_RIGHT
predict_position   0 noop, 1 TURN_LEFT, 2 TURN_RIGHT, 3 ATTACK, 4 TURN_LEFT+ATTACK, 5 TURN_RIGHT+ATTACK
basic              0 noop, 1 MOVE_LEFT, 2 MOVE_RIGHT, 3 ATTACK, 4 MOVE_LEFT+ATTACK, 5 MOVE_RIGHT+ATTACK
deadly_corridor    0 noop, 1 MOVE_LEFT, 2 MOVE_RIGHT, 3 ATTACK, 4 MOVE_FORWARD, 5 MOVE_BACKWARD, 6 TURN_LEFT,
                   7 TURN_RIGHT, 8 MOVE_FORWARD+ATTACK, 9 MOVE_FORWARD+TURN_LEFT, 10 MOVE_FORWARD+TURN_RIGHT,
                   11 TURN_LEFT+ATTACK, 12 TURN_RIGHT+ATTACK, 13 MOVE_LEFT+ATTACK, 14 MOVE_RIGHT+ATTACK
(Turn/move rates are not given; measure them. Buttons are binary: held or not for the whole agent step.)

SCENARIOS: reward and episode end (from the public scenario configs / descriptions)
-----------------------------------------------------------------------------------
take_cover         reward +1 per tic alive. Ends (terminated) only on death; no timeout in the config. The agent
                   has no weapon and can only strafe.
defend_the_center  reward +1 per kill (verified), -1 on death. Ends on death or tic timeout 2100. The agent can
                   only turn and shoot (no translation).
defend_the_line    reward +1 per kill (verified), -1 on death. No timeout in the config (ends on death).
health_gathering   reward +1 per tic alive, -100 on death (both verified). Timeout 2100 tics. Health decreases over
                   time without any attacker; objects named Medikit lie around the room.
predict_position   living reward -0.001 per tic (verified); +1 for killing the target (verified). Observed: random play
                   mostly ends (terminated) after ~80 tics with no kill, i.e. the episode also ends by other
                   game-defined conditions (the agent holds one shot, ammo 1), or at tic timeout 300.
basic              living reward -1 per tic (verified); the step in which the single target is killed paid
                   ~ +97 in total in my tests (public description: +100 for the kill, -5 per shot). Ends when the
                   target dies (terminated) or at tic timeout 300.
deadly_corridor    reward shaped by progress along the corridor towards the goal (public description; per-step
                   rewards of order +-10 seen), -100 on death (config); timeout 2100 tics.
(Reward terms not marked "verified" are taken from the public configs/descriptions and were not checked in detail.)

OTHER FUNCTIONS
---------------
floor_plan(scenario) -> {"scenario", "sectors": [{"id", "floor", "ceiling", "lines": [{x1,y1,x2,y2,blocking}]}],
                         "walls": [deduplicated blocking segments], "bounds": {xmin,xmax,ymin,ymax}}
    Static geometry only (no objects): floor / ceiling heights per sector and the sector boundary line segments in
    the same x,y frame as the observations. "blocking" lines stop movement. Not metered.
make(..., record_path="x.mp4" or "x.gif", record_format="mp4"|"gif"): optionally records the rendered screen with a
    text overlay (one frame per 2 tics) for human inspection; sensors never use pixels.

DETERMINISM (verified)
----------------------
For all 7 scenarios, the same reset(seed) followed by the same action sequence reproduced observations (including
visible objects, depth scan, rewards, ground truth) bit-for-bit across: separate env instances, repeated reset(seed)
within one instance, and different worker processes (reallab.doom.parallel). Different seeds give different
trajectories. reset(None) draws a seed from the RNG given to make(seed=...) so it is reproducible only if make()
got a seed. Caveats: verified on one machine/engine version only; frame_skip must be the same.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np

from cawm import ledger
from reallab.doom import _game

CAL_CAP_STEPS = 20_000
CAL_CAP_EPISODES = 5     # lives per (scenario, pilot) for purpose="calibration"


def _spec(cfg, singles, combos=()):
    return dict(cfg=cfg, actions=[(b,) for b in singles] + [tuple(c) for c in combos])


SCENARIOS = {
    "take_cover": _spec("take_cover.cfg", ["MOVE_LEFT", "MOVE_RIGHT"]),
    "defend_the_center": _spec("defend_the_center.cfg", ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
                               [("TURN_LEFT", "ATTACK"), ("TURN_RIGHT", "ATTACK")]),
    "defend_the_line": _spec("defend_the_line.cfg", ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
                             [("TURN_LEFT", "ATTACK"), ("TURN_RIGHT", "ATTACK")]),
    "health_gathering": _spec("health_gathering.cfg", ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"],
                              [("MOVE_FORWARD", "TURN_LEFT"), ("MOVE_FORWARD", "TURN_RIGHT")]),
    "predict_position": _spec("predict_position.cfg", ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
                              [("TURN_LEFT", "ATTACK"), ("TURN_RIGHT", "ATTACK")]),
    "basic": _spec("basic.cfg", ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
                   [("MOVE_LEFT", "ATTACK"), ("MOVE_RIGHT", "ATTACK")]),
    "deadly_corridor": _spec("deadly_corridor.cfg",
                             ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "MOVE_FORWARD", "MOVE_BACKWARD",
                              "TURN_LEFT", "TURN_RIGHT"],
                             [("MOVE_FORWARD", "ATTACK"), ("MOVE_FORWARD", "TURN_LEFT"),
                              ("MOVE_FORWARD", "TURN_RIGHT"), ("TURN_LEFT", "ATTACK"),
                              ("TURN_RIGHT", "ATTACK"), ("MOVE_LEFT", "ATTACK"), ("MOVE_RIGHT", "ATTACK")]),
}


def _action_name(combo) -> str:
    return "+".join(combo)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def _blackbox_dir(scenario: str, pilot: str = "") -> Path:
    base = os.environ.get("CAWM_BLACKBOX_DIR") or Path(__file__).resolve().parents[2] / "runs" / "doom" / "blackbox"
    return Path(base) / scenario / pilot if pilot else Path(base) / scenario


class RealDoom:
    """A metered, black-box episode interface to one scenario. Create with make()."""

    def __init__(self, scenario, purpose, tag, seed, frame_skip, record_path, ledger_path,
                 record_format="mp4", record_title=None, pilot=None):
        if scenario not in SCENARIOS:
            raise KeyError(f"unknown scenario {scenario!r}; known: {sorted(SCENARIOS)}")
        if purpose not in ledger.PURPOSES:
            raise ValueError(f"purpose must be one of {ledger.PURPOSES}")
        self._calib = purpose == "calibration"
        if self._calib and not pilot:
            raise ValueError("purpose='calibration' requires pilot='<name>' (lives are per pilot)")
        self.pilot = pilot
        if pilot and not self._calib:
            tag = f"{pilot}/{tag}" if tag else pilot
        self.scenario, self.purpose, self.tag, self.frame_skip = scenario, purpose, tag, int(frame_skip)
        self._ledger_key = f"{scenario}:{pilot}" if self._calib else scenario
        spec = SCENARIOS[scenario]
        self.action_names = ["noop"] + [_action_name(c) for c in spec["actions"]]
        self._rng = np.random.default_rng(seed)
        self.record_path, self.record_format, self.record_title = record_path, record_format, record_title
        self._rec = None
        self._ledger_path = ledger_path or os.environ.get("CAWM_LEDGER", ledger.DEFAULT_PATH)
        self.meter = ledger.Meter("doom", self._ledger_key, purpose, tag,
                                  cap_steps=CAL_CAP_STEPS if self._calib else None,
                                  cap_episodes=CAL_CAP_EPISODES if self._calib else None,
                                  path=self._ledger_path)
        self._core = _game.Core(spec["cfg"], spec["actions"], self.frame_skip, rgb=True if record_path else False)
        self._active = False     # an episode is in progress (reset called, not finished)
        self._ep_steps = 0
        self._closed = False
        self._bb = None          # black-box log file (calibration)

    # ---- recording (optional) -------------------------------------------------------------
    def _open_writer(self):
        if self.record_path and self._rec is None:
            from reallab.doom.render import Recorder
            self._rec = Recorder(self.record_path, self.record_format,
                                 self.record_title or f"{self.scenario} [{self.purpose}]")

    def _frame(self, screen, tic=None, health=None, kills=None):
        if self._rec is not None and screen is not None:
            p = self._core.game
            self._rec.add(screen, self._core.episode_time() if tic is None else tic,
                          p.get_game_variable(_game.GV.HEALTH) if health is None else health,
                          int(p.get_game_variable(_game.GV.KILLCOUNT)) if kills is None else kills)

    def _stop_writer(self):
        if self._rec is not None:
            self._rec.close()
            self._rec = None

    # ---- black-box log (calibration) --------------------------------------------------------
    def _bb_write(self, rec):
        if self._bb is not None:
            self._bb.write(json.dumps(_jsonable(rec)) + "\n")

    def _bb_close(self):
        if self._bb is not None:
            self._bb.close()
            self._bb = None

    # ---- episode interface ----------------------------------------------------------------
    def _abandon(self):
        if self._active:
            self.meter.episode()
        self._active = False
        self._bb_close()

    def reset(self, seed: int | None = None) -> dict:
        if self._closed:
            raise RuntimeError("env is closed")
        self._abandon()
        self.meter.start_episode()   # raises BudgetExceeded when the lives cap is used up
        if seed is None:
            seed = int(self._rng.integers(0, 2**31 - 1))
        self._core.new_episode(seed)
        self._active, self._ep_steps = True, 0
        self._stop_writer()
        self._open_writer()
        obs, gt, screen = self._core.obs(self._calib)
        self._frame(screen)
        self._last_gt = gt
        if self._calib:
            k = ledger.totals(self._ledger_path).get(("doom", self._ledger_key, "calibration"), {"episodes": 1})["episodes"]
            d = _blackbox_dir(self.scenario, self.pilot)
            d.mkdir(parents=True, exist_ok=True)
            self._bb = gzip.open(d / f"life_{k}.jsonl.gz", "wt")
            self._bb_write({"meta": dict(scenario=self.scenario, seed=seed, purpose=self.purpose, tag=self.tag,
                                         frame_skip=self.frame_skip, action_names=self.action_names, life=k, pilot=self.pilot)})
            self._bb_write({"reset": {"obs": obs, "info": {"gt_objects": gt}}})
        return obs

    @property
    def last_gt_objects(self):
        """Calibration only: ground-truth object list of the most recent reset observation."""
        if not self._calib:
            raise PermissionError("gt_objects only exist for purpose='calibration'")
        return self._last_gt

    def step(self, action: int):
        if self._closed:
            raise RuntimeError("env is closed")
        if not self._active:
            raise RuntimeError("episode is over (or reset() not called); call reset()")
        action = int(action)
        if not 0 <= action < len(self.action_names):
            raise ValueError(f"action must be in [0, {len(self.action_names)})")
        self.meter.step()  # raises cawm.ledger.BudgetExceeded *before* the step is executed
        self._ep_steps += 1
        c = self._core
        if self._rec is not None:
            chunk = 2 if self.frame_skip % 2 == 0 else 1
            reward, frames = c.act(action, self.frame_skip, chunk)
            for fr in frames:
                self._frame(*fr)
        else:
            reward = c.act(action, self.frame_skip)
        finished = c.finished
        dead = bool(c.game.get_game_variable(_game.GV.DEAD)) if finished else False
        truncated = finished and (not dead) and c.timed_out()
        terminated = finished and not truncated
        obs, gt, screen = c.obs(self._calib)
        info = {"timeout": truncated}
        if self._calib:
            info["gt_objects"] = gt
        self._bb_write(dict(action=action, reward=reward, terminated=terminated, truncated=truncated,
                            obs=obs, info=info))
        if finished:
            self._active = False
            self.meter.episode()
            self._bb_close()
        return obs, reward, terminated, truncated, info

    def stop_recording(self) -> None:
        self._stop_writer()

    def close(self) -> None:
        if self._closed:
            return
        self._abandon()
        self._stop_writer()
        self.meter.flush()
        self._core.close()
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make(scenario: str, purpose: str, tag: str = "", seed: int | None = None, frame_skip: int = 4,
         record_path: str | None = None, ledger_path=None, record_format: str = "mp4",
         record_title: str | None = None, pilot: str | None = None) -> RealDoom:
    """Create a metered real-world environment. See the module docstring for the full manual."""
    return RealDoom(scenario, purpose, tag, seed, frame_skip, record_path, ledger_path, record_format, record_title, pilot)


def floor_plan(scenario: str) -> dict:
    """One-off static site survey: sectors with floor/ceiling heights and wall line segments."""
    spec = SCENARIOS[scenario]
    core = _game.Core(spec["cfg"], spec["actions"], 4, sectors=True)
    try:
        core.new_episode(0)
        st = core.game.get_state()
        sectors, walls, seen = [], [], set()
        for i, s in enumerate(st.sectors):
            lines = []
            for l in s.lines:
                seg = dict(x1=l.x1, y1=l.y1, x2=l.x2, y2=l.y2, blocking=bool(l.is_blocking))
                lines.append(seg)
                if l.is_blocking:
                    k = tuple(sorted([(l.x1, l.y1), (l.x2, l.y2)]))
                    if k not in seen:
                        seen.add(k)
                        walls.append(seg)
            sectors.append(dict(id=i, floor=s.floor_height, ceiling=s.ceiling_height, lines=lines))
        xs = [v for w in walls for v in (w["x1"], w["x2"])]
        ys = [v for w in walls for v in (w["y1"], w["y2"])]
        bounds = dict(xmin=min(xs), xmax=max(xs), ymin=min(ys), ymax=max(ys)) if walls else None
        return dict(scenario=scenario, sectors=sectors, walls=walls, bounds=bounds)
    finally:
        core.close()
