"""MuJoCo model of the cable rig.  Internal: pilots must not read this module.

The cable is a 1-D elastic flexcomp (discrete elastic rod, MuJoCo >= 4: bending + torsion, inextensible edges via edge
equality constraints) lying on a table plane.  Its first two points are held by connect constraints to fixed world
points (the clamp); the anchor force is the net constraint force of the clamp.  The last point is held by a connect
constraint to the gripper, a ball on two position-controlled slide joints (x, y) at a fixed height.  Three cylinders
are the pegs; their positions are written into the model at every reset.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from reallab.station import _limits as S

N_EDGES = 38                    # 39 points; markers are every second point
DT = 0.002
SUBSTEPS = int(round(1.0 / (S.CONTROL_HZ * DT)))   # 25
SETTLE_STEPS = 500              # 1.0 s of physics after placing the initial shape (gripper holding still)
KP, KV = 3000.0, 110.0          # gripper position servo
GRIP_MASS = 0.4
_EQ = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
LOADCELL = os.environ.get("CAWM_STATION_LOADCELL", "max")   # lab tuning switch; the rig uses "max"


def build_xml(phys: dict) -> str:
    r = S.CABLE_RADIUS
    I = math.pi * r ** 4 / 4.0
    E = phys["bend_EI"] / I
    L = S.CABLE_LENGTH
    h = L / N_EDGES
    mass = phys["mass_per_m"] * L
    pegs = "\n".join(
        f'<body name="post{i}b" mocap="true" pos="{0.3 + 0.2 * i} 5.0 {S.PEG_HEIGHT / 2}"><geom name="post{i}" '
        f'type="capsule" size="{S.PEG_RADIUS} {S.PEG_HEIGHT / 2}" friction="{phys["mu_peg"]} 0.005 0.0001" '
        f'contype="3" conaffinity="3" rgba="0.55 0.55 0.6 1"/></body>'
        for i in range(3))
    return f"""
<mujoco model="station">
  <compiler angle="radian"/>
  <option timestep="{DT}" integrator="discrete" solver="Newton" iterations="50" ls_iterations="20" cone="pyramidal"
          tolerance="1e-8"><flag refsafe="disable"/></option>
  <size memory="64M"/>
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="tbl" type="2d" builtin="checker" rgb1="0.82 0.80 0.74" rgb2="0.78 0.76 0.70" width="256" height="256"/>
    <material name="tbl" texture="tbl" texrepeat="12 8"/>
  </asset>
  <worldbody>
    <light pos="0.6 -0.6 2.0" dir="0 0.3 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="plane" pos="0.6 0 0" size="0.6 0.4 0.01" material="tbl"
          friction="{phys["mu_table"]} 0.005 0.0001" contype="1" conaffinity="1"/>
    {pegs}
    <site name="target" type="cylinder" size="{S.CLIP_TOL} 0.001" pos="0.9 0 0.001" rgba="0.1 0.7 0.2 0.6"/>
    <geom name="fix" type="box" size="0.02 0.02 0.012" pos="{S.ANCHOR[0] - 0.02} {S.ANCHOR[1]} 0.012"
          contype="0" conaffinity="0" rgba="0.25 0.25 0.3 1"/>
    <camera name="top" pos="0.6 0 1.0" xyaxes="1 0 0 0 1 0" projection="orthographic" fovy="0.84"/>
    <body name="tool" pos="0 0 {S.GRIP_HEIGHT}">
      <joint name="gx" type="slide" axis="1 0 0" damping="1"/>
      <joint name="gy" type="slide" axis="0 1 0" damping="1"/>
      <geom name="tool" type="sphere" size="{S.GRIP_RADIUS}" mass="{GRIP_MASS}" contype="2" conaffinity="2"
            rgba="0.2 0.35 0.85 1"/>
    </body>
    <flexcomp name="s" type="grid" dim="1" count="{N_EDGES + 1} 1 1" spacing="{h} 1 1" pos="{L / 2} 0 {r}"
              radius="{r}" mass="{mass}" rgba="0.85 0.35 0.1 1">
      <edge equality="true" solref="0.001 1" solimp="0.99 0.9999 0.0001"/>
      <elasticity young="{E}" poisson="0.3" damping="0.002"/>
      <contact condim="3" friction="0.01 0.005 0.0001" selfcollide="none" contype="1" conaffinity="1"
               solref="0.004 1"/>
    </flexcomp>
  </worldbody>
  <equality>
    <connect name="fix0" solref="0.001 1" solimp="0.99 0.9999 0.0001" body1="s_0" body2="world" anchor="0 0 0"/>
    <connect name="fix1" solref="0.001 1" solimp="0.99 0.9999 0.0001" body1="s_1" body2="world" anchor="0 0 0"/>
    <connect name="hold" solref="0.001 1" solimp="0.99 0.9999 0.0001" body1="s_{N_EDGES}" body2="tool" anchor="0 0 0"/>
  </equality>
  <actuator>
    <position name="ax" joint="gx" kp="{KP}" kv="{KV}" forcerange="-{S.GRIP_FMAX} {S.GRIP_FMAX}" forcelimited="true"/>
    <position name="ay" joint="gy" kp="{KP}" kv="{KV}" forcerange="-{S.GRIP_FMAX} {S.GRIP_FMAX}" forcelimited="true"/>
  </actuator>
</mujoco>
"""


class Scene:
    def __init__(self, phys: dict):
        self.phys = dict(phys)
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.phys))
        m = self.model
        self.data = mujoco.MjData(m)
        self.vbody = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"s_{i}") for i in range(N_EDGES + 1)])
        self.vqadr = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self.vbody])
        self.vdadr = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in self.vbody])
        self.vpos0 = m.body_pos[self.vbody].copy()      # parent = world
        self.peg_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"post{i}") for i in range(3)]
        self.peg_mocap = [m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"post{i}b")] for i in range(3)]
        self.clip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.gq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ("gx", "gy")]
        self.gd = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ("gx", "gy")]
        self.eq = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n) for n in ("fix0", "fix1", "hold")}
        # connect targets: anchor point and clamp point in the world; the cable end hangs below the gripper centre
        h = S.CABLE_LENGTH / N_EDGES
        r = S.CABLE_RADIUS
        m.eq_data[self.eq["fix0"], 3:6] = (S.ANCHOR[0], S.ANCHOR[1], r)
        m.eq_data[self.eq["fix1"], 3:6] = (S.ANCHOR[0] + h * S.CLAMP_DIR[0], S.ANCHOR[1] + h * S.CLAMP_DIR[1], r)
        m.eq_data[self.eq["hold"], 3:6] = (0.0, 0.0, r - S.GRIP_HEIGHT)
        # viscous drag of the cable on the table (per unit length), as damping of the vertex x/y slide joints
        for i in range(N_EDGES + 1):
            m.dof_damping[self.vdadr[i]:self.vdadr[i] + 2] = self.phys.get("drag", 0.0) * h
        self.ref = np.zeros(2)          # rate-limited motion target of the gripper
        self.cmd = np.zeros(2)
        self.n_edges = N_EDGES
        self.last_load = 0.0

    def tracked_points(self, n: int) -> np.ndarray:
        """n points evenly spaced in rest arc length (the vertices are equally spaced; n-1 must divide N_EDGES or we
        interpolate linearly between vertices)."""
        P = self.points()
        u = np.linspace(0.0, N_EDGES, n)
        i0 = np.minimum(np.floor(u).astype(int), N_EDGES - 1)
        f = (u - i0)[:, None]
        return (1 - f) * P[i0] + f * P[i0 + 1]

    def tool_pos(self):
        return self.gripper_pos()

    def tool_vel(self):
        return self.gripper_vel()

    def pose_from_points(self, pts20: np.ndarray, tool: np.ndarray) -> None:
        """Replay helper (rendering only): place the vertices on the 20 tracked points (midpoints in between)."""
        m, d = self.model, self.data
        P = np.zeros((N_EDGES + 1, 3))
        P[0::2] = pts20
        P[1::2] = 0.5 * (pts20[:-1] + pts20[1:])
        for i in range(N_EDGES + 1):
            d.qpos[self.vqadr[i]:self.vqadr[i] + 3] = P[i] - self.vpos0[i]
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = tool
        mujoco.mj_forward(m, d)

    def load_now(self) -> float:
        return self.load_force()

    # ---------------------------------------------------------------- state
    def points(self) -> np.ndarray:
        return self.data.xpos[self.vbody].copy()

    def gripper_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[self.gq[0]], self.data.qpos[self.gq[1]]])

    def gripper_vel(self) -> np.ndarray:
        return np.array([self.data.qvel[self.gd[0]], self.data.qvel[self.gd[1]]])

    def anchor_force(self) -> float:
        """Net force of the cable on the clamp (both clamp constraints), N."""
        d = self.data
        ne = d.ne
        typ, ids = d.efc_type[:ne], d.efc_id[:ne]
        rows = np.nonzero((typ == _EQ) & ((ids == self.eq["fix0"]) | (ids == self.eq["fix1"])))[0]
        if len(rows) != 6:
            return 0.0
        f = d.efc_force[rows].reshape(2, 3).sum(axis=0)   # connect rows come in (x, y, z) triples
        return float(np.sqrt(f @ f))

    def tool_force(self) -> float:
        """Horizontal force of the cable on the tool (the connect constraint at the cable end), N."""
        d = self.data
        ne = d.ne
        rows = np.nonzero((d.efc_type[:ne] == _EQ) & (d.efc_id[:ne] == self.eq["hold"]))[0]
        if len(rows) != 3:
            return 0.0
        f = d.efc_force[rows[:2]]
        return float(np.sqrt(f @ f))

    def load_force(self) -> float:
        if LOADCELL == "tool":
            return self.tool_force()
        if LOADCELL == "anchor":
            return self.anchor_force()
        return max(self.anchor_force(), self.tool_force())

    # ---------------------------------------------------------------- episode
    def reset(self, shape: np.ndarray, pegs: np.ndarray | None = None, clip: np.ndarray | None = None):
        """shape: (N_EDGES+1, 2) xy of the cable points (first = anchor).  The pegs are parked off the table while the
        cable relaxes for SETTLE_STEPS (gripper holding the end still); then place(pegs, clip) if given."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for i, k in enumerate(self.peg_mocap):
            d.mocap_pos[k, :2] = (0.3 + 0.2 * i, 5.0)
        for i in range(N_EDGES + 1):
            p = np.array([shape[i, 0], shape[i, 1], S.CABLE_RADIUS])
            d.qpos[self.vqadr[i]:self.vqadr[i] + 3] = p - self.vpos0[i]
        end = shape[-1]
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = end
        self.ref = end.copy()
        self.cmd = end.copy()
        d.ctrl[:] = end
        mujoco.mj_forward(m, d)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(m, d)
        if pegs is not None:
            self.place(pegs, clip)

    def place(self, pegs, clip):
        m, d = self.model, self.data
        for i, k in enumerate(self.peg_mocap):
            d.mocap_pos[k, :2] = pegs[i]
        m.site_pos[self.clip_sid, :2] = clip
        mujoco.mj_forward(m, d)

    def step(self, target: np.ndarray) -> float:
        """One control period with the gripper commanded to `target`.  Returns the mean anchor force (the load)."""
        m, d = self.model, self.data
        self.cmd = np.asarray(target, float).copy()
        vstep = S.GRIP_VMAX * DT
        acc, mx = 0.0, 0.0
        for _ in range(SUBSTEPS):
            delta = self.cmd - self.ref
            n = float(np.linalg.norm(delta))
            self.ref = self.cmd.copy() if n <= vstep else self.ref + delta * (vstep / n)
            d.ctrl[:] = self.ref
            mujoco.mj_step(m, d)
            f = self.load_force()
            acc += f
            mx = max(mx, f)
        self.last_load = acc / SUBSTEPS
        self.last_peak = mx
        return self.last_load
