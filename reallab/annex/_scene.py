"""MuJoCo model of the annex installation (a rope on a table, one end held by a tool, one post).  Internal: pilots must
not read this module.

The rope is a 1-D elastic flexcomp (discrete elastic rod: bending + torsion, inextensible edges via stiff edge
equality constraints, refsafe disabled).  Vertex 0 is the free tip (tracked point 1, carries an extra lumped mass),
vertex N_EDGES is held by a stiff connect constraint to the tool, a ball on two slide joints (x, y) at rope height
that does not collide with anything.  The tool is driven by a force-limited PD servo (with velocity feed-forward)
towards a motion reference; the reference travels towards the commanded point with limited speed and acceleration.
The post is a mocap capsule (static geoms moved at runtime do not collide with the flex).  Viscous table drag acts on
every vertex (dof damping of the x/y slides).
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from reallab.annex import _limits as S

N_EDGES = 38                    # 39 vertices; tracked points are every second vertex
DT = 0.002
SUBSTEPS = int(round(1.0 / (S.CONTROL_HZ * DT)))   # 10
SETTLE_STEPS = 400              # 0.8 s of physics after placing the initial shape (tool holding still)
PARK = (3.0, 3.0)


def build_xml(phys: dict) -> str:
    r = S.THICK
    I = math.pi * r ** 4 / 4.0
    E = phys["bend_EI"] / I
    L = S.LENGTH
    h = L / N_EDGES
    mass = phys["mass_per_m"] * L
    return f"""
<mujoco model="annex">
  <compiler angle="radian"/>
  <option timestep="{DT}" integrator="discrete" solver="Newton" iterations="30" ls_iterations="10"
          cone="pyramidal" tolerance="1e-7"><flag refsafe="disable"/></option>
  <size memory="64M"/>
  <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="tbl" type="2d" builtin="checker" rgb1="0.82 0.80 0.74" rgb2="0.78 0.76 0.70" width="256" height="256"/>
    <material name="tbl" texture="tbl" texrepeat="16 12"/>
  </asset>
  <worldbody>
    <light pos="0.7 -0.6 2.0" dir="0 0.3 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="plane" pos="0.7 0 0" size="0 0 0.01" material="tbl"
          friction="{phys["mu_table"]} 0.005 0.0001" contype="1" conaffinity="1"/>
    <body name="postb" mocap="true" pos="{PARK[0]} {PARK[1]} {S.POST_HEIGHT / 2}"><geom name="post" type="capsule"
          size="{S.POST_RADIUS} {S.POST_HEIGHT / 2}" friction="{phys["mu_post"]} 0.005 0.0001" contype="1"
          conaffinity="1" rgba="0.55 0.55 0.6 1"/></body>
    <site name="zone" type="box" size="{(S.ZONE_HIGH[0] - S.ZONE_LOW[0]) / 2} {(S.ZONE_HIGH[1] - S.ZONE_LOW[1]) / 2} 0.0005"
          pos="{S.ZONE_CENTRE[0]} {S.ZONE_CENTRE[1]} 0.0005" rgba="0.2 0.4 0.9 0.12"/>
    <site name="target" type="cylinder" size="{S.POCKET_R} 0.001" pos="{PARK[0]} {PARK[1]} 0.001" rgba="0.1 0.7 0.2 0.5"/>
    <camera name="top" pos="0.75 0 1.6" xyaxes="1 0 0 0 1 0" projection="orthographic" fovy="1.0"/>
    <body name="tool" pos="0 0 {S.TOOL_HEIGHT}">
      <joint name="gx" type="slide" axis="1 0 0"/>
      <joint name="gy" type="slide" axis="0 1 0"/>
      <geom name="tool" type="sphere" size="0.012" mass="{S.TOOL_MASS}" contype="0" conaffinity="0"
            rgba="0.2 0.35 0.85 1"/>
    </body>
    <flexcomp name="s" type="grid" dim="1" count="{N_EDGES + 1} 1 1" spacing="{h} 1 1" pos="{L / 2} 0 {r}"
              radius="{r}" mass="{mass}" rgba="0.85 0.35 0.1 1">
      <edge equality="true" solref="0.001 1" solimp="0.99 0.9999 0.0001"/>
      <elasticity young="{E}" poisson="0.3" damping="{phys["damping"]}"/>
      <contact condim="3" friction="0.01 0.005 0.0001" selfcollide="none" contype="1" conaffinity="1"
               solref="0.004 1"/>
    </flexcomp>
  </worldbody>
  <equality>
    <connect name="hold" solref="0.001 1" solimp="0.99 0.9999 0.0001" body1="s_{N_EDGES}" body2="tool" anchor="0 0 0"/>
  </equality>
  <actuator>
    <motor name="ax" joint="gx" ctrlrange="-{S.TOOL_FMAX} {S.TOOL_FMAX}" ctrllimited="true"/>
    <motor name="ay" joint="gy" ctrlrange="-{S.TOOL_FMAX} {S.TOOL_FMAX}" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def _fill(pts20: np.ndarray) -> np.ndarray:
    """Vertices from the 20 tracked points (every second vertex): each in-between vertex is put at the apex of the
    isosceles triangle with legs h over its two neighbours (edge lengths preserved), bulging away from the local
    centre of curvature (in x, y); z is averaged."""
    h = S.LENGTH / N_EDGES
    P = np.zeros((N_EDGES + 1, 3))
    P[0::2] = pts20
    n = len(pts20)
    for j in range(n - 1):
        a, b = pts20[j], pts20[j + 1]
        c = b[:2] - a[:2]
        L = float(np.hypot(*c))
        mid = 0.5 * (a + b)
        if L < 1e-9 or L >= 2 * h:
            P[2 * j + 1] = mid
            continue
        off = np.sqrt(max(h * h - (L / 2) ** 2, 0.0))
        nrm = np.array([-c[1], c[0]]) / L                     # left of the travel direction
        turn = 0.0                                            # > 0: the curve turns left here
        if j > 0:
            d0 = a[:2] - pts20[j - 1][:2]
            turn += d0[0] * c[1] - d0[1] * c[0]
        if j + 2 < n:
            d2 = pts20[j + 2][:2] - b[:2]
            turn += c[0] * d2[1] - c[1] * d2[0]
        sgn = -1.0 if turn > 0 else 1.0                       # bulge to the outside of the turn
        P[2 * j + 1] = mid
        P[2 * j + 1, :2] = mid[:2] + sgn * off * nrm
    return P


class Scene:
    def __init__(self, phys: dict, vmax: float = S.TOOL_VMAX, amax: float = S.TOOL_AMAX):
        self.phys = dict(phys)
        self.vmax, self.amax = float(vmax), float(amax)
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.phys))
        m = self.model
        self.data = mujoco.MjData(m)
        self.vbody = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"s_{i}") for i in range(N_EDGES + 1)])
        self.vqadr = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self.vbody])
        self.vdadr = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in self.vbody])
        self.vpos0 = m.body_pos[self.vbody].copy()
        self.post_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "postb")]
        self.target_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.gq = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ("gx", "gy")]
        self.gd = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ("gx", "gy")]
        h = S.LENGTH / N_EDGES
        for i in range(N_EDGES + 1):
            m.dof_damping[self.vdadr[i]:self.vdadr[i] + 2] = self.phys["drag"] * h
        m.body_mass[self.vbody[0]] += self.phys["tip_mass"]
        mujoco.mj_setConst(m, self.data)    # NB: recomputes connect anchors from qpos0, so set them afterwards
        hold = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "hold")
        m.eq_data[hold, 3:6] = (0.0, 0.0, 0.0)
        self.ref = np.zeros(2)          # motion reference of the tool and its velocity
        self.rv = np.zeros(2)
        self.cmd = np.zeros(2)
        self.n_edges = N_EDGES
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._buf = np.zeros(mujoco.mj_stateSize(m, self._spec))

    # ---------------------------------------------------------------- readings
    def vertices(self) -> np.ndarray:
        return self.data.xpos[self.vbody].copy()

    def tracked_points(self) -> np.ndarray:
        return self.vertices()[0::2].copy()

    def tool_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[self.gq[0]], self.data.qpos[self.gq[1]]])

    def tool_vel(self) -> np.ndarray:
        return np.array([self.data.qvel[self.gd[0]], self.data.qvel[self.gd[1]]])

    # ---------------------------------------------------------------- state save / restore (lab search)
    def get_state(self):
        mujoco.mj_getState(self.model, self.data, self._buf, self._spec)
        return (self._buf.copy(), self.ref.copy(), self.rv.copy(), self.cmd.copy())

    def set_state(self, st) -> None:
        buf, ref, rv, cmd = st
        mujoco.mj_setState(self.model, self.data, buf, self._spec)
        self.ref, self.rv, self.cmd = ref.copy(), rv.copy(), cmd.copy()
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- episode
    def reset(self, shape: np.ndarray, post=None, target=None):
        """shape: (N_EDGES+1, 2) xy of the vertices (first = free tip, last = at the tool).  The post is parked off
        the table while the body relaxes for SETTLE_STEPS (tool holding still); then place(post, target)."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.mocap_pos[self.post_mocap, :2] = PARK
        for i in range(N_EDGES + 1):
            p = np.array([shape[i, 0], shape[i, 1], S.THICK])
            d.qpos[self.vqadr[i]:self.vqadr[i] + 3] = p - self.vpos0[i]
        end = np.asarray(shape[-1], float)
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = end
        self.ref, self.rv, self.cmd = end.copy(), np.zeros(2), end.copy()
        mujoco.mj_forward(m, d)
        for _ in range(SETTLE_STEPS):
            self._servo()
            mujoco.mj_step(m, d)
        if post is not None:
            self.place(post, target)

    def place(self, post, target=None):
        m, d = self.model, self.data
        d.mocap_pos[self.post_mocap, :2] = post
        m.site_pos[self.target_sid, :2] = PARK if target is None else target
        mujoco.mj_forward(m, d)

    def _servo(self):
        d = self.data
        x, v = self.tool_pos(), self.tool_vel()
        f = S.TOOL_KP * (self.ref - x) + S.TOOL_KV * (self.rv - v)
        d.ctrl[:] = np.clip(f, -S.TOOL_FMAX, S.TOOL_FMAX)

    def _advance_ref(self):
        delta = self.cmd - self.ref
        dist = float(np.hypot(delta[0], delta[1]))
        if dist < 1e-9:
            vdes = np.zeros(2)
        else:
            sp = min(self.vmax, math.sqrt(2.0 * self.amax * dist), dist / DT)
            vdes = delta * (sp / dist)
        dv = vdes - self.rv
        n = float(np.hypot(dv[0], dv[1]))
        lim = self.amax * DT
        if n > lim:
            dv *= lim / n
        self.rv = self.rv + dv
        self.ref = self.ref + self.rv * DT

    def step(self, target: np.ndarray) -> None:
        """One control period with the tool commanded to `target` (already clipped to the zone)."""
        m, d = self.model, self.data
        self.cmd = np.asarray(target, float).copy()
        for _ in range(SUBSTEPS):
            self._advance_ref()
            self._servo()
            mujoco.mj_step(m, d)

    def set_from_reading(self, pts20, prev_pts20, tool, prev_tool, hz: float) -> None:
        """Lab: build a simulator state from what a pilot observes - the 20 tracked points and the tool position of the
        current and the previous reading (velocities by finite differences, intermediate vertices interpolated,
        torsion at rest, tool motion reference = tool reading).  The post / target placement is kept."""
        m, d = self.model, self.data
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0.0
        d.qacc_warmstart[:] = 0.0
        P = _fill(np.asarray(pts20, float))
        Q = _fill(np.asarray(prev_pts20, float))
        V = (P - Q) * hz
        for i in range(N_EDGES + 1):
            d.qpos[self.vqadr[i]:self.vqadr[i] + 3] = P[i] - self.vpos0[i]
            d.qvel[self.vdadr[i]:self.vdadr[i] + 3] = V[i]
        tv = (np.asarray(tool) - np.asarray(prev_tool)) * hz
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = tool
        d.qvel[self.gd[0]], d.qvel[self.gd[1]] = tv
        self.ref, self.rv, self.cmd = np.array(tool, float), tv.copy(), np.array(tool, float)
        mujoco.mj_forward(m, d)

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
