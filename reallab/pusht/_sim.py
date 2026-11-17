"""Lab-internal wrapper around gym-pusht's PushTEnv (pymunk).  Pilots must not read this module."""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
from gym_pusht.envs import PushTEnv  # noqa: E402

from reallab.pusht._deviations import DEVIATIONS, DEVIATIONS_L, NOMINAL, NOMINAL_L  # noqa: E402


class SimPushT(PushTEnv):
    """PushTEnv with (optionally) the hidden deviations applied, plus full-state get/set for lab-side use."""

    def __init__(self, deviations: dict | None = None, vis_size: int = 384, object: str = "T"):
        if object not in ("T", "L"):
            raise ValueError("object must be 'T' or 'L'")
        self.object = object
        if deviations is None:      # lab switch for the "deviations off" reference run
            nodev = os.environ.get("CAWM_PUSHT_NODEV")
            deviations = (NOMINAL if nodev else DEVIATIONS) if object == "T" else (NOMINAL_L if nodev else DEVIATIONS_L)
        dev = deviations
        self.deviations = dict(dev)
        super().__init__(obs_type="state", render_mode="rgb_array", block_cog=dev.get("block_cog"), damping=dev.get("damping"),
                         visualization_width=vis_size, visualization_height=vis_size)
        self.k_p, self.k_v = float(dev.get("k_p", 100.0)), float(dev.get("k_v", 20.0))
        self.reset(seed=0)

    @staticmethod
    def add_ell(space, position, angle, mask=None):
        """The L block (see _geometry.py); mass 1."""
        import pymunk
        import pygame
        if mask is None:
            mask = pymunk.ShapeFilter.ALL_MASKS()
        from reallab.pusht._geometry import L_FOOT, L_LEG
        v1, v2 = list(L_LEG), list(L_FOOT)
        body = pymunk.Body(1, pymunk.moment_for_poly(1, vertices=v1) + pymunk.moment_for_poly(1, vertices=v2))
        s1, s2 = pymunk.Poly(body, v1), pymunk.Poly(body, v2)
        for sh in (s1, s2):
            sh.color = pygame.Color("LightSlateGray")
            sh.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
        body.angle = angle
        body.position = position
        body.friction = 1
        space.add(body, s1, s2)
        return body, [s1, s2]

    def _setup(self):
        if self.object == "T":
            super()._setup()
        else:
            import pymunk
            self.space = pymunk.Space()
            self.space.gravity = 0, 0
            self.space.damping = self.damping if self.damping is not None else 0.0
            self.teleop = False
            self.space.add(self.add_segment(self.space, (5, 506), (5, 5), 2), self.add_segment(self.space, (5, 5), (506, 5), 2),
                           self.add_segment(self.space, (506, 5), (506, 506), 2), self.add_segment(self.space, (5, 506), (506, 506), 2))
            self.agent = self.add_circle(self.space, (256, 400), 15)
            self.block, self._block_shapes = self.add_ell(self.space, (256, 300), 0)
            from reallab.pusht._geometry import L_GOAL
            self.goal_pose = np.array(L_GOAL)
            if self.block_cog is not None:
                self.block.center_of_gravity = self.block_cog
            self.collision_handeler = self.space.add_collision_handler(0, 0)
            self.collision_handeler.post_solve = self._handle_collision
            self.n_contact_points = 0
        f = self.deviations.get("friction")
        if f is not None:
            for sh in (*self.block.shapes, *self.agent.shapes):
                sh.friction = float(f)

    # full physical state (exact, includes velocities) ------------------------------------------------------
    def get_full_state(self) -> np.ndarray:
        a, b = self.agent, self.block
        return np.array([*a.position, *a.velocity, *b.position, b.angle, *b.velocity, b.angular_velocity])

    def set_full_state(self, s) -> None:
        a, b = self.agent, self.block
        a.position = (float(s[0]), float(s[1]))
        a.velocity = (float(s[2]), float(s[3]))
        b.angle = float(s[6])               # angle first: it rotates about the centre of gravity
        b.position = (float(s[4]), float(s[5]))
        b.velocity = (float(s[7]), float(s[8]))
        b.angular_velocity = float(s[9])

    @staticmethod
    def wrap(theta: float) -> float:
        return float(theta % (2 * np.pi))
