"""Internal VizDoom wrapper used by api.py. Not part of the documented interface."""
from __future__ import annotations

import os

import numpy as np
import vizdoom as vzd

GV = vzd.GameVariable
N_SCAN = 64
RES_W, RES_H = 320, 240
PLAYER_NAME = "DoomPlayer"

# game variables we always enable (independent of what the cfg lists)
_VARS = [GV.POSITION_X, GV.POSITION_Y, GV.POSITION_Z, GV.ANGLE, GV.PITCH,
         GV.VELOCITY_X, GV.VELOCITY_Y, GV.VELOCITY_Z,
         GV.HEALTH, GV.ARMOR, GV.SELECTED_WEAPON, GV.SELECTED_WEAPON_AMMO, GV.ATTACK_READY,
         GV.KILLCOUNT, GV.ITEMCOUNT, GV.DEAD, GV.ON_GROUND,
         GV.DAMAGE_TAKEN, GV.HITS_TAKEN, GV.DAMAGECOUNT, GV.HITCOUNT]


def player_dict(game: vzd.DoomGame) -> dict:
    g = game.get_game_variable
    return dict(
        x=g(GV.POSITION_X), y=g(GV.POSITION_Y), z=g(GV.POSITION_Z), angle=g(GV.ANGLE), pitch=g(GV.PITCH),
        vx=g(GV.VELOCITY_X), vy=g(GV.VELOCITY_Y), vz=g(GV.VELOCITY_Z),
        health=g(GV.HEALTH), armor=g(GV.ARMOR),
        selected_weapon=int(g(GV.SELECTED_WEAPON)), ammo=g(GV.SELECTED_WEAPON_AMMO),
        attack_ready=bool(g(GV.ATTACK_READY)),
        killcount=int(g(GV.KILLCOUNT)), itemcount=int(g(GV.ITEMCOUNT)),
        dead=bool(g(GV.DEAD)), on_ground=bool(g(GV.ON_GROUND)),
        damage_taken=g(GV.DAMAGE_TAKEN), hits_taken=int(g(GV.HITS_TAKEN)),
        damage_dealt=g(GV.DAMAGECOUNT), hits_dealt=int(g(GV.HITCOUNT)),
    )


def _obj(o, box=None) -> dict:
    d = dict(id=int(o.id), name=o.name, x=o.position_x, y=o.position_y, z=o.position_z, angle=o.angle,
             vx=o.velocity_x, vy=o.velocity_y, vz=o.velocity_z)
    if box is None:
        d.update(sx=None, sy=None, sw=None, sh=None)
    else:
        d.update(sx=int(box.x), sy=int(box.y), sw=int(box.width), sh=int(box.height))
    return d


class Core:
    """One DoomGame instance configured for structured-sensor output."""

    def __init__(self, cfg: str, buttons_combo_spec: list, frame_skip: int, rgb: bool = False,
                 sectors: bool = False):
        self.game = g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, cfg))
        g.set_window_visible(False)
        g.set_console_enabled(False)
        g.set_sound_enabled(False)
        g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        g.set_screen_format(vzd.ScreenFormat.RGB24 if rgb else vzd.ScreenFormat.GRAY8)
        g.set_labels_buffer_enabled(True)
        g.set_depth_buffer_enabled(True)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(sectors)
        g.set_mode(vzd.Mode.PLAYER)
        g.clear_available_game_variables()
        for v in _VARS:
            g.add_available_game_variable(v)
        g.init()
        self.buttons = [b.name for b in g.get_available_buttons()]
        self.timeout = g.get_episode_timeout()
        self.start_time = g.get_episode_start_time()
        self.n_buttons = len(self.buttons)
        self.frame_skip = frame_skip
        self.last_tic = -1
        # discrete action table -> button vectors
        self.table = [[0] * self.n_buttons]
        for combo in buttons_combo_spec:
            vec = [0] * self.n_buttons
            for b in combo:
                vec[self.buttons.index(b)] = 1
            self.table.append(vec)

    def new_episode(self, seed: int) -> None:
        self.game.set_seed(int(seed))
        self.game.new_episode()
        self.last_tic = -1

    def act(self, a: int, frame_skip: int, chunk: int | None = None):
        """chunk=None: returns reward. chunk=k: advances in k-tic pieces and returns (reward, frames) where
        frames = [(screen, tic, health, kills)] after each piece (same engine tics as one make_action)."""
        if chunk is None:
            return float(self.game.make_action(self.table[a], frame_skip))
        g, r, left, frames = self.game, 0.0, frame_skip, []
        while left > 0 and not g.is_episode_finished():
            n = min(chunk, left)
            r += g.make_action(self.table[a], n)
            left -= n
            st = g.get_state()
            if st is not None:
                frames.append((st.screen_buffer, self.episode_time(), g.get_game_variable(GV.HEALTH),
                               int(g.get_game_variable(GV.KILLCOUNT))))
        return float(r), frames

    @property
    def finished(self) -> bool:
        return self.game.is_episode_finished()

    def episode_time(self) -> int:
        return int(self.game.get_episode_time())

    def timed_out(self) -> bool:
        return self.timeout > 0 and self.episode_time() >= self.timeout

    def obs(self, want_gt: bool):
        """Returns (obs, gt_objects_or_None, screen_or_None)."""
        g = self.game
        st = g.get_state()
        t = self.episode_time()
        if st is None and t <= self.last_tic:   # engine reports 0 after some terminal events: upper bound
            t = self.last_tic + (self.frame_skip if self.last_tic >= 0 else 0)
        self.last_tic = t
        obs = dict(tic=t, player=player_dict(g))
        if st is None:  # episode finished: no sensors rendered any more
            obs["objects"] = []
            obs["depth_scan"] = np.zeros(N_SCAN, dtype=np.float32)
            return obs, ([] if want_gt else None), None
        by_id = {o.id: o for o in st.objects}
        boxes = {}
        for lab in st.labels:
            if lab.object_name == PLAYER_NAME:
                continue
            boxes[lab.object_id] = lab
        vis = [_obj(by_id[i], b) for i, b in sorted(boxes.items()) if i in by_id]
        obs["objects"] = vis
        row = st.depth_buffer[RES_H // 2].astype(np.float32)
        obs["depth_scan"] = row.reshape(N_SCAN, -1).mean(axis=1).astype(np.float32)
        gt = None
        if want_gt:
            gt = [_obj(o, boxes.get(o.id)) for o in sorted(st.objects, key=lambda o: o.id) if o.name != PLAYER_NAME]
            for d in gt:
                d["visible"] = d["sx"] is not None
        return obs, gt, st.screen_buffer

    def close(self):
        self.game.close()
