"""GIF recording of the real world for human inspection (not a sensor).  Frames are the env's own rgb rendering
(goal in light green, T in grey, pusher in blue, a red cross marks the last commanded target) with a text strip."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_BYTES = 6_000_000
FPS = 10.0            # one frame per agent step (control rate is 10 Hz) -> real-time playback


def _font(size=14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def overlay(frame: np.ndarray, text: str, banner: str | None = None) -> Image.Image:
    im = Image.fromarray(np.ascontiguousarray(np.asarray(frame).astype(np.uint8)))
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, im.width, 20], fill=(0, 0, 0, 160))
    d.text((4, 2), text, fill=(255, 255, 255, 255), font=_font(14))
    if banner:
        d.rectangle([0, im.height - 24, im.width, im.height], fill=(0, 0, 0, 170))
        d.text((4, im.height - 21), banner, fill=(255, 255, 255, 255), font=_font(15))
    return im


class Recorder:
    """Collects one frame per agent step and writes a GIF (<= ~6 MB, >= 384 px when possible) on close()."""

    def __init__(self, path: str, title: str = ""):
        if not str(path).endswith(".gif"):
            raise ValueError("record_path must end with .gif")
        self.path, self.title = str(path), title
        self.frames: list[np.ndarray] = []
        self.texts: list[str] = []

    def add(self, frame, step: int, coverage: float) -> None:
        self.frames.append(np.array(frame, copy=True))
        self.texts.append(f"{self.title}  step {step}  coverage {coverage:.2f}")

    def close(self, outcome: str | None = None) -> None:
        if not self.frames:
            return
        ims = [overlay(f, t) for f, t in zip(self.frames, self.texts)]
        if outcome:
            last = overlay(self.frames[-1], self.texts[-1], banner=outcome)
            ims = ims[:-1] + [last] * 8       # hold the final frame
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._write(ims)
        self.frames, self.texts = [], []

    def _write(self, ims: list[Image.Image]) -> None:
        step, scale = 1, 1.0
        while True:
            sel = ims[::step]
            if scale != 1.0:
                sel = [im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR) for im in sel]
            pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in sel]
            dur = int(round(1000 / FPS * step / 10)) * 10
            pal[0].save(self.path, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)
            if os.path.getsize(self.path) <= MAX_BYTES or (step >= 8 and scale <= 0.75):
                return
            if scale > 0.75:
                scale -= 0.125
            else:
                step *= 2


def render_life(npz_path: str, out_path: str, title: str | None = None) -> str:
    """Re-render a saved black-box life (runs/pusht/blackbox/<pilot>/life_<k>.npz) as a GIF titled
    'life k/5 - <pilot>' (default).  Pure replay of the logged poses; no physics, no ledger."""
    from reallab.pusht._sim import SimPushT
    d = np.load(npz_path, allow_pickle=False)
    pilot = str(d["pilot"]) if "pilot" in d.files else ""
    life = int(d["life"]) if "life" in d.files else 0
    obj = str(d["object"]) if "object" in d.files else "T"
    title = title or f"life {life}/{5 if obj == 'T' else 2} - {pilot}"
    sim = SimPushT(object=obj)
    T = len(d["actions"])
    rec = Recorder(out_path, title)
    for i in range(T + 1):
        s = np.zeros(10)
        s[:2], s[2:4] = d["agent_pos"][i], d["agent_vel"][i]
        s[4:6], s[6] = d["block_pose"][i][:2], d["block_pose"][i][2]
        sim.set_full_state(s)
        sim._last_action = d["actions"][i - 1] if i > 0 else None
        rec.add(sim.render(), i, float(d["coverage"][i]))
    rec.close("SUCCESS" if bool(d["success"]) else f"timeout - max coverage {float(d['max_coverage']):.2f}")
    return out_path
