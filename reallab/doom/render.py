"""Screen recording for human inspection (GIF / MP4) with a small text overlay. Not a sensor."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_BYTES = 6_000_000
GAME_FPS = 35.0


def _font(size=12):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def overlay(frame: np.ndarray, text: str) -> Image.Image:
    """frame: HxWx3 uint8 (or CxHxW) -> PIL image with a one-line text strip at the top."""
    f = np.asarray(frame)
    if f.ndim == 3 and f.shape[0] == 3:
        f = np.transpose(f, (1, 2, 0))
    if f.ndim == 2:
        f = np.stack([f] * 3, -1)
    im = Image.fromarray(np.ascontiguousarray(f.astype(np.uint8)))
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0, 150))
    d.text((3, 1), text, fill=(255, 255, 255, 255), font=_font(12))
    return im


class Recorder:
    """Collects frames (one every `frame_tics` game tics) and writes gif or mp4 on close()."""

    def __init__(self, path: str, fmt: str = "mp4", title: str = "", frame_tics: int = 2):
        if fmt not in ("mp4", "gif"):
            raise ValueError("record_format must be 'mp4' or 'gif'")
        self.path, self.fmt, self.title, self.frame_tics = str(path), fmt, title, frame_tics
        self.frames: list[Image.Image] = []

    def add(self, screen, tic: int, health: float, kills: int) -> None:
        self.frames.append(overlay(screen, f"{self.title}  tic {tic}  hp {int(health)}  kills {kills}"))

    def close(self) -> None:
        if not self.frames:
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        fps = GAME_FPS / self.frame_tics
        if self.fmt == "mp4":
            import imageio
            with imageio.get_writer(self.path, fps=fps) as w:
                for im in self.frames:
                    w.append_data(np.asarray(im))
        else:
            self._write_gif(fps)
        self.frames = []

    def _write_gif(self, fps: float) -> None:
        frames, step, scale = self.frames, 1, 1.0
        while True:
            sel = frames[::step]
            if scale != 1.0:
                sel = [im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR) for im in sel]
            pal = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in sel]
            dur = int(round(1000 / fps * step / 10)) * 10      # GIF delays are multiples of 10 ms
            pal[0].save(self.path, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)
            if os.path.getsize(self.path) <= MAX_BYTES or (step >= 8 and scale <= 0.5):
                return
            if scale > 0.5:
                scale -= 0.25
            else:
                step *= 2
