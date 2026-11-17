"""Lab-side GIF recording of the real scene (MuJoCo off-screen rendering, top-down camera).  Internal: never shown to
pilots.  Posts are annotated with an arrow for the required side (counter-clockwise arrow = +1), the target ring is
green, and a text strip shows title, time, load vs limit, sides satisfied and the outcome."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAX_BYTES = 6_000_000
FPS = 20.0
W, H = 720, 480
_SCALE = (H / 2) / 0.42          # px per metre for the top camera (fovy 45 deg at 1.014 m)


def _font(size=14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _px(x, y):
    return 360 + (x - 0.6) * _SCALE, 240 - y * _SCALE


_RENDERERS = {}


def _renderer(model):
    import mujoco
    key = id(model)
    if key not in _RENDERERS:
        _RENDERERS[key] = mujoco.Renderer(model, height=H, width=W)
    return _RENDERERS[key]


def frame(scene, obs, text: str) -> Image.Image:
    r = _renderer(scene.model)
    r.update_scene(scene.data, camera="top")
    im = Image.fromarray(r.render())
    d = ImageDraw.Draw(im, "RGBA")
    for (x, y), s in zip(obs["posts"], obs["post_sides"]):
        col = (40, 90, 230, 235) if s > 0 else (220, 60, 160, 235)
        R = 0.045
        phi = np.linspace(-0.75 * np.pi * s, 0.75 * np.pi * s, 30) - np.pi / 2
        pts = [_px(x + R * np.cos(f), y + R * np.sin(f)) for f in phi]
        d.line(pts, fill=col, width=3)
        f = phi[-1]
        tip = np.array(pts[-1])
        hd = np.array([-np.sin(f) * s, -np.cos(f) * s])          # world tangent (s * (-sin, cos)) with y flipped
        hd /= np.linalg.norm(hd) + 1e-9
        nn = np.array([-hd[1], hd[0]])
        d.polygon([tuple(tip + 10 * hd), tuple(tip + 6 * nn), tuple(tip - 6 * nn)], fill=col)
        cx, cy = _px(x, y)
        d.text((cx + R * _SCALE + 4, cy - R * _SCALE - 4), "+1" if s > 0 else "-1", fill=col, font=_font(13))
    tx, ty = _px(*obs["target"])
    rt = 0.02 * _SCALE
    d.ellipse([tx - rt, ty - rt, tx + rt, ty + rt], outline=(20, 160, 40, 255), width=2)
    d.rectangle([0, 0, W, 20], fill=(0, 0, 0, 170))
    d.text((4, 3), text, fill=(255, 255, 255, 255), font=_font(13))
    return im


class Recorder:
    def __init__(self, path: str, title: str = ""):
        if not str(path).endswith(".gif"):
            raise ValueError("path must end with .gif")
        self.path, self.title = str(path), title
        self.ims: list[Image.Image] = []
        self._max = 0.0

    def text(self, obs):
        ok = int(np.sum(obs["sides_now"] == obs["post_sides"]))
        self._max = max(self._max, float(obs["load"]))
        return (f"{self.title}   t={obs['t'] / 20.0:5.2f}s   load {obs['load']:5.2f}/10 N (max {self._max:4.1f})"
                f"   sides {ok}/3")

    def add(self, scene, obs) -> None:
        self.ims.append(frame(scene, obs, self.text(obs)))

    def add_image(self, im) -> None:
        self.ims.append(im)

    def close(self, outcome: str | None = None) -> None:
        if not self.ims:
            return
        ims = self.ims
        if outcome:
            last = ims[-1].copy()
            d = ImageDraw.Draw(last, "RGBA")
            d.rectangle([0, H - 28, W, H], fill=(0, 0, 0, 180))
            d.text((6, H - 24), outcome, fill=(255, 255, 255, 255), font=_font(17))
            ims = ims[:-1] + [last] * 20
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        _write(ims, self.path)
        self.ims = []


def _write(ims, path):
    step, scale, colors = 1, 1.0, 64
    while True:
        sel = ims[::step]
        if scale != 1.0:
            sel = [im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR) for im in sel]
        pal = [im.convert("P", palette=Image.ADAPTIVE, colors=colors) for im in sel]
        dur = int(round(1000 / FPS * step / 10)) * 10
        pal[0].save(path, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)
        if os.path.getsize(path) <= MAX_BYTES or step >= 8:
            return
        if colors > 32:
            colors = 32
        elif scale > 0.75 and int(W * (scale - 0.125)) >= 480:
            scale -= 0.125
        else:
            step *= 2


def render_life(npz_path: str, out_path: str, title: str | None = None) -> str:
    """Re-render a saved black-box life / observation episode from its log: the scene is rebuilt for the seed and the
    logged tracked points are written into it frame by frame (pure replay, no ledger)."""
    from reallab.station import _internal, _scene
    d = np.load(npz_path, allow_pickle=False)
    pilot = str(d["pilot"]) if "pilot" in d.files else ""
    life = int(d["life"]) if "life" in d.files else 0
    seed = int(d["seed"])
    if title is None:
        title = f"life {life}/5 - {pilot}" if life > 0 else f"observation episode {seed}"
    sc = _scene.Scene(_internal.physical())
    sc.place(d["posts"][0], d["target"][0])
    rec = Recorder(out_path, title)
    T = len(d["actions"])
    for i in range(T + 1):
        obs = {k: d[k][i] for k in ("points", "tool_pos", "posts", "post_sides", "sides_now", "target", "load", "t")}
        sc.pose_from_points(obs["points"], obs["tool_pos"])
        rec.add(sc, obs)
    out = ("SUCCESS" if bool(d["success"]) else "OVERLOAD" if bool(d["overload"]) else
           f"TIME OUT - dist {float(d['final_dist']):.3f} m, sides {int(d['sides_ok'])}/3")
    rec.close(out)
    return out_path
