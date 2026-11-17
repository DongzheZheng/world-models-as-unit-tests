"""Lab-side top-down rendering (PIL drawing of the logged readings; no simulator needed).  Internal: never shown to
pilots.  The tool zone is a pale blue rectangle, the post grey with an arrow for the required side (counter-clockwise
arrow = +1), the target (if any) a green ring, the tracked points an orange polyline (point 1 = darker dot), the tool
blue.  A text strip shows the title, time, the signed winding (side * W) vs the requirement and the outcome.

render_log(npz_path, out_gif, title)   re-renders any black-box life or evaluation step log (parallel.py log_dir)
Recorder                              frame-by-frame recorder used by api.py's lab hook
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reallab.annex import _limits as S

MAX_BYTES = 6_000_000
FPS = 25.0                       # every second agent step is drawn (50 Hz readings)
W, H = 720, 560
VIEW = dict(x=(-0.05, 1.25), y=(-0.50, 0.50))
_SCALE = min(W / (VIEW["x"][1] - VIEW["x"][0]), (H - 40) / (VIEW["y"][1] - VIEW["y"][0]))


def _font(size=14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _px(x, y):
    return (x - VIEW["x"][0]) * _SCALE, 20 + (VIEW["y"][1] - y) * _SCALE


def _pts_obs(o):
    return o["points"], o["tool_pos"], o["post"], int(o.get("post_side", 0)), o.get("target")


def frame(o: dict, text: str, bottom: str | None = None) -> Image.Image:
    pts, tool, post, side, target = _pts_obs(o)
    im = Image.new("RGB", (W, H), (226, 222, 210))
    d = ImageDraw.Draw(im, "RGBA")
    for gx in np.arange(0.0, 1.3, 0.1):
        d.line([_px(gx, VIEW["y"][0]), _px(gx, VIEW["y"][1])], fill=(210, 206, 194), width=1)
    for gy in np.arange(-0.5, 0.51, 0.1):
        d.line([_px(VIEW["x"][0], gy), _px(VIEW["x"][1], gy)], fill=(210, 206, 194), width=1)
    z0, z1 = _px(S.ZONE_LOW[0], S.ZONE_HIGH[1]), _px(S.ZONE_HIGH[0], S.ZONE_LOW[1])
    d.rectangle([z0, z1], fill=(60, 110, 230, 40), outline=(60, 110, 230, 160), width=2)
    if target is not None and np.all(np.isfinite(target)):
        tx, ty = _px(*target)
        rt = S.POCKET_R * _SCALE
        d.ellipse([tx - rt, ty - rt, tx + rt, ty + rt], outline=(20, 160, 40, 255), width=3)
    cx, cy = _px(*post)
    rp = S.POST_RADIUS * _SCALE
    d.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], fill=(130, 130, 140, 255))
    col = (40, 90, 230, 235) if side > 0 else (220, 60, 160, 235)
    R = 0.055
    if side == 0:
        P = [_px(p[0], p[1]) for p in pts]
        return _finish(im, d, P, tool, text, bottom)
    phi = np.linspace(-0.75 * np.pi * side, 0.75 * np.pi * side, 30) - np.pi / 2
    arc = [_px(post[0] + R * np.cos(f), post[1] + R * np.sin(f)) for f in phi]
    d.line(arc, fill=col, width=3)
    tip = np.array(arc[-1])
    hd = np.array(arc[-1]) - np.array(arc[-3])
    hd /= np.linalg.norm(hd) + 1e-9
    nn = np.array([-hd[1], hd[0]])
    d.polygon([tuple(tip + 10 * hd), tuple(tip + 6 * nn), tuple(tip - 6 * nn)], fill=col)
    d.text((cx + R * _SCALE + 4, cy - R * _SCALE - 4), "+1" if side > 0 else "-1", fill=col, font=_font(13))
    P = [_px(p[0], p[1]) for p in pts]
    return _finish(im, d, P, tool, text, bottom)


def _finish(im, d, P, tool, text, bottom):
    d.line(P, fill=(215, 90, 25, 255), width=4, joint="curve")
    x, y = P[0]
    d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(140, 40, 10, 255))
    tx, ty = _px(*tool)
    d.ellipse([tx - 7, ty - 7, tx + 7, ty + 7], fill=(40, 70, 200, 255))
    d.rectangle([0, 0, W, 20], fill=(0, 0, 0, 170))
    d.text((4, 3), text, fill=(255, 255, 255, 255), font=_font(13))
    if bottom:
        d.rectangle([0, H - 28, W, H], fill=(0, 0, 0, 180))
        d.text((6, H - 24), bottom, fill=(255, 255, 255, 255), font=_font(17))
    return im


def _winding(pts, post):
    a = np.arctan2(pts[:, 1] - post[1], pts[:, 0] - post[0])
    return float(((np.diff(a) + np.pi) % (2 * np.pi) - np.pi).sum())


class Recorder:
    def __init__(self, path: str, title: str = "", every: int = 2):
        if not str(path).endswith(".gif"):
            raise ValueError("path must end with .gif")
        self.path, self.title, self.every = str(path), title, every
        self.ims: list[Image.Image] = []
        self._last = None

    def text(self, o):
        d = float(np.linalg.norm(np.asarray(o["points"])[0, :2] - np.asarray(o["target"])))
        return f"{self.title}   t={int(o['t']) / S.CONTROL_HZ:5.2f}s   far end to target {d:5.3f} m (need <= {S.POCKET_R:.2f})"

    def add(self, o) -> None:
        self._last = o
        if int(o["t"]) % self.every == 0:
            self.ims.append(frame(o, self.text(o)))

    def close(self, outcome: str | None = None) -> None:
        if self._last is None:
            return
        last = frame(self._last, self.text(self._last), outcome)
        ims = self.ims + [last] * 20
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        _write(ims, self.path)
        self.ims, self._last = [], None


def _write(ims, path):
    step, colors = 1, 64
    while True:
        sel = ims[::step]
        pal = [im.convert("P", palette=Image.ADAPTIVE, colors=colors) for im in sel]
        dur = int(round(1000 / FPS * step / 10)) * 10
        pal[0].save(path, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)
        if os.path.getsize(path) <= MAX_BYTES or step >= 8:
            return
        if colors > 32:
            colors = 32
        else:
            step *= 2


def outcome_text(d) -> str:
    if bool(d["success"]):
        return "SUCCESS"
    return f"TIME OUT - far end {float(d['final_tip_dist']):.3f} m from target"


def render_log(npz_path: str, out_path: str, title: str | None = None) -> str:
    """Re-render a saved black-box life or an evaluation step log (parallel.evaluate log_dir) into a GIF with the
    caller's title.  Pure replay of the logged readings; no simulator, no ledger."""
    d = np.load(npz_path, allow_pickle=False)
    if title is None:
        life = int(d["life"]) if "life" in d.files else 0
        title = f"life {life}/5" if life > 0 else f"seed {int(d['seed'])}"
    rec = Recorder(out_path, title)
    T = len(d["t"])
    keys = [k for k in ("points", "tool_pos", "post", "target", "t") if k in d.files]
    for i in range(T):
        rec.add({k: d[k][i] for k in keys})
    if "success" in d.files:
        out = "SUCCESS" if bool(d["success"]) else "TIME OUT"
    else:   # parallel step log: success means the last reading met the goal
        out = "SUCCESS" if (len(d["rewards"]) and d["rewards"][-1] > 5.0) else "TIME OUT" if len(d["rewards"]) else ""
    rec.close(out)
    return out_path


def contact_sheet(frames: list[dict], out_png: str, every: int = 5, cols: int = 4, title: str = "") -> str:
    """Lab tuning aid: a PNG grid of every `every`-th frame."""
    sel = frames[::every][: cols * 6]
    ims = [frame(o, f"{title} t={int(o['t']) / S.CONTROL_HZ:.2f}s").resize((W // 2, H // 2)) for o in sel]
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W // 2, rows * H // 2), (255, 255, 255))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * W // 2, (i // cols) * H // 2))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png
