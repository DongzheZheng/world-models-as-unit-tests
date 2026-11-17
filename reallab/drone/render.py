"""GIF rendering of logged flights (CPU only: PyBullet TinyRenderer in DIRECT mode, PIL, matplotlib/Agg).

  load_log(path_or_dict)                      -> dict(pos, quat, t, relays, relay_index, outcome, pilot, life, ...)
  render_episode_gif(log, out_path, title="") -> 3-D replay with relay markers, trail, overlay, other aircraft
  render_topdown_gif(logs, out_path, labels)  -> matplotlib top-down map comparing several flights
Both accept reveal_truth=True (lab-only extra overlay, drawn by an internal helper module)."""
from __future__ import annotations

import gzip
import io
import json
import os

import numpy as np

OUTCOME_COLOR = {"success": (40, 170, 70), "collision": (210, 40, 40), "out_of_bounds": (210, 120, 20),
                 "timeout": (200, 150, 0), "abandoned": (120, 120, 120), "unknown": (120, 120, 120)}
BOX_COLORS = [(0.80, 0.35, 0.20, 1), (0.20, 0.45, 0.70, 1), (0.65, 0.20, 0.25, 1), (0.25, 0.55, 0.45, 1),
              (0.75, 0.60, 0.20, 1)]


# ----------------------------------------------------------------------------------------------- loading
def load_log(src) -> dict:
    """Accepts a black-box .npz, an evaluate() .jsonl.gz episode log, or an already loaded dict."""
    if isinstance(src, dict):
        return src
    src = str(src)
    out: dict = {}
    if src.endswith(".npz"):
        z = np.load(src, allow_pickle=False)
        pos = z["pos"] if "pos" in z.files else z["lin_pos"]
        out = dict(pos=pos, quat=z["quat"], t=z["t"].reshape(-1) if "t" in z.files else np.arange(len(pos)) / 30.0,
                   relays=z["relays"][0] if "relays" in z.files and z["relays"].ndim == 3 else None,
                   relay_index=z["relay_index"].reshape(-1) if "relay_index" in z.files else None,
                   actions=z["actions"], outcome=str(z["end_cause"]), pilot=str(z["pilot"]) if "pilot" in z.files else "",
                   life=int(z["life"]) if "life" in z.files else None,
                   mission=str(z["mission"]) if "mission" in z.files else "A",
                   mode=str(z["command_mode"]) if "command_mode" in z.files else "ctbr",
                   goal=z["goal"] if "goal" in z.files else None,
                   movers=(z["movers_true_pos"][:, : int(z["movers_true_count"].max())] if "movers_true_pos" in z.files else None))
        return out
    rows = [json.loads(l) for l in gzip.open(src, "rt")]
    obs = [rows[0]["obs"]] + [r["obs"] for r in rows if r["type"] == "step"]
    info = rows[-1]["info"] if rows[-1]["type"] == "end" else {}
    mv = [r.get("render_movers") for r in rows if r["type"] in ("reset", "step")]
    mv = np.array(mv) if mv and all(m is not None for m in mv) else None
    key = "pos" if "pos" in obs[0] else "lin_pos"
    out = dict(pos=np.array([o[key] for o in obs]), quat=np.array([o["quat"] for o in obs]),
               t=np.arange(len(obs)) / 30.0,
               relays=np.array(obs[0]["relays"]) if "relays" in obs[0] else None,
               relay_index=np.array([o["relay_index"][0] for o in obs]) if "relay_index" in obs[0] else None,
               actions=np.array([r["action"] for r in rows if r["type"] == "step"]),
               outcome=next((k for k in ("success", "collision", "out_of_bounds", "timeout") if info.get(k)), "unknown"),
               pilot="", life=None, mission="A", mode="?", goal=np.array([o["goal"] for o in obs]) if "goal" in obs[0] else None,
               relays_reached=info.get("relays_reached"), movers=mv)
    return out


def _reached(log, i):
    """Number of relays reached by sample i."""
    ri = log.get("relay_index")
    if ri is None:
        return 0
    n = int(ri[min(i, len(ri) - 1)])
    if i >= len(ri) - 1 and log["outcome"] == "success":
        return 3
    return n


def _default_boxes(reveal=False):
    from .api import site_map
    return site_map()["boxes"]


def _font(size):
    from PIL import ImageFont
    try:
        import matplotlib
        return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def _project(pts, view, proj, W, H):
    V = np.array(view).reshape(4, 4).T
    P = np.array(proj).reshape(4, 4).T
    p = np.c_[np.asarray(pts, float).reshape(-1, 3), np.ones(len(np.atleast_2d(pts)))]
    c = (P @ V @ p.T).T
    w = c[:, 3:4]
    ndc = c[:, :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)
    ok = (w[:, 0] > 0)
    return np.c_[(ndc[:, 0] * 0.5 + 0.5) * W, (1 - (ndc[:, 1] * 0.5 + 0.5)) * H], ok


def _save_gif(frames, path, fps, max_mb):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stride = 1
    while True:
        fr = frames[::stride]
        if stride > 1 and (len(frames) - 1) % stride:
            fr = fr + [frames[-1]]
        pal = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in fr]
        dur = int(1000 / fps) * stride
        pal[0].save(path, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True, disposal=1)
        if os.path.getsize(path) <= max_mb * 1e6 or len(fr) < 20:
            return path
        stride += 1


# ----------------------------------------------------------------------------------------------- 3-D replay
def render_episode_gif(log, out_path, title="", boxes=None, fps=15, size=(480, 360), speed=2.0,
                       reveal_truth=False, max_mb=6.0, label=None, cutaway=True):
    import pybullet as pb
    from PIL import Image, ImageDraw
    log = load_log(log)
    W, H = size
    pos, quat = np.asarray(log["pos"], float), np.asarray(log["quat"], float)
    boxes = boxes or _default_boxes()
    relays = log.get("relays")
    cid = pb.connect(pb.DIRECT)
    kw = dict(physicsClientId=cid)
    try:
        def vbox(half, rgba):
            return pb.createVisualShape(pb.GEOM_BOX, halfExtents=half, rgbaColor=rgba, **kw)
        def body(vs, p, q=(0, 0, 0, 1)):
            return pb.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=p, baseOrientation=q, **kw)
        body(vbox([45, 20, 0.05], (0.62, 0.64, 0.58, 1)), [30, 0, -0.05])
        body(vbox([30, 12, 0.06], (0.70, 0.70, 0.66, 1)), [30, 0, 0.0])
        for i, b in enumerate(boxes):
            he, c = list(b["half_extents"]), list(b["center"])
            if cutaway and c[1] < 0 and b.get("kind") == "container_stack":      # cut away the near (south) wall
                he[2], c[2] = 0.5, 0.5
            body(vbox(he, BOX_COLORS[i % len(BOX_COLORS)]), c)
        relay_ids = []
        if relays is not None:
            for r in relays:
                relay_ids.append(pb.createMultiBody(baseMass=0, basePosition=r, baseVisualShapeIndex=pb.createVisualShape(
                    pb.GEOM_SPHERE, radius=0.7, rgbaColor=(1, 0.9, 0.1, 0.55), **kw), **kw))
        if reveal_truth:
            from ._render_truth import add_truth_bodies
            add_truth_bodies(pb, cid, body, vbox)
        mv = log.get("movers")
        mv_ids = []
        if mv is not None:
            sph = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.5, rgbaColor=(0.85, 0.2, 0.85, 1), **kw)
            mv_ids = [pb.createMultiBody(baseMass=0, baseVisualShapeIndex=sph, basePosition=[0, 0, -9], **kw)
                      for _ in range(mv.shape[1])]
        arm = vbox([0.55, 0.07, 0.05], (0.05, 0.05, 0.05, 1))
        arms = [body(arm, [0, 0, 0]), body(arm, [0, 0, 0])]
        hub = body(pb.createVisualShape(pb.GEOM_SPHERE, radius=0.22, rgbaColor=(0.95, 0.1, 0.1, 1), **kw), [0, 0, 0])
        nose = body(pb.createVisualShape(pb.GEOM_SPHERE, radius=0.13, rgbaColor=(1, 1, 1, 1), **kw), [0, 0, 0])
        marker = None
        if reveal_truth:
            marker = body(pb.createVisualShape(pb.GEOM_SPHERE, radius=0.33, rgbaColor=(1, 0, 1, 1), **kw), [0, 0, 0])
        proj = pb.computeProjectionMatrixFOV(60, W / H, 0.1, 200, **kw)
        step = max(1, int(round(30 / fps * speed)))
        idx = list(range(0, len(pos), step))
        if idx[-1] != len(pos) - 1:
            idx.append(len(pos) - 1)
        font, font_b = _font(12), _font(15)
        frames = []
        cam = pos[0] + np.array([-3.0, -11.0, 7.0])
        for n, i in enumerate(idx):
            p = pos[i]
            R = np.array(pb.getMatrixFromQuaternion(quat[i].tolist())).reshape(3, 3)
            for k, a in enumerate(arms):
                rz = pb.getQuaternionFromEuler([0, 0, np.pi / 4 + k * np.pi / 2])
                _, q = pb.multiplyTransforms([0, 0, 0], quat[i].tolist(), [0, 0, 0], rz)
                pb.resetBasePositionAndOrientation(a, p.tolist(), q, **kw)
            pb.resetBasePositionAndOrientation(hub, p.tolist(), [0, 0, 0, 1], **kw)
            pb.resetBasePositionAndOrientation(nose, (p + R @ np.array([0.6, 0, 0])).tolist(), [0, 0, 0, 1], **kw)
            if marker is not None:
                from ._render_truth import marker_offset
                pb.resetBasePositionAndOrientation(marker, (p + R @ marker_offset(0.65)).tolist(), [0, 0, 0, 1], **kw)
            for k, mid in enumerate(mv_ids):
                pb.resetBasePositionAndOrientation(mid, np.asarray(mv[min(i, len(mv) - 1)][k]).tolist(), [0, 0, 0, 1], **kw)
            nrel = _reached(log, i)
            for k, rid in enumerate(relay_ids):
                col = (0.1, 0.85, 0.2, 0.7) if k < nrel else ((1, 0.55, 0.05, 0.75) if k == nrel else (1, 0.9, 0.1, 0.5))
                pb.changeVisualShape(rid, -1, rgbaColor=col, **kw)
            tgt = p + np.array([3.0, 0.0, 1.0])
            want = p + np.array([-3.0, -11.0, 7.0])
            cam = cam + 0.25 * (want - cam)
            view = pb.computeViewMatrix(cam.tolist(), tgt.tolist(), [0, 0, 1], **kw)
            img = pb.getCameraImage(W, H, view, proj, renderer=pb.ER_TINY_RENDERER, **kw)[2]
            im = Image.fromarray(np.reshape(np.array(img, dtype=np.uint8), (H, W, 4))[:, :, :3])
            d = ImageDraw.Draw(im, "RGBA")
            tr, ok = _project(pos[: i + 1][-240:], view, proj, W, H)
            pts = [tuple(x) for x, o in zip(tr, ok) if o]
            if len(pts) > 1:
                d.line(pts, fill=(255, 255, 255, 230), width=2)
            if relays is not None:
                rp, rok = _project(relays, view, proj, W, H)
                for k in range(len(relays)):
                    if rok[k]:
                        c = (30, 200, 60) if k < nrel else (255, 140, 20)
                        d.text((rp[k][0] - 4, rp[k][1] - 26), f"R{k + 1}", fill=c + (255,), font=font_b)
            d.rectangle([0, 0, W, 38], fill=(0, 0, 0, 150))
            d.text((6, 3), title or "", fill=(255, 255, 255, 255), font=font_b)
            tt = log["t"][i] if log.get("t") is not None and len(log["t"]) > i else i / 30.0
            d.text((6, 21), f"t = {tt:5.1f} s    relays {nrel}/3    z = {p[2]:.1f} m", fill=(230, 230, 230, 255), font=font)
            if i == len(pos) - 1:
                oc = log["outcome"]
                c = OUTCOME_COLOR.get(oc, (120, 120, 120))
                d.rectangle([0, H - 34, W, H], fill=c + (200,))
                d.text((8, H - 29), (label or "") + oc.upper().replace("_", " "), fill=(255, 255, 255, 255), font=font_b)
            frames.append(im)
        frames += [frames[-1]] * int(fps * 1.2)
        return _save_gif(frames, out_path, fps, max_mb)
    finally:
        pb.disconnect(cid)


def render_life_gif(npz_path, out_path, reveal_truth=False, **kw):
    """Render one black-box life, labelled 'life k/5 - pilot'."""
    log = load_log(npz_path)
    title = f"life {log['life']}/5 - {log['pilot']}   (mission {log['mission']}, {log['mode']})"
    return render_episode_gif(log, out_path, title=title, reveal_truth=reveal_truth, **kw)


# ----------------------------------------------------------------------------------------------- top-down comparison
def render_topdown_gif(logs, out_path, labels=None, boxes=None, title="", fps=15, speed=3.0, reveal_truth=False,
                       max_mb=6.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    logs = [load_log(l) for l in logs]
    labels = labels or [f"flight {i + 1}" for i in range(len(logs))]
    boxes = boxes or _default_boxes()
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=80)
    ax.set_xlim(-1, 61); ax.set_ylim(-12.5, 12.5); ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, -12), 60, 24, fill=False, ec="0.5", lw=1))
    for b in boxes:
        c, h = b["center"], b["half_extents"]
        ax.add_patch(Rectangle((c[0] - h[0], c[1] - h[1]), 2 * h[0], 2 * h[1], fc="0.45", ec="0.25"))
    if reveal_truth:
        from ._render_truth import draw_truth_topdown
        draw_truth_topdown(ax)
    relays = next((l["relays"] for l in logs if l.get("relays") is not None), None)
    rel_art = []
    if relays is not None:
        for k, r in enumerate(relays):
            rel_art.append(ax.plot(r[0], r[1], "o", ms=11, mfc="gold", mec="k", zorder=5)[0])
            ax.text(r[0], r[1] + 1.1, f"R{k + 1}", ha="center", fontsize=8, zorder=6)
    cols = plt.cm.tab10(np.arange(len(logs)) % 10)
    lines = [ax.plot([], [], "-", lw=1.8, color=c, label=lab)[0] for c, lab in zip(cols, labels)]
    dots = [ax.plot([], [], "o", ms=7, color=c, mec="k", zorder=7)[0] for c in cols]
    mlog = next((l for l in logs if l.get("movers") is not None), None)
    mdots = ax.plot([], [], "o", ms=8, color="m", mec="k", alpha=0.8, zorder=6)[0] if mlog is not None else None
    ax.legend(loc="lower left", fontsize=7, ncol=3, framealpha=0.85)
    ax.set_title(title, fontsize=10)
    txt = ax.text(0.995, 0.03, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, zorder=9, bbox=dict(fc="w", ec="none", alpha=0.7))
    ax.set_xlabel("x [m]", fontsize=8); ax.set_ylabel("y [m]", fontsize=8)
    fig.tight_layout()
    nmax = max(len(l["pos"]) for l in logs)
    step = max(1, int(round(30 / fps * speed)))
    idx = list(range(0, nmax, step)) + [nmax - 1]
    frames = []
    for i in idx:
        for l, ln, dt, c in zip(logs, lines, dots, cols):
            j = min(i, len(l["pos"]) - 1)
            ln.set_data(l["pos"][: j + 1, 0], l["pos"][: j + 1, 1])
            dt.set_data([l["pos"][j, 0]], [l["pos"][j, 1]])
            if j == len(l["pos"]) - 1:
                dt.set_marker({"success": "*", "collision": "X"}.get(l["outcome"], "s")); dt.set_markersize(11)
        if relays is not None:
            reach = max(_reached(l, min(i, len(l["pos"]) - 1)) for l in logs)
            for k, a in enumerate(rel_art):
                a.set_markerfacecolor("limegreen" if k < reach else "gold")
        if mdots is not None:
            M = mlog["movers"][min(i, len(mlog["movers"]) - 1)]
            mdots.set_data(M[:, 0], M[:, 1])
        txt.set_text(f"t = {i / 30:5.1f} s")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(buf.copy()))
    plt.close(fig)
    frames += [frames[-1]] * int(fps * 1.5)
    return _save_gif(frames, out_path, fps, max_mb)
