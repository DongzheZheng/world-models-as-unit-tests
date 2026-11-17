"""English showdown video for the cable task (station).  Run by the orchestrator.

Three programs on the same real episode (evaluation seed 1000), re-rendered from their real step logs with the
lab's renderer.  The simple constant-speed program is run here once on seeds 1000-1049 (cached) to get its log and
its success rate; the two pilots' panels come from the step logs of their official evaluations.
The code world renders nothing: every frame is the simulator that plays the real world.

usage: python experiments/audit/station_montage.py  -> runs/station/gifs/station_showdown_en.{mp4,gif} (+ _last/_mid png)
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

SEED, SPEED, FPS = 1000, 2, 20
NAIVE_DIR = Path("runs/station/audit_logs/gif_naive")
PANEL = (480, 320)


def naive_stats():
    cache = NAIVE_DIR / "result.json"
    if not cache.exists():
        from reallab.station.parallel import evaluate
        eps = evaluate("reallab.station._reference:naive_router", {}, seeds=list(range(1000, 1050)), purpose="eval",
                       pilot="audit", tag="gif_naive", n_workers=12, log_dir=str(NAIVE_DIR))
        cache.write_text(json.dumps(eps))
    return json.loads(cache.read_text())


def frames_of(log_path):
    from reallab.station import _internal, _render, _scene
    d = np.load(log_path, allow_pickle=False)
    sc = _scene.Scene(_internal.physical())
    sc.place(d["posts"][0], d["target"][0])
    load = np.asarray(d["load"], float)
    T = len(d["actions"])
    out, peak = [], 0.0
    for i in list(range(0, T + 1, SPEED)) + ([T] if T % SPEED else []):
        obs = {k: d[k][i] for k in ("points", "tool_pos", "posts", "post_sides", "sides_now", "target")}
        sc.pose_from_points(obs["points"], obs["tool_pos"])
        peak = max(peak, float(load[: i + 1].max()))
        im = _render.frame(sc, obs, f"t = {i / 20.0:4.1f} s     load {load[i]:4.1f} / 10 N")
        out.append(im.convert("RGB").resize(PANEL, Image.LANCZOS))
    overload = bool(load.max() > 10.0)
    ok = (not overload) and T < 600
    return out, dict(T=T, max_load=float(load.max()), overload=overload, ok=ok, load=load)


def clip(title, sub, log_path):
    frames, info = frames_of(log_path)
    final = (f"done in {info['T'] / 20.0:.1f} s · peak load {info['max_load']:.1f} N" if info["ok"] else
             f"OVERLOAD at {info['T'] / 20.0:.1f} s (load above 10 N)" if info["overload"] else "out of time")
    load = info["load"]
    return dict(title=title, sub=sub, frames=frames, ok=info["ok"], final=final,
                live=lambda k: f"load {load[min(k * SPEED, len(load) - 1)]:.1f} N of 10")


if __name__ == "__main__":
    from experiments.audit.montage_en import compose
    naive = naive_stats()
    rate = lambda eps: f"{sum(bool(e['success']) for e in eps)}/{len(eps)}"
    res = lambda c: json.loads(Path(f"runs/station/results/station__{c}.json").read_text())
    n_ok = lambda c: f"{round(res(c)['eval']['mean'] * res(c)['eval']['n'])}/{res(c)['eval']['n']}"
    clips = [
        clip("Existing program", f"constant 15 cm/s · no real trials · {rate(naive)} episodes", NAIVE_DIR / f"ep_{SEED}.npz"),
        clip("Code as Policies agent", f"5 real trials · {n_ok('aspire_style')} episodes", f"runs/station/aspire_style/eval_logs/ep_{SEED}.npz"),
        clip("Code-as-world-model agent", f"3 real trials · {n_ok('code_wm')} episodes", f"runs/station/code_wm/eval_logs/ep_{SEED}.npz"),
    ]
    compose(clips, "Routing a cable around three posts · real MuJoCo · same episode (seed 1000) · 2x speed",
            "Pull too fast and the load passes 10 N. Neither agent was told what the setup is; both worked it out from logs.",
            3, "runs/station/gifs/station_showdown_en", fps=FPS)
