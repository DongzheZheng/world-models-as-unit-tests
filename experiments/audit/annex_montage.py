"""English video for the dynamic rope task (annex).  Run by the orchestrator.

Several programs on the same real episode, re-rendered from their real step logs with the lab's renderer (a pure
replay of logged readings).  The code world renders nothing: every frame is the simulator that plays the real world.

usage: python experiments/audit/annex_montage.py <seed> <cols> <stem> "<title>|<sub>|<log_dir>" ...
  each <sub> is the caption under the panel title (lives and success count), passed in by the caller from verified
  result files.
"""
import sys

import numpy as np
from PIL import Image

FPS = 25                      # every second 50 Hz reading -> real time
PANEL = (480, 373)


def clip(title, sub, log_path):
    from reallab.annex import _render
    d = np.load(log_path, allow_pickle=False)
    T = len(d["actions"])
    dist = np.linalg.norm(d["points"][:, 0, :2] - d["target"], axis=1)
    idx = list(range(0, T + 1, 2)) + ([T] if T % 2 else [])
    frames = []
    for i in idx:
        o = {k: d[k][i] for k in ("points", "tool_pos", "post", "target", "t")}
        im = _render.frame(o, f"t = {i / 50.0:4.2f} s")
        frames.append(im.convert("RGB").resize(PANEL, Image.LANCZOS))
    ok = bool(len(d["rewards"]) and float(d["rewards"].max()) > 5.0)
    moving = float(np.linalg.norm(d["points"][-1][:, :2] - d["points"][-2][:, :2], axis=1).max()) > 0.02 / 50
    final = (f"at rest inside the target after {T / 50.0:.2f} s" if ok else
             f"out of time, still moving, far end {dist[-1]:.2f} m from the target" if moving else
             f"came to rest {dist[-1]:.2f} m from the target")
    return dict(title=title, sub=sub, frames=frames, ok=ok, final=final,
                live=lambda k: f"far end {dist[min(idx[min(k, len(idx) - 1)], len(dist) - 1)]:.2f} m from the target (needs 0.10)")


if __name__ == "__main__":
    from experiments.audit.montage_en import compose
    seed, cols, stem = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    clips = []
    for spec in sys.argv[4:]:
        title, sub, log_dir = spec.split("|")
        clips.append(clip(title, sub, f"{log_dir}/ep_{seed}.npz"))
    compose(clips, f"Casting a rope past a post onto a target · real MuJoCo · same episode (seed {seed}) · real time",
            "One fast throw, 4.7 s, no force reading: the far end must come to rest inside the green ring. Neither agent was told what the setup is.",
            cols, stem, fps=FPS)
