"""Reset readings ("looking is free") for the audits of the cable (station) and rope (annex) tasks.

The audits that score a program inside a code world start every imagined episode from the reset reading of a real
seed.  These readings are taken with the labs' own snapshot tools, which reset the real environment and never send
a command (zero agent steps; the ledger counts them under purpose "smoke"):

    python -m reallab.station._snapshots <first> <last_exclusive> <out.npz> [workers]
    python -m reallab.annex._snapshots   <first> <last_exclusive> <out.npz> [workers]

layouts(task, first, n) runs the right tool once and caches the file under runs/<lab>/audit/.
init_obs_station / init_obs_annex turn the rows into lab-format reset observations.
Call these only under `if __name__ == "__main__":` (the snapshot tools start worker processes with "spawn").
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = {"cable": ("reallab.station._snapshots", "runs/station/audit"),
         "rope": ("reallab.annex._snapshots", "runs/annex/audit")}


def layouts(task: str, first: int, n: int, workers: int = 8) -> Path:
    """Path of an .npz with the reset readings of seeds first..first+n-1 (generated on first use)."""
    mod, rel = TOOLS[task]
    path = ROOT / rel / f"layouts_{first}_{first + n}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        nw = max(1, min(12, workers, n))
        print(f"[snapshots] reset readings of {task} seeds {first}..{first + n - 1} -> {path.relative_to(ROOT)}",
              flush=True)
        subprocess.run([sys.executable, "-m", mod, str(first), str(first + n), str(path), str(nw)], check=True, cwd=ROOT)
    return path


def _rows(task, first, n, workers):
    d = np.load(layouts(task, first, n, workers))
    seeds = [int(s) for s in d["seeds"]]
    return d, [seeds.index(s) for s in range(first, first + n)]


def init_obs_station(first: int, n: int, workers: int = 8) -> list[dict]:
    from reallab.station import api
    d, idx = _rows("cable", first, n, workers)
    out = []
    for i in idx:
        pts, posts = d["points"][i], d["posts"][i]
        out.append(dict(points=pts, tool_pos=d["tool_pos"][i], tool_vel=np.zeros(2), posts=posts,
                        post_sides=d["post_sides"][i], sides_now=api.sides(pts, posts), target=d["target"][i],
                        load=float(d["load"][i]), t=0))
    return out


def init_obs_annex(first: int, n: int, workers: int = 8) -> list[dict]:
    d, idx = _rows("rope", first, n, workers)
    return [dict(points=d["points"][i].copy(), tool_pos=d["tool_pos"][i].copy(), tool_vel=np.zeros(2),
                 post=d["post"][i].copy(), zone=d["zone"][i].copy(), target=d["target"][i].copy(), t=0) for i in idx]
