"""Lab-side reference numbers on the real installation (purpose=eval, seeds 1000-1049) -> runs/station/_lab/baselines.json.

    python -m reallab.station._baselines all            # every reference below, then GIFs
    python -m reallab.station._baselines <name> [...]   # random noop naive_router naive_router_nodev informed informed_nodev
    python -m reallab.station._baselines speed          # single-process agent steps per second
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("runs/station/_lab/baselines.json")
GIFS = Path("runs/station/_lab/gifs")
SEEDS = list(range(1000, 1050))
REFS = {
    "random": ("reallab.station.parallel:random_policy", {"seed": 0}, False),
    "noop": ("reallab.station.parallel:noop_policy", {}, False),
    "naive_router": ("reallab.station._reference:naive_router", {}, False),
    "naive_router_nodev": ("reallab.station._reference:naive_router", {}, True),
    "informed": ("reallab.station._reference:informed", {}, False),
    "informed_nodev": ("reallab.station._reference:informed", {}, True),
}


def summary(res):
    from reallab.station._tune import classify
    br = {}
    for r in res:
        br[classify(r)] = br.get(classify(r), 0) + 1
    return dict(n=len(res), success=float(np.mean([r["success"] for r in res])), breakdown=br,
                mean_max_load=float(np.mean([r["max_load"] for r in res])),
                mean_final_dist=float(np.mean([r["final_dist"] for r in res])),
                mean_steps=float(np.mean([r["n_steps"] for r in res])),
                mean_return=float(np.mean([r["ret"] for r in res])),
                per_seed=[dict(seed=r["seed"], outcome=classify(r), steps=r["n_steps"], max_load=round(r["max_load"], 3),
                               final_dist=round(r["final_dist"], 4), sides_ok=r["sides_ok"]) for r in res])


def run(name):
    from reallab.station._reference import lab_evaluate
    fac, kw, nodev = REFS[name]
    if nodev:
        os.environ["CAWM_STATION_NODEV"] = "1"
    else:
        os.environ.pop("CAWM_STATION_NODEV", None)
    rec = name in ("naive_router", "informed")
    t0 = time.time()
    res = lab_evaluate(fac, kw, SEEDS, purpose="eval", tag=f"lab_ref_{name}", n_workers=24,
                       log_dir=f"runs/station/_lab/logs/{name}", record_first=1 if rec else 0, video_dir=str(GIFS),
                       record_title=f"{name} (deviations {'off' if nodev else 'on'}) seed {{seed}}")
    if rec:
        g = GIFS / f"ep_{SEEDS[0]}.gif"
        if g.exists():
            g.replace(GIFS / f"{name}.gif")
    entry = dict(summary(res), factory=fac, kwargs=kw, deviations="off" if nodev else "on", wall_s=round(time.time() - t0, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur[name] = entry
    OUT.write_text(json.dumps(cur, indent=1))
    print(name, {k: entry[k] for k in ("success", "breakdown", "mean_max_load", "mean_steps")}, flush=True)


def speed():
    from reallab.station import api
    from reallab.station._reference import naive_router
    env = api.make("smoke", tag="lab_speed")
    pol = naive_router()
    pol.bind(api.task_card())
    n, t_reset, t_step = 0, 0.0, 0.0
    for seed in range(5):
        pol.reset()
        t0 = time.time()
        obs = env.reset(seed)
        t_reset += time.time() - t0
        done = False
        t0 = time.time()
        while not done:
            obs, r, te, tr, info = env.step(pol.act(obs))
            done = te or tr
            n += 1
        t_step += time.time() - t0
    env.close()
    import mujoco
    from reallab.station import _scene
    out = dict(agent_steps_per_s=n / t_step, physics_steps_per_s=n * _scene.SUBSTEPS / t_step,
               reset_s=t_reset / 5, timestep=_scene.DT, substeps=_scene.SUBSTEPS, n_edges=_scene.N_EDGES,
               mujoco=mujoco.__version__, render_backend=os.environ.get("MUJOCO_GL"))
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur["speed"] = out
    OUT.write_text(json.dumps(cur, indent=1))
    print("speed", out, flush=True)


if __name__ == "__main__":
    names = sys.argv[1:]
    if names == ["all"]:
        names = list(REFS) + ["speed"]
    for nm in names:
        speed() if nm == "speed" else run(nm)
