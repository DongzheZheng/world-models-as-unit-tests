# Controller tuning / robustness inside the code world (snapshot situations + observation reset readings).
#   python experiments/station/code_wm/tune.py '<json list of kwargs dicts>' [randomize]
import sys, json, time
import numpy as np
from experiments.station.code_wm.screen import run_in_world, layouts_as_obs, summarize


def obs_reset_readings():
    out = []
    for s in range(900, 910):
        E = np.load(f"runs/station/observation/episode_{s}.npz")
        out.append(dict(points=E["points"][0], tool_pos=E["tool_pos"][0], tool_vel=E["tool_vel"][0], posts=E["posts"][0],
                        post_sides=E["post_sides"][0].astype(np.int64), target=E["target"][0], load=float(E["load"][0]), t=0))
    return out


if __name__ == "__main__":
    cands = json.loads(sys.argv[1])
    rnd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    obs = layouts_as_obs() + obs_reset_readings()
    tot_steps, t0 = 0, time.time()
    for kw in cands:
        res = run_in_world("experiments.station.code_wm.policy:make_policy", kw, obs, randomize=rnd, seed=1)
        tot_steps += sum(r["n_steps"] for r in res)
        s1 = summarize(res[:200]); s2 = summarize(res[200:])
        print(json.dumps(kw), "| snapshots", json.dumps(s1), "| obs-situations succ", s2["success"],
              "ov", s2["overload"], "maxload", [round(r["max_load"], 1) for r in res[200:]], flush=True)
    print("world steps", tot_steps, "wall", round(time.time() - t0, 1))
