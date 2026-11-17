"""Blind screening on the rope task (annex), run by the orchestrator.  Every program version the code-as-policy pilot
left behind (pilots/aspire_annex/policy_v1..v19) is run (a) for real and (b) inside the code-world pilot's official
world (worlds/annex/annex_world.py with runs/annex/code_wm/world_params.json, via experiments/annex/code_wm/screen.py),
started from the reset readings of the same seeds only.

usage (from the repository root):
    python -m experiments.audit.annex_screen real  <first_seed> <n> [workers]
    python -m experiments.audit.annex_screen world <first_seed> <n> [device] [procs]
    python -m experiments.audit.annex_screen speed <device>
    python -m experiments.audit.annex_screen report
Results: runs/annex/audit/screen/{real,world,report}_<first_seed>.json.  The reset readings are generated once with
`python -m reallab.annex._snapshots` (see experiments/audit/snapshots.py) into runs/annex/audit/layouts_*.npz.

The world files used here are byte-identical to the frozen copy the screening was originally run from
(experiments/annex/code_wm/screen.py, worlds/annex/annex_world.py, runs/annex/code_wm/world_params.json).
"""
import json
import sys
import time
from pathlib import Path

OUT = Path("runs/annex/audit/screen")
VERS = [f"v{k}" for k in range(1, 20)]
FAC = "pilots.aspire_annex.policy_{}:make_policy"


def world_one(a):
    v, init, dev = a
    import torch
    torch.set_num_threads(1)
    from experiments.annex.code_wm.screen import run_in_world
    t0 = time.time()
    r = run_in_world(FAC.format(v), {}, init, device=dev)
    return v, [dict(success=bool(e["success"]), tip_dist=float(e["tip_dist"]), n_steps=int(e["n_steps"])) for e in r], time.time() - t0


if __name__ == "__main__":
    import numpy as np
    from experiments.audit.snapshots import init_obs_annex
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1]
    if mode == "speed":
        v, r, w = world_one(("v6", init_obs_annex(1000, 2), sys.argv[2]))
        print("speed", sys.argv[2], "2 episodes", round(w, 1), "s", r)
    elif mode == "real":
        from reallab.annex.parallel import evaluate
        s0, n = int(sys.argv[2]), int(sys.argv[3])
        nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        res = {}
        for v in VERS:
            eps = evaluate(FAC.format(v), {}, seeds=list(range(s0, s0 + n)), purpose="eval", pilot="", tag=f"screen_{v}", n_workers=nw)
            res[v] = [dict(success=bool(e["success"]), tip_dist=float(e["tip_dist"]), n_steps=int(e["n_steps"])) for e in eps]
            print("real", v, sum(e["success"] for e in res[v]), "/", n, "tip median", round(float(np.median([e["tip_dist"] for e in res[v]])), 3), flush=True)
        json.dump(res, open(OUT / f"real_{s0}.json", "w"))
    elif mode == "world":
        import multiprocessing as mp
        s0, n = int(sys.argv[2]), int(sys.argv[3])
        dev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
        procs = min(12, int(sys.argv[5]) if len(sys.argv) > 5 else 6)
        init = init_obs_annex(s0, n)
        res = {}
        with mp.get_context("spawn").Pool(procs) as pool:
            for v, r, w in pool.imap_unordered(world_one, [(v, init, dev) for v in VERS]):
                res[v] = r
                print("world", v, sum(e["success"] for e in r), "/", n, "tip median", round(float(np.median([e["tip_dist"] for e in r])), 3), "wall", round(w), flush=True)
        json.dump(res, open(OUT / f"world_{s0}.json", "w"))
    elif mode == "report":
        for s0 in (1000, 5000, 6000):
            fr, fw = OUT / f"real_{s0}.json", OUT / f"world_{s0}.json"
            if not (fr.exists() and fw.exists()):
                continue
            R, Wd = json.load(open(fr)), json.load(open(fw))
            rows, dr, dw, agree, tot = [], [], [], 0, 0
            for v in VERS:
                r, w = R[v], Wd[v]
                rs, ws = np.mean([e["success"] for e in r]), np.mean([e["success"] for e in w])
                rt, wt = np.array([e["tip_dist"] for e in r]), np.array([e["tip_dist"] for e in w])
                dr += list(rt); dw += list(wt)
                agree += sum(a["success"] == b["success"] for a, b in zip(r, w)); tot += len(r)
                rows.append(dict(version=v, real=float(rs), world=float(ws), real_tip_median=float(np.median(rt)), world_tip_median=float(np.median(wt)),
                                 tip_abs_err_median=float(np.median(np.abs(rt - wt)))))
                print(s0, v, "real", round(100 * rs), "world", round(100 * ws), "| tip median real", round(float(np.median(rt)), 2), "world", round(float(np.median(wt)), 2), "| per-episode |diff| median", round(float(np.median(np.abs(rt - wt))), 2))
            dr, dw = np.array(dr), np.array(dw)
            summ = dict(seeds=s0, n_programs=len(VERS), mae_points=float(np.mean([abs(x["real"] - x["world"]) for x in rows]) * 100),
                        max_err_points=float(max(abs(x["real"] - x["world"]) for x in rows) * 100), episode_agreement=agree / tot,
                        world_false_success=int(sum((not a["success"]) and b["success"] for v in VERS for a, b in zip(R[v], Wd[v]))),
                        real_success_total=int(sum(a["success"] for v in VERS for a in R[v])), world_success_total=int(sum(b["success"] for v in VERS for b in Wd[v])),
                        tip_corr=float(np.corrcoef(dr, dw)[0, 1]), tip_abs_err_median=float(np.median(np.abs(dr - dw))), rows=rows)
            json.dump(summ, open(OUT / f"report_{s0}.json", "w"), indent=1)
            print({k: v for k, v in summ.items() if k != "rows"})
