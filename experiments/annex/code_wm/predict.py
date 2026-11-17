"""In-world success prediction for a controller over snapshot situations, 6 worker processes (spawn).
usage: python predict.py <factory> <first> <count> <out.json> [params_override.json]"""
import json
import sys
import time


def _work(args):
    fac, idx, override = args
    import torch
    torch.set_num_threads(1)
    from experiments.annex.code_wm.screen import run_in_world, snapshot_obs
    res = run_in_world(fac, {}, snapshot_obs(idx), params=override)
    for i, r in zip(idx, res):
        r["snapshot"] = int(i)
    return res


if __name__ == "__main__":
    import multiprocessing as mp
    fac, a, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    override = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
    idx = list(range(a, a + n))
    chunks = [(fac, idx[k::6], override) for k in range(6) if idx[k::6]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        res = [r for part in pool.map(_work, chunks) for r in part]
    res.sort(key=lambda r: r["snapshot"])
    s = sum(r["success"] for r in res) / len(res)
    json.dump(dict(factory=fac, override=override, success_rate=s, episodes=res, wall_s=time.time() - t0,
                   steps=int(sum(r["n_steps"] for r in res))), open(out, "w"), indent=1)
    print("success", s, "wall", time.time() - t0)
