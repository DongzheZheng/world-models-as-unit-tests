"""Lab-side (orchestrator): scene snapshots.  Internal: pilots must not read this module.

Looking is free, acting is not.  For a range of seeds, take the reset reading only (no command is ever sent) and
store them in one file that every pilot may study:  <out>.npz  with arrays
    seeds (N,), points (N,20,3), tool_pos (N,2), post (N,2), zone (N,2,2), target (N,2)
Counted in the ledger under purpose "smoke" (zero agent steps).

usage: python -m reallab.annex._snapshots <first_seed> <last_seed_exclusive> <out.npz> [n_workers (<= 12)]
"""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

KEYS = ("points", "tool_pos", "post", "zone", "target")


def _chunk(args):
    seeds, tag = args
    from reallab.annex import api
    env = api.RealAnnex("smoke", "", tag, None)
    rows = []
    for s in seeds:
        o = env.reset(int(s))
        rows.append({k: np.array(o[k]) for k in KEYS})
    env.close()
    return list(seeds), rows


def main():
    a, b, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    nw = min(12, int(sys.argv[4]) if len(sys.argv) > 4 else 8)
    seeds = list(range(a, b))
    chunks = [(seeds[i::nw], "lab/scene_snapshots") for i in range(nw) if seeds[i::nw]]
    with mp.get_context("spawn").Pool(len(chunks)) as pool:
        res = pool.map(_chunk, chunks)
    by_seed = {s: r for ss, rr in res for s, r in zip(ss, rr)}
    data = {k: np.stack([by_seed[s][k] for s in seeds]) for k in KEYS}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, seeds=np.array(seeds), **data)
    print("wrote", out, {k: v.shape for k, v in data.items()})


if __name__ == "__main__":
    main()
