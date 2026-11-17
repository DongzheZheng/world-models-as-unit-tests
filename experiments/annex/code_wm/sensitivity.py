"""Sensitivity of the final controller's outcome to world-parameter error, measured inside the code world.
1) run the controller closed-loop in the NOMINAL world on snapshot situations (6 processes), recording its commands;
2) replay those commands open-loop in perturbed worlds (each parameter x (1 +- 3/5/10 %)) and with a tiny initial
   state perturbation (node positions + N(0, 0.1 mm) / N(0, 1 mm)); report success rate and p1 landing shift.
usage: python sensitivity.py <first> <count> <out.json>"""
import json
import sys
import time

import numpy as np


def _work(idx):
    import torch
    torch.set_num_threads(1)
    from experiments.annex.code_wm.screen import run_in_world, snapshot_obs
    res = run_in_world("experiments.annex.code_wm.throw:make_final", {}, snapshot_obs(idx), record_actions=True)
    for i, r in zip(idx, res):
        r["snapshot"] = int(i)
    return res


def replay(eps, prm, pos_noise=0.0, seed=0):
    """Open-loop replay with the episode rules; returns success flags and final p1 (stops at success like the lab)."""
    import torch
    from worlds.annex import annex_world as W
    from experiments.annex.code_wm.screen import snapshot_obs
    obs = snapshot_obs([e["snapshot"] for e in eps])
    st = W.init_from_obs(obs, None, prm, "cuda")
    if pos_noise > 0:
        g = torch.Generator(device="cuda").manual_seed(seed)
        nz = torch.randn(st["pos"].shape, device="cuda", generator=g) * pos_noise
        nz[:, -1] = 0
        st["pos"] = st["pos"] + nz
        st["prev_pos"] = st["pos"].clone()
    T = max(len(e["actions"]) for e in eps)
    A = np.stack([np.concatenate([e["actions"], np.repeat(e["actions"][-1:], T - len(e["actions"]), 0)]) for e in eps])
    A = torch.tensor(A, dtype=torch.float32, device="cuda")
    for k in range(W.TIME_LIMIT):
        a = A[:, min(k, T - 1)]
        st, r, d = W.step(st, a)
        if bool(d.all()):
            break
    return st["success"].cpu().numpy(), st["pos"][:, 0].cpu().numpy()


if __name__ == "__main__":
    import multiprocessing as mp
    a, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    idx = list(range(a, a + n))
    t0 = time.time()
    with mp.get_context("spawn").Pool(6) as pool:
        eps = [r for part in pool.map(_work, [idx[k::6] for k in range(6)]) for r in part]
    eps.sort(key=lambda r: r["snapshot"])
    for e in eps:
        e["actions"] = np.array(e["actions"])
    from experiments.annex.code_wm.screen import world_params
    nom = world_params()
    s0, p0 = replay(eps, nom)
    rep = dict(n_episodes=len(eps), closed_loop_nominal_success=float(np.mean([e["success"] for e in eps])),
               open_loop_replay_nominal_success=float(s0.mean()), params={}, initial_state={})
    keys = ["k_stretch", "c_stretch", "k_bend", "c_bend", "mu", "aniso", "c_visc", "c_quad", "lat_mass", "tip_mass",
            "servo_w", "servo_z", "servo_amax"]
    for k in keys:
        rep["params"][k] = {}
        for f in (-0.10, -0.05, -0.03, 0.03, 0.05, 0.10):
            p = dict(nom)
            p[k] = nom[k] * (1 + f)
            s, pp = replay(eps, p)
            sh = np.linalg.norm(pp - p0, axis=1)
            rep["params"][k][f"{f:+.2f}"] = dict(success=float(s.mean()), p1_shift_median=float(np.median(sh)),
                                                  p1_shift_mean=float(sh.mean()), p1_shift_p90=float(np.quantile(sh, 0.9)))
        print(k, rep["params"][k], flush=True)
    for nzs in (1e-4, 1e-3):
        s, pp = replay(eps, nom, pos_noise=nzs, seed=1)
        sh = np.linalg.norm(pp - p0, axis=1)
        rep["initial_state"][f"node_pos_noise_{nzs}"] = dict(success=float(s.mean()), p1_shift_median=float(np.median(sh)),
                                                             p1_shift_mean=float(sh.mean()), p1_shift_p90=float(np.quantile(sh, 0.9)))
    print(rep["initial_state"])
    rep["wall_s"] = time.time() - t0
    rep["wm_steps"] = int(sum(len(e["actions"]) for e in eps))
    json.dump(rep, open(out, "w"), indent=1)
