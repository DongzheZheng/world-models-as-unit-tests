"""Extract the behaviour of the other aircraft from mission-C black-box instrumentation (true mover states).

Per mover: segments of constant velocity separated by stops/turns -> kind (shuttle along one axis / wanderer),
speed, turn points, dwell times, z level, and the relation of its motion to our own drone (does it head for us?).
   python experiments/drone/code_wm/fit_movers.py out.json life_1.npz [life_2.npz ...]"""
import json
import sys

import numpy as np


def segments(p, v, eps=0.05):
    """Split into moving / dwelling runs by speed; returns list of dicts."""
    sp = np.linalg.norm(v, axis=1)
    mov = sp > eps
    runs, s = [], 0
    for i in range(1, len(sp) + 1):
        if i == len(sp) or mov[i] != mov[s]:
            runs.append(dict(moving=bool(mov[s]), i0=s, i1=i, p0=p[s].tolist(), p1=p[i - 1].tolist(),
                             speed=float(sp[s:i].mean()), dir=(v[s:i].mean(0) / max(sp[s:i].mean(), 1e-9)).tolist()))
            s = i
    return runs


def analyse(path):
    z = np.load(path)
    P, V = z["movers_true_pos"], z["movers_true_vel"]
    own, ownv = z["pos"], z["lin_vel"]
    n = int(z["movers_true_count"].max())
    out = []
    for m in range(n):
        p, v = P[:, m], V[:, m]
        runs = segments(p, v)
        mv = [r for r in runs if r["moving"]]
        dw = [r for r in runs if not r["moving"]]
        ext = p.max(0) - p.min(0)
        axes = [int(a) for a in np.where(ext > 0.3)[0]]
        # direction changes within moving runs (velocity jumps without stopping) -> also turn points
        dv = np.linalg.norm(np.diff(v, axis=0), axis=1)
        turns = int((dv > 0.5).sum())
        kind = "shuttle_" + "xyz"[axes[0]] if len(axes) == 1 else ("wanderer" if len(axes) >= 2 else "static")
        # heading of moving runs relative to our drone: cos(angle between mover velocity and direction to us)
        cosines = []
        for r in mv:
            i = (r["i0"] + r["i1"]) // 2
            to_me = own[min(i, len(own) - 1)] - p[i]
            dvec = np.array(r["dir"])
            cosines.append(float(dvec @ to_me / max(np.linalg.norm(to_me), 1e-9)))
        d_me = np.linalg.norm(p[: len(own)] - own, axis=1)
        out.append(dict(
            mover=m, kind=kind, axes=axes, z_level=[float(p[:, 2].min()), float(p[:, 2].max())],
            x_range=[float(p[:, 0].min()), float(p[:, 0].max())], y_range=[float(p[:, 1].min()), float(p[:, 1].max())],
            speeds=sorted({round(r["speed"], 2) for r in mv if r["i1"] - r["i0"] > 3}),
            dwell_s=[round((r["i1"] - r["i0"]) / 30, 2) for r in dw],
            hop_lengths=[round(float(np.linalg.norm(np.array(r["p1"]) - np.array(r["p0"]))), 2) for r in mv],
            stop_points=[[round(c, 2) for c in r["p0"]] for r in dw],
            n_turns_without_stop=turns, cos_heading_to_me=[round(c, 2) for c in cosines],
            min_dist_to_me=float(d_me.min()), t_min_dist=float(d_me.argmin() / 30)))
    return dict(life=path, end=str(z["end_cause"]), steps=int(z["n_steps"]), own_start=own[0].tolist(), movers=out)


if __name__ == "__main__":
    res = [analyse(p) for p in sys.argv[2:]]
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    for L in res:
        print("=====", L["life"], L["end"], L["steps"])
        for m in L["movers"]:
            print(m["mover"], m["kind"], "z", np.round(m["z_level"], 2).tolist(), "x", np.round(m["x_range"], 1).tolist(),
                  "y", np.round(m["y_range"], 2).tolist(), "v", m["speeds"], "dwell", m["dwell_s"][:8], "hops", m["hop_lengths"][:8],
                  "turns", m["n_turns_without_stop"], "cos->me", m["cos_heading_to_me"][:8], "dmin", round(m["min_dist_to_me"], 2))
