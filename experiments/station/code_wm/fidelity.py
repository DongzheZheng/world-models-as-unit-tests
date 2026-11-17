# Open-loop fidelity of the code world -> runs/station/code_wm/fidelity.json
import json, glob
import numpy as np
import torch
from worlds.station import station_world as W
from experiments.station.code_wm.replay import restart_state

HELD_OUT = ["runs/station/observation/episode_908.npz", "runs/station/observation/episode_909.npz",
            "runs/station/blackbox/code_wm/life_2.npz", "runs/station/blackbox/code_wm/life_3.npz"]
FIT = [f"runs/station/observation/episode_{s}.npz" for s in range(900, 908)] + ["runs/station/blackbox/code_wm/life_1.npz"]
HS = (5, 20, 60)


def horizon_errors(files, par, stride=10):
    out = {h: dict(pos=[], load=[]) for h in HS}
    for f in files:
        E = np.load(f); T = len(E["actions"])
        A = torch.tensor(E["actions"], dtype=torch.float32, device="cuda")
        for t0 in range(1, T - 5, stride):
            st = restart_state(E, t0, par)
            for h in range(1, max(HS) + 1):
                if t0 + h > T:
                    break
                st, _, _ = W.step(st, A[t0 + h - 1:t0 + h]); st["done"][:] = False
                if h in HS:
                    x = st["x"][0].cpu().numpy()
                    out[h]["pos"].append(float(np.linalg.norm(x - E["points"][t0 + h][:, :2], axis=1).mean()))
                    out[h]["load"].append(abs(st["load"].item() - float(E["load"][t0 + h])))
    return {f"{h}_steps": dict(n=len(v["pos"]), marker_pos_err_mean_m=float(np.mean(v["pos"])), marker_pos_err_median_m=float(np.median(v["pos"])),
                               load_abs_err_mean_N=float(np.mean(v["load"])), load_abs_err_median_N=float(np.median(v["load"])))
            for h, v in out.items()}


def outcome_replay(f, par):
    E = np.load(f); T = len(E["actions"])
    st = W.init_from_obs(dict(points=E["points"][0][None], tool_pos=E["tool_pos"][0][None], tool_vel=E["tool_vel"][0][None],
                              posts=E["posts"][0][None], post_sides=E["post_sides"][0][None], target=E["target"][0][None],
                              load=np.array([E["load"][0]])), par=par)
    A = torch.tensor(E["actions"], dtype=torch.float32, device="cuda")
    pred = None
    for t in range(T):
        st, _, done = W.step(st, A[t:t + 1])
        if done.item():
            pred = ("success" if st["success"].item() else "overload" if st["overload"].item() else "timeout", t + 1)
            break
    real = ("success" if bool(E["success"]) else "overload" if bool(E["overload"]) else "timeout", int(E["n_steps"]))
    if pred is None:
        pred = ("still running when the log ends", T)
    return dict(log=f.split("/")[-1], used_for_fit=f in FIT, real=dict(outcome=real[0], step=real[1], max_load=float(E["max_load"])),
                world=dict(outcome=pred[0], step=pred[1], max_load=float(st["max_load"].item())),
                outcome_match=pred[0] == real[0])


if __name__ == "__main__":
    par = dict(W.DEFAULT_PAR)
    with torch.no_grad():
        F = dict(note="open-loop replay of the logged commands from the logged state (velocities by finite difference); "
                      "error = mean over the 20 markers of |predicted - logged| position, and |predicted - logged| load. "
                      "Restart every 10 steps.",
                 held_out_logs=HELD_OUT, held_out=horizon_errors(HELD_OUT, par), fitted_logs_for_reference=horizon_errors(FIT, par),
                 rope_stays_still_baseline=None,
                 observation_episode_outcomes=[outcome_replay(f"runs/station/observation/episode_{s}.npz", par) for s in range(900, 910)])
        # baseline: markers do not move (except nothing) -> error after h steps
        base = {}
        for h in HS:
            errs = []
            for f in HELD_OUT:
                E = np.load(f); P = E["points"][:, :, :2]
                for t0 in range(1, len(P) - h, 10):
                    errs.append(np.linalg.norm(P[t0 + h] - P[t0], axis=1).mean())
            base[f"{h}_steps"] = float(np.mean(errs))
        F["rope_stays_still_baseline"] = dict(held_out_marker_pos_err_mean_m=base)
    json.dump(F, open("runs/station/code_wm/fidelity.json", "w"), indent=1)
    print(json.dumps(F, indent=1))
