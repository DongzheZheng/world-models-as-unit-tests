"""Fit the code world's parameters to black-box lives by gradient descent through dyn_step (multi-step open loop).

  python experiments/drone/code_wm/fit_world.py --lives life_1.npz ... [--holdout life_k.npz ...] --out fitted.json
Each window starts from a logged state and replays the logged commands for H steps; the loss compares predicted and
logged pos / vel / body rates / attitude.  Also reports open-loop errors at 0.5 s and 1 s (datasheet vs fitted).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from worlds.drone import site_world as W

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FREE = ["CMD_DELAY", "RATE_GAIN", "TAU_RATE", "RATE_BIAS_P", "RATE_BIAS_Q", "RATE_BIAS_R", "THRUST_GAIN", "DRAG",
        "WIND0_X", "WIND0_Y", "JET_N_AY", "JET_N_SIG", "JET_N_X", "JET_N_GY", "JET_N_GZ", "JET_N_SP",
        "JET_S_AY", "JET_S_SIG", "JET_S_X", "JET_S_GY", "JET_S_GZ", "JET_S_SP"]
POS = {"TAU_RATE", "DRAG", "JET_N_SIG", "JET_S_SIG"}     # positive via softplus


def load(path, drop_end=3):
    z = np.load(path)
    if str(z["command_mode"]) != "ctbr":
        return None
    n = int(z["n_steps"]) - (drop_end if str(z["end_cause"]) in ("collision", "out_of_bounds") else 0)
    t = lambda k: torch.tensor(z[k], dtype=torch.float32, device=DEV)
    return dict(pos=t("pos")[: n + 1], vel=t("lin_vel")[: n + 1], quat=t("quat")[: n + 1], w=t("ang_vel")[: n + 1],
                a=t("actions")[:n], name=str(path).split("/")[-1], cause=str(z["end_cause"]))


def windows(lives, H, stride=1):
    """Stack all windows [i, i+H] with i >= 2 (two previous commands known)."""
    S = {k: [] for k in ("pos", "vel", "R", "w", "a_prev", "a_prev2", "acts", "tgt_pos", "tgt_vel", "tgt_w", "tgt_R", "x0")}
    for L in lives:
        n = L["a"].shape[0]
        R = W.quat_to_R(L["quat"])
        for i in range(2, n - H + 1, stride):
            S["pos"].append(L["pos"][i]); S["vel"].append(L["vel"][i]); S["R"].append(R[i]); S["w"].append(L["w"][i])
            S["a_prev"].append(L["a"][i - 1]); S["a_prev2"].append(L["a"][i - 2]); S["acts"].append(L["a"][i:i + H])
            S["tgt_pos"].append(L["pos"][i + 1:i + H + 1]); S["tgt_vel"].append(L["vel"][i + 1:i + H + 1])
            S["tgt_w"].append(L["w"][i + 1:i + H + 1]); S["tgt_R"].append(R[i + 1:i + H + 1]); S["x0"].append(L["pos"][i, 0])
    return {k: torch.stack(v) for k, v in S.items()}


def rollout(prm, Wd, H):
    B = Wd["pos"].shape[0]
    T_prev = Wd["a_prev"][:, 3].clamp(0, 1)
    s = dict(pos=Wd["pos"], vel=Wd["vel"], R=Wd["R"], w=Wd["w"], f=W.G * prm["THRUST_GAIN"] * (T_prev / 0.3) ** 2,
             prev_a=Wd["a_prev"], prev_a2=Wd["a_prev2"], gust=torch.zeros(B, 3, device=DEV))
    P, V, Wr, Rr = [], [], [], []
    for t in range(H):
        s = W.dyn_step(s, Wd["acts"][:, t], prm)
        P.append(s["pos"]); V.append(s["vel"]); Wr.append(s["w"]); Rr.append(s["R"])
    return torch.stack(P, 1), torch.stack(V, 1), torch.stack(Wr, 1), torch.stack(Rr, 1)


def loss_fn(prm, Wd, H):
    P, V, Wr, Rr = rollout(prm, Wd, H)
    lp = ((P - Wd["tgt_pos"]) ** 2).sum(-1).mean() / 0.05 ** 2
    lv = ((V - Wd["tgt_vel"]) ** 2).sum(-1).mean() / 0.1 ** 2
    lw = ((Wr - Wd["tgt_w"]) ** 2).sum(-1).mean() / 0.1 ** 2
    lr = ((Rr - Wd["tgt_R"]) ** 2).sum((-1, -2)).mean() / 0.02 ** 2
    return lp + lv + lw + lr


@torch.no_grad()
def errors(prm, lives, steps=(15, 30)):
    """Open-loop errors after k steps (pos m, vel m/s), median over windows, per life."""
    out = {}
    for L in lives:
        H = max(steps)
        if L["a"].shape[0] < H + 3:
            continue
        Wd = windows([L], H)
        P, V, _, _ = rollout(prm, Wd, H)
        out[L["name"]] = {f"{k / 30:.1f}s": dict(pos_err_m_median=float((P[:, k - 1] - Wd["tgt_pos"][:, k - 1]).norm(dim=-1).median()),
                                                pos_err_m_mean=float((P[:, k - 1] - Wd["tgt_pos"][:, k - 1]).norm(dim=-1).mean()),
                                                vel_err_ms_median=float((V[:, k - 1] - Wd["tgt_vel"][:, k - 1]).norm(dim=-1).median()),
                                                vel_err_ms_mean=float((V[:, k - 1] - Wd["tgt_vel"][:, k - 1]).norm(dim=-1).mean()))
                          for k in steps}
        out[L["name"]]["n_windows"] = int(Wd["pos"].shape[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lives", nargs="+", required=True)
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--free", nargs="*", default=None)
    ap.add_argument("--init", default=None, help="json of starting parameter values")
    ap.add_argument("--set", nargs="*", default=[], help="K=V overrides of the starting values")
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lives = [l for l in (load(p) for p in args.lives) if l is not None]
    hold = [l for l in (load(p) for p in args.holdout) if l is not None]
    base = dict(W.DATASHEET)
    if args.init:
        j = json.load(open(args.init))
        base.update(j.get("full", j))
    for kv in args.set:
        k, v = kv.split("=")
        base[k] = float(v)
    free = args.free if args.free is not None else FREE
    raw = {}
    for k in free:
        v = float(base[k])
        if k in POS:
            v = math.log(math.expm1(max(v, 1e-3)))
        elif k == "CMD_DELAY":
            v = math.log((v / 2 + 1e-3) / (1 - v / 2 + 1e-3))
        raw[k] = torch.tensor(v, device=DEV, requires_grad=True)

    def build():
        prm = dict(base)
        for k, r in raw.items():
            prm[k] = torch.nn.functional.softplus(r) if k in POS else (2 * torch.sigmoid(r) if k == "CMD_DELAY" else r)
        return prm

    Wd = windows(lives, args.H)
    opt = torch.optim.Adam(list(raw.values()) or [torch.zeros(1, requires_grad=True)], lr=0.03)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    l0 = float(loss_fn(build(), Wd, args.H))
    for it in range(args.iters):
        opt.zero_grad()
        L = loss_fn(build(), Wd, args.H)
        if raw:
            L.backward()
        opt.step(); sched.step()
        if it % 50 == 0 or it == args.iters - 1:
            print(it, f"{float(L):.2f}", {k: round(float(v), 4) for k, v in build().items() if k in raw})
    fitted = {k: float(v) for k, v in build().items() if k in raw}
    prm_f = dict(base); prm_f.update(fitted)
    res = dict(lives=[l["name"] for l in lives], holdout=[l["name"] for l in hold], H=args.H, loss_init=l0,
               loss_final=float(loss_fn(prm_f, Wd, args.H)), fitted=fitted, full=prm_f,
               err_datasheet=errors(dict(W.DATASHEET), lives + hold), err_fitted=errors(prm_f, lives + hold))
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("loss_init", "loss_final", "fitted")}, indent=1))
    for nm in res["err_fitted"]:
        print(nm, "datasheet", {k: round(v["pos_err_m_median"], 3) for k, v in res["err_datasheet"][nm].items() if isinstance(v, dict)},
              "fitted", {k: round(v["pos_err_m_median"], 3) for k, v in res["err_fitted"][nm].items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
