"""Add the L-block results to runs/pusht/code_wm/{summary,fidelity,diagnosis}.json (object_L blocks).
python experiments/pusht/code_wm/make_reports_L.py"""
import json
from pathlib import Path

import numpy as np

from experiments.pusht.code_wm import fidelity as FID
from worlds.pusht import pusht_world as W

O = Path("runs/pusht/code_wm")
BB = Path("runs/pusht/blackbox_L/code_wm")


def J(p):
    p = O / p
    return json.loads(p.read_text()) if p.exists() else None


def main():
    W.set_object("L")
    P = dict(W.PARAMS)
    T = W.PARAMS_T
    lives = []
    purposes = {1: "scripted probe (9 pushes on different faces at 3-7 u/step with release phases), no wall contact; "
                   "used to fit the L's mass distribution and contact friction",
                2: "end-to-end test of the T planner settings on the L world fitted on life 1; also the held-out "
                   "fidelity set"}
    total = 0
    for k in (1, 2):
        d = np.load(BB / f"life_{k}.npz")
        total += int(d["n_steps"])
        lives.append(dict(life=k, seed=int(d["seed"]), purpose=purposes[k], n_steps=int(d["n_steps"]),
                          success=bool(d["success"]), max_coverage=float(d["max_coverage"]),
                          gif=f"runs/pusht/code_wm/gifs/L_code_wm_life_{k}.gif"))
    lives[1]["outcome_note"] = ("failed (max cov 0.59): the planner pushed the block into the top wall and dragged it "
                                "along it for ~100 steps; the real block penetrated the wall by up to 25 u and was "
                                "later ejected, which the world's wall clamp cannot reproduce")
    runs = ["wmL_w0", "wmL_w1", "wmL_w1_mu2_jit", "wmL_w1_mu02_jit"] + [f"wmL_d{i}" for i in range(6)] + \
           ["wmL_final_pred", "wmL_final_robust"]
    tun = {r: J(f"{r}.json") for r in runs}
    tun = {r: t for r, t in tun.items() if t}
    ws = sum(t["wm_steps"] for t in tun.values()); wall = sum(t["wall_s"] for t in tun.values())
    pred, rob = tun.get("wmL_final_pred"), tun.get("wmL_final_robust")
    ev = J("eval_L_code_wm.json")
    real = None
    if ev:
        e = ev["episodes"]
        real = dict(seeds="1000-1049", success_rate=float(np.mean([x["success"] for x in e])),
                    mean_max_coverage=float(np.mean([x["max_coverage"] for x in e])),
                    mean_n_steps=float(np.mean([x["n_steps"] for x in e])), wall_s=ev["wall_s"],
                    gifs="runs/pusht/code_wm/gifs/eval_L_code_wm/ep_1000..1002.gif")
    fidl = dict(
        held_out_world_fit_on_life1_only=dict(note="params of the life-1 fit (mu = 5) on life 2", **{
            w: J(f"fidL_l1_on_l2_{w}.json") for w in ("clear", "near")}),
        final_world_on_life2=dict(note="final L params (mu = 0.4 was chosen using life 2's wall-free windows, so this "
                                       "is NOT held out for mu)",
                                  clear=FID.evaluate(P, [str(BB / "life_2.npz")], wall="clear"),
                                  near=FID.evaluate(P, [str(BB / "life_2.npz")], wall="near")),
        mu_scan=J("L_mu_scan.json"),
    )
    f = json.loads((O / "fidelity.json").read_text())
    f["object_L"] = fidl
    (O / "fidelity.json").write_text(json.dumps(f, indent=1))
    d = json.loads((O / "diagnosis.json").read_text())
    d["object_L"] = dict(
        carried_over_unchanged=dict(kp=P["kp"], kv=P["kv"], lin_damp=P["lin_damp"], ang_damp=P["ang_damp"],
                                    lin_coul=P["lin_coul"], ang_coul=P["ang_coul"], restitution=P["restitution"],
                                    bias=P["bias"]),
        evidence_for_carry_over="on L life 1 refitting ALL parameters lowers the 10-step replay loss only from 0.217 "
                                "(mass + mu refit, rest from the T) to 0.195; the free pusher fit gives kp 100.4, kv "
                                "20.1 (T: 100, 20)",
        refitted=dict(com_u=P["com_u"], com_v=P["com_v"], k_gyr=P["k_gyr"], mu=P["mu"]),
        drawn_centroid=list(W.CENTROID_L),
        com_offset_from_centroid=[P["com_u"] - W.CENTROID_L[0], P["com_v"] - W.CENTROID_L[1]],
        T_com_offset_from_centroid=[T["com_u"] - W.CENTROID_T[0], T["com_v"] - W.CENTROID_T[1]],
        evidence=dict(com="fits with different free sets: com_u 22.8-23.6, com_v 54.2-57.2 (L life 1, and lives 1+2 "
                          "wall-free); COM at the drawn centroid raises the loss 0.22 -> 0.79",
                      k_gyr="83-101 across fits (uniform L would be ~40): the L turns sluggishly; partly traded "
                            "against mu",
                      mu="L life 1 is flat for mu >= 0.36 and bad at 0 (0.65 vs 0.22); L life 2 wall-free windows "
                         "have a minimum at 0.36 -> 0.4, plausible 0.3-1. The T's contact was frictionless (mu ~ 0)"),
        unusual="like the T, the L's mass is off its drawn centroid (here ~14 u toward the leg side and ~13 u up), "
                "it has a large rotational inertia, and -- unlike the T -- the pusher grips it (contact friction ~0.4). "
                "'Same batch' held for the pusher and the floor, not for contact friction or mass distribution.",
        walls="M5 (hard clamp) is badly wrong: on life 2 the real block was pushed up to 25 u through the top wall "
              "and later ejected at ~25 u/step; 5-step replay error near walls 28 u mean vs 4 u (median 0.5 u) away "
              "from walls. Not fixed in the world; the planner was given a wall-clearance cost instead.",
    )
    (O / "diagnosis.json").write_text(json.dumps(d, indent=1))
    s = json.loads((O / "summary.json").read_text())
    s["object_L"] = dict(
        lives_used=2, lives=lives, real_steps_total=total,
        world_changes=["geometry from the L datasheet (boxes, outline, centroid, goal) via set_object('L')",
                       "COM, radius of gyration and contact friction refitted (see diagnosis.object_L)"],
        carried_over_from_T=["all mechanisms M1-M5 and their code", "pusher PD gains", "floor friction",
                             "restitution and penetration bias", "the primitive planner and fitting tools"],
        fitted=dict(com_u=P["com_u"], com_v=P["com_v"], k_gyr=P["k_gyr"], mu=P["mu"]),
        controller=dict(type="push-primitive sampling MPC with two-primitive lookahead, in the code world",
                        settings=pred["planner"] if pred else None,
                        changes_vs_T=["angle weight 60 -> 300 (failures in the world were 'right place, wrong angle')",
                                      "time penalty removed", "wall-clearance cost (wall_w 1)",
                                      "two-primitive lookahead (32 x 64)"]),
        code_world=dict(steps=ws, wall_s_summed=wall, steps_per_s_per_run=[t["wm_steps"] / t["wall_s"] for t in tun.values()],
                        runs={r: dict(success=t["success_rate"], n=t["n_ep"], mean_max_cov=t["mean_max_coverage"],
                                      planner=t["planner"], reality_override=t.get("reality_override"),
                                      params_jitter=t["params_jitter"]) for r, t in tun.items()},
                        note="wall_s summed over runs that partly ran concurrently; 4 early runs (world with mu = 5) "
                             "were killed and are not counted"),
        predicted_success_code_world=pred["success_rate"] if pred else None,
        predicted_success_code_world_perturbed=rob["success_rate"] if rob else None,
        real_eval=real,
    )
    (O / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s["object_L"], indent=1)[:3000])


if __name__ == "__main__":
    main()
