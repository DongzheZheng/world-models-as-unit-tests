"""Assemble runs/pusht/code_wm/{diagnosis,fidelity,summary}.json from the fit / fidelity / tuning / eval files.
python experiments/pusht/code_wm/make_reports.py <cfg.json>   (cfg: lives purposes, chosen tuning file, eval files)"""
import json
import sys
from pathlib import Path

import numpy as np

from experiments.pusht.code_wm import fidelity as FID
from worlds.pusht import pusht_world as W

O = Path("runs/pusht/code_wm")
BB = Path("runs/pusht/blackbox/code_wm")


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    P = W.PARAMS
    f1, f2, f12 = J(O / "fit_l1.json"), J(O / "fit_l2.json"), J(O / "fit_l12.json")
    per = {k: [f1["params"][k], f2["params"][k], f12["params"][k]] for k in f12["params"]}
    diag = dict(
        world_file="worlds/pusht/pusht_world.py",
        fitted_on=["life_1 (probe)", "life_2 (probe)"],
        method="cross-entropy search over parameters minimising 10-step open-loop replay error of block pose "
               "(pos + 60 u/rad * angle) on all segments where the block moves; pusher gains fitted first on "
               "5-step pusher replays. Values per fit: [life1 only, life2 only, lives 1+2].",
        mechanisms=[
            dict(name="M1 pusher_pd", params=dict(kp=P["kp"], kv=P["kv"]), per_fit=dict(kp=per["kp"], kv=per["kv"]),
                 evidence="step-command and ramp responses in life 1 (steps 0-67); replay error 0.01-0.02 u. Pusher "
                          "velocity is not changed by contact (life 1 steps 68-80 constant 28.7,-22.5 u/s) -> "
                          "the pusher is kinematic w.r.t. the block.", confidence="high"),
            dict(name="M3 centre_of_mass (UNUSUAL)", params=dict(com_u=P["com_u"], com_v=P["com_v"]),
                 per_fit=dict(com_u=per["com_u"], com_v=per["com_v"]),
                 drawing_centroid=[0.0, 40.714],
                 evidence="axial push on the stem tip turned the block -18 deg in 13 steps (life 1, steps 68-81); "
                          "independent fits on life 1 and life 2 agree within 1 u; replay loss with the COM forced "
                          "to the drawing centroid is 7x larger (life 1: %.2f vs %.2f)."
                          % (f1["block_loss_com_at_centroid"], f1["block_loss"]),
                 confidence="high",
                 conclusion="the block's mass is concentrated toward the right end of the bar (+u) and lower than "
                            "the drawn centroid: COM ~ (24, 30) instead of (0, 40.7). Pushes through the drawn "
                            "centroid therefore rotate the block (counter-intuitive to a planner using the drawing)."),
            dict(name="M3 radius_of_gyration", params=dict(k_gyr=P["k_gyr"]), per_fit=dict(k_gyr=per["k_gyr"]),
                 evidence="rotation rate vs translation in contact segments; spread 50-56 across fits",
                 confidence="medium"),
            dict(name="M2 contact", params=dict(restitution=P["restitution"], mu=P["mu"], bias=P["bias"]),
                 per_fit=dict(restitution=per["restitution"], mu=per["mu"], bias=per["bias"]),
                 evidence="contact friction converges to ~0 in every fit (the pusher slides along faces); "
                          "restitution small and poorly determined (0-0.1)", confidence="medium (mu), low (e)"),
            dict(name="M4 table_friction", params={k: P[k] for k in ("lin_damp", "ang_damp", "lin_coul", "ang_coul")},
                 per_fit={k: per[k] for k in ("lin_damp", "ang_damp", "lin_coul", "ang_coul")},
                 evidence="release-and-watch phases (life 1 steps 81-90 etc.): the block keeps sliding after the "
                          "pusher leaves, displacement per step shrinking geometrically (~x0.63/step) -> mostly "
                          "viscous decay (~4.6/s), small Coulomb part", confidence="high (viscous), low (Coulomb)"),
            dict(name="M5 walls", params=dict(inner=[W.WALL_LO, W.WALL_HI]),
                 evidence="task_card only; never exercised in the lives (no wall contact observed) -> crude clamp",
                 confidence="low"),
        ],
        unusual=("The block's centre of mass is offset ~24 u to the +u side of the symmetry axis and ~11 u below the "
                 "drawn centroid; it also coasts after release (inertia + viscous floor friction) and the pusher-block "
                 "contact is nearly frictionless."),
        uncertainty="CEM elite spreads are over-confident; the spread across the three independent fits is the honest "
                    "uncertainty (see per_fit). Not represented stochastically in the planner (the open-loop error "
                    "is ~0.6 u / 0.4 deg after 20 steps).",
    )
    (O / "diagnosis.json").write_text(json.dumps(diag, indent=1))

    # fidelity: world (fit lives 1+2) on unseen lives; plus cross-fits between the two probe lives
    fid = dict(definition="open-loop replay of the logged pusher commands from a logged state (block twist from "
                          "the previous logged pose); errors of block position [u] and angle [deg] after h steps, over "
                          "all windows in which the block moves; static_baseline = 'block does not move'.")
    fid["world_fit_l12_on_unseen_controller_lives"] = dict(
        lives=cfg["controller_lives"], **FID.evaluate(dict(P), [str(BB / f"life_{k}.npz") for k in cfg["controller_lives"]]))
    fid["fit_l1_on_l2"] = J(O / "fid_l1_on_l2.json")
    fid["fit_l2_on_l1"] = J(O / "fid_l2_on_l1.json")
    (O / "fidelity.json").write_text(json.dumps(fid, indent=1))

    lives = []
    total = 0
    for k, purpose in cfg["lives"].items():
        d = np.load(BB / f"life_{k}.npz")
        n = int(d["n_steps"]); total += n
        lives.append(dict(life=int(k), seed=int(d["seed"]), purpose=purpose, n_steps=n, success=bool(d["success"]),
                          max_coverage=float(d["max_coverage"]), gif=f"runs/pusht/code_wm/gifs/code_wm_life_{k}.gif"))
    tunes = [J(p) for p in cfg["tuning_files"]]
    wm_steps = sum(t["wm_steps"] for t in tunes if t)
    wall = sum(t["wall_s"] for t in tunes if t)
    chosen = J(cfg["chosen"])
    ev = J(O / "eval_code_wm.json")
    summ = dict(
        lives_used=len(lives), lives=lives, real_steps_total=total,
        world_file_bytes=Path("worlds/pusht/pusht_world.py").stat().st_size,
        controller=dict(type="push-primitive sampling MPC (random shooting + 2 CEM refinements) in the code world",
                        settings=chosen["planner"] if chosen else None),
        code_world=dict(steps_controller_synthesis_and_tuning=wm_steps, wall_s=wall,
                        steps_per_s=wm_steps / wall if wall else None,
                        tuning_runs=[dict(file=f, success=t["success_rate"], n=t["n_ep"], planner=t["planner"],
                                          params_jitter=t["params_jitter"]) for f, t in zip(cfg["tuning_files"], tunes) if t],
                        final_controller_success=chosen["success_rate"] if chosen else None,
                        final_controller_n_ep=chosen["n_ep"] if chosen else None),
        real_eval=None if ev is None else dict(
            seeds="1000-1049", success_rate=float(np.mean([e["success"] for e in ev["episodes"]])),
            mean_max_coverage=float(np.mean([e["max_coverage"] for e in ev["episodes"]])),
            mean_n_steps=float(np.mean([e["n_steps"] for e in ev["episodes"]])),
            gifs="runs/pusht/code_wm/gifs/eval_code_wm/ep_1000..1002.gif", wall_s=ev["wall_s"]),
        gifs=dict(lives="runs/pusht/code_wm/gifs/code_wm_life_<k>.gif", eval="runs/pusht/code_wm/gifs/eval_code_wm/"),
    )
    (O / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
