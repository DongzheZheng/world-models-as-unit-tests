"""Write runs/drone/code_wm/{diagnosis,fidelity,summary}.json from the fits, tuning runs, lives and evaluation rows.
   python experiments/drone/code_wm/make_outputs.py <outputs_cfg.json>"""
import json
import os
import sys
from pathlib import Path

import numpy as np

O = Path("runs/drone/code_wm")
BB = Path("runs/drone/blackbox")


def jl(p):
    return json.load(open(p))


def life_rows(pilot, purposes):
    out = []
    for p in sorted((BB / pilot).glob("life_*.npz"), key=lambda q: int(q.stem.split("_")[1])):
        z = np.load(p)
        k = int(z["life"])
        out.append(dict(pilot=pilot, life=k, seed=int(z["seed"]), command_mode=str(z["command_mode"]),
                        real_steps=int(z["n_steps"]), end_cause=str(z["end_cause"]), relays_reached=int(z["relays_reached"]),
                        time_s=round(int(z["n_steps"]) / 30.0, 3), purpose=purposes.get(f"{pilot}/{k}", ""),
                        gif=f"runs/drone/code_wm/gifs/{pilot}_life_{k}.gif"))
    return out


def ends(rows):
    c = {"success": 0, "collision": 0, "out_of_bounds": 0, "timeout": 0}
    for r in rows:
        for k in c:
            if r.get(k):
                c[k] += 1
                break
    return c


def main():
    cfg = jl(sys.argv[1])
    lives = []
    for pilot in cfg["pilots"]:
        lives += life_rows(pilot, cfg["purposes"])
    total = sum(l["real_steps"] for l in lives)
    # ------------------------------------------------------------------ fidelity
    fid = dict(metric="open-loop replay of the logged commands from every logged state (windows start at each step); "
                      "median over windows of |position error| [m] and |velocity error| [m/s] after 0.5 s and 1.0 s",
               note="'held_out' = world fitted only on earlier lives, scored on a life it never saw",
               comparisons=[])
    for name, path, desc in cfg["fidelity_files"]:
        d = jl(O / path)
        for life in d["err_fitted"]:
            fid["comparisons"].append(dict(
                world=name, world_fitted_on=d["lives"], life=life, held_out=life in d["holdout"], description=desc,
                datasheet={k: v for k, v in d["err_datasheet"][life].items()}, code_world={k: v for k, v in d["err_fitted"][life].items()}))
    json.dump(fid, open(O / "fidelity.json", "w"), indent=1)
    # ------------------------------------------------------------------ diagnosis
    final = jl(O / cfg["final_world"])
    diag = dict(world_file="worlds/drone/site_world.py", fitted_on=final["lives"], fit_loss_datasheet_vs_final=None,
                method="gradient descent (Adam) through dyn_step on 20-step (0.67 s) open-loop windows of every logged "
                       "state; loss on position, velocity, body rates and attitude",
                mechanisms=cfg["mechanisms"], final_parameters=final["full"])
    json.dump(diag, open(O / "diagnosis.json", "w"), indent=1)
    # ------------------------------------------------------------------ summary
    tj = jl(O / cfg["tune_json"])
    summ = dict(lives_used=len(lives), lives=lives, real_steps_total=total,
                site_world_py_bytes=os.path.getsize("worlds/drone/site_world.py"),
                controller_type=cfg["controller_type"],
                synthesis=dict(tuning_file=cfg["tune_json"], wm_steps=tj["wm_steps"], wall_s=round(tj["wall_s"], 1),
                               wm_steps_per_s=round(tj["wm_steps_per_s"]), all_synthesis_runs=cfg.get("all_tunes", {}),
                               code_world_success=tj["finals"][tj["pick"]]["success"],
                               code_world_time_success_s=tj["finals"][tj["pick"]]["t_success"],
                               code_world_end_causes={k: tj["finals"][tj["pick"]][k] for k in ("success", "collision", "oob", "timeout")},
                               controller=tj["controller"]),
                gifs=sorted(str(p) for p in (O / "gifs").glob("*.gif")))
    for tag, rows_path in cfg.get("evals", {}).items():
        if not (O / rows_path).exists():
            continue
        rows = jl(O / rows_path)
        ts = [r["time_s"] for r in rows if r["success"]]
        summ[f"real_eval_{tag}"] = dict(episodes=len(rows), seeds=[rows[0]["seed"], rows[-1]["seed"]],
                                        success_rate=sum(bool(r["success"]) for r in rows) / len(rows),
                                        mean_time_success_s=float(np.mean(ts)) if ts else None,
                                        end_causes=ends(rows), mean_relays_reached=float(np.mean([r["relays_reached"] for r in rows])))
    summ.update(cfg.get("extra", {}))
    json.dump(summ, open(O / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "lives"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
