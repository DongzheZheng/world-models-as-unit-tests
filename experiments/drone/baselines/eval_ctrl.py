"""Evaluate one controller on mission A, 50 real eval episodes (seeds 1000-1049), and record it."""
import argparse, json, os
from reallab.drone.parallel import evaluate
from cawm import results

def run(factory, kwargs, controller, family, real_steps, train_wall_s, notes, n_workers=4, seeds=range(1000, 1050), pilot="ppo_real",
        tag=None, log=True):
    logdir = f"runs/drone/baselines_rl/evlogs/{controller}" if log else None
    rows = evaluate(factory, kwargs, "site", list(seeds), purpose="eval", tag=tag or controller, n_workers=n_workers,
                    log_dir=logdir, pilot=pilot, mission="A")
    for r in rows:
        r["success"] = bool(r.get("success"))
        r["time_success_s"] = r["time_s"] if r["success"] else None
    rec = results.record("drone", "site_A", controller, family, "real", rows, "success",
                         other_metrics=("time_success_s", "relays_reached", "collision", "out_of_bounds", "timeout"),
                         real_steps_training=int(real_steps), train_wall_s=float(train_wall_s), notes=notes)
    ts = [r["time_success_s"] for r in rows if r["success"]]
    out = dict(controller=controller, success=rec["eval"]["mean"], n_success=len(ts), mean_time=(sum(ts) / len(ts) if ts else None),
               relays=rec["other"]["relays_reached"]["mean"], collision=rec["other"]["collision"]["mean"],
               timeout=rec["other"]["timeout"]["mean"], oob=rec["other"]["out_of_bounds"]["mean"])
    print(json.dumps(out), flush=True)
    os.makedirs("runs/drone/baselines_rl/evals", exist_ok=True)
    json.dump(dict(summary=out, rows=rows), open(f"runs/drone/baselines_rl/evals/{controller}.json", "w"))
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt"); p.add_argument("--controller"); p.add_argument("--real_steps", type=int)
    p.add_argument("--wall", type=float, default=0.0); p.add_argument("--notes", default=""); p.add_argument("--workers", type=int, default=4); p.add_argument("--kind", default="ppo")
    a = p.parse_args()
    NOTES = {"ppo_real_5lives": "PPO updated after each of the 5 real calibration lives (seeds 0-4), 40 epochs per update on all real data so far, deterministic mean action at eval; 212 real transitions in total"}
    if not a.notes:
        a.notes = NOTES.get(a.controller, "PPO trained directly in the real site (purpose=oracle_train), 16 parallel real workers, deterministic mean action at eval")
    if a.kind == "wm":
        run("experiments.drone.baselines.wm:wm_policy", dict(ckpt=a.ckpt), a.controller, "neural_wm", a.real_steps, a.wall,
            "Ensemble of 5 MLPs (state-delta) trained only on the 5 black-box lives (life 1 stabiliser+dither exploration, lives 2-5 MPPI); MPPI H=30, 384 samples, novelty penalty 0.5 + tilt limit 0.5 rad + public-geometry collision checks",
            a.workers, pilot="neural_wm")
    else:
        run("experiments.drone.baselines.common:rl_policy", dict(ckpt=a.ckpt), a.controller, "model_free_rl", a.real_steps, a.wall,
            a.notes, a.workers)
