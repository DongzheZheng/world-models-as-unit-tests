"""Official evaluation (once): final program, seeds 1000-1049, purpose eval, mission A; record + GIF of episode 1.
   python pilots/aspire_drone/evaluate_final.py <version> <controller_name> <real_steps_training> [mission A|B]"""
import json
import sys
from pathlib import Path

NOTES = {
    "aspire_style": "ASPIRE-style code-as-policy: policy_v5 (ctbr), refined over 5 real lives (v1 crash, v2-v5 success); "
                    "rate-bias feed-forward, identified thrust constant, online disturbance observer + map-indexed priors "
                    "for the two side-aisle crosswind jets. real_steps_training = all steps of the 5 official lives.",
    "aspire_style_B": "Phase 2, mission B, zero B flights: policy_v6 = v5 (validated on A) + route planner that replaces "
                      "wall-crossing relay segments by L-shaped legs via a corner (R1->corner in canyon->R2 up the north "
                      "aisle->same corner->R3), turn-dependent speed reference (1.5 m/s at 90-deg corners, 0.5 at the R2 "
                      "reversal, 3.5 m/s cap off-canyon), north-jet profile assumed to continue into the aisle at 70%, "
                      "a_max 16->24 m/s^2, observer clip 20, generic box-distance wall push. Lives used: 5 (all mission A).",
    "aspire_style_C": "Phase 3, mission C (other aircraft): policy_v11 = v6 flight core + constant-velocity CPA "
                      "avoidance (velocity-level escape, larger keep-out from hovering 'darters', vertical preference), relay "
                      "funnel + missed-relay recovery, altitude band and vertical velocity barrier. 5 mission-C lives "
                      "(v7 success, v8 out-of-bounds after missing relay 1, v9 success, v10 success x2); v11 adds the "
                      "vertical barrier after life 5 and was NOT flown before this evaluation. real_steps_training = "
                      "steps of the 5 mission-C lives.",
    "aspire_style_extended_C": "Phase 3 extended run, mission C: policy_v14, same lineage as the official v11 (49/100) "
                               "continued over 15 extra lives (x1: v11, x2: v12, x3: v13 x2 + v14 x3). Total mission-C "
                               "lives: 20 (5 official + 15 extended); real_steps_training = steps of all 20 lives. v14 adds "
                               "hover/moving-dependent emergency law, relay funnel 0.5+0.2d, slow missed-relay recovery, "
                               "wall-face velocity barrier with attitude lag, no speed cut near aircraft, moving keep-out 1.2 m.",
}


def main():
    from cawm.results import record
    from reallab.drone.parallel import evaluate
    from reallab.drone.render import render_episode_gif
    ver, controller, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    mission = sys.argv[4] if len(sys.argv) > 4 else "A"
    notes = NOTES[controller if mission == "A" else f"{controller}_{mission}"]
    tagm = "" if mission == "A" else f"_{mission}"
    out = Path("runs/drone/aspire_style")
    log_dir = out / f"eval_logs_{controller}{tagm}"
    seeds = list(range(1000, 1100 if mission == "C" else 1050))
    rows = evaluate(f"pilots.aspire_drone.policy_{ver}:make_policy", {}, "site", seeds=seeds, purpose="eval",
                    tag=f"{controller}_eval{tagm}", n_workers=16, log_dir=str(log_dir), pilot="aspire_style", mission=mission,
                    command_mode="ctbr")
    for r in rows:
        r["time_success_s"] = r["time_s"] if r["success"] else None
    (out / f"eval_{controller}{tagm}.json").write_text(json.dumps(rows, indent=1, default=str))
    rec = record(demo="drone", scenario=f"site_{mission}", controller=controller, family="code_as_policy", trained_in="real",
                 episodes=rows, metric="success",
                 other_metrics=("time_success_s", "relays_reached", "collision", "out_of_bounds", "timeout"),
                 real_steps_calibration=0, real_steps_training=steps, notes=notes)
    print(json.dumps({k: rec[k] for k in ("eval", "other")}, indent=1))
    first = log_dir / "site_eval_1000.jsonl.gz"
    render_episode_gif(str(first), str(out / f"{controller}{tagm}_eval_seed1000.gif"),
                       title=f"{controller} ({ver}) - eval seed 1000, mission {mission}")
    for r in rows:
        print(r["seed"], r["success"], round(r["time_s"], 2), r["relays_reached"], r["collision"], r["out_of_bounds"],
              r["timeout"], round(r["min_range_seen"], 2))


if __name__ == "__main__":
    main()
