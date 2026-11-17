"""Add the cap_in_code_world block to runs/pusht/aspire_style/summary.json."""
import json
import os
from pilots.aspire_pusht.make_summary import evalstats

if __name__ == "__main__":
    S = json.load(open("runs/pusht/aspire_style/summary.json"))
    scr = [json.loads(l) for l in open("runs/pusht/aspire_style/cw_screen.jsonl")]
    rows = [dict(name=r["name"], program=f"policy_v{r['version']}.py", overrides=r["kwargs"], imagined_episodes=r["n"],
                 seeds=f"{r['seed0']}..{r['seed0'] + r['n'] - 1}", success=round(r["success"], 3),
                 mean_max_coverage=round(r["max_coverage"], 3), success_away_from_walls=round(r["success_away_from_walls"], 3),
                 success_near_wall=r["success_near_wall"] and round(r["success_near_wall"], 3),
                 by_initial_angle_error=r["by_initial_angle_error"]) for r in scr]
    n_eps = sum(r["n"] for r in scr) + 120 + 19
    S["cap_in_code_world"] = dict(
        world="worlds/pusht/pusht_world.py via experiments/pusht/code_wm/screen.run_in_world (object L)",
        screening=rows,
        reference_on_heldout_seeds_74000=[x for x in rows if x["seeds"].startswith("74000")],
        submitted=dict(program="pilots/aspire_pusht/policy_v17.py", bytes=os.path.getsize("pilots/aspire_pusht/policy_v17.py"),
                       code_world_prediction=dict(success=0.840, mean_max_coverage=0.866, seeds="74000..74479 (held out)"),
                       changes_vs_v15=["S10c removed (precision depth back to push units)", "k_depth 0.5 -> 0.8, d_max 40 -> 60 on the L",
                                       "coast factor 1.7 -> 1.0 on the L", "S14 approach-stall watchdog"]),
        what_the_world_showed=[
            "S10c (adopted after one real flight) was harmful in aggregate: v15 37.7% vs v14-rules 60.4% (480 imagined episodes); it slowed the end of long episodes (5% vs 56% success at initial |eth| 90-135 deg). One real flight per change could not reveal this.",
            "Failures were dominated by time: success fell from ~0.9 at initial |eth| < 45 deg to 0.0-0.2 above 135 deg; pushing faster (k_depth 0.8, d_max 60) was the one lever that moved every band (60% -> 79%).",
            "Many single-knob changes (COM, g, commitment, translation weight, thresholds) were within noise (+-0.03 at n=240); the world's COM (23.4, 55.6) did not help my program.",
            "0-coverage failures near walls came from a navigation livelock (pusher pinned at the clipped wall line); the orbit-direction fix hurt (81.5%), only a stall watchdog was kept."],
        distrust=[
            "Near-wall starts (block within 40 u of a wall): the world makes the wall a hard stop; success there 0.67-0.71 vs 0.92-0.94 away from walls; I did not tune for wall behaviour and kept only a wall-independent watchdog.",
            "Faster pushes extrapolate: in contact, imagined v17 pusher steps have median 12 / p90 25 / max 51 u per step vs 5.6 / 16 / 37 in my 8 real L lives.",
            "Selection bias: ~45 candidates on the same 240/480 seeds; the submitted program was re-scored on held-out imagined seeds 74000-74479."],
        imagined_episodes=n_eps, wm_steps_recorded=4496746, screening_wall_s=249.0,
        real=dict(eval=evalstats("runs/pusht/aspire_style/eval_L_cap_in_code_world/episodes.json"),
                  result_file="runs/pusht/results/pusht_L__cap_in_code_world.json",
                  gifs="runs/pusht/aspire_style/gifs/L_cap_in_code_world_eval/ep_1000.gif .. ep_1002.gif"),
    )
    json.dump(S, open("runs/pusht/aspire_style/summary.json", "w"), indent=1)
    print(json.dumps(S["cap_in_code_world"]["real"], indent=1), n_eps)
