"""Real evaluation (once) of the program chosen in the code world: python pilots/aspire_pusht/evaluate_cw.py <version> <notes_file>"""
import json
import sys

if __name__ == "__main__":
    from reallab.pusht.parallel import evaluate
    from cawm.results import record
    ver = sys.argv[1]; notes = open(sys.argv[2]).read().strip()
    scr = [json.loads(l) for l in open("runs/pusht/aspire_style/cw_screen.jsonl")]
    wm_steps = sum(r["imagined_steps"] for r in scr) + (120 + 19) * 300      # + upper bound for failure-listing/trace runs
    wall = sum(r["wall_s"] for r in scr)
    out = "runs/pusht/aspire_style/eval_L_cap_in_code_world"
    eps = evaluate(f"pilots.aspire_pusht.policy_v{ver}:make_policy", {}, list(range(1000, 1050)), purpose="eval",
                   pilot="cap_in_world", tag="L_cap_in_code_world", n_workers=12, log_dir=out, record_first=3,
                   video_dir="runs/pusht/aspire_style/gifs/L_cap_in_code_world_eval", object="L")
    rec = record(demo="pusht", scenario="pusht_L", controller="cap_in_code_world", family="code_as_policy_in_code_world",
                 trained_in="code_world", episodes=eps, metric="success", other_metrics=("max_coverage", "final_coverage", "n_steps"),
                 real_steps_calibration=600, real_steps_training=2367, wm_steps=int(wm_steps), train_wall_s=float(wall), notes=notes)
    json.dump(eps, open(f"{out}/episodes.json", "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("eval", "other", "wm_steps", "train_wall_s")}, indent=1))
