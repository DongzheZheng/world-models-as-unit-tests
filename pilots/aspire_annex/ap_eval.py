"""Official evaluation: python pilots/aspire_annex/ap_eval.py <version> <controller> <log_dir> <blackbox pilots comma-sep> <notes>"""
import json, sys
from pathlib import Path

if __name__ == "__main__":
    import numpy as np
    from reallab.annex.parallel import evaluate
    from cawm.results import record
    ver, controller, log_dir, pilots = sys.argv[1:5]
    NOTES = {"aspire_style": "ASPIRE-style code-as-policy, official run: 5 real lives; final program v6 = snap-back cast "
                             "(full-rate strike towards an aim point beside the post, immediate haul back, hold)",
             "aspire_style_extended": "ASPIRE-style code-as-policy, extended run continuing the same lineage",
             "aspire_style_extended_final": "ASPIRE-style code-as-policy, extended run to the full budget: 5 official + 15 extra = 20 real lives; final program v19 = snap-back cast (centroid-aligned line, 0.27 m full-rate stroke + immediate haul) with pre-straightening of a curled cord and a timed 0.08 m follow-through at turnover"}
    notes = NOTES.get(controller, controller)
    if len(sys.argv) > 5:
        notes = NOTES[sys.argv[5]]
    steps, lives = 0, 0
    for pl in pilots.split(","):
        for f in sorted(Path(f"runs/annex/blackbox/{pl}").glob("life_*.npz")):
            steps += int(np.load(f)["n_steps"]); lives += 1
    res = evaluate(f"pilots.aspire_annex.policy_{ver}:make_policy", {}, seeds=list(range(1000, 1050)), purpose="eval",
                   pilot="aspire_style", tag=controller, n_workers=6, log_dir=log_dir)
    rec = record(demo="annex", scenario="annex", controller=controller, family="code_as_policy", trained_in="real",
                 episodes=res, metric="success", other_metrics=("tip_dist", "n_steps"), real_steps_calibration=0,
                 real_steps_training=steps, notes=notes + f" | lives={lives}, real steps of lives={steps}, program=policy_{ver}.py")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / "results.json").write_text(json.dumps(res, indent=1))
    print("lives", lives, "steps", steps)
    print(json.dumps(rec, indent=1))
    td = np.array([r["tip_dist"] for r in res]); print("tip_dist pct", np.percentile(td, [10, 25, 50, 75, 90]), "succ", np.mean([r["success"] for r in res]))
