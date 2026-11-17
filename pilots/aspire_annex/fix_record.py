"""Re-write the extended-final result record with corrected notes from the saved episode list (no new episodes)."""
import json
from pathlib import Path
if __name__ == "__main__":
    from cawm.results import record
    res = json.loads(Path("runs/annex/aspire_style/eval_logs_extended_final/results.json").read_text())
    rec = record(demo="annex", scenario="annex", controller="aspire_style_extended", family="code_as_policy", trained_in="real",
                 episodes=res, metric="success", other_metrics=("tip_dist", "n_steps"), real_steps_calibration=0,
                 real_steps_training=4484,
                 notes="ASPIRE-style code-as-policy, extended run: 19 real lives in total (5 official + 14 extra: x1 4, x2 5, x3 5; "
                       "one of the 15 extra lives was left unused by a counting slip); final program policy_v19.py = snap-back cast "
                       "(centroid-aligned line, 0.27 m full-rate stroke + immediate haul) with pre-straightening of a curled cord and "
                       "a timed 0.08 m follow-through at turnover; real steps of all lives = 4484")
    print(rec["eval"], rec["notes"])
