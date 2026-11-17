"""FINAL evaluation, run once: 100 real episodes, seeds 1000-1099, purpose='eval', with the controller chosen by
code-world score (runs/doom/code_wm/selection_v3world.json).  Writes runs/doom/code_wm/eval.json and the
cawm.results record.  usage: python experiments/doom/code_wm/final_eval.py <wm_steps> <train_wall_s>"""
import gzip
import json
import sys
import time
from pathlib import Path

from cawm import results
from reallab.doom.parallel import evaluate

if __name__ == "__main__":
    wm_steps, wall = int(float(sys.argv[1])), float(sys.argv[2])
    out = Path("runs/doom/code_wm")
    assert not (out / "eval.json").exists(), "final evaluation already done; never rerun"
    kw = dict(horizon=24, margin=2.0, w_center=0.3)
    cal = 0
    for p in sorted(Path("runs/doom/blackbox/take_cover/code_wm").glob("life_*.jsonl.gz")):
        cal += sum(1 for _ in gzip.open(p, "rt")) - 2
    t0 = time.time()
    eps = evaluate("experiments.doom.code_wm.planner:planner", kw, "take_cover", list(range(1000, 1100)),
                   purpose="eval", pilot="code_wm", tag="final", n_workers=50, record_first=1,
                   record_format="gif", video_dir=str(out / "gifs"))
    (out / "gifs" / "ep_1000.gif").rename(out / "gifs" / "eval_seed1000.gif")
    rec = results.record(demo="doom", scenario="take_cover", controller="code_wm", family="code_as_wm",
                         trained_in="code_world", episodes=eps, metric="ret", real_steps_calibration=cal,
                         real_steps_training=0, wm_steps=wm_steps, train_wall_s=wall,
                         other_metrics=("length_tics", "n_steps"),
                         notes="code world worlds/fps/take_cover_world.py (fit to 4 lives); controller = hand-written "
                               "feature-space planner (24-tic lookahead over 18 strafe plans on remembered fireballs), "
                               "parameters and choice over PPO candidates by code-world score only")
    (out / "eval.json").write_text(json.dumps(dict(episodes=eps, record=rec, eval_wall_s=time.time() - t0,
                                                   controller=kw), indent=1))
    print(json.dumps(rec["eval"]), "cal steps", cal)
