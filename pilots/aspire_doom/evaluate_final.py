"""Official evaluation (once): python pilots/aspire_doom/evaluate_final.py <version> <controller> <real_steps_training> <notes>"""
import json
import sys
from pathlib import Path

from reallab.doom.parallel import evaluate
from cawm.results import record


def main():
    ver, controller, steps, notes = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    out = Path("runs/doom/aspire_style")
    res = evaluate(f"pilots.aspire_doom.policy_{ver}:make_policy", {}, scenario="take_cover",
                   seeds=list(range(1000, 1100)), purpose="eval", pilot="aspire_style", record_first=1,
                   record_format="gif", video_dir=str(out / "gifs"), n_workers=32)
    g = out / "gifs" / "ep_1000.gif"
    if g.exists():
        g.rename(g.with_name(f"{controller}_eval_seed1000_{ver}.gif"))
    (out / f"eval_{controller}.json").write_text(json.dumps(res))
    rec = record(demo="doom", scenario="take_cover", controller=controller, family="code_as_policy",
                 trained_in="real", episodes=res, metric="ret", real_steps_calibration=0,
                 real_steps_training=steps, notes=notes)
    print(json.dumps(rec["eval"]))


if __name__ == "__main__":
    main()
