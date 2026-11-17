"""python -m reallab.doom.make_gif --scenario take_cover --seed 1000 --out runs/doom/gifs/take_cover_random.gif"""
import argparse
import shutil
import tempfile
from pathlib import Path

from reallab.doom.parallel import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="take_cover")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--policy", default="reallab.doom.parallel:random_policy")
    ap.add_argument("--tag", default="gif_random")
    ap.add_argument("--purpose", default="eval")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        r = evaluate(a.policy, {"seed": 0} if "random" in a.policy else {}, a.scenario, [a.seed], purpose=a.purpose,
                     tag=a.tag, n_workers=1, record_first=1, record_format="gif", video_dir=d)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(d) / f"ep_{a.seed}.gif", a.out)
    print(r[0])
