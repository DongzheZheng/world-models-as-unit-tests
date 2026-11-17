"""Fly ONE calibration life (metered).  usage: python experiments/doom/code_wm/fly_life.py <seed> <factory> '<kwargs json>'
The env logs the life itself to runs/doom/blackbox/take_cover/code_wm/life_<k>.jsonl.gz; GIF goes to runs/doom/code_wm/gifs/.
"""
import json
import sys
from pathlib import Path

from reallab.doom.parallel import evaluate

if __name__ == "__main__":
    seed, factory, kw = int(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3] if len(sys.argv) > 3 else "{}")
    assert seed < 1000
    out = Path("runs/doom/code_wm/gifs")
    out.mkdir(parents=True, exist_ok=True)
    res = evaluate(factory, kw, "take_cover", [seed], purpose="calibration", pilot="code_wm", tag="life",
                   record_first=1, record_format="gif", video_dir=str(out))
    print(json.dumps(res))
    with open("runs/doom/code_wm/lives.jsonl", "a") as f:
        f.write(json.dumps(dict(seed=seed, factory=factory, kwargs=kw, result=res[0])) + "\n")
