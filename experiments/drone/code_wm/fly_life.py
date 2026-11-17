"""Fly ONE real calibration life with the code_wm Pilot and render it.
   python experiments/drone/code_wm/fly_life.py <pilot_name> <seed> <pilot kwargs .json file> [ctbr|position] [factory]"""
import json
import sys
from pathlib import Path


def main():
    from reallab.drone.parallel import evaluate
    from reallab.drone.render import render_life_gif
    pilot, seed = sys.argv[1], int(sys.argv[2])
    kw = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}   # kwargs from a JSON file (avoids shell quoting)
    mode = sys.argv[4] if len(sys.argv) > 4 else "ctbr"
    factory = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else "experiments.drone.code_wm.controller:make_pilot"
    mission = sys.argv[6] if len(sys.argv) > 6 else "A"
    assert seed < 1000
    rows = evaluate(factory, kw, "site", seeds=[seed], purpose="calibration", tag=f"code_wm life seed {seed}",
                    pilot=pilot, mission=mission, command_mode=mode)
    print(json.dumps(rows, default=str))
    bb = sorted(Path("runs/drone/blackbox", pilot).glob("life_*.npz"), key=lambda p: p.stat().st_mtime)[-1]
    out = Path("runs/drone/code_wm/gifs") / f"{pilot}_{bb.stem}.gif"
    render_life_gif(str(bb), str(out))
    print("blackbox", bb, "gif", out)


if __name__ == "__main__":
    main()
