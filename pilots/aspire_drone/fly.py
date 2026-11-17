"""Fly ONE real life (costs a life) and render its GIF.
   python pilots/aspire_drone/fly.py <version> <seed> [ctbr|position] [pilot] [policy_kwargs_json] [mission]"""
import json
import sys
from pathlib import Path


def main():
    from reallab.drone.parallel import evaluate
    from reallab.drone.render import render_life_gif
    ver, seed = sys.argv[1], int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "ctbr"
    pilot = sys.argv[4] if len(sys.argv) > 4 else "aspire_style"
    kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    mission = sys.argv[6] if len(sys.argv) > 6 else "A"
    assert seed < 1000
    rows = evaluate(f"pilots.aspire_drone.policy_{ver}:make_policy", kw, "site", seeds=[seed], purpose="calibration",
                    tag=f"aspire_style_{ver}", pilot=pilot, mission=mission, command_mode=mode)
    print(json.dumps(rows, default=str))
    bb = sorted(Path(f"runs/drone/blackbox/{pilot}").glob("life_*.npz"), key=lambda p: p.stat().st_mtime)[-1]
    out = Path("runs/drone/aspire_style") / f"{pilot}_{bb.stem}.gif"
    render_life_gif(str(bb), str(out))
    print("blackbox", bb, "gif", out)


if __name__ == "__main__":
    main()
