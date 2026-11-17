"""Fly one calibration life: python pilots/aspire_pusht/fly.py <version> <pilot> <seed> [object]"""
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    from reallab.pusht.parallel import evaluate
    from reallab.pusht.render import render_life
    ver, pilot, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    obj = sys.argv[4] if len(sys.argv) > 4 else "T"
    res = evaluate(f"pilots.aspire_pusht.policy_v{ver}:make_policy", {}, [seed], purpose="calibration", pilot=pilot,
                   n_workers=1, object=obj)
    print(json.dumps(res))
    bbdir = "blackbox" if obj == "T" else f"blackbox_{obj}"
    bb = sorted(Path(f"runs/pusht/{bbdir}/{pilot}").glob("life_*.npz"), key=lambda p: p.stat().st_mtime)[-1]
    tag = "" if obj == "T" else f"{obj}_"
    out = Path(f"runs/pusht/aspire_style/gifs/{tag}{pilot}_{bb.stem}_v{ver}_seed{seed}.gif")
    render_life(str(bb), str(out))
    print("life file", bb, "gif", out)
