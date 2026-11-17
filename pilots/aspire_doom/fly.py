"""Fly one real life: python pilots/aspire_doom/fly.py <version> <seed> [pilot]"""
import json
import sys
from pathlib import Path

from reallab.doom.parallel import evaluate


def main():
    ver, seed = sys.argv[1], int(sys.argv[2])
    pilot = sys.argv[3] if len(sys.argv) > 3 else "aspire_style"
    out = Path("runs/doom/aspire_style/lives") / f"{pilot}_seed{seed}_{ver}"
    res = evaluate(f"pilots.aspire_doom.policy_{ver}:make_policy", {}, scenario="take_cover", seeds=[seed],
                   purpose="calibration", pilot=pilot, record_first=1, record_format="gif",
                   video_dir="runs/doom/aspire_style/gifs", log_dir=str(out))
    src = Path(f"runs/doom/aspire_style/gifs/ep_{seed}.gif")
    if src.exists():
        src.rename(src.with_name(f"{pilot}_life_seed{seed}_{ver}.gif"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
