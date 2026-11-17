"""Fly ONE calibration life.  usage: python experiments/annex/code_wm/fly_life.py <seed> <factory> [json kwargs] [pilot]"""
import json
import sys

if __name__ == "__main__":
    from reallab.annex.parallel import evaluate
    seed = int(sys.argv[1])
    factory = sys.argv[2]
    kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    pilot = sys.argv[4] if len(sys.argv) > 4 else "code_wm"
    assert seed < 1000
    res = evaluate(factory, kw, seeds=[seed], purpose="calibration", pilot=pilot, n_workers=1)
    print(json.dumps(res))
