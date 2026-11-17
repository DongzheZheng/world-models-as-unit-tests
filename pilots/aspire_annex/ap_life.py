"""Fly ONE calibration life:  python pilots/aspire_annex/ap_life.py <version> <seed> [pilot] [kwargs-json]"""
import json, sys

if __name__ == "__main__":
    from reallab.annex.parallel import evaluate
    ver, seed = sys.argv[1], int(sys.argv[2])
    pilot = sys.argv[3] if len(sys.argv) > 3 else "aspire_style"
    kw = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    assert seed < 1000
    res = evaluate(f"pilots.aspire_annex.policy_{ver}:make_policy", kw, seeds=[seed], purpose="calibration",
                   pilot=pilot, tag=f"{ver}", n_workers=1)
    print(json.dumps(res))
