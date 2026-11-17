"""Fly ONE calibration life: python pilots/aspire_blind/fly_life.py <policy_module> <seed> [pilot]"""
import json
import sys

if __name__ == "__main__":
    from reallab.station import parallel
    mod, seed = sys.argv[1], int(sys.argv[2])
    pilot = sys.argv[3] if len(sys.argv) > 3 else "aspire_blind"
    assert seed < 1000
    res = parallel.evaluate(f"pilots.aspire_blind.{mod}:make_policy", {}, seeds=[seed], purpose="calibration",
                            pilot=pilot, n_workers=1)
    print(json.dumps(res))
