"""Fly ONE calibration life (or the final evaluation) through the lab's parallel.evaluate.
usage: python pilots/aspire_station/fly.py life <version> <seed> [pilot]
       python pilots/aspire_station/fly.py eval <version> [controller_name]
"""
import json
import sys


def main():
    from reallab.station import parallel
    mode, ver = sys.argv[1], sys.argv[2]
    factory = f"pilots.aspire_station.policy_v{ver}:make_policy"
    if mode == "life":
        seed = int(sys.argv[3])
        pilot = sys.argv[4] if len(sys.argv) > 4 else "aspire_style"
        assert seed < 1000
        res = parallel.evaluate(factory, {}, seeds=[seed], purpose="calibration", pilot=pilot, n_workers=1)
        print(json.dumps(res))
    elif mode == "eval":
        name = sys.argv[3] if len(sys.argv) > 3 else "aspire_style"
        res = parallel.evaluate(factory, {}, seeds=list(range(1000, 1050)), purpose="eval", pilot="aspire_style",
                                n_workers=10, log_dir=f"runs/station/aspire_style/eval_logs" +
                                ("" if name == "aspire_style" else "_" + name))
        out = f"runs/station/aspire_style/eval_{name}_v{ver}.json"
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        n = len(res)
        print("success", sum(r["success"] for r in res) / n, "overload", sum(r["overload"] for r in res) / n)
    else:
        raise SystemExit("bad mode")


if __name__ == "__main__":
    main()
