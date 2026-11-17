# Fly the controller for one real calibration life:  python experiments/station/code_wm/fly_ctrl.py SEED '<kwargs json>' [pilot]
import sys, json

if __name__ == "__main__":
    from reallab.station import parallel
    seed = int(sys.argv[1]); kw = json.loads(sys.argv[2]); pilot = sys.argv[3] if len(sys.argv) > 3 else "code_wm"
    assert seed < 1000
    res = parallel.evaluate("experiments.station.code_wm.policy:make_policy", kw, seeds=[seed], purpose="calibration",
                            pilot=pilot, tag="ctrl_test", n_workers=1)
    print(res)
