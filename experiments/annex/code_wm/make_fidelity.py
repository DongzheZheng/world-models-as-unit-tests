"""Assemble fidelity.json from the out-of-sample fidelity runs (each life scored by params fitted only to earlier lives)."""
import json

if __name__ == "__main__":
    runs = [("fit6 (life 1)", "runs/annex/code_wm/fid_fit6_on_life2.json", "life_2"),
            ("fit7 (lives 1-2)", "runs/annex/code_wm/fid_fit7_on_life3.json", "life_3"),
            ("fit9 (lives 1-3)", "runs/annex/code_wm/fid_fit9_on_life4.json", "life_4")]
    out = dict(method="open-loop replay of the logged commands from a logged reading (node velocities from the two "
                      "previous readings), every 4th reading as a start; error = |world - real| of tracked points' "
                      "(x,y) after h steps; each life is scored with parameters fitted only to EARLIER lives",
               per_life={}, rest_point={})
    for fit, path, life in runs:
        d = json.load(open(path))
        out["per_life"][life] = dict(params=fit, horizons=d["horizons"])
        for k, w in d["whole_life"].items():
            out["rest_point"][life] = dict(params=fit, real_final_p1=w["real_final_p1"],
                                           world_final_p1_from_reset=w["world_final_p1_from_reset"],
                                           dist_from_reset_m=w["dist_from_reset"],
                                           dist_from_reading2_with_velocity_m=w["dist_from_reading2_with_velocity"],
                                           note="no life ended at rest; this is the final reading at the 236-step limit")
    d = json.load(open("runs/annex/code_wm/fid_fit10_insample.json"))
    out["in_sample_final_world_fit10_lives1to4"] = d["horizons"]
    out["rest_point"]["life_1"] = dict(params="fit6 (life 1, in-sample)", **{k: v for k, v in json.load(open("runs/annex/code_wm/fid_fit6_on_life1.json"))["whole_life"]["runs/annex/blackbox/code_wm/life_1.npz"].items()})
    json.dump(out, open("runs/annex/code_wm/fidelity.json", "w"), indent=1)
    print(json.dumps(out["per_life"], indent=0)[:3000])
