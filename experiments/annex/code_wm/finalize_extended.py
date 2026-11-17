"""After the extended evaluation: add extended results to summary.json, the v2 out-of-sample errors to fidelity.json,
and the extended findings to diagnosis.json (official fields kept)."""
import json
import subprocess

R = "runs/annex/code_wm/"


def hz(path):
    d = json.load(open(path))
    return {h: dict(node_mean=round(v["node_mean"], 4), p1_mean=round(v["p1_mean"], 4), p1_median=round(v["p1_median"], 4))
            for h, v in d["horizons"].items()}


if __name__ == "__main__":
    ev = json.load(open(R + "eval_code_wm_extended.json"))
    split = json.loads(subprocess.check_output(["python", "experiments/annex/code_wm/eval_split.py", R + "eval_logs_extended"]))
    s = json.load(open(R + "summary.json"))
    eps = ev["episodes"]
    s["extended"].update(
        real_success_rate=sum(e["success"] for e in eps) / len(eps), real_successes=int(sum(e["success"] for e in eps)),
        real_failures=dict(rest_outside_target=split["rest_outside"], moving_at_time_limit=split["moving_at_limit"]),
        tip_dist_mean=sum(e["tip_dist"] for e in eps) / len(eps), eval_wall_s=ev["wall_s"],
        compute_per_step_s=ev["wall_s"] * 6 / sum(e["n_steps"] for e in eps),
        prediction_error_points=round(100 * (s["extended"]["predicted_success_rate"] - sum(e["success"] for e in eps) / len(eps)), 1),
        recorded_as="code_wm_extended (runs/annex/results/annex__code_wm_extended.json)")
    json.dump(s, open(R + "summary.json", "w"), indent=1)

    f = json.load(open(R + "fidelity.json"))
    f["extended"] = dict(
        world_selection_held_out=dict(
            note="all three fitted on the same 6 lives (code_wm 1-4, x1 1-2), scored on 2 unseen probe lives (code_wm 5, x1 3)",
            official_structure=hz(R + "xfid_v1_ho.json"), v2_without_link_projection=hz(R + "xfid_v2nopbd_ho.json"),
            v2=hz(R + "xfid_v2_ho.json")),
        controller_lives_out_of_sample=dict(
            note="lives x1 4, x1 5, x2 1 (flown with throw_v2, never fitted); official world fit10 vs v2 world_params_v2",
            official_world_fit10=hz(R + "xfid_v1_on_ctrl_lives.json"), v2=hz(R + "xfid_v2_on_ctrl_lives.json")))
    json.dump(f, open(R + "fidelity.json", "w"), indent=1)

    d = json.load(open(R + "diagnosis.json"))
    p2 = json.load(open(R + "world_params_v2.json"))["params"]
    d["extended"] = dict(
        lift_off_regime=("Lift-off was seen ONCE: life 2, an uncapped 0.4 m reversing stroke at 4 m/s made while the rod "
                         "was extended and still moving fast (z up to 0.27 m). In every other life z <= 0.0067 m: lives 1, "
                         "3, 4, the four stroke probes (strokes of 0.15-0.25 m at 1-4 m/s starting from a lying rod) and "
                         "all 12 lives flown with the 3 m/s-capped v2 controllers. So a planar world is enough for the "
                         "regime the capped controller operates in, not for violent reversals of an extended moving rod."),
        friction_form=dict(mu_at_speed=p2["mu"], mu_low_speed=p2["mu_lo"], v_transition=p2["v_s"],
                           finding="table friction is speed dependent: high (~0.28) for fast sliding, low (~0.09) at low speed; "
                                   "this explains why fast strokes lose energy quickly while a slowly turning rod coasts and "
                                   "settles slowly (25 of 34 official failures were still moving)"),
        links="near-inextensible with inelastic snap (position projection) fits better than stiff damped springs",
        still_wrong="tip_mass sits at its bound (6): the lumped-mass inertia distribution is still not right",
        what_mattered_most=("on held-out lives at 60 steps: official 24.6 cm -> speed-dependent friction alone 19.3 cm -> "
                            "plus link projection 14.5 cm; on unseen controller lives the v2 world halves the error at "
                            "5/20/60 steps (2.9/13.7/40 cm -> 1.2/5.8/17 cm)"),
        params_v2=p2)
    json.dump(d, open(R + "diagnosis.json", "w"), indent=1)
    print(json.dumps(s["extended"], indent=0)[-900:])
