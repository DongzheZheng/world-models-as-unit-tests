"""sensitivity.json from sensitivity_raw.json (official world fit10 + official controller throw:make_final)."""
import json

if __name__ == "__main__":
    r = json.load(open("runs/annex/code_wm/sensitivity_raw.json"))
    worst = {}
    for k, d in r["params"].items():
        worst[k] = {f: dict(success=v["success"], p1_shift_median_m=round(v["p1_shift_median"], 4)) for f, v in d.items()}
    pct3 = [d["+0.03"]["p1_shift_median"] for d in r["params"].values()] + [d["-0.03"]["p1_shift_median"] for d in r["params"].values()]
    out = dict(
        setup="official world (fit10) and official controller (throw:make_final); 12 snapshot situations 150-161; "
              "controller run closed-loop in the nominal world, its command sequence then replayed OPEN-LOOP in a "
              "world with one parameter scaled by (1+f), or with node positions perturbed by Gaussian noise",
        closed_loop_nominal_success=r["closed_loop_nominal_success"],
        open_loop_replay_nominal_success=r["open_loop_replay_nominal_success"],
        per_parameter=r["params"], initial_state=r["initial_state"],
        summary=dict(median_over_params_of_p1_shift_median_at_3pct_m=sorted(pct3)[len(pct3) // 2], compact=worst),
        implication=("A 3% error in most rod parameters (mu, k_bend, aniso, lat_mass, k_stretch) moves the open-loop "
                     "landing point by several cm (median) and tens of cm (p90), comparable to the 0.10 m target "
                     "radius, and drops open-loop success from nominal to ~0.3-0.6; 10% errors drop it to 0-0.5. "
                     "Drag terms (c_visc, c_quad) and the servo matter little. Measured open-loop fidelity "
                     "(fidelity.json) is far worse than 3% parameter error implies, so an open-loop throw cannot be "
                     "trusted; the task demands either ~1% model accuracy over 1-3 s or feedback."),
        how_the_controller_copes=("receding-horizon re-planning every 8 steps from live readings (the world is only "
                                  "trusted over the next few tens of steps), domain randomisation over +-25% of the "
                                  "uncertain parameters when choosing the throw, a command-rate cap, and one "
                                  "re-throw if no imagined world still predicts success"),
        wall_s=r.get("wall_s"))
    json.dump(out, open("runs/annex/code_wm/sensitivity.json", "w"), indent=1)
    print(json.dumps(out["initial_state"]), out["summary"]["median_over_params_of_p1_shift_median_at_3pct_m"],
          out["closed_loop_nominal_success"], out["open_loop_replay_nominal_success"])
