import json, numpy as np
from pathlib import Path
if __name__ == "__main__":
    lives = []
    meta = {
     ("aspire_style", 1): ("v1", "tethered cast: back up, one full-rate 0.17 m stroke at 52 deg to the cord's lie, hold",
        "tool reached 3.0 m/s; cord folded, tip whipped past the tool by 0.09 m, then the cord slowly (0.5 m/s for ~2 s) re-straightened back to its initial direction; very little energy survived the stop",
        "v2/v3: align the stroke with the cord's lie, longest stroke, avoid slack; add reel-in"),
     ("aspire_style", 2): ("v3", "overcast + reel-in, aligned 0.26 m stroke",
        "whole straight cord moved at 3.5 m/s with the tool (rigid, axial); at the stop KE fell 12.8 -> 1.4 in 5 steps, bend 0.3 -> 4.7 rad; tip travelled ~0.9 m but stalled 0.3 m from the tool and the cord re-straightened sideways/back (rotated ~110 deg); reel-in never triggered",
        "v4: turn the strike into a swing about the parked tool and use the post as a stop"),
     ("aspire_style", 3): ("v4", "swing-into-post: perpendicular strike ending at a tool point whose tangent line past the post carries the tip to the target",
        "cord direction estimated from its centroid, but the near part lay along the strike: the tool pushed into it, lagged (1.5-2 m/s vs 4 m/s reference, servo saturation), cord hardly swung; reel-in fired wrongly (bug) and dragged the tool forward",
        "skills S2/S3/S7; v5: kinematic track-laying + tip-first push"),
     ("aspire_style", 4): ("v5", "lay a 0.7 m track with the tool so the tip lies on it heading away from the target, then reverse along it to push the cord out tip-first",
        "the cord does not follow the tool's path: tip moved <0.25 m over 0.9 m of tool path, only while taut; push did nothing useful",
        "abandon kinematic plans (S5); v6: snap-back cast with a haul"),
     ("aspire_style", 5): ("v6", "snap-back cast: full-rate strike from the back of the zone towards the aim point, immediate reversal (haul) to the start, hold",
        "strike reached 2.5 m/s; after the haul the tip passed 0.25 m ahead of the tool and the cord ended straight and pointing FORWARD (tip 0.65 m from the tool, x=0.77) for the first time, but at -46 deg instead of +8 deg (direction set by the near part after the haul)",
        "submitted v6 as final program (S1, S6)"),
     ("aspire_style_x1", 1): ("v7", "v6 with the haul triggered at 85% instead of 55% of the stroke",
        "trajectory practically identical to life 5 (the reference was already braking; the switch moved by one step at most)",
        "extended run stopped here (time); v7 evaluated once"),
    }
    for (pl, k), (ver, prog, saw, changed) in meta.items():
        L = np.load(f"runs/annex/blackbox/{pl}/life_{k}.npz")
        P = L["points"][:, :, :2]; tool = L["tool_pos"]
        rest = bool(np.linalg.norm(P[-1] - P[-2], axis=1).max() <= 0.0004 and np.linalg.norm(tool[-1] - tool[-2]) <= 0.0004)
        outcome = "success" if bool(L["success"]) else ("at rest outside the target" if rest else "still moving at the time limit")
        lives.append(dict(pilot=pl, life=k, program=ver, program_idea=prog, seed=int(L["seed"]), outcome=outcome,
                          n_steps=int(L["n_steps"]), final_tip_dist_m=round(float(L["final_tip_dist"]), 3),
                          trace_showed=saw, changed=changed))
    off = json.load(open("runs/annex/results/annex__aspire_style.json"))
    ext = json.load(open("runs/annex/results/annex__aspire_style_extended.json"))
    S = dict(
     lives=lives,
     versions=[
      dict(v="v1", desc="tethered cast from the back to a parked point at reach 0.66 m beside the post; plain stop; hold"),
      dict(v="v2", desc="v1 + stroke direction chosen opposite to the cord's lie (never flown)"),
      dict(v="v3", desc="overcast beyond the target + slow feedback reel-in of the tip; slack penalty"),
      dict(v="v4", desc="perpendicular strike to swing the cord about the parked tool into the post as a stop; reel-in"),
      dict(v="v5", desc="kinematic: lay a track and push the cord out tip-first along it"),
      dict(v="v6", desc="snap-back cast: full-rate strike towards the aim point then immediate haul back to the start; hold  [SUBMITTED]"),
      dict(v="v7", desc="v6 with later haul switch (85%) [extended run]"),
     ],
     submitted="policy_v6.py", final_program_bytes=3603,
     official=dict(lives=5, real_steps=1180, eval_seeds="1000-1049", success_rate=off["eval"]["mean"],
                   split=dict(success=1, at_rest_outside_target=47, still_moving_at_limit=2),
                   tip_dist_median_m=round(off["other"]["tip_dist"]["median"], 3), tip_dist_mean_m=round(off["other"]["tip_dist"]["mean"], 3)),
     extended=dict(lives_extra=1, lives_total=6, real_steps_total=1416, program="policy_v7.py", success_rate=ext["eval"]["mean"],
                   split=dict(success=1, at_rest_outside_target=47, still_moving_at_limit=2),
                   note="cut short after 1 extra life (time budget); v7 behaved like v6 on every eval seed"),
     conclusion=("A flexible, bending-elastic cord (~0.70 m, ~1 cm thick) lying on a table, one end held by the tool. It slides "
                 "quickly when pulled taut and fast (the whole straight cord moved at 3.5 m/s with the tool), stops within ~0.1 s "
                 "when the motion has to bend it, and after any throw re-straightens slowly (0.3-0.5 m/s over 1.5-2 s) along the "
                 "direction of its near part. The task is to throw (cast/flick) it so the free end comes to rest in a disc placed "
                 "0.12 m straight behind a short post, out of the tool's reach."),
     field_note=dict(one_go="once thrown, the cord can only be pulled back (and slow pulls only move a taut tip towards the tool); a second throw needs the cord behind the tool again",
                     slowly_nowhere="confirmed in life 4: slow tool paths leave the tip where it is; only a fast, taut throw moves the cord beyond the zone",
                     small_change="confirmed: the end direction depends on how the near part ends up after the hairpin (life 5: 54 deg off), which shifts with small changes of stroke and initial curl"),
    )
    Path("runs/annex/aspire_style/summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: S[k] for k in ("official", "extended")}, indent=1))
    for l in lives: print(l["pilot"], l["life"], l["program"], l["seed"], l["outcome"], l["final_tip_dist_m"], l["n_steps"])
