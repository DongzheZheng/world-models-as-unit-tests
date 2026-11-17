import json, numpy as np
from pathlib import Path
EXT = {
 ("aspire_style_x1", 1): ("v7", "v6 with the haul at 85% of the stroke", "practically identical to life 5", "nothing; timing irrelevant for short strokes"),
 ("aspire_style_x1", 2): ("v8", "cast line most opposite to the near part (p20->p14), re-plan on arrival", "near mis 1.9 deg but centroid mis 26.5: tip got 0.15 m ahead, cord swung to +y and unfolded at 112 deg (delta +139)", "v9: measure the lie by the centroid"),
 ("aspire_style_x1", 3): ("v9", "cast line most opposite to the tool->centroid direction (stroke 0.27 m)", "centroid mis -4.8: tip flew straight at the target at 6 m/s, reached 0.057 m from its centre at 4.5 m/s, snapped at chord 0.61 and recoiled behind the tool", "v10: cut energy"),
 ("aspire_style_x1", 4): ("v10", "v9 with a 0.12 m stroke (closed-form energy estimate)", "no whip at all, tip peak 1.3 m/s", "bracket the energy: v11"),
 ("aspire_style_x2", 1): ("v11", "v9 with stroke <= 0.20 m", "tip stalled 0.26 m short of the target, then fell back; final 0.389 m", "v12: interpolate"),
 ("aspire_style_x2", 2): ("v12", "v9 with stroke 0.23 m (energy interpolation)", "tip slowed to ~0 at the post's side (0.136 m from the target), kinked cord dragged it back", "v13: more energy and more clearance"),
 ("aspire_style_x2", 3): ("v13", "stroke 0.25 m, line >= 0.07 m from the post", "different parked point; tip stalled 0.33 m away beside the post, cord swung laterally and went back", "v14"),
 ("aspire_style_x2", 4): ("v14", "v9 geometry, stroke 0.25 m + soft give (<= 0.06 m)", "tip reached 0.082 m from the target centre (inside the disc) at 0.8 m/s, grazed the post, recoiled; give only 0.02 m", "v15: keep the cord taut"),
 ("aspire_style_x2", 5): ("v15", "stroke 0.25, park 0.60 m, chord servo pulling the tool back (<= 0.12 m)", "tip at the target (0.077 m) at 3.6 m/s, snapped at chord 0.611, recoiled despite the servo", "v16: let the post absorb it"),
 ("aspire_style_x3", 1): ("v16", "cast aimed at the post (post 0.50 m from the tool), stroke 0.27", "cord missed the post by 0.07 m; tip reached 0.054 m from the target at 1.4 m/s and recoiled at chord 0.609", "v17: follow-through"),
 ("aspire_style_x3", 2): ("v17", "v9 cast (park 0.58 m, clearance 0.05, stroke 0.27) + 0.08 m follow-through at turnover", "recoil largely suppressed: tip 0.089 m from the target at turnover, then crept to 0.30 m with the cord folded; best extended life", "v18: fire earlier, give more"),
 ("aspire_style_x3", 3): ("v18", "v17 with the follow-through at chord 0.30 and 0.12 m, park 0.56 m", "tip 0.107 m from the target, then hit the post (0.030 m) and collapsed back", "keep v17 timing; test generalisation"),
 ("aspire_style_x3", 4): ("v17", "v17 on a new situation", "curled cord (chord 0.375): no whip, tip only 0.2 m ahead", "v19: pre-straighten"),
 ("aspire_style_x3", 5): ("v19", "v17 + slow pre-straightening walk", "chord 0.375 -> 0.58, whip formed, tip 0.124 m from the target at 0.85 m/s, then grazed the post (0.031 m) and came back", "budget used; v19 submitted"),
}
if __name__ == "__main__":
    S = json.loads(Path("runs/annex/aspire_style/summary.json").read_text())
    S["extended_after_6_lives"] = S.pop("extended")
    S["extended_after_6_lives"]["record_copy"] = "runs/annex/aspire_style/result_extended_after_6_lives.json"
    S["lives"] = [l for l in S["lives"] if l["pilot"] == "aspire_style"]
    for (pl, k), (ver, prog, saw, changed) in EXT.items():
        L = np.load(f"runs/annex/blackbox/{pl}/life_{k}.npz")
        P = L["points"][:, :, :2]; tool = L["tool_pos"]; tgt = L["target"][0]
        rest = bool(np.linalg.norm(P[-1] - P[-2], axis=1).max() <= 0.0004 and np.linalg.norm(tool[-1] - tool[-2]) <= 0.0004)
        outcome = "success" if bool(L["success"]) else ("at rest outside the target" if rest else "still moving at the time limit")
        S["lives"].append(dict(pilot=pl, life=k, program=ver, program_idea=prog, seed=int(L["seed"]), outcome=outcome,
                               n_steps=int(L["n_steps"]), final_tip_dist_m=round(float(L["final_tip_dist"]), 3),
                               closest_tip_to_target_m=round(float(np.linalg.norm(P[:, 0] - tgt, axis=1).min()), 3),
                               trace_showed=saw, changed=changed))
    S["versions"] = S["versions"][:7] + [
      dict(v="v8", desc="cast line chosen opposite to the near part; one live re-plan"),
      dict(v="v9", desc="cast line chosen opposite to the cord's centroid"),
      dict(v="v10", desc="v9, stroke 0.12 m"), dict(v="v11", desc="v9, stroke 0.20 m"), dict(v="v12", desc="v9, stroke 0.23 m"),
      dict(v="v13", desc="stroke 0.25, clearance 0.07"), dict(v="v14", desc="stroke 0.25 + small soft give"),
      dict(v="v15", desc="park 0.60 + chord servo (tool pulls back)"), dict(v="v16", desc="cast aimed at the post"),
      dict(v="v17", desc="v9 cast (park 0.58, stroke 0.27) + 0.08 m follow-through at turnover"),
      dict(v="v18", desc="v17 with earlier/larger follow-through, park 0.56"),
      dict(v="v19", desc="v17 + pre-straightening walk for curled cords  [SUBMITTED for the extended evaluation]")]
    S["versions"][5]["desc"] = S["versions"][5]["desc"].replace("[SUBMITTED]", "[SUBMITTED, official]")
    rec = json.loads(Path("runs/annex/results/annex__aspire_style_extended.json").read_text())
    S["extended_final"] = dict(lives_extra=14, lives_total=19, unused_extra_lives=1, real_steps_total=4484, program="policy_v19.py",
                              program_bytes=Path("pilots/aspire_annex/policy_v19.py").stat().st_size,
                              success_rate=rec["eval"]["mean"], split=dict(success=0, at_rest_outside_target=50, still_moving_at_limit=0),
                              tip_dist_median_m=round(rec["other"]["tip_dist"]["median"], 3), tip_dist_mean_m=round(rec["other"]["tip_dist"]["mean"], 3),
                              log_dir="runs/annex/aspire_style/eval_logs_extended_final",
                              note="one of the 15 allowed extra lives was not used (counting slip: x1 got 4 lives, not 5); noticed after the evaluation")
    S["conclusion"] += (" Extended run: a full-rate stroke of >= 0.25 m with an immediate haul, on a straight cord aligned so its "
                        "centroid lies straight behind the cast line, throws the tip into the target disc reliably on one situation "
                        "(5 of 6 such casts passed within 0.06-0.13 m of its centre), but the cord turns over at a flight chord of 0.61 m "
                        "and recoils, or grazes the post and collapses; a follow-through softens the recoil. Few layouts leave room for "
                        "the needed stroke, and nothing made the tip come to rest in the disc.")
    Path("runs/annex/aspire_style/summary.json").write_text(json.dumps(S, indent=1))
    for l in S["lives"]: print(l["pilot"], l["life"], l["program"], l["seed"], l["final_tip_dist_m"], l.get("closest_tip_to_target_m"))
