"""Add the object_L block to runs/pusht/aspire_style/summary.json."""
import json
import os
import numpy as np
from pilots.aspire_pusht.make_summary import evalstats

LIVES = [
 ("aspire_style", 1, 9, "failure, max 0.945 at t=263: angle error fell only +69 -> +2.5 deg in a fixed 4-contact cycle (left side, leg right side, leg top, bottom), ~6-10 deg per ~60-step cycle; every push ended by alignment falling below the keep threshold; rotation regression k = 0.00008 vs 0.00015 on the T",
  "v10: S10 rotation-response scaling g = 2.5, S11 relative thresholds, COM shrunk to (39, 47)"),
 ("aspire_style", 2, 10, "failure, max 0.428: eth +83 deg while the body origin was 2 u from its goal; 280 steps of short pushes and 16-21-step walks; eth 83 -> 27 deg",
  "none before the official evaluation (v10 evaluated: 8/50); then v11: S12 COM-based twist + rotation phase"),
 ("aspire_style_x1", 1, 11, "seed 29 again: failure, max 0.783 (was 0.428); first push turned 19 deg in 9 steps, later pushes 1-4 deg in 6 steps (depth ~8 u)",
  "v12: S10b push depth in push units, COM (37.5, 48) from pooled regression"),
 ("aspire_style_x1", 2, 12, "seed 29 again: SUCCESS at step 267 (0.952); still ~80% walking, pushes 2-4 steps", "none"),
 ("aspire_style_x2", 1, 12, "new seed 41 (start eth -135 deg): failure, max 0.862; pushes 2-4 steps then 9-22-step walks", "v13: S13 push commitment (>= 6 steps)"),
 ("aspire_style_x2", 2, 13, "seed 41 again: failure, max 0.882; phase breakdown: 76 radial-exit steps at 10.9 u/step, 37 orbit, 48 direct, 48 corridor, 91 push",
  "v14: S7b fast radial exit with early orbiting, corridor 2x"),
 ("aspire_style_x3", 1, 14, "seed 41 a third time: failure, max 0.942 (coverage > 0.9 from t=231); coverage rose in walks and fell in each precision push",
  "v15: S10c T-validated depth inside the precision phase"),
 ("aspire_style_x3", 2, 15, "new seed 53 (start eth +151 deg): failure, max 0.890, never fully in the precision phase", "none (all 6 extra lives used; v15 evaluated)"),
]

if __name__ == "__main__":
    S = json.load(open("runs/pusht/aspire_style/summary.json"))
    lives = []
    for pilot, k, v, saw, change in LIVES:
        d = np.load(f"runs/pusht/blackbox_L/{pilot}/life_{k}.npz")
        lives.append(dict(pilot=pilot, life=k, seed=int(d["seed"]), program=f"policy_v{v}.py", success=bool(d["success"]),
                          max_coverage=round(float(d["max_coverage"]), 4), n_steps=int(d["n_steps"]), trace_showed=saw, changed=change))
    off = [l for l in lives if l["pilot"] == "aspire_style"]; ext = [l for l in lives if l["pilot"] != "aspire_style"]
    sk = json.load(open("pilots/aspire_pusht/skills.json"))
    S["object_L"] = dict(
        lives=lives,
        official=dict(lives_used=len(off), real_steps=sum(l["n_steps"] for l in off), final_program="pilots/aspire_pusht/policy_v10.py",
                      final_program_bytes=os.path.getsize("pilots/aspire_pusht/policy_v10.py"),
                      eval=evalstats("runs/pusht/aspire_style/eval_L_aspire_style/episodes.json"),
                      result_file="runs/pusht/results/pusht_L__aspire_style.json",
                      eval_gifs="runs/pusht/aspire_style/gifs/L_aspire_style_eval/ep_1000.gif .. ep_1002.gif"),
        extended=dict(extra_lives_used=len(ext), total_L_lives=len(lives), real_steps_total=sum(l["n_steps"] for l in lives),
                      final_program="pilots/aspire_pusht/policy_v15.py", final_program_bytes=os.path.getsize("pilots/aspire_pusht/policy_v15.py"),
                      eval=evalstats("runs/pusht/aspire_style/eval_L_aspire_style_extended/episodes.json"),
                      result_file="runs/pusht/results/pusht_L__aspire_style_extended.json",
                      eval_gifs="runs/pusht/aspire_style/gifs/L_aspire_style_extended_eval/ep_1000.gif .. ep_1002.gif"),
        life_gifs="runs/pusht/aspire_style/gifs/L_<pilot>_life_<k>_v<ver>_seed<seed>.gif",
        skills_transfer=sk["transfer_to_L"],
        conclusions="Same rig behaviour as the T (pusher response, coasting) carried over unchanged. The L turns ~2-3x less per unit push relative to how far it translates; COM regressions put the mass near (37, 48) vs the area centroid (37.6, 42.6), i.e. shifted towards the far end of the long leg as the T's was shifted up its stem (weak, per-life fits scatter 26-40 in u). Large initial angle errors (>100 deg) are the hard cases: the program cannot turn the L fast enough within 300 steps.",
    )
    json.dump(S, open("runs/pusht/aspire_style/summary.json", "w"), indent=1)
    print(json.dumps({k: S["object_L"][k] for k in ("official", "extended")}, indent=1))
