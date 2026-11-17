"""Assemble runs/pusht/aspire_style/summary.json from the black-box lives, eval outputs and the change log."""
import json
import os
import numpy as np

LIVES = [  # (pilot, life, version, trace finding, change made after it)
 ("aspire_style", 1, 1, "success at step 278 (cov 0.965); 228/278 steps walking; alternated two contacts with ~28-step walks per 7-step push; pusher closes ~40% of its target gap per step; block coasts after release (decay ~0.63/step); rotation regression COM (1.4, 42.2)",
  "v2: travel cost in contact choice + persistence (S1), faster navigation (S2)"),
 ("aspire_style", 2, 2, "success at step 193 (cov 0.951); 4 pushes near the bar corner slid off laterally and re-approached; pooled COM fit (5.4, 42.9) sd (1.9, 2.7); pusher response dq = 0.30 (target-q) + 0.30 v dt",
  "v3: COM (5, 42.5) (S3), faces trimmed away from corners, lateral tol 8 (S4), orbit look-ahead, v_free 90 (S2b)"),
 ("aspire_style", 3, 3, "success at step 226 (cov 0.953); same-face slip/re-approach cycles (push 5, walk 7) x4; overshoot near goal 0.943 -> 0.916; per-life COM u estimate -5.1 (pooled 3.0, 44.0)",
  "v4: re-anchor sliding contact along the face (S4b), coast-aware push stop (S5), COM (3, 44)"),
 ("aspire_style", 4, 4, "success at step 169 (cov 0.953); contact slid 25->27->28 without re-approach (S4b works); 0.8 -> 0.95 took 83 steps, ~2/3 walking",
  "v5: precision phase (twist error < 20): travel weight 0.6, keep 0.3 (S6)"),
 ("aspire_style", 5, 5, "success at step 105 (cov 0.951); 0.79 -> 0.95 in 30 steps", "none (official run ended; v5 evaluated)"),
 ("aspire_style_x1", 1, 6, "success at step 96 (cov 0.955); walks now 6-10 steps", "none"),
 ("aspire_style_x1", 2, 6, "FAILURE (max cov 0.822): 272 steps parked in 'push' on a contact with alignment ~0.01; travel cost made the useless adjacent contact win; coast stop held depth -4",
  "v7: contacts below keep threshold never selected; stall watchdog bans a motionless contact for 25 steps (S8)"),
 ("aspire_style_x1", 3, 7, "success at step 93 (cov 0.954)", "none"),
 ("aspire_style_x1", 4, 7, "success at step 115 (cov 0.954)", "none"),
 ("aspire_style_x1", 5, 7, "success at step 246 (cov 0.950); coverage lost in pushes started on contacts with alignment 0.33-0.55",
  "v8: a new contact needs alignment >= 0.6 (S6b)"),
 ("aspire_style_x2", 1, 8, "success at step 100 (cov 0.951)", "none"),
 ("aspire_style_x2", 2, 8, "success at step 92 (cov 0.950)", "none"),
 ("aspire_style_x2", 3, 8, "success at step 81 (cov 0.950)", "none (stopped: 8 of 15 extra lives used)"),
]

def evalstats(path):
    eps = json.load(open(path))
    return dict(n=len(eps), success_rate=float(np.mean([e["success"] for e in eps])),
                mean_max_coverage=float(np.mean([e["max_coverage"] for e in eps])),
                mean_n_steps=float(np.mean([e["n_steps"] for e in eps])))

if __name__ == "__main__":
    lives = []
    for pilot, k, v, saw, change in LIVES:
        d = np.load(f"runs/pusht/blackbox/{pilot}/life_{k}.npz")
        lives.append(dict(pilot=pilot, life=k, seed=int(d["seed"]), program=f"policy_v{v}.py", success=bool(d["success"]),
                          max_coverage=round(float(d["max_coverage"]), 4), n_steps=int(d["n_steps"]),
                          trace_showed=saw, changed=change))
    off = [l for l in lives if l["pilot"] == "aspire_style"]; ext = [l for l in lives if l["pilot"] != "aspire_style"]
    S = dict(
        pilot="aspire_style", paradigm="code as policy (ASPIRE-style: write program, execute for real, read trace, repair, distil skills)",
        lives=lives,
        official=dict(lives_used=len(off), real_steps=sum(l["n_steps"] for l in off), lives_succeeded=sum(l["success"] for l in off),
                      final_program="pilots/aspire_pusht/policy_v5.py", final_program_bytes=os.path.getsize("pilots/aspire_pusht/policy_v5.py"),
                      eval=evalstats("runs/pusht/aspire_style/eval_aspire_style/episodes.json"),
                      result_file="runs/pusht/results/pusht__aspire_style.json",
                      eval_gifs="runs/pusht/aspire_style/gifs/aspire_style_eval/ep_1000.gif .. ep_1002.gif"),
        extended=dict(extra_lives_used=len(ext), total_lives=len(lives), real_steps_total=sum(l["n_steps"] for l in lives),
                      extra_lives_succeeded=sum(l["success"] for l in ext),
                      final_program="pilots/aspire_pusht/policy_v8.py", final_program_bytes=os.path.getsize("pilots/aspire_pusht/policy_v8.py"),
                      eval=evalstats("runs/pusht/aspire_style/eval_aspire_style_extended/episodes.json"),
                      result_file="runs/pusht/results/pusht__aspire_style_extended.json",
                      eval_gifs="runs/pusht/aspire_style/gifs/aspire_style_extended_eval/ep_1000.gif .. ep_1002.gif"),
        life_gifs="runs/pusht/aspire_style/gifs/<pilot>_life_<k>_v<ver>_seed<seed>.gif",
        skill_library="pilots/aspire_pusht/skills.json",
    )
    json.dump(S, open("runs/pusht/aspire_style/summary.json", "w"), indent=1)
    print(json.dumps({k: S[k] for k in ("official", "extended")}, indent=1))
