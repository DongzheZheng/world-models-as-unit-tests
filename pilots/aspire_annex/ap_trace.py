"""Summarise a life log: python pilots/aspire_annex/ap_trace.py <npz> [every]"""
import sys, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
if __name__ == "__main__":
    L = np.load(sys.argv[1]); ev = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    P, tool, tv, A = L["points"], L["tool_pos"], L["tool_vel"], L["actions"]
    post, tgt = L["post"][0], L["target"][0]
    print("seed", L["seed"], "success", L["success"], "n", L["n_steps"], "final tip dist", L["final_tip_dist"], "post", post, "target", tgt)
    sp = np.r_[0, np.linalg.norm(np.diff(P[:, :, :2], axis=0), axis=2).max(1) * 50]
    tipv = np.r_[0, np.linalg.norm(np.diff(P[:, 0, :2], axis=0), axis=1) * 50]
    dpost = np.linalg.norm(P[:, :, :2] - post, axis=2).min(1)
    for t in list(range(0, len(P), ev)) + [len(P) - 1]:
        a = A[t - 1] if t > 0 else [np.nan, np.nan]
        print(f"t{t:3d} cmd {np.round(a,3)} tool {np.round(tool[t],3)} tv {np.round(tv[t],2)} tip {np.round(P[t,0],3)} tipv {tipv[t]:.2f} "
              f"maxv {sp[t]:.2f} zmax {P[t,:,2].max():.3f} d_tgt {np.linalg.norm(P[t,0,:2]-tgt):.3f} d_post_min {dpost[t]:.3f} "
              f"tip-tool {np.linalg.norm(P[t,0,:2]-tool[t]):.3f}")
    print("final points\n", P[-1])
