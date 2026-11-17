"""Per-life snap-back cast features: python ap_snap.py <npz> ...  (strike start, aim, cord lie, resting direction, reach)"""
import sys, numpy as np
np.set_printoptions(precision=3, suppress=True)
def ang(v): return float(np.degrees(np.arctan2(v[1], v[0])))
def wrap(a): return (a + 180) % 360 - 180
if __name__ == "__main__":
    for f in sys.argv[1:]:
        L = np.load(f); P = L["points"][:, :, :2]; tool = L["tool_pos"]; A = L["actions"]
        sp = np.linalg.norm(np.diff(A, axis=0), axis=1)
        k = int(np.argmax(sp > 0.05)) + 1          # first big command jump = strike start (action index)
        F = tool[k]; G = A[k]
        d = ang(G - F)
        rn = ang(P[k, 13] - P[k, 19]); rc = ang(P[k].mean(0) - F)
        mis = wrap(rn - (d + 180))
        vmax = np.linalg.norm(L["tool_vel"], axis=1).max()
        tend = tool[-1]; phi = ang(P[-1, 0] - tend); reach = np.linalg.norm(P[-1, 0] - tend)
        chord0 = np.linalg.norm(P[k, 0] - P[k, 19])
        print(f"{f.split('/')[-2]}/{f.split('/')[-1]} k={k} F={F} d={d:.1f} near={rn:.1f} cen={rc:.1f} mis_near={mis:.1f} mis_cen={wrap(rc-(d+180)):.1f} chord0={chord0:.3f} vtool_max={vmax:.2f} | rest_dir={phi:.1f} delta={wrap(phi-d):.1f} reach={reach:.3f} tipdist={float(L['final_tip_dist']):.3f}")
