"""Per-step rope kinematics: mean speed, momentum, KE-proxy, tip speed, chord, curvature."""
import sys, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
if __name__ == "__main__":
    L = np.load(sys.argv[1]); P = L["points"][:, :, :2]; t0, t1 = int(sys.argv[2]), int(sys.argv[3])
    V = np.diff(P, axis=0) * 50
    for t in range(t0, t1):
        v = V[t - 1]; sp = np.linalg.norm(v, axis=1)
        seg = np.diff(P[t], axis=0); ang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0])); bend = np.abs(np.diff(ang)).sum()
        acc = (np.linalg.norm(V[t], axis=1) - sp) * 50 if t < len(V) else sp * 0
        print(f"t{t} meanspd {sp.mean():.3f} spd[0,5,10,15] {sp[[0,5,10,15]]} mom {v.mean(0)} KE {np.mean(sp**2):.3f} chord {np.linalg.norm(P[t,0]-P[t,19]):.3f} bend {bend:.2f} dspd/dt(med pts 0-14) {np.median(acc[:15]):.2f}")
