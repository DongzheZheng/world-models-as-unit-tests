"""Step-by-step view of one mover vs our drone in a mission-C life: velocity direction vs direction to us."""
import sys
import numpy as np
z = np.load(sys.argv[1]); m = int(sys.argv[2]); st = int(sys.argv[3]) if len(sys.argv) > 3 else 6
P, V, own, ov = z["movers_true_pos"][:, m], z["movers_true_vel"][:, m], z["pos"], z["lin_vel"]
for i in range(0, len(own), st):
    to = own[i] - P[i]; d = np.linalg.norm(to); sp = np.linalg.norm(V[i])
    c = V[i] @ to / max(d * sp, 1e-9)
    print(f"t={i/30:5.2f} mp={P[i].round(2)} mv={V[i].round(2)} |v|={sp:.2f} own={own[i].round(1)} d={d:.2f} cos={c:+.2f}")
