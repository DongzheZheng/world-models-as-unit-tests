"""Print the instrumented trajectories of the other aircraft in mission-C lives (numpy only)."""
import sys
import numpy as np

for path in sys.argv[1:]:
    z = np.load(path)
    P, V, C = z["movers_true_pos"], z["movers_true_vel"], z["movers_true_count"]
    pos = z["pos"]
    print("=====", path, "end", z["end_cause"], "steps", int(z["n_steps"]), "count", np.unique(C), "own start", pos[0].round(2))
    n = int(C.max())
    for m in range(n):
        p, v = P[:, m], V[:, m]
        sp = np.linalg.norm(v, axis=1)
        print(f"-- mover {m}: x [{p[:,0].min():.1f},{p[:,0].max():.1f}] y [{p[:,1].min():.1f},{p[:,1].max():.1f}] z [{p[:,2].min():.2f},{p[:,2].max():.2f}] "
              f"speed mean {sp.mean():.2f} min {sp.min():.2f} max {sp.max():.2f}  min dist to me {np.linalg.norm(p - pos, axis=1).min():.2f}")
        for i in range(0, len(p), 45):
            print(f"   t={i/30:5.1f} p={p[i].round(2)} v={v[i].round(2)} d_me={np.linalg.norm(p[i]-pos[i]):.1f}")
    if "movers" in z.files:
        nm = z["n_movers"].reshape(-1)
        print("observed n_movers over time (every 1 s):", nm[::30].astype(int).tolist())
