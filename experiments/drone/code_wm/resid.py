"""One-step acceleration residual (logged - predicted) of a fitted world along one life, smoothed over 5 steps."""
import sys
import numpy as np
import torch
from experiments.drone.code_wm.fit_world import load, windows, rollout
from experiments.drone.code_wm.imagine import load_world

prm = load_world(sys.argv[1])
L = load(sys.argv[2])
Wd = windows([L], 1)
with torch.no_grad():
    P, V, _, _ = rollout(prm, Wd, 1)
res = ((Wd["tgt_vel"][:, 0] - V[:, 0]) * 30).cpu().numpy()
pos = Wd["pos"].cpu().numpy()
k = np.ones(5) / 5
rs = np.stack([np.convolve(res[:, j], k, "same") for j in range(3)], 1)
step = int(sys.argv[3]) if len(sys.argv) > 3 else 6
for i in range(0, len(rs), step):
    print(f"{(i + 2) / 30:5.2f} pos {pos[i].round(2)}  res {rs[i].round(2)}")
print("rms res", np.sqrt((res ** 2).mean(0)).round(3), "smoothed", np.sqrt((rs ** 2).mean(0)).round(3))
