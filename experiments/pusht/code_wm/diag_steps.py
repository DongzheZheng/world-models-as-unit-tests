"""Per-step 1-step and 5-step replay errors of the current world on one life (diagnostics).
python experiments/pusht/code_wm/diag_steps.py <object> <npz> [every]"""
import sys
import numpy as np
import torch
from experiments.pusht.code_wm import fit_world as F
from worlds.pusht import pusht_world as W

W.set_object(sys.argv[1])
L = F.load([sys.argv[2]])
d = L[0]
H = 5
S = [(0, t) for t in range(1, len(d["a"]) - H + 1)]
p0, pv0, q0, qm1, A, Q, P = F.batch(L, S, H)
prm = {k: torch.tensor(float(v), device=F.DEV) for k, v in W.PARAMS.items()}
Qp, _ = F.rollout(prm, p0, pv0, q0, qm1, A, H)
dpos, dth = F.pose_err(Qp, Q)
np.set_printoptions(precision=1, suppress=True, linewidth=220)
ev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
for i, (_, t) in enumerate(S):
    if i % ev == 0:
        print(t, d["a"][t], d["p"][t], d["q"][t], "| 1-step err %.2f  5-step err %.2f  dth5 %.1f" % (
            dpos[i, 0], dpos[i, 4], np.degrees(float(dth[i, 4]))), "pred1", Qp[i, 0].cpu().numpy(), "real1", d["q"][t + 1])
