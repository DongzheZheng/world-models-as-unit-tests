"""Replay loss of the L world vs contact friction mu on L life 1, L life 2 (wall-free windows) and both."""
import json
import torch
from experiments.pusht.code_wm import fit_world as F
from experiments.pusht.code_wm.fidelity import clearance
from worlds.pusht import pusht_world as W

W.set_object("L")
B = "runs/pusht/blackbox_L/code_wm"
mus = [0.0, 0.2, 0.36, 0.6, 1.0, 2.0, 5.0]
out = {}
for name, lives in (("life1", [f"{B}/life_1.npz"]), ("life2_clear", [f"{B}/life_2.npz"])):
    L = F.load(lives)
    S = F.segments(L, 10)
    cl = [clearance(d["q"]) for d in L]
    S = [(li, t) for li, t in S if cl[li][t - 1:t + 11].min() > 15]
    lf = F.batched_loss(L, S, 10, len(mus))
    prm = {k: torch.tensor([float(v)] * len(mus)) for k, v in W.PARAMS.items()}
    prm["mu"] = torch.tensor(mus)
    out[name] = dict(n_segments=len(S), loss={str(m): round(float(l), 3) for m, l in zip(mus, lf(prm))})
print(json.dumps(out))
json.dump(out, open("runs/pusht/code_wm/L_mu_scan.json", "w"), indent=1)
