"""Static geometry: shortest homotopic cable length anchor -> (tool point) for the planned tool path, vs 1.0 m.
Also: does the life log's cable look taut at the end?"""
import importlib
import sys
import numpy as np
from pilots.aspire_blind import geom

ANCHOR = np.array([0.05, 0.0])


def taut_len(mod, q, posts, sides, r=0.034):
    pts, S = mod.corridor_path(ANCHOR, posts, sides, q, R=r, iters=600)
    return S[-1], pts


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "policy_v1"
    mod = importlib.import_module(f"pilots.aspire_blind.{ver}")
    d = np.load("runs/station/observation/layouts.npz")
    tool, posts, ps, tgt = d["tool_pos"], d["posts"], d["post_sides"], d["target"]
    Lend, Lmax, Wr = [], [], []
    for i in range(200):
        path, S = mod.corridor_path(tool[i], posts[i], ps[i], tgt[i])
        L_end, rope = taut_len(mod, tgt[i], posts[i], ps[i])
        Lend.append(L_end)
        Wr.append((geom.winding(rope, posts[i]) * ps[i]).min())
        idx = np.linspace(0, len(path) - 1, 25).astype(int)
        Lmax.append(max(taut_len(mod, path[k], posts[i], ps[i])[0] for k in idx))
    Lend, Lmax, Wr = map(np.array, (Lend, Lmax, Wr))
    print("taut rope length at target: mean %.3f max %.3f n>0.97 %d" % (Lend.mean(), Lend.max(), (Lend > 0.97).sum()))
    print("max taut length along tool path: mean %.3f max %.3f n>0.97 %d n>1.0 %d" % (Lmax.mean(), Lmax.max(), (Lmax > 0.97).sum(), (Lmax > 1.0).sum()))
    print("taut-rope winding min: min %.2f n<1.8 %d" % (Wr.min(), (Wr < 1.8).sum()))
    print("seed 38: Lend %.3f Lmax %.3f" % (Lend[38], Lmax[38]))
    print("tight seeds", np.where(Lmax > 0.95)[0], Lmax[Lmax > 0.95])
    print("percentiles Lmax", np.percentile(Lmax, [50, 90, 99]))
