"""Replay logged observations through a program version: what it WOULD have commanded vs what was commanded.
usage: python pilots/aspire_station/an_replay.py <version> <npz> ..."""
import importlib
import sys
import numpy as np

KEYS = ("points", "tool_pos", "tool_vel", "posts", "post_sides", "sides_now", "target", "load", "t")

if __name__ == "__main__":
    mod = importlib.import_module(f"pilots.aspire_station.policy_v{sys.argv[1]}")
    for path in sys.argv[2:]:
        d = np.load(path, allow_pickle=True)
        pol = mod.make_policy()
        pol.reset()
        diffs = []
        for t in range(len(d["actions"])):
            obs = {k: d[k][t] for k in KEYS}
            obs["load"] = float(obs["load"]); obs["t"] = int(obs["t"])
            diffs.append(np.linalg.norm(pol.act(obs) - d["actions"][t]))
        diffs = np.array(diffs)
        print(path, "max cmd diff", diffs.max().round(5), "first t diff>1mm", int(np.argmax(diffs > 1e-3)) if (diffs > 1e-3).any() else None)
