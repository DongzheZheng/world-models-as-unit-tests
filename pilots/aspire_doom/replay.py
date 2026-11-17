"""Open-loop replay: feed LOGGED observations of a past life to a program and print what it WOULD have output
next to what was actually done. No environment, no dynamics: the observations are fixed recordings.
python pilots/aspire_doom/replay.py <life.jsonl.gz> <version> [from_step]"""
import gzip
import importlib
import json
import sys

path, ver = sys.argv[1], sys.argv[2]
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
L = [json.loads(l) for l in gzip.open(path, "rt")]
pol = importlib.import_module(f"pilots.aspire_doom.policy_{ver}").make_policy()
pol.bind("take_cover", L[0]["meta"]["action_names"])
pol.reset()
obs = L[1]["reset"]["obs"]
for i, s in enumerate(L[2:]):
    a = pol.act(obs)
    if i >= start:
        print(i, "logged", s["action"], "would", a, "y", round(obs["player"]["y"]), "tracks",
              {k: (round(v[0]), round(v[1])) for k, v in pol.tracks.items()} if hasattr(pol, "tracks") else "")
    obs = s["obs"]
