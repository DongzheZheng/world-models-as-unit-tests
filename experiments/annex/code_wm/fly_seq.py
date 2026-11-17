"""Fly calibration lives one after another (one worker each).
usage: python fly_seq.py <pilot>:<seed>:<factory>[:<schedule path>] ...   (schedule path -> kwargs {"path": ...})"""
import json
import sys

if __name__ == "__main__":
    from reallab.annex.parallel import evaluate
    for spec in sys.argv[1:]:
        parts = spec.split(":")
        pilot, seed = parts[0], int(parts[1])
        factory = parts[2] + ":" + parts[3]
        kw = {"path": parts[4]} if len(parts) > 4 else {}
        assert seed < 1000
        res = evaluate(factory, kw, seeds=[seed], purpose="calibration", pilot=pilot, n_workers=1)
        print(pilot, json.dumps(res), flush=True)
