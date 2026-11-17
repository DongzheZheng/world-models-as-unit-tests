"""Evaluate saved checkpoints (100 real episodes, seeds 1000-1099), record results, write curve.json (+ best GIF)."""
import argparse
import json
from pathlib import Path

import numpy as np

from cawm import results
from reallab.doom import parallel

OUT = Path("runs/doom/rl_baseline")
FACT = "experiments.doom.rl_baseline.ppo_real:make_policy"


def ev(name, workers, **kw):
    return parallel.evaluate(FACT, {"ckpt": str(OUT / "ckpt" / f"{name}.pt")}, "take_cover", list(range(1000, 1100)),
                             purpose="eval", tag=name, n_workers=workers, pilot="ppo_real", **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", default=None)
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    idx = [json.loads(l) for l in open(OUT / "ckpt_index.jsonl")]
    meta = {}
    for m in idx:
        meta[m["name"]] = m          # last entry wins
    done = json.load(open(OUT / "eval_done.json")) if (OUT / "eval_done.json").exists() else {}
    for name, m in meta.items():
        if (a.only and name not in a.only.split(",")) or name in done or name.startswith("ppo_real_final"):
            continue
        eps = ev(name, a.workers)
        rets = [e["ret"] for e in eps]
        results.record("doom", "take_cover", name, "model_free_rl", "real", eps, "ret",
                       real_steps_training=m["steps"], train_wall_s=m["wall_s"],
                       notes=f"{m['lives']} real episodes" + (" (no reuse of data)" if name.endswith("noreuse") else ""))
        done[name] = dict(real_episodes=m["lives"], real_steps=m["steps"], eval_mean=float(np.mean(rets)),
                          eval_sem=float(np.std(rets) / np.sqrt(len(rets))), train_wall_s=m["wall_s"])
        print(name, done[name], flush=True)
        json.dump(done, open(OUT / "eval_done.json", "w"), indent=1)
    rows = sorted([dict(name=k, **v) for k, v in done.items() if not k.endswith("noreuse")],
                  key=lambda r: r["real_episodes"])
    json.dump([{k: r[k] for k in ("real_episodes", "real_steps", "eval_mean", "eval_sem", "train_wall_s")} for r in rows],
              open(OUT / "curve.json", "w"), indent=1)
    hit = [r for r in rows if r["eval_mean"] >= 750]
    print("lives-to-750:", hit[0]["real_episodes"] if hit else "never", flush=True)
    if a.gif and rows:
        best = max(rows, key=lambda r: r["eval_mean"])["name"]
        ev_ = parallel.evaluate(FACT, {"ckpt": str(OUT / "ckpt" / f"{best}.pt")}, "take_cover", [1000], purpose="eval",
                                tag=best + "_gif", n_workers=1, pilot="ppo_real", record_first=1, record_format="gif",
                                video_dir=str(OUT))
        print("gif for", best, ev_[0]["ret"], flush=True)
