"""One record per (demo, scenario, controller): how it was obtained and how it did in the real world.

Every controller result is a JSON file under runs/<demo>/results/.  `python -m cawm.results`
collects all of them into runs/summary.json and prints one table per demo/scenario.

A record separates three kinds of cost that must never be mixed up:
  real_steps_calibration   real-world steps the world-model author looked at (shared by all code-world controllers)
  real_steps_training      real-world steps used to train THIS controller (0 for anything trained in the code world)
  wm_steps                 code-world steps used to train / tune this controller
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
# paradigm of the method that produced the controller
FAMILIES = ("classic", "model_free_rl", "neural_wm", "code_as_policy", "code_as_wm", "code_as_policy_in_code_world", "reference")
TRAINED_IN = ("code_world", "real", "none")


def _stats(xs: list[float]) -> dict:
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:        # e.g. completion time when no episode succeeded
        return dict(n=0, mean=None, std=None, sem=None, median=None, min=None, max=None)
    sd = statistics.pstdev(xs) if n > 1 else 0.0
    return dict(n=n, mean=statistics.fmean(xs), std=sd, sem=sd / math.sqrt(n) if n else 0.0,
                median=statistics.median(xs), min=min(xs), max=max(xs))


def record(demo: str, scenario: str, controller: str, family: str, trained_in: str, episodes: list[dict],
           metric: str, real_steps_calibration: int = 0, real_steps_training: int = 0, wm_steps: int = 0,
           train_wall_s: float = 0.0, other_metrics: tuple = (), notes: str = "", runs: Path = RUNS) -> dict:
    """Write the result of evaluating one controller on real-world episodes.

    episodes: the list returned by the lab's `evaluate` (one dict per episode, must contain `seed` and `metric`).
    metric: the key of the headline per-episode number (e.g. "ret", "success").
    other_metrics: further per-episode keys to summarise (booleans are averaged into rates).
    """
    assert family in FAMILIES and trained_in in TRAINED_IN
    rec = dict(
        demo=demo, scenario=scenario, controller=controller, family=family, trained_in=trained_in,
        real_steps_calibration=int(real_steps_calibration), real_steps_training=int(real_steps_training),
        wm_steps=int(wm_steps), train_wall_s=float(train_wall_s), metric=metric,
        eval=_stats([e[metric] for e in episodes]), seeds=[e["seed"] for e in episodes],
        other={k: _stats([e[k] for e in episodes if k in e and e[k] is not None]) for k in other_metrics},
        notes=notes, written=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    out = runs / demo / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{scenario}__{controller}.json").write_text(json.dumps(rec, indent=1))
    return rec


def collect(runs: Path = RUNS) -> list[dict]:
    recs = [json.loads(p.read_text()) for p in sorted(runs.glob("*/results/*.json"))]
    order = {f: i for i, f in enumerate(FAMILIES)}
    return sorted(recs, key=lambda r: (r["demo"], r["scenario"], order[r["family"]], r["controller"]))


def table(recs: list[dict]) -> str:
    lines, key = [], None
    for r in recs:
        if (r["demo"], r["scenario"]) != key:
            key = (r["demo"], r["scenario"])
            lines += ["", f"== {r['demo']} / {r['scenario']}   (metric: {r['metric']}, n={r['eval']['n']} real episodes)",
                      f"{'controller':30s} {'family':9s} {'trained in':10s} {'mean':>9s} {'± sem':>8s} {'median':>9s} "
                      f"{'real cal':>9s} {'real train':>10s} {'wm steps':>10s} {'train s':>8s}"]
        e = r["eval"]
        lines.append(f"{r['controller']:30s} {r['family']:9s} {r['trained_in']:10s} {e['mean']:9.2f} {e['sem']:8.2f} "
                     f"{e['median']:9.2f} {r['real_steps_calibration']:9d} {r['real_steps_training']:10d} "
                     f"{r['wm_steps']:10.3g} {r['train_wall_s']:8.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    recs = collect()
    (RUNS / "summary.json").write_text(json.dumps(recs, indent=1))
    print(table(recs))
