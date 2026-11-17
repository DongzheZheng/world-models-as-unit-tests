"""Read-only summary of runs/probe/*.json (no real-world interaction).  usage: python experiments/audit/probe_report.py"""
import json
from pathlib import Path

P = Path("runs/probe")


def short(v):
    return {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()
            if k not in ("per_seed", "n", "success", "lives_before")}


if (P / "seedrange.json").exists():
    d = json.loads((P / "seedrange.json").read_text())
    print("== seed-range check: final programs on seeds 0-99 (the range lives are drawn from)")
    for k, v in d.items():
        print(f"  {k:22s} n={v['n']:3d} success={v['success'] * 100:5.1f}%  {short(v)}")
    for key, seeds, label in [("push_T/aspire_v5", (11, 23, 37, 52, 68), "Push-T v5 on the 5 seeds flown as official lives"),
                              ("push_T/aspire_v1", (11, 23, 37, 52, 68), "Push-T v1 on the same seeds"),
                              ("push_T/aspire_v8", (11, 23, 37, 52, 68, 101, 202, 303, 404, 505), "Push-T v8 on all life seeds < 100.."),
                              ("drone_C/aspire_v11", (21, 22, 23, 24, 25, 31, 32, 33), "mission C v11 on life seeds 21-25, 31-33"),
                              ("drone_C/aspire_v14", (21, 22, 23, 24, 25, 31, 32, 33), "mission C v14 on the same seeds")]:
        if key in d:
            ps = d[key]["per_seed"]
            print(f"  {label}: {[ps.get(str(s)) for s in seeds]}")

if (P / "curves.json").exists():
    d = json.loads((P / "curves.json").read_text())
    print("\n== lives -> true success rate on unseen seeds (6000+)")
    task = None
    for k, v in d.items():
        t, name = k.split("/")
        if t != task:
            task = t
            print(f"-- {t}")
        extra = f"mean {v['mean']:.0f} median {v['median']:.0f}" if "mean" in v else short(v)
        print(f"  {name:24s} lives_before={v['lives_before']:2d}  success={v['success'] * 100:5.1f}%  n={v['n']}  {extra}")
