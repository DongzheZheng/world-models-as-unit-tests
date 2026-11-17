"""Lab-side reference numbers on the real installation (purpose=eval, seeds 1000..1049) -> runs/annex/_lab/baselines.json
(step logs under runs/annex/_lab/logs/<name>/, plans under runs/annex/_lab/plans/).

    python -m reallab.annex._baselines plan <priv|slow>          plan per eval seed in the lab simulator (12 workers)
    python -m reallab.annex._baselines run <name> [json kwargs]  random noop priv slow geo   (replay through api.py)
    python -m reallab.annex._baselines gifs                       two GIFs into runs/annex/_lab/gifs/
    python -m reallab.annex._baselines speed                      agent steps per second; 50 episodes on 12 workers
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path("runs/annex/_lab")
OUT = LAB / "baselines.json"
SEEDS = list(range(1000, 1050))


def _plan_one(args):
    seed, kind, TL = args
    from reallab.annex._plan import Family, search
    from reallab.annex._sim import Sim
    from reallab.annex import _limits as S
    if kind == "priv":
        sim = Sim()
        ep = sim.setup(seed)
        fam = Family("cast")
        r = search(sim, fam, "pocket", budget=1000, T=TL, rng=np.random.default_rng(seed), extra_after_success=60,
                   robust_k=5)
    else:
        sim = Sim(vmax=0.2)
        ep = sim.setup(seed)
        best = None
        for fam in (Family("cast", m=4, dmax=1.0), Family("spin", slow=True)):
            r = search(sim, fam, "pocket", budget=300, T=TL, rng=np.random.default_rng(seed))
            if best is None or r["score"] > best[0]["score"]:
                best = (r, fam)
        r, fam = best
    if r["params"] is None:
        return seed, None, False
    sched = fam.make(r["params"], ep)
    lo, hi = np.array(S.ZONE_LOW), np.array(S.ZONE_HIGH)
    cmds = [np.clip(sched(k), lo, hi).tolist() for k in range(TL)]
    return seed, cmds, bool(r["success"])


def classify(log):
    pts, tool = log["points"], log["tool_pos"]
    if len(log["rewards"]) and log["rewards"][-1] > 5.0:      # +10 on success; the shaped part is tiny
        return "success"
    rest = (np.linalg.norm(pts[-1, :, :2] - pts[-2, :, :2], axis=1).max() <= 0.02 / 50
            and np.linalg.norm(tool[-1] - tool[-2]) <= 0.02 / 50)
    d = float(np.linalg.norm(pts[-1, 0, :2] - log["target"][-1]))
    if not rest:
        return "still_moving_at_time_limit"
    return "at_rest_outside_target" if d > 0.10 else "other"


def summarise(name, res, extra):
    br = {}
    per = []
    for r in res:
        d = np.load(LAB / "logs" / name / f"ep_{r['seed']}.npz")
        c = classify(d)
        br[c] = br.get(c, 0) + 1
        per.append(dict(seed=r["seed"], outcome=c, steps=r["n_steps"], tip_dist=round(r["tip_dist"], 4)))
    return dict(n=len(res), success=float(np.mean([r["success"] for r in res])), breakdown=br,
                median_final_tip_dist=float(np.median([r["tip_dist"] for r in res])), per_seed=per, **extra)


def save(name, entry):
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur[name] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cur, indent=1))


def main():
    from reallab.annex import api
    TL = api.TIME_LIMIT
    cmd = sys.argv[1]
    if cmd == "plan":
        kind = sys.argv[2]
        t0 = time.time()
        with mp.get_context("spawn").Pool(12) as pool:
            res = pool.map(_plan_one, [(s, kind, TL) for s in SEEDS], chunksize=1)
        (LAB / "plans").mkdir(parents=True, exist_ok=True)
        (LAB / "plans" / f"{kind}.json").write_text(json.dumps(dict(
            plans={str(s): c for s, c, _ in res if c is not None},
            planned_success={str(s): ok for s, _, ok in res}, wall_s=time.time() - t0)))
        print(kind, "planned success in lab sim", np.mean([ok for _, _, ok in res]))
    elif cmd == "run":
        name = sys.argv[2]
        from reallab.annex._reference import lab_evaluate
        from reallab.annex.parallel import evaluate
        log_dir = str(LAB / "logs" / name)
        t0 = time.time()
        if name in ("random", "noop"):
            fac = "reallab.annex.parallel:random_policy" if name == "random" else "reallab.annex.parallel:noop_policy"
            kw = {"seed": 0} if name == "random" else {}
            res = evaluate(fac, kw, SEEDS, purpose="eval", tag=f"lab_ref_{name}", n_workers=12, log_dir=log_dir)
            extra = dict(factory=fac)
        elif name in ("priv", "slow"):
            plans = json.loads((LAB / "plans" / f"{name}.json").read_text())["plans"]
            res = lab_evaluate("reallab.annex._reference:replay_policy", {"plans": plans}, SEEDS, tag=f"lab_ref_{name}",
                               log_dir=log_dir, vmax=0.2 if name == "slow" else None)
            extra = dict(note="per-episode search in the true simulator, replayed" +
                         (" with the tool capped at 0.2 m/s" if name == "slow" else ""))
        elif name == "geo":
            consts = json.loads(sys.argv[3])
            res = lab_evaluate("reallab.annex._reference:geo_policy", {"consts": consts}, SEEDS, tag="lab_ref_geo",
                               log_dir=log_dir)
            extra = dict(consts=consts)
        else:
            raise SystemExit(__doc__)
        entry = summarise(name, res, dict(extra, wall_s=round(time.time() - t0, 1)))
        save(name, entry)
        print(name, entry["success"], entry["breakdown"])
    elif cmd == "resum":                     # re-classify outcomes from the saved step logs
        b = json.loads(OUT.read_text())
        for name, e in b.items():
            if "per_seed" not in e:
                continue
            br = {}
            for p in e["per_seed"]:
                p["outcome"] = classify(np.load(LAB / "logs" / name / f"ep_{p['seed']}.npz"))
                br[p["outcome"]] = br.get(p["outcome"], 0) + 1
            e["breakdown"] = br
            print(name, e["success"], br)
        OUT.write_text(json.dumps(b, indent=1))
    elif cmd == "gifs":
        from reallab.annex._render import render_log
        g = LAB / "gifs"
        g.mkdir(parents=True, exist_ok=True)
        b = json.loads(OUT.read_text())
        s_ok = next(p["seed"] for p in b["priv"]["per_seed"] if p["outcome"] == "success")
        render_log(str(LAB / "logs" / "priv" / f"ep_{s_ok}.npz"), str(g / "privileged_success.gif"),
                   f"privileged reference, eval seed {s_ok}")
        s_g = b["geo"]["per_seed"][0]["seed"]
        render_log(str(LAB / "logs" / "geo" / f"ep_{s_g}.npz"), str(g / "geometry_script.gif"),
                   f"geometry-scheduled script, eval seed {s_g} ({b['geo']['per_seed'][0]['outcome']})")
        print("gifs written")
    elif cmd == "speed":
        env = api.RealAnnex("smoke", "", "lab_speed", None)
        rng = np.random.default_rng(0)
        n, t0 = 0, time.time()
        for s in range(3):
            obs = env.reset(2000 + s)
            done = False
            while not done:
                obs, r, te, tr, info = env.step(rng.uniform(api.ZONE[0], api.ZONE[1]))
                n += 1
                done = te or tr
        sps = n / (time.time() - t0)
        env.close()
        from reallab.annex.parallel import evaluate
        t0 = time.time()
        evaluate("reallab.annex.parallel:random_policy", {"seed": 0}, list(range(2000, 2050)), purpose="smoke",
                 tag="lab_speed", n_workers=12)
        w = time.time() - t0
        save("speed", dict(agent_steps_per_s_one_process=round(sps, 1), seconds_50_episodes_12_workers=round(w, 1),
                           time_limit_steps=TL))
        print("speed", sps, w)


if __name__ == "__main__":
    main()
