"""Lab-side acceptance criteria runs (smoke seeds 3000..3049 by default, true simulator unless stated).
Outputs JSON under runs/annex/_lab/criteria/.  At most 12 worker processes.

    python -m reallab.annex._criteria priv   [first n budget family]   C2: per-episode search, T = 200 steps
    python -m reallab.annex._criteria slow   TL [first n budget]       C1: best path search with the tool capped at 0.2 m/s
    python -m reallab.annex._criteria twice  TL [first n budget]       C3: cast, look (wait for rest), cast again, within TL
    python -m reallab.annex._criteria perturb TL draw [first n budget] C4: plan in a +-10% simulator, execute in the true one
    python -m reallab.annex._criteria geo    TL [first n]              C5a: optimise the geometry-scheduled script
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("runs/annex/_lab/criteria")
FAMILY = os.environ.get("CAWM_ANNEX_FAMILY", "spin")
ROBUST = int(os.environ.get("CAWM_ANNEX_ROBUST", "5"))
PSCALE = float(os.environ.get("CAWM_ANNEX_PSCALE", "0.1"))   # relative error of every hidden parameter (criterion 4)


def perturbed_phys(draw: int) -> dict:
    from reallab.annex import _internal
    p = _internal.physical()
    rng = np.random.default_rng([4242, draw])
    mult = {}
    for k in _internal.PERTURBABLE:
        f = 1.0 + PSCALE * float(rng.choice([-1.0, 1.0]))
        mult[k] = f
        p[k] = p[k] * f
    return p, mult


def _priv(args):
    seed, budget, family, T = args
    from reallab.annex._plan import Family, search
    from reallab.annex._sim import Sim
    sim = Sim()
    ep = sim.setup(seed)
    fam = Family(family)
    t0 = time.time()
    r = search(sim, fam, "pocket", budget=budget, T=T, rng=np.random.default_rng(seed), extra_after_success=60,
               robust_k=ROBUST)
    out = dict(seed=seed, dist=ep["dist"], bearing=ep["bearing"], success=r["success"], t_done=r["t_done"],
               used=r["used"], tip_dist=r["tip_dist"], family=family, robust_frac=r["robust_frac"],
               params=None if r["params"] is None else [float(x) for x in r["params"]], wall=time.time() - t0)
    if r["success"]:
        rng = np.random.default_rng([seed, 5])
        base = np.array(r["tip_final"])
        mv, ok = [], []
        for _ in range(10):
            q = np.clip(r["params"] * rng.uniform(0.95, 1.05, len(r["params"])), fam.lo, fam.hi)
            rr = sim.rollout(fam.make(q, ep), T, stop_when=())
            mv.append(float(np.linalg.norm(rr["tip_final"] - base)))
            ok.append(rr["pocket_t"] is not None)
        out.update(sens_median_m=float(np.median(mv)), sens_p90_m=float(np.percentile(mv, 90)),
                   sens_success=float(np.mean(ok)))
    return out


def _slow(args):
    seed, budget, TL = args
    from reallab.annex._plan import Family, search
    from reallab.annex._sim import Sim
    sim = Sim(vmax=0.2)
    sim.setup(seed)
    best = None
    for fam in (Family("cast", m=4, dmax=1.0), Family("spin", slow=True)):
        r = search(sim, fam, "pocket", budget=budget, T=TL, rng=np.random.default_rng(seed))
        if best is None or r["score"] > best["score"]:
            best = dict(r, family=fam.kind)
    return dict(seed=seed, success=best["success"], tip_dist=best["tip_dist"], family=best["family"],
                params=None if best["params"] is None else [float(x) for x in best["params"]])


def _twice(args):
    seed, budget, TL, prev = args
    from reallab.annex._plan import Family, search
    from reallab.annex._sim import Sim
    sim = Sim()
    ep = sim.setup(seed)
    fam = Family(prev["family"])
    if prev["params"] is None:
        return dict(seed=seed, single=False, twice=False)
    r1 = sim.rollout(fam.make(np.array(prev["params"]), ep), TL, stop_when=("pocket",))
    single = r1["pocket_t"] is not None
    if single:
        return dict(seed=seed, single=True, twice=True, k_look=None)
    k = r1["n"]
    left = TL - k
    if left < 5:
        return dict(seed=seed, single=False, twice=False, k_look=k)
    st = sim.scene.get_state()
    r2 = search(sim, Family(prev["family"]), "pocket", budget=budget, T=left, rng=np.random.default_rng(seed + 7),
                start_state=st, k0=k)
    return dict(seed=seed, single=False, twice=bool(r2["success"]), k_look=k, left=left)


def _perturb(args):
    seed, budget, TL, draw, family = args
    from reallab.annex._plan import Family, search
    from reallab.annex._sim import Sim
    p, mult = perturbed_phys(draw)
    sim_m = Sim(p)
    ep = sim_m.setup(seed)
    fam = Family(family)
    r = search(sim_m, fam, "pocket", budget=budget, T=TL, rng=np.random.default_rng(seed), extra_after_success=60,
               robust_k=ROBUST)
    real = Sim()
    real.setup(seed)
    ok = False
    if r["params"] is not None:
        rr = real.rollout(fam.make(r["params"], real.ep), TL, stop_when=("pocket",))
        ok = rr["pocket_t"] is not None
        td = rr["tip_dist"]
    return dict(seed=seed, model_success=r["success"], real_success=bool(ok), robust_frac=r["robust_frac"],
                real_tip_dist=float(td) if r["params"] is not None else None, mult=mult)


_G = {}


def _geo_init(seeds, TL):
    from reallab.annex._sim import Sim
    _G["sims"] = []
    for s in seeds:
        sim = Sim()
        sim.setup(s)
        _G["sims"].append((s, sim, dict(sim.ep), sim.s0))
    _G["TL"] = TL


def _geo_eval(consts_list):
    from reallab.annex._sim import Cast, geo_params
    out = []
    for consts in consts_list:
        row = []
        for s, sim, ep, s0 in _G["sims"]:
            sim.ep, sim.s0 = ep, s0
            p = geo_params(consts, np.linalg.norm(ep["post"] - np.array([0.425, 0.0])))
            r = sim.rollout(Cast(p, ep["post"], 1, 3), _G["TL"], stop_when=("pocket",))
            row.append((r["pocket_t"] is not None, r["tip_dist"]))
        out.append(row)
    return out


def geo_optimise(seeds, TL, gens=12, pop=20, scales=(0.8, 1.25)):
    """CEM over the 11 script constants, objective = success rate (+ small closeness bonus) on `seeds`."""
    from reallab.annex._sim import GEO_HI, GEO_LO
    nw = min(12, len(seeds))
    groups = [seeds[i::nw] for i in range(nw)]
    pools = [mp.get_context("spawn").Pool(1, initializer=_geo_init, initargs=(g, TL)) for g in groups]
    rng = np.random.default_rng(11)
    mu, sd = (GEO_LO + GEO_HI) / 2, (GEO_HI - GEO_LO) / 2
    best = (-1, None, None)

    def evaluate(cands):
        parts = [p.apply_async(_geo_eval, (cands,)) for p in pools]
        rows = [pr.get() for pr in parts]
        succ = np.zeros(len(cands))
        close = np.zeros(len(cands))
        for g in rows:
            for i, row in enumerate(g):
                succ[i] += sum(a for a, _ in row)
                close[i] += sum(min(d, 0.5) for _, d in row)
        return succ / len(seeds), close / len(seeds)

    hist = []
    for g in range(gens):
        X = np.clip(mu + sd * rng.standard_normal((pop, len(mu))), GEO_LO, GEO_HI)
        if g == 0:
            X[: pop // 2] = rng.uniform(GEO_LO, GEO_HI, (pop // 2, len(mu)))
        sr, cl = evaluate([x.tolist() for x in X])
        sc = sr - 0.2 * cl
        i = int(np.argmax(sc))
        if sc[i] > best[0]:
            best = (sc[i], X[i].copy(), float(sr[i]))
        hist.append(float(sr.max()))
        el = X[np.argsort(-sc)[:5]]
        mu, sd = el.mean(0), np.maximum(el.std(0), (GEO_HI - GEO_LO) * 0.02)
        print("geo gen", g, "best success", round(float(sr.max()), 3), flush=True)
    consts = best[1]
    scaled = {}
    for f in scales:
        sr, _ = evaluate([(consts * f).tolist()])
        scaled[str(f)] = float(sr[0])
    for p in pools:
        p.close()
        p.join()
    return dict(consts=consts.tolist(), success=best[2], scaled=scaled, history=hist)


def _pool(fn, tasks):
    with mp.get_context("spawn").Pool(min(12, len(tasks))) as pool:
        return pool.map(fn, tasks, chunksize=1)


def main():
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if cmd == "priv":
        first, n, budget = (int(x) for x in (sys.argv[2:5] if len(sys.argv) > 4 else (3000, 50, 1000)))
        family = sys.argv[5] if len(sys.argv) > 5 else FAMILY
        res = _pool(_priv, [(s, budget, family, int(os.environ.get("CAWM_ANNEX_T", "200"))) for s in range(first, first + n)])
        ts = [r["t_done"] for r in res if r["success"]]
        p90 = float(np.percentile(ts, 90)) if ts else None
        summ = dict(success=float(np.mean([r["success"] for r in res])), n=len(res), p90_steps=p90,
                    TL_rule=int(np.ceil(1.3 * p90)) if p90 else None, mean_used=float(np.mean([r["used"] for r in res])),
                    sens_median_m=float(np.median([r["sens_median_m"] for r in res if "sens_median_m" in r] or [np.nan])),
                    sens_p90_m=float(np.median([r["sens_p90_m"] for r in res if "sens_p90_m" in r] or [np.nan])),
                    sens_success=float(np.mean([r["sens_success"] for r in res if "sens_success" in r] or [np.nan])),
                    wall_s=time.time() - t0)
        name = f"priv_{first}_{n}.json"
    elif cmd == "slow":
        TL = int(sys.argv[2])
        first, n, budget = (int(x) for x in (sys.argv[3:6] if len(sys.argv) > 5 else (3000, 50, 300)))
        res = _pool(_slow, [(s, budget, TL) for s in range(first, first + n)])
        summ = dict(success=float(np.mean([r["success"] for r in res])), TL=TL, budget_per_family=budget,
                    median_tip_dist=float(np.median([r["tip_dist"] for r in res])), wall_s=time.time() - t0)
        name = f"slow_{first}_{n}_TL{TL}.json"
    elif cmd == "twice":
        TL = int(sys.argv[2])
        first, n, budget = (int(x) for x in (sys.argv[3:6] if len(sys.argv) > 5 else (3000, 50, 400)))
        prev = {r["seed"]: r for r in json.loads((OUT / f"priv_{first}_{n}.json").read_text())["per_seed"]}
        res = _pool(_twice, [(s, budget, TL, prev[s]) for s in range(first, first + n)])
        summ = dict(single=float(np.mean([r["single"] for r in res])), twice=float(np.mean([r["twice"] for r in res])),
                    TL=TL, wall_s=time.time() - t0)
        name = f"twice_{first}_{n}_TL{TL}.json"
    elif cmd == "perturb":
        TL, draw = int(sys.argv[2]), int(sys.argv[3])
        first, n, budget = (int(x) for x in (sys.argv[4:7] if len(sys.argv) > 6 else (3000, 50, 1000)))
        res = _pool(_perturb, [(s, budget, TL, draw, FAMILY) for s in range(first, first + n)])
        summ = dict(draw=draw, mult=res[0]["mult"], model_success=float(np.mean([r["model_success"] for r in res])),
                    real_success=float(np.mean([r["real_success"] for r in res])), TL=TL, wall_s=time.time() - t0)
        summ["scale"] = PSCALE
        name = f"perturb_{first}_{n}_TL{TL}_d{draw}" + ("" if PSCALE == 0.1 else f"_s{PSCALE}") + ".json"
    elif cmd == "bins":
        d = json.loads((OUT / "priv_3000_50.json").read_text())["per_seed"]
        dist = np.array([r["dist"] for r in d])
        bear = np.array([r["bearing"] for r in d])
        ok = np.array([r["success"] for r in d], float)
        summ = dict(by_distance={f"{a:.2f}-{b:.2f}": dict(n=int(((dist >= a) & (dist < b)).sum()),
                                                            success=float(ok[(dist >= a) & (dist < b)].mean()))
                                 for a, b in ((0.40, 0.45), (0.45, 0.50), (0.50, 0.56))},
                    by_abs_bearing={f"{a:.2f}-{b:.2f}": dict(n=int(((abs(bear) >= a) & (abs(bear) < b)).sum()),
                                                               success=float(ok[(abs(bear) >= a) & (abs(bear) < b)].mean()))
                                    for a, b in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.76))})
        res = []
        name = "priv_bins.json"
    elif cmd == "geo":
        TL = int(sys.argv[2])
        first, n = (int(x) for x in (sys.argv[3:5] if len(sys.argv) > 4 else (3000, 50)))
        summ = geo_optimise(list(range(first, first + n)), TL)
        summ["wall_s"] = time.time() - t0
        res = []
        name = f"geo_{first}_{n}_TL{TL}.json"
    else:
        raise SystemExit(__doc__)
    (OUT / name).write_text(json.dumps(dict(summary=summ, per_seed=res), indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
