"""Lab-side amended criterion 4: planning only in wrong models, executing in the true simulator (seeds 3000..3023).
Every planning model starts from what a pilot observes (the reset reading; later readings for re-planning), with the
true layout (post, target) - never from the true simulator state.  At most 12 worker processes.

    python -m reallab.annex._closed open     <scale> <draw>   open loop, one model with every parameter off by +-scale
    python -m reallab.annex._closed closed   <scale> <draw>   receding-horizon re-planning in that model
    python -m reallab.annex._closed ensemble <scale>          one open-loop cast chosen across 6 models within +-scale
Output: runs/annex/_lab/criteria/c4b_<mode>_<scale>_<draw>.json
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("runs/annex/_lab/criteria")
SEEDS = list(range(3000, 3024))
TL = 236
REPLAN_EVERY = 15
REPLAN_MAX_K = 150
REPLAN_BUDGET = 90          # rollouts per re-plan (3 CEM generations of 30), warm-started from the current plan
N_MODELS = 6


def model_phys(scale, draw, uniform=False):
    from reallab.annex import _internal
    p = _internal.physical()
    rng = np.random.default_rng([4242, int(draw), int(round(scale * 1000))])
    mult = {}
    for k in _internal.PERTURBABLE:
        f = 1.0 + (scale * float(rng.uniform(-1, 1)) if uniform else scale * float(rng.choice([-1.0, 1.0])))
        mult[k] = round(f, 4)
        p[k] *= f
    return p, mult


def _true(seed):
    from reallab.annex._sim import Sim
    t = Sim()
    t.setup(seed)
    return t


def _model_from_reading(phys, true, pts, prev_pts, tool, prev_tool):
    from reallab.annex._sim import Sim
    m = Sim(phys)
    m.ep = dict(true.ep)
    m.scene.place(true.ep["post"], true.ep["target"])
    m.scene.set_from_reading(pts, prev_pts, tool, prev_tool, 50.0)
    m.s0 = m.scene.get_state()
    return m


def _exec_true(true, fam, params, T):
    r = true.rollout(fam.make(params, true.ep), T, stop_when=("pocket",))
    return r["pocket_t"] is not None, r["tip_dist"]


def _open(args):
    seed, scale, draw = args
    from reallab.annex._plan import Family, search
    true = _true(seed)
    phys, mult = model_phys(scale, draw)
    m = _model_from_reading(phys, true, true.pts0, true.pts0, true.tool0, true.tool0)
    fam = Family("cast")
    r = search(m, fam, "pocket", budget=1000, T=TL, rng=np.random.default_rng(seed), extra_after_success=60,
               robust_k=5)
    ok, d = (False, None) if r["params"] is None else _exec_true(true, fam, r["params"], TL)
    return dict(seed=seed, model_success=r["success"], real_success=bool(ok), real_tip_dist=d, mult=mult)


def _closed(args):
    seed, scale, draw = args
    from reallab.annex._plan import Family, search
    from reallab.annex._sim import ZHI, ZLO, at_rest
    true = _true(seed)
    phys, mult = model_phys(scale, draw)
    m = _model_from_reading(phys, true, true.pts0, true.pts0, true.tool0, true.tool0)
    fam = Family("cast")
    rng = np.random.default_rng(seed)
    r = search(m, fam, "pocket", budget=1000, T=TL, rng=rng, extra_after_success=60, robust_k=5)
    p = r["params"]
    used = r["used"]
    sc = true.scene
    sc.set_state(true.s0)
    prev_p, prev_t = true.pts0, true.tool0
    pp, pt = prev_p, prev_t           # reading before the latest one
    ok, n_replans, t_done = False, 0, None
    for k in range(TL):
        if k > 0 and k % REPLAN_EVERY == 0 and k <= REPLAN_MAX_K and not at_rest(pp, prev_p, pt, prev_t):
            m.scene.set_from_reading(prev_p, pp, prev_t, pt, 50.0)
            st = m.scene.get_state()
            r2 = search(m, fam, "pocket", budget=REPLAN_BUDGET, pop=30, T=TL - k, rng=rng, start_state=st, k0=k, init=p,
                        init_sd=0.05, absolute=True, stop_on_success=False)
            p = r2["params"]
            used += r2["used"]
            n_replans += 1
        cmd = np.minimum(np.maximum(fam.make(p, true.ep)(k), ZLO), ZHI)
        sc.step(cmd)
        cur_p, cur_t = sc.tracked_points(), sc.tool_pos()
        if (at_rest(prev_p, cur_p, prev_t, cur_t) and np.all(cur_t >= ZLO - 0.01) and np.all(cur_t <= ZHI + 0.01)
                and np.linalg.norm(cur_p[0, :2] - true.ep["target"]) <= true.pocket_r):
            ok, t_done = True, k + 1
            break
        pp, pt, prev_p, prev_t = prev_p, prev_t, cur_p, cur_t
    d = float(np.linalg.norm(sc.tracked_points()[0, :2] - true.ep["target"]))
    return dict(seed=seed, model_success_initial=r["success"], real_success=ok, t_done=t_done, n_replans=n_replans,
                rollouts=used, real_tip_dist=d, mult=mult)


def _ensemble(args):
    seed, scale = args
    from reallab.annex._plan import Family, score, search
    true = _true(seed)
    models, mults = [], []
    for j in range(N_MODELS):
        phys, mult = model_phys(scale, 100 + j, uniform=True)
        models.append(_model_from_reading(phys, true, true.pts0, true.pts0, true.tool0, true.tool0))
        mults.append(mult)
    fam = Family("cast")
    rng = np.random.default_rng(seed)
    r0 = search(models[0], fam, "pocket", budget=400, T=TL, rng=rng, extra_after_success=30)
    lo, hi = fam.lo, fam.hi
    mu = np.array(r0["params"])
    sd = 0.1 * (hi - lo)

    def ens(x):
        rs = [mm.rollout(fam.make(x, mm.ep), TL, stop_when=("pocket",)) for mm in models]
        nsucc = sum(rr["pocket_t"] is not None for rr in rs)
        return nsucc + 0.01 * float(np.mean([max(score(rr, "pocket", mm), -1.0) for rr, mm in zip(rs, models)])), nsucc

    best = (-1e9, mu.copy(), 0)
    v, ns = ens(mu)
    best = (v, mu.copy(), ns)
    used = r0["used"] + N_MODELS
    for g in range(6):
        X = np.clip(mu + sd * rng.standard_normal((24, len(mu))), lo, hi)
        X[0] = best[1]
        vals = []
        for x in X:
            v, ns = ens(x)
            used += N_MODELS
            vals.append(v)
            if v > best[0]:
                best = (v, x.copy(), ns)
        el = X[np.argsort(-np.array(vals))[:6]]
        mu, sd = el.mean(0), np.maximum(el.std(0), 0.01 * (hi - lo))
    ok, d = _exec_true(true, fam, best[1], TL)
    return dict(seed=seed, models_succeeding=int(best[2]), real_success=bool(ok), real_tip_dist=d, rollouts=used,
                mults=mults)


def main():
    mode, scale = sys.argv[1], float(sys.argv[2])
    draw = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    fn = dict(open=_open, closed=_closed, ensemble=_ensemble)[mode]
    tasks = [(s, scale) if mode == "ensemble" else (s, scale, draw) for s in SEEDS]
    t0 = time.time()
    with mp.get_context("spawn").Pool(12) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    summ = dict(mode=mode, scale=scale, draw=draw, n=len(res), real_success=float(np.mean([r["real_success"] for r in res])),
                median_real_tip_dist=float(np.median([r["real_tip_dist"] for r in res if r["real_tip_dist"] is not None])),
                wall_s=round(time.time() - t0, 1))
    if mode == "closed":
        summ.update(replan_every=REPLAN_EVERY, replan_budget=REPLAN_BUDGET, mean_replans=float(np.mean([r["n_replans"] for r in res])),
                    mean_rollouts=float(np.mean([r["rollouts"] for r in res])))
    if mode == "ensemble":
        summ.update(n_models=N_MODELS, mean_models_succeeding=float(np.mean([r["models_succeeding"] for r in res])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"c4b_{mode}_{scale}_{draw}.json").write_text(json.dumps(dict(summary=summ, per_seed=res), indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
