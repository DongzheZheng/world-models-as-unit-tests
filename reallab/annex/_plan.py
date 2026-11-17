"""Lab-side planners on the true (or a perturbed) simulator.  Internal: pilots must not read this module.

search(sim, family, goal, budget, ...)  per-episode cross-entropy search over a low-dimensional parametrised cast
                                         (Spin: circle then release; Cast: timed waypoints) on Sim; returns the best
                                         parameters, whether they succeed, the completion step and rollouts used.
Scores: wrap   -> success first, then side*W at the end (at rest), plus a little for an earlier finish;
        pocket -> success first, then minus the tip's distance to the target.
"""
from __future__ import annotations

import numpy as np

from reallab.annex import _limits as S
from reallab.annex._sim import Cast, Shifted, Spin, Spin2, cast_box


class Family:
    def __init__(self, kind: str, m: int = 3, dmax: float = 0.6, slow: bool = False):
        self.kind, self.m = kind, m
        if kind in ("spin", "spin2"):
            self.lo, self.hi = (Spin if kind == "spin" else Spin2).LO.copy(), (Spin if kind == "spin" else Spin2).HI.copy()
            if slow:
                self.hi[3] = 4.0          # angular speed small enough for the slow tool
                self.lo[3] = -4.0
                self.hi[5] = 3.0
        else:
            self.lo, self.hi = cast_box(m, dmax)

    def make(self, p, ep):
        if self.kind == "spin":
            return Spin(p, ep["post"], ep["side"])
        if self.kind == "spin2":
            return Spin2(p, ep["post"], ep["side"])
        return Cast(p, ep["post"], ep["side"], self.m)


def score(r, goal, sim):
    t = r[goal + "_t"]
    if t is not None:
        return 100.0 - 0.1 * t / S.CONTROL_HZ + (min(r["final_w"] - sim.w_min, 3.0) if goal == "wrap" else
                                                 -r["tip_dist"])
    if goal == "wrap":
        return r["final_w"] - (0.0 if r["at_rest"] else 1.0)
    side_pen = 0.0 if sim.pocket_w is None else 0.1 * max(0.0, sim.pocket_w - r["final_w"])
    return -r["tip_dist"] - side_pen - (0.0 if r["at_rest"] else 0.2)


def search(sim, fam: Family, goal: str, budget: int = 300, pop: int = 30, n_elite: int = 6, T: int = 200,
           rng=None, stop_on_success: bool = True, extra_after_success: int = 0, init=None, start_state=None,
           k0: int = 0, robust_k: int = 0, n_jit: int = 6, jit: float = 0.02, init_sd: float | None = None,
           absolute: bool = False):
    """Cross-entropy search.  With robust_k > 0, the up-to-robust_k best successful candidates are each re-run with
    n_jit jittered copies (each parameter moved by U(-jit, jit) times its range) and the one that succeeds most often
    under jitter is returned (ties: higher score)."""
    rng = rng or np.random.default_rng(0)
    ep = sim.ep
    lo, hi = fam.lo, fam.hi
    mu = (lo + hi) / 2 if init is None else np.asarray(init, float)
    sd = (hi - lo) / 2 if init_sd is None else init_sd * (hi - lo)
    best = (-1e9, None, None)
    used = 0
    first_success_at = None
    succ = []
    while used < budget:
        n = min(pop, budget - used)
        X = mu + sd * rng.standard_normal((n, len(lo)))
        X = np.clip(X, lo, hi)
        if used == 0 and init is None:
            X[: n // 2] = rng.uniform(lo, hi, (n // 2, len(lo)))
        elif used == 0:
            X[0] = np.clip(init, lo, hi)
        sc = np.zeros(n)
        for i in range(n):
            sched = fam.make(X[i], ep) if (k0 == 0 or absolute) else Shifted(fam.make(X[i], ep), k0)
            r = sim.rollout(sched, T, stop_when=(goal,), start_state=start_state, k0=k0)
            sc[i] = score(r, goal, sim)
            used += 1
            if sc[i] > best[0]:
                best = (sc[i], X[i].copy(), r)
            if r[goal + "_t"] is not None:
                succ.append((sc[i], X[i].copy(), r))
                if first_success_at is None:
                    first_success_at = used
            if first_success_at is not None and stop_on_success and used >= first_success_at + extra_after_success:
                break
        if first_success_at is not None and stop_on_success and used >= first_success_at + extra_after_success:
            break
        el = X[np.argsort(-sc)[:n_elite]]
        mu = el.mean(0)
        sd = np.maximum(el.std(0), (hi - lo) * 0.02)
    rob = None
    if robust_k and succ:
        succ.sort(key=lambda z: -z[0])
        cands = []
        for c in succ:
            if all(np.abs(c[1] - d[1]).max() > 1e-9 for d in cands):
                cands.append(c)
            if len(cands) >= robust_k:
                break
        best_rob = None
        for c in cands:
            ok = 0
            for _ in range(n_jit):
                q = np.clip(c[1] + jit * (hi - lo) * rng.uniform(-1, 1, len(lo)), lo, hi)
                sched = fam.make(q, ep) if (k0 == 0 or absolute) else Shifted(fam.make(q, ep), k0)
                rr = sim.rollout(sched, T, stop_when=(goal,), start_state=start_state, k0=k0)
                used += 1
                ok += rr[goal + "_t"] is not None
            key = (ok, c[0])
            if best_rob is None or key > best_rob[0]:
                best_rob = (key, c)
        best = best_rob[1]
        rob = best_rob[0][0] / n_jit
    r = best[2]
    return dict(robust_frac=rob, params=best[1], success=r[goal + "_t"] is not None, t_done=r[goal + "_t"], used=used,
                tip_final=r["tip_final"].tolist(),
                final_w=r["final_w"], tip_dist=r["tip_dist"], score=best[0])
