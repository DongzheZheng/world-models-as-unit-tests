#!/usr/bin/env python
"""Re-run the frozen controllers of the study and compare with the recorded results.

    python reproduce.py list
        every registered (task, controller) with its recorded result, and the model-based (world) harnesses
    python reproduce.py real <task> <controller> [--seeds a-b | --episodes n] [--workers n] [--log-dir DIR]
        re-run that frozen controller in the environment on the evaluation seeds of its recorded result and print
        the reproduced number next to the recorded one (also on the same seeds, episode by episode, when the
        per-episode record is in runs/)
    python reproduce.py world <task> [program ...] [--episodes n] [--seeds a-b] [--workers n] [--device cpu|cuda]
        re-run the model-based evaluation: programs scored inside a world-model agent's code world, zero real steps

Tasks: doom, drone_A, drone_B, drone_C, pusht_T, pusht_L, cable, rope   (world mode: doom, drone, pusht, cable, rope;
drone_A/drone_C and pusht_T/pusht_L select one half of a screening).  Run from anywhere; the script works in the
repository root (several modules read files under runs/ by relative path).

Every real episode is metered by cawm.ledger into runs/ledger.jsonl (purpose "eval"; reset-only snapshots "smoke").
Outputs of a run go to runs/reproduce/ (ignored by git).  Worker processes: default 8, at most 12.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_WORKERS = 12
DEFAULT_WORKERS = 8

# --------------------------------------------------------------------------------------------------------------------
# Registry of frozen controllers.  Factories and keyword arguments are exactly those of the recorded evaluations
# (the agents' own eval scripts and experiments/audit/*.py); `episodes` is the per-episode record of that evaluation.
# --------------------------------------------------------------------------------------------------------------------
DOOM_CW_KW = dict(horizon=24, margin=2.0, w_center=0.3)          # experiments/doom/code_wm/final_eval.py
STATION_CW_KW = {"clear": 0.09, "load_soft": 3.0, "load_hard": 6.0}  # experiments/station/code_wm/official_eval.py
DRONE_SITE = dict(command_mode="ctbr")

TASKS = {   # task -> (lab, demo, scenario, extra keyword arguments of the lab's evaluate())
    "doom": ("doom", "doom", "take_cover", {}),
    "drone_A": ("drone", "drone", "site_A", dict(mission="A")),
    "drone_B": ("drone", "drone", "site_B", dict(mission="B")),
    "drone_C": ("drone", "drone", "site_C", dict(mission="C")),
    "pusht_T": ("pusht", "pusht", "pusht", dict(object="T")),
    "pusht_L": ("pusht", "pusht", "pusht_L", dict(object="L")),
    "cable": ("station", "station", "station", {}),
    "rope": ("annex", "annex", "annex", {}),
}
ALIASES = {"pusht": "pusht_T", "station": "cable", "annex": "rope", "take_cover": "doom"}


def _e(task, controller, factory, kwargs=None, episodes=None, env=None, what=""):
    return dict(task=task, controller=controller, factory=factory, kwargs=kwargs or {}, episodes=episodes,
                env=env or {}, what=what)


REGISTRY = [
    # ---- Doom (take cover) --------------------------------------------------------------------------------------
    _e("doom", "aspire_style", "pilots.aspire_doom.policy_v5:make_policy", episodes="runs/doom/aspire_style/eval_aspire_style.json",
       what="Code-as-Policies agent, submitted program"),
    _e("doom", "code_wm", "experiments.doom.code_wm.planner:planner", DOOM_CW_KW, episodes="runs/doom/code_wm/eval.json",
       what="world-model agent, controller chosen in its code world"),
    *[_e("doom", f"ppo_real_{n}lives", "experiments.doom.rl_baseline.ppo_real:make_policy",
         {"ckpt": f"runs/doom/rl_baseline/ckpt/ppo_real_{n}lives.pt"}, what="baseline: PPO trained in the real environment")
      for n in (5, 20, 50, 100, 200, 250, 300, 350, 400, 450, 500, 1000, 2000, 4672)],
    _e("doom", "ppo_real_5lives_noreuse", "experiments.doom.rl_baseline.ppo_real:make_policy",
       {"ckpt": "runs/doom/rl_baseline/ckpt/ppo_real_5lives_noreuse.pt"}, what="baseline: PPO, 5 lives, no data reuse"),
    # ---- Drone, mission A ---------------------------------------------------------------------------------------
    _e("drone_A", "aspire_style", "pilots.aspire_drone.policy_v5:make_policy", env=DRONE_SITE,
       episodes="runs/drone/aspire_style/eval_aspire_style.json", what="Code-as-Policies agent, submitted program"),
    _e("drone_A", "code_wm", "experiments.drone.code_wm.controller:make_pilot",
       ("runs/drone/code_wm/cfg/eval_code_wm.json", "pilot_kwargs"), env=DRONE_SITE,
       episodes="runs/drone/code_wm/eval_code_wm_rows.json", what="world-model agent's controller"),
    *[_e("drone_A", f"ppo_real_{n}lives", "experiments.drone.baselines.common:rl_policy", {"ckpt": ck},
         episodes=f"runs/drone/baselines_rl/evals/ppo_real_{n}lives.json", what="baseline: PPO trained in the real environment")
      for n, ck in [(5, "runs/drone/baselines_rl/ppo5/ppo_5lives.pt"), (395, "runs/drone/baselines_rl/oracle/ckpt_ep395.pt"),
                    (1137, "runs/drone/baselines_rl/oracle/ckpt_ep1137.pt"), (3065, "runs/drone/baselines_rl/oracle/ckpt_ep3065.pt"),
                    (10082, "runs/drone/baselines_rl/oracle/ckpt_ep10082.pt"), (14811, "runs/drone/baselines_rl/oracle/final.pt")]],
    _e("drone_A", "neural_wm_5lives", "experiments.drone.baselines.wm:wm_policy",
       {"ckpt": "runs/drone/baselines_rl/neural_wm/model_final.pt"}, episodes="runs/drone/baselines_rl/evals/neural_wm_5lives.json",
       what="baseline: neural world model (MLP ensemble) + MPPI, 5 lives"),
    # ---- Drone, mission B (zero mission-B flights) --------------------------------------------------------------
    _e("drone_B", "aspire_style", "pilots.aspire_drone.policy_v6:make_policy", env=DRONE_SITE,
       episodes="runs/drone/aspire_style/eval_aspire_style_B.json", what="Code-as-Policies agent, submitted program"),
    _e("drone_B", "code_wm", "experiments.drone.code_wm.controller:make_pilot",
       ("runs/drone/code_wm/cfg/eval_code_wm_B.json", "pilot_kwargs"), env=DRONE_SITE,
       episodes="runs/drone/code_wm/eval_code_wm_B_rows.json", what="world-model agent's controller"),
    # ---- Drone, mission C (other aircraft) ----------------------------------------------------------------------
    _e("drone_C", "aspire_style", "pilots.aspire_drone.policy_v11:make_policy", env=DRONE_SITE,
       episodes="runs/drone/aspire_style/eval_aspire_style_C.json", what="Code-as-Policies agent, submitted program"),
    _e("drone_C", "aspire_style_extended", "pilots.aspire_drone.policy_v14:make_policy", env=DRONE_SITE,
       episodes="runs/drone/aspire_style/eval_aspire_style_extended_C.json", what="Code-as-Policies agent, extended run (20 lives)"),
    _e("drone_C", "cap_in_code_world", "pilots.aspire_drone.policy_v15:make_policy", env=DRONE_SITE,
       episodes="runs/drone/aspire_style/eval_cap_in_code_world_C.json",
       what="Code as Policies with the world model as its tests"),
    _e("drone_C", "code_wm", "experiments.drone.code_wm.controller:make_pilot",
       ("runs/drone/code_wm/cfg/eval_code_wm_C.json", "pilot_kwargs"), env=DRONE_SITE,
       episodes="runs/drone/code_wm/eval_code_wm_C_rows.json", what="world-model agent's controller"),
    # ---- Push-T ---------------------------------------------------------------------------------------------------
    _e("pusht_T", "aspire_style", "pilots.aspire_pusht.policy_v5:make_policy",
       episodes="runs/pusht/aspire_style/eval_aspire_style/episodes.json", what="Code-as-Policies agent, submitted program"),
    _e("pusht_T", "aspire_style_extended", "pilots.aspire_pusht.policy_v8:make_policy",
       episodes="runs/pusht/aspire_style/eval_aspire_style_extended/episodes.json", what="Code-as-Policies agent, extended run (13 lives)"),
    _e("pusht_T", "code_wm", "experiments.pusht.code_wm.controller:mpc_policy",
       ("runs/pusht/code_wm/eval_code_wm.json", "cfg", "planner"), episodes="runs/pusht/code_wm/eval_code_wm.json",
       what="world-model agent's controller (sampling MPC in its code world)"),
    # ---- Push-L ---------------------------------------------------------------------------------------------------
    _e("pusht_L", "aspire_style", "pilots.aspire_pusht.policy_v10:make_policy",
       episodes="runs/pusht/aspire_style/eval_L_aspire_style/episodes.json", what="Code-as-Policies agent, submitted program"),
    _e("pusht_L", "aspire_style_extended", "pilots.aspire_pusht.policy_v15:make_policy",
       episodes="runs/pusht/aspire_style/eval_L_aspire_style_extended/episodes.json", what="Code-as-Policies agent, extended run (8 L lives)"),
    _e("pusht_L", "cap_in_code_world", "pilots.aspire_pusht.policy_v17:make_policy",
       episodes="runs/pusht/aspire_style/eval_L_cap_in_code_world/episodes.json",
       what="Code as Policies with the world model as its tests"),
    _e("pusht_L", "code_wm", "experiments.pusht.code_wm.controller:mpc_policy",
       ("runs/pusht/code_wm/eval_L_code_wm.json", "cfg", "planner"), episodes="runs/pusht/code_wm/eval_L_code_wm.json",
       what="world-model agent's controller (sampling MPC in its code world)"),
    # ---- Cable routing (station) ----------------------------------------------------------------------------------
    _e("cable", "aspire_style", "pilots.aspire_station.policy_v3:make_policy",
       episodes="runs/station/aspire_style/eval_aspire_style_v3.json", what="Code-as-Policies agent, submitted program"),
    _e("cable", "aspire_blind", "pilots.aspire_blind.policy_v2:make_policy",
       episodes="runs/station/aspire_blind/eval_aspire_blind.json", what="Code-as-Policies agent without observation logs"),
    _e("cable", "code_wm", "experiments.station.code_wm.policy:make_policy", STATION_CW_KW,
       episodes="runs/station/code_wm/eval_result.json", what="world-model agent's controller"),
    # ---- Rope throw (annex) ---------------------------------------------------------------------------------------
    _e("rope", "aspire_style", "pilots.aspire_annex.policy_v6:make_policy",
       episodes="runs/annex/aspire_style/eval_logs/results.json", what="Code-as-Policies agent, submitted program"),
    _e("rope", "aspire_style_extended", "pilots.aspire_annex.policy_v19:make_policy",
       episodes="runs/annex/aspire_style/eval_logs_extended_final/results.json", what="Code-as-Policies agent, extended run (19 lives)"),
    _e("rope", "code_wm", "experiments.annex.code_wm.throw:make_final",
       episodes="runs/annex/code_wm/eval_code_wm.json", what="world-model agent's controller (CEM planning in its code world, GPU)"),
    _e("rope", "code_wm_extended", "experiments.annex.code_wm.throw_v2:make_robust",
       episodes="runs/annex/code_wm/eval_code_wm_extended.json", what="world-model agent, extended run (world v2, GPU)"),
]


def _norm_task(t):
    return ALIASES.get(t, t)


def _resolve_kwargs(kw):
    """kwargs are a dict, or (json file, key, key, ...) pointing at the dict used in the recorded evaluation."""
    if isinstance(kw, tuple):
        d = json.loads((ROOT / kw[0]).read_text())
        for k in kw[1:]:
            d = d[k]
        return d
    return kw


def _record(entry):
    lab, demo, scen, _ = TASKS[entry["task"]]
    p = ROOT / "runs" / demo / "results" / f"{scen}__{entry['controller']}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _recorded_episodes(entry):
    if not entry["episodes"]:
        return None
    p = ROOT / entry["episodes"]
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if isinstance(d, dict):
        d = d.get("episodes", d.get("rows"))
    return {int(e["seed"]): e for e in d} if d else None


def _metric(entry):
    return "ret" if TASKS[entry["task"]][0] == "doom" else "success"


def _parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _workers(n):
    return max(1, min(MAX_WORKERS, int(n)))


def _fmt(metric, x):
    if x is None:
        return "-"
    return f"{x:9.2f}" if metric == "ret" else f"{100 * x:6.1f}%"


def _save(kind, name, data):
    out = ROOT / "runs" / "reproduce"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{kind}_{name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(data, indent=1, default=float))
    return p


# --------------------------------------------------------------------------------------------------------------------
# real
# --------------------------------------------------------------------------------------------------------------------
def run_real(entry, seeds, workers, log_dir=None):
    lab, demo, scen, env = TASKS[entry["task"]]
    fac, kw = entry["factory"], _resolve_kwargs(entry["kwargs"])
    env = {**env, **entry["env"]}
    tag = f"reproduce_{entry['task']}_{entry['controller']}"
    if lab == "doom":
        from reallab.doom.parallel import evaluate
        return evaluate(fac, kw, "take_cover", seeds, purpose="eval", tag=tag, n_workers=workers, log_dir=log_dir,
                        pilot="reproduce")
    if lab == "drone":
        from reallab.drone.parallel import evaluate
        return evaluate(fac, kw, "site", seeds, purpose="eval", tag=tag, n_workers=workers, log_dir=log_dir,
                        pilot="reproduce", **env)
    if lab == "pusht":
        from reallab.pusht.parallel import evaluate
        return evaluate(fac, kw, seeds=seeds, purpose="eval", pilot="reproduce", tag=tag, n_workers=workers,
                        log_dir=log_dir, **env)
    if lab == "station":
        from reallab.station.parallel import evaluate
        return evaluate(fac, kw, seeds=seeds, purpose="eval", pilot="reproduce", tag=tag, n_workers=workers, log_dir=log_dir)
    if lab == "annex":
        from reallab.annex.parallel import evaluate
        return evaluate(fac, kw, seeds=seeds, purpose="eval", pilot="reproduce", tag=tag, n_workers=workers, log_dir=log_dir)
    raise ValueError(lab)


def cmd_real(a):
    task = _norm_task(a.task)
    hits = [e for e in REGISTRY if e["task"] == task and e["controller"] == a.controller]
    if not hits:
        sys.exit(f"no registered controller {a.controller!r} for task {task!r}; see `python reproduce.py list`")
    entry = hits[0]
    rec, eps_rec, metric = _record(entry), _recorded_episodes(entry), _metric(entry)
    seeds = _parse_seeds(a.seeds) if a.seeds else (rec["seeds"] if rec else list(range(1000, 1050)))
    if a.episodes:
        seeds = seeds[:a.episodes]
    w = _workers(a.workers)
    print(f"{task} / {entry['controller']}: {entry['factory']} {json.dumps(_resolve_kwargs(entry['kwargs']))}")
    print(f"  {len(seeds)} real episodes, seeds {seeds[0]}..{seeds[-1]}, {w} workers", flush=True)
    t0 = time.time()
    eps = run_real(entry, seeds, w, a.log_dir)
    wall = time.time() - t0
    vals = [float(e[metric]) for e in eps]
    mean = statistics.fmean(vals)
    line = f"  reproduced {_fmt(metric, mean)}"
    if metric == "success":
        line += f"  ({int(sum(vals))}/{len(vals)})"
    same = None
    if eps_rec and all(s in eps_rec for s in seeds):
        rv = [float(eps_rec[s][metric]) for s in seeds]
        same = statistics.fmean(rv)
        agree = sum((abs(x - y) < 1e-9) for x, y in zip(vals, rv))
        line += f"   recorded on the same seeds {_fmt(metric, same)}"
        if metric == "success":
            line += f"  ({int(sum(rv))}/{len(rv)})"
        line += f"   identical episodes {agree}/{len(seeds)}"
    if rec:
        line += f"   recorded (all {rec['eval']['n']} seeds) {_fmt(metric, rec['eval']['mean'])}"
    print(line + f"   wall {wall:.0f} s", flush=True)
    p = _save("real", f"{task}_{entry['controller']}", dict(task=task, controller=entry["controller"],
                                                            factory=entry["factory"], kwargs=_resolve_kwargs(entry["kwargs"]),
                                                            seeds=seeds, reproduced=mean, recorded_same_seeds=same,
                                                            recorded=rec["eval"]["mean"] if rec else None, wall_s=wall,
                                                            episodes=eps))
    print(f"  -> {p.relative_to(ROOT)}")


# --------------------------------------------------------------------------------------------------------------------
# world (model-based evaluation: programs scored inside a world-model agent's code world)
# --------------------------------------------------------------------------------------------------------------------
def _chunks(seeds, size=25):
    return [seeds[i:i + size] for i in range(0, len(seeds), size)]


def _w_doom(args):
    v, n, seed0 = args
    from experiments.doom.code_wm.screen import run_in_world
    return v, run_in_world(f"pilots.aspire_doom.policy_{v}:make_policy", {}, n, seed0)


def _w_drone(args):
    v, mission, n, seed0 = args
    from experiments.drone.code_wm.screen import run_in_world, summarise
    t0 = time.time()
    s = summarise(run_in_world(f"pilots.aspire_drone.policy_{v}:make_policy", {}, mission, n, seed0))
    s["wall_s"] = time.time() - t0
    return mission, v, s


def _w_pusht(args):
    from experiments.pusht.code_wm.screen import _run_chunk
    return args[0], args[1], _run_chunk(args[2])


def _w_rope(args):
    v, init, dev = args
    import torch
    torch.set_num_threads(1)
    from experiments.annex.code_wm.screen import run_in_world
    t0 = time.time()
    r = run_in_world(f"pilots.aspire_annex.policy_{v}:make_policy", {}, init, device=dev)
    return v, [dict(success=bool(e["success"]), tip_dist=float(e["tip_dist"]), n_steps=int(e["n_steps"])) for e in r], time.time() - t0


def _pool(workers):
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(workers)


def _pick(names, wanted):
    if not wanted:
        return names
    bad = [w for w in wanted if w not in names]
    if bad:
        sys.exit(f"unknown program(s) {bad}; available: {names}")
    return [n for n in names if n in wanted]


def _table(rows, metric_name):
    print(f"\n  {'program':28s} {'reproduced':>11s} {'recorded':>10s}   n")
    for name, rep, recd, n in rows:
        f = (lambda x: "-" if x is None else (f"{x:10.1f}" if metric_name == "ret" else f"{100 * x:9.1f}%"))
        print(f"  {name:28s} {f(rep):>11s} {f(recd):>10s}   {n}")


def world_doom(a, programs, w):
    rec = json.loads((ROOT / "runs/probe/screen_doom.json").read_text())
    n = a.episodes or 200
    vs = _pick([f"v{k}" for k in range(1, 6)], programs)
    # same world seeds for every version: chunk c of 25 episodes uses seed 20000 + 1000 c (experiments/doom/code_wm/screen.py)
    tasks = [(v, min(25, n - c * 25), 20000 + 1000 * c) for v in vs for c in range((n + 24) // 25)]
    t0 = time.time()
    with _pool(w) as pool:
        outs = pool.map(_w_doom, tasks, chunksize=1)
    rows, res = [], {}
    for v in vs:
        rets = [e["ret"] for vv, r in outs if vv == v for e in r]
        res[v] = dict(n=len(rets), mean=statistics.fmean(rets), median=statistics.median(rets))
        rows.append((f"aspire_doom {v} (mean ret)", res[v]["mean"], rec[v]["mean"] if n == rec[v]["n"] else None, len(rets)))
    _table(rows, "ret")
    return res, time.time() - t0


def world_drone(a, programs, w, missions):
    rec = json.loads((ROOT / "runs/probe/screen_drone.json").read_text())
    jobs = []
    if "A" in missions:
        jobs += [(f"v{i}", "A", a.episodes or 100, 5000) for i in range(1, 6)]
    if "C" in missions:
        jobs += [(f"v{i}", "C", a.episodes or 200, 6000) for i in range(6, 15)]
    if programs:      # names like v3 or A/v3
        keep = set(_pick([f"{m}/{v}" for v, m, _, _ in jobs] + [v for v, _, _, _ in jobs], programs))
        jobs = [j for j in jobs if f"{j[1]}/{j[0]}" in keep or j[0] in keep]
    t0 = time.time()
    with _pool(w) as pool:
        outs = pool.map(_w_drone, jobs, chunksize=1)
    rows, res = [], {}
    for m, v, s in outs:
        res[f"{m}/{v}"] = s
        r = rec[m][v]
        rows.append((f"mission {m} aspire_drone {v}", s["success"], r["success"] if s["n"] == r["n"] else None, s["n"]))
    _table(rows, "success")
    return res, time.time() - t0


def world_pusht(a, programs, w, objs):
    rec = json.loads((ROOT / "runs/probe/screen_pusht.json").read_text())
    n = a.episodes or 200
    jobs = ([("T", v) for v in range(1, 9)] if "T" in objs else []) + ([("L", v) for v in range(9, 16)] if "L" in objs else [])
    if programs:      # names like v3 or T/v3
        keep = set(_pick([f"{o}/v{v}" for o, v in jobs] + [f"v{v}" for o, v in jobs], programs))
        jobs = [(o, v) for o, v in jobs if f"{o}/v{v}" in keep or f"v{v}" in keep]
    # episode i uses seed 50000 + i for the initial state and the policy (same states for every version of an object)
    chunks = [(o, v, (f"pilots.aspire_pusht.policy_v{v}:make_policy", {}, o, s))
              for o, v in jobs for s in _chunks(list(range(50000, 50000 + n)))]
    t0 = time.time()
    with _pool(w) as pool:
        outs = pool.map(_w_pusht, chunks, chunksize=1)
    rows, res = [], {}
    for o, v in jobs:
        eps = [e for oo, vv, r in outs if (oo, vv) == (o, v) for e in r]
        s = statistics.fmean(float(e["success"]) for e in eps)
        res[f"{o}/v{v}"] = dict(n=len(eps), success=s)
        r = rec[o][f"v{v}"]
        rows.append((f"{o} aspire_pusht v{v}", s, r["success"] if r["n"] == len(eps) else None, len(eps)))
    _table(rows, "success")
    return res, time.time() - t0


def _cable_programs():
    from experiments.audit.station_probe import GRID
    progs = [(lb, fac, kw) for lb, fac, kw in GRID]
    progs += [(f"blind_v{k}", f"pilots.aspire_blind.policy_v{k}:make_policy", {}) for k in (1, 2, 3)]
    return progs


def world_cable(a, programs, w, dev):
    from experiments.audit.snapshots import init_obs_station
    from experiments.station.code_wm.screen import run_in_world
    first = _parse_seeds(a.seeds)[0] if a.seeds else 6000
    n = a.episodes or 50
    rec = json.loads((ROOT / "runs/probe/station_probe.json").read_text())
    alias = {"aspire_style": "cap_v3", "code_wm": "code_wm_ctrl", "aspire_blind": "blind_v2"}
    progs = _cable_programs()
    want = [alias.get(p, p) for p in programs] if programs else None
    names = _pick([p[0] for p in progs], want)
    init = init_obs_station(first, n, w)
    t0 = time.time()
    rows, res = [], {}
    for lb, fac, kw in progs:
        if lb not in names:
            continue
        t1 = time.time()
        eps = run_in_world(fac, kw, init, device=dev)
        s = statistics.fmean(float(bool(e["success"])) for e in eps)
        res[lb] = dict(n=len(eps), success=s, per_seed=[int(bool(e["success"])) for e in eps], wall_s=time.time() - t1)
        r = rec["world"].get(f"{lb}@{first}")
        rr = rec["real"].get(f"{lb}@{first}")
        rows.append((lb, s, r["success"] if r and r["n"] == len(eps) else None, len(eps)))
        pct = (lambda d: "  -   " if not d else "%5.1f%%" % (100 * d["success"]))
        print(f"  {lb:28s} world {100 * s:5.1f}%   recorded world {pct(r)}   real (recorded) {pct(rr)}"
              f"   {time.time() - t1:.0f} s", flush=True)
    _table(rows, "success")
    return res, time.time() - t0


def world_rope(a, programs, w, dev):
    from experiments.audit.snapshots import init_obs_annex
    first = _parse_seeds(a.seeds)[0] if a.seeds else 1000
    n = a.episodes or 50
    alias = {"aspire_style": "v6", "aspire_style_extended": "v19"}
    vs = _pick([f"v{k}" for k in range(1, 20)], [alias.get(p, p) for p in programs] if programs else None)
    init = init_obs_annex(first, n, w)
    rec_w = ROOT / f"runs/annex/audit/screen/world_{first}.json"
    rec_r = ROOT / f"runs/annex/audit/screen/real_{first}.json"
    RW = json.loads(rec_w.read_text()) if rec_w.exists() else {}
    RR = json.loads(rec_r.read_text()) if rec_r.exists() else {}
    t0 = time.time()
    res = {}
    with _pool(min(w, len(vs))) as pool:
        for v, r, wall in pool.imap_unordered(_w_rope, [(v, init, dev) for v in vs]):
            res[v] = dict(n=len(r), success=statistics.fmean(float(e["success"]) for e in r), episodes=r, wall_s=wall)
            print(f"  aspire_annex {v:4s} world {sum(e['success'] for e in r)}/{len(r)}   {wall:.0f} s", flush=True)
    rows = []
    for v in vs:
        rw = RW.get(v)
        same = rw is not None and len(rw) >= n
        recd = statistics.fmean(float(e["success"]) for e in rw[:n]) if same else None
        if same:
            res[v]["identical_episodes"] = sum(x["success"] == y["success"] for x, y in zip(res[v]["episodes"], rw[:n]))
        rows.append((f"aspire_annex {v}", res[v]["success"], recd, n))
    _table(rows, "success")
    if RR:
        print("  (real outcomes of the same seeds, recorded: " +
              ", ".join(f"{v} {sum(e['success'] for e in RR[v][:n])}/{n}" for v in vs if v in RR) + ")")
    return res, time.time() - t0


def cmd_world(a):
    task = _norm_task(a.task)
    w = _workers(a.workers)
    progs = a.programs
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    print(f"world {a.task}: {w} workers", flush=True)
    if task == "doom":
        res, wall = world_doom(a, progs, w)
    elif task in ("drone", "drone_A", "drone_C"):
        res, wall = world_drone(a, progs, w, "AC" if task == "drone" else task[-1])
    elif task in ("pusht_T", "pusht_L") or a.task == "pusht":
        res, wall = world_pusht(a, progs, w, "TL" if a.task == "pusht" else task[-1])
    elif task == "cable":
        res, wall = world_cable(a, progs, w, dev)
    elif task == "rope":
        # the recorded rope screening ran on CPU; pass --device cuda to use a GPU
        res, wall = world_rope(a, progs, w, a.device or "cpu")
    else:
        sys.exit(f"no world harness for {task!r} (doom, drone, pusht, cable, rope)")
    print(f"\n  wall {wall:.1f} s")
    p = _save("world", task, dict(task=a.task, programs=progs, episodes=a.episodes, wall_s=wall, results=res))
    print(f"  -> {p.relative_to(ROOT)}")


# --------------------------------------------------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------------------------------------------------
WORLD_HARNESSES = [
    ("doom", "experiments/doom/code_wm/screen.py", "aspire_doom v1-v5, 200 imagined episodes each", "runs/probe/screen_doom.json"),
    ("drone", "experiments/drone/code_wm/screen.py", "aspire_drone v1-v5 (mission A, 100) and v6-v14 (mission C, 200)", "runs/probe/screen_drone.json"),
    ("pusht", "experiments/pusht/code_wm/screen.py", "aspire_pusht v1-v8 (T) and v9-v15 (L), 200 each", "runs/probe/screen_pusht.json"),
    ("cable", "experiments/station/code_wm/screen.py", "pilot programs + 16 open-loop routers, reset readings of seeds 6000-6049", "runs/probe/station_probe.json"),
    ("rope", "experiments/annex/code_wm/screen.py", "aspire_annex v1-v19, reset readings of seeds 1000-1049", "runs/annex/audit/screen/world_1000.json"),
]


def cmd_list(_):
    print(f"{'task':9s} {'controller':24s} {'recorded':>10s} {'n':>4s}  seeds       factory")
    for e in REGISTRY:
        rec, metric = _record(e), _metric(e)
        r = _fmt(metric, rec["eval"]["mean"]) if rec else "-"
        n = rec["eval"]["n"] if rec else 0
        sd = f"{rec['seeds'][0]}-{rec['seeds'][-1]}" if rec else ""
        print(f"{e['task']:9s} {e['controller']:24s} {r:>10s} {n:4d}  {sd:11s} {e['factory']}")
    print("\nworld harnesses (python reproduce.py world <task> [program ...]):")
    for t, f, what, out in WORLD_HARNESSES:
        print(f"  {t:6s} {f:42s} {what}  [recorded: {out}]")


def main():
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("real")
    r.add_argument("task"); r.add_argument("controller")
    r.add_argument("--seeds", help="a-b (inclusive) or a,b,c; default: the seeds of the recorded evaluation")
    r.add_argument("--episodes", type=int, help="only the first n of those seeds")
    r.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    r.add_argument("--log-dir", help="write the lab's step logs here")
    wd = sub.add_parser("world")
    wd.add_argument("task"); wd.add_argument("programs", nargs="*")
    wd.add_argument("--episodes", type=int, help="episodes per program (default: as recorded)")
    wd.add_argument("--seeds", help="cable/rope: first seed of the reset readings (default 6000 / 1000)")
    wd.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    wd.add_argument("--device", help="torch device for the cable/rope worlds (default: cuda if available for cable, cpu for rope)")
    a = ap.parse_args()
    {"list": cmd_list, "real": cmd_real, "world": cmd_world}[a.cmd](a)


if __name__ == "__main__":
    main()
