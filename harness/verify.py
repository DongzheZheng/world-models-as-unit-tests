"""`verify`: score what the agent left in its workspace.  Run by the harness, never by the agent.

1. Which deliverables are there (harness/contract.py), and can policy.py and world.py be loaded at all.
2. Real: policy.py on the held-out seeds in the environment.
3. Predicted: policy.py in the delivered world.py, on `[verify] model_episodes` seeds of the model; and, if World has
   reset_from(), once more started from the reset observations of the held-out episodes (same situations as 2).
4. report.json and report.md in the run directory; per-episode results and logs in <run dir>/verify/.

Delivered files are untrusted code: everything in 1 to 3 that touches them happens in subprocesses with time limits
(harness/evaluate.py).  A file that cannot be imported, an object with the wrong methods, a wrong return shape, an
exception or a timeout is written into the report; none of them stops the harness.  An episode in which the program
raises counts as a failure and is flagged.

report.json:
    run            name, role, environment, config file
    trials         used, budget, hypothesis_at_first_trial (was notes/hypothesis.json there when trial 1 started)
    agent          outcome of the agent command (state/agent.json), if one was run
    deliverables   file -> {status: present | missing | broken, required, detail}
    predicted_by_agent        the number in prediction.json, or null
    model          policy.py in world.py: episodes, successes, success_rate, flagged, flags, seeds, error,
                   format_problems (observation keys or shapes that differ from the environment's)
    model_from_held_out_starts   the same from the held-out reset observations, plus agreement (fraction of held-out
                   seeds on which model and environment gave the same outcome); null without reset_from()
    real           policy.py in the environment on the held-out seeds: the same keys as model
    files          where the per-episode results are
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import time
from pathlib import Path

from harness import __version__, budget, config, contract, evaluate, report, scaffold, wire


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)) or "policy"


def _span(seeds: list[int]) -> list[int] | None:
    return [seeds[0], seeds[-1]] if seeds else None


def score(*, config_json, cfg: dict, env_info: dict, policies: list[dict], world, model_starts: list[dict],
          real_seeds: list[int], workers: int, log_dir, paired: bool) -> list[dict]:
    """Episodes of each policy file in the environment and in a world-model file.

    policies: [{"file", "cwd", "name"}].  `world` may be None (no model episodes) and `real_seeds` empty (no real
    episodes).  Returns one dict per policy: {"real", "model", "paired", "episodes": {...}}; a part that was not run
    is None."""
    limits = evaluate.limits_of(cfg)
    out = [{"real": None, "model": None, "paired": None, "episodes": {}, "has_reset_from": None} for _ in policies]
    signature, reset_obs = None, {}
    if real_seeds:
        results = evaluate.real_episodes(config_json, cfg, policies, real_seeds, workers=workers, log_dir=log_dir,
                                         keep_reset_obs=True)
        for o, r in zip(out, results):
            for e in r["episodes"]:
                obs = e.pop("reset_obs", None)
                if obs is not None:
                    reset_obs.setdefault(e["seed"], obs)
            o["real"] = {**evaluate.summarise(r["episodes"]), "seeds": _span(real_seeds),
                         "error": r["error"] or r["load_error"]}
            o["episodes"]["real"] = r["episodes"]
        if reset_obs:
            signature = contract.obs_signature(wire.unpack(next(iter(reset_obs.values()))))
    if world is not None and model_starts:
        results = evaluate.model_episodes(world, policies, model_starts, task_card=env_info["task_card"],
                                          max_steps=env_info["max_steps"], limits=limits, workers=workers,
                                          log_dir=log_dir, tag="model")
        for o, r in zip(out, results):
            o["model"] = {**evaluate.summarise(r["episodes"]), "seeds": _span([s["seed"] for s in model_starts]),
                          "error": r["load_error"], "error_in": r["load_what"],
                          "format_problems": contract.compare_signatures(signature, r["obs_signature"])}
            o["episodes"]["model"] = r["episodes"]
            o["has_reset_from"] = r["has_reset_from"]
    starts = [{"seed": s, "obs": reset_obs[s]} for s in real_seeds if s in reset_obs]
    which = [i for i, o in enumerate(out) if o["has_reset_from"] and o["model"] and not o["model"]["error"]]
    if paired and world is not None and starts and which:
        results = evaluate.model_episodes(world, [policies[i] for i in which], starts, task_card=env_info["task_card"],
                                          max_steps=env_info["max_steps"], limits=limits, workers=workers,
                                          log_dir=log_dir, tag="model_paired")
        for i, r in zip(which, results):
            o = out[i]
            real_by_seed = {e["seed"]: e["success"] for e in o["episodes"]["real"]}
            same = [e["success"] == real_by_seed[e["seed"]] for e in r["episodes"] if e["seed"] in real_by_seed]
            o["paired"] = {**evaluate.summarise(r["episodes"]), "seeds": _span([s["seed"] for s in starts]),
                           "error": r["load_error"], "agreement": sum(same) / len(same) if same else None}
            o["episodes"]["paired"] = r["episodes"]
    return out


def deliverables(cfg: dict, env_info: dict, log_dir: Path):
    """Presence of every deliverable, and whether policy.py and world.py can be loaded.
    Returns (table: file -> {status, required, detail}, predicted_by_agent, world_has_reset_from)."""
    ws = config.paths(cfg).workspace
    role = contract.ROLES[cfg["agent"]["role"]]
    load_timeout = evaluate.limits_of(cfg)["load_timeout_s"]
    jobs = {}
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for kind, name in (("policy", contract.POLICY_FILE), ("world", contract.WORLD_FILE)):
            if (ws / name).is_file():
                jobs[kind] = pool.submit(evaluate.check_file, kind, ws / name, ws, env_info["task_card"], load_timeout,
                                         log_dir / f"check_{kind}.log")
        checks = {kind: f.result() for kind, f in jobs.items()}

    def loaded(name, kind, required):
        if not (ws / name).is_file():
            return {"status": "missing", "required": required, "detail": None}
        if not checks[kind]["ok"]:
            return {"status": "broken", "required": required, "detail": checks[kind]["error"]}
        return {"status": "present", "required": required, "detail": None}

    table = {contract.POLICY_FILE: loaded(contract.POLICY_FILE, "policy", True),
             contract.WORLD_FILE: loaded(contract.WORLD_FILE, "world", role["world"])}
    predicted, problem = contract.read_prediction(ws / contract.PREDICTION_FILE)
    table[contract.PREDICTION_FILE] = {
        "status": "present" if problem is None else "missing" if problem == "missing" else "broken",
        "required": role["prediction"], "detail": None if problem in (None, "missing") else problem}
    for name in contract.NOTE_FILES:
        f = ws / name
        entry = {"status": "present" if f.is_file() else "missing", "required": False, "detail": None}
        if f.is_file():
            try:
                json.loads(f.read_text())
            except (OSError, ValueError):
                entry.update(status="broken", detail="not readable as JSON")
        table[name] = entry
    has_reset_from = bool(checks.get("world", {}).get("has_reset_from"))
    return table, predicted, has_reset_from


def verify(cfg: dict, *, workers: int | None = None, model_episodes: int | None = None, skip_real: bool = False,
           skip_model: bool = False) -> dict:
    """Score the workspace of an initialised run directory; writes and returns the report."""
    p = config.paths(cfg)
    env_info = scaffold.environment_info(cfg)
    out_dir, log_dir = p.verify, p.verify / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = int(workers or cfg["verify"]["workers"])
    t0 = time.time()

    table, predicted, _ = deliverables(cfg, env_info, log_dir)
    policy_ok = table[contract.POLICY_FILE]["status"] == "present"
    world_ok = table[contract.WORLD_FILE]["status"] == "present"
    ws = p.workspace
    policies = [{"file": ws / contract.POLICY_FILE, "cwd": ws, "name": "policy"}]
    real_seeds = config.held_out_seeds(cfg) if policy_ok and not skip_real else []
    starts = [{"seed": s} for s in config.model_seeds(cfg, model_episodes)]
    use_world = policy_ok and world_ok and not skip_model
    result = score(config_json=p.config_json, cfg=cfg, env_info=env_info, policies=policies,
                   world=ws / contract.WORLD_FILE if use_world else None, model_starts=starts,
                   real_seeds=real_seeds, workers=workers, log_dir=log_dir, paired=bool(cfg["verify"]["paired"]))[0]

    files = {}
    for part, episodes in result["episodes"].items():
        name = {"real": "real_episodes.json", "model": "model_episodes.json", "paired": "model_paired_episodes.json"}[part]
        (out_dir / name).write_text(json.dumps(episodes, indent=1))
        files[part] = f"verify/{name}"
    counter = budget.read(p.budget) or {"trials": [], "trial_budget": cfg["budget"]["trials"]}
    agent = json.loads(p.agent.read_text()) if p.agent.is_file() else None
    rep = {
        "harness_version": __version__,
        "run": {"name": cfg["run"]["name"], "role": cfg["agent"]["role"], "environment": cfg["environment"]["name"],
                "config": cfg["config_file"]},
        "trials": {"used": len(counter["trials"]), "budget": counter["trial_budget"],
                   "hypothesis_at_first_trial": counter.get("hypothesis_at_first_trial")},
        "agent": None if agent is None else {k: agent.get(k) for k in ("kind", "exit_code", "timed_out", "wall_s", "error")},
        "deliverables": table,
        "predicted_by_agent": predicted,
        "model": result["model"],
        "model_from_held_out_starts": result["paired"],
        "real": result["real"],
        "files": {"report": "report.md", **files, "logs": "verify/logs"},
        "wall_s": round(time.time() - t0, 2),
    }
    p.report_json.write_text(json.dumps(rep, indent=1))
    p.report_md.write_text(markdown(rep))
    return rep


def markdown(rep: dict) -> str:
    """The report as plain tables."""
    real, model, paired = rep["real"], rep["model"], rep["model_from_held_out_starts"]

    def seeds(s):
        return f"{s['seeds'][0]}-{s['seeds'][1]}" if s and s.get("seeds") else "-"

    predicted = "-" if rep["predicted_by_agent"] is None else f"{rep['predicted_by_agent']:.3f}"
    rows = [
        ["real trials used", f"{rep['trials']['used']} of {rep['trials']['budget']}", "-", "-", "-"],
        ["predicted by the agent (prediction.json)", predicted, "-", "-", "-"],
        ["predicted: policy.py in world.py", report.rate(model), report.count(model), seeds(model), report.flagged(model)],
        ["predicted: policy.py in world.py from held-out reset observations", report.rate(paired), report.count(paired),
         seeds(paired), report.flagged(paired)],
        ["real: policy.py in the environment, held-out", report.rate(real), report.count(real), seeds(real),
         report.flagged(real)],
    ]
    parts = [f"# {rep['run']['name']}: {rep['run']['role']}", "",
             report.table(["quantity", "value", "successes/episodes", "seeds", "flagged episodes"], rows), ""]
    extra = []
    if paired and paired.get("agreement") is not None:
        extra.append(["same outcome in world.py and in the environment, per held-out seed", f"{paired['agreement']:.3f}"])
    hyp = rep["trials"]["hypothesis_at_first_trial"]
    if hyp is not None:
        extra.append(["notes/hypothesis.json present when the first real trial started", "yes" if hyp else "no"])
    if rep["agent"]:
        a = rep["agent"]
        extra.append(["agent command", f"exit code {a['exit_code']}, {a['wall_s']} s"
                      + (", stopped at the time limit" if a["timed_out"] else "") + (f", {a['error']}" if a["error"] else "")])
    for name, part in (("world.py", model), ("environment", real)):
        if part and part.get("error"):
            extra.append([f"error ({name})", part["error"]])
        if part and part.get("first_error"):
            extra.append([f"first flagged episode ({name})", part["first_error"]])
    for problem in (model or {}).get("format_problems") or []:
        extra.append(["world.py format", problem])
    if extra:
        parts += [report.table(["check", "result"], extra), ""]
    rows = [[name, d["status"], "yes" if d["required"] else "no", d["detail"] or ""]
            for name, d in rep["deliverables"].items()]
    parts += [report.table(["deliverable", "status", "required", "detail"], rows), ""]
    rows = [[k, v] for k, v in rep["files"].items()]
    parts += [report.table(["output", "file"], rows), ""]
    return "\n".join(parts)
