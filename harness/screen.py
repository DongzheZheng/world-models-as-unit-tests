"""`screen`: score any list of policy files in a given world model and in the environment.

This is the generic form of the comparison "success predicted by the world model against real success": every
program gets the same model episodes (seeds first, first + 1, ...) and the same real episodes (by default the
held-out seeds of the config).  It needs a config with an [environment] section but no run directory.

Outputs in the output directory:
    screen.json            per program: model, real (and paired) summaries; over all programs: Pearson r and the
                           mean absolute difference between predicted and real success rate
    screen.md              the same as a table
    screen.svg             scatter plot, predicted against real
    screen_episodes.json   every episode
    logs/                  output of the delivered code and of the environment processes

The policy files and the world model are untrusted code and run in subprocesses with time limits, each policy with
its own directory as working directory (harness/evaluate.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from harness import __version__, config, report, scaffold
from harness.verify import safe_name, score


class ScreenError(Exception):
    pass


def screen(cfg: dict, *, world=None, policies=None, names=None, out_dir=None, model_episodes: int | None = None,
           model_seed_first: int | None = None, real_seeds: list[int] | None = None, workers: int | None = None,
           paired: bool | None = None, skip_real: bool = False, skip_model: bool = False) -> dict:
    """Score policy files; arguments left at None come from the [screen], [verify] and [budget] sections."""
    world = Path(world).resolve() if world else (Path(cfg["screen"]["world"]) if cfg["screen"]["world"] else None)
    files = [Path(f).resolve() for f in policies] if policies else [Path(f) for f in cfg["screen"]["policies"]]
    if names is None and not policies:
        names = cfg["screen"]["names"] or None
    if not files:
        raise ScreenError("no policy files: give them on the command line or as [screen] policies")
    for f in files + ([world] if world else []):
        if not f.is_file():
            raise ScreenError(f"no file {f}")
    if names and len(names) != len(files):
        raise ScreenError("give one name per policy file")
    labels = []
    for i, f in enumerate(files):
        label = safe_name(names[i] if names and names[i] else f.stem)
        while label in labels:
            label += "_"
        labels.append(label)
    if skip_model or world is None:
        world = None
    out_dir = Path(out_dir).resolve() if out_dir else Path(cfg["screen"]["out"])
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = int(workers or cfg["verify"]["workers"])
    paired = cfg["screen"]["paired"] if paired is None else paired
    if real_seeds is None:
        real_seeds = config.held_out_seeds(cfg)
    if skip_real:
        real_seeds = []
    starts = [{"seed": s} for s in config.model_seeds(cfg, model_episodes, model_seed_first)]

    config_json = out_dir / "config.json"
    config_json.write_text(json.dumps(cfg, indent=1))
    env_info = scaffold.describe_environment(config_json, out_dir / "environment.json", log_dir / "describe.log",
                                             float(cfg["server"]["start_timeout_s"]))
    t0 = time.time()
    entries = [{"file": f, "cwd": f.parent, "name": label} for f, label in zip(files, labels)]
    results = score(config_json=config_json, cfg=cfg, env_info=env_info, policies=entries, world=world,
                    model_starts=starts, real_seeds=real_seeds, workers=workers, log_dir=log_dir, paired=paired)

    rows, episodes = [], {}
    for entry, res in zip(entries, results):
        rows.append({"name": entry["name"], "file": str(entry["file"]), "model": res["model"], "real": res["real"],
                     "paired": res["paired"]})
        episodes[entry["name"]] = res["episodes"]
    both = [(r["name"], r["model"]["success_rate"], r["real"]["success_rate"]) for r in rows
            if r["model"] and r["real"] and r["model"]["success_rate"] is not None and r["real"]["success_rate"] is not None]
    xs, ys = [b[1] for b in both], [b[2] for b in both]
    out = {
        "harness_version": __version__,
        "environment": cfg["environment"]["name"],
        "config": cfg["config_file"],
        "world": str(world) if world else None,
        "model_seeds": [starts[0]["seed"], starts[-1]["seed"]] if world else None,
        "real_seeds": [real_seeds[0], real_seeds[-1]] if real_seeds else None,
        "policies": rows,
        "summary": {"programs": len(both), "pearson_r": report.pearson(xs, ys),
                    "mean_abs_difference": report.mean_abs_error(xs, ys)},
        "wall_s": round(time.time() - t0, 2),
        "files": {"table": "screen.md", "plot": "screen.svg" if both else None, "episodes": "screen_episodes.json",
                  "logs": "logs"},
    }
    (out_dir / "screen.json").write_text(json.dumps(out, indent=1))
    (out_dir / "screen_episodes.json").write_text(json.dumps(episodes, indent=1))
    (out_dir / "screen.md").write_text(markdown(out))
    if both:
        s = out["summary"]
        note = f"{len(both)} programs" + (f", r = {s['pearson_r']:.3f}" if s["pearson_r"] is not None else "") \
            + f", mean |difference| = {s['mean_abs_difference']:.3f}"
        (out_dir / "screen.svg").write_text(report.scatter_svg(
            both, "success rate predicted in the world model", "real success rate (environment)", note))
    out["out_dir"] = str(out_dir)
    return out


def markdown(out: dict) -> str:
    def seeds(s):
        return f"{s[0]}-{s[1]}" if s else "-"

    header = ["program", "predicted (world model)", "successes/episodes", "flagged", "real (environment)",
              "successes/episodes", "flagged"]
    with_paired = any(r["paired"] for r in out["policies"])
    if with_paired:
        header += ["predicted from real resets", "same outcome"]
    rows = []
    for r in out["policies"]:
        row = [r["name"], report.rate(r["model"]), report.count(r["model"]), report.flagged(r["model"]),
               report.rate(r["real"]), report.count(r["real"]), report.flagged(r["real"])]
        if with_paired:
            agreement = (r["paired"] or {}).get("agreement")
            row += [report.rate(r["paired"]), "-" if agreement is None else f"{agreement:.3f}"]
        rows.append(row)
    s = out["summary"]
    facts = [["world model", out["world"] or "-"], ["model seeds", seeds(out["model_seeds"])],
             ["real seeds", seeds(out["real_seeds"])], ["programs with both numbers", s["programs"]],
             ["Pearson r", "-" if s["pearson_r"] is None else f"{s['pearson_r']:.3f}"],
             ["mean absolute difference", "-" if s["mean_abs_difference"] is None else f"{s['mean_abs_difference']:.3f}"]]
    errors = []
    for r in out["policies"]:
        for part in ("model", "real", "paired"):
            if r[part] and r[part].get("error"):
                errors.append([r["name"], part, r[part]["error"]])
            elif r[part] and r[part].get("first_error"):
                errors.append([r["name"], f"{part}, first flagged episode", r[part]["first_error"]])
        for problem in (r["model"] or {}).get("format_problems") or []:
            errors.append([r["name"], "world model format", problem])
    parts = [f"# screen: {out['environment']}", "", report.table(header, rows), "",
             report.table(["quantity", "value"], facts), ""]
    if errors:
        parts += [report.table(["program", "where", "problem"], errors), ""]
    return "\n".join(parts)
