"""Would the code world have flagged the stall bug in the no-logs code-as-policy pilot's submitted program?
Orchestrator only.  The reset readings of the official evaluation episodes (seeds 1000-1049) are taken from the
first row of that pilot's own evaluation step logs (no extra real steps); each program version is then run in the
code world from those readings and compared, episode by episode, with what really happened.
usage: python -m experiments.audit.station_blind_check -> runs/probe/station_blind_check.json

The evaluation step logs are not part of the public repository.  When they are absent, the same reset readings are
taken again with the lab's snapshot tool (`python -m reallab.station._snapshots`, zero agent steps; cached in
runs/station/audit/), which gives the first row of those logs.
"""
import json
from pathlib import Path

import numpy as np

LOGS = Path("runs/station/aspire_blind/eval_logs")


def init_obs():
    out, seeds = [], list(range(1000, 1050))
    if not all((LOGS / f"ep_{s}.npz").exists() for s in seeds):
        from experiments.audit.snapshots import init_obs_station
        return seeds, init_obs_station(1000, 50)
    for s in seeds:
        d = np.load(LOGS / f"ep_{s}.npz", allow_pickle=False)
        out.append(dict(points=d["points"][0], tool_pos=d["tool_pos"][0], tool_vel=np.zeros(2), posts=d["posts"][0],
                        post_sides=d["post_sides"][0], sides_now=d["sides_now"][0], target=d["target"][0],
                        load=float(d["load"][0]), t=0))
    return seeds, out


if __name__ == "__main__":
    from experiments.station.code_wm.screen import run_in_world
    seeds, obs = init_obs()
    raw = json.loads(Path("runs/station/aspire_blind/eval_aspire_blind.json").read_text())
    real = {e["seed"]: e for e in (raw if isinstance(raw, list) else raw.get("episodes", []))}
    out = {}
    for k in (1, 2, 3):
        res = run_in_world(f"pilots.aspire_blind.policy_v{k}:make_policy", {}, obs)
        fails = [s for s, e in zip(seeds, res) if not e["success"]]
        out[f"v{k}"] = dict(world_success=sum(e["success"] for e in res) / len(res), world_failed_seeds=fails,
                            world_fail_kinds={s: ("overload" if e["overload"] else f"not done in {e['n_steps']} steps, sides {e['sides_ok']}/3, dist {e['final_dist']:.3f}")
                                              for s, e in zip(seeds, res) if not e["success"]})
        print(f"v{k}: code world predicts {out[f'v{k}']['world_success']:.2f}; failed seeds {fails}")
    if real:
        rf = sorted(s for s, e in real.items() if not e["success"])
        out["real_v2_failed_seeds"] = rf
        print("real (official, v2) failed seeds:", rf, {s: (real[s]["n_steps"], real[s]["overload"], real[s]["sides_ok"]) for s in rf})
    Path("runs/probe/station_blind_check.json").write_text(json.dumps(out, indent=1))
