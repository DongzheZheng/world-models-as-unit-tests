"""The replay "agent": follows a recorded script in the workspace instead of thinking.  It spends no tokens.

    python -P _replay_agent.py <script.json>          (working directory: the workspace; started by harness/agents.py)

It does what an agent would do through the same door: real trials go through lab.py, so the environment server
counts and records them, and files are written into the workspace.  The script format is described in
harness/agents.py.  Source paths in the script are relative to the script file.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    script_path = Path(sys.argv[1]).resolve()
    script = json.loads(script_path.read_text())
    base = script_path.parent
    workspace = Path(os.getcwd())
    sys.path.insert(0, str(workspace))
    import lab
    import numpy as np

    for number, step in enumerate(script["steps"], 1):
        if "copy" in step:
            for dest, source in step["copy"].items():
                target = workspace / dest
                if ".." in Path(dest).parts or Path(dest).is_absolute():
                    raise ValueError(f"step {number}: {dest!r} is not a path inside the workspace")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base / source, target)
                print(f"step {number}: copied {dest}")
        elif "trial" in step:
            trial = step["trial"]
            if "record" in trial:
                with np.load(base / trial["record"], allow_pickle=False) as data:
                    seed, actions = int(data["seed"]), data["actions"]
            else:
                seed, actions = int(trial["seed"]), []
                for item in trial["actions"]:           # an action, or {"repeat": n, "action": [...]}
                    actions += [item["action"]] * int(item["repeat"]) if isinstance(item, dict) else [item]
            lab.reset(seed)
            done, info, n = False, {}, 0
            for action in actions:
                _, done, info = lab.step(action)
                n += 1
                if done:
                    break
            print(f"step {number}: real trial with seed {seed}: {n} steps, "
                  + (f"success={info.get('success')}" if done else "abandoned before its end"))
        else:
            raise ValueError(f"step {number}: expected \"copy\" or \"trial\", got {sorted(step)}")
    print("budget:", json.dumps({k: v for k, v in lab.status().items() if k.startswith("trial")}))
    lab.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
