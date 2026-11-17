"""A stand-in for an agent command in the tests.  It calls no model and spends no tokens.

    python stub_agent.py <brief file> <workspace> <reference dir> <brief text>      (working directory: the workspace)

It does what the harness expects of an agent, in the smallest way: reads BRIEF.md, writes notes/hypothesis.json,
spends two real trials through lab.py, and writes the deliverables by copying the toy reference files.  It also
records how it was started (agent_args.json) so that the test can check the placeholders of the command.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    brief_file, workspace, reference, brief_text = sys.argv[1:5]
    here = Path.cwd()
    brief = (here / "BRIEF.md").read_text()
    sys.path.insert(0, str(here))
    import lab

    (here / "notes" / "hypothesis.json").write_text(json.dumps({"what_it_is": "a stub's guess"}))
    before = lab.status()
    for seed in (1, 2):
        lab.reset(seed)
        done = False
        while not done:
            _, done, info = lab.step([0.5, -0.5])
    for name in ("world.py", "policy.py", "prediction.json"):
        shutil.copyfile(Path(reference) / name, here / name)
    (here / "agent_args.json").write_text(json.dumps({
        "cwd": str(here), "brief_file": brief_file, "workspace": workspace,
        "brief_text_is_the_brief": brief_text == brief, "stdin_is_the_brief": sys.stdin.read() == brief,
        "budget_at_start": before["trial_budget"], "trials_left_at_end": lab.status()["trials_left"],
        "marker": os.environ.get("STUB_MARKER")}))
    print("stub agent done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
