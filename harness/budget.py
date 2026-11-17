"""The budget counter: how many real trials the agent has used, kept in a file outside its workspace.

state/budget.json:

    {"trial_budget": 5,
     "trials": [{"trial": 1, "seed": 3, "started": <unix time>, "ended": <unix time or null>, "n_steps": 80,
                 "outcome": "finished" | "abandoned" | "running", "success": true | false | null,
                 "record": "trial_1.npz"}, ...],
     "hypothesis_at_first_trial": true | false | null}      # was notes/hypothesis.json there when trial 1 started

A trial counts from the moment its reset succeeded, whether or not the episode is finished.  Only the environment
server writes this file; it is rewritten atomically on every change, so the count survives a restart of the server.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, path, trial_budget: int):
        self.path = Path(path)
        self.trial_budget = int(trial_budget)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            self.data["trial_budget"] = self.trial_budget
            for t in self.data["trials"]:           # a server that died mid-episode leaves a trial "running"
                if t["outcome"] == "running":
                    t["outcome"] = "abandoned"
        else:
            self.data = {"trial_budget": self.trial_budget, "trials": [], "hypothesis_at_first_trial": None}
        self._write()

    @property
    def used(self) -> int:
        return len(self.data["trials"])

    @property
    def left(self) -> int:
        return max(0, self.trial_budget - self.used)

    def check(self) -> None:
        if self.used >= self.trial_budget:
            raise BudgetExceeded(f"all {self.trial_budget} real trials of this run are used up")

    def start(self, seed: int) -> int:
        """Count one trial and return its number (1, 2, ...)."""
        self.check()
        k = self.used + 1
        self.data["trials"].append({"trial": k, "seed": int(seed), "started": time.time(), "ended": None,
                                    "n_steps": 0, "outcome": "running", "success": None, "record": None})
        self._write()
        return k

    def end(self, k: int, n_steps: int, finished: bool, success, record: str | None) -> None:
        t = self.data["trials"][k - 1]
        t.update(ended=time.time(), n_steps=int(n_steps), outcome="finished" if finished else "abandoned",
                 success=None if success is None else bool(success), record=record)
        self._write()

    def note_hypothesis(self, present: bool) -> None:
        if self.data.get("hypothesis_at_first_trial") is None:
            self.data["hypothesis_at_first_trial"] = bool(present)
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)


def read(path) -> dict | None:
    """The counter as stored, or None if no server has run yet."""
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
