"""Real-world interaction ledger.

The formal simulators (VizDoom, PyBullet) play the role of the *real world*.  Every single
interaction with them is metered here, tagged with a purpose, and appended to one JSONL file.
This is the accounting backbone of the anti-inverse-crime protocol:

  calibration   the "lives": real episodes whose black-box logs the world-model author may study
                to write / fit the code world.  Hard-capped (episodes and/or steps) per scenario.
  eval          final evaluation episodes (never used to change a controller)
  oracle_train  RL trained directly in the real world -- a privileged reference, reported with its cost
  smoke         the lab's own plumbing tests (building the lab, not the world model)
  observation   real episodes run by an existing controller (not by a pilot) whose logs every pilot may study:
                watching the environment is cheap and safe; only a pilot's own attempts are lives

Controllers trained in the code world must show 0 real steps beyond `calibration`.
"""
from __future__ import annotations

import atexit
import fcntl
import json
import os
import time
from collections import defaultdict
from pathlib import Path

PURPOSES = ("calibration", "eval", "oracle_train", "smoke", "observation")


def default_path() -> Path:
    """The ledger file: $CAWM_LEDGER if set (tests point it at a temporary file), else runs/ledger.jsonl.
    Resolved at call time, so setting the variable after this module was imported still takes effect."""
    return Path(os.environ.get("CAWM_LEDGER", Path(__file__).resolve().parents[1] / "runs" / "ledger.jsonl"))


DEFAULT_PATH = default_path()   # import-time value, kept for callers that pass it explicitly


class BudgetExceeded(RuntimeError):
    pass


def _parse(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:                  # shared lock: never read a record another process is still writing
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return _parse(f.read())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _sum(records: list[dict]) -> dict:
    out: dict = defaultdict(lambda: {"steps": 0, "episodes": 0})
    for r in records:
        key = (r["demo"], r["scenario"], r["purpose"])
        out[key]["steps"] += r["steps"]
        out[key]["episodes"] += r["episodes"]
    return dict(out)


def totals(path: Path | None = None) -> dict:
    """{(demo, scenario, purpose): {"steps": int, "episodes": int}} over the whole ledger."""
    return _sum(_read(Path(path) if path is not None else default_path()))


class Meter:
    """Counts real steps/episodes for one environment instance and writes them to the ledger.

    Usage inside a real-world wrapper:
        self.meter = Meter("drone", "site", purpose, tag, cap_episodes=3)
        ... self.meter.start_episode()   # in reset(): enforces the lives cap, records the life at once
        ... self.meter.step()            # once per agent step (not per simulator tic)
        ... self.meter.episode()         # when the episode ends or is abandoned (flushes the steps)
    Wrappers that never call start_episode() still work: episode() then counts the episode itself.
    Caps are checked against everything already in the ledger for (demo, scenario, purpose),
    across processes.
    """

    def __init__(self, demo: str, scenario: str, purpose: str, tag: str = "",
                 cap_steps: int | None = None, path: Path | None = None, cap_episodes: int | None = None):
        if purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
        self.demo, self.scenario, self.purpose, self.tag = demo, scenario, purpose, tag
        self.cap_steps, self.cap_episodes = cap_steps, cap_episodes
        self.path = Path(path) if path is not None else default_path()
        self._steps = 0
        self._episodes = 0
        self._open = False      # an episode registered by start_episode() is in progress
        self._spent = None      # lazily read: steps already in the ledger for this key
        atexit.register(self.flush)

    @property
    def _key(self):
        return (self.demo, self.scenario, self.purpose)

    def _append(self, steps: int, episodes: int, check=None) -> None:
        """Append one record under an exclusive lock; `check(totals_for_key)` may veto by raising."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                if check is not None:
                    f.seek(0)
                    check(_sum(_parse(f.read())).get(self._key, {"steps": 0, "episodes": 0}))
                rec = dict(t=time.time(), demo=self.demo, scenario=self.scenario, purpose=self.purpose,
                           tag=self.tag, steps=steps, episodes=episodes, pid=os.getpid())
                f.seek(0, 2)
                f.write(json.dumps(rec) + "\n")
                f.flush()                  # the record must be on disk before the lock is released
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def start_episode(self) -> None:
        """Register a new real episode (a "life").  Raises BudgetExceeded when the lives cap is used up."""
        if self._open:
            self.episode()

        def check(spent):
            if self.cap_episodes is not None and spent["episodes"] >= self.cap_episodes:
                raise BudgetExceeded(f"{self.demo}/{self.scenario}: all {self.cap_episodes} "
                                     f"'{self.purpose}' episodes (lives) are used up")

        self._append(0, 1, check)
        self._open = True

    def _already_spent(self) -> int:
        if self._spent is None:
            self._spent = totals(self.path).get(self._key, {"steps": 0})["steps"]
        return self._spent

    def step(self, n: int = 1) -> None:
        if self.cap_steps is not None and self._already_spent() + self._steps + n > self.cap_steps:
            self.flush()
            raise BudgetExceeded(
                f"{self.demo}/{self.scenario}: '{self.purpose}' budget of {self.cap_steps} real steps is exhausted")
        self._steps += n

    def episode(self) -> None:
        if self._open:
            self._open = False       # already counted by start_episode()
        else:
            self._episodes += 1
        self.flush()

    def flush(self) -> None:
        if self._steps == 0 and self._episodes == 0:
            return
        self._append(self._steps, self._episodes)
        if self._spent is not None:
            self._spent += self._steps
        self._steps = 0
        self._episodes = 0


def summary(path: Path | None = None) -> str:
    rows = sorted(totals(path).items())
    lines = [f"{'demo':8s} {'scenario':22s} {'purpose':13s} {'steps':>10s} {'episodes':>9s}"]
    for (demo, scen, purpose), v in rows:
        lines.append(f"{demo:8s} {scen:22s} {purpose:13s} {v['steps']:>10d} {v['episodes']:>9d}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
