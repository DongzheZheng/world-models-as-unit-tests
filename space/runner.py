"""Run one Code-as-Policies pushing program for a batch of episodes and record trajectories.

    run_world(obj, version, seed0, n, record)   n episodes inside the world model (worlds/pusht/pusht_world.py)
    run_real(obj, version, seed0, n, record)    n episodes in the environment (reallab.pusht)

obj is "T" or "L", version a program number of that object (PROGRAMS), episode i uses seed seed0 + i.
record: False (outcomes only), True (also the trajectory of the first episode) or "all" (of every episode).
Both return a dict:
    side, task, version, seed0, n, time_limit (steps)
    success   [bool] * n          per-episode outcome (coverage > 0.95 within 300 steps)
    n_steps   [int] * n
    first     None, or the first episode: seed, success, n_steps and
              trajectory = [[pusher x, pusher y, block x, block y, block theta], ...], one row for the initial state
              and one per step (x, y rounded to 0.1 u, theta in [0, 2 pi) rounded to 0.001 rad)
    trajectories   only with record="all": one such trajectory per episode
    geometry  block = outline polygons in the block's body frame, goal = (x, y, theta), pusher_radius,
              workspace = (width, height), wall = (low, high) inner faces of the walls

The episodes are run by the repository's own evaluation code, which is only tapped to record trajectories:
  world        experiments.pusht.code_wm.screen._run_chunk runs the batch.  While it runs, the world module's
               make_state and physics are wrapped so that the pusher positions and block poses of the batch are
               copied after every step.
  environment  reallab.pusht.parallel._worker (the worker loop of evaluate()) is called in this process on plain
               queues with step logging switched on, and the step logs are read back.
               Every episode is metered by cawm.ledger (purpose "eval", so seeds must be >= 1000).
space/check_runner.py compares the outcomes with _run_chunk and with the recorded evaluations.

serve(conn) is the loop of one worker process of the web app (space/app.py).
"""
from __future__ import annotations

import contextlib
import math
import queue
import signal
import tempfile
from pathlib import Path

PROGRAMS = {"T": tuple(range(1, 9)), "L": tuple(range(9, 18))}     # pilots/aspire_pusht/policy_v{k}.py
TAGS = {("T", 5): "submitted", ("T", 8): "extended", ("L", 10): "submitted", ("L", 15): "extended",
        ("L", 17): "with_tests"}
DEFAULT = dict(task="T", version=5)                                # the selection a page starts with
LEDGER_TAG = "space_demo"

# Episodes per side of one run on the live page.  Measured on two cores of a fast workstation with warm workers,
# the slowest programs take 1.7 s (world model) + 1.3 s (environment) for 20 episodes; this keeps a full run well
# under 20 s on shared 2-vCPU hardware several times slower.
N_EPISODES = 20


def factory(obj: str, version: int) -> str:
    """The policy factory of program `version`; only the listed programs of the given object are accepted."""
    if obj not in PROGRAMS or isinstance(version, bool) or not isinstance(version, int) or version not in PROGRAMS[obj]:
        raise ValueError(f"unknown program {obj!r} / {version!r}")
    return f"pilots.aspire_pusht.policy_v{version}:make_policy"


def _row(px, py, bx, by, theta):
    return [round(float(px), 1), round(float(py), 1), round(float(bx), 1), round(float(by), 1),
            round(float(theta) % (2 * math.pi), 3)]


def _result(side, obj, version, seed0, episodes, trajectories, record, geometry):
    """trajectories: one per episode in order, for the first episode only or for all of them (see record)."""
    from reallab.pusht import api
    out = dict(side=side, task=obj, version=version, seed0=seed0, n=len(episodes), time_limit=api.TIME_LIMIT,
               success=[bool(e["success"]) for e in episodes], n_steps=[int(e["n_steps"]) for e in episodes],
               first=None, geometry=geometry)
    if record:
        e = episodes[0]
        out["first"] = dict(seed=int(e["seed"]), success=bool(e["success"]), n_steps=int(e["n_steps"]),
                            trajectory=trajectories[0])
    if record == "all":
        out["trajectories"] = trajectories
    return out


# ---------------------------------------------------------------------------------------------------- world model
@contextlib.contextmanager
def _tap(W, frames):
    """Copy (pusher positions, block poses) of the batch from every state the world module creates or steps to."""
    make_state, physics = W.make_state, W.physics

    def copy(state):
        frames.append((state["p"].tolist(), state["q"].tolist()))
        return state

    W.make_state = lambda *a, **k: copy(make_state(*a, **k))
    W.physics = lambda *a, **k: copy(physics(*a, **k))
    try:
        yield
    finally:
        W.make_state, W.physics = make_state, physics


def world_geometry(obj: str) -> dict:
    from worlds.pusht import pusht_world as W
    o = W.OBJECTS[obj]
    return dict(block=[[[float(u), float(v)] for u, v in o["verts"]]], goal=[float(x) for x in o["goal"]],
                pusher_radius=float(W.R_PUSHER), workspace=[512.0, 512.0], wall=[float(W.WALL_LO), float(W.WALL_HI)])


def run_world(obj: str, version: int, seed0: int, n: int, record=True) -> dict:
    from experiments.pusht.code_wm import screen
    from worlds.pusht import pusht_world as W
    args = (factory(obj, version), {}, obj, list(range(int(seed0), int(seed0) + int(n))))
    frames = []
    with _tap(W, frames) if record else contextlib.nullcontext():
        episodes = screen._run_chunk(args)
    # the batch keeps stepping until its last episode ends: keep each row up to the end of its own episode
    trajectories = [[_row(*p[i], *q[i]) for p, q in frames[:episodes[i]["n_steps"] + 1]]
                    for i in range(len(episodes) if record == "all" else 1 if record else 0)]
    return _result("world", obj, version, int(seed0), episodes, trajectories, record, world_geometry(obj))


# ---------------------------------------------------------------------------------------------------- environment
def real_geometry(obj: str) -> dict:
    """Drawn from the public task card: the block is the union of its two rectangles."""
    import shapely.geometry as sg
    from shapely.ops import unary_union
    from reallab.pusht import api
    card = api.task_card(obj) if obj != "T" else api.task_card()
    rects = [sg.box(r["u"][0], r["v"][0], r["u"][1], r["v"][1]) for r in card["block"].values()
             if isinstance(r, dict) and "u" in r and "v" in r]
    shape = sg.polygon.orient(unary_union(rects).simplify(0.0))
    half = card["walls"]["half_thickness"]
    (lo, _), (hi, _) = card["walls"]["corners"]
    return dict(block=[[[float(u), float(v)] for u, v in list(shape.exterior.coords)[:-1]]],
                goal=[float(x) for x in card["goal_pose"]], pusher_radius=float(card["pusher"]["radius"]),
                workspace=[float(x) for x in card["workspace"]["high"]], wall=[lo + half, hi - half])


def run_real(obj: str, version: int, seed0: int, n: int, record=True, tag: str = LEDGER_TAG) -> dict:
    import numpy as np
    from reallab.pusht import parallel
    seeds = list(range(int(seed0), int(seed0) + int(n)))
    cfg = dict(policy_factory=factory(obj, version), policy_kwargs={}, purpose="eval", pilot="", tag=tag,
               log_dir=None, record_first=0, record_format="gif", video_dir=None, object=obj)
    tasks, results = queue.SimpleQueue(), queue.SimpleQueue()
    for i, s in enumerate(seeds):
        tasks.put((i, s))
    tasks.put(None)
    trajectories = []
    with tempfile.TemporaryDirectory(prefix="space_run_") as tmp:
        if record:
            cfg["log_dir"] = tmp
        parallel._worker(0, tasks, results, cfg)
        episodes = {}
        while not results.empty():
            kind, idx, payload = results.get()
            if kind != "ok":
                raise RuntimeError(f"{payload[0]}: {payload[1]}")
            episodes[idx] = payload
        if len(episodes) != len(seeds):
            raise RuntimeError(f"{len(episodes)} of {len(seeds)} episodes finished")
        for s in seeds if record == "all" else seeds[:1] if record else []:
            with np.load(Path(tmp) / f"ep_{s}.npz") as log:
                trajectories.append([_row(*p, *b) for p, b in zip(log["agent_pos"], log["block_pose"])])
    return _result("real", obj, version, int(seed0), [episodes[i] for i in range(len(seeds))], trajectories, record,
                   real_geometry(obj))


# ---------------------------------------------------------------------------------------------------- worker process
def serve(conn) -> None:
    """Loop of one worker process: import the simulators once, then run the jobs received on the pipe `conn`.

    job = (side, obj, version, seed0, n, record) with side "world" or "real"; the reply is ("ok", result) or
    ("error", text).  The first message sent is ("ready", None) or ("failed", text).  None, or the pipe closing,
    ends the loop."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)        # the parent decides when a worker stops
    try:
        run_world("T", PROGRAMS["T"][-1], 0, 1, record=False)       # imports torch, the world model and a program
        import gym_pusht.envs  # noqa: F401  the simulator behind reallab.pusht
        from reallab.pusht import parallel  # noqa: F401
        real_geometry("T")
        conn.send(("ready", None))
    except Exception as ex:
        conn.send(("failed", f"{type(ex).__name__}: {ex}"))
        return
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        side, obj, version, seed0, n, record = job
        try:
            run = run_world if side == "world" else run_real
            conn.send(("ok", run(obj, version, seed0, n, record)))
        except Exception as ex:
            conn.send(("error", f"{type(ex).__name__}: {ex}"))
