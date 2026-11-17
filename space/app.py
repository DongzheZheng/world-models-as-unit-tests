"""Web app of the interactive article: the built page plus an API that runs a pushing program on the server.

    uvicorn space.app:app --host 0.0.0.0 --port 7860        (from the repository root; one uvicorn process)

GET  /                the article with the interactive panel (built by space/build_page.py; built here if missing)
GET  /media/...       the article's videos and posters
GET  /api/programs    tasks, selectable programs with their tags and recorded results, N, the default selection
POST /api/run/world   body {"task": "T" | "L", "version": int}: N episodes inside the world model
POST /api/run/real    same body: N episodes in the environment
GET  /api/health      worker and queue state (503 until both workers are ready)

A run request is answered with newline-delimited JSON, one object per line, written as things happen:
    {"status": "queued", "ahead": 2}       only while other runs are before this one (repeated every few seconds)
    {"status": "running", "queued_s": 3.1}
    {"status": "done", "queued_s": 3.1, "elapsed_s": 1.6, "k": 7, ...the result of space.runner...}
or, instead of the last line, {"status": "error", "detail": "..."}.  Invalid input is refused with 422 and a full
queue with 429 before anything is streamed.

One run executes at a time, first come first served; at most MAX_WAITING further requests wait.  The work is done
by two worker processes started once at startup (space.runner.serve); a batch is split over both.  The server draws
the first seed of every batch itself.  A batch that does not finish within RUN_TIMEOUT_S is abandoned and its
workers are replaced.

Every environment episode is metered in runs/ledger.jsonl by cawm.ledger.  To keep that file bounded, it is cut
down to its newest LEDGER_KEEP_BYTES whenever it is larger than LEDGER_MAX_BYTES (checked at startup and before
every environment batch).
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from cawm import ledger  # noqa: E402
from space import build_page, runner  # noqa: E402

N = runner.N_EPISODES
N_WORKERS = 2
MAX_WAITING = 6                 # requests waiting behind the running one; the next is refused with 429
RUN_TIMEOUT_S = 60.0            # one batch (world or environment)
START_TIMEOUT_S = 180.0         # worker start-up: importing the simulators
QUEUE_BEAT_S = 4.0              # a waiting request is told its place this often
SEEDS = (1_000_000, 2_000_000_000)   # first seeds are drawn from here: far above the seeds of the recorded evaluations
#                                      (1000-6999 in the environment, 50000-50199 in the world model), and >= 1000
#                                      as purpose "eval" of the environment requires
LEDGER_MAX_BYTES, LEDGER_KEEP_BYTES = 256_000, 64_000
PAGE_DIR = Path(os.environ.get("SPACE_PAGE_DIR") or build_page.DEFAULT_OUT)

TAGS, DEFAULT = runner.TAGS, runner.DEFAULT

log = logging.getLogger("uvicorn.error")


def catalog() -> dict:
    """What the panel offers.  Recorded pairs: success inside the world model (runs/probe/screen_pusht.json, 200
    episodes) and in the environment (runs/probe/curves.json, 50 episodes) of programs v1 to v15."""
    world = json.loads((ROOT / "runs/probe/screen_pusht.json").read_text())
    real = json.loads((ROOT / "runs/probe/curves.json").read_text())
    programs = []
    for obj, versions in runner.PROGRAMS.items():
        for v in versions:
            w, r = world[obj].get(f"v{v}"), real.get(f"push_{obj}/aspire_v{v}")
            rec = dict(world=w["success"], real=r["success"], world_n=w["n"], real_n=r["n"]) if w and r else None
            programs.append(dict(task=obj, version=v, tag=TAGS.get((obj, v)), recorded=rec))
    if sum(p["recorded"] is not None for p in programs) != 15:
        raise RuntimeError("expected recorded results for programs v1 to v15")
    return dict(n=N, default=DEFAULT, tasks=list(runner.PROGRAMS), programs=programs,
                geometry={obj: runner.real_geometry(obj) for obj in runner.PROGRAMS})


def trim_ledger() -> None:
    path = ledger.default_path()
    try:
        if path.stat().st_size <= LEDGER_MAX_BYTES:
            return
        with open(path, "rb") as f:
            f.seek(-LEDGER_KEEP_BYTES, os.SEEK_END)
            tail = f.read()
        tmp = path.with_name(path.name + ".trim")
        tmp.write_bytes(tail.partition(b"\n")[2])           # whole records only
        os.replace(tmp, path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------------------------------- workers
class RunFailed(Exception):
    pass


class Worker:
    """One spawned process running space.runner.serve, driven over a pipe (blocking calls: use from a thread)."""

    def __init__(self):
        self.proc = self.conn = None

    def start(self) -> None:
        self.stop()
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=runner.serve, args=(child,), daemon=True, name="space-worker")
        self.proc.start()
        child.close()
        try:
            kind, payload = self._reply(START_TIMEOUT_S)
        except Exception as ex:
            kind, payload = "failed", f"{type(ex).__name__}: {ex}"
        if kind != "ready":
            self.stop()
            raise RuntimeError(f"worker did not start: {payload}")

    def _reply(self, timeout: float):
        if not self.conn.poll(timeout):
            raise TimeoutError(f"no reply within {timeout:.0f} s")
        return self.conn.recv()             # EOFError if the process died

    def call(self, job: tuple, timeout: float) -> dict:
        if not self.alive():
            self.start()
        self.conn.send(job)
        kind, payload = self._reply(timeout)
        if kind != "ok":
            raise RunFailed(payload)        # the job raised; the worker itself is fine
        return payload

    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.join(10)
        if self.conn is not None:
            self.conn.close()
        self.proc = self.conn = None


class Pool:
    def __init__(self, size: int):
        self.workers = [Worker() for _ in range(size)]

    async def start(self) -> None:
        await asyncio.gather(*(asyncio.to_thread(w.start) for w in self.workers))

    async def stop(self) -> None:
        await asyncio.gather(*(asyncio.to_thread(w.stop) for w in self.workers))

    async def run(self, side: str, obj: str, version: int, seed0: int, n: int) -> dict:
        """Split the batch over the workers (the first part records the first episode) and join the parts."""
        sizes = [n // len(self.workers) + (i < n % len(self.workers)) for i in range(len(self.workers))]
        calls, s = [], seed0
        for i, (w, m) in enumerate(zip(self.workers, sizes)):
            if m:
                calls.append((w, asyncio.to_thread(w.call, (side, obj, version, s, m, i == 0), RUN_TIMEOUT_S)))
                s += m
        parts = await asyncio.gather(*(c for _, c in calls), return_exceptions=True)
        failed = [(w, p) for (w, _), p in zip(calls, parts) if isinstance(p, BaseException)]
        if failed:
            for w, p in failed:
                log.error("space: %s %s/v%d seed %d: %s: %s", side, obj, version, seed0, type(p).__name__, p)
                if not isinstance(p, RunFailed):            # timed out or died: replace the process
                    try:
                        await asyncio.to_thread(w.start)
                    except Exception as ex:
                        log.error("space: worker restart failed: %s", ex)
            timed_out = any(isinstance(p, TimeoutError) for _, p in failed)
            raise RunFailed("the run timed out" if timed_out else "the run failed on the server")
        out = dict(parts[0], seed0=seed0, n=n)
        out["success"] = [x for p in parts for x in p["success"]]
        out["n_steps"] = [x for p in parts for x in p["n_steps"]]
        return out


class Line:
    """One run at a time, first come first served, with a bounded number of waiting requests."""

    def __init__(self, max_waiting: int):
        self.max_waiting, self.busy = max_waiting, False
        self._waiting: collections.deque[asyncio.Future] = collections.deque()

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    def join(self) -> asyncio.Future | None:
        """A future that completes when it is this request's turn; None if the line is full."""
        turn = asyncio.get_running_loop().create_future()
        if not self.busy:
            self.busy = True
            turn.set_result(None)
        elif len(self._waiting) >= self.max_waiting:
            return None
        else:
            self._waiting.append(turn)
        return turn

    def ahead(self, turn: asyncio.Future) -> int:
        """Runs before this one, the running one included (0 once it is its turn)."""
        return self._waiting.index(turn) + 1 if turn in self._waiting else 0

    def leave(self, turn: asyncio.Future) -> None:
        """Give up a place in the line, or end the run and pass the turn on."""
        if turn in self._waiting:
            self._waiting.remove(turn)
        elif self._waiting:
            self._waiting.popleft().set_result(None)
        else:
            self.busy = False


pool, line = Pool(N_WORKERS), Line(MAX_WAITING)
state = dict(ready=False, started=time.time(), runs=0, catalog=None)
_rng = random.SystemRandom()


async def execute(side: str, obj: str, version: int) -> dict:
    if side == "real":
        trim_ledger()
    res = await pool.run(side, obj, version, _rng.randrange(SEEDS[0], SEEDS[1] - N), N)
    state["runs"] += 1
    return res


class RunStream(StreamingResponse):
    """Streams the events of one run; however the response ends, its place in the line is given back (once the
    workers are free again, if the client went away while they were busy)."""

    def __init__(self, side: str, obj: str, version: int, turn: asyncio.Future):
        super().__init__(self.events(), media_type="application/x-ndjson",
                         headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        self.side, self.obj, self.version, self.turn, self.job = side, obj, version, turn, None

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self.job is not None and not self.job.done():
                self.job.add_done_callback(lambda _: line.leave(self.turn))
            else:
                line.leave(self.turn)

    async def events(self):
        def event(**kw) -> bytes:
            return (json.dumps(kw, separators=(",", ":")) + "\n").encode()

        t0 = time.monotonic()
        while not self.turn.done():
            yield event(status="queued", ahead=line.ahead(self.turn))
            await asyncio.wait([self.turn], timeout=QUEUE_BEAT_S)
        queued_s = round(time.monotonic() - t0, 2)
        yield event(status="running", queued_s=queued_s)
        t1 = time.monotonic()
        self.job = asyncio.ensure_future(execute(self.side, self.obj, self.version))
        try:
            res = await asyncio.shield(self.job)        # a run that has started is not cancelled half-way
        except RunFailed as ex:
            yield event(status="error", detail=str(ex), queued_s=queued_s)
            return
        except Exception as ex:
            log.exception("space: %s %s/v%d: %s", self.side, self.obj, self.version, ex)
            yield event(status="error", detail="the run failed on the server", queued_s=queued_s)
            return
        elapsed_s = round(time.monotonic() - t1, 2)
        log.info("space: %s %s/v%d seed %d: %d/%d in %.2f s (queued %.2f s)", self.side, self.obj, self.version,
                 res["seed0"], sum(res["success"]), res["n"], elapsed_s, queued_s)
        yield event(status="done", queued_s=queued_s, elapsed_s=elapsed_s, k=sum(res["success"]), **res)


# ---------------------------------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_: FastAPI):
    if not (PAGE_DIR / "index.html").exists():
        build_page.build(PAGE_DIR)
    state["catalog"] = catalog()
    trim_ledger()
    await pool.start()
    state["ready"] = True
    log.info("space: %d workers ready, %d episodes per batch", N_WORKERS, N)
    yield
    state["ready"] = False
    await pool.stop()


app = FastAPI(title="World Models as Unit Tests", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: Literal["T", "L"]
    version: int


def start_run(side: str, req: RunRequest) -> RunStream:
    if req.version not in runner.PROGRAMS[req.task]:
        raise HTTPException(422, f"task {req.task} has no program v{req.version}")
    if not state["ready"]:
        raise HTTPException(503, "the server is starting")
    turn = line.join()
    if turn is None:
        raise HTTPException(429, "too many runs are waiting", headers={"Retry-After": "15"})
    return RunStream(side, req.task, req.version, turn)


@app.post("/api/run/world")
async def run_world(req: RunRequest):
    return start_run("world", req)


@app.post("/api/run/real")
async def run_real(req: RunRequest):
    return start_run("real", req)


@app.get("/api/programs")
async def programs():
    return JSONResponse(state["catalog"], headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
async def health():
    alive = [w.alive() for w in pool.workers]
    ok = state["ready"] and all(alive)
    return JSONResponse(dict(status="ok" if ok else "starting" if not state["ready"] else "degraded", workers=alive,
                             busy=line.busy, waiting=line.waiting, max_waiting=MAX_WAITING, n=N, runs=state["runs"],
                             uptime_s=round(time.time() - state["started"])),
                        status_code=200 if ok else 503, headers={"Cache-Control": "no-store"})


@app.api_route("/", methods=["GET", "HEAD"])
async def index():
    return FileResponse(PAGE_DIR / "index.html", media_type="text/html", headers={"Cache-Control": "no-cache"})


app.mount("/media", StaticFiles(directory=PAGE_DIR / "media", check_dir=False), name="media")
