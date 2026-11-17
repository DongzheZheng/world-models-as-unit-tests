"""A small harness for the procedure of this repository, for your own environment.

A coding agent is pointed at a black-box environment with a budget of real trials.  It writes the environment's
world model as one Python file, obtains a controller inside that model and hands both in.  The harness then runs the
controller in the delivered world model (predicted success) and on held-out seeds in the environment (real success).

    python -m harness init    --config run.toml     create the run directory and the agent's workspace
    python -m harness serve   --run-dir DIR         keep the environment server up (for an agent you start yourself)
    python -m harness run     --config run.toml     init if needed, serve, run the agent, stop, verify
    python -m harness verify  --run-dir DIR         score the delivered policy in the delivered world and for real
    python -m harness screen  --config run.toml     score any policy files in a world model and in the environment
    python -m harness status  --run-dir DIR         trials used and deliverables present

Modules: contract (what an environment and the deliverables must look like), config (the TOML file), server (the
environment process and its rules), client/lab.py (the file the agent gets), scaffold (init), agents (command and
replay adapters), evaluate (episodes of untrusted programs, in subprocesses), verify, screen, report.

The core needs only the standard library and numpy.  Tested on Linux with Python 3.12; it needs Python 3.11 or
later and a POSIX system (inherited sockets, process groups).
"""

__version__ = "0.1.0"
