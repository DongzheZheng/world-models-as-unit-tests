"""pytest setup of the harness tests (helpers: harness_support.py)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_support as hs  # noqa: E402

_counter = itertools.count()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes minutes (the full pusht reproduction)")


@pytest.fixture
def make_run(tmp_path):
    """make_run(overrides=None, init=True, files=None) -> the resolved config of a fresh run under tmp_path."""

    def make(overrides=None, init=True, files=None):
        return hs.new_run(tmp_path / f"case{next(_counter)}", overrides, init, files)

    return make
