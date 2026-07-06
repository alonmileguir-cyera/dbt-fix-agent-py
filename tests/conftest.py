"""Global pytest configuration: proves the offline-testing contract.

Every test module in this suite must be able to run with zero real network
access and zero real subprocess execution, *except* an explicitly marked
`real_process` module (added in a later sprint, once there is anything real
to integration-test against -- the auditor subprocess, a real `dbt`
executable, etc.).

This is enforced actively, not just by convention: an autouse fixture
monkeypatches `socket.socket` and `subprocess.Popen.__init__` to raise for
every test *unless* the test (or its module) carries the `real_process`
marker. Sprint 1 has no `real_process` tests at all -- every Sprint 1 module
is pure, offline computation -- so this guard is simply always active for
now, and stays available for later sprints to build on.
"""

from __future__ import annotations

import socket
import subprocess

import pytest


class BlockedNetworkAccessError(RuntimeError):
    """Raised when a test tries to open a real socket without the `real_process` marker."""


class BlockedSubprocessAccessError(RuntimeError):
    """Raised when a test tries to spawn a real subprocess without the `real_process` marker."""


def _blocked_socket(*_args, **_kwargs):
    raise BlockedNetworkAccessError(
        "real network access is blocked in offline tests; inject a fake instead "
        "(mark the test `real_process` if it genuinely needs the network)"
    )


def _blocked_popen_init(self, *_args, **_kwargs):
    raise BlockedSubprocessAccessError(
        "real subprocess access is blocked in offline tests; inject a fake instead "
        "(mark the test `real_process` if it genuinely needs a real subprocess)"
    )


@pytest.fixture(autouse=True)
def _block_network_and_subprocess(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("real_process") is not None:
        yield
        return

    original_socket = socket.socket
    original_popen_init = subprocess.Popen.__init__

    socket.socket = _blocked_socket  # type: ignore[assignment]
    subprocess.Popen.__init__ = _blocked_popen_init  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment]
        subprocess.Popen.__init__ = original_popen_init  # type: ignore[assignment]
