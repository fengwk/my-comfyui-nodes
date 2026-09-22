"""Shared helpers for the video-enhance DNR3 tests."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from my_nodes.core.video_enhance import FEATURE_NR, FEATURE_SR
from my_nodes.core.video_enhance.runtime import CORE_DLL, NR_DLL, SR_DLL

FAKE_WORKER = Path(__file__).with_name("video_enhance_fake_dnr3_worker.py")

_GONE_TIMEOUT_SECONDS = 3.0


def create_runtime_dir(
    root: Path,
    features: int,
    *,
    nr_name: str | None = None,
    skip: tuple[str, ...] = (),
) -> Path:
    """Create a runtime directory holding the placeholder files of `features`.

    Only the files the features require are written, which is exactly what the
    feature-specific validation tests want to prove. `skip` leaves a file out
    on purpose.
    """
    root.mkdir(parents=True, exist_ok=True)
    names = [CORE_DLL]
    if features & FEATURE_SR:
        names.append(SR_DLL)
    if features & FEATURE_NR:
        names.append(nr_name or NR_DLL)
    for name in names:
        if name in skip:
            continue
        (root / name).write_bytes(b"fake NVIDIA runtime placeholder")
    return root


def fake_worker_command(mode: str = "ok") -> tuple[str, ...]:
    """argv that runs the fake DNR3 worker in `mode` instead of the real host."""
    return (sys.executable, str(FAKE_WORKER), mode)


def read_report(path: Path) -> dict[str, Any]:
    """Read the fake worker's JSON report (empty when it never wrote one)."""
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def wait_for_report(path: Path, timeout: float = 5.0) -> dict[str, Any]:
    """Wait until the fake worker has recorded its session header."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        report = read_report(path)
        if "header" in report:
            return report
        time.sleep(0.02)
    raise AssertionError(f"fake DNR3 worker never reported its header: {read_report(path)}")


def assert_process_gone(pid: int | None, timeout: float = _GONE_TIMEOUT_SECONDS) -> None:
    """Fail unless `pid` is gone: the strongest cleanup evidence we can get."""
    if pid is None:
        raise AssertionError("no pid was recorded, so cleanup cannot be verified")
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:  # pragma: no cover - different user
            return
        if time.monotonic() >= deadline:
            raise AssertionError(f"process {pid} is still alive after cleanup")
        time.sleep(0.02)
