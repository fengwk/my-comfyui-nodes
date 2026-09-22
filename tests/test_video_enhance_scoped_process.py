from __future__ import annotations

import sys
import unittest

from my_nodes.core.video_enhance import ProcessError, ProcessInterrupted, ProcessIOError, ScopedProcess
from my_nodes.core.video_enhance.scoped_process import ProcessTimeout

from .video_enhance_fixtures import assert_process_gone

# Child programs. Every one of them is a real process talking over real pipes:
# the point is the process owner, not the payload.
ECHO_SIZE = 4 * 1024 * 1024  # far beyond the default 64 KiB pipe buffer
DRAIN_THEN_REPLY = f"""
import sys
remaining = {ECHO_SIZE}
handle = sys.stdin.buffer
while remaining:
    chunk = handle.read(min(65536, remaining))
    if not chunk:
        break
    remaining -= len(chunk)
sys.stdout.buffer.write(b"ok:" + str({ECHO_SIZE} - remaining).encode())
sys.stdout.buffer.flush()
"""
FLOOD_STDOUT = f"""
import sys
sys.stdout.buffer.write(b"x" * {ECHO_SIZE})
sys.stdout.buffer.flush()
"""
EXIT_AFTER_STDIN_EOF = """
import sys
data = sys.stdin.buffer.read()
print(len(data), file=sys.stderr, flush=True)
"""
SILENT_FOREVER = """
import time
time.sleep(60)
"""
IGNORE_TERM_AND_LEAK_GRANDCHILD = """
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print("grandchild", child.pid, file=sys.stderr, flush=True)
time.sleep(60)
"""
RETURN_GROUP = """
import os, sys
# Fixed 64-byte line so the parent can read it with a single exact read.
line = f"{os.getpid()} {os.getpgrp()} {os.getsid(0)}"
print(line.ljust(63))
sys.stdout.flush()
"""
CRASH_WITH_STDERR = """
import sys
print("panic in the fake worker", file=sys.stderr, flush=True)
sys.exit(3)
"""
CHATTY_STDERR = """
import sys, time
# ~16 KiB of stderr: more than the bounded capture keeps.
for index in range(400):
    print(f"line {index:04d} " + "x" * 32, file=sys.stderr)
sys.stderr.flush()
time.sleep(1)
"""
GRANDCHILD_MARKER = "grandchild "


class ScopedProcessTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._pids: list[int] = []
        self.addCleanup(self._assert_all_gone)

    def _assert_all_gone(self) -> None:
        for pid in self._pids:
            assert_process_gone(pid)

    def track(self, process: ScopedProcess) -> ScopedProcess:
        if process.pid is not None:
            self._pids.append(process.pid)
        return process

    def spawn(self, program: str, **kwargs) -> ScopedProcess:
        return self.track(ScopedProcess((sys.executable, "-c", program), **kwargs))

    def grandchild_pids(self, process: ScopedProcess) -> list[int]:
        """PIDs the child reported before it was killed, and leak-check them."""
        pids = [
            int(line.split()[1])
            for line in process.stderr_text().splitlines()
            if line.startswith(GRANDCHILD_MARKER)
        ]
        self.assertTrue(pids, f"no grandchild was reported: {process.stderr_text()!r}")
        self._pids.extend(pids)
        return pids

    def await_marker(self, process: ScopedProcess, marker: str, timeout: float = 5.0) -> None:
        """Wait until the child's stderr mentions `marker` (drained while waiting)."""
        deadline = timeout
        while deadline > 0:
            if marker in process.stderr_text():
                return
            try:
                process.read_exactly(1, timeout=0.05, what="probe")
            except ProcessError:
                pass
            deadline -= 0.05
        self.fail(f"child never reported {marker!r}: {process.stderr_text()!r}")


class ExchangeTests(ScopedProcessTestCase):
    def test_large_payload_round_trip_uses_partial_reads_and_writes(self) -> None:
        # A 4 MiB request cannot fit in one pipe write, so this only passes if
        # the owner loops over partial os.write/os.read results.
        expected = b"ok:" + str(ECHO_SIZE).encode()
        with self.spawn(DRAIN_THEN_REPLY, timeout=20.0) as process:
            process.write(b"y" * ECHO_SIZE, what="bulk request")
            self.assertEqual(
                process.read_exactly(len(expected), what="bulk reply"), expected
            )
        self.assertEqual(process.returncode, 0)

    def test_large_reply_is_read_completely(self) -> None:
        with self.spawn(FLOOD_STDOUT, timeout=20.0) as process:
            self.assertEqual(process.read_exactly(ECHO_SIZE, what="bulk reply"), b"x" * ECHO_SIZE)
        self.assertEqual(process.returncode, 0)

    def test_child_runs_in_its_own_session(self) -> None:
        # killpg() is only safe when the child leads its own group.
        with self.spawn(RETURN_GROUP, timeout=5.0) as process:
            pid, pgrp, sid = process.read_exactly(64, what="group report").split()
            self.assertEqual(int(pid), process.pid)
            self.assertEqual(int(pgrp), process.pid)
            self.assertEqual(int(sid), process.pid)

    def test_clean_close_waits_for_the_child_and_reaps_it(self) -> None:
        process = self.spawn(EXIT_AFTER_STDIN_EOF, timeout=5.0)
        with process:
            pass
        self.assertFalse(process.is_alive())
        self.assertEqual(process.returncode, 0)
        # "0" is the byte count the child read, i.e. the EOF our close sent.
        self.assertIn("0", process.stderr_text())

    def test_close_is_idempotent_after_a_clean_run(self) -> None:
        process = self.spawn(EXIT_AFTER_STDIN_EOF, timeout=5.0)
        with process:
            pass
        process.close()
        process.terminate()
        self.assertFalse(process.is_alive())

    def test_single_use_guard_and_stream_guards(self) -> None:
        process = self.spawn(SILENT_FOREVER, timeout=1.0)
        process.__enter__()
        with self.assertRaises(ProcessError):
            process.__enter__()
        process.terminate()
        with self.assertRaises(ProcessError):
            process.write(b"x")
        with self.assertRaises(ProcessError):
            process.read_exactly(1)

    def test_write_before_start_is_rejected(self) -> None:
        process = ScopedProcess((sys.executable, "-c", "pass"))
        with self.assertRaises(ProcessError):
            process.write(b"x")


class FailureTests(ScopedProcessTestCase):
    def test_short_reply_is_reported_with_exit_code_and_stderr(self) -> None:
        with self.assertRaises(ProcessIOError) as raised:
            with self.spawn(CRASH_WITH_STDERR, timeout=5.0) as process:
                process.read_exactly(16, what="reply header")
        message = str(raised.exception)
        self.assertIn("reply header", message)
        self.assertIn("exit code 3", message)
        self.assertIn("panic in the fake worker", message)

    def test_timeout_on_a_silent_child(self) -> None:
        with self.assertRaises(ProcessTimeout) as raised:
            with self.spawn(SILENT_FOREVER, timeout=0.3, poll_interval=0.02) as process:
                process.read_exactly(4, what="reply header")
        self.assertIn("0.300s", str(raised.exception))

    def test_timeout_on_a_stalled_write(self) -> None:
        # The child never reads, so the 4 MiB write fills the pipe and must hit
        # the deadline instead of blocking the caller forever.
        with self.assertRaises(ProcessTimeout) as raised:
            with self.spawn(SILENT_FOREVER, timeout=0.3, poll_interval=0.02) as process:
                process.write(b"z" * ECHO_SIZE, what="bulk request")
        self.assertIn("bulk request", str(raised.exception))

    def test_stderr_capture_is_bounded_and_keeps_the_tail(self) -> None:
        process = self.spawn(CHATTY_STDERR, timeout=5.0)
        with process:
            with self.assertRaises(ProcessIOError):
                process.read_exactly(1, what="reply")
        text = process.stderr_text()
        self.assertTrue(text.startswith("..."), text[:40])
        self.assertLessEqual(len(text), 8192 + 4)
        self.assertIn("line 0399", text)


class CleanupTests(ScopedProcessTestCase):
    def test_body_exception_still_kills_and_reaps_the_child(self) -> None:
        process = self.spawn(SILENT_FOREVER, timeout=5.0)
        with self.assertRaises(ValueError):
            with process:
                raise ValueError("stage failed")
        self.assertFalse(process.is_alive())
        assert_process_gone(process.pid)

    def test_interrupt_returning_true_stops_the_wait_and_cleans_up(self) -> None:
        calls: list[int] = []

        def interrupt() -> bool:
            calls.append(1)
            return True

        process = self.spawn(SILENT_FOREVER, timeout=5.0, interrupt=interrupt)
        with self.assertRaises(ProcessInterrupted):
            with process:
                process.read_exactly(4, what="reply header")
        self.assertTrue(calls)
        self.assertFalse(process.is_alive())
        assert_process_gone(process.pid)

    def test_custom_baseexception_from_the_interrupt_kills_the_process_group(self) -> None:
        class ComfyInterrupt(BaseException):
            """Stands in for a ComfyUI interrupt: a BaseException, not an Exception."""

        state: dict[str, ScopedProcess] = {}

        def interrupt() -> bool:
            # Only cancel once the child really owns a grandchild, so the group
            # cleanup is what we are actually asserting.
            if GRANDCHILD_MARKER in state["process"].stderr_text():
                raise ComfyInterrupt("cancelled by the user")
            return False

        process = self.spawn(
            IGNORE_TERM_AND_LEAK_GRANDCHILD,
            timeout=5.0,
            interrupt=interrupt,
            poll_interval=0.02,
            shutdown_grace=0.3,
        )
        state["process"] = process
        with self.assertRaises(ComfyInterrupt):
            with process:
                process.read_exactly(4, what="reply header")
        self.assertFalse(process.is_alive())
        assert_process_gone(process.pid)
        for pid in self.grandchild_pids(process):
            assert_process_gone(pid)

    def test_sigterm_is_escalated_to_sigkill_for_a_stubborn_child(self) -> None:
        process = self.spawn(IGNORE_TERM_AND_LEAK_GRANDCHILD, timeout=5.0, shutdown_grace=0.3)
        with process:
            self.await_marker(process, GRANDCHILD_MARKER)
        self.assertFalse(process.is_alive())
        self.assertEqual(process.returncode, -9)
        for pid in self.grandchild_pids(process):
            assert_process_gone(pid)

    def test_exit_before_start_is_harmless(self) -> None:
        process = ScopedProcess((sys.executable, "-c", "pass"))
        process.close()
        process.terminate()
        self.assertIsNone(process.pid)

    def test_spawn_failure_is_actionable(self) -> None:
        with self.assertRaises(ProcessError) as raised:
            ScopedProcess(("/nonexistent/dnr3-worker",)).__enter__()
        self.assertIn("/nonexistent/dnr3-worker", str(raised.exception))
