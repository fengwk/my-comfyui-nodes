"""One child process owned by one context.

`ScopedProcess` is the smallest possible owner of an external worker: it spawns
exactly one process for the lifetime of one `with` block, exchanges raw bytes
over its pipes, and guarantees that the child is gone *and reaped* when the
block ends, whatever happened inside it.

* `shell=False`, piped stdio, `start_new_session=True` on POSIX, so the child
  leads its own process group and signalling can never reach anything else.
* reads and writes are deadline aware (`select` + non-blocking `os.read` /
  `os.write`), so a stalled worker cannot block the caller forever, and a
  payload larger than the pipe buffer is handled in partial writes.
* stderr is drained while waiting and kept bounded (tail) for error messages.
* the interrupt callback is polled while waiting; a truthy return raises
  `ProcessInterrupted` and an exception it raises propagates unchanged.
* every exit path closes stdin, terminates the process *group* with SIGTERM,
  escalates to SIGKILL and reaps - including for a `BaseException` such as a
  ComfyUI interrupt. Cleanup errors never mask an exception already on its way
  out.

Deliberately not here: no daemon, no watchdog thread, no idle timer, no global
process cache and no `wineserver -k`. This class only ever signals the group it
created itself.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 600.0
TERMINATE_GRACE_SECONDS = 2.0
POLL_INTERVAL_SECONDS = 0.05
STDERR_LIMIT_BYTES = 8 * 1024
STDERR_CHUNK_BYTES = 4096
STDERR_SETTLE_SECONDS = 0.5
WRITE_CHUNK_BYTES = 1 << 16


class ProcessError(RuntimeError):
    """Base class for every scoped-process failure."""


class ProcessIOError(ProcessError):
    """The child closed a stream, refused a write or died unexpectedly."""


class ProcessTimeout(ProcessError, TimeoutError):
    """The child did not answer within the allowed time."""


class ProcessInterrupted(ProcessError):
    """The interrupt callback asked to cancel while waiting."""


class ScopedProcess:
    """Run one child process for the lifetime of one `with` block."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        interrupt: Callable[[], object] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        stderr_limit: int = STDERR_LIMIT_BYTES,
        shutdown_grace: float = TERMINATE_GRACE_SECONDS,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if shutdown_grace < 0:
            raise ValueError("shutdown_grace must not be negative")
        self._command = tuple(str(part) for part in command)
        self._cwd = cwd
        self._env = None if env is None else {str(k): str(v) for k, v in env.items()}
        self._interrupt = interrupt
        self._timeout = float(timeout)
        self._poll_interval = float(poll_interval)
        self._stderr_limit = max(int(stderr_limit), 0)
        self._shutdown_grace = float(shutdown_grace)
        self._process: subprocess.Popen[bytes] | None = None
        self._selector = selectors.DefaultSelector()
        self._in_fd: int | None = None
        self._out_fd: int | None = None
        self._err_fd: int | None = None
        self._writing = False
        self._stderr = bytearray()
        self._stderr_truncated = False
        self._released = False
        self._closed = False

    # ------------------------------------------------------------------ public

    @property
    def pid(self) -> int | None:
        """PID of the child, or None when it was never spawned."""
        return None if self._process is None else self._process.pid

    @property
    def returncode(self) -> int | None:
        """Exit status of the reaped child, or None while it is still running."""
        return None if self._process is None else self._process.returncode

    @property
    def command(self) -> tuple[str, ...]:
        return self._command

    def is_alive(self) -> bool:
        """True while the child has not been reaped."""
        return self._process is not None and self._process.poll() is None

    def stderr_text(self) -> str:
        """Bounded stderr tail, decoded for error messages."""
        text = bytes(self._stderr).decode("utf-8", errors="replace")
        return f"...{text}" if self._stderr_truncated else text

    def __enter__(self) -> ScopedProcess:
        if self._process is not None:
            raise ProcessError("a scoped process is single use; create one per context")
        self._spawn()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        # A clean block may end gracefully and may report a worker that refuses
        # to die; an aborted block must never mask the propagating exception.
        if exc_type is None:
            self._shut_down(graceful=True, strict=True)
        else:
            try:
                self._shut_down(graceful=False, strict=False)
            except BaseException:
                pass
        return False

    def write(self, data: bytes | bytearray | memoryview, *, timeout: float | None = None,
              what: str = "request") -> None:
        """Write every byte of `data`, raising rather than blocking forever."""
        view = memoryview(data).cast("B")
        fd = self._require_stream(self._in_fd, "send")
        limit, deadline = self._deadline(timeout)
        self._register_write()
        try:
            offset = 0
            while offset < len(view):
                self._wait_for(fd, selectors.EVENT_WRITE, deadline, limit, what)
                try:
                    written = os.write(fd, view[offset : offset + WRITE_CHUNK_BYTES])
                except BlockingIOError:
                    continue
                except (BrokenPipeError, OSError) as exc:
                    raise ProcessIOError(
                        f"cannot send {what} to {self._describe()}: {exc}"
                        f"{self._exit_detail(settle=True)}"
                    ) from exc
                offset += written
        finally:
            self._unregister_write()

    def read_exactly(self, size: int, *, timeout: float | None = None, what: str = "data") -> bytes:
        """Read exactly `size` bytes, raising on timeout, interrupt or EOF."""
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        fd = self._require_stream(self._out_fd, "read")
        limit, deadline = self._deadline(timeout)
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            self._wait_for(fd, selectors.EVENT_READ, deadline, limit, what)
            try:
                chunk = os.read(fd, remaining)
            except BlockingIOError:
                continue
            except OSError as exc:
                raise ProcessIOError(
                    f"cannot read {what} from {self._describe()}: {exc}"
                    f"{self._exit_detail(settle=True)}"
                ) from exc
            if not chunk:
                raise ProcessIOError(
                    f"{self._describe()} closed its output after {size - remaining} of "
                    f"{size} bytes of {what}{self._exit_detail(settle=True)}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def wait_exit(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the child; True once it was reaped."""
        process = self._process
        if process is None:
            return True
        if process.poll() is not None:
            return True
        if timeout <= 0:
            return process.poll() is not None
        try:
            process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def close_stdin(self) -> None:
        """Close the child's stdin: the usual "no more input" signal."""
        if self._in_fd is None:
            return
        self._unregister_write()
        self._in_fd = None
        stream = self._process.stdin if self._process is not None else None
        if stream is not None:
            self._close_stream(stream)

    def close(self) -> None:
        """Reap the child, gracefully when it is still running; idempotent."""
        self._shut_down(graceful=True, strict=True)

    def terminate(self) -> None:
        """Reap the child now, killing the whole group; idempotent and quiet."""
        try:
            self._shut_down(graceful=False, strict=False)
        except BaseException:
            # Never mask the exception this cleanup is running next to.
            pass

    # ----------------------------------------------------------------- process

    def _spawn(self) -> None:
        options: dict[str, Any] = {
            "shell": False,
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
            "cwd": self._cwd,
            "env": self._env,
        }
        if os.name == "posix":
            # Own session: the worker and everything it forks share one group,
            # and that group contains nothing but the worker.
            options["start_new_session"] = True
        try:
            self._process = subprocess.Popen(self._command, **options)
        except OSError as exc:
            raise ProcessError(f"cannot start {self._command[0]}: {exc}") from exc
        process = self._process
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        self._in_fd = process.stdin.fileno()
        self._out_fd = process.stdout.fileno()
        self._err_fd = process.stderr.fileno()
        os.set_blocking(self._in_fd, False)
        os.set_blocking(self._out_fd, False)
        os.set_blocking(self._err_fd, False)
        self._selector.register(self._out_fd, selectors.EVENT_READ, "stdout")
        self._selector.register(self._err_fd, selectors.EVENT_READ, "stderr")

    def _register_write(self) -> None:
        if self._writing or self._in_fd is None:
            return
        self._selector.register(self._in_fd, selectors.EVENT_WRITE, "stdin")
        self._writing = True

    def _unregister_write(self) -> None:
        if not self._writing or self._in_fd is None:
            self._writing = False
            return
        try:
            self._selector.unregister(self._in_fd)
        except (KeyError, ValueError, OSError):
            pass
        self._writing = False

    # ------------------------------------------------------------------ streams

    def _deadline(self, timeout: float | None) -> tuple[float, float]:
        limit = self._timeout if timeout is None else float(timeout)
        if limit <= 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")
        return limit, time.monotonic() + limit

    def _require_stream(self, fd: int | None, action: str) -> int:
        if self._process is None:
            raise ProcessError("scoped process was never started")
        if self._closed:
            raise ProcessError(f"scoped process is closed; cannot {action} {self._describe()}")
        if fd is None:
            raise ProcessError(f"cannot {action}: the stream is already closed")
        return fd

    def _wait_for(
        self, fd: int, events: int, deadline: float, limit: float, what: str
    ) -> None:
        """Block until `fd` is ready for `events`, draining stderr meanwhile."""
        while True:
            self._check_interrupt(what)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProcessTimeout(
                    f"{self._describe()} did not answer within {limit:.3f}s while waiting "
                    f"for {what}{self._exit_detail()}"
                )
            try:
                ready = self._selector.select(min(remaining, self._poll_interval))
            except OSError as exc:
                raise ProcessIOError(f"cannot poll {self._describe()}: {exc}") from exc
            for key, mask in ready:
                if key.fd == fd and mask & events:
                    return
                if key.data == "stderr" and key.fd == self._err_fd:
                    self._read_stderr()

    def _check_interrupt(self, what: str) -> None:
        callback = self._interrupt
        if callback is None:
            return
        if callback():
            raise ProcessInterrupted(
                f"interrupted while waiting for {what} from {self._describe()}"
            )

    def _read_stderr(self) -> None:
        """Drain whatever the child wrote to stderr into the bounded tail."""
        if self._err_fd is None:
            return
        try:
            chunk = os.read(self._err_fd, STDERR_CHUNK_BYTES)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._drop_stderr_stream()
            return
        if not chunk:
            # EOF: unregister, otherwise select() reports a readable fd forever.
            self._drop_stderr_stream()
            return
        self._append_stderr(chunk)

    def _drop_stderr_stream(self) -> None:
        if self._err_fd is None:
            return
        try:
            self._selector.unregister(self._err_fd)
        except (KeyError, ValueError, OSError):
            pass
        self._err_fd = None

    def _append_stderr(self, chunk: bytes) -> None:
        if self._stderr_limit <= 0:
            self._stderr_truncated = True
            return
        self._stderr.extend(chunk)
        excess = len(self._stderr) - self._stderr_limit
        if excess > 0:
            # Keep the tail: the last lines before a crash say the most.
            del self._stderr[:excess]
            self._stderr_truncated = True

    def _drain_stderr(self, timeout: float) -> None:
        """Collect the remaining stderr of a child that has stopped writing."""
        deadline = time.monotonic() + timeout
        while self._err_fd is not None and time.monotonic() < deadline:
            self._read_stderr()

    # ------------------------------------------------------------------- cleanup

    def _shut_down(self, *, graceful: bool, strict: bool) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.close_stdin()
            if graceful:
                self.wait_exit(self._shutdown_grace)
            if not self.wait_exit(0.0):
                self._signal_group(force=False)
                if not self.wait_exit(self._shutdown_grace):
                    self._signal_group(force=True)
                    self.wait_exit(self._shutdown_grace)
            self._drain_stderr(STDERR_SETTLE_SECONDS)
            if strict and self.is_alive():
                raise ProcessError(
                    f"{self._describe()} ignored SIGKILL; giving up on a clean reap"
                )
        except BaseException:
            self._kill_quietly()
            raise
        finally:
            self._release()

    def _signal_group(self, *, force: bool) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        if os.name == "posix":
            try:
                # start_new_session made the child its own group leader, so the
                # PID is the group id and no process outside it is signalled.
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                return
            except OSError:
                pass
        try:
            process.kill() if force else process.terminate()
        except OSError:
            pass

    def _kill_quietly(self) -> None:
        """Last resort used while an exception is already propagating."""
        try:
            self._signal_group(force=True)
            self.wait_exit(self._shutdown_grace)
        except BaseException:
            pass

    def _release(self) -> None:
        """Close every fd and the selector; only the captured stderr survives."""
        if self._released:
            return
        self._released = True
        self._unregister_write()
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                self._close_stream(stream)
        self._in_fd = self._out_fd = None
        self._err_fd = None
        try:
            self._selector.close()
        except (OSError, ValueError):
            pass

    @staticmethod
    def _close_stream(stream: Any) -> None:
        if stream is None:
            return
        try:
            stream.close()
        except OSError:
            pass

    # ----------------------------------------------------------------- reporting

    def _describe(self) -> str:
        pid = self.pid
        return "worker" if pid is None else f"worker pid {pid}"

    def _exit_detail(self, *, settle: bool = False) -> str:
        """Suffix describing the child state, for error messages."""
        process = self._process
        if process is None:
            return ""
        if settle and process.poll() is None:
            # EOF/BrokenPipe can become visible just before waitpid reports the
            # exit. Reap that short race so diagnostics include the real code.
            try:
                process.wait(timeout=STDERR_SETTLE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            return "; still running"
        if not self._released:
            self._drain_stderr(STDERR_SETTLE_SECONDS)
        stderr = self.stderr_text().strip()
        prefix = f"; exit code {process.returncode}"
        return f"{prefix}; stderr: {stderr}" if stderr else prefix
