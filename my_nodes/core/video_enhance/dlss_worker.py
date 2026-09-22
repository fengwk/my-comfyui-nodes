"""One DNR3 worker session: one execution, one host process, one protocol.

`Dnr3Worker` drives the whole exchange for a single video-enhance execution:

* it validates the runtime files the requested features need (via `HostDriver`),
* starts the vendored host once (`ScopedProcess`: own process group, pipes,
  deadline-aware reads/writes, no daemon and no reuse between executions),
* sends the DNR3 header, streams frames and returns the worker's RGB output,
* finishes with the worker's END1 and reaps the process.

Leaving the `with` block always ends with the worker process group gone: a
clean block asks for END1 first, anything else - timeout, protocol error, child
crash or a `BaseException` interrupt - terminates the group immediately.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

import numpy as np

from my_nodes.core.video_enhance import dnr3, scoped_process
from my_nodes.core.video_enhance.runtime import HostDriver, RuntimeFiles

DEFAULT_TIMEOUT_SECONDS = scoped_process.DEFAULT_TIMEOUT_SECONDS


class Dnr3Worker:
    """Run one DNR3 host process for one execution (SR, NR or SR+NR)."""

    def __init__(
        self,
        header: dnr3.Header,
        *,
        driver: HostDriver,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        shutdown_grace: float = scoped_process.TERMINATE_GRACE_SECONDS,
        interrupt: Callable[[], object] | None = None,
    ) -> None:
        if driver.files.features != header.features:
            raise dnr3.Dnr3ValidationError(
                f"driver was resolved for features {driver.files.features:#x} but the header "
                f"requests {header.features:#x}; resolve the runtime for the same features"
            )
        self._header = header
        self._driver = driver
        self._grace = float(shutdown_grace)
        self._process = scoped_process.ScopedProcess(
            driver.command,
            cwd=driver.cwd,
            env=driver.env,
            interrupt=interrupt,
            timeout=timeout,
            shutdown_grace=shutdown_grace,
        )
        self._frame_index = 0
        self._finishing = False
        self._finished = False

    # ------------------------------------------------------------------ public

    @property
    def header(self) -> dnr3.Header:
        return self._header

    @property
    def files(self) -> RuntimeFiles:
        """Runtime files this session was resolved and validated against."""
        return self._driver.files

    @property
    def frame_index(self) -> int:
        """Index of the next frame to send."""
        return self._frame_index

    @property
    def pid(self) -> int | None:
        return self._process.pid

    def stderr_text(self) -> str:
        """Bounded worker stderr tail, for error messages and progress logs."""
        return self._process.stderr_text()

    def __enter__(self) -> Dnr3Worker:
        self._process.__enter__()
        try:
            self._process.write(self._header.pack(), what="DNR3 header")
        except BaseException:
            self._process.terminate()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        forced = exc_type is not None
        try:
            if not forced:
                self.finish()
        except BaseException:
            forced = True
            raise
        finally:
            # Never leave the worker behind, whatever happened above.
            if forced:
                self._process.terminate()
            else:
                self._process.close()
        return False

    def enhance(
        self,
        rgb: np.ndarray,
        motion: np.ndarray | None = None,
        *,
        reset: bool = False,
        timeout: float | None = None,
    ) -> np.ndarray:
        """Send one frame and return the worker's processed frame.

        `rgb` is float32 `(input_height, input_width, 3)`; `motion` is float16
        or uint16 `(input_height, input_width, 2)` and may be None for static
        (zero) motion. The returned array is a read-only float32
        `(output_height, output_width, 3)` view over the received payload.
        """
        if self._finished or self._finishing:
            raise dnr3.Dnr3Error("worker session is finishing; no more frames accepted")
        index = self._frame_index
        if index >= self._header.frame_count:
            raise dnr3.Dnr3Error(
                f"frame_count {self._header.frame_count} is exhausted; "
                "declare every frame in the session header"
            )
        header = self._header
        payload = dnr3.rgb_payload(rgb, header.input_width, header.input_height)
        vectors = dnr3.motion_payload(motion, header.input_width, header.input_height)
        self._process.write(
            dnr3.pack_frame_header(index, reset), timeout=timeout, what=f"frame {index} header"
        )
        self._process.write(payload, timeout=timeout, what=f"frame {index} RGB")
        self._process.write(vectors, timeout=timeout, what=f"frame {index} motion")
        output = self._read_reply(index, timeout)
        self._frame_index = index + 1
        return output

    def finish(self) -> None:
        """Read the worker's END1 and reap it; idempotent."""
        if self._finished:
            return
        if self._frame_index != self._header.frame_count:
            raise dnr3.Dnr3ProtocolError(
                f"session declared {self._header.frame_count} frames but {self._frame_index} "
                "were sent; the worker cannot complete this session"
            )
        self._finishing = True
        # The host leaves its read loop after the last frame; closing stdin only
        # makes an early failure (wrong frame count) surface instead of hanging.
        self._process.close_stdin()
        marker = self._process.read_exactly(dnr3.END_SIZE, what="END1 marker")
        if marker != dnr3.END_MAGIC:
            raise dnr3.Dnr3ProtocolError(
                f"expected the END1 marker, got {marker!r}{self._detail()}"
            )
        self._finished = True
        if not self._process.wait_exit(self._grace):
            # Some NGX runtimes block in teardown after END1; every frame is
            # already delivered, so reap the group instead of hanging.
            return
        code = self._process.returncode
        if code != 0:
            raise dnr3.Dnr3ProtocolError(
                f"DLSS worker exited with code {code} after END1{self._detail()}"
            )

    # ----------------------------------------------------------------- internal

    def _read_reply(self, index: int, timeout: float | None) -> np.ndarray:
        reply_index, ok, float_count = dnr3.parse_reply_header(
            self._process.read_exactly(dnr3.REPLY_HEADER_SIZE, timeout=timeout, what="reply header")
        )
        if reply_index != index:
            raise dnr3.Dnr3ProtocolError(
                f"reply is for frame {reply_index} but frame {index} was sent{self._detail()}"
            )
        if not ok:
            raise dnr3.Dnr3RemoteError(
                f"DLSS worker failed on frame {index}: {self._read_error_text(timeout)}"
            )
        expected = self._header.output_floats
        if float_count != expected:
            raise dnr3.Dnr3ProtocolError(
                f"frame {index} reply declares {float_count} floats for "
                f"{self._header.output_width}x{self._header.output_height}, expected {expected}"
                f"{self._detail()}"
            )
        payload = self._process.read_exactly(
            expected * 4, timeout=timeout, what=f"frame {index} RGB output"
        )
        return dnr3.decode_rgb(payload, self._header.output_width, self._header.output_height)

    def _read_error_text(self, timeout: float | None) -> str:
        length = dnr3.check_error_length(
            self._process.read_exactly(dnr3.LENGTH.size, timeout=timeout, what="error length")
        )
        if not length:
            return "no message"
        text = self._process.read_exactly(length, timeout=timeout, what="error text")
        message = text.decode("utf-8", errors="replace").strip()
        return message or "no message"

    def _detail(self) -> str:
        stderr = self.stderr_text().strip()
        return f" (worker log: {stderr})" if stderr else ""
