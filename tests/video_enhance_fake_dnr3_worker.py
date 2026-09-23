"""Protocol-speaking fake DNR3 worker used by the video-enhance backend tests.

It speaks the real DNR3 protocol over stdin/stdout with no GPU, no DLSS runtime
and no Wine: it parses the header with the production parser
(`my_nodes.core.video_enhance.dnr3`), reads each frame payload with the exact
byte counts the protocol declares, and answers with the `OUT1` reply layout the
production host writes.

The first argument selects the failure behaviour; every run also appends what
it saw to a JSON report file (`FAKE_DNR3_REPORT`) so tests can assert the exact
bytes the parent put on the wire - including the session feature flags - and the
launch environment the parent wrote explicitly.

Modes
-----
ok                answer every frame with a deterministic nearest-neighbour
                  enlargement (or an echo when the sizes match), then END1
hang              accept frames and never answer (parent must time out)
stubborn          like `hang` but ignores SIGTERM (parent must SIGKILL)
stubborn-child    like `stubborn` and leaves a grandchild in the same group
crash             write a line to stderr and die before answering frame 0
error             answer with a worker error reply, then stay alive
bad-magic         answer with a complete reply carrying the wrong magic
bad-index         answer for the wrong frame index
bad-count         answer declaring one float too many
short-reply       answer with a truncated reply header
no-end            answer every frame but never send END1
exit-after-end    send END1 and then exit non-zero
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402 - after sys.path bootstrap

from my_nodes.core.video_enhance import dnr3  # noqa: E402

MODES: tuple[str, ...] = (
    "ok",
    "hang",
    "stubborn",
    "stubborn-child",
    "crash",
    "error",
    "bad-magic",
    "bad-index",
    "bad-count",
    "short-reply",
    "no-end",
    "exit-after-end",
)
REPORT_ENV = "FAKE_DNR3_REPORT"
ERROR_MESSAGE = "fake DNR3 worker: neural-rendering runtime unavailable"
# Launch-time controls the stage writes explicitly; recorded so a test can prove
# what the real child process received, including that a stale parent value was
# replaced.
WATCHED_ENV: tuple[str, ...] = (
    "DLSS5NR_UI_CORRECTION",
    "DLSS5NR_DETAIL",
    "DLSS5NR_COLOR",
    "DLSS5NR_SR_PRESET",
    "DLSS5NR_GPU_INDEX",
    "DLSS5NR_CHANNEL_ORDER",
)


class Report:
    """Append-only JSON report of what the worker received."""

    def __init__(self, path: str | None) -> None:
        self._path = path
        self._data: dict[str, object] = {"frames": []}

    def set(self, key: str, value: object) -> None:
        self._data[key] = value
        self._flush()

    def append_frame(self, entry: dict[str, object]) -> None:
        frames = self._data["frames"]
        assert isinstance(frames, list)
        frames.append(entry)
        self._flush()

    def _flush(self) -> None:
        if not self._path:
            return
        target = Path(self._path)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self._data, handle, sort_keys=True)
            handle.flush()
        os.replace(temporary, target)


def _nearest(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Deterministic enlargement: nearest neighbour, same for every frame."""
    in_h, in_w = rgb.shape[0], rgb.shape[1]
    rows = (np.arange(height) * in_h) // height
    cols = (np.arange(width) * in_w) // width
    return rgb[np.ix_(rows, cols)]


def _read_frame(header: dnr3.Header, report: Report, index: int) -> tuple[np.ndarray, bytes]:
    raw = dnr3.read_exact(sys.stdin.buffer, dnr3.FRAME_HEADER_SIZE, "frame header")
    frame_index, reset = dnr3.parse_frame_header(raw)
    if frame_index != index:
        raise dnr3.Dnr3ProtocolError(f"expected frame {index}, got {frame_index}")
    rgb_bytes = dnr3.read_exact(sys.stdin.buffer, header.input_floats * 4, "input RGB")
    motion = dnr3.read_exact(sys.stdin.buffer, header.motion_words * 2, "motion")
    rgb = np.frombuffer(rgb_bytes, dtype="<f4").reshape(
        (header.input_height, header.input_width, 3)
    )
    report.append_frame(
        {
            "index": frame_index,
            "reset": reset,
            "rgb_first": [float(value) for value in rgb[0, 0]],
            "rgb_max": float(rgb.max()),
            "motion_nonzero": int(np.count_nonzero(np.frombuffer(motion, dtype="<u2"))),
            "motion_bytes": len(motion),
        }
    )
    return rgb, motion


def _serve(header: dnr3.Header, mode: str, report: Report) -> int:
    out = sys.stdout.buffer
    for index in range(header.frame_count):
        rgb, _motion = _read_frame(header, report, index)
        if mode == "hang" or mode == "stubborn" or mode == "stubborn-child":
            time.sleep(60.0)
            return 0
        if mode == "crash":
            print("fake DNR3 worker aborted", file=sys.stderr, flush=True)
            os._exit(3)
        output = _nearest(
            np.ascontiguousarray(rgb, dtype=np.float32),
            header.output_height,
            header.output_width,
        )
        payload = output.tobytes()
        if mode == "error":
            print(f"fake DNR3 worker: {ERROR_MESSAGE}", file=sys.stderr, flush=True)
            out.write(dnr3.pack_error_reply(index, ERROR_MESSAGE))
            out.flush()
            time.sleep(60.0)  # stay alive: the parent must abort and clean up
            return 0
        if mode == "bad-magic":
            out.write(dnr3.REPLY_HEADER.pack(b"XXXX", index, 1, header.output_floats))
            out.write(payload)
            out.flush()
            return 0
        if mode == "bad-index":
            out.write(dnr3.pack_reply_header(index + 1, header.output_floats))
            out.write(payload)
            out.flush()
            return 0
        if mode == "bad-count":
            out.write(dnr3.pack_reply_header(index, header.output_floats + 1))
            out.write(payload)
            out.flush()
            return 0
        if mode == "short-reply":
            out.write(dnr3.pack_reply_header(index, header.output_floats)[:8])
            out.flush()
            return 0
        out.write(dnr3.pack_reply_header(index, header.output_floats))
        out.write(payload)
        out.flush()
    if mode == "no-end":
        report.set("ended", False)
        return 0
    out.write(dnr3.END_MAGIC)
    out.flush()
    report.set("ended", True)
    if mode == "exit-after-end":
        return 3
    return 0


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else "ok"
    if mode not in MODES:
        return 5
    report = Report(os.environ.get(REPORT_ENV) or None)
    report.set("mode", mode)
    report.set("pid", os.getpid())
    report.set("pgrp", os.getpgrp())
    report.set("sid", os.getsid(0))
    report.set("env", {name: os.environ.get(name) for name in WATCHED_ENV})
    if mode == "stubborn" or mode == "stubborn-child":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "stubborn-child":
        # Same process group: only a group signal can reach this grandchild.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        report.set("grandchild", child.pid)

    raw = dnr3.read_exact(sys.stdin.buffer, dnr3.HEADER_SIZE, "DNR3 header")
    header = dnr3.Header.parse(raw)
    report.set(
        "header",
        {
            "input_width": header.input_width,
            "input_height": header.input_height,
            "output_width": header.output_width,
            "output_height": header.output_height,
            "warmup_frames": header.warmup_frames,
            "frame_count": header.frame_count,
            "perf_quality": header.perf_quality,
            "features": header.features,
            "preset": header.preset,
            "style": header.style,
            "automask": header.automask,
            "ui_correction": header.ui_correction,
            "intensity": header.intensity,
            "tone": header.tone,
            "structure": header.structure,
            "skin": header.skin,
            "global_tone": header.global_tone,
        },
    )
    return _serve(header, mode, report)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except dnr3.Dnr3Error as exc:  # malformed request: fail loudly, not silently
        print(f"fake DNR3 worker rejected the request: {exc}", file=sys.stderr, flush=True)
        sys.exit(6)
