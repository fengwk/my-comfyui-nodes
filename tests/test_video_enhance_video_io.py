"""Tests for the constant-memory FFmpeg/FFprobe I/O layer.

Most cases drive the module against generated fake `ffprobe`/`ffmpeg`
executables: they make the interesting situations deterministic - a VFR
timestamp stream, a decoder that ends early or late, an encoder that fails
after the first frame, a child that has to be killed - which no real clip can
produce reliably. The fake tools are argv driven and read/write the same pipes
the real ones do, so the process lifecycle under test is the production one.

`RealFfmpegTests` closes the loop with one tiny real clip: it proves the exact
frame counts, the CFR rejection and the audio remux against the binaries the
node actually calls.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import tracemalloc
import unittest
from collections.abc import Iterator
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from my_nodes.core.video_enhance.video_io import (
    FFmpegFrameReader,
    FFmpegFrameWriter,
    VideoIOError,
    VideoSpec,
    probe_cfr_video,
    remux_audio,
)

from .video_enhance_fixtures import assert_process_gone

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


# ------------------------------------------------------------------- fixtures


def write_tool(directory: Path, name: str, body: str) -> Path:
    """An executable fake tool that the module can spawn through PATH/argv."""
    path = directory / name
    path.write_text(f"#!{sys.executable}\n{textwrap.dedent(body).lstrip()}", encoding="utf-8")
    path.chmod(0o755)
    return path


FLOOD_BYTES = 256 * 1024
"""More stderr than a pipe holds (64 KiB), so it only fits while it is drained."""


def _stderr_flood(flood_bytes: int, flood_forever: bool) -> str:
    """Statements that close stdout and then overfill the stderr pipe.

    A pipe holds 64 KiB: a child that writes more than that before exiting only
    reaches its exit if the parent keeps draining stderr while it waits, which is
    exactly what the draining waits are for. The data is not meant to be kept,
    so each tool writes `FLOOD_BYTES` of it and the module may drop most.
    """
    if flood_bytes <= 0 and not flood_forever:
        return ""
    return f"""
        if {flood_bytes!r} or {flood_forever!r}:
            sys.stdout.flush()
            os.close(1)
            line = b"flood " + b"x" * 58
            written = 0
            while {flood_forever!r} or written < {flood_bytes!r}:
                try:
                    sys.stderr.buffer.write(line)
                    sys.stderr.buffer.flush()
                except OSError:
                    break
                written += len(line)
            try:
                sys.stderr.buffer.write(b"END-OF-FLOOD")
                sys.stderr.buffer.flush()
            except OSError:
                pass
    """


def _path_text(path: Path | None) -> str:
    """A path literal for the generated tools, or None when there is no path."""
    return "None" if path is None else repr(str(path))


def _pid_recorder(pid_file: Path | None) -> str:
    """Source of the pid probe the fake tools start with.

    Every line after the first is indented like the template line it continues,
    so `textwrap.dedent` can still strip the common prefix of the whole script.
    """
    statements = [
        f"pid_file = {_path_text(pid_file)}",
        "if pid_file:",
        '    with open(pid_file, "w", encoding="utf-8") as handle:',
        "        handle.write(str(os.getpid()))",
    ]
    return "\n        ".join(statements)


def video_stream(**overrides: Any) -> dict[str, Any]:
    """One 64x48 30 fps 8-bit SDR video stream, as FFprobe reports it."""
    stream: dict[str, Any] = {
        "index": 0,
        "codec_type": "video",
        "width": 64,
        "height": 48,
        "pix_fmt": "yuv420p",
        "bits_per_raw_sample": "8",
        "color_transfer": "unknown",
        "color_primaries": "unknown",
        "r_frame_rate": "30/1",
        "avg_frame_rate": "30/1",
        "sample_aspect_ratio": "1:1",
        "disposition": {"attached_pic": 0},
    }
    stream.update(overrides)
    return stream


IDENTITY_MATRIX = "\n00000000:        65536           0           0\n00000001:            0       65536           0\n00000002:            0           0  1073741824\n"
"""An identity display matrix, as FFprobe prints one."""

VFLIP_MATRIX = "\n00000000:        65536           0           0\n00000001:            0      -65536           0\n00000002:            0           0  1073741824\n"
"""A vertical flip: FFprobe reports `rotation 0` for it, FFmpeg still applies it."""


def display_matrix(rotation: float | str | None = None, matrix: str | None = None) -> list[dict[str, Any]]:
    """One side data entry as the FFprobe `stream_side_data` selector prints it."""
    entry: dict[str, Any] = {"side_data_type": "Display Matrix"}
    if rotation is not None:
        entry["rotation"] = rotation
    if matrix is not None:
        entry["displaymatrix"] = matrix
    return [entry]


def timed_thumbnail(**overrides: Any) -> dict[str, Any]:
    """A preview track: the `V` selector skips these the way it skips cover art."""
    stream = video_stream(
        width=16, height=16, pix_fmt="yuvj420p", disposition={"timed_thumbnails": 1}
    )
    stream.update(overrides)
    return stream


def attached_picture(**overrides: Any) -> dict[str, Any]:
    """Cover art: a video stream the `V:0` selection has to skip."""
    stream = video_stream(
        codec_name="mjpeg",
        width=32,
        height=32,
        pix_fmt="yuvj420p",
        bits_per_raw_sample=None,
        r_frame_rate="25/1",
        avg_frame_rate="25/1",
        disposition={"attached_pic": 1},
    )
    stream.update(overrides)
    return stream


def audio_stream(index: int = 1) -> dict[str, Any]:
    return {"index": index, "codec_type": "audio", "codec_name": "aac"}


def report(*streams: dict[str, Any]) -> dict[str, Any]:
    """The JSON envelope of one `-show_entries stream=...` answer."""
    return {"programs": [], "stream_groups": [], "streams": list(streams)}


def uniform_timestamps(count: int, fps: int = 30) -> list[float]:
    """`count` presentation timestamps of one exact frame rate."""
    return [index / fps for index in range(count)]


def quantized_timestamps(count: int, fps: int = 30, *, unit: int = 1000) -> list[float]:
    """The timestamps a millisecond container timebase reports for a CFR clip.

    Every timestamp is rounded on its own, so the values alternate between the
    two nearest timebase ticks (33 ms and 34 ms at 30 fps) while the cadence
    stays constant. The scan has to accept that quantization.
    """
    return [round(index * unit / fps) / unit for index in range(count)]


def frame_lines(timestamps: list[float], *, side_data: str | None = None) -> str:
    """One CSV frame line per decoded frame, as FFprobe's frame printer writes them.

    Frames that carry side data get it appended behind the timestamp in the same
    line, which is why the scan reads the first field only.
    """
    lines = [f"{value:.6f}" for value in timestamps]
    if side_data is not None and lines:
        lines[0] = f"{lines[0]},{side_data}"
    return "".join(f"{line}\n" for line in lines)


def fake_ffprobe(
    directory: Path,
    *,
    streams: dict[str, Any] | str | None = None,
    timestamps: list[float] | str = (),
    stderr: str = "",
    exit_code: int = 0,
    delay: float = 0.0,
    pid_file: Path | None = None,
    report_file: Path | None = None,
    stderr_flood: int = 0,
    stderr_flood_forever: bool = False,
) -> Path:
    """A fake ffprobe: stream JSON for the probe call, frame lines for the scan.

    With `report_file` every invocation appends its argv, which is how the tests
    prove which stream selection and which entries the module asks for.
    """
    streams_text = streams if isinstance(streams, str) else json.dumps(
        streams if streams is not None else report(video_stream())
    )
    if isinstance(timestamps, str):
        timestamps_text = timestamps
    else:
        timestamps_text = frame_lines(timestamps)
    body = f"""
        import json
        import os
        import sys
        import time

        {_pid_recorder(pid_file)}
        argv = sys.argv[1:]
        report_file = {_path_text(report_file)}
        if report_file:
            try:
                with open(report_file, encoding="utf-8") as handle:
                    commands = json.load(handle)
            except (FileNotFoundError, ValueError):
                commands = []
            commands.append(argv)
            with open(report_file, "w", encoding="utf-8") as handle:
                json.dump(commands, handle)
        kind = "stream" if any(part.startswith("stream=") for part in argv) else "frames"
        payload = {{"stream": {streams_text!r}, "frames": {timestamps_text!r}}}[kind]
        if {delay!r}:
            time.sleep({delay!r})
        sys.stdout.write(payload)
        sys.stdout.flush()
        if {stderr!r}:
            sys.stderr.write({stderr!r})
            sys.stderr.flush()
        {_stderr_flood(stderr_flood, stderr_flood_forever)}
        sys.exit({exit_code!r})
    """
    return write_tool(directory, "ffprobe", body)


def fake_ffmpeg_decode(
    directory: Path,
    *,
    frame_size: int,
    frames: int,
    stderr: str = "",
    exit_code: int = 0,
    delay: float = 0.0,
    pid_file: Path | None = None,
    report_file: Path | None = None,
    stderr_flood: int = 0,
) -> Path:
    """A fake ffmpeg that streams `frames` raw frames of constant bytes."""
    body = f"""
        import json
        import os
        import sys
        import time

        argv = sys.argv[1:]
        report_file = {_path_text(report_file)}
        if report_file:
            with open(report_file, "w", encoding="utf-8") as handle:
                json.dump([argv], handle)
        {_pid_recorder(pid_file)}
        for index in range({frames!r}):
            sys.stdout.buffer.write(bytes([index % 256]) * {frame_size!r})
            sys.stdout.buffer.flush()
            if {delay!r}:
                time.sleep({delay!r})
        if {stderr!r}:
            sys.stderr.write({stderr!r})
            sys.stderr.flush()
        {_stderr_flood(stderr_flood, False)}
        sys.exit({exit_code!r})
    """
    return write_tool(directory, "ffmpeg", body)


def fake_ffmpeg_encode(
    directory: Path,
    *,
    width: int,
    height: int,
    report_file: Path,
    pid_file: Path | None = None,
    exit_code: int = 0,
    stderr: str = "",
    output_bytes: int | None = None,
    skip_output: bool = False,
    frames_before_exit: int | None = None,
    delay: float = 0.0,
    stderr_flood: int = 0,
    stderr_flood_forever: bool = False,
) -> Path:
    """A fake ffmpeg that consumes raw RGB frames and writes its argv report.

    `frames_before_exit` stops reading mid-stream, the way a real encoder dies
    when its input cannot be encoded any further; `delay` keeps the encoder
    alive on purpose, so a cancel can be observed while it works.
    """
    body = f"""
        import json
        import os
        import sys
        import time

        argv = sys.argv[1:]
        output = argv[-1]
        frames_before_exit = {frames_before_exit!r}
        {_pid_recorder(pid_file)}
        size = {width!r} * {height!r} * 3
        payload = bytearray()
        frames = 0
        while True:
            chunk = sys.stdin.buffer.read(size)
            if not chunk:
                break
            payload.extend(chunk)
            frames += 1
            if {delay!r}:
                time.sleep({delay!r})
            if frames_before_exit is not None and frames >= frames_before_exit:
                break
        report_path = {str(report_file)!r}
        report_temp = report_path + ".tmp"
        with open(report_temp, "w", encoding="utf-8") as handle:
            json.dump({{"argv": argv, "frames": frames, "payload": payload.hex()}}, handle)
        os.replace(report_temp, report_path)
        if not {skip_output!r}:
            with open(output, "wb") as handle:
                handle.write(bytes(payload[:{output_bytes!r}]))
        if {stderr!r}:
            sys.stderr.write({stderr!r})
            sys.stderr.flush()
        {_stderr_flood(stderr_flood, stderr_flood_forever)}
        sys.exit({exit_code!r})
    """
    return write_tool(directory, "ffmpeg", body)


def fake_ffmpeg_remux(
    directory: Path,
    *,
    report_file: Path,
    pid_file: Path | None = None,
    exit_code: int = 0,
    stderr: str = "",
    skip_output: bool = False,
    delay: float = 0.0,
    stderr_flood: int = 0,
) -> Path:
    """A fake ffmpeg that records the remux command and writes an output file."""
    body = f"""
        import json
        import os
        import sys
        import time

        argv = sys.argv[1:]
        {_pid_recorder(pid_file)}
        if {delay!r}:
            time.sleep({delay!r})
        with open({str(report_file)!r}, "w", encoding="utf-8") as handle:
            json.dump({{"argv": argv}}, handle)
        if not {skip_output!r}:
            with open(argv[-1], "wb") as handle:
                handle.write(b"matroska-ish remux")
        if {stderr!r}:
            sys.stderr.write({stderr!r})
            sys.stderr.flush()
        {_stderr_flood(stderr_flood, False)}
        sys.exit({exit_code!r})
    """
    return write_tool(directory, "ffmpeg", body)


# --------------------------------------------------------------- base helpers


class VideoIOTestCase(unittest.TestCase):
    """One temporary workspace per test, removed whatever the test does."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="video_io_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()

    def spec(
        self,
        frame_count: int = 4,
        *,
        width: int = 4,
        height: int = 2,
        has_audio: bool = False,
        fps: Fraction = Fraction(30),
    ) -> VideoSpec:
        """A spec of a placeholder clip; the fake tools ignore its content."""
        path = self.work / "clip.mkv"
        if not path.exists():
            path.write_bytes(b"placeholder clip")
        return VideoSpec(
            path=path,
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            has_audio=has_audio,
            pixel_format="yuv420p",
        )

    def frame(self, value: float = 0.5) -> np.ndarray:
        """One 4x2 RGB frame of a single value."""
        return np.full((2, 4, 3), value, dtype=np.float32)

    def read_report(self, path: Path) -> dict[str, Any]:
        """The JSON report a fake tool wrote, once it wrote it."""
        if not path.is_file():
            return {}
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def read_commands(self, report_file: Path) -> list[list[str]]:
        """The argv of every invocation a fake tool recorded."""
        with open(report_file, encoding="utf-8") as handle:
            return json.load(handle)

    def await_pid(self, pid_file: Path, timeout: float = 5.0) -> int:
        """Wait until a fake tool recorded its pid, then return it."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return int(pid_file.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                time.sleep(0.01)
        raise AssertionError(f"the fake tool never recorded its pid in {pid_file}")

    def interrupt_when_running(self, pid_file: Path) -> Callable[[], bool]:
        """An interrupt callback that cancels as soon as the child is up."""

        def interrupt() -> bool:
            return pid_file.exists()

        return interrupt

    def assert_alive(self, pid: int) -> None:
        self.assertTrue(is_alive(pid), f"process {pid} died too early")

    def assert_file(self, path: Path) -> None:
        self.assertTrue(path.is_file(), f"{path} was not written")

    @contextlib.contextmanager
    def failing_setup(self) -> Iterator[list[subprocess.Popen[bytes]]]:
        """Spawn for real, then fail the setup step that follows Popen.

        `os.set_blocking` runs after the child exists, so this is exactly the
        window in which the owner alone is responsible for the child.
        """
        spawned: list[subprocess.Popen[bytes]] = []
        real_popen = subprocess.Popen

        def recording_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        with mock.patch("subprocess.Popen", recording_popen), mock.patch(
            "os.set_blocking", side_effect=KeyboardInterrupt("cancel during setup")
        ):
            yield spawned


def is_alive(pid: int) -> bool:
    """True while a process with that pid exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - different user
        return True
    return True


# ----------------------------------------------------------------- VideoSpec


class VideoSpecTests(unittest.TestCase):
    """The spec is the contract the node author passes around, so it is frozen."""

    def test_is_immutable(self) -> None:
        spec = VideoSpec(
            path=Path("/tmp/clip.mkv"),
            width=64,
            height=48,
            frame_count=30,
            fps=Fraction(30),
            has_audio=True,
            pixel_format="yuv420p",
        )
        with self.assertRaises(FrozenInstanceError):
            spec.frame_count = 31  # type: ignore[misc]

    def test_rejects_invalid_fields(self) -> None:
        base: dict[str, Any] = {
            "path": Path("/tmp/clip.mkv"),
            "width": 64,
            "height": 48,
            "frame_count": 30,
            "fps": Fraction(30),
            "has_audio": True,
            "pixel_format": "yuv420p",
        }
        for field, value, error in (
            ("path", "/tmp/clip.mkv", TypeError),
            ("width", 0, ValueError),
            ("height", -2, ValueError),
            ("frame_count", True, TypeError),
            ("fps", 30, TypeError),
            ("fps", Fraction(0), ValueError),
            ("has_audio", "yes", TypeError),
            ("pixel_format", "", ValueError),
            ("video_start_time", 0.0, TypeError),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(error):
                    VideoSpec(**{**base, field: value})


# --------------------------------------------------------------------- probe


class ProbeTests(VideoIOTestCase):
    def test_cfr_spec_comes_from_the_metadata_and_the_scan(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream(), audio_stream()),
            timestamps=uniform_timestamps(7),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        spec = probe_cfr_video(source, ffprobe_path=ffprobe)

        # The count is the number of scanned frame lines, the rest is metadata.
        self.assertEqual(spec, VideoSpec(
            path=source,
            width=64,
            height=48,
            frame_count=7,
            fps=Fraction(30),
            has_audio=True,
            pixel_format="yuv420p",
        ))

    def test_the_first_decoded_video_pts_is_kept_for_audio_remux(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream(), audio_stream()),
            timestamps=[1.25 + value for value in uniform_timestamps(4)],
        )
        source = self.work / "offset.mkv"
        source.write_bytes(b"clip")

        spec = probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertEqual(spec.video_start_time, Fraction(5, 4))

    def test_the_scan_decodes_the_selected_stream_instead_of_packets(self) -> None:
        report_file = self.root / "ffprobe.json"
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            timestamps=uniform_timestamps(3),
            report_file=report_file,
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        probe_cfr_video(source, ffprobe_path=ffprobe)

        metadata, scan = self.read_commands(report_file)
        # Both passes look at the same video stream, and it is the non-attached
        # one, so cover art can never be the stream that gets decoded.
        self.assertIn("stream_disposition=attached_pic", " ".join(metadata))
        self.assertNotIn("-select_streams", metadata)
        self.assertEqual(scan[scan.index("-select_streams") + 1], "V:0")
        # Decoded frames, one line each, not packet timestamps.
        self.assertIn("-show_frames", scan)
        self.assertIn("frame=best_effort_timestamp_time", scan)
        self.assertNotIn("packet=pts_time", " ".join(scan))
        self.assertEqual(scan[scan.index("-of") + 1], "csv=p=0")

    def test_the_count_is_the_number_of_decoded_frame_lines(self) -> None:
        # Frames that carry side data append it to their own line; the count is
        # still exactly one line per decoded frame.
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            timestamps=frame_lines(
                uniform_timestamps(5),
                side_data="H.26[45] User Data Unregistered SEI message",
            ),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        spec = probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertEqual(spec.frame_count, 5)

    def test_quantized_timestamps_are_accepted(self) -> None:
        # A millisecond timebase rounds every timestamp on its own, which must
        # not be mistaken for a variable cadence.
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            timestamps=quantized_timestamps(60),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        spec = probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertEqual(spec.frame_count, 60)

    def test_quantized_timestamps_of_a_long_clip_are_accepted(self) -> None:
        # Quantization must not accumulate over the length of a clip.
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            timestamps=quantized_timestamps(5000),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        self.assertEqual(probe_cfr_video(source, ffprobe_path=ffprobe).frame_count, 5000)

    def test_duplicated_and_out_of_order_timestamps_are_rejected(self) -> None:
        for timestamps in (
            [0.0, 1 / 30, 1 / 30, 2 / 30],  # a repeated timestamp
            [0.0, 2 / 30, 1 / 30, 3 / 30],  # a backwards step
        ):
            with self.subTest(timestamps=timestamps):
                ffprobe = fake_ffprobe(
                    self.bin, streams=report(video_stream()), timestamps=timestamps
                )
                source = self.work / "clip.mp4"
                source.write_bytes(b"clip")

                with self.assertRaises(VideoIOError) as caught:
                    probe_cfr_video(source, ffprobe_path=ffprobe)

                self.assertIn("not constant frame rate", str(caught.exception))

    def test_a_repeated_timestamp_inside_the_drift_bound_is_rejected(self) -> None:
        # At 1000 fps the 2 ms drift floor is wider than a whole interval, so
        # only strict monotonicity can reject the repeated timestamp.
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream(r_frame_rate="1000/1", avg_frame_rate="1000/1")),
            timestamps=[0.0, 0.001, 0.001, 0.002],
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        self.assertIn("not constant frame rate", message)
        self.assertIn("out of order", message)

    def test_a_dropped_interval_is_rejected(self) -> None:
        # A frame is missing: every later timestamp is a whole interval off the
        # grid, even though each gap is a legal multiple of the interval.
        timestamps = [0.0, 1 / 30, 2 / 30, 4 / 30, 5 / 30, 6 / 30]
        ffprobe = fake_ffprobe(
            self.bin, streams=report(video_stream()), timestamps=timestamps
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        self.assertIn("not constant frame rate", message)
        self.assertIn("ms off", message)

    def test_non_finite_timestamps_are_rejected(self) -> None:
        for text in ("inf\n0.033333\n", "nan\n0.033333\n"):
            with self.subTest(timestamp=text.strip()):
                ffprobe = fake_ffprobe(
                    self.bin, streams=report(video_stream()), timestamps=text
                )
                source = self.work / "clip.mp4"
                source.write_bytes(b"clip")

                with self.assertRaises(VideoIOError) as caught:
                    probe_cfr_video(source, ffprobe_path=ffprobe)

                self.assertIn("non-finite", str(caught.exception))

    def test_vfr_timestamps_are_rejected(self) -> None:
        mixed = uniform_timestamps(5) + uniform_timestamps(5, fps=60)
        ffprobe = fake_ffprobe(
            self.bin, streams=report(video_stream()), timestamps=mixed
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        # The error has to name the reason a user can act on.
        self.assertIn("variable frame rate", message.lower())
        self.assertIn("timestamps", message.lower())

    def test_timestamps_of_the_wrong_rate_are_rejected(self) -> None:
        ffprobe = fake_ffprobe(  # declared 30 fps, delivered as 24 fps
            self.bin,
            streams=report(video_stream()),
            timestamps=uniform_timestamps(6, fps=24),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("not constant frame rate", str(caught.exception))

    def test_missing_timestamps_are_rejected(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin, streams=report(video_stream()), timestamps="N/A\nN/A\n"
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("no presentation timestamp", str(caught.exception))

    def test_empty_clip_is_rejected(self) -> None:
        ffprobe = fake_ffprobe(self.bin, streams=report(video_stream()), timestamps=[])
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("no video frames", str(caught.exception))

    def test_missing_source_is_rejected(self) -> None:
        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(self.work / "nope.mp4", ffprobe_path="/nonexistent/ffprobe")

        self.assertIn("not a readable local file", str(caught.exception))

    def test_non_local_source_is_rejected(self) -> None:
        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video("rtsp://camera.local/stream", ffprobe_path="/nonexistent/ffprobe")

        self.assertIn("not a local file", str(caught.exception))

    def test_directory_source_is_rejected(self) -> None:
        with self.assertRaises(VideoIOError):
            probe_cfr_video(self.work, ffprobe_path="/nonexistent/ffprobe")

    def test_metadata_selects_the_first_non_attached_video_stream(self) -> None:
        # Cover art first, the real clip second: the spec has to describe the
        # clip, because that is the stream the scan and the reader select.
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(attached_picture(index=0), video_stream(index=1), audio_stream(2)),
            timestamps=uniform_timestamps(4),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        spec = probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertEqual((spec.width, spec.height), (64, 48))
        self.assertEqual(spec.pixel_format, "yuv420p")
        self.assertTrue(spec.has_audio)

    def test_cover_art_alone_is_not_a_video_stream(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(attached_picture(index=1), audio_stream(0)),
            timestamps=uniform_timestamps(3),
        )
        source = self.work / "song.mp3"
        source.write_bytes(b"song")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        self.assertIn("no video stream", message)
        self.assertIn("attached pictures", message)

    def test_timed_thumbnail_streams_are_skipped_like_the_V_selector(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(timed_thumbnail(index=0), video_stream(index=1), audio_stream(2)),
            timestamps=uniform_timestamps(4),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        spec = probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertEqual((spec.width, spec.height), (64, 48))
        self.assertEqual(spec.pixel_format, "yuv420p")

    def test_only_timed_thumbnails_is_not_a_video_stream(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(timed_thumbnail(index=0), audio_stream(1)),
            timestamps=uniform_timestamps(3),
        )
        source = self.work / "clip.mkv"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        self.assertIn("no video stream", message)
        self.assertIn("timed thumbnails", message)

    def test_clip_without_video_stream_is_rejected(self) -> None:
        ffprobe = fake_ffprobe(self.bin, streams=report(audio_stream()))
        source = self.work / "clip.m4a"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("no video stream", str(caught.exception))

    def test_invalid_dimensions_are_rejected(self) -> None:
        for overrides in ({"width": 0}, {"height": "N/A"}, {"width": None}):
            with self.subTest(overrides=overrides):
                ffprobe = fake_ffprobe(self.bin, streams=report(video_stream(**overrides)))
                source = self.work / "clip.mp4"
                source.write_bytes(b"clip")

                with self.assertRaises(VideoIOError) as caught:
                    probe_cfr_video(source, ffprobe_path=ffprobe)

                self.assertIn("invalid dimensions", str(caught.exception))

    def test_invalid_frame_rate_is_rejected(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream(r_frame_rate="0/0", avg_frame_rate="N/A")),
            timestamps=uniform_timestamps(3),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("no usable frame rate", str(caught.exception))

    def test_the_average_rate_is_preferred_over_the_representation_rate(self) -> None:
        # 29.97 fps reports r_frame_rate 60000/1001 (the LCM of the timestamp
        # grid) and avg_frame_rate 30000/1001 (the playback rate); the cadence is
        # the average one, so the timestamps are checked against it.
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(
                video_stream(r_frame_rate="60000/1001", avg_frame_rate="30000/1001")
            ),
            timestamps=uniform_timestamps(20, fps=Fraction(30000, 1001)),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        spec = probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertEqual(spec.fps, Fraction(30000, 1001))
        self.assertEqual(spec.frame_count, 20)

    def test_an_absurd_representation_rate_is_ignored(self) -> None:
        # Mixed 25 and 30 fps content reports r_frame_rate 150, which is even
        # representable; only the average rate describes the playback cadence.
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream(r_frame_rate="150/1", avg_frame_rate="25/1")),
            timestamps=uniform_timestamps(10, fps=25),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        self.assertEqual(probe_cfr_video(source, ffprobe_path=ffprobe).fps, Fraction(25))

    def test_representation_rate_is_the_fallback(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream(r_frame_rate="24/1", avg_frame_rate="N/A")),
            timestamps=uniform_timestamps(4, fps=24),
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        self.assertEqual(probe_cfr_video(source, ffprobe_path=ffprobe).fps, Fraction(24))

    def test_a_rotated_source_is_refused_before_any_frame_is_decoded(self) -> None:
        report_file = self.root / "ffprobe.json"
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream(side_data_list=display_matrix(rotation=90))),
            timestamps=uniform_timestamps(4),
            report_file=report_file,
        )
        source = self.work / "rotated.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        self.assertIn("90 degree display rotation", message)
        self.assertIn("Normalize the file first", message)
        # Only the metadata probe ran: a rotated clip is never frame-scanned or
        # decoded, because its frames would not match the coded dimensions.
        self.assertEqual(len(self.read_commands(report_file)), 1)

    def test_a_flipped_source_is_refused_even_though_its_rotation_is_zero(self) -> None:
        # FFmpeg applies a vertical flip while decoding while FFprobe reports
        # `rotation 0`, so the matrix itself has to decide.
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(
                video_stream(side_data_list=display_matrix(rotation=0, matrix=VFLIP_MATRIX))
            ),
            timestamps=uniform_timestamps(4),
        )
        source = self.work / "flipped.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        self.assertIn("display matrix", message)
        self.assertIn("Normalize the file first", message)

    def test_an_identity_display_matrix_is_accepted(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(
                video_stream(side_data_list=display_matrix(rotation=0, matrix=IDENTITY_MATRIX))
            ),
            timestamps=uniform_timestamps(4),
        )
        source = self.work / "plain.mp4"
        source.write_bytes(b"clip")

        self.assertEqual(probe_cfr_video(source, ffprobe_path=ffprobe).frame_count, 4)

    def test_an_unreadable_rotation_or_matrix_is_refused(self) -> None:
        for label, side_data, expected in (
            ("rotation", display_matrix(rotation="N/A"), "unreadable display rotation"),
            ("matrix", display_matrix(rotation=0, matrix="not a matrix"), "unreadable display matrix"),
            ("non-finite", display_matrix(rotation="inf"), "non-finite display rotation"),
        ):
            with self.subTest(field=label):
                ffprobe = fake_ffprobe(
                    self.bin,
                    streams=report(video_stream(side_data_list=side_data)),
                    timestamps=uniform_timestamps(4),
                )
                source = self.work / "odd.mp4"
                source.write_bytes(b"clip")

                with self.assertRaises(VideoIOError) as caught:
                    probe_cfr_video(source, ffprobe_path=ffprobe)

                self.assertIn(expected, str(caught.exception))

    def test_a_non_square_sample_aspect_ratio_is_refused(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream(sample_aspect_ratio="4:3")),
            timestamps=uniform_timestamps(4),
        )
        source = self.work / "anamorphic.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        self.assertIn("non-square pixels", message)
        self.assertIn("4:3", message)
        self.assertIn("setsar=1", message)

    def test_square_and_unspecified_sample_aspect_ratios_are_accepted(self) -> None:
        for ratio in ("1:1", "N/A", "0:1", "1/1"):
            with self.subTest(sample_aspect_ratio=ratio):
                ffprobe = fake_ffprobe(
                    self.bin,
                    streams=report(video_stream(sample_aspect_ratio=ratio)),
                    timestamps=uniform_timestamps(3),
                )
                source = self.work / "square.mp4"
                source.write_bytes(b"clip")

                self.assertEqual(
                    probe_cfr_video(source, ffprobe_path=ffprobe).frame_count, 3
                )

    def test_an_unreadable_sample_aspect_ratio_is_refused(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream(sample_aspect_ratio="four thirds")),
            timestamps=uniform_timestamps(3),
        )
        source = self.work / "odd.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("unreadable sample aspect ratio", str(caught.exception))

    def test_deep_and_float_pixel_formats_are_rejected(self) -> None:
        cases = (
            ({"pix_fmt": "yuv420p10le", "bits_per_raw_sample": "10"}, "10-bit"),
            ({"pix_fmt": "gbrp12le", "bits_per_raw_sample": None}, "12-bit"),
            ({"pix_fmt": "yuv420p", "bits_per_raw_sample": "10"}, "10-bit"),
            ({"pix_fmt": "yuv420p", "bits_per_raw_sample": "7"}, "7-bit"),
            ({"pix_fmt": "gbrpf32le", "bits_per_raw_sample": None}, "floating point"),
        )
        for overrides, expected in cases:
            with self.subTest(pixel_format=overrides["pix_fmt"]):
                ffprobe = fake_ffprobe(self.bin, streams=report(video_stream(**overrides)))
                source = self.work / "clip.mp4"
                source.write_bytes(b"clip")

                with self.assertRaises(VideoIOError) as caught:
                    probe_cfr_video(source, ffprobe_path=ffprobe)

                self.assertIn(expected, str(caught.exception))
                self.assertIn("8-bit SDR", str(caught.exception))

    def test_hdr_tags_are_rejected(self) -> None:
        for overrides in (
            {"color_transfer": "smpte2084"},
            {"color_transfer": "arib-std-b67"},
            {"color_primaries": "bt2020"},
        ):
            with self.subTest(overrides=overrides):
                ffprobe = fake_ffprobe(self.bin, streams=report(video_stream(**overrides)))
                source = self.work / "clip.mp4"
                source.write_bytes(b"clip")

                with self.assertRaises(VideoIOError) as caught:
                    probe_cfr_video(source, ffprobe_path=ffprobe)

                self.assertIn("not SDR", str(caught.exception))

    def test_missing_pixel_format_is_rejected(self) -> None:
        ffprobe = fake_ffprobe(self.bin, streams=report(video_stream(pix_fmt=None)))
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("no pixel format", str(caught.exception))

    def test_stream_report_is_bounded(self) -> None:
        # An answer of many megabytes must fail instead of being buffered whole.
        ffprobe = fake_ffprobe(self.bin, streams="{" + "x" * (2 << 20))
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("more than", str(caught.exception))

    def test_invalid_json_is_rejected(self) -> None:
        ffprobe = fake_ffprobe(self.bin, streams="not json at all")
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("valid JSON", str(caught.exception))

    def test_over_long_timestamp_line_is_rejected(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin, streams=report(video_stream()), timestamps="0.0" + "0" * 8192 + "\n"
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertIn("over-long line", str(caught.exception))

    def test_long_timestamp_scan_is_streamed(self) -> None:
        # Far more timestamps than a pipe buffer holds: the scan has to drain them.
        ffprobe = fake_ffprobe(
            self.bin, streams=report(video_stream()), timestamps=uniform_timestamps(20000)
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        self.assertEqual(probe_cfr_video(source, ffprobe_path=ffprobe).frame_count, 20000)

    def test_the_scan_holds_no_timestamps(self) -> None:
        # Holding every timestamp would cost megabytes here; the drift check
        # keeps two floats, so the scan stays in the tens of kilobytes whatever
        # the clip length is.
        ffprobe = fake_ffprobe(
            self.bin, streams=report(video_stream()), timestamps=uniform_timestamps(200_000)
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        tracemalloc.start()
        try:
            frame_count = probe_cfr_video(source, ffprobe_path=ffprobe).frame_count
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        self.assertEqual(frame_count, 200_000)
        self.assertLess(peak, 4 << 20, f"the timestamp scan peaked at {peak} bytes")

    def test_the_probe_finishes_when_ffprobe_floods_stderr_before_exiting(self) -> None:
        # FFprobe closes stdout and then writes 256 KiB - four pipe buffers - of
        # stderr before it exits. Only a wait that keeps draining gets to the
        # exit; a plain waitpid would sit there until the timeout.
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            timestamps=uniform_timestamps(3),
            stderr_flood=FLOOD_BYTES,
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        spec = probe_cfr_video(source, ffprobe_path=ffprobe)

        self.assertEqual(spec.frame_count, 3)

    def test_a_flooded_probe_failure_keeps_the_bounded_tail(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            timestamps=uniform_timestamps(3),
            stderr_flood=FLOOD_BYTES,
            exit_code=3,
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        self.assertIn("exit code 3", message)
        # The tail of the flood is what survives, and it stays bounded.
        self.assertIn("END-OF-FLOOD", message)
        self.assertLess(len(message), 4 * 8192)

    def test_probe_failure_reports_the_diagnostics(self) -> None:
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            stderr="clip.mkv: Invalid data found when processing input\n",
            exit_code=1,
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path=ffprobe)

        message = str(caught.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("Invalid data found", message)

    def test_missing_ffprobe_binary_is_actionable(self) -> None:
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(source, ffprobe_path="ffprobe-that-does-not-exist")

        message = str(caught.exception)
        self.assertIn("ffprobe was not found", message)
        self.assertIn("PATH", message)

    def test_interrupt_stops_the_probe_and_reaps_it(self) -> None:
        pid_file = self.root / "ffprobe.pid"
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            timestamps=uniform_timestamps(10),
            delay=30.0,
            pid_file=pid_file,
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(
                source, ffprobe_path=ffprobe, interrupt=self.interrupt_when_running(pid_file)
            )

        self.assertIn("interrupted", str(caught.exception))
        assert_process_gone(self.await_pid(pid_file))

    def test_a_cancel_during_the_probe_finish_is_prompt(self) -> None:
        # FFprobe answers and then floods stderr forever, so the wait for its exit
        # can only end through the interrupt callback.
        pid_file = self.root / "ffprobe.pid"
        fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            timestamps=uniform_timestamps(3),
            pid_file=pid_file,
            stderr_flood_forever=True,
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")
        started = time.monotonic()

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(
                source,
                ffprobe_path=self.bin / "ffprobe",
                interrupt=lambda: time.monotonic() > started + 0.3,
            )

        self.assertIn("interrupted", str(caught.exception))
        self.assertLess(time.monotonic() - started, 10.0)
        assert_process_gone(self.await_pid(pid_file))

    def test_base_exception_from_the_callback_propagates_and_reaps(self) -> None:
        pid_file = self.root / "ffprobe.pid"
        ffprobe = fake_ffprobe(
            self.bin,
            streams=report(video_stream()),
            timestamps=uniform_timestamps(10),
            delay=30.0,
            pid_file=pid_file,
        )
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        def interrupt() -> None:
            if pid_file.exists():
                raise KeyboardInterrupt("comfy interrupt")

        with self.assertRaises(KeyboardInterrupt):
            probe_cfr_video(source, ffprobe_path=ffprobe, interrupt=interrupt)

        assert_process_gone(self.await_pid(pid_file))


# -------------------------------------------------------------------- reader


class SetupFailureTests(VideoIOTestCase):
    """A spawn that fails after the child exists still has to reap it.

    `os.set_blocking` runs in every owner right after `Popen`, so failing it with
    a `BaseException` is exactly the window in which the child has nobody else
    looking after it.
    """

    def test_a_failed_probe_setup_leaves_no_ffprobe_behind(self) -> None:
        fake_ffprobe(self.bin, delay=30.0)
        source = self.work / "clip.mp4"
        source.write_bytes(b"clip")

        with self.failing_setup() as spawned:
            with self.assertRaises(KeyboardInterrupt):
                probe_cfr_video(source, ffprobe_path=self.bin / "ffprobe")

        self.assertEqual(len(spawned), 1)
        assert_process_gone(spawned[0].pid)

    def test_a_failed_reader_setup_leaves_no_decoder_behind(self) -> None:
        fake_ffmpeg_decode(self.bin, frame_size=4 * 2 * 3, frames=1, delay=30.0)

        with self.failing_setup() as spawned:
            with self.assertRaises(KeyboardInterrupt):
                FFmpegFrameReader(self.spec(1), ffmpeg_path=self.bin / "ffmpeg").__enter__()

        self.assertEqual(len(spawned), 1)
        assert_process_gone(spawned[0].pid)

    def test_a_failed_writer_setup_leaves_no_encoder_behind(self) -> None:
        fake_ffmpeg_encode(
            self.bin, width=4, height=2, report_file=self.root / "encode.json"
        )
        writer = FFmpegFrameWriter(
            self.work / "video_only.mkv",
            width=4,
            height=2,
            fps=Fraction(30),
            expected_frames=1,
            ffmpeg_path=self.bin / "ffmpeg",
        )

        with self.failing_setup() as spawned:
            with self.assertRaises(KeyboardInterrupt):
                writer.__enter__()

        self.assertEqual(len(spawned), 1)
        assert_process_gone(spawned[0].pid)


class ReaderTests(VideoIOTestCase):
    def test_frames_are_float32_rgb_in_unit_range(self) -> None:
        spec = self.spec(3, width=4, height=2)
        fake_ffmpeg_decode(
            self.bin, frame_size=4 * 2 * 3, frames=3, pid_file=self.root / "ffmpeg.pid"
        )

        with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
            frames = list(reader)

        self.assertEqual(len(frames), 3)
        for index, frame in enumerate(frames):
            self.assertEqual(frame.shape, (2, 4, 3))
            self.assertEqual(frame.dtype, np.float32)
            self.assertGreaterEqual(float(frame.min()), 0.0)
            self.assertLessEqual(float(frame.max()), 1.0)
            # Frame `index` is decoded from the byte value the fake wrote.
            self.assertAlmostEqual(float(frame[0, 0, 0]), index / 255.0, places=6)

    def test_it_maps_the_same_non_attached_video_stream_as_the_probe(self) -> None:
        report_file = self.root / "decode.json"
        spec = self.spec(1)
        fake_ffmpeg_decode(
            self.bin, frame_size=4 * 2 * 3, frames=1, report_file=report_file
        )

        with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
            list(reader)

        argv = self.read_commands(report_file)[0]
        # `0:V:0` is the probe's `V:0`: cover art is never the decoded stream.
        self.assertEqual(argv[argv.index("-map") + 1], "0:V:0")
        self.assertNotIn("0:v:0", argv)

    def test_the_decoder_streams_while_the_caller_consumes(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        spec = self.spec(4)
        fake_ffmpeg_decode(
            self.bin, frame_size=4 * 2 * 3, frames=4, delay=0.05, pid_file=pid_file
        )

        with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
            first = next(reader)
            pid = self.await_pid(pid_file)
            # One frame read, three still to come, and the decoder is still alive:
            # the reader streams frames instead of collecting the clip.
            self.assertEqual(reader.frames_read, 1)
            self.assertEqual(first.shape, (2, 4, 3))
            self.assert_alive(pid)
            self.assertEqual(len(list(reader)), 3)
            self.assertEqual(reader.frames_read, 4)

    def test_truncated_stream_is_an_error(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        spec = self.spec(4)
        fake_ffmpeg_decode(self.bin, frame_size=4 * 2 * 3, frames=2, pid_file=pid_file)

        with self.assertRaises(VideoIOError) as caught:
            with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
                list(reader)

        message = str(caught.exception)
        self.assertIn("ended after", message)
        self.assertIn("of 4 frames", message)
        assert_process_gone(self.await_pid(pid_file))

    def test_extra_frames_are_an_error(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        spec = self.spec(3)
        fake_ffmpeg_decode(self.bin, frame_size=4 * 2 * 3, frames=5, pid_file=pid_file)

        with self.assertRaises(VideoIOError) as caught:
            with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
                list(reader)

        self.assertIn("more than the 3 frames", str(caught.exception))
        assert_process_gone(self.await_pid(pid_file))

    def test_the_reader_finishes_when_the_decoder_floods_stderr(self) -> None:
        # The decoder closes stdout after its frames (so the reader sees EOF and
        # knows the clip ended) and then floods stderr before exiting.
        spec = self.spec(2)
        fake_ffmpeg_decode(
            self.bin, frame_size=4 * 2 * 3, frames=2, stderr_flood=FLOOD_BYTES
        )

        with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
            frames = list(reader)

        self.assertEqual(len(frames), 2)

    def test_a_flooded_decoder_failure_is_reported_with_its_tail(self) -> None:
        spec = self.spec(2)
        fake_ffmpeg_decode(
            self.bin,
            frame_size=4 * 2 * 3,
            frames=2,
            stderr_flood=FLOOD_BYTES,
            exit_code=1,
        )

        with self.assertRaises(VideoIOError) as caught:
            with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
                list(reader)

        message = str(caught.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("END-OF-FLOOD", message)
        self.assertLess(len(message), 4 * 8192)

    def test_nonzero_exit_reports_the_diagnostics(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        spec = self.spec(2)
        fake_ffmpeg_decode(
            self.bin,
            frame_size=4 * 2 * 3,
            frames=2,
            stderr="Error while decoding stream #0:0: Invalid data\n",
            exit_code=1,
            pid_file=pid_file,
        )

        with self.assertRaises(VideoIOError) as caught:
            with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
                list(reader)

        message = str(caught.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("Invalid data", message)
        assert_process_gone(self.await_pid(pid_file))

    def test_early_exit_reaps_the_decoder(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        spec = self.spec(50)
        fake_ffmpeg_decode(
            self.bin, frame_size=4 * 2 * 3, frames=50, delay=0.02, pid_file=pid_file
        )

        with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
            for index, _frame in enumerate(reader):
                if index == 1:
                    break
            pid = self.await_pid(pid_file)
            self.assert_alive(pid)

        assert_process_gone(pid)

    def test_base_exception_inside_the_block_reaps_the_decoder(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        spec = self.spec(50)
        fake_ffmpeg_decode(
            self.bin, frame_size=4 * 2 * 3, frames=50, delay=0.02, pid_file=pid_file
        )

        with self.assertRaises(KeyboardInterrupt):
            with FFmpegFrameReader(spec, ffmpeg_path=self.bin / "ffmpeg") as reader:
                next(reader)
                pid = self.await_pid(pid_file)
                raise KeyboardInterrupt("comfy interrupt")

        assert_process_gone(pid)

    def test_interrupt_callback_stops_the_decode(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        spec = self.spec(50)
        fake_ffmpeg_decode(
            self.bin, frame_size=4 * 2 * 3, frames=50, delay=0.05, pid_file=pid_file
        )

        with self.assertRaises(VideoIOError) as caught:
            with FFmpegFrameReader(
                spec,
                ffmpeg_path=self.bin / "ffmpeg",
                interrupt=self.interrupt_when_running(pid_file),
            ) as reader:
                list(reader)

        self.assertIn("interrupted", str(caught.exception))
        assert_process_gone(self.await_pid(pid_file))

    def test_missing_ffmpeg_binary_is_actionable(self) -> None:
        spec = self.spec(1)

        with self.assertRaises(VideoIOError) as caught:
            with FFmpegFrameReader(spec, ffmpeg_path="ffmpeg-that-does-not-exist"):
                pass

        self.assertIn("ffmpeg was not found", str(caught.exception))

    def test_reading_needs_a_context_manager(self) -> None:
        reader = FFmpegFrameReader(self.spec(1), ffmpeg_path=self.bin / "ffmpeg")

        with self.assertRaises(VideoIOError) as caught:
            next(reader)

        self.assertIn("context manager", str(caught.exception))

    def test_reader_is_single_use(self) -> None:
        fake_ffmpeg_decode(self.bin, frame_size=4 * 2 * 3, frames=1)
        reader = FFmpegFrameReader(self.spec(1), ffmpeg_path=self.bin / "ffmpeg")

        with reader:
            with self.assertRaises(VideoIOError) as caught:
                reader.__enter__()

        self.assertIn("single use", str(caught.exception))


# -------------------------------------------------------------------- writer


class WriterTests(VideoIOTestCase):
    def writer(
        self,
        expected_frames: int = 2,
        *,
        name: str = "video_only.mkv",
        width: int = 4,
        height: int = 2,
        **kwargs: Any,
    ) -> FFmpegFrameWriter:
        """A writer over the fake ffmpeg in this test's bin directory.

        Every case can override one contract argument, so the tests read like the
        API they exercise.
        """
        self.encode_report = kwargs.pop("report_file", self.root / "encode.json")
        return FFmpegFrameWriter(
            self.work / name,
            width=kwargs.pop("width", width),
            height=kwargs.pop("height", height),
            fps=kwargs.pop("fps", Fraction(30)),
            expected_frames=kwargs.pop("expected_frames", expected_frames),
            ffmpeg_path=self.bin / "ffmpeg",
            **kwargs,
        )

    def test_it_encodes_the_exact_frames_it_was_given(self) -> None:
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            pid_file=self.root / "ffmpeg.pid",
        )
        writer = self.writer(2)
        frames = [self.frame(0.0), self.frame(0.5)]

        with writer as sink:
            for frame in frames:
                sink.write(frame)

        report = self.read_report(self.encode_report)
        self.assertEqual(report["frames"], 2)
        self.assertEqual(bytes.fromhex(report["payload"]), b"\x00" * 24 + b"\x80" * 24)
        self.assert_file(writer.path)

    def test_it_maps_the_encode_options_a_streaming_node_needs(self) -> None:
        fake_ffmpeg_encode(
            self.bin, width=4, height=2, report_file=self.root / "encode.json"
        )
        with self.writer(1) as sink:
            sink.write(self.frame(0.5))

        argv = self.read_report(self.encode_report)["argv"]
        for expected in (
            "-f", "rawvideo", "rgb24", "-s", "4x2", "-r", "30",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-f", "matroska",
        ):
            self.assertIn(expected, argv)
        # The output path is the last argument: argv, never a shell string.
        self.assertEqual(argv[-1], str(self.work / "video_only.mkv"))

    def test_a_fractional_frame_rate_stays_exact(self) -> None:
        # 29.97 fps has to reach FFmpeg as 30000/1001, not as a rounded float.
        fake_ffmpeg_encode(self.bin, width=4, height=2, report_file=self.root / "encode.json")

        with self.writer(1, fps=Fraction(30000, 1001)) as sink:
            sink.write(self.frame(0.5))

        argv = self.read_report(self.encode_report)["argv"]
        self.assertEqual(argv[argv.index("-r") + 1], "30000/1001")

    def test_clipping_and_rounding_of_the_frames(self) -> None:
        fake_ffmpeg_encode(self.bin, width=4, height=2, report_file=self.root / "encode.json")
        frame = np.zeros((2, 4, 3), dtype=np.float32)
        frame[0, 0] = (0.0, 0.25, 1.0)
        frame[0, 1] = (-1.0, 2.0, 0.5)

        with self.writer(1) as sink:
            sink.write(frame)

        payload = bytes.fromhex(self.read_report(self.encode_report)["payload"])
        self.assertEqual(payload[:6], bytes([0, 64, 255, 0, 255, 128]))

    def test_nvenc_uses_cq_rate_control(self) -> None:
        fake_ffmpeg_encode(self.bin, width=4, height=2, report_file=self.root / "encode.json")

        with self.writer(1, codec="h264_nvenc", quality=23) as sink:
            sink.write(self.frame(0.5))

        argv = self.read_report(self.encode_report)["argv"]
        for expected in ("h264_nvenc", "-rc", "vbr", "-cq", "23", "-b:v", "0"):
            self.assertIn(expected, argv)
        self.assertNotIn("-crf", argv)

    def test_too_few_frames_is_an_error_and_leaves_nothing_behind(self) -> None:
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            pid_file=self.root / "ffmpeg.pid",
        )
        writer = self.writer(3)

        with self.assertRaises(VideoIOError) as caught:
            with writer as sink:
                self.await_pid(self.root / "ffmpeg.pid")
                sink.write(self.frame(0.5))

        self.assertIn("needs 3 frames but only 1", str(caught.exception))
        self.assertFalse(writer.path.exists())
        assert_process_gone(self.await_pid(self.root / "ffmpeg.pid"))

    def test_too_many_frames_is_an_error_and_leaves_nothing_behind(self) -> None:
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            pid_file=self.root / "ffmpeg.pid",
        )
        writer = self.writer(2)

        with self.assertRaises(VideoIOError) as caught:
            with writer as sink:
                sink.write(self.frame(0.5))
                self.await_pid(self.root / "ffmpeg.pid")
                for _ in range(2):
                    sink.write(self.frame(0.5))

        self.assertIn("called for frame 3", str(caught.exception))
        self.assertFalse(writer.path.exists())
        assert_process_gone(self.await_pid(self.root / "ffmpeg.pid"))

    def test_invalid_frames_are_refused_before_anything_is_encoded(self) -> None:
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            pid_file=self.root / "ffmpeg.pid",
        )
        cases = {
            "shape": np.zeros((2, 3, 3), dtype=np.float32),
            "channels": np.zeros((2, 4, 4), dtype=np.float32),
            "non-finite": np.full((2, 4, 3), np.nan, dtype=np.float32),
            "text": np.full((2, 4, 3), "x"),
        }
        for name, frame in cases.items():
            with self.subTest(case=name):
                # The prior subtest's pid file must not satisfy await_pid for
                # this process before it has started reading stdin.
                (self.root / "ffmpeg.pid").unlink(missing_ok=True)
                writer = self.writer(1)
                with self.assertRaises(VideoIOError):
                    with writer as sink:
                        pid = self.await_pid(self.root / "ffmpeg.pid")
                        sink.write(frame)

                self.assertFalse(writer.path.exists())
                assert_process_gone(pid)
                self.assertEqual(self.read_report(self.encode_report), {})

        assert_process_gone(self.await_pid(self.root / "ffmpeg.pid"))

    def test_unsupported_codec_quality_and_dimensions_are_refused(self) -> None:
        for kwargs, expected in (
            ({"codec": "libx265"}, "unsupported video codec"),
            ({"codec": "mpeg4"}, "unsupported video codec"),
            ({"codec": "libx264", "quality": 52}, "0..51"),
            ({"codec": "libx264", "quality": -1}, "0..51"),
            ({"codec": "libx264", "quality": "18"}, "must be an int"),
            ({"codec": "libx264", "width": 5}, "even dimensions"),
            ({"codec": "libx264", "height": 3}, "even dimensions"),
            ({"codec": "libx264", "expected_frames": 0}, "positive int"),
            ({"codec": "libx264", "fps": Fraction(0)}, "positive"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(VideoIOError) as caught:
                    self.writer(**kwargs)

                self.assertIn(expected, str(caught.exception))

    def test_unknown_output_container_is_refused(self) -> None:
        with self.assertRaises(VideoIOError) as caught:
            self.writer(1, name="video_only.webm")

        message = str(caught.exception)
        self.assertIn("container", message)
        self.assertIn(".mkv", message)

    def test_the_encode_finalization_waits_through_a_stderr_flood(self) -> None:
        # After its input pipe closes, this encoder reports 256 KiB on stderr and
        # only then exits (and only then does its output file exist).
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            stderr_flood=FLOOD_BYTES,
        )
        writer = self.writer(2)

        with writer as sink:
            sink.write(self.frame(0.0))
            sink.write(self.frame(0.5))

        self.assert_file(writer.path)
        self.assertEqual(self.read_report(self.encode_report)["frames"], 2)

    def test_a_flooded_encoder_failure_reports_its_tail(self) -> None:
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            stderr_flood=FLOOD_BYTES,
            exit_code=1,
        )
        writer = self.writer(2)

        with self.assertRaises(VideoIOError) as caught:
            with writer as sink:
                sink.write(self.frame(0.0))
                sink.write(self.frame(0.5))

        message = str(caught.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("END-OF-FLOOD", message)
        self.assertLess(len(message), 4 * 8192)
        self.assertFalse(writer.path.exists())

    def test_a_cancel_during_a_stderr_flood_is_prompt(self) -> None:
        # The encoder floods stderr forever: it never exits, so only the
        # interrupt callback can end the finalization wait - promptly, and with
        # the partial output removed.
        pid_file = self.root / "ffmpeg.pid"
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            pid_file=pid_file,
            stderr_flood_forever=True,
        )
        polls = {"count": 0}

        def interrupt() -> bool:
            # Cancelling while the encoder is being written to would abort the
            # write instead of the flood; the third poll is inside the wait.
            polls["count"] += 1
            return polls["count"] >= 3

        writer = self.writer(1, interrupt=interrupt)
        started = time.monotonic()

        with self.assertRaises(VideoIOError) as caught:
            with writer as sink:
                sink.write(self.frame(0.5))

        self.assertIn("cancel", str(caught.exception).lower())
        self.assertLess(time.monotonic() - started, 10.0)
        self.assertFalse(writer.path.exists())
        assert_process_gone(self.await_pid(pid_file))

    def test_encoder_failure_reports_the_diagnostics_and_cleans_up(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            pid_file=pid_file,
            stderr="[libx264 @ 0x1] width not divisible by 2\n",
            exit_code=1,
        )
        writer = self.writer(2)

        with self.assertRaises(VideoIOError) as caught:
            with writer as sink:
                sink.write(self.frame(0.5))
                sink.write(self.frame(0.5))

        message = str(caught.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("width not divisible by 2", message)
        self.assertFalse(writer.path.exists())
        assert_process_gone(self.await_pid(pid_file))

    def test_encoder_that_dies_mid_stream_is_reported(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        fake_ffmpeg_encode(
            self.bin,
            width=512,
            height=512,
            report_file=self.root / "encode.json",
            pid_file=pid_file,
            frames_before_exit=1,
            stderr="[libx264 @ 0x1] Error while opening encoder\n",
            exit_code=1,
        )
        # Frames larger than the pipe buffer: the writer cannot hide the failure
        # inside the pipe, it has to see the dead encoder.
        frame = np.zeros((512, 512, 3), dtype=np.float32)

        with self.assertRaises(VideoIOError) as caught:
            with self.writer(4, width=512, height=512) as sink:
                for _ in range(4):
                    sink.write(frame)

        self.assertIn("stopped accepting frames", str(caught.exception))
        assert_process_gone(self.await_pid(pid_file))

    def test_success_without_bytes_is_an_error(self) -> None:
        fake_ffmpeg_encode(
            self.bin, width=4, height=2, report_file=self.root / "encode.json", skip_output=True
        )
        writer = self.writer(1)

        with self.assertRaises(VideoIOError) as caught:
            with writer as sink:
                sink.write(self.frame(0.5))

        self.assertIn("wrote no video", str(caught.exception))
        self.assertFalse(writer.path.exists())

    def test_empty_output_is_an_error(self) -> None:
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            output_bytes=0,
        )
        writer = self.writer(1)

        with self.assertRaises(VideoIOError) as caught:
            with writer as sink:
                sink.write(self.frame(0.5))

        self.assertIn("wrote no video", str(caught.exception))
        self.assertFalse(writer.path.exists())

    def test_missing_ffmpeg_binary_is_actionable(self) -> None:
        with self.assertRaises(VideoIOError) as caught:
            with FFmpegFrameWriter(
                self.work / "video_only.mkv",
                width=4,
                height=2,
                fps=Fraction(30),
                expected_frames=1,
                ffmpeg_path="ffmpeg-that-does-not-exist",
            ):
                pass

        self.assertIn("ffmpeg was not found", str(caught.exception))

    def test_writing_needs_a_context_manager(self) -> None:
        writer = self.writer(1)

        with self.assertRaises(VideoIOError) as caught:
            writer.write(self.frame(0.5))

        self.assertIn("context manager", str(caught.exception))

    def test_interrupt_cancels_the_encode_and_removes_the_output(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            pid_file=pid_file,
            delay=5.0,
        )
        writer = self.writer(3, interrupt=self.interrupt_when_running(pid_file))

        with self.assertRaises(VideoIOError) as caught:
            with writer as sink:
                for _ in range(3):
                    sink.write(self.frame(0.5))

        self.assertIn("interrupted", str(caught.exception))
        self.assertFalse(writer.path.exists())
        assert_process_gone(self.await_pid(pid_file))

    def test_base_exception_cancels_the_encode_and_removes_the_output(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        fake_ffmpeg_encode(
            self.bin,
            width=4,
            height=2,
            report_file=self.root / "encode.json",
            pid_file=pid_file,
        )
        writer = self.writer(3)

        with self.assertRaises(KeyboardInterrupt):
            with writer as sink:
                sink.write(self.frame(0.5))
                pid = self.await_pid(pid_file)
                raise KeyboardInterrupt("comfy interrupt")

        self.assertFalse(writer.path.exists())
        assert_process_gone(pid)


# --------------------------------------------------------------------- remux


class RemuxTests(VideoIOTestCase):
    def video_only(self, name: str = "video_only.mkv", payload: bytes = b"encoded video") -> Path:
        path = self.work / name
        path.write_bytes(payload)
        return path

    def test_a_source_without_audio_is_moved_without_ffmpeg(self) -> None:
        video_only = self.video_only()
        spec = self.spec(2, has_audio=False)

        output = remux_audio(
            spec,
            video_only,
            self.work / "final.mkv",
            ffmpeg_path="/nonexistent/ffmpeg",  # proves no FFmpeg is started
        )

        self.assertEqual(output, self.work / "final.mkv")
        self.assertEqual(output.read_bytes(), b"encoded video")
        self.assertFalse(video_only.exists())

    def test_a_cancelled_move_without_audio_publishes_nothing(self) -> None:
        video_only = self.video_only()
        spec = self.spec(2, has_audio=False)
        output = self.work / "final.mkv"

        with self.assertRaises(VideoIOError) as caught:
            remux_audio(spec, video_only, output, interrupt=lambda: True)

        self.assertIn("interrupted", str(caught.exception))
        self.assertFalse(output.exists())
        # The caller's encoded video is still its own file.
        self.assertEqual(video_only.read_bytes(), b"encoded video")

    def test_a_base_exception_cancel_without_audio_publishes_nothing(self) -> None:
        class Cancel(BaseException):
            """Stands in for a ComfyUI interrupt."""

        def interrupt() -> bool:
            raise Cancel("cancelled by the user")

        video_only = self.video_only()
        spec = self.spec(2, has_audio=False)
        output = self.work / "final.mkv"

        with self.assertRaises(Cancel):
            remux_audio(spec, video_only, output, interrupt=interrupt)

        self.assertFalse(output.exists())
        self.assertEqual(video_only.read_bytes(), b"encoded video")

    def test_audio_is_copied_with_the_encoded_video(self) -> None:
        report_file = self.root / "remux.json"
        fake_ffmpeg_remux(self.bin, report_file=report_file)
        video_only = self.video_only()
        spec = self.spec(2, has_audio=True)

        output = remux_audio(
            spec, video_only, self.work / "final.mkv", ffmpeg_path=self.bin / "ffmpeg"
        )

        argv = self.read_report(report_file)["argv"]
        # Both streams are mapped and copied, and nothing may shorten the clip.
        self.assertEqual(
            argv[argv.index("-map") + 1 : argv.index("-map") + 5],
            ["0:v:0", "-map", "1:a:0", "-c"],
        )
        self.assertIn("copy", argv)
        self.assertIn(str(spec.path), argv)
        self.assertNotIn("-shortest", argv)
        self.assertIn("matroska", argv)
        self.assert_file(output)
        # The caller's video-only file is left alone on success.
        self.assert_file(video_only)
        self.assertNotEqual(video_only, output)

    def test_audio_is_shifted_by_the_video_pts_without_per_input_normalization(self) -> None:
        report_file = self.root / "remux.json"
        fake_ffmpeg_remux(self.bin, report_file=report_file)
        video_only = self.video_only()
        base = self.spec(2, has_audio=True)
        spec = VideoSpec(
            path=base.path,
            width=base.width,
            height=base.height,
            frame_count=base.frame_count,
            fps=base.fps,
            has_audio=base.has_audio,
            pixel_format=base.pixel_format,
            video_start_time=Fraction(5, 4),
        )

        remux_audio(
            spec, video_only, self.work / "final.mkv", ffmpeg_path=self.bin / "ffmpeg"
        )

        argv = self.read_report(report_file)["argv"]
        self.assertLess(argv.index("-copyts"), argv.index("-i"))
        source_input = argv.index(str(spec.path))
        self.assertEqual(
            argv[source_input - 3 : source_input], ["-itsoffset", "-1.25", "-i"]
        )
        self.assertEqual(
            argv[argv.index("-avoid_negative_ts") : argv.index("-avoid_negative_ts") + 2],
            ["-avoid_negative_ts", "make_zero"],
        )

    def test_the_remux_waits_through_a_stderr_flood(self) -> None:
        fake_ffmpeg_remux(
            self.bin, report_file=self.root / "remux.json", stderr_flood=FLOOD_BYTES
        )
        video_only = self.video_only()
        spec = self.spec(2, has_audio=True)

        output = remux_audio(
            spec, video_only, self.work / "final.mkv", ffmpeg_path=self.bin / "ffmpeg"
        )

        self.assert_file(output)

    def test_a_flooded_remux_failure_reports_its_tail(self) -> None:
        fake_ffmpeg_remux(
            self.bin,
            report_file=self.root / "remux.json",
            stderr_flood=FLOOD_BYTES,
            exit_code=1,
        )
        video_only = self.video_only()
        spec = self.spec(2, has_audio=True)

        with self.assertRaises(VideoIOError) as caught:
            remux_audio(
                spec, video_only, self.work / "final.mkv", ffmpeg_path=self.bin / "ffmpeg"
            )

        message = str(caught.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("END-OF-FLOOD", message)
        self.assertLess(len(message), 4 * 8192)

    def test_a_failed_remux_keeps_the_video_only_file(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        fake_ffmpeg_remux(
            self.bin,
            report_file=self.root / "remux.json",
            pid_file=pid_file,
            stderr="Could not find tag for codec pcm_s24le\n",
            exit_code=1,
        )
        video_only = self.video_only()
        spec = self.spec(2, has_audio=True)

        with self.assertRaises(VideoIOError) as caught:
            remux_audio(
                spec, video_only, self.work / "final.mkv", ffmpeg_path=self.bin / "ffmpeg"
            )

        message = str(caught.exception)
        self.assertIn("exit code 1", message)
        self.assertIn("Could not find tag", message)
        self.assertFalse((self.work / "final.mkv").exists())
        self.assert_file(video_only)
        assert_process_gone(self.await_pid(pid_file))

    def test_interrupt_keeps_the_video_only_file(self) -> None:
        pid_file = self.root / "ffmpeg.pid"
        fake_ffmpeg_remux(
            self.bin,
            report_file=self.root / "remux.json",
            pid_file=pid_file,
            delay=30.0,
        )
        video_only = self.video_only()
        spec = self.spec(2, has_audio=True)

        with self.assertRaises(VideoIOError) as caught:
            remux_audio(
                spec,
                video_only,
                self.work / "final.mkv",
                ffmpeg_path=self.bin / "ffmpeg",
                interrupt=self.interrupt_when_running(pid_file),
            )

        self.assertIn("interrupted", str(caught.exception))
        self.assertFalse((self.work / "final.mkv").exists())
        self.assert_file(video_only)

    def test_missing_inputs_are_refused(self) -> None:
        spec = self.spec(2, has_audio=True)
        with self.assertRaises(VideoIOError) as caught:
            remux_audio(spec, self.work / "missing.mkv", self.work / "final.mkv")

        self.assertIn("does not exist", str(caught.exception))

    def test_a_missing_audio_source_is_refused(self) -> None:
        video_only = self.video_only()
        spec = self.spec(2, has_audio=True)
        spec.path.unlink()

        with self.assertRaises(VideoIOError) as caught:
            remux_audio(spec, video_only, self.work / "final.mkv")

        self.assertIn("audio source", str(caught.exception))
        self.assert_file(video_only)

    def test_unknown_output_container_is_refused(self) -> None:
        video_only = self.video_only()
        spec = self.spec(2, has_audio=True)

        with self.assertRaises(VideoIOError) as caught:
            remux_audio(spec, video_only, self.work / "final.webm")

        self.assertIn("container", str(caught.exception))
        self.assert_file(video_only)

    def test_a_remux_without_output_is_an_error(self) -> None:
        fake_ffmpeg_remux(
            self.bin, report_file=self.root / "remux.json", skip_output=True
        )
        video_only = self.video_only()
        spec = self.spec(2, has_audio=True)

        with self.assertRaises(VideoIOError) as caught:
            remux_audio(
                spec, video_only, self.work / "final.mkv", ffmpeg_path=self.bin / "ffmpeg"
            )

        self.assertIn("wrote no video", str(caught.exception))
        self.assertFalse((self.work / "final.mkv").exists())
        self.assert_file(video_only)

    def test_a_non_spec_source_is_refused(self) -> None:
        with self.assertRaises(TypeError):
            remux_audio(  # type: ignore[arg-type]
                "source.mkv", self.video_only(), self.work / "final.mkv"
            )


# ------------------------------------------------------- real FFmpeg round trip


@unittest.skipUnless(FFMPEG and FFPROBE, "requires the ffmpeg and ffprobe binaries")
class RealFfmpegTests(unittest.TestCase):
    """One tiny real clip through probe, decode, encode and remux.

    The fake tools above prove the lifecycle; these tests prove the exact frame
    counts, the CFR rejection and the audio remux against the real binaries.
    """

    WIDTH = 32
    HEIGHT = 24
    FPS = 8
    FRAMES = 8

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="video_io_real_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.build_source()

    def build_source(self, *, with_audio: bool = True, name: str = "source.mp4") -> Path:
        """A `FRAMES` frame CFR clip at `FPS`, with an audio tone by default."""
        duration = self.FRAMES / self.FPS
        command = [
            FFMPEG,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={self.WIDTH}x{self.HEIGHT}:rate={self.FPS}:duration={duration}",
        ]
        if with_audio:
            command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if with_audio:
            command += ["-c:a", "aac"]
        else:
            command += ["-an"]
        command.append(str(self.root / name))
        subprocess.run(command, check=True)
        return self.root / name

    def ffmpeg(self, *arguments: str) -> None:
        subprocess.run([FFMPEG, "-y", "-v", "error", *arguments], check=True)

    def ffprobe(self, *arguments: str) -> str:
        """The csv report of one ffprobe question about a real file."""
        result = subprocess.run(
            [FFPROBE, "-v", "error", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def first_packet_pts(self, path: Path, selector: str) -> Fraction:
        """The first packet PTS of one stream in a tiny real fixture."""
        report = self.ffprobe(
            "-select_streams",
            selector,
            "-show_packets",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            str(path),
        )
        return Fraction(report.splitlines()[0])

    def test_probe_is_exact_and_deterministic(self) -> None:
        spec = probe_cfr_video(self.source)
        second = probe_cfr_video(self.source)

        self.assertEqual(spec, VideoSpec(
            path=self.source,
            width=self.WIDTH,
            height=self.HEIGHT,
            frame_count=self.FRAMES,
            fps=Fraction(self.FPS),
            has_audio=True,
            pixel_format="yuv420p",
        ))
        self.assertEqual(spec, second)

    def test_the_probed_count_is_the_decoded_frame_count(self) -> None:
        spec = probe_cfr_video(self.source)
        decided = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "V:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1",
             str(self.source)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        self.assertEqual(int(decided), spec.frame_count)

    def test_a_cover_art_container_is_not_a_clip(self) -> None:
        cover = self.root / "cover.png"
        song = self.root / "song.mp3"
        self.ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=1:duration=1",
            "-frames:v", "1", str(cover),
        )
        self.ffmpeg(
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-i", str(cover), "-map", "0:a", "-map", "1:v",
            "-c:a", "libmp3lame", "-c:v", "mjpeg",
            "-disposition:v:0", "attached_pic", "-id3v2_version", "3",
            "-f", "mp3", str(song),
        )

        # The only video stream is the cover: `v:0` selects it, `V:0` (what the
        # module asks for) selects nothing, and so does the decoded frame scan.
        self.assertEqual(self.ffprobe("-select_streams", "v:0", "-show_entries",
                                      "stream=index", "-of", "csv=p=0", str(song)), "1")
        self.assertEqual(self.ffprobe("-select_streams", "V:0", "-show_entries",
                                      "stream=index", "-of", "csv=p=0", str(song)), "")
        self.assertEqual(self.ffprobe("-select_streams", "V:0", "-show_frames",
                                      "-show_entries", "frame=best_effort_timestamp_time",
                                      "-of", "csv=p=0", str(song)), "")

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(song)

        message = str(caught.exception)
        self.assertIn("no video stream", message)
        self.assertIn("attached pictures", message)

    def test_a_clip_with_cover_art_still_decodes_the_clip(self) -> None:
        cover = self.root / "cover.png"
        tagged = self.root / "tagged.mkv".replace(".mkv", ".mp4")
        self.ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=1:duration=1",
            "-frames:v", "1", str(cover),
        )
        self.ffmpeg(
            "-i", str(cover), "-i", str(self.source),
            "-map", "0:v", "-map", "1:v", "-map", "1:a",
            "-c:v:0", "mjpeg", "-c:v:1", "copy", "-c:a", "copy",
            "-disposition:v:0", "attached_pic", str(tagged),
        )

        spec = probe_cfr_video(tagged)

        # The metadata pass picks the same stream as the `V:0` selector (the MP4
        # muxer writes the attached picture last, so the fake test above covers
        # the cover-first ordering), and the cover art is 32x32 and one frame
        # where the clip is neither.
        self.assertEqual(
            self.ffprobe("-select_streams", "V:0", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0", str(tagged)),
            f"{self.WIDTH},{self.HEIGHT}",
        )
        self.assertEqual((spec.width, spec.height), (self.WIDTH, self.HEIGHT))
        self.assertEqual(spec.frame_count, self.FRAMES)
        self.assertTrue(spec.has_audio)
        with FFmpegFrameReader(spec) as reader:
            self.assertEqual(len(list(reader)), self.FRAMES)

    def test_reader_yields_the_probed_frames_one_at_a_time(self) -> None:
        spec = probe_cfr_video(self.source)
        frames: list[np.ndarray] = []
        with FFmpegFrameReader(spec) as reader:
            for frame in reader:
                self.assertEqual(frame.shape, (self.HEIGHT, self.WIDTH, 3))
                self.assertEqual(frame.dtype, np.float32)
                self.assertGreaterEqual(float(frame.min()), 0.0)
                self.assertLessEqual(float(frame.max()), 1.0)
                frames.append(frame)
            self.assertEqual(reader.frames_read, self.FRAMES)

        self.assertEqual(len(frames), self.FRAMES)
        # A real clip, not a constant colour: the decode actually happened.
        self.assertGreater(float(frames[0].std()), 0.01)

    def test_round_trip_keeps_frames_rate_and_audio(self) -> None:
        spec = probe_cfr_video(self.source)
        with FFmpegFrameReader(spec) as reader:
            frames = list(reader)

        video_only = self.root / "video_only.mkv"
        with FFmpegFrameWriter(
            video_only,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            expected_frames=spec.frame_count,
        ) as writer:
            for frame in frames:
                writer.write(frame)

        final = remux_audio(spec, video_only, self.root / "final.mkv")

        result = probe_cfr_video(final)
        self.assertEqual(result.frame_count, spec.frame_count)
        self.assertEqual(result.width, spec.width)
        self.assertEqual(result.height, spec.height)
        self.assertEqual(result.fps, spec.fps)
        self.assertTrue(result.has_audio)

        with FFmpegFrameReader(result) as reader:
            decoded = list(reader)
        self.assertEqual(len(decoded), self.FRAMES)
        for source_frame, decoded_frame in zip(frames, decoded):
            # Lossy, but the same clip: a frame mix-up would be far larger.
            self.assertLess(float(np.abs(source_frame - decoded_frame).mean()), 0.05)

    def test_round_trip_preserves_audio_that_starts_before_video(self) -> None:
        source = self.root / "offset_source.mkv"
        duration = self.FRAMES / self.FPS
        self.ffmpeg(
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={self.WIDTH}x{self.HEIGHT}:rate={self.FPS}:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration + 0.5}",
            "-vf",
            "setpts=PTS+1/TB",
            "-af",
            "asetpts=PTS+0.5/TB",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            str(source),
        )
        spec = probe_cfr_video(source)
        self.assertEqual(spec.video_start_time, Fraction(1))
        source_offset = (
            self.first_packet_pts(source, "a:0") - self.first_packet_pts(source, "v:0")
        )
        self.assertEqual(source_offset, Fraction(-1, 2))

        video_only = self.root / "offset_video_only.mkv"
        with FFmpegFrameReader(spec) as reader:
            with FFmpegFrameWriter(
                video_only,
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
                expected_frames=spec.frame_count,
            ) as writer:
                for frame in reader:
                    writer.write(frame)

        final = remux_audio(spec, video_only, self.root / "offset_final.mkv")
        final_offset = (
            self.first_packet_pts(final, "a:0") - self.first_packet_pts(final, "v:0")
        )

        self.assertEqual(final_offset, source_offset)

    def test_a_source_without_audio_is_just_moved(self) -> None:
        silent = self.build_source(with_audio=False, name="silent.mp4")
        spec = probe_cfr_video(silent)
        self.assertFalse(spec.has_audio)

        with FFmpegFrameReader(spec) as reader:
            frames = list(reader)
        video_only = self.root / "silent_only.mkv"
        with FFmpegFrameWriter(
            video_only,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            expected_frames=spec.frame_count,
        ) as writer:
            for frame in frames:
                writer.write(frame)

        output = remux_audio(spec, video_only, self.root / "silent_final.mkv")

        self.assertTrue(output.is_file())
        self.assertFalse(video_only.exists())
        self.assertFalse(probe_cfr_video(output).has_audio)

    def test_a_rotated_or_flipped_source_is_rejected(self) -> None:
        # 90 degrees keeps the pixel count (so the reader would reshape
        # transposed frames), a horizontal flip is reported as a rotation and a
        # vertical flip only as a non-identity matrix; FFmpeg applies all three
        # while decoding, so all three are refused.
        for name, flag in (
            ("rotated.mp4", "-display_rotation"),
            ("hflipped.mp4", "-display_hflip"),
            ("vflipped.mp4", "-display_vflip"),
        ):
            with self.subTest(transform=name):
                derived = self.root / name
                arguments = (
                    ["-display_rotation", "90"] if flag == "-display_rotation" else [flag]
                )
                self.ffmpeg(*arguments, "-i", str(self.source), "-c", "copy", str(derived))
                self.assertTrue(self.ffprobe("-show_entries", "stream_side_data=rotation",
                                             "-of", "default=nw=1:nk=1", str(derived)) != "")

                with self.assertRaises(VideoIOError) as caught:
                    probe_cfr_video(derived)

                message = str(caught.exception)
                self.assertIn("Normalize the file first", message)
                self.assertTrue(
                    "display rotation" in message or "display matrix" in message, message
                )

    def test_a_non_square_pixel_source_is_rejected(self) -> None:
        derived = self.root / "anamorphic.mp4"
        self.ffmpeg(
            "-i", str(self.source), "-vf", "setsar=4/3", "-c:v", "libx264", "-an", str(derived)
        )

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(derived)

        message = str(caught.exception)
        self.assertIn("non-square pixels", message)
        self.assertIn("setsar=1", message)

    def test_a_vfr_source_is_rejected(self) -> None:
        slow = self.root / "slow.mkv"
        fast = self.root / "fast.mkv"
        for rate, path in ((self.FPS, slow), (self.FPS * 3, fast)):
            self.ffmpeg(
                "-f", "lavfi", "-i",
                f"testsrc2=size={self.WIDTH}x{self.HEIGHT}:rate={rate}:duration=0.5",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
            )
        listing = self.root / "concat.txt"
        listing.write_text(f"file '{slow}'\nfile '{fast}'\n", encoding="utf-8")
        vfr = self.root / "vfr.mkv"
        self.ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(vfr))

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(vfr)

        message = str(caught.exception)
        self.assertIn("variable frame rate", message.lower())
        self.assertIn("timestamps", message.lower())

    def test_a_ten_bit_source_is_rejected(self) -> None:
        ten_bit = self.root / "ten_bit.mp4"
        self.ffmpeg(
            "-f", "lavfi", "-i",
            f"testsrc2=size={self.WIDTH}x{self.HEIGHT}:rate={self.FPS}:duration=0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p10le", str(ten_bit),
        )

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(ten_bit)

        self.assertIn("10-bit", str(caught.exception))

    def test_an_hdr_tagged_source_is_rejected(self) -> None:
        hdr = self.root / "hdr.mkv"
        self.ffmpeg(
            "-f", "lavfi", "-i",
            f"testsrc2=size={self.WIDTH}x{self.HEIGHT}:rate={self.FPS}:duration=0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-x264-params", "colorprim=bt2020:transfer=smpte2084", str(hdr),
        )

        with self.assertRaises(VideoIOError) as caught:
            probe_cfr_video(hdr)

        self.assertIn("not SDR", str(caught.exception))
