"""Constant-memory FFmpeg/FFprobe I/O for the native VIDEO streaming node.

The streaming node never holds a clip: it describes one file, walks its frames
through FFmpeg pipes and hands them to the enhancement stages one at a time.
This module is the only place where that node touches an external tool.

* `probe_cfr_video` describes a local constant frame rate 8-bit SDR file. The
  stream probe is one bounded JSON answer; the exact frame count and the CFR
  proof come from a second streaming scan of the *decoded* frames' presentation
  timestamps: every timestamp must sit on the ideal `first + index / fps` grid
  within an explicit bound and the sequence must be strictly increasing, which
  rejects variable frame rate input that the reported frame rates hide. The
  average frame rate is preferred over `r_frame_rate`, which is a representation
  rate rather than the playback rate (mixed 25 and 30 fps content reports 150).
  Both passes select `V:0`, the first video stream that is neither an attached
  picture nor a timed thumbnail, and the reader maps exactly that stream, so
  cover art is never mistaken for a clip. A display transform - a rotation, or
  a non-identity matrix such as a flip, which FFprobe reports with `rotation 0`
  even though FFmpeg applies it - and a non-square sample aspect ratio are both
  refused: the decode path returns the *stored* pixels, and neither can be
  applied without reshaping, mirroring or rescaling them.
* `FFmpegFrameReader` decodes raw RGB frames from one FFmpeg pipe and yields
  exactly the probed frame count, one float32 `[H,W,3]` frame in [0,1] at a
  time.
* `FFmpegFrameWriter` encodes those frames straight back into an FFmpeg pipe,
  finalizes only after exactly the expected frame count and drops a partial
  output on every failure, including a cancel.
* `remux_audio` stream-copies the encoded video together with the first source
  audio track, or just moves the video-only file when the source has no audio.

Every external command is an argv sequence with `shell=False`, runs in its own
process group and is reaped on success, on an exception, on a `BaseException`
cancel and when the caller's `interrupt` callback asks for it. The reader and
the writer exchange fixed size binary payloads, which is what `ScopedProcess`
already owns; the probe and the timestamp scan need bounded line oriented text
output, so they use the minimal read only owner below. No daemon, no watchdog
and no background service is involved.

Every wait for a child to finish is a draining wait
(`ScopedProcess.wait_exit_draining`): FFmpeg reports a failed encode or remux on
stderr while it runs, and a child that fills that pipe before exiting would
wedge a plain `waitpid` forever. The waits therefore keep reading both pipes
(stdout is discarded, stderr stays in the bounded tail), poll the interrupt
callback and end on their deadline; a spawn whose fd setup fails after the child
exists is killed and reaped by its owner.
"""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from my_nodes.core.video_enhance.scoped_process import (
    ProcessError,
    ProcessInterrupted,
    ProcessIOError,
    ScopedProcess,
)

# --- external tool budgets -------------------------------------------------

FRAME_TIMEOUT_SECONDS = 120.0
"""Longest wait for one decoded or encoded frame."""

ENCODE_TIMEOUT_SECONDS = 600.0
"""Longest wait for FFmpeg to finish after its input pipe was closed."""

REMUX_TIMEOUT_SECONDS = 1800.0
"""Longest wait for a stream copy of the whole clip."""

PROBE_TIMEOUT_SECONDS = 600.0
"""Longest silence tolerated while FFprobe answers or finishes."""

POLL_INTERVAL_SECONDS = 0.05
TERMINATE_GRACE_SECONDS = 2.0
STDERR_SETTLE_SECONDS = 0.5
STDERR_LIMIT_BYTES = 8 * 1024
STREAM_PROBE_LIMIT_BYTES = 1 << 20
READ_CHUNK_BYTES = 1 << 16
SCAN_LINE_LIMIT_BYTES = 4096

# --- probe rules -----------------------------------------------------------

CADENCE_DRIFT_FLOOR_SECONDS = 0.002
"""Floor of the cadence bound: a millisecond container timebase rounds that far.

One millisecond timebases round each timestamp by at most half a unit and the
rounding does not accumulate, because every timestamp is rounded on its own.
"""

CADENCE_DRIFT_RATIO = 0.1
"""Relative cadence bound: a longer or shorter interval misses the grid by more.

A duplicated, dropped or otherwise off-cadence interval shifts every later
timestamp by a whole nominal interval, which this bound always rejects; only
quantization noise stays inside it.
"""

HDR_TRANSFERS = frozenset({"smpte2084", "smpte428", "arib-std-b67"})

# --- encode rules ----------------------------------------------------------

SUPPORTED_CODECS = ("libx264", "h264_nvenc")
OUTPUT_PIXEL_FORMAT = "yuv420p"
QUALITY_MIN = 0
QUALITY_MAX = 51
FRAME_CHANNELS = 3
CONTAINER_FORMATS = {".mkv": "matroska", ".mp4": "mp4", ".m4v": "mp4", ".mov": "mov"}

_STREAM_FIELDS = (
    "stream=index,codec_type,width,height,pix_fmt,bits_per_raw_sample,"
    "color_transfer,color_primaries,r_frame_rate,avg_frame_rate,sample_aspect_ratio"
    ":stream_disposition=attached_pic,timed_thumbnails"
    ":stream_side_data=rotation,displaymatrix"
)
_SKIPPED_VIDEO_DISPOSITIONS = ("attached_pic", "timed_thumbnails")
"""The video streams FFmpeg's `V` selector leaves out."""

_IDENTITY_DISPLAY_MATRIX = (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824)
"""FFprobe's identity display matrix: nine 16.16/2.30 fixed point values."""

_MATRIX_ROW_PREFIX = re.compile(r"^[0-9a-fA-F]+:", re.MULTILINE)
_FRAME_TIMESTAMP_FIELDS = "frame=best_effort_timestamp_time"
_VIDEO_STREAM_SELECTOR = "V:0"
"""FFprobe selection of the first video stream that is not an attached picture."""
_VIDEO_STREAM_MAP = "0:V:0"
"""FFmpeg mapping of that same stream, for the decoder and the encoder."""
_URL_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_FLOAT_PIXEL_FORMAT_PATTERN = re.compile(r"f(?:16|32|64)(?:le|be)?$")
_DEEP_PIXEL_FORMAT_PATTERN = re.compile(
    r"(?:gbrp|gbrap|gray|ya|xyz|rgb|p)(9|10|12|14|16)(?:le|be)?$"
)


class VideoIOError(RuntimeError):
    """The video file, the FFmpeg tools or the frame exchange cannot be used."""


@dataclass(frozen=True)
class VideoSpec:
    """Immutable description of one local constant frame rate video file.

    `path` is the probed file, `width`/`height` its coded size, `frame_count`
    the exact number of frames `FFmpegFrameReader` yields, `fps` its nominal
    frame rate, `has_audio` whether it carries an audio stream and
    `pixel_format` the source pixel format as FFprobe reported it.
    """

    path: Path
    width: int
    height: int
    frame_count: int
    fps: Fraction
    has_audio: bool
    pixel_format: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError(f"path must be a pathlib.Path, got {type(self.path).__name__}")
        for name in ("width", "height", "frame_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if not isinstance(self.fps, Fraction):
            raise TypeError(f"fps must be a fractions.Fraction, got {type(self.fps).__name__}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not isinstance(self.has_audio, bool):
            raise TypeError(f"has_audio must be a bool, got {type(self.has_audio).__name__}")
        if not isinstance(self.pixel_format, str) or not self.pixel_format:
            raise ValueError(f"pixel_format must be a non-empty string, got {self.pixel_format!r}")


# --------------------------------------------------------------------- probe


def probe_cfr_video(
    source: str | os.PathLike[str],
    *,
    ffprobe_path: str | os.PathLike[str] = "ffprobe",
    interrupt: Callable[[], object] | None = None,
) -> VideoSpec:
    """Describe one local constant frame rate 8-bit SDR video file.

    A missing or non-local source, a file without a video stream, invalid
    dimensions or frame rate, a clip without frames, a source that is not 8-bit
    SDR, a rotated or non-square-pixel source and a variable frame rate file all
    fail here, before any decoding starts. The container's own metadata and one
    streaming pass over the decoded frames' timestamps are read; no frame is
    buffered, so the probe is O(1) in memory whatever the length of the clip.
    """
    path = _local_source(source)
    ffprobe = _resolve_tool(ffprobe_path, "ffprobe")
    _check_interrupt(interrupt)
    streams = _probe_streams(path, ffprobe, interrupt)
    stream = _video_stream(streams, path)
    width, height = _dimensions(stream, path)
    _checked_geometry(stream, path)
    fps = _nominal_fps(stream, path)
    pixel_format = _checked_pixel_format(stream, path)
    frame_count = _scan_frame_timestamps(path, ffprobe, fps, interrupt)
    return VideoSpec(
        path=path,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        has_audio=any(entry.get("codec_type") == "audio" for entry in streams),
        pixel_format=pixel_format,
    )


def _local_source(source: str | os.PathLike[str]) -> Path:
    """The local file behind `source`; URLs, directories and pipes are refused."""
    try:
        text = os.fspath(source)
    except TypeError as error:
        raise VideoIOError(
            f"a video source must be a path, got {type(source).__name__}"
        ) from error
    if isinstance(text, bytes):
        text = os.fsdecode(text)
    if _URL_PATTERN.match(text):
        raise VideoIOError(
            f"{text} is not a local file; streaming enhancement needs a local seekable "
            "video file, and URLs, pipes and live streams cannot be probed"
        )
    path = Path(text)
    if not path.is_file():
        raise VideoIOError(
            f"the video source {path} is not a readable local file; streaming enhancement "
            "needs a local seekable video file that already exists"
        )
    return path


def _resolve_tool(executable: str | os.PathLike[str], name: str) -> str:
    """Absolute path of one external tool, or an actionable error."""
    resolved = shutil.which(os.fspath(executable))
    if resolved is None:
        raise VideoIOError(
            f"{name} was not found: {os.fspath(executable)!r}. Install FFmpeg so that "
            f"{name} is on PATH, or pass the path of the binary explicitly."
        )
    return resolved


def _probe_streams(
    path: Path, ffprobe: str, interrupt: Callable[[], object] | None
) -> tuple[dict[str, Any], ...]:
    """Stream level metadata of `path`, as a bounded FFprobe JSON answer."""
    command = (
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        _STREAM_FIELDS,
        "-of",
        "json",
        str(path),
    )
    with _ProbeProcess(command, interrupt=interrupt) as probe:
        raw = probe.read_all(STREAM_PROBE_LIMIT_BYTES, what="the stream information")
        probe.finish(what=f"probing {path}")
    try:
        report = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as error:
        raise VideoIOError(f"ffprobe did not answer with valid JSON for {path}: {error}") from error
    streams = report.get("streams") if isinstance(report, dict) else None
    if not isinstance(streams, list) or not streams:
        raise VideoIOError(f"ffprobe found no streams in {path}")
    return tuple(entry for entry in streams if isinstance(entry, dict))


def _video_stream(streams: tuple[dict[str, Any], ...], path: Path) -> dict[str, Any]:
    """The first video stream the `V` selector picks.

    FFprobe's `V:0`, the timestamp scan and the reader's `-map 0:V:0` all mean
    this stream: the first video stream whose disposition is neither
    `attached_pic` (cover art) nor `timed_thumbnails` (the preview track of e.g.
    some MKV files). Filtering the same dispositions here is what keeps the
    metadata, the frame count and the decoded frames describing one stream.
    """
    for stream in streams:
        if stream.get("codec_type") == "video" and not _is_skipped_video(stream):
            return stream
    if any(stream.get("codec_type") == "video" for stream in streams):
        raise VideoIOError(
            f"{path} has no video stream, only attached pictures (cover art) or timed "
            "thumbnails"
        )
    raise VideoIOError(f"{path} has no video stream")


def _is_skipped_video(stream: dict[str, Any]) -> bool:
    """True for a video stream the `V` selector leaves out."""
    disposition = stream.get("disposition")
    if not isinstance(disposition, dict):
        return False
    return any(bool(disposition.get(name)) for name in _SKIPPED_VIDEO_DISPOSITIONS)


def _checked_geometry(stream: dict[str, Any], path: Path) -> None:
    """Refuse input whose stored pixels are not the pixels a frame walker sees.

    The reader reshapes raw decoded frames with the coded width and height, so
    any geometry FFmpeg applies while decoding has to be baked into the pixels
    first. A quarter turn is worst: it keeps the pixel count, so transposed
    frames would be reshaped into the coded dimensions and silently distorted. A
    non-square sample aspect ratio only distorts at display time, but the
    enhancement stages treat pixels as square. Both are refused with the
    normalizing step to run first, instead of being silently distorted.
    """
    transform = _display_transform(stream, path)
    if transform is not None:
        raise VideoIOError(
            f"{path} carries {transform}; the streaming node walks the stored pixels and "
            "cannot apply a display matrix. Normalize the file first, for example "
            f"`ffmpeg -i {path} -c:v libx264 -crf 18 -pix_fmt yuv420p normalized.mp4` "
            "(FFmpeg applies the transform and writes the resulting pixels), then enhance "
            "the normalized file."
        )
    ratio = _sample_aspect_ratio(stream, path)
    if ratio is not None and ratio != 1:
        raise VideoIOError(
            f"{path} stores non-square pixels (sample_aspect_ratio "
            f"{ratio.numerator}:{ratio.denominator}); the streaming node treats the stored "
            "pixels as square. Normalize the file first, for example "
            f"`ffmpeg -i {path} -vf scale=iw*sar:ih,setsar=1 -c:v libx264 -crf 18 "
            "-pix_fmt yuv420p normalized.mp4`, then enhance the normalized file."
        )


def _display_transform(stream: dict[str, Any], path: Path) -> str | None:
    """Describe the transform FFmpeg applies while decoding, or None.

    The rotation angle alone is not enough: a vertical flip is reported with
    `rotation 0` although its display matrix is not the identity, and FFmpeg
    still flips the decoded frames (verified against the installed FFmpeg, whose
    output differs from `-noautorotate` for such a file). So the matrix is
    inspected whenever FFprobe reports one, and any non-identity matrix counts as
    a transform the reader cannot represent.
    """
    side_data = stream.get("side_data_list")
    if not isinstance(side_data, list):
        return None
    for entry in side_data:
        if not isinstance(entry, dict):
            continue
        rotation = _display_rotation(entry, path)
        if rotation:
            return f"a {rotation:g} degree display rotation"
        matrix = entry.get("displaymatrix")
        if matrix is None:
            continue
        if not isinstance(matrix, str):
            raise VideoIOError(
                f"ffprobe reported the unreadable display matrix {matrix!r} for {path}"
            )
        identity = _matrix_is_identity(matrix)
        if identity is None:
            raise VideoIOError(
                f"ffprobe reported the unreadable display matrix {matrix!r} for {path}"
            )
        if not identity:
            return "a display matrix that is not the identity (for example a flip)"
    return None


def _display_rotation(entry: dict[str, Any], path: Path) -> float:
    """The rotation of one side data entry in degrees; 0 when it has none."""
    reported = entry.get("rotation")
    if reported is None:
        return 0.0
    try:
        rotation = float(reported)
    except (TypeError, ValueError):
        raise VideoIOError(
            f"ffprobe reported the unreadable display rotation {reported!r} for {path}"
        ) from None
    if not math.isfinite(rotation):
        raise VideoIOError(
            f"ffprobe reported the non-finite display rotation {reported!r} for {path}"
        )
    return rotation


def _matrix_is_identity(matrix: str) -> bool | None:
    """True/False for a parsable display matrix, None when it is not one.

    FFprobe prints the matrix as three numbered rows of fixed point values
    (`00000000: 65536 0 0`), which is nine integers once the row prefixes are
    gone. Anything else is refused as unreadable rather than assumed harmless.
    """
    rows = _MATRIX_ROW_PREFIX.sub("", matrix)
    values = [int(token) for token in re.findall(r"-?\d+", rows)]
    if len(values) != len(_IDENTITY_DISPLAY_MATRIX):
        return None
    return tuple(values) == _IDENTITY_DISPLAY_MATRIX


def _sample_aspect_ratio(stream: dict[str, Any], path: Path) -> Fraction | None:
    """The sample aspect ratio, or None when the pixels count as square.

    FFprobe reports `N/A` for an unspecified ratio and `0:1` for a stream that
    declares no sample aspect ratio at all; both mean the coded pixels are taken
    as square, which is exactly what the streaming node does.
    """
    value = stream.get("sample_aspect_ratio")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text in {"N/A", "0:1", "0/1"}:
        return None
    ratio = _parse_ratio(text)
    if ratio is None:
        raise VideoIOError(
            f"ffprobe reported the unreadable sample aspect ratio {value!r} for {path}"
        )
    return ratio


def _dimensions(stream: dict[str, Any], path: Path) -> tuple[int, int]:
    width = _positive_int(stream.get("width"))
    height = _positive_int(stream.get("height"))
    if width is None or height is None:
        raise VideoIOError(
            f"ffprobe reported invalid dimensions {stream.get('width')}x"
            f"{stream.get('height')} for {path}"
        )
    return width, height


def _nominal_fps(stream: dict[str, Any], path: Path) -> Fraction:
    """The average frame rate, falling back to the representation rate.

    `r_frame_rate` is not the playback rate: FFmpeg computes it as the lowest
    rate that can represent every timestamp, so mixed cadence content reports a
    multiple of the real one (25 and 30 fps together report 150). The average
    rate is what the CFR scan can hold the timestamps against.
    """
    for field in ("avg_frame_rate", "r_frame_rate"):
        fps = _parse_frame_rate(stream.get(field))
        if fps is not None:
            return fps
    raise VideoIOError(
        f"ffprobe reported no usable frame rate for {path}: avg_frame_rate="
        f"{stream.get('avg_frame_rate')!r}, r_frame_rate={stream.get('r_frame_rate')!r}"
    )


def _parse_frame_rate(value: object) -> Fraction | None:
    """`"30000/1001"` as a Fraction; None when the field carries no rate."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text == "N/A":
        return None
    try:
        rate = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _parse_ratio(text: str) -> Fraction | None:
    """`"4:3"` or `"4/3"` as a positive Fraction; None when it is neither."""
    try:
        ratio = Fraction(text.replace(":", "/"))
    except (ValueError, ZeroDivisionError):
        return None
    return ratio if ratio > 0 else None


def _checked_pixel_format(stream: dict[str, Any], path: Path) -> str:
    """The source pixel format, refusing anything outside 8-bit SDR."""
    pixel_format = stream.get("pix_fmt")
    if not isinstance(pixel_format, str) or not pixel_format or pixel_format == "unknown":
        raise VideoIOError(f"ffprobe reported no pixel format for {path}")
    bits = _bit_depth(pixel_format, stream.get("bits_per_raw_sample"))
    if bits is None:
        raise VideoIOError(
            f"{path} uses the floating point pixel format {pixel_format}; the streaming "
            "node needs 8-bit SDR input"
        )
    if bits > 8:
        raise VideoIOError(
            f"{path} is {bits}-bit ({pixel_format}); the streaming node needs 8-bit SDR input"
        )
    transfer = str(stream.get("color_transfer", "")).strip().lower()
    primaries = str(stream.get("color_primaries", "")).strip().lower()
    if transfer in HDR_TRANSFERS or primaries == "bt2020":
        raise VideoIOError(
            f"{path} is not SDR (color_transfer={transfer or 'unknown'}, "
            f"color_primaries={primaries or 'unknown'}); the streaming node needs "
            "8-bit SDR input"
        )
    return pixel_format


def _bit_depth(pixel_format: str, bits_per_raw_sample: object) -> int | None:
    """Bit depth of one pixel format, or None for a floating point format.

    FFprobe reports `bits_per_raw_sample` for most codecs; the pixel format name
    is the fallback and the only signal for the deep formats whose codecs do not
    declare their depth.
    """
    reported = _positive_int(bits_per_raw_sample)
    if reported is not None:
        return reported
    if _FLOAT_PIXEL_FORMAT_PATTERN.search(pixel_format):
        return None
    match = _DEEP_PIXEL_FORMAT_PATTERN.search(pixel_format)
    return int(match.group(1)) if match else 8


def _scan_frame_timestamps(
    path: Path, ffprobe: str, fps: Fraction, interrupt: Callable[[], object] | None
) -> int:
    """Count the decoded frames of `path` and prove that its cadence is constant.

    The frames are decoded and reported in display order, which is the order the
    enhancement pipeline walks them in; a packet count is not a decoded frame
    contract, so it is not used. Only the first and the previous timestamp are
    kept, so the pass is O(1) in memory and time per frame whatever the length
    of the clip.

    Every timestamp is checked against the ideal `first + index / fps` grid
    rather than against the previous gap, so the verdict does not depend on
    where a drift starts: quantization noise stays inside the bound (it does not
    accumulate) while a duplicated, dropped or otherwise off-cadence interval
    moves every later timestamp by a whole nominal interval, which the relative
    part of the bound always catches.
    """
    command = (
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        _VIDEO_STREAM_SELECTOR,
        "-show_frames",
        "-show_entries",
        _FRAME_TIMESTAMP_FIELDS,
        "-of",
        "csv=p=0",
        str(path),
    )
    nominal = float(fps.denominator) / float(fps.numerator)
    bound = max(CADENCE_DRIFT_FLOOR_SECONDS, CADENCE_DRIFT_RATIO * nominal)
    first: float | None = None
    previous: float | None = None
    count = 0
    with _ProbeProcess(command, interrupt=interrupt) as probe:
        for line in probe.lines(limit=SCAN_LINE_LIMIT_BYTES, what="frame timestamps"):
            count += 1
            timestamp = _parse_timestamp(line, count, path)
            if previous is not None and timestamp <= previous:
                raise VideoIOError(
                    f"{path} is not constant frame rate: frame {count} has the timestamp "
                    f"{timestamp:.6f} s after {previous:.6f} s, so the timestamps are "
                    "duplicated or out of order. Variable frame rate (VFR) video and "
                    "timestamps that do not match the declared frame rate are not supported."
                )
            if first is None:
                first = timestamp
            else:
                expected = first + (count - 1) * nominal
                drift = timestamp - expected
                if abs(drift) > bound:
                    raise VideoIOError(
                        f"{path} is not constant frame rate: frame {count} is timestamped "
                        f"{timestamp:.6f} s where {fps} fps puts it at {expected:.6f} s, "
                        f"{abs(drift) * 1000:.3f} ms off (tolerated {bound * 1000:.3f} ms). "
                        "Variable frame rate (VFR) video and timestamps that do not match "
                        "the declared frame rate are not supported."
                    )
            previous = timestamp
        probe.finish(what=f"scanning the frame timestamps of {path}")
    if count == 0:
        raise VideoIOError(f"{path} contains no video frames")
    return count


def _parse_timestamp(line: str, index: int, path: Path) -> float:
    """One `csv=p=0` frame line as seconds.

    The timestamp is the line's first field: the frame printer appends the side
    data of the frames that carry any (SEI, display matrix) behind it.
    """
    text = line.split(",", 1)[0].strip()
    if not text or text == "N/A":
        raise VideoIOError(
            f"ffprobe reported no presentation timestamp for frame {index} of {path}, so "
            "its constant frame rate cannot be verified"
        )
    try:
        timestamp = float(text)
    except ValueError as error:
        raise VideoIOError(
            f"ffprobe reported the unusable presentation timestamp {text!r} for frame "
            f"{index} of {path}"
        ) from error
    if not math.isfinite(timestamp):
        raise VideoIOError(
            f"ffprobe reported the non-finite presentation timestamp {text!r} for frame "
            f"{index} of {path}"
        )
    return timestamp


class _ProbeProcess:
    """One read only FFprobe child: bounded text output, always reaped.

    `ScopedProcess` owns the fixed size byte exchange of the reader and the
    writer; FFprobe answers in text of unknown length instead (one bounded JSON
    blob, or one timestamp per line), which needs streaming reads with a
    deadline, the interrupt callback polled while waiting and a bounded stderr
    tail. This is the smallest owner that covers exactly that.
    """

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        interrupt: Callable[[], object] | None = None,
        timeout: float = PROBE_TIMEOUT_SECONDS,
        stderr_limit: int = STDERR_LIMIT_BYTES,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self._command = tuple(str(part) for part in command)
        self._interrupt = interrupt
        self._timeout = float(timeout)
        self._stderr_limit = max(int(stderr_limit), 0)
        self._process: subprocess.Popen[bytes] | None = None
        self._selector = selectors.DefaultSelector()
        self._out_fd: int | None = None
        self._err_fd: int | None = None
        self._stderr = bytearray()
        self._stderr_truncated = False
        self._closed = False

    @property
    def returncode(self) -> int | None:
        """Exit status of the reaped child, or None when it never ran or runs on."""
        return None if self._process is None else self._process.returncode

    def stderr_text(self) -> str:
        """Bounded stderr tail, decoded for error messages."""
        text = bytes(self._stderr).decode("utf-8", errors="replace")
        return f"...{text}" if self._stderr_truncated else text

    def __enter__(self) -> _ProbeProcess:
        if self._process is not None:
            raise VideoIOError("a probe process is single use; create one per context")
        try:
            self._spawn()
        except BaseException:
            # `_spawn` can fail after Popen succeeded (fd setup, selector
            # registration, a BaseException cancel). Nothing owns the child yet,
            # so this is the only place that can take it down.
            self.terminate()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        # One exit path for success, exceptions and a BaseException cancel: the
        # child's group is killed and reaped, and cleanup never masks the error.
        self.terminate()
        return False

    def read_all(self, limit: int, *, what: str = "output") -> bytes:
        """Every byte the child writes, refusing more than `limit` of them."""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = self._read_chunk(what)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise VideoIOError(
                    f"ffprobe wrote more than {limit} bytes of {what} for "
                    f"{self._command[-1]}; the output is not the expected report"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def lines(self, *, limit: int = SCAN_LINE_LIMIT_BYTES, what: str = "output") -> Iterator[str]:
        """The child's stdout line by line, holding at most one bounded line."""
        pending = bytearray()
        while True:
            chunk = self._read_chunk(what)
            if not chunk:
                break
            pending.extend(chunk)
            while True:
                index = pending.find(b"\n")
                if index < 0:
                    break
                self._check_line_length(index, limit, what)
                line = bytes(pending[:index])
                del pending[: index + 1]
                yield line.decode("utf-8", errors="replace").rstrip("\r")
            self._check_line_length(len(pending), limit, what)
        if pending:
            yield pending.decode("utf-8", errors="replace").rstrip("\r")

    def _check_line_length(self, length: int, limit: int, what: str) -> None:
        """Refuse a line longer than `limit`: FFprobe never legitimately writes one."""
        if length > limit:
            raise VideoIOError(
                f"ffprobe wrote an over-long line of {what} (more than {limit} bytes) for "
                f"{self._command[-1]}"
            )

    def finish(self, *, what: str) -> None:
        """Require the child to exit successfully, with its stderr in the error.

        The wait drains stdout and stderr instead of blocking in `waitpid`, which
        is what keeps an FFprobe that writes a long report or a long warning
        before exiting from wedging on a full pipe, and it polls the caller's
        interrupt callback while it waits.
        """
        if not self._wait_exit_draining(self._timeout, what=what):
            raise VideoIOError(
                f"{what} did not finish within {self._timeout:.0f}s{self._stderr_suffix()}"
            )
        returncode = self.returncode
        if returncode != 0:
            raise VideoIOError(f"{what} failed with exit code {returncode}{self._stderr_suffix()}")

    def terminate(self) -> None:
        """Kill the child's group when it is still alive and reap it; idempotent."""
        try:
            if self._process is not None and self._process.poll() is None:
                self._signal_group(force=False)
                if not self._wait_exit(TERMINATE_GRACE_SECONDS):
                    self._signal_group(force=True)
                    self._wait_exit(TERMINATE_GRACE_SECONDS)
            self._drain_stderr(STDERR_SETTLE_SECONDS)
            self._release()
        except BaseException:  # pragma: no cover - cleanup never masks an error
            pass

    # ----------------------------------------------------------------- private

    def _spawn(self) -> None:
        options: dict[str, Any] = {
            "shell": False,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
        }
        if os.name == "posix":
            # Own session: the child and everything it forks share one group, and
            # that group holds nothing but this child.
            options["start_new_session"] = True
        try:
            self._process = subprocess.Popen(self._command, **options)
        except OSError as error:
            raise VideoIOError(f"cannot start {self._command[0]}: {error}") from error
        process = self._process
        assert process.stdout is not None and process.stderr is not None
        self._out_fd = process.stdout.fileno()
        self._err_fd = process.stderr.fileno()
        os.set_blocking(self._out_fd, False)
        os.set_blocking(self._err_fd, False)
        self._selector.register(self._out_fd, selectors.EVENT_READ, "stdout")
        self._selector.register(self._err_fd, selectors.EVENT_READ, "stderr")

    def _read_chunk(self, what: str) -> bytes:
        """Next stdout chunk; b"" once the child closed stdout.

        `timeout` bounds the silence before one chunk, not the length of the
        whole answer, so a long clip answers for as long as it keeps writing.
        """
        deadline = time.monotonic() + self._timeout
        while True:
            if self._out_fd is None:
                return b""
            _check_interrupt(self._interrupt)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VideoIOError(
                    f"ffprobe stopped writing {what} for {self._timeout:.0f}s"
                    f"{self._stderr_suffix()}"
                )
            try:
                ready = self._selector.select(min(remaining, POLL_INTERVAL_SECONDS))
            except OSError as error:
                raise VideoIOError(f"cannot wait for ffprobe {what}: {error}") from error
            for key, _mask in ready:
                if key.data != "stdout":
                    self._read_stderr()
                    continue
                try:
                    chunk = os.read(self._out_fd, READ_CHUNK_BYTES)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError as error:
                    raise VideoIOError(f"cannot read ffprobe {what}: {error}") from error
                if chunk:
                    return chunk
                self._drop_stdout()
                return b""

    def _drop_stdout(self) -> None:
        if self._out_fd is None:
            return
        try:
            self._selector.unregister(self._out_fd)
        except (KeyError, ValueError, OSError):  # pragma: no cover - already gone
            pass
        self._out_fd = None

    def _read_stderr(self) -> None:
        """Drain whatever the child wrote into the bounded tail."""
        if self._err_fd is None:
            return
        try:
            chunk = os.read(self._err_fd, STDERR_LIMIT_BYTES)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._drop_stderr()
            return
        if not chunk:
            self._drop_stderr()
            return
        if self._stderr_limit <= 0:
            self._stderr_truncated = True
            return
        self._stderr.extend(chunk)
        excess = len(self._stderr) - self._stderr_limit
        if excess > 0:
            # Keep the tail: the last lines before a failure say the most.
            del self._stderr[:excess]
            self._stderr_truncated = True

    def _drop_stderr(self) -> None:
        if self._err_fd is None:
            return
        try:
            self._selector.unregister(self._err_fd)
        except (KeyError, ValueError, OSError):  # pragma: no cover - already gone
            pass
        self._err_fd = None

    def _drain_stderr(self, timeout: float) -> None:
        """Collect the stderr of a child that has stopped writing."""
        deadline = time.monotonic() + timeout
        while self._err_fd is not None and time.monotonic() < deadline:
            self._read_stderr()

    def _stderr_suffix(self) -> str:
        self._drain_stderr(STDERR_SETTLE_SECONDS)
        text = self.stderr_text().strip()
        return f": {text}" if text else ""

    def _wait_exit_draining(self, timeout: float, *, what: str) -> bool:
        """Wait for the child while draining both pipes and polling the cancel.

        Only the previous `wait_exit` call is replaced: stdout is read and
        dropped (the caller is past reading it), stderr goes into the same
        bounded tail the rest of this class uses, the deadline bounds the wait
        and `_check_interrupt` turns the caller's cancel into an error.
        """
        process = self._process
        if process is None:
            return True
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while True:
            if process.poll() is not None:
                self._drain_stderr(STDERR_SETTLE_SECONDS)
                return True
            _check_interrupt(self._interrupt)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                ready = self._selector.select(min(remaining, POLL_INTERVAL_SECONDS))
            except OSError as error:
                raise VideoIOError(f"cannot wait for ffprobe: {error}") from error
            for key, _mask in ready:
                if key.data == "stderr":
                    self._read_stderr()
                else:
                    self._read_chunk(what)

    def _wait_exit(self, timeout: float) -> bool:
        """Wait up to `timeout` seconds for the child; True once it was reaped.

        Only for a child that was just killed or is about to be: it never reads
        the pipes, so a child that blocks writing them would keep this wait
        running until the timeout.
        """
        process = self._process
        if process is None:
            return True
        try:
            process.wait(timeout=max(float(timeout), 0.0))
            return True
        except subprocess.TimeoutExpired:
            return False

    def _signal_group(self, *, force: bool) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        if os.name == "posix":
            try:
                # start_new_session made the child its own group leader, so its
                # PID is the group id and nothing outside the group is signalled.
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                return
            except OSError:
                pass
        try:
            process.kill() if force else process.terminate()
        except OSError:  # pragma: no cover - the child is already gone
            pass

    def _release(self) -> None:
        """Close every fd and the selector; only the captured stderr survives."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is None:
                    continue
                try:
                    stream.close()
                except OSError:  # pragma: no cover - closing twice is harmless
                    pass
        self._out_fd = None
        self._err_fd = None
        try:
            self._selector.close()
        except (OSError, ValueError):  # pragma: no cover - already closed
            pass


# -------------------------------------------------------------------- reader


class FFmpegFrameReader:
    """Decode the frames of one probed file, one frame at a time.

    Used as a context manager and as an iterator. Every step yields exactly one
    float32 `[H,W,3]` RGB frame in [0,1]; a pipe that ends early, a decoder that
    keeps going past `spec.frame_count` and a failing FFmpeg are all errors, so
    a caller that reaches the end has the whole clip and nothing else. Leaving
    the loop early is allowed and kills the decoder on the way out.
    """

    def __init__(
        self,
        spec: VideoSpec,
        *,
        ffmpeg_path: str | os.PathLike[str] = "ffmpeg",
        interrupt: Callable[[], object] | None = None,
    ) -> None:
        if not isinstance(spec, VideoSpec):
            raise TypeError(f"spec must be a VideoSpec, got {type(spec).__name__}")
        self._spec = spec
        self._ffmpeg_path = ffmpeg_path
        self._interrupt = interrupt
        self._process: ScopedProcess | None = None
        self._index = 0
        self._finished = False

    def __enter__(self) -> FFmpegFrameReader:
        if self._process is not None:
            raise VideoIOError("a frame reader is single use; create one per clip")
        process = ScopedProcess(
            self._command(_resolve_tool(self._ffmpeg_path, "ffmpeg")),
            interrupt=self._interrupt,
            timeout=FRAME_TIMEOUT_SECONDS,
        )
        try:
            process.__enter__()
        except ProcessError as error:
            raise VideoIOError(f"cannot start ffmpeg for {self._spec.path}: {error}") from error
        self._process = process
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        # terminate() is idempotent and quiet, so this reaps the decoder on the
        # paths `__next__` does not reach, without masking a propagating error.
        process, self._process = self._process, None
        if process is not None:
            process.terminate()
        return False

    def __iter__(self) -> FFmpegFrameReader:
        return self

    def __next__(self) -> np.ndarray:
        process = self._require_process()
        if self._index >= self._spec.frame_count:
            self._require_end(process)
            raise StopIteration
        self._index += 1
        return _rgb_frame(self._read_frame_bytes(process), self._spec, self._index)

    @property
    def frames_read(self) -> int:
        """Frames handed to the caller so far."""
        return self._index

    def _command(self, ffmpeg: str) -> tuple[str, ...]:
        # `0:V:0` is the stream the probe described: the first video stream that
        # is not an attached picture. No `-frames:v`: the reader stops after
        # `spec.frame_count` frames itself and treats anything behind them as a
        # wrong frame count instead of silently truncating the clip.
        return (
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(self._spec.path),
            "-map",
            _VIDEO_STREAM_MAP,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        )

    def _require_process(self) -> ScopedProcess:
        process = self._process
        if process is None:
            raise VideoIOError("a frame reader must be used as a context manager")
        return process

    def _read_frame_bytes(self, process: ScopedProcess) -> bytes:
        """The next raw RGB frame, or an error naming the truncated clip."""
        size = self._spec.width * self._spec.height * FRAME_CHANNELS
        try:
            return process.read_exactly(
                size, what=f"frame {self._index} of {self._spec.frame_count}"
            )
        except ProcessInterrupted as error:
            raise VideoIOError(f"reading {self._spec.path} was interrupted: {error}") from error
        except ProcessError as error:
            raise VideoIOError(
                f"{self._spec.path} ended after {self._index - 1} of "
                f"{self._spec.frame_count} frames: {error}"
            ) from error

    def _require_end(self, process: ScopedProcess) -> None:
        """Require EOF after the last frame and a successful FFmpeg exit."""
        if self._finished:
            return
        self._finished = True
        try:
            trailing = process.read_exactly(1, what="frames beyond the probed count")
        except ProcessIOError:
            # EOF: the decoder closed stdout, which is what the probed frame
            # count predicts. Every other failure is reported below or here.
            trailing = b""
        except ProcessError as error:
            raise VideoIOError(f"cannot finish reading {self._spec.path}: {error}") from error
        if trailing:
            raise VideoIOError(
                f"ffmpeg produced more than the {self._spec.frame_count} frames probed for "
                f"{self._spec.path}; the file is not the clip that was probed"
            )
        _await_exit(
            process,
            what=f"reading {self._spec.path}",
            timeout=FRAME_TIMEOUT_SECONDS,
        )
        # terminate() is idempotent: the decoder is already reaped, and this is
        # what drains its stderr tail for the message below.
        process.terminate()
        if process.returncode != 0:
            raise VideoIOError(
                f"ffmpeg failed with exit code {process.returncode} while reading "
                f"{self._spec.path}: {process.stderr_text().strip()}"
            )


# -------------------------------------------------------------------- writer


class FFmpegFrameWriter:
    """Encode one video-only file from streamed float32 RGB frames.

    A context manager: every `write(frame)` streams one validated frame to the
    encoder, and the block finalizes successfully only after exactly
    `expected_frames` frames. Frames are interpreted on the [0,1] scale, clipped
    to it and rounded to 8-bit before they are sent. The output container comes
    from the file suffix. Any failure, cancel or frame count mismatch reaps the
    encoder and removes the partial output.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        width: int,
        height: int,
        fps: Fraction | int | float | str,
        expected_frames: int,
        codec: str = "libx264",
        quality: int = 18,
        ffmpeg_path: str | os.PathLike[str] = "ffmpeg",
        interrupt: Callable[[], object] | None = None,
    ) -> None:
        self._path = Path(os.fspath(path))
        self._container = _container_format(self._path)
        self._width = _positive_amount("width", width)
        self._height = _positive_amount("height", height)
        if self._width % 2 or self._height % 2:
            raise VideoIOError(
                f"{OUTPUT_PIXEL_FORMAT} encoding needs even dimensions, got "
                f"{self._width}x{self._height}"
            )
        self._fps = _as_fps(fps)
        self._expected_frames = _positive_amount("expected_frames", expected_frames)
        self._codec = _codec(codec)
        self._quality = _quality(quality)
        self._ffmpeg_path = ffmpeg_path
        self._interrupt = interrupt
        self._process: ScopedProcess | None = None
        self._written = 0

    def __enter__(self) -> FFmpegFrameWriter:
        if self._process is not None:
            raise VideoIOError("a frame writer is single use; create one per output")
        process = ScopedProcess(
            self._command(_resolve_tool(self._ffmpeg_path, "ffmpeg")),
            interrupt=self._interrupt,
            timeout=FRAME_TIMEOUT_SECONDS,
        )
        try:
            process.__enter__()
        except ProcessError as error:
            raise VideoIOError(f"cannot start ffmpeg for {self._path}: {error}") from error
        self._process = process
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            if exc_type is None:
                self._finish()
            else:
                self._discard()
        except BaseException:
            # A failed finalize cleans up the same way an aborted block does, and
            # cleanup must never replace the error already on its way out.
            self._discard()
            raise
        finally:
            self._release()
        return False

    def write(self, frame: np.ndarray) -> None:
        """Validate and encode one float32 `[H,W,3]` RGB frame in [0,1]."""
        process = self._require_process()
        if self._written >= self._expected_frames:
            raise VideoIOError(
                f"{self._path} was told to hold {self._expected_frames} frames but write() "
                f"was called for frame {self._written + 1}"
            )
        payload = _uint8_payload(frame, self._width, self._height, self._written + 1, self._path)
        try:
            process.write(payload, what="video frame")
        except ProcessInterrupted as error:
            raise VideoIOError(f"encoding {self._path} was interrupted: {error}") from error
        except ProcessError as error:
            raise VideoIOError(
                f"ffmpeg stopped accepting frames for {self._path}: {error}"
            ) from error
        self._written += 1

    @property
    def frames_written(self) -> int:
        """Frames accepted by the encoder so far."""
        return self._written

    @property
    def path(self) -> Path:
        """The file this writer produces."""
        return self._path

    def _command(self, ffmpeg: str) -> tuple[str, ...]:
        if self._codec == "h264_nvenc":
            # NVENC takes the same constant quality scale as CQ and needs its
            # rate control mode spelled out.
            rate_control = ("-rc", "vbr", "-cq", str(self._quality), "-b:v", "0")
        else:
            rate_control = ("-crf", str(self._quality))
        return (
            ffmpeg,
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            _fps_text(self._fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            self._codec,
            *rate_control,
            "-pix_fmt",
            OUTPUT_PIXEL_FORMAT,
            "-f",
            self._container,
            str(self._path),
        )

    def _require_process(self) -> ScopedProcess:
        process = self._process
        if process is None:
            raise VideoIOError("a frame writer must be used as a context manager")
        return process

    def _finish(self) -> None:
        """Close the encoder's input and require a complete, playable output."""
        if self._written != self._expected_frames:
            raise VideoIOError(
                f"{self._path} needs {self._expected_frames} frames but only {self._written} "
                "were written; the partial output is removed"
            )
        process = self._require_process()
        process.close_stdin()
        _await_exit(process, what=f"encoding {self._path}", timeout=ENCODE_TIMEOUT_SECONDS)
        # terminate() is idempotent: the encoder is already reaped, and this is
        # what drains its stderr tail for the messages below.
        process.terminate()
        if process.returncode != 0:
            raise VideoIOError(
                f"ffmpeg failed with exit code {process.returncode} while encoding "
                f"{self._path}: {process.stderr_text().strip()}"
            )
        if not self._path.is_file() or self._path.stat().st_size == 0:
            raise VideoIOError(f"ffmpeg reported success but wrote no video to {self._path}")

    def _discard(self) -> None:
        """Kill the encoder and remove the partial output; never raises."""
        self._release()
        _unlink_quietly(self._path)

    def _release(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.terminate()


# --------------------------------------------------------------------- remux


def remux_audio(
    source: VideoSpec,
    video_only_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    ffmpeg_path: str | os.PathLike[str] = "ffmpeg",
    interrupt: Callable[[], object] | None = None,
) -> Path:
    """Put the encoded video and the source's first audio track into one file.

    Both streams are copied, never re-encoded, and no `-shortest` shortens
    either of them. The output muxer follows the output suffix, so the `.mkv`
    file the streaming node uses is Matroska. When the source has no audio the
    video-only file simply replaces the output, without starting FFmpeg at all.
    A failed remux removes the partial output but keeps the caller's video-only
    file.
    """
    if not isinstance(source, VideoSpec):
        raise TypeError(f"source must be a VideoSpec, got {type(source).__name__}")
    video_only = Path(os.fspath(video_only_path))
    output = Path(os.fspath(output_path))
    if not video_only.is_file():
        raise VideoIOError(f"the encoded video {video_only} does not exist")
    if not source.has_audio:
        # The shortcut is a move, not an FFmpeg run, so it has to check the
        # cancel itself: a cancelled job must not publish an output file.
        _check_interrupt(interrupt)
        _replace_file(video_only, output)
        return output
    if not source.path.is_file():
        raise VideoIOError(f"the audio source {source.path} does not exist")
    container = _container_format(output)
    command = (
        _resolve_tool(ffmpeg_path, "ffmpeg"),
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video_only),
        "-i",
        str(source.path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-f",
        container,
        str(output),
    )
    process = ScopedProcess(command, interrupt=interrupt)
    try:
        try:
            process.__enter__()
        except ProcessError as error:
            raise VideoIOError(f"cannot start ffmpeg for {output}: {error}") from error
        _await_exit(process, what=f"remuxing {output}", timeout=REMUX_TIMEOUT_SECONDS)
        # terminate() is idempotent: the child is already reaped, and this is
        # what drains its stderr tail for the message below.
        process.terminate()
        if process.returncode != 0:
            raise VideoIOError(
                f"ffmpeg failed with exit code {process.returncode} while remuxing {output}: "
                f"{process.stderr_text().strip()}"
            )
        if not output.is_file() or output.stat().st_size == 0:
            raise VideoIOError(f"ffmpeg reported success but wrote no video to {output}")
    except BaseException:
        # The caller owns `video_only`; only our own partial output goes away.
        _unlink_quietly(output)
        raise
    finally:
        process.terminate()
    return output


# ------------------------------------------------------------------- helpers


def _check_interrupt(callback: Callable[[], object] | None) -> None:
    """Abort when the caller's interrupt callback asks for it.

    A callback that raises (ComfyUI's `throw_exception_if_processing_interrupted`)
    propagates its own BaseException, which the callers' cleanup turns into a
    killed child; a callback that returns a truthy value aborts here.
    """
    if callback is not None and callback():
        raise VideoIOError("the video I/O operation was interrupted")


def _await_exit(process: ScopedProcess, *, what: str, timeout: float) -> None:
    """Wait for a child to exit, draining its output, its cancel and its clock.

    A plain `wait_exit` here would deadlock any FFmpeg that reports a failure on
    stderr (a failed encode, a refused remux, a decoder that stops early): the
    message fills the pipe, FFmpeg blocks writing it and the parent blocks in
    `waitpid` waiting for an exit that cannot happen. This keeps both pipes
    drained - stdout discarded, stderr in the bounded tail - while it waits. The
    process was built with the caller's interrupt callback, so a cancel is
    detected while waiting rather than after the deadline.
    """
    try:
        exited = process.wait_exit_draining(timeout, what=what)
    except ProcessInterrupted as error:
        raise VideoIOError(f"{what} was cancelled: {error}") from error
    except ProcessError as error:
        raise VideoIOError(f"cannot wait for {what}: {error}") from error
    if not exited:
        raise VideoIOError(
            f"{what} did not finish within {timeout:.0f}s"
            f"{_stderr_suffix(process)}"
        )


def _stderr_suffix(process: ScopedProcess) -> str:
    """The bounded FFmpeg diagnostics, for the messages of this module."""
    text = process.stderr_text().strip()
    return f": {text}" if text else ""


def _positive_int(value: object) -> int | None:
    """An int from an int or a numeric string, when it is positive."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _positive_amount(name: str, value: object) -> int:
    """One positive frame count, size or dimension, or an actionable error."""
    parsed = _positive_int(value)
    if parsed is None:
        raise VideoIOError(f"{name} must be a positive int, got {value!r}")
    return parsed


def _codec(value: object) -> str:
    if value not in SUPPORTED_CODECS:
        raise VideoIOError(
            f"unsupported video codec {value!r}; the streaming node encodes with "
            f"{' or '.join(SUPPORTED_CODECS)}"
        )
    return str(value)


def _quality(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VideoIOError(f"quality must be an int in {QUALITY_MIN}..{QUALITY_MAX}, got {value!r}")
    if not QUALITY_MIN <= value <= QUALITY_MAX:
        raise VideoIOError(
            f"quality must be a CRF/CQ value in {QUALITY_MIN}..{QUALITY_MAX}, got {value}"
        )
    return value


def _as_fps(value: Fraction | int | float | str) -> Fraction:
    """One exact frame rate from a Fraction, an int, a float or `"num/den"`."""
    if isinstance(value, Fraction):
        fps = value
    elif isinstance(value, bool):
        raise VideoIOError(f"fps must be a frame rate, got {value!r}")
    elif isinstance(value, int):
        fps = Fraction(value)
    elif isinstance(value, float):
        # `Fraction(29.97)` is the binary noise; the decimal reading is what the
        # caller meant.
        fps = Fraction(str(value))
    elif isinstance(value, str):
        try:
            fps = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise VideoIOError(
                f"fps must be a frame rate like '30000/1001', got {value!r}"
            ) from error
    else:
        raise VideoIOError(
            f"fps must be a Fraction, an int, a float or a 'num/den' string, got "
            f"{type(value).__name__}"
        )
    if fps <= 0:
        raise VideoIOError(f"fps must be positive, got {value!r}")
    return fps


def _fps_text(fps: Fraction) -> str:
    """The `-r` argument of one exact frame rate."""
    if fps.denominator == 1:
        return str(fps.numerator)
    return f"{fps.numerator}/{fps.denominator}"


def _container_format(path: Path) -> str:
    """The muxer of `path`, taken from its suffix."""
    suffix = path.suffix.lower()
    try:
        return CONTAINER_FORMATS[suffix]
    except KeyError:
        supported = ", ".join(sorted(CONTAINER_FORMATS))
        raise VideoIOError(
            f"cannot infer the output container of {path}; use one of {supported}"
        ) from None


def _uint8_payload(frame: np.ndarray, width: int, height: int, index: int, path: Path) -> bytes:
    """One frame as raw 8-bit RGB, clipped to [0,1] and rounded."""
    array = np.asarray(frame)
    if array.shape != (height, width, FRAME_CHANNELS):
        raise VideoIOError(
            f"frame {index} for {path} must be {height}x{width} RGB "
            f"{(height, width, FRAME_CHANNELS)}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating) and not np.issubdtype(array.dtype, np.integer):
        raise VideoIOError(
            f"frame {index} for {path} must be numeric, got dtype {array.dtype}"
        )
    values = array.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise VideoIOError(f"frame {index} for {path} contains NaN or infinite values")
    scaled = np.clip(values, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)
    return scaled.astype(np.uint8).tobytes()


def _rgb_frame(raw: bytes, spec: VideoSpec, index: int) -> np.ndarray:
    """One raw RGB frame from the decoder as float32 `[H,W,3]` in [0,1]."""
    expected = spec.width * spec.height * FRAME_CHANNELS
    if len(raw) != expected:  # pragma: no cover - the reader reads exact frames
        raise VideoIOError(
            f"frame {index} of {spec.path} arrived with {len(raw)} of {expected} bytes"
        )
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(spec.height, spec.width, FRAME_CHANNELS)
    return frame.astype(np.float32) / np.float32(255.0)


def _replace_file(source: Path, output: Path) -> None:
    """Move `source` onto `output`, across filesystems if it has to."""
    try:
        shutil.move(os.fspath(source), os.fspath(output))
    except OSError as error:
        raise VideoIOError(f"cannot move {source} to {output}: {error}") from error


def _unlink_quietly(path: Path) -> None:
    """Remove a partial output without replacing an error on its way out."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:  # pragma: no cover - e.g. a filesystem that refuses removal
        pass
