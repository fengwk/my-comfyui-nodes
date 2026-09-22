"""DNR3: the frame protocol shared by the node core and the DLSS worker host.

The wire format is the upstream MIT ``ComfyUI-DLSS5-NR`` (RH-RunningHub) DNR2
host transport with one deliberate change: the header field upstream leaves
unused (``profile``) carries the feature bitmask here, so super resolution,
neural rendering and both together are independently selectable while the
header keeps its fixed 72-byte size. Everything else (``FRM2``, ``OUT1``,
``END1`` and the field order) is unchanged, so one build of the vendored
native bridge/host serves all three combinations.

Layout, all integers little-endian, one process per execution::

    header   : 4s + 12 uint32 + 5 float32                 (72 bytes, b"DNR3")
    frame    : 4s + index + reset                         (12 bytes, b"FRM2")
               input RGB float32 (w*h*3), motion fp16 bits (w*h*2)
    reply    : 4s + index + ok + float_count              (16 bytes, b"OUT1")
               ok=1: output RGB float32 (out_w*out_h*3)
               ok=0: uint32 length + UTF-8 error text
    end      : 4s b"END1" after the last frame

This module is pure: it validates and (de)serializes; it never starts a
process. `my_nodes.core.video_enhance.scoped_process` owns the process and
`my_nodes.core.video_enhance.dlss_worker` drives the exchange.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import BinaryIO, Mapping

import numpy as np

MAGIC = b"DNR3"
FRAME_MAGIC = b"FRM2"
REPLY_MAGIC = b"OUT1"
END_MAGIC = b"END1"

# Header feature bitmask (replaces the upstream unused `profile` field).
FEATURE_SR = 0x1
FEATURE_NR = 0x2
FEATURE_MASK = FEATURE_SR | FEATURE_NR

HEADER = struct.Struct("<4sIIIIIIIIIIII5f")
FRAME_HEADER = struct.Struct("<4sII")
REPLY_HEADER = struct.Struct("<4sIII")
LENGTH = struct.Struct("<I")

HEADER_SIZE = HEADER.size
FRAME_HEADER_SIZE = FRAME_HEADER.size
REPLY_HEADER_SIZE = REPLY_HEADER.size
END_SIZE = len(END_MAGIC)

# Envelope limits, mirrored by the native host so a bad header fails early on
# both sides instead of inside NGX.
MAX_DIM = 16384
MAX_LONG_EDGE = 7680
MAX_SHORT_EDGE = 4320
MAX_PIXELS = 1 << 28
MAX_FRAMES = 1_000_000
MAX_ERROR_BYTES = 65535

# NGX DLSS `PerfQualityValue` -> the geometric ratio the runtime derives from
# it (native: FixedScalingRatio in dlss5nr_bridge.cpp). 5 is native size: it is
# the only selector a neural-rendering-only session may use.
PERF_QUALITY_RATIOS: Mapping[int, float] = {5: 1.0, 2: 1.5, 1: 1.724, 0: 2.0, 3: 3.0}
NATIVE_PERF_QUALITY = 5
# Absolute tolerance accepted when comparing requested dimensions to the ratio
# the runtime derives from PerfQualityValue (native default: 0.03).
RATIO_TOLERANCE = 0.03

# The scales the combined node offers (see plan.SR_SCALES) and the selector
# each one has to run as. 1.0 is native DLAA (feature 1), not neural rendering.
SR_SCALE_PERF_QUALITY: Mapping[float, int] = {1.0: 5, 1.5: 2, 2.0: 0, 3.0: 3}

_RGB_DTYPE = np.dtype("<f4")
_MOTION_BITS_DTYPE = np.dtype("<u2")


class Dnr3Error(RuntimeError):
    """Base class for every DNR3 protocol, validation and runtime failure."""


class Dnr3ValidationError(Dnr3Error):
    """The caller handed this side something that cannot be sent as DNR3."""


class Dnr3ProtocolError(Dnr3Error):
    """The worker answered with something that is not a valid DNR3 stream."""


class Dnr3RemoteError(Dnr3Error):
    """The worker reported a failure for one frame."""


def _require_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Dnr3ValidationError(f"{name} must be an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise Dnr3ValidationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise Dnr3ValidationError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Dnr3ValidationError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise Dnr3ValidationError(f"{name} must be finite, got {value!r}")
    return number


def check_features(features: object) -> int:
    """Validate a feature bitmask: some known bits, no unknown ones."""
    value = _require_int(features, "features", minimum=0)
    if value == 0:
        raise Dnr3ValidationError("features must enable FEATURE_SR, FEATURE_NR or both")
    unknown = value & ~FEATURE_MASK
    if unknown:
        raise Dnr3ValidationError(
            f"features {value:#x} contains unknown bits {unknown:#x}; "
            f"expected FEATURE_SR={FEATURE_SR:#x}, FEATURE_NR={FEATURE_NR:#x}"
        )
    return value


def perf_quality_for_scale(scale: float) -> int:
    """PerfQualityValue that makes the runtime produce exactly `scale`."""
    try:
        return SR_SCALE_PERF_QUALITY[float(scale)]
    except KeyError:
        raise Dnr3ValidationError(
            f"unsupported super-resolution scale {scale!r}; expected one of "
            f"{sorted(SR_SCALE_PERF_QUALITY)}"
        ) from None


def check_dimensions(name: str, width: int, height: int) -> None:
    """Envelope checks shared by the header validation on both sides."""
    _require_int(width, f"{name} width", minimum=1)
    _require_int(height, f"{name} height", minimum=1)
    if width > MAX_DIM or height > MAX_DIM:
        raise Dnr3ValidationError(f"{name} {width}x{height} exceeds the {MAX_DIM} pixel limit")
    pixels = width * height
    if pixels > MAX_PIXELS:
        raise Dnr3ValidationError(
            f"{name} {width}x{height} exceeds the {MAX_PIXELS} pixel frame budget"
        )


@dataclass(frozen=True)
class Header:
    """Validated DNR3 session header: one execution of one worker process.

    Fields are declared in wire order. `features` selects what the worker may
    load and create: `FEATURE_SR` means feature 1, either native DLAA (1.0x)
    or a larger scale, and never touches the neural-rendering runtime;
    `FEATURE_NR` alone stays at native resolution and creates only feature 18;
    both together run feature 1 first (DLAA or SR) and feed its output to NR.
    """

    input_width: int
    input_height: int
    output_width: int
    output_height: int
    warmup_frames: int
    frame_count: int
    perf_quality: int
    features: int
    preset: int
    style: int
    automask: bool
    ui_correction: bool
    intensity: float
    tone: float
    structure: float
    skin: float
    global_tone: float

    def __post_init__(self) -> None:
        check_dimensions("input", self.input_width, self.input_height)
        check_dimensions("output", self.output_width, self.output_height)
        long_edge = max(self.output_width, self.output_height)
        short_edge = min(self.output_width, self.output_height)
        if long_edge > MAX_LONG_EDGE or short_edge > MAX_SHORT_EDGE:
            raise Dnr3ValidationError(
                f"output {self.output_width}x{self.output_height} exceeds the DLSS "
                f"{MAX_LONG_EDGE}x{MAX_SHORT_EDGE} envelope"
            )
        _require_int(self.frame_count, "frame_count", minimum=1)
        if self.frame_count > MAX_FRAMES:
            raise Dnr3ValidationError(f"frame_count must be <= {MAX_FRAMES}, got {self.frame_count}")
        _require_int(self.warmup_frames, "warmup_frames", minimum=0)
        if self.warmup_frames > self.frame_count:
            raise Dnr3ValidationError(
                f"warmup_frames {self.warmup_frames} exceeds frame_count {self.frame_count}"
            )
        features = check_features(self.features)
        _require_int(self.perf_quality, "perf_quality", minimum=0)
        if self.perf_quality not in PERF_QUALITY_RATIOS:
            raise Dnr3ValidationError(
                f"perf_quality must be one of {sorted(PERF_QUALITY_RATIOS)}, "
                f"got {self.perf_quality!r}"
            )
        _require_int(self.preset, "preset", minimum=0)
        _require_int(self.style, "style", minimum=0)
        _require_bool(self.automask, "automask")
        _require_bool(self.ui_correction, "ui_correction")
        for name in ("intensity", "tone", "structure", "skin", "global_tone"):
            _require_float(getattr(self, name), name)
        self._check_scale(features)

    def _check_scale(self, features: int) -> None:
        """Reject dimension/quality combinations the runtime cannot honour.

        The native side enforces the same contract (ProcessFrame in
        dlss5nr_bridge.cpp); checking it here turns an opaque NGX failure into
        an actionable error before Wine is even started.
        """
        ratio_x = self.output_width / self.input_width
        ratio_y = self.output_height / self.input_height
        if abs(ratio_x - ratio_y) > RATIO_TOLERANCE:
            raise Dnr3ValidationError(
                f"output {self.output_width}x{self.output_height} does not preserve the "
                f"input aspect ratio {self.input_width}x{self.input_height}"
            )
        if not features & FEATURE_SR:
            if (self.input_width, self.input_height) != (self.output_width, self.output_height):
                raise Dnr3ValidationError(
                    "neural rendering without super resolution must stay at native "
                    f"resolution, got {self.input_width}x{self.input_height} -> "
                    f"{self.output_width}x{self.output_height}"
                )
            if self.perf_quality != NATIVE_PERF_QUALITY:
                raise Dnr3ValidationError(
                    f"neural rendering without super resolution must use "
                    f"perf_quality={NATIVE_PERF_QUALITY} (native size), got {self.perf_quality}"
                )
            return
        expected = PERF_QUALITY_RATIOS[self.perf_quality]
        if abs(ratio_x - expected) > RATIO_TOLERANCE or abs(ratio_y - expected) > RATIO_TOLERANCE:
            raise Dnr3ValidationError(
                f"output {self.output_width}x{self.output_height} is {ratio_x:.3f}x the input "
                f"but perf_quality {self.perf_quality} means {expected:.3f}x; use "
                f"perf_quality_for_scale() with a supported scale"
            )
        # FEATURE_SR at 1.0 is native DLAA (feature 1). A non-SR session is
        # already rejected above if it tries to leave native size.

    @property
    def sr_enabled(self) -> bool:
        return bool(self.features & FEATURE_SR)

    @property
    def nr_enabled(self) -> bool:
        return bool(self.features & FEATURE_NR)

    @property
    def input_pixels(self) -> int:
        return self.input_width * self.input_height

    @property
    def output_pixels(self) -> int:
        return self.output_width * self.output_height

    @property
    def input_floats(self) -> int:
        return self.input_pixels * 3

    @property
    def motion_words(self) -> int:
        return self.input_pixels * 2

    @property
    def output_floats(self) -> int:
        return self.output_pixels * 3

    def pack(self) -> bytes:
        """Serialize this header into the fixed 72-byte wire form."""
        return HEADER.pack(
            MAGIC,
            self.input_width,
            self.input_height,
            self.output_width,
            self.output_height,
            self.warmup_frames,
            self.frame_count,
            self.perf_quality,
            self.features,
            self.preset,
            self.style,
            int(self.automask),
            int(self.ui_correction),
            self.intensity,
            self.tone,
            self.structure,
            self.skin,
            self.global_tone,
        )

    @classmethod
    def parse(cls, raw: bytes) -> Header:
        """Deserialize and validate a wire header (worker side)."""
        if len(raw) != HEADER_SIZE:
            raise Dnr3ProtocolError(
                f"header must be exactly {HEADER_SIZE} bytes, got {len(raw)}"
            )
        (
            magic,
            input_width,
            input_height,
            output_width,
            output_height,
            warmup_frames,
            frame_count,
            perf_quality,
            features,
            preset,
            style,
            automask,
            ui_correction,
            intensity,
            tone,
            structure,
            skin,
            global_tone,
        ) = HEADER.unpack(raw)
        if magic != MAGIC:
            raise Dnr3ProtocolError(f"unexpected header magic {magic!r}, expected {MAGIC!r}")
        # Wire flags are 0/1, not arbitrary integers that bool() would accept.
        for name, flag in (("automask", automask), ("ui_correction", ui_correction)):
            if flag not in (0, 1):
                raise Dnr3ProtocolError(f"header {name} must be 0 or 1, got {flag}")
        try:
            return cls(
                input_width=input_width,
                input_height=input_height,
                output_width=output_width,
                output_height=output_height,
                warmup_frames=warmup_frames,
                frame_count=frame_count,
                perf_quality=perf_quality,
                features=features,
                preset=preset,
                style=style,
                automask=automask == 1,
                ui_correction=ui_correction == 1,
                intensity=float(intensity),
                tone=float(tone),
                structure=float(structure),
                skin=float(skin),
                global_tone=float(global_tone),
            )
        except Dnr3ValidationError as exc:
            raise Dnr3ProtocolError(f"invalid DNR3 header: {exc}") from exc


def pack_frame_header(index: int, reset: bool) -> bytes:
    """Serialize the per-frame header that precedes one frame's payload."""
    return FRAME_HEADER.pack(
        FRAME_MAGIC,
        _require_int(index, "index", minimum=0),
        int(_require_bool(reset, "reset")),
    )


def parse_frame_header(raw: bytes) -> tuple[int, bool]:
    """Deserialize a frame header; returns (index, reset)."""
    magic, index, reset = FRAME_HEADER.unpack(_exact(raw, FRAME_HEADER_SIZE, "frame header"))
    if magic != FRAME_MAGIC:
        raise Dnr3ProtocolError(f"unexpected frame magic {magic!r}, expected {FRAME_MAGIC!r}")
    if reset not in (0, 1):
        raise Dnr3ProtocolError(f"frame reset flag must be 0 or 1, got {reset}")
    return index, bool(reset)


def pack_reply_header(index: int, float_count: int) -> bytes:
    """Serialize an OUT1 reply header for a successful frame."""
    return REPLY_HEADER.pack(
        REPLY_MAGIC,
        _require_int(index, "index", minimum=0),
        1,
        _require_int(float_count, "float_count", minimum=0),
    )


def pack_error_reply(index: int, message: str) -> bytes:
    """Serialize a complete failed-frame reply: header, length, UTF-8 text."""
    text = str(message).encode("utf-8")[:MAX_ERROR_BYTES]
    header = REPLY_HEADER.pack(REPLY_MAGIC, _require_int(index, "index", minimum=0), 0, 0)
    return header + LENGTH.pack(len(text)) + text


def parse_reply_header(raw: bytes) -> tuple[int, bool, int]:
    """Deserialize an OUT1 reply header; returns (index, ok, float_count)."""
    magic, index, ok, float_count = REPLY_HEADER.unpack(
        _exact(raw, REPLY_HEADER_SIZE, "reply header")
    )
    if magic != REPLY_MAGIC:
        raise Dnr3ProtocolError(f"unexpected reply magic {magic!r}, expected {REPLY_MAGIC!r}")
    if ok not in (0, 1):
        raise Dnr3ProtocolError(f"reply ok flag must be 0 or 1, got {ok}")
    return index, bool(ok), float_count


def check_error_length(raw: bytes) -> int:
    """Read the declared error text length, bounded before anything is read."""
    (length,) = LENGTH.unpack(_exact(raw, LENGTH.size, "error length"))
    if length > MAX_ERROR_BYTES:
        raise Dnr3ProtocolError(
            f"error text of {length} bytes exceeds the {MAX_ERROR_BYTES} byte limit"
        )
    return length


def _exact(raw: bytes, size: int, what: str) -> bytes:
    if len(raw) != size:
        raise Dnr3ProtocolError(f"{what} must be exactly {size} bytes, got {len(raw)}")
    return raw


def rgb_payload(rgb: object, width: int, height: int) -> bytes:
    """Validate a frame's RGB array and serialize it as little-endian float32."""
    array = _checked_rgb(rgb, width, height)
    return np.ascontiguousarray(array, dtype=_RGB_DTYPE).tobytes()


def decode_rgb(payload: bytes, width: int, height: int) -> np.ndarray:
    """Validate a received RGB payload; returns a read-only (h, w, 3) view."""
    expected = width * height * 3 * _RGB_DTYPE.itemsize
    if len(payload) != expected:
        raise Dnr3ProtocolError(
            f"RGB payload must be {expected} bytes for {width}x{height}, got {len(payload)}"
        )
    frame = np.frombuffer(payload, dtype=_RGB_DTYPE).reshape((height, width, 3))
    if not np.isfinite(frame).all():
        raise Dnr3ProtocolError(f"RGB payload for {width}x{height} contains non-finite values")
    return frame


def _checked_rgb(rgb: object, width: int, height: int) -> np.ndarray:
    array = np.asarray(rgb)
    if array.dtype != np.float32:
        raise Dnr3ValidationError(f"rgb must be float32, got {array.dtype}")
    if array.shape != (height, width, 3):
        raise Dnr3ValidationError(
            f"rgb must have shape {(height, width, 3)}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise Dnr3ValidationError(f"rgb for {width}x{height} contains non-finite values")
    return array


def motion_payload(motion: object, width: int, height: int) -> bytes:
    """Serialize motion vectors as fp16 bits; `None` means static (zero) motion."""
    if motion is None:
        return bytes(width * height * 2 * _MOTION_BITS_DTYPE.itemsize)
    array = np.ascontiguousarray(motion)
    if array.shape != (height, width, 2):
        raise Dnr3ValidationError(
            f"motion must have shape {(height, width, 2)}, got {array.shape}"
        )
    if array.dtype == np.float16:
        if not np.isfinite(array).all():
            raise Dnr3ValidationError(
                f"motion for {width}x{height} contains non-finite float16 values"
            )
        return array.view(_MOTION_BITS_DTYPE).tobytes()
    if array.dtype == np.uint16:
        return array.view(_MOTION_BITS_DTYPE).tobytes()
    raise Dnr3ValidationError(f"motion must be float16 or uint16, got {array.dtype}")


def read_exact(stream: BinaryIO, size: int, what: str) -> bytes:
    """Read exactly `size` bytes from a blocking stream (worker side helper)."""
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise Dnr3ProtocolError(
                f"stream ended after {size - remaining} of {size} bytes of {what}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
