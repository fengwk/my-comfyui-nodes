"""Temporal motion guides for one DNR3 session.

OpenCV DIS dense flow is the only optical-flow implementation. A scene cut or
an implausible flow magnitude resets the worker history and sends zero motion
for that frame. `none` never imports OpenCV: every frame is reset with zero
motion. Selecting optical flow when cv2 is missing is an error, not a fallback.
NGX consumes backward-reprojection vectors (current pixel -> previous pixel),
so DIS receives the current grayscale frame first and the previous frame second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOTION_OPTICAL_FLOW = "optical_flow"
MOTION_NONE = "none"
MOTION_MODES: tuple[str, ...] = (MOTION_OPTICAL_FLOW, MOTION_NONE)

# DIS pixels that are larger than this fraction of the shorter edge are treated
# as a cut (or a broken estimate) rather than usable temporal guidance.
_IMPLausible_FLOW_FRACTION = 0.25


class MotionGuideError(RuntimeError):
    """Optical flow was requested but OpenCV is not importable."""


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise MotionGuideError(
            "DLSS motion is set to optical_flow but OpenCV (cv2) is not installed. "
            "Install opencv-python-headless in the ComfyUI environment, or set "
            "motion to none. There is no hidden fallback."
        ) from exc
    return cv2


@dataclass(frozen=True)
class MotionGuide:
    """Motion vectors and the worker reset flag for one input frame."""

    motion: np.ndarray | None
    reset: bool


def _gray(frame: np.ndarray) -> np.ndarray:
    rgb = np.clip(frame, 0.0, 1.0)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.ascontiguousarray((gray * 255.0).astype(np.uint8))


def _mean_abs_diff(current: np.ndarray, previous: np.ndarray) -> float:
    return float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))


class MotionGuides:
    """Build one guide per frame, in order, for a single worker session."""

    def __init__(self, mode: str, scene_cut_threshold: float) -> None:
        if mode not in MOTION_MODES:
            raise ValueError(f"motion mode must be one of {MOTION_MODES}, got {mode!r}")
        if not np.isfinite(scene_cut_threshold) or scene_cut_threshold < 0.0:
            raise ValueError(f"scene_cut_threshold must be a finite value >= 0, got {scene_cut_threshold!r}")
        self.mode = mode
        self.scene_cut_threshold = float(scene_cut_threshold)
        self._previous: np.ndarray | None = None
        self._flow = None

    def guide(self, frame: np.ndarray) -> MotionGuide:
        if self.mode == MOTION_NONE:
            return MotionGuide(motion=None, reset=True)
        return self._optical_flow(frame)

    def _optical_flow(self, frame: np.ndarray) -> MotionGuide:
        gray = _gray(frame)
        previous = self._previous
        self._previous = gray
        if previous is None or previous.shape != gray.shape:
            return MotionGuide(motion=None, reset=True)
        if self.scene_cut_threshold > 0.0 and _mean_abs_diff(gray, previous) >= self.scene_cut_threshold * 255.0:
            return MotionGuide(motion=None, reset=True)
        flow = self._dis_current_to_previous(gray, previous)
        height, width = gray.shape
        limit = _IMPLausible_FLOW_FRACTION * float(min(height, width))
        if not np.isfinite(flow).all() or float(np.max(np.abs(flow))) > limit:
            return MotionGuide(motion=None, reset=True)
        return MotionGuide(motion=np.ascontiguousarray(flow, dtype=np.float16), reset=False)

    def _dis_current_to_previous(
        self, current: np.ndarray, previous: np.ndarray
    ) -> np.ndarray:
        if self._flow is None:
            cv2 = _import_cv2()
            self._flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        # OpenCV maps pixels from its first image into its second image. NGX
        # needs prev_pixel - current_pixel, not the forward temporal flow.
        return self._flow.calc(current, previous, None)
