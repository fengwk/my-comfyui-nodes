"""Immutable execution plan for the combined video-enhance node.

The plan is the single validated description of one execution: it is built from
the public node settings, exposes the deterministic stage order and never starts
a backend by itself. A plan with every stage disabled is a valid pass-through
plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 1.0 is native-resolution DLAA (feature 1, perf_quality 5), not a no-op.
SR_SCALES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
NR_PROFILES: tuple[str, ...] = ("light", "standard", "portrait", "detail")
INTERPOLATION_FACTORS: tuple[int, ...] = (2,)
NR_INTENSITY_RANGE: tuple[float, float] = (0.0, 2.0)

STAGE_DLSS: str = "dlss"
STAGE_VFI: str = "vfi"

# Explicit order of the two stages when both are active. Only the combined run
# is ambiguous; a single active stage keeps its own behavior either way.
STAGE_ORDER_DLSS_THEN_VFI: str = "dlss_then_vfi"
STAGE_ORDER_VFI_THEN_DLSS: str = "vfi_then_dlss"
STAGE_ORDERS: tuple[str, ...] = (STAGE_ORDER_DLSS_THEN_VFI, STAGE_ORDER_VFI_THEN_DLSS)


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class VideoEnhancePlan:
    """Validated, immutable description of one video-enhance execution.

    Every setting is validated, including the parameters of disabled stages:
    the values must stay meaningful and serializable so they can be forwarded to
    the worker. A disabled stage is only excluded from `stages`, so it never
    causes backend startup.

    `stages` is the deterministic execution order. `stage_order` only chooses
    between the two combinations of both stages: the legacy default runs DLSS
    (super resolution and neural rendering) first and interpolates its output,
    the alternative interpolates first and enhances the interpolated frames.
    When a single stage is active its order is not ambiguous and unchanged.
    """

    enable_super_resolution: bool = False
    sr_scale: float = 2.0
    enable_neural_rendering: bool = False
    nr_profile: str = "standard"
    nr_intensity: float = 1.0
    enable_frame_interpolation: bool = False
    interpolation_factor: int = 2
    stage_order: str = STAGE_ORDER_DLSS_THEN_VFI

    def __post_init__(self) -> None:
        _require_bool(self.enable_super_resolution, "enable_super_resolution")
        scale = _require_number(self.sr_scale, "sr_scale")
        if scale not in SR_SCALES:
            raise ValueError(f"sr_scale must be one of {SR_SCALES}, got {self.sr_scale!r}")
        _require_bool(self.enable_neural_rendering, "enable_neural_rendering")
        profile = _require_str(self.nr_profile, "nr_profile")
        if profile not in NR_PROFILES:
            raise ValueError(f"nr_profile must be one of {NR_PROFILES}, got {profile!r}")
        intensity = _require_number(self.nr_intensity, "nr_intensity")
        if not NR_INTENSITY_RANGE[0] <= intensity <= NR_INTENSITY_RANGE[1]:
            raise ValueError(
                f"nr_intensity must be within {NR_INTENSITY_RANGE}, got {self.nr_intensity!r}"
            )
        _require_bool(self.enable_frame_interpolation, "enable_frame_interpolation")
        factor = _require_int(self.interpolation_factor, "interpolation_factor")
        if factor not in INTERPOLATION_FACTORS:
            raise ValueError(
                f"interpolation_factor must be one of {INTERPOLATION_FACTORS}, got {factor!r}"
            )
        order = _require_str(self.stage_order, "stage_order")
        if order not in STAGE_ORDERS:
            raise ValueError(f"stage_order must be one of {STAGE_ORDERS}, got {order!r}")
        # Normalize numbers so equal settings always produce equal plans.
        object.__setattr__(self, "sr_scale", scale)
        object.__setattr__(self, "nr_intensity", intensity)
        object.__setattr__(self, "stage_order", order)

    @property
    def uses_dlss(self) -> bool:
        """True when the DLSS worker stage must run for this execution."""
        return self.enable_super_resolution or self.enable_neural_rendering

    @property
    def uses_frame_interpolation(self) -> bool:
        """True when the frame-interpolation stage must run."""
        return self.enable_frame_interpolation

    @property
    def uses_both_stages(self) -> bool:
        """True when `stage_order` decides between two active stages."""
        return self.uses_dlss and self.uses_frame_interpolation

    @property
    def stages(self) -> tuple[str, ...]:
        """Deterministic stage order for this execution; empty when all off."""
        if self.uses_both_stages:
            if self.stage_order == STAGE_ORDER_VFI_THEN_DLSS:
                return (STAGE_VFI, STAGE_DLSS)
            return (STAGE_DLSS, STAGE_VFI)
        if self.uses_dlss:
            return (STAGE_DLSS,)
        if self.uses_frame_interpolation:
            return (STAGE_VFI,)
        return ()

    @property
    def is_pass_through(self) -> bool:
        """True when no stage is enabled and frames are returned unchanged."""
        return not self.stages
