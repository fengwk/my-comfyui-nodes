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
# `custom` is the last choice: every built-in profile keeps its own fixed model
# fields, so it resolves exactly as before and only `custom` reads the
# advanced fields below.
NR_PROFILES: tuple[str, ...] = ("light", "standard", "portrait", "detail", "custom")
CUSTOM_NR_PROFILE: str = "custom"
INTERPOLATION_FACTORS: tuple[int, ...] = (2,)
NR_INTENSITY_RANGE: tuple[float, float] = (0.0, 2.0)

# Advanced neural-rendering controls. The strings are the user-facing choices;
# the profile resolver maps them onto the integer model selectors.
NR_STYLES: tuple[str, ...] = ("Default", "Natural", "Cinematic")
NR_PRESETS: tuple[str, ...] = ("Default", "Preset 1", "Preset 2", "Preset 3")
NR_LOCAL_STRUCTURE_RANGE: tuple[float, float] = (0.0, 2.0)
NR_LOCAL_TONE_RANGE: tuple[float, float] = (0.0, 2.0)
NR_SKIN_RANGE: tuple[float, float] = (-1.0, 2.0)
NR_DETAIL_RANGE: tuple[float, float] = (0.0, 2.0)
NR_COLOR_RANGE: tuple[float, float] = (0.0, 1.0)
# DLSS super-resolution preset. `Default` leaves the runtime default in place,
# the letters are the model selector a specific DLSS release pins.
SR_PRESETS: tuple[str, ...] = ("Default", "E", "F", "J", "K", "L", "M")
GPU_INDEX_RANGE: tuple[int, int] = (0, 15)

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


def _require_in_range(value: object, name: str, bounds: tuple[float, float]) -> float:
    """A finite number inside the closed `bounds` interval."""
    number = _require_number(value, name)
    if not bounds[0] <= number <= bounds[1]:
        raise ValueError(f"{name} must be within {bounds}, got {value!r}")
    return number


def _require_int_in_range(value: object, name: str, bounds: tuple[int, int]) -> int:
    """An int inside the closed `bounds` interval (a bool is never an int)."""
    number = _require_int(value, name)
    if not bounds[0] <= number <= bounds[1]:
        raise ValueError(f"{name} must be within {bounds}, got {value!r}")
    return number


def _require_choice(value: object, name: str, choices: tuple[str, ...]) -> str:
    """A string that is one of `choices`."""
    selected = _require_str(value, name)
    if selected not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {selected!r}")
    return selected


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

    The advanced neural-rendering controls (`nr_style`, `nr_preset`,
    `nr_local_structure`, `nr_local_tone`, `nr_skin`, `nr_auto_mask`,
    `nr_ui_correction`) are read only by the `custom` profile; every built-in
    profile keeps its own fixed model fields. `nr_detail` and `nr_color` are the
    post-neural-rendering composite controls and apply whenever neural rendering
    runs, their defaults preserving the raw model output. `sr_preset` and
    `gpu_index` belong to the super-resolution and launch paths.
    """

    enable_super_resolution: bool = False
    sr_scale: float = 2.0
    enable_neural_rendering: bool = False
    nr_profile: str = "standard"
    nr_intensity: float = 1.0
    nr_style: str = "Cinematic"
    nr_preset: str = "Default"
    nr_local_structure: float = 1.0
    nr_local_tone: float = 1.0
    nr_skin: float = -1.0
    nr_detail: float = 1.0
    nr_color: float = 1.0
    nr_ui_correction: bool = False
    nr_auto_mask: bool = False
    sr_preset: str = "Default"
    gpu_index: int = 0
    enable_frame_interpolation: bool = False
    interpolation_factor: int = 2
    stage_order: str = STAGE_ORDER_DLSS_THEN_VFI

    def __post_init__(self) -> None:
        _require_bool(self.enable_super_resolution, "enable_super_resolution")
        scale = _require_number(self.sr_scale, "sr_scale")
        if scale not in SR_SCALES:
            raise ValueError(f"sr_scale must be one of {SR_SCALES}, got {self.sr_scale!r}")
        _require_bool(self.enable_neural_rendering, "enable_neural_rendering")
        profile = _require_choice(self.nr_profile, "nr_profile", NR_PROFILES)
        intensity = _require_in_range(self.nr_intensity, "nr_intensity", NR_INTENSITY_RANGE)
        # Every advanced control is validated even when neural rendering is off,
        # so the plan stays the one serializable description of the execution.
        style = _require_choice(self.nr_style, "nr_style", NR_STYLES)
        preset = _require_choice(self.nr_preset, "nr_preset", NR_PRESETS)
        structure = _require_in_range(
            self.nr_local_structure, "nr_local_structure", NR_LOCAL_STRUCTURE_RANGE
        )
        tone = _require_in_range(self.nr_local_tone, "nr_local_tone", NR_LOCAL_TONE_RANGE)
        skin = _require_in_range(self.nr_skin, "nr_skin", NR_SKIN_RANGE)
        detail = _require_in_range(self.nr_detail, "nr_detail", NR_DETAIL_RANGE)
        color = _require_in_range(self.nr_color, "nr_color", NR_COLOR_RANGE)
        _require_bool(self.nr_ui_correction, "nr_ui_correction")
        _require_bool(self.nr_auto_mask, "nr_auto_mask")
        sr_preset = _require_choice(self.sr_preset, "sr_preset", SR_PRESETS)
        gpu_index = _require_int_in_range(self.gpu_index, "gpu_index", GPU_INDEX_RANGE)
        _require_bool(self.enable_frame_interpolation, "enable_frame_interpolation")
        factor = _require_int(self.interpolation_factor, "interpolation_factor")
        if factor not in INTERPOLATION_FACTORS:
            raise ValueError(
                f"interpolation_factor must be one of {INTERPOLATION_FACTORS}, got {factor!r}"
            )
        order = _require_choice(self.stage_order, "stage_order", STAGE_ORDERS)
        # Normalize numbers so equal settings always produce equal plans.
        for name, value in (
            ("sr_scale", scale),
            ("nr_intensity", intensity),
            ("nr_style", style),
            ("nr_preset", preset),
            ("nr_local_structure", structure),
            ("nr_local_tone", tone),
            ("nr_skin", skin),
            ("nr_detail", detail),
            ("nr_color", color),
            ("sr_preset", sr_preset),
            ("gpu_index", gpu_index),
            ("stage_order", order),
        ):
            object.__setattr__(self, name, value)

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
