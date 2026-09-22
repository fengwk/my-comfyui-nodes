"""UX neural-rendering profiles mapped onto DNR3 header fields.

These are local convenience presets, not NVIDIA official presets. The field
names match the DNR3 header (`style`, `preset`, `tone`, `structure`, `skin`,
`global_tone`, `automask`). The widget intensity (0..2) replaces the profile's
base intensity; the other fields stay fixed.

`ui_correction` is intentionally absent: the native bridge reads that switch
from `DLSS5NR_UI_CORRECTION`, not from the header.
"""

from __future__ import annotations

from dataclasses import dataclass

from my_nodes.core.video_enhance.plan import NR_PROFILES

# style / preset are the integer selectors the bridge forwards as
# DLSSNR.Style and DLSSNR.Hint.Render.Preset. Negative skin/global_tone leave
# those parameters at the model default (the bridge skips values below 0).
_PROFILE_FIELDS: dict[str, tuple[int, int, float, float, float, float, bool]] = {
    # (style, preset, tone, structure, skin, global_tone, automask)
    "light": (0, 0, 1.0, 0.8, -1.0, -1.0, False),
    "standard": (0, 0, 1.0, 1.0, -1.0, -1.0, False),
    "portrait": (1, 0, 1.0, 1.0, 1.0, -1.0, False),
    "detail": (0, 0, 1.0, 1.5, -1.0, -1.0, True),
}


@dataclass(frozen=True)
class NeuralRenderingSettings:
    """One documented mapping from a UX profile plus intensity to header fields."""

    profile: str
    style: int
    preset: int
    intensity: float
    tone: float
    structure: float
    skin: float
    global_tone: float
    automask: bool


def neural_rendering_settings(profile: str, intensity: float) -> NeuralRenderingSettings:
    """Map a UX profile and the widget intensity onto DNR3 neural-rendering fields.

    `intensity` is the user control (already validated to 0..2 by the plan). It
    does not rescale the other profile fields.
    """
    if profile not in NR_PROFILES:
        raise ValueError(f"nr_profile must be one of {NR_PROFILES}, got {profile!r}")
    style, preset, tone, structure, skin, global_tone, automask = _PROFILE_FIELDS[profile]
    return NeuralRenderingSettings(
        profile=profile,
        style=style,
        preset=preset,
        intensity=float(intensity),
        tone=tone,
        structure=structure,
        skin=skin,
        global_tone=global_tone,
        automask=automask,
    )
