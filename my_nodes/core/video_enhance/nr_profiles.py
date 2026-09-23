"""UX neural-rendering profiles mapped onto DNR3 header fields.

These are local convenience presets, not NVIDIA official presets. The field
names match the DNR3 header (`style`, `preset`, `tone`, `structure`, `skin`,
`global_tone`, `automask`). The widget intensity (0..2) replaces the profile's
base intensity; the other fields stay fixed.

The last profile, `custom`, is not a table entry: it maps the plan's advanced
controls onto the same fields, so `style`/`preset` become the integer model
selectors and local structure, local tone, skin, auto mask and UI correction come
from the plan. Global tone stays at the model default and is not exposed.

`ui_correction` travels both ways: it is a header field here and the native
bridge also reads the `DLSS5NR_UI_CORRECTION` environment value, which the stage
sets from the same resolved setting.
"""

from __future__ import annotations

from dataclasses import dataclass

from my_nodes.core.video_enhance.plan import (
    CUSTOM_NR_PROFILE,
    NR_PROFILES,
    VideoEnhancePlan,
)

# style / preset are the integer selectors the bridge forwards as
# DLSSNR.Style and DLSSNR.Hint.Render.Preset. Negative skin/global_tone leave
# those parameters at the model default (the bridge skips values below 0).
_PROFILE_FIELDS: dict[str, tuple[int, int, float, float, float, float, bool, bool]] = {
    # (style, preset, tone, structure, skin, global_tone, automask, ui_correction)
    "light": (0, 0, 1.0, 0.8, -1.0, -1.0, False, False),
    "standard": (0, 0, 1.0, 1.0, -1.0, -1.0, False, False),
    "portrait": (1, 0, 1.0, 1.0, 1.0, -1.0, False, False),
    "detail": (0, 0, 1.0, 1.5, -1.0, -1.0, True, False),
}

# Model selectors the `custom` profile resolves its string choices into.
STYLE_IDS: dict[str, int] = {"Default": 0, "Natural": 1, "Cinematic": 2}
PRESET_IDS: dict[str, int] = {"Default": 0, "Preset 1": 1, "Preset 2": 2, "Preset 3": 3}

# Not exposed as a widget: a negative value keeps the model's own global tone.
GLOBAL_TONE_UNSET: float = -1.0


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
    ui_correction: bool = False


def neural_rendering_settings(profile: str, intensity: float) -> NeuralRenderingSettings:
    """Map a built-in UX profile and the widget intensity onto DNR3 fields.

    `intensity` is the user control (already validated to 0..2 by the plan). It
    does not rescale the other profile fields. The `custom` profile has no fixed
    fields and is resolved from the plan by `plan_neural_rendering_settings`.
    """
    if profile == CUSTOM_NR_PROFILE:
        raise ValueError(
            "nr_profile 'custom' takes its model fields from the plan; "
            "use plan_neural_rendering_settings"
        )
    if profile not in NR_PROFILES:
        raise ValueError(f"nr_profile must be one of {NR_PROFILES}, got {profile!r}")
    style, preset, tone, structure, skin, global_tone, automask, ui_correction = (
        _PROFILE_FIELDS[profile]
    )
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
        ui_correction=ui_correction,
    )


def plan_neural_rendering_settings(plan: VideoEnhancePlan) -> NeuralRenderingSettings:
    """Resolve one plan's neural-rendering fields for the DNR3 header.

    A built-in profile resolves exactly as before and ignores every advanced
    field. `custom` maps the plan onto the model parameters: style and preset
    become their integer selectors, local structure and local tone become
    structure and tone, and skin, auto mask and UI correction are taken as is.
    """
    if plan.nr_profile != CUSTOM_NR_PROFILE:
        return neural_rendering_settings(plan.nr_profile, plan.nr_intensity)
    return NeuralRenderingSettings(
        profile=CUSTOM_NR_PROFILE,
        style=STYLE_IDS[plan.nr_style],
        preset=PRESET_IDS[plan.nr_preset],
        intensity=float(plan.nr_intensity),
        tone=plan.nr_local_tone,
        structure=plan.nr_local_structure,
        skin=plan.nr_skin,
        global_tone=GLOBAL_TONE_UNSET,
        automask=plan.nr_auto_mask,
        ui_correction=plan.nr_ui_correction,
    )
