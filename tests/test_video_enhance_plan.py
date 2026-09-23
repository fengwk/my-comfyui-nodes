from __future__ import annotations

import dataclasses
import unittest

from my_nodes.core.video_enhance.plan import (
    CUSTOM_NR_PROFILE,
    GPU_INDEX_RANGE,
    NR_COLOR_RANGE,
    NR_DETAIL_RANGE,
    NR_LOCAL_STRUCTURE_RANGE,
    NR_LOCAL_TONE_RANGE,
    NR_PRESETS,
    NR_PROFILES,
    NR_SKIN_RANGE,
    NR_STYLES,
    SR_PRESETS,
    SR_SCALES,
    STAGE_DLSS,
    STAGE_ORDER_DLSS_THEN_VFI,
    STAGE_ORDER_VFI_THEN_DLSS,
    STAGE_ORDERS,
    STAGE_VFI,
    VideoEnhancePlan,
)


class PlanValidationTests(unittest.TestCase):
    def test_defaults_are_a_pass_through_plan(self) -> None:
        plan = VideoEnhancePlan()
        self.assertEqual(plan.stages, ())
        self.assertTrue(plan.is_pass_through)
        self.assertFalse(plan.uses_dlss)
        self.assertFalse(plan.uses_frame_interpolation)

    def test_allowed_scale_values_are_accepted(self) -> None:
        for scale in SR_SCALES:
            with self.subTest(scale=scale):
                plan = VideoEnhancePlan(enable_super_resolution=True, sr_scale=scale)
                self.assertEqual(plan.sr_scale, float(scale))
        # 1.0 is native DLAA and still a DLSS stage, not a disabled no-op.
        dlaa = VideoEnhancePlan(enable_super_resolution=True, sr_scale=1.0)
        self.assertEqual(dlaa.stages, (STAGE_DLSS,))
        self.assertTrue(dlaa.uses_dlss)

    def test_disallowed_scale_is_rejected_even_when_disabled(self) -> None:
        # Disabled-stage parameters stay validated so they remain serializable.
        for scale in (1.25, 2.5, 4.0):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    VideoEnhancePlan(enable_super_resolution=False, sr_scale=scale)

    def test_disallowed_interpolation_factor_is_rejected_even_when_disabled(self) -> None:
        for factor in (0, 1, 3, 4):
            with self.subTest(factor=factor):
                with self.assertRaises(ValueError):
                    VideoEnhancePlan(enable_frame_interpolation=False, interpolation_factor=factor)

    def test_allowed_profiles_are_accepted(self) -> None:
        for profile in NR_PROFILES:
            with self.subTest(profile=profile):
                plan = VideoEnhancePlan(nr_profile=profile)
                self.assertEqual(plan.nr_profile, profile)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VideoEnhancePlan(enable_neural_rendering=True, nr_profile="cinematic")

    def test_custom_is_appended_after_the_builtin_profiles(self) -> None:
        # The built-in order is part of the node schema, so `custom` is appended
        # instead of being inserted anywhere.
        self.assertEqual(NR_PROFILES[:-1], ("light", "standard", "portrait", "detail"))
        self.assertEqual(NR_PROFILES[-1], CUSTOM_NR_PROFILE)
        self.assertEqual(VideoEnhancePlan(nr_profile="custom").nr_profile, "custom")

    def test_nr_intensity_range_boundaries(self) -> None:
        self.assertEqual(VideoEnhancePlan(nr_intensity=0.0).nr_intensity, 0.0)
        self.assertEqual(VideoEnhancePlan(nr_intensity=2.0).nr_intensity, 2.0)
        self.assertEqual(VideoEnhancePlan(nr_intensity=1).nr_intensity, 1.0)

    def test_nr_intensity_outside_range_or_not_finite_is_rejected(self) -> None:
        for intensity in (-0.1, 2.1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(intensity=intensity):
                with self.assertRaises(ValueError):
                    VideoEnhancePlan(nr_intensity=intensity)

    def test_wrong_types_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            VideoEnhancePlan(enable_super_resolution="yes")
        with self.assertRaises(TypeError):
            VideoEnhancePlan(sr_scale="2.0")
        with self.assertRaises(TypeError):
            VideoEnhancePlan(enable_neural_rendering=1)
        with self.assertRaises(TypeError):
            VideoEnhancePlan(nr_profile=2)
        with self.assertRaises(TypeError):
            VideoEnhancePlan(nr_intensity=True)
        with self.assertRaises(TypeError):
            VideoEnhancePlan(interpolation_factor=2.0)
        with self.assertRaises(TypeError):
            VideoEnhancePlan(interpolation_factor=True)

    def test_plan_is_immutable(self) -> None:
        plan = VideoEnhancePlan()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.enable_super_resolution = True


class PlanStageTests(unittest.TestCase):
    def test_stage_order_is_dlss_then_interpolation(self) -> None:
        plan = VideoEnhancePlan(
            enable_super_resolution=True,
            enable_neural_rendering=True,
            enable_frame_interpolation=True,
        )
        self.assertEqual(plan.stages, (STAGE_DLSS, STAGE_VFI))
        self.assertTrue(plan.uses_dlss)
        self.assertTrue(plan.uses_frame_interpolation)
        self.assertFalse(plan.is_pass_through)

    def test_super_resolution_alone_enables_the_dlss_stage(self) -> None:
        plan = VideoEnhancePlan(enable_super_resolution=True)
        self.assertEqual(plan.stages, (STAGE_DLSS,))
        self.assertFalse(plan.uses_frame_interpolation)

    def test_neural_rendering_alone_enables_the_dlss_stage(self) -> None:
        plan = VideoEnhancePlan(enable_neural_rendering=True)
        self.assertEqual(plan.stages, (STAGE_DLSS,))
        self.assertTrue(plan.uses_dlss)

    def test_interpolation_alone_enables_only_the_vfi_stage(self) -> None:
        plan = VideoEnhancePlan(enable_frame_interpolation=True)
        self.assertEqual(plan.stages, (STAGE_VFI,))
        self.assertFalse(plan.uses_dlss)

    def test_disabling_a_stage_removes_it_from_the_order(self) -> None:
        plan = VideoEnhancePlan(
            enable_super_resolution=False,
            enable_neural_rendering=True,
            enable_frame_interpolation=False,
        )
        self.assertEqual(plan.stages, (STAGE_DLSS,))
        self.assertNotIn(STAGE_VFI, plan.stages)

    def test_equal_settings_produce_equal_plans(self) -> None:
        self.assertEqual(
            VideoEnhancePlan(enable_super_resolution=True, sr_scale=2),
            VideoEnhancePlan(enable_super_resolution=True, sr_scale=2.0),
        )


class PlanStageOrderTests(unittest.TestCase):
    def test_legacy_default_order_is_dlss_then_vfi(self) -> None:
        self.assertEqual(VideoEnhancePlan().stage_order, STAGE_ORDER_DLSS_THEN_VFI)
        self.assertEqual(STAGE_ORDERS[0], STAGE_ORDER_DLSS_THEN_VFI)
        plan = VideoEnhancePlan(
            enable_super_resolution=True, enable_frame_interpolation=True
        )
        self.assertEqual(plan.stages, (STAGE_DLSS, STAGE_VFI))
        self.assertTrue(plan.uses_both_stages)

    def test_vfi_then_dlss_reverses_the_two_active_stages(self) -> None:
        plan = VideoEnhancePlan(
            enable_super_resolution=True,
            enable_frame_interpolation=True,
            stage_order=STAGE_ORDER_VFI_THEN_DLSS,
        )
        self.assertEqual(plan.stages, (STAGE_VFI, STAGE_DLSS))

    def test_stage_order_does_not_change_a_single_active_stage(self) -> None:
        for order in STAGE_ORDERS:
            with self.subTest(order=order):
                dlss_only = VideoEnhancePlan(enable_neural_rendering=True, stage_order=order)
                self.assertEqual(dlss_only.stages, (STAGE_DLSS,))
                self.assertFalse(dlss_only.uses_both_stages)
                vfi_only = VideoEnhancePlan(enable_frame_interpolation=True, stage_order=order)
                self.assertEqual(vfi_only.stages, (STAGE_VFI,))
        self.assertEqual(
            VideoEnhancePlan(enable_neural_rendering=True).stages,
            VideoEnhancePlan(
                enable_neural_rendering=True, stage_order=STAGE_ORDER_VFI_THEN_DLSS
            ).stages,
        )

    def test_disabled_stage_is_still_removed_from_the_requested_order(self) -> None:
        plan = VideoEnhancePlan(
            enable_frame_interpolation=True,
            stage_order=STAGE_ORDER_VFI_THEN_DLSS,
        )
        self.assertEqual(plan.stages, (STAGE_VFI,))
        self.assertNotIn(STAGE_DLSS, plan.stages)

    def test_unknown_stage_order_is_rejected_even_when_disabled(self) -> None:
        with self.assertRaises(ValueError):
            VideoEnhancePlan(stage_order="vfi_dlss_parallel")

    def test_wrong_stage_order_type_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            VideoEnhancePlan(stage_order=None)

    def test_stage_order_is_part_of_plan_equality(self) -> None:
        self.assertNotEqual(
            VideoEnhancePlan(
                enable_super_resolution=True,
                enable_frame_interpolation=True,
                stage_order=STAGE_ORDER_DLSS_THEN_VFI,
            ),
            VideoEnhancePlan(
                enable_super_resolution=True,
                enable_frame_interpolation=True,
                stage_order=STAGE_ORDER_VFI_THEN_DLSS,
            ),
        )


class AdvancedControlTests(unittest.TestCase):
    """The advanced DLSS controls are validated plan fields, always."""

    def test_defaults_keep_the_existing_behaviour(self) -> None:
        plan = VideoEnhancePlan()
        self.assertEqual(plan.nr_style, "Cinematic")
        self.assertEqual(plan.nr_preset, "Default")
        self.assertEqual(plan.nr_local_structure, 1.0)
        self.assertEqual(plan.nr_local_tone, 1.0)
        self.assertEqual(plan.nr_skin, -1.0)
        self.assertEqual(plan.nr_detail, 1.0)
        self.assertEqual(plan.nr_color, 1.0)
        self.assertIs(plan.nr_ui_correction, False)
        self.assertIs(plan.nr_auto_mask, False)
        self.assertEqual(plan.sr_preset, "Default")
        self.assertEqual(plan.gpu_index, 0)
        # A plan built without any of them equals one that spells them out.
        self.assertEqual(plan, VideoEnhancePlan(**{
            "nr_style": "Cinematic", "nr_preset": "Default", "nr_local_structure": 1.0,
            "nr_local_tone": 1.0, "nr_skin": -1.0, "nr_detail": 1.0, "nr_color": 1.0,
            "nr_ui_correction": False, "nr_auto_mask": False, "sr_preset": "Default",
            "gpu_index": 0,
        }))

    def test_choice_lists_are_the_documented_ones(self) -> None:
        self.assertEqual(NR_STYLES, ("Default", "Natural", "Cinematic"))
        self.assertEqual(NR_PRESETS, ("Default", "Preset 1", "Preset 2", "Preset 3"))
        self.assertEqual(SR_PRESETS, ("Default", "E", "F", "J", "K", "L", "M"))
        for style in NR_STYLES:
            self.assertEqual(VideoEnhancePlan(nr_style=style).nr_style, style)
        for preset in NR_PRESETS:
            self.assertEqual(VideoEnhancePlan(nr_preset=preset).nr_preset, preset)
        for sr_preset in SR_PRESETS:
            self.assertEqual(VideoEnhancePlan(sr_preset=sr_preset).sr_preset, sr_preset)

    def test_unknown_choices_are_rejected_even_when_disabled(self) -> None:
        for name, value in (
            ("nr_style", "cinematic"),
            ("nr_preset", "Preset 4"),
            ("sr_preset", "G"),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    VideoEnhancePlan(enable_neural_rendering=False, **{name: value})

    def test_numeric_ranges_are_enforced_even_when_disabled(self) -> None:
        ranges = (
            ("nr_local_structure", NR_LOCAL_STRUCTURE_RANGE),
            ("nr_local_tone", NR_LOCAL_TONE_RANGE),
            ("nr_skin", NR_SKIN_RANGE),
            ("nr_detail", NR_DETAIL_RANGE),
            ("nr_color", NR_COLOR_RANGE),
        )
        for name, bounds in ranges:
            low, high = bounds
            with self.subTest(name=name, value=low):
                # Both boundaries are values the user can really set.
                self.assertEqual(getattr(VideoEnhancePlan(**{name: low}), name), float(low))
            with self.subTest(name=name, value=high):
                self.assertEqual(getattr(VideoEnhancePlan(**{name: high}), name), float(high))
            for value in (low - 0.01, high + 0.01):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        VideoEnhancePlan(enable_neural_rendering=False, **{name: value})

    def test_gpu_index_range_is_enforced(self) -> None:
        self.assertEqual(GPU_INDEX_RANGE, (0, 15))
        self.assertEqual(VideoEnhancePlan(gpu_index=15).gpu_index, 15)
        for value in (-1, 16):
            with self.subTest(gpu_index=value):
                with self.assertRaises(ValueError):
                    VideoEnhancePlan(enable_super_resolution=False, gpu_index=value)

    def test_wrong_advanced_types_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            VideoEnhancePlan(nr_ui_correction=1)
        with self.assertRaises(TypeError):
            VideoEnhancePlan(nr_auto_mask=0)
        with self.assertRaises(TypeError):
            VideoEnhancePlan(nr_detail="1.0")
        with self.assertRaises(TypeError):
            VideoEnhancePlan(nr_style=2)
        with self.assertRaises(TypeError):
            VideoEnhancePlan(sr_preset=None)
        with self.assertRaises(TypeError):
            VideoEnhancePlan(gpu_index=1.0)
        with self.assertRaises(TypeError):
            VideoEnhancePlan(gpu_index=True)

    def test_advanced_numbers_are_normalized_for_plan_equality(self) -> None:
        left = VideoEnhancePlan(nr_detail=1, nr_color=1, gpu_index=0)
        right = VideoEnhancePlan(nr_detail=1.0, nr_color=1.0, gpu_index=0)
        self.assertEqual(left, right)
        self.assertIsInstance(left.nr_detail, float)
        # A different advanced value is a different plan, so it cannot be
        # silently dropped from a cache key or a comparison.
        self.assertNotEqual(left, VideoEnhancePlan(nr_detail=1.5))
        self.assertNotEqual(left, VideoEnhancePlan(gpu_index=1))


if __name__ == "__main__":
    unittest.main()
