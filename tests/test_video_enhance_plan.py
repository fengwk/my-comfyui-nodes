from __future__ import annotations

import dataclasses
import unittest

from my_nodes.core.video_enhance.plan import (
    NR_PROFILES,
    SR_SCALES,
    STAGE_DLSS,
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


if __name__ == "__main__":
    unittest.main()
