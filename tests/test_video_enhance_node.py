"""MyVideoEnhance and MyDLSSRuntimeProbe: planning, routing and backend boundaries.

The DLSS path drives the existing real-pipe fake DNR3 worker. GIMM is represented
by stand-in node classes and a real torch.nn.Module; the S-Lab implementation is
not imported. Comfy's model manager is stubbed so the test records ModelPatcher
lifecycle calls, including cleanup after a BaseException. The node itself is
covered through the shared frame pipeline, whose stage order and disk staging are
asserted in `test_video_enhance_frame_pipeline.py`.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from my_nodes.core.video_enhance import FEATURE_NR, FEATURE_SR
from my_nodes.core.video_enhance.channel_order import select_channel_order, swap_rb
from my_nodes.core.video_enhance.dlss_stage import (
    SR_PRESET_IDS,
    FrameValidationError,
    apply_worker_environment,
    build_header,
    output_dimensions,
    prepare_frames,
    resolve_runtime_dir,
    run_dlss_stage,
    worker_environment,
)
from my_nodes.core.video_enhance.frame_pipeline import FrameSpec, PipelineResult
from my_nodes.core.video_enhance.gimm_vfi import (
    GIMM_FLOW_NAME,
    GIMM_MODEL_NAME,
    GimmVfiError,
    _clear_cublas_workspaces,
    _clear_gimm_backwarp_cache,
    _quiet_interpolate_method,
    clear_patcher_cache,
    interpolate_offline,
    iter_interpolate_offline,
    require_offline_weights,
    resolve_gimm_nodes,
)
from my_nodes.core.video_enhance.motion import MOTION_NONE, MotionGuideError, MotionGuides
from my_nodes.core.video_enhance.nr_profiles import (
    PRESET_IDS,
    STYLE_IDS,
    plan_neural_rendering_settings,
    neural_rendering_settings,
)
from my_nodes.core.video_enhance.plan import (
    CUSTOM_NR_PROFILE,
    NR_PRESETS,
    NR_PROFILES,
    NR_STYLES,
    SR_PRESETS,
    STAGE_ORDER_DLSS_THEN_VFI,
    STAGE_ORDER_VFI_THEN_DLSS,
    STAGE_ORDERS,
    VideoEnhancePlan,
)
from my_nodes.core.video_enhance.runtime import HostDriver, RuntimeFiles
from my_nodes.nodes.video_enhance import (
    ADVANCED_OPTIONAL,
    CUSTOM_PROFILE_ONLY,
    SPATIAL_LABELS,
    InsufficientRamError,
    MyDLSSRuntimeProbe,
    MyVideoEnhance,
    OUTPUT_RAM_FRACTION,
    _plan,
    spatial_scale,
)
from my_nodes.registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

from .video_enhance_fake_dnr3_worker import WATCHED_ENV
from .video_enhance_fixtures import (
    ADVANCED_CONTROLS,
    ADVANCED_NAMES,
    ADVANCED_VALUES,
    assert_process_gone,
    create_runtime_dir,
    fake_worker_command,
    read_report,
)


def _batch(*frames: np.ndarray) -> np.ndarray:
    return np.stack(frames, axis=0).astype(np.float32)


def _frame(width: int = 4, height: int = 6, value: float = 0.2) -> np.ndarray:
    frame = np.full((height, width, 3), value, dtype=np.float32)
    frame[..., 0] = value
    frame[..., 2] = 1.0 - value
    return frame


def _advanced_options(name: str) -> dict:
    """The classic widget options of one advanced control."""
    return ADVANCED_OPTIONAL[name][1]


class ProfileAndHeaderTests(unittest.TestCase):
    def test_profile_mapping_is_explicit_and_not_scaled_by_intensity(self) -> None:
        # The widget intensity replaces only the intensity field. Documented here
        # so a later change cannot silently retune the other UX fields.
        expected = {
            "light": dict(style=0, preset=0, tone=1.0, structure=0.8, skin=-1.0, global_tone=-1.0, automask=False),
            "standard": dict(style=0, preset=0, tone=1.0, structure=1.0, skin=-1.0, global_tone=-1.0, automask=False),
            "portrait": dict(style=1, preset=0, tone=1.0, structure=1.0, skin=1.0, global_tone=-1.0, automask=False),
            "detail": dict(style=0, preset=0, tone=1.0, structure=1.5, skin=-1.0, global_tone=-1.0, automask=True),
        }
        # `custom` is appended: it is not a table entry but the plan's own fields.
        self.assertEqual(tuple(expected) + (CUSTOM_NR_PROFILE,), NR_PROFILES)
        for profile, fields in expected.items():
            settings = neural_rendering_settings(profile, 0.5)
            for name, value in fields.items():
                self.assertEqual(getattr(settings, name), value, msg=f"{profile}.{name}")
            self.assertEqual(settings.intensity, 0.5)
            # No built-in profile turns the UI correction on by itself.
            self.assertIs(settings.ui_correction, False)

    def test_custom_profile_has_no_fixed_table_and_maps_every_plan_field(self) -> None:
        # `custom` cannot be resolved from the table, so a caller cannot confuse
        # it with a built-in preset.
        with self.assertRaises(ValueError):
            neural_rendering_settings(CUSTOM_NR_PROFILE, 1.0)
        plan = VideoEnhancePlan(
            enable_neural_rendering=True,
            nr_profile=CUSTOM_NR_PROFILE,
            nr_intensity=1.5,
            nr_style="Natural",
            nr_preset="Preset 3",
            nr_local_structure=0.5,
            nr_local_tone=1.25,
            nr_skin=0.75,
            nr_detail=1.6,
            nr_color=0.5,
            nr_ui_correction=True,
            nr_auto_mask=True,
            sr_preset="K",
        )
        settings = plan_neural_rendering_settings(plan)
        self.assertEqual(settings.profile, CUSTOM_NR_PROFILE)
        self.assertEqual(settings.style, STYLE_IDS["Natural"])
        self.assertEqual(settings.preset, PRESET_IDS["Preset 3"])
        self.assertEqual(settings.intensity, 1.5)
        self.assertEqual(settings.tone, 1.25)
        self.assertEqual(settings.structure, 0.5)
        self.assertEqual(settings.skin, 0.75)
        self.assertEqual(settings.automask, True)
        self.assertEqual(settings.ui_correction, True)
        # Global tone is not exposed: it keeps the model default.
        self.assertEqual(settings.global_tone, -1.0)

    def test_every_style_and_preset_choice_resolves_to_its_model_selector(self) -> None:
        expected_styles = {"Default": 0, "Natural": 1, "Cinematic": 2}
        expected_presets = {"Default": 0, "Preset 1": 1, "Preset 2": 2, "Preset 3": 3}
        self.assertEqual(STYLE_IDS, expected_styles)
        self.assertEqual(PRESET_IDS, expected_presets)
        self.assertEqual(NR_STYLES, tuple(expected_styles))
        self.assertEqual(NR_PRESETS, tuple(expected_presets))
        for style, style_id in expected_styles.items():
            for preset, preset_id in expected_presets.items():
                with self.subTest(style=style, preset=preset):
                    plan = VideoEnhancePlan(
                        nr_profile=CUSTOM_NR_PROFILE, nr_style=style, nr_preset=preset
                    )
                    plan_settings = plan_neural_rendering_settings(plan)
                    self.assertEqual(plan_settings.style, style_id)
                    self.assertEqual(plan_settings.preset, preset_id)
                    header = build_header(
                        VideoEnhancePlan(
                            enable_neural_rendering=True,
                            nr_profile=CUSTOM_NR_PROFILE,
                            nr_style=style,
                            nr_preset=preset,
                        ),
                        _batch(_frame()),
                    )
                    self.assertEqual(header.style, style_id)
                    self.assertEqual(header.preset, preset_id)

    def test_custom_header_carries_the_plan_fields(self) -> None:
        plan = VideoEnhancePlan(
            enable_super_resolution=True,
            enable_neural_rendering=True,
            sr_scale=2.0,
            nr_profile=CUSTOM_NR_PROFILE,
            nr_intensity=0.25,
            nr_style="Cinematic",
            nr_preset="Preset 2",
            nr_local_structure=0.5,
            nr_local_tone=1.75,
            nr_skin=0.25,
            nr_ui_correction=True,
            nr_auto_mask=True,
        )
        header = build_header(plan, _batch(_frame()))
        self.assertEqual(header.features, FEATURE_SR | FEATURE_NR)
        self.assertEqual(header.style, STYLE_IDS["Cinematic"])
        self.assertEqual(header.preset, PRESET_IDS["Preset 2"])
        self.assertEqual(header.intensity, 0.25)
        self.assertEqual(header.tone, 1.75)
        self.assertEqual(header.structure, 0.5)
        self.assertEqual(header.skin, 0.25)
        self.assertEqual(header.automask, True)
        self.assertEqual(header.ui_correction, True)
        self.assertEqual(header.global_tone, -1.0)

    def test_builtin_profiles_ignore_every_advanced_field(self) -> None:
        # An existing workflow never set them, and setting them must not move a
        # built-in profile's header either.
        base = VideoEnhancePlan(enable_neural_rendering=True, nr_profile="portrait")
        tweaked = VideoEnhancePlan(
            enable_neural_rendering=True,
            nr_profile="portrait",
            nr_style="Natural",
            nr_preset="Preset 1",
            nr_local_structure=0.25,
            nr_local_tone=0.25,
            nr_skin=1.75,
            nr_ui_correction=True,
            nr_auto_mask=True,
        )
        for name in ("style", "preset", "intensity", "tone", "structure", "skin",
                     "global_tone", "automask", "ui_correction"):
            with self.subTest(field=name):
                self.assertEqual(
                    getattr(build_header(base, _batch(_frame())), name),
                    getattr(build_header(tweaked, _batch(_frame())), name),
                )
        self.assertEqual(
            plan_neural_rendering_settings(tweaked),
            neural_rendering_settings("portrait", 1.0),
        )

    def test_scale_one_keeps_native_size_and_larger_scales_round_even(self) -> None:
        self.assertEqual(output_dimensions(5, 7, 1.0), (5, 7))
        # floor(n * scale + 0.5), then the next even number when odd.
        # 7 * 1.5 rounds to 11, then the next even edge is 12.
        self.assertEqual(output_dimensions(5, 7, 1.5), (8, 12))
        self.assertEqual(output_dimensions(4, 6, 2.0), (8, 12))

    def test_header_uses_the_profile_and_rejects_a_bad_aspect(self) -> None:
        plan = VideoEnhancePlan(enable_super_resolution=True, enable_neural_rendering=True, sr_scale=2.0, nr_profile="portrait", nr_intensity=1.25)
        header = build_header(plan, _batch(_frame()))
        self.assertEqual(header.features, FEATURE_SR | FEATURE_NR)
        self.assertEqual(header.style, 1)
        self.assertEqual(header.intensity, 1.25)
        self.assertEqual((header.output_width, header.output_height), (8, 12))
        odd = VideoEnhancePlan(enable_super_resolution=True, sr_scale=1.5)
        with self.assertRaises(Exception):
            # 5x7 at 1.5 becomes 8x10, which no longer matches the source aspect
            # closely enough for the header contract. That must fail before Wine.
            build_header(odd, _batch(_frame(width=5, height=7)))

    def test_prepare_frames_rejects_empty_and_non_finite_batches(self) -> None:
        with self.assertRaises(ValueError):
            prepare_frames(np.zeros((0, 4, 4, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            prepare_frames(np.array([[[[np.nan, 0.0, 0.0]]]], dtype=np.float32))
        staged = prepare_frames(_batch(_frame()))
        self.assertTrue(staged.flags["C_CONTIGUOUS"])


class _Touched:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def free(self) -> None:
        self.calls.append("free")

    def empty(self) -> None:
        self.calls.append("empty")

    def interrupt(self) -> None:
        self.calls.append("interrupt")


class DlssStageIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._pids: list[int] = []
        self.addCleanup(self._assert_gone)

    def _assert_gone(self) -> None:
        for pid in self._pids:
            assert_process_gone(pid)

    def _factory(self, features: int, report: Path):
        directory = create_runtime_dir(self.root / f"runtime-{features}", features)

        def factory(*, runtime_dir, features, wine_prefix):
            del runtime_dir, wine_prefix
            env = dict(os.environ)
            env["FAKE_DNR3_REPORT"] = str(report)
            return HostDriver.direct(fake_worker_command("ok"), runtime_dir=directory, features=features, env=env)

        return factory

    def test_one_worker_processes_the_batch_and_exits(self) -> None:
        report = self.root / "report.json"
        plan = VideoEnhancePlan(enable_super_resolution=True, sr_scale=2.0, nr_profile="detail", nr_intensity=0.4)
        touched = _Touched()
        seen: list[int] = []

        def progress(done: int, total: int) -> None:
            seen.append(done)
            self.assertEqual(total, 2)

        result = run_dlss_stage(
            plan,
            _batch(_frame(value=0.2), _frame(value=0.4)),
            runtime_dir="unused",
            wine_prefix="",
            channel_order="RGBA",
            motion_mode=MOTION_NONE,
            scene_cut_threshold=0.2,
            progress=progress,
            interrupt=touched.interrupt,
            driver_factory=self._factory(FEATURE_SR, report),
            memory_hooks=(touched.free, touched.empty),
        )
        self.assertEqual(result.frames.shape, (2, 12, 8, 3))
        self.assertEqual(seen, [1, 2])
        self.assertEqual(touched.calls[:2], ["free", "empty"])
        wire = read_report(report)
        self._pids.append(int(wire["pid"]))
        self.assertTrue(wire["ended"])
        self.assertTrue(all(frame["reset"] for frame in wire["frames"]))
        self.assertEqual(wire["header"]["style"], 0)
        self.assertEqual(wire["header"]["structure"], 1.5)
        self.assertEqual(wire["header"]["automask"], 1)
        self.assertAlmostEqual(wire["header"]["intensity"], 0.4)

    def test_auto_channel_order_is_chosen_once_for_the_batch(self) -> None:
        # The fake worker echoes RGB. A swapped first frame must not be required;
        # an explicit BGRA request swaps every frame the same way.
        report = self.root / "bgra.json"
        plan = VideoEnhancePlan(enable_neural_rendering=True)
        result = run_dlss_stage(
            plan,
            _batch(_frame(value=0.1), _frame(value=0.3)),
            runtime_dir="unused",
            wine_prefix="",
            channel_order="BGRA",
            motion_mode=MOTION_NONE,
            scene_cut_threshold=0.2,
            driver_factory=self._factory(FEATURE_NR, report),
            memory_hooks=(lambda: None, lambda: None),
        )
        self.assertEqual(result.channel_order, "BGRA")
        np.testing.assert_allclose(result.frames[0], swap_rb(_frame(value=0.1)))
        np.testing.assert_allclose(result.frames[1], swap_rb(_frame(value=0.3)))
        self._pids.append(int(read_report(report)["pid"]))

    def test_the_custom_factory_driver_gets_the_explicit_launch_environment(self) -> None:
        # The child process itself reports the six values: a stale variable of
        # the same name in this process must not survive into the worker.
        report = self.root / "custom-env.json"
        plan = VideoEnhancePlan(
            enable_neural_rendering=True,
            nr_profile=CUSTOM_NR_PROFILE,
            nr_intensity=1.25,
            nr_style="Natural",
            nr_preset="Preset 1",
            nr_local_tone=1.5,
            nr_skin=0.5,
            nr_detail=1.5,
            nr_color=0.5,
            nr_ui_correction=True,
            nr_auto_mask=True,
            sr_preset="L",
            gpu_index=3,
        )
        with mock.patch.dict(os.environ, {name: "stale" for name in WATCHED_ENV}):
            result = run_dlss_stage(
                plan,
                _batch(_frame(value=0.2), _frame(value=0.4)),
                runtime_dir="unused",
                wine_prefix="",
                channel_order="auto",
                motion_mode=MOTION_NONE,
                scene_cut_threshold=0.2,
                driver_factory=self._factory(FEATURE_NR, report),
                memory_hooks=(lambda: None, lambda: None),
            )
        self.assertEqual(result.frames.shape[0], 2)
        wire = read_report(report)
        self._pids.append(int(wire["pid"]))
        self.assertEqual(
            wire["env"],
            {
                "DLSS5NR_UI_CORRECTION": "1",
                "DLSS5NR_DETAIL": "1.5",
                "DLSS5NR_COLOR": "0.5",
                "DLSS5NR_SR_PRESET": str(SR_PRESET_IDS["L"]),
                "DLSS5NR_GPU_INDEX": "3",
                "DLSS5NR_CHANNEL_ORDER": "auto",
            },
        )
        # The header carries the same resolved UI correction as the environment.
        self.assertEqual(wire["header"]["ui_correction"], 1)
        self.assertEqual(wire["header"]["style"], STYLE_IDS["Natural"])
        self.assertEqual(wire["header"]["preset"], PRESET_IDS["Preset 1"])
        self.assertEqual(wire["header"]["structure"], 1.0)

    def test_the_builtin_driver_is_asked_for_the_selected_gpu_and_the_plan_environment(
        self,
    ) -> None:
        # The built-in path is the only one that may receive the new keyword, and
        # mocking the factory proves the argument without spawning Wine.
        report = self.root / "builtin.json"
        plan = VideoEnhancePlan(enable_neural_rendering=True, gpu_index=2)
        directory = create_runtime_dir(self.root / "runtime-builtin", FEATURE_NR)
        seen: list[dict] = []

        def fake_wine(**kwargs):
            seen.append(kwargs)
            env = dict(os.environ)
            env["FAKE_DNR3_REPORT"] = str(report)
            return HostDriver.direct(
                fake_worker_command("ok"),
                runtime_dir=kwargs["runtime_dir"],
                features=kwargs["features"],
                env=env,
            )

        with mock.patch.object(HostDriver, "wine", side_effect=fake_wine):
            run_dlss_stage(
                plan,
                _batch(_frame()),
                runtime_dir=str(directory),
                wine_prefix="",
                channel_order="RGBA",
                motion_mode=MOTION_NONE,
                scene_cut_threshold=0.2,
                memory_hooks=(lambda: None, lambda: None),
            )
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["gpu_index"], 2)
        self.assertEqual(seen[0]["features"], FEATURE_NR)
        self.assertEqual(seen[0]["runtime_dir"], str(directory))
        wire = read_report(report)
        self._pids.append(int(wire["pid"]))
        self.assertEqual(wire["env"]["DLSS5NR_GPU_INDEX"], "2")
        self.assertEqual(wire["env"]["DLSS5NR_CHANNEL_ORDER"], "RGBA")

    def test_a_custom_factory_that_only_knows_the_old_keywords_still_works(self) -> None:
        # The documented signature of a custom factory is unchanged: three
        # keywords and no gpu_index. Passing a fourth keyword would raise a
        # TypeError here instead of quietly changing the factory contract.
        report = self.root / "old-factory.json"
        plan = VideoEnhancePlan(enable_neural_rendering=True, gpu_index=1)
        directory = create_runtime_dir(self.root / "runtime-old-factory", FEATURE_NR)
        calls: list[str] = []

        def factory(*, runtime_dir, features, wine_prefix):
            calls.append(wine_prefix or "")
            env = dict(os.environ)
            env["FAKE_DNR3_REPORT"] = str(report)
            return HostDriver.direct(
                fake_worker_command("ok"), runtime_dir=runtime_dir, features=features, env=env
            )

        run_dlss_stage(
            plan,
            _batch(_frame()),
            runtime_dir=str(directory),
            wine_prefix="",
            channel_order="auto",
            motion_mode=MOTION_NONE,
            scene_cut_threshold=0.2,
            driver_factory=factory,
            memory_hooks=(lambda: None, lambda: None),
        )
        self.assertEqual(calls, [""])
        wire = read_report(report)
        self._pids.append(int(wire["pid"]))
        # The GPU index still reaches the worker, through the overlaid driver env.
        self.assertEqual(wire["env"]["DLSS5NR_GPU_INDEX"], "1")

    def test_worker_is_reaped_when_the_interrupt_raises_base_exception(self) -> None:
        report = self.root / "interrupt.json"
        plan = VideoEnhancePlan(enable_super_resolution=True, sr_scale=2.0)

        class Cancel(BaseException):
            pass

        def interrupt() -> None:
            if report.is_file() and read_report(report).get("pid"):
                raise Cancel()

        with self.assertRaises(Cancel):
            run_dlss_stage(
                plan,
                _batch(_frame(), _frame(value=0.5)),
                runtime_dir="unused",
                wine_prefix="",
                channel_order="RGBA",
                motion_mode=MOTION_NONE,
                scene_cut_threshold=0.2,
                interrupt=interrupt,
                driver_factory=self._factory(FEATURE_SR, report),
                memory_hooks=(lambda: None, lambda: None),
            )
        wire = read_report(report)
        self._pids.append(int(wire["pid"]))

    def test_optical_flow_without_cv2_fails_before_a_worker(self) -> None:
        plan = VideoEnhancePlan(enable_super_resolution=True, sr_scale=2.0)
        started = []

        def factory(**_kwargs):
            started.append(True)
            raise AssertionError("worker must not start")

        with mock.patch(
            "my_nodes.core.video_enhance.motion._import_cv2",
            side_effect=MotionGuideError("cv2 is not installed"),
        ):
            with self.assertRaises(MotionGuideError) as raised:
                run_dlss_stage(
                    plan,
                    _batch(_frame(), _frame(value=0.6)),
                    runtime_dir="unused",
                    wine_prefix="",
                    channel_order="auto",
                    motion_mode="optical_flow",
                    scene_cut_threshold=0.2,
                    driver_factory=factory,
                    memory_hooks=(lambda: None, lambda: None),
                )
        self.assertIn("cv2", str(raised.exception))
        self.assertEqual(started, [])

    def test_runtime_dir_resolution_order(self) -> None:
        self.assertEqual(resolve_runtime_dir("  /explicit  ", env={}, models_dir="/models"), "/explicit")
        self.assertEqual(
            resolve_runtime_dir("", env={"DLSS5_RUNTIME_DIR": "/from-env"}, models_dir="/models"),
            "/from-env",
        )
        self.assertEqual(resolve_runtime_dir("  ", env={}, models_dir="/models"), os.path.join("/models", "dlss5"))


class WorkerEnvironmentTests(unittest.TestCase):
    """The launch environment is explicit, plan-derived and overlay-only."""

    def _driver(self, env: dict[str, str]) -> HostDriver:
        """A frozen driver with the given environment, without touching a disk."""
        files = RuntimeFiles(
            features=FEATURE_NR,
            directory=Path("/runtime"),
            core=Path("/runtime/_nvngx.dll"),
            sr=None,
            nr=Path("/runtime/nvngx_dlssnr.dll"),
            nr_name="nvngx_dlssnr.dll",
        )
        return HostDriver(files=files, command=("/runtime/host.exe",), env=env)

    def test_defaults_are_the_documented_ones(self) -> None:
        # A workflow that never set a new widget keeps the raw model output: the
        # values below are exactly what the old build's child environment implied.
        self.assertEqual(
            worker_environment(VideoEnhancePlan(), "auto"),
            {
                "DLSS5NR_UI_CORRECTION": "0",
                "DLSS5NR_DETAIL": "1.0",
                "DLSS5NR_COLOR": "1.0",
                "DLSS5NR_SR_PRESET": "0",
                "DLSS5NR_GPU_INDEX": "0",
                "DLSS5NR_CHANNEL_ORDER": "auto",
            },
        )

    def test_every_plan_control_reaches_its_own_variable(self) -> None:
        plan = VideoEnhancePlan(
            nr_profile=CUSTOM_NR_PROFILE,
            nr_detail=1.5,
            nr_color=0.5,
            nr_ui_correction=True,
            sr_preset="M",
            gpu_index=7,
        )
        self.assertEqual(
            worker_environment(plan, "BGRA"),
            {
                "DLSS5NR_UI_CORRECTION": "1",
                "DLSS5NR_DETAIL": "1.5",
                "DLSS5NR_COLOR": "0.5",
                "DLSS5NR_SR_PRESET": "13",
                "DLSS5NR_GPU_INDEX": "7",
                "DLSS5NR_CHANNEL_ORDER": "BGRA",
            },
        )
        # Detail and color apply to every profile, not only to `custom`.
        builtin = VideoEnhancePlan(nr_detail=0.5, nr_color=0.25)
        self.assertEqual(worker_environment(builtin, "auto")["DLSS5NR_DETAIL"], "0.5")
        self.assertEqual(worker_environment(builtin, "auto")["DLSS5NR_COLOR"], "0.25")
        # Only `custom` can turn the UI correction on; the built-in profiles
        # ignore the field, exactly like the header does.
        self.assertEqual(worker_environment(VideoEnhancePlan(nr_ui_correction=True), "auto")[
            "DLSS5NR_UI_CORRECTION"
        ], "0")

    def test_the_sr_preset_ids_cover_the_offered_choices(self) -> None:
        # The mapping is the wire contract with the native bridge, so a new
        # choice cannot be added without deciding its model selector.
        self.assertEqual(tuple(SR_PRESET_IDS), SR_PRESETS)
        self.assertEqual(
            SR_PRESET_IDS,
            {"Default": 0, "E": 5, "F": 6, "J": 10, "K": 11, "L": 12, "M": 13},
        )
        for preset, preset_id in SR_PRESET_IDS.items():
            with self.subTest(preset=preset):
                plan = VideoEnhancePlan(sr_preset=preset)
                self.assertEqual(worker_environment(plan, "auto")["DLSS5NR_SR_PRESET"], str(preset_id))

    def test_an_unknown_channel_order_fails_before_a_launch(self) -> None:
        with self.assertRaises(FrameValidationError):
            worker_environment(VideoEnhancePlan(), "bgra")

    def test_equal_plans_write_equal_environments(self) -> None:
        self.assertEqual(
            worker_environment(VideoEnhancePlan(nr_detail=1, nr_color=1), "auto"),
            worker_environment(VideoEnhancePlan(nr_detail=1.0, nr_color=1.0), "auto"),
        )

    def test_stale_parent_values_are_replaced_and_the_rest_is_kept(self) -> None:
        stale = {name: "stale" for name in WATCHED_ENV}
        stale.update({"DISPLAY": ":99", "PATH": "/usr/bin", "WINEPREFIX": "/prefix"})
        driver = apply_worker_environment(
            self._driver(stale),
            VideoEnhancePlan(nr_detail=1.25, sr_preset="J"),
            "RGBA",
        )
        self.assertEqual(driver.env["DLSS5NR_DETAIL"], "1.25")
        self.assertEqual(driver.env["DLSS5NR_SR_PRESET"], "10")
        self.assertEqual(driver.env["DLSS5NR_CHANNEL_ORDER"], "RGBA")
        self.assertNotIn("stale", set(driver.env.values()))
        # The Wine setup and every unrelated variable survive untouched.
        self.assertEqual(driver.env["DISPLAY"], ":99")
        self.assertEqual(driver.env["PATH"], "/usr/bin")
        self.assertEqual(driver.env["WINEPREFIX"], "/prefix")
        # Only the environment changed: the driver stays the same frozen object
        # with the same command and runtime files.
        self.assertEqual(driver.command, ("/runtime/host.exe",))
        self.assertEqual(driver.files, self._driver({}).files)
        self.assertEqual(self._driver(stale).env, stale)


class MotionResetTests(unittest.TestCase):
    def test_none_resets_every_frame_without_importing_cv2(self) -> None:
        guides = MotionGuides(MOTION_NONE, 0.2)
        with mock.patch.dict("sys.modules", {"cv2": None}):
            first = guides.guide(_frame())
            second = guides.guide(_frame(value=0.9))
        self.assertTrue(first.reset and second.reset)
        self.assertIsNone(first.motion)

    def test_scene_cut_resets_and_drops_motion(self) -> None:
        class _Flow:
            def calc(self, previous, current, _none):
                del previous, current
                return np.zeros((6, 4, 2), dtype=np.float32)

        class _Cv2:
            DISOPTICAL_FLOW_PRESET_FAST = 1

            @staticmethod
            def DISOpticalFlow_create(_preset):
                return _Flow()

        guides = MotionGuides("optical_flow", scene_cut_threshold=0.05)
        with mock.patch.dict("sys.modules", {"cv2": _Cv2}):
            self.assertTrue(guides.guide(_frame(value=0.0)).reset)
            steady = guides.guide(_frame(value=0.01))
            cut = guides.guide(_frame(value=0.9))
        self.assertFalse(steady.reset)
        self.assertIsNotNone(steady.motion)
        self.assertTrue(cut.reset)
        self.assertIsNone(cut.motion)

    def test_optical_flow_maps_current_pixels_back_to_the_previous_frame(self) -> None:
        calls: list[tuple[np.ndarray, np.ndarray]] = []

        class _Flow:
            def calc(self, source, target, _none):
                calls.append((source.copy(), target.copy()))
                return np.zeros((6, 4, 2), dtype=np.float32)

        class _Cv2:
            DISOPTICAL_FLOW_PRESET_FAST = 1

            @staticmethod
            def DISOpticalFlow_create(_preset):
                return _Flow()

        guides = MotionGuides("optical_flow", scene_cut_threshold=0.0)
        previous = _frame(value=0.1)
        current = _frame(value=0.3)
        with mock.patch.dict("sys.modules", {"cv2": _Cv2}):
            self.assertTrue(guides.guide(previous).reset)
            guide = guides.guide(current)

        # OpenCV maps its first image into its second; this ordering therefore
        # proves the NGX-required current -> previous reprojection direction.
        current_gray = int((0.299 * 0.3 + 0.587 * 0.3 + 0.114 * 0.7) * 255)
        previous_gray = int((0.299 * 0.1 + 0.587 * 0.1 + 0.114 * 0.9) * 255)
        np.testing.assert_array_equal(calls[0][0], np.full((6, 4), current_gray, dtype=np.uint8))
        np.testing.assert_array_equal(calls[0][1], np.full((6, 4), previous_gray, dtype=np.uint8))
        self.assertFalse(guide.reset)


def _load_module():
    import torch

    module = torch.nn.Linear(1, 1)
    module.dtype = torch.float32
    module.flow_estimator = torch.nn.Linear(1, 1)
    return module


class _Loader:
    def loadmodel(self, model, precision="fp32", torch_compile=False):
        del model, precision, torch_compile
        return (_load_module(),)


class _Interpolator:
    calls: list[dict] = []

    def interpolate(self, module, images, ds_factor, interpolation_factor, seed, output_flows=False):
        import torch

        if not isinstance(module, torch.nn.Module):
            raise AssertionError(f"interpolator received {type(module).__name__}, not a real module")
        type(self).calls.append(
            {
                "device": str(module.device),
                "factor": interpolation_factor,
                "seed": seed,
                "flows": output_flows,
                "ds": ds_factor,
                "count": int(images.shape[0]),
            }
        )
        left = images[0]
        right = images[1]
        middle = (left + right) / 2
        return (torch.stack((left, middle, right)), torch.zeros(1))


class _Memory:
    def __init__(self) -> None:
        self.loaded: list[object] = []
        self.unloaded: list[object] = []
        self.freed: list[tuple[float, str]] = []
        self.events: list[str] = []
        self.devices_while_loaded: list[str] = []
        self.caches = 0
        self.fail_on_load = False
        self.fail_on_pair: int | None = None
        self.pairs = 0

    def free_memory(self, memory_required, device):
        self.events.append("free")
        self.freed.append((float(memory_required), str(device)))
        return []

    def load_models_gpu(self, models, memory_required=0, force_full_load=False):
        del memory_required
        self.events.append("load")
        self.loaded.append((models[0], force_full_load))
        if self.fail_on_load:
            raise RuntimeError("simulated model load failure")
        import torch

        loaded = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        models[0].model.to(loaded)
        models[0].model.device = loaded
        self.devices_while_loaded.append(str(next(models[0].model.parameters()).device))

    def unload_model_and_clones(self, patcher):
        import torch

        self.events.append("unload")
        patcher.model.to(torch.device("cpu"))
        self.unloaded.append(patcher)

    def soft_empty_cache(self, force=False):
        del force
        self.events.append("cache")
        self.caches += 1

    def get_torch_device(self):
        import torch

        return torch.device("cpu")

    def throw_exception_if_processing_interrupted(self) -> None:
        self.pairs += 1
        if self.fail_on_pair is not None and self.pairs >= self.fail_on_pair:
            raise _Cancel()


class _Cancel(BaseException):
    pass


class GimmLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
            from comfy.model_patcher import ModelPatcher  # noqa: F401
        except ImportError:
            self.skipTest("GIMM lifecycle needs the ComfyUI interpreter (torch and ModelPatcher)")
        clear_patcher_cache()
        self.addCleanup(clear_patcher_cache)
        _Interpolator.calls = []
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.models = Path(self._tmp.name)
        directory = self.models / "interpolation" / "gimm-vfi"
        directory.mkdir(parents=True)
        (directory / GIMM_MODEL_NAME).write_bytes(b"gimm")
        (directory / GIMM_FLOW_NAME).write_bytes(b"raft")

    def test_missing_weights_name_the_files_and_do_not_download(self) -> None:
        (self.models / "interpolation" / "gimm-vfi" / GIMM_FLOW_NAME).unlink()
        with self.assertRaises(GimmVfiError) as raised:
            require_offline_weights(self.models)
        self.assertIn(GIMM_FLOW_NAME, str(raised.exception))
        self.assertIn("does not download", str(raised.exception))

    def test_missing_plugin_names_the_install_path(self) -> None:
        with self.assertRaises(GimmVfiError) as raised:
            resolve_gimm_nodes({}, plugin_path="/opt/custom_nodes/ComfyUI-GIMM-VFI")
        self.assertIn("/opt/custom_nodes/ComfyUI-GIMM-VFI", str(raised.exception))

    def test_resolution_keyed_plugin_cuda_cache_is_cleared(self) -> None:
        cache = {"cuda:0-(1,2,256,256)": object()}

        def template():
            pass

        warp = types.FunctionType(template.__code__, {"backwarp_tenGrid": cache})
        warp_w_mask = types.FunctionType(template.__code__, {"warp": warp})
        module = types.SimpleNamespace(warp_w_mask=warp_w_mask)
        _clear_gimm_backwarp_cache(module)
        self.assertEqual(cache, {})

    def test_cublas_workspace_clear_is_version_guarded(self) -> None:
        # RAFT's correlation matmul creates a persistent cuBLAS workspace. This
        # proves cleanup invokes the supported hook but tolerates older Torch.
        calls = []
        torch_module = types.SimpleNamespace(
            _C=types.SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append("clear"))
        )
        _clear_cublas_workspaces(torch_module)
        _clear_cublas_workspaces(types.SimpleNamespace(_C=types.SimpleNamespace()))
        self.assertEqual(calls, ["clear"])

    def _mappings(self, memory: _Memory) -> dict:
        class _RecordingLoader(_Loader):
            def loadmodel(self, model, precision="fp32", torch_compile=False):
                memory.events.append("model_loader")
                return super().loadmodel(model, precision, torch_compile)

        return {
            "DownloadAndLoadGIMMVFIModel": _RecordingLoader,
            "GIMMVFI_interpolate": _Interpolator,
        }

    @contextlib.contextmanager
    def _comfy(self, memory: _Memory):
        """Patch the external model manager and caches the stages talk to."""
        import sys
        import comfy.model_management as mm

        # The production functions import model_management locally, so patch the
        # module object they receive rather than the caller's name binding.
        with mock.patch.object(mm, "free_memory", memory.free_memory), \
             mock.patch.object(mm, "load_models_gpu", memory.load_models_gpu), \
             mock.patch.object(mm, "unload_model_and_clones", memory.unload_model_and_clones), \
             mock.patch.object(mm, "soft_empty_cache", memory.soft_empty_cache), \
             mock.patch.object(mm, "get_torch_device", memory.get_torch_device), \
             mock.patch(
                 "my_nodes.core.video_enhance.gimm_vfi._clear_cublas_workspaces",
                 side_effect=lambda _torch: memory.events.append("cublas"),
             ), \
             mock.patch.dict(sys.modules, {"comfy.model_management": mm}):
            yield

    def _run(self, frames: np.ndarray, memory: _Memory):
        import torch
        from comfy.model_patcher import ModelPatcher

        mappings = self._mappings(memory)
        with self._comfy(memory):
            output = interpolate_offline(
                frames,
                precision="fp32",
                ds_factor=1.0,
                models_dir=self.models,
                node_mappings=mappings,
                load_device=torch.device("cpu"),
                interrupt=memory.throw_exception_if_processing_interrupted,
            )
        self.assertIsInstance(memory.loaded[0][0], ModelPatcher)
        self.assertIsInstance(memory.loaded[0][0].model, torch.nn.Module)
        return output

    def _open_stream(self, frames: np.ndarray, memory: _Memory, count: int | None = None):
        """Open the incremental stage under the same patched model manager."""
        import torch

        return iter_interpolate_offline(
            frames,
            int(frames.shape[0]) if count is None else count,
            precision="fp32",
            ds_factor=1.0,
            models_dir=self.models,
            node_mappings=self._mappings(memory),
            load_device=torch.device("cpu"),
            interrupt=memory.throw_exception_if_processing_interrupted,
        )

    def _drain(self, frames: np.ndarray, memory: _Memory, count: int | None = None) -> np.ndarray:
        with self._comfy(memory):
            stream = self._open_stream(frames, memory, count)
            collected = []
            try:
                for frame in stream:
                    collected.append(np.array(frame))
            finally:
                stream.close()
        return np.stack(collected) if collected else np.empty((0, frames.shape[1], frames.shape[2], 3), dtype=np.float32)

    def test_the_incremental_stage_returns_what_the_wrapper_returns(self) -> None:
        # The wrapper is a collector over the iterator: same frames, same order.
        frames = _batch(_frame(value=0.0), _frame(value=0.4), _frame(value=0.8))
        wrapper = self._run(frames, _Memory())
        incremental = self._drain(frames, _Memory())
        np.testing.assert_array_equal(incremental, wrapper)
        self.assertEqual(incremental.dtype, np.float32)

    def test_the_incremental_stage_yields_frames_while_the_model_is_loaded(self) -> None:
        # The streaming contract: a frame arrives before the next pair is built,
        # and closing the abandoned stage still unloads the model.
        frames = _batch(_frame(value=0.0), _frame(value=0.4), _frame(value=0.8))
        memory = _Memory()
        with self._comfy(memory):
            stream = self._open_stream(frames, memory)
            try:
                first = np.array(next(stream))
                self.assertIn("load", memory.events)
                self.assertNotIn("unload", memory.events)
                second = np.array(next(stream))
                self.assertNotIn("unload", memory.events)
            finally:
                stream.close()
        np.testing.assert_allclose(first, frames[0])
        np.testing.assert_allclose(second, (frames[0] + frames[1]) / 2)
        self.assertIn("unload", memory.events)
        self.assertEqual(len(memory.unloaded), 1)

    def test_the_incremental_stage_rejects_a_short_source_and_unloads(self) -> None:
        frames = _batch(_frame(value=0.0), _frame(value=0.4), _frame(value=0.8))
        memory = _Memory()
        with self.assertRaises(GimmVfiError) as raised:
            self._drain(frames, memory, count=4)
        self.assertIn("ended early", str(raised.exception))
        self.assertEqual(len(memory.unloaded), 1)
        self.assertEqual(memory.caches, 2)

    def test_pairs_are_assembled_without_duplicate_boundaries(self) -> None:
        memory = _Memory()
        frames = _batch(_frame(value=0.0), _frame(value=0.4), _frame(value=0.8))
        output = self._run(frames, memory)
        self.assertEqual(output.shape[0], 5)
        np.testing.assert_allclose(output[0], frames[0])
        np.testing.assert_allclose(output[2], frames[1])
        np.testing.assert_allclose(output[4], frames[2])
        np.testing.assert_allclose(output[1], (frames[0] + frames[1]) / 2)
        self.assertEqual(_Interpolator.calls[0]["factor"], 2)
        self.assertEqual(_Interpolator.calls[0]["seed"], 0)
        self.assertFalse(_Interpolator.calls[0]["flows"])
        self.assertEqual(len(memory.unloaded), 1)
        self.assertEqual(memory.caches, 2)
        self.assertEqual(memory.freed, [(1e30, "cpu")])
        self.assertEqual(
            memory.events,
            ["free", "cache", "model_loader", "load", "unload", "cublas", "cache"],
        )
        self.assertTrue(memory.devices_while_loaded[0].startswith("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        self.assertEqual(str(next(memory.unloaded[0].model.parameters()).device), "cpu")
        self.assertTrue(memory.loaded[0][1])

    def test_plugin_progress_is_call_local_while_original_method_stays_noisy(self) -> None:
        # This models the installed method's ProgressBar/tqdm globals. A concurrent
        # original invocation proves those globals remain noisy even while the
        # production call path is using its quiet clone.
        import threading

        import torch

        progress_events: list[tuple[str, int]] = []
        tqdm_calls: list[bool] = []
        globals_intact: list[bool] = []
        bindings: list[tuple[object, object, object, bool]] = []
        closure_value = object()
        positional_default = object()
        keyword_default = object()

        class _NoisyProgressBar:
            def __init__(self, total) -> None:
                progress_events.append(("init", int(total)))

            def update(self, amount) -> None:
                progress_events.append(("update", int(amount)))

        def noisy_tqdm(iterable, *_args, disable=False, **_kwargs):
            tqdm_calls.append(bool(disable))
            return iterable

        def make_template(captured):
            def interpolate(
                self,
                module,
                images,
                ds_factor,
                interpolation_factor,
                seed,
                output_flows=False,
                tag=positional_default,
                *,
                token=keyword_default,
            ):
                del module, ds_factor, seed
                bindings.append((captured, tag, token, self.invoke_original))
                pbar = ProgressBar(images.shape[0] - 1)  # noqa: F821
                if self.invoke_original:
                    invoke_original(images, interpolation_factor, output_flows)  # noqa: F821
                for _index in tqdm(range(images.shape[0] - 1)):  # noqa: F821
                    pbar.update(1)
                left, right = images
                return (torch.stack((left, (left + right) / 2, right)), torch.zeros(1))

            return interpolate

        template = make_template(closure_value)
        plugin_globals = template.__globals__.copy()
        plugin_globals.update(ProgressBar=_NoisyProgressBar, tqdm=noisy_tqdm)
        plugin_function = types.FunctionType(
            template.__code__,
            plugin_globals,
            name=template.__name__,
            argdefs=template.__defaults__,
            closure=template.__closure__,
        )
        plugin_function.__kwdefaults__ = template.__kwdefaults__
        plugin_function.__annotations__ = template.__annotations__
        metadata = object()
        plugin_function.test_metadata = metadata
        thread_errors: list[BaseException] = []

        class _NoisyInterpolator:
            interpolate = plugin_function

            def __init__(self) -> None:
                self.invoke_original = True

        def invoke_original(images, factor, output_flows) -> None:
            def run() -> None:
                try:
                    globals_intact.append(
                        plugin_function.__globals__ is plugin_globals
                        and plugin_globals["ProgressBar"] is _NoisyProgressBar
                        and plugin_globals["tqdm"] is noisy_tqdm
                    )
                    standalone = _NoisyInterpolator()
                    standalone.invoke_original = False
                    standalone.interpolate(
                        object(), images, 1.0, factor, 0, output_flows=output_flows
                    )
                except BaseException as exc:
                    thread_errors.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            thread.join()

        plugin_globals["invoke_original"] = invoke_original
        bound = _NoisyInterpolator().interpolate
        quiet = _quiet_interpolate_method(bound)
        self.assertIs(quiet.__self__, bound.__self__)
        self.assertIs(quiet.__func__.__defaults__, plugin_function.__defaults__)
        self.assertIs(quiet.__func__.__kwdefaults__, plugin_function.__kwdefaults__)
        self.assertIs(quiet.__func__.__closure__, plugin_function.__closure__)
        self.assertIs(quiet.__func__.test_metadata, metadata)

        frames = _batch(_frame(value=0.0), _frame(value=0.4), _frame(value=0.8))
        mappings = {
            "DownloadAndLoadGIMMVFIModel": _Loader,
            "GIMMVFI_interpolate": _NoisyInterpolator,
        }
        seen: list[tuple[int, int]] = []
        memory = _Memory()
        with self._comfy(memory):
            output = interpolate_offline(
                frames,
                precision="fp32",
                ds_factor=1.0,
                models_dir=self.models,
                node_mappings=mappings,
                load_device=torch.device("cpu"),
                progress=lambda done, total: seen.append((done, total)),
            )

        np.testing.assert_allclose(
            output,
            _batch(
                frames[0],
                (frames[0] + frames[1]) / 2,
                frames[1],
                (frames[1] + frames[2]) / 2,
                frames[2],
            ),
        )
        self.assertEqual(seen, [(1, 2), (2, 2)])
        # Only the two concurrent standalone calls publish their own 1/1.
        self.assertEqual(
            progress_events,
            [("init", 1), ("update", 1), ("init", 1), ("update", 1)],
        )
        self.assertEqual(tqdm_calls, [False, True, False, True])
        self.assertEqual(globals_intact, [True, True])
        self.assertEqual(thread_errors, [])
        self.assertTrue(
            all(
                item[:3] == (closure_value, positional_default, keyword_default)
                for item in bindings
            )
        )
        self.assertIs(plugin_function.__globals__, plugin_globals)
        self.assertIs(plugin_globals["ProgressBar"], _NoisyProgressBar)
        self.assertIs(plugin_globals["tqdm"], noisy_tqdm)

    def test_non_python_interpolator_fails_instead_of_running_noisily(self) -> None:
        # Built-in/plugin callables cannot safely receive a copied globals dict.
        with self.assertRaisesRegex(GimmVfiError, "ordinary Python bound instance method"):
            _quiet_interpolate_method([].append)

    def test_single_frame_does_not_load_the_model(self) -> None:
        memory = _Memory()
        frames = _batch(_frame())
        import torch

        mappings = {"DownloadAndLoadGIMMVFIModel": _Loader, "GIMMVFI_interpolate": _Interpolator}
        with mock.patch("comfy.model_management.load_models_gpu", memory.load_models_gpu):
            output = interpolate_offline(
                frames,
                precision="fp32",
                ds_factor=1.0,
                models_dir=self.models,
                node_mappings=mappings,
                load_device=torch.device("cpu"),
            )
        np.testing.assert_allclose(output, frames)
        self.assertEqual(memory.loaded, [])

    def test_base_exception_still_unloads(self) -> None:
        memory = _Memory()
        memory.fail_on_pair = 2
        frames = _batch(_frame(value=0.0), _frame(value=0.2), _frame(value=0.4))
        # The second check is immediately after interpolate: cancellation raises
        # from the first next(), before either produced frame can be yielded.
        with self._comfy(memory):
            stream = self._open_stream(frames, memory)
            with self.assertRaises(_Cancel):
                next(stream)
        self.assertEqual(len(_Interpolator.calls), 1)
        self.assertEqual(len(memory.unloaded), 1)
        self.assertEqual(memory.caches, 2)
        self.assertTrue(memory.devices_while_loaded[0].startswith("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        self.assertEqual(str(next(memory.unloaded[0].model.parameters()).device), "cpu")

    def test_plugin_exceptions_keep_original_globals_and_unload(self) -> None:
        # Both ordinary failures and cancellation-like BaseExceptions must cross
        # the cloned call unchanged, with no global suppression left behind.
        import torch

        frames = _batch(_frame(value=0.0), _frame(value=0.2))

        class _OriginalProgressBar:
            def __init__(self, _total) -> None:
                pass

            def update(self, _amount) -> None:
                pass

        def original_tqdm(iterable, **_kwargs):
            return iterable

        for failure in (RuntimeError("plugin failed"), _Cancel()):
            with self.subTest(failure=type(failure).__name__):
                def interpolate(
                    self,
                    module,
                    images,
                    ds_factor,
                    interpolation_factor,
                    seed,
                    output_flows=False,
                ):
                    del self, module, images, ds_factor, interpolation_factor, seed, output_flows
                    ProgressBar(1).update(1)  # noqa: F821
                    list(tqdm(range(1)))  # noqa: F821
                    raise plugin_failure  # noqa: F821

                plugin_globals = interpolate.__globals__.copy()
                plugin_globals.update(
                    ProgressBar=_OriginalProgressBar,
                    tqdm=original_tqdm,
                    plugin_failure=failure,
                )
                plugin_function = types.FunctionType(
                    interpolate.__code__,
                    plugin_globals,
                    name="interpolate",
                    argdefs=interpolate.__defaults__,
                    closure=interpolate.__closure__,
                )

                class _FailingInterpolator:
                    pass

                _FailingInterpolator.interpolate = plugin_function
                memory = _Memory()
                mappings = self._mappings(memory)
                mappings["GIMMVFI_interpolate"] = _FailingInterpolator
                with self._comfy(memory):
                    stream = iter_interpolate_offline(
                        frames,
                        2,
                        precision="fp32",
                        ds_factor=1.0,
                        models_dir=self.models,
                        node_mappings=mappings,
                        load_device=torch.device("cpu"),
                    )
                    with self.assertRaises(type(failure)) as raised:
                        next(stream)

                self.assertIs(raised.exception, failure)
                self.assertIs(plugin_function.__globals__, plugin_globals)
                self.assertIs(plugin_globals["ProgressBar"], _OriginalProgressBar)
                self.assertIs(plugin_globals["tqdm"], original_tqdm)
                self.assertEqual(len(memory.unloaded), 1)

    def test_model_manager_load_failure_still_unloads_and_clears_cache(self) -> None:
        memory = _Memory()
        memory.fail_on_load = True
        with self.assertRaisesRegex(RuntimeError, "simulated model load failure"):
            self._run(_batch(_frame(value=0.0), _frame(value=0.2)), memory)
        self.assertEqual(len(memory.unloaded), 1)
        self.assertEqual(memory.caches, 2)
        self.assertEqual(
            memory.events,
            ["free", "cache", "model_loader", "load", "unload", "cublas", "cache"],
        )


class NodeContractTests(unittest.TestCase):
    def test_nodes_are_registered_and_legacy_schema_is_present(self) -> None:
        self.assertIs(NODE_CLASS_MAPPINGS["MyVideoEnhance"], MyVideoEnhance)
        self.assertIs(NODE_CLASS_MAPPINGS["MyDLSSRuntimeProbe"], MyDLSSRuntimeProbe)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["MyVideoEnhance"], "My Video Enhance")
        types = MyVideoEnhance.INPUT_TYPES()
        self.assertIn("images", types["required"])
        self.assertEqual(types["required"]["spatial_mode"][1]["default"], "2.0x")
        self.assertEqual(MyVideoEnhance.RETURN_TYPES, ("IMAGE", "INT", "STRING"))
        self.assertIn("advanced", types["optional"]["vfi_precision"][1])
        self.assertIn("DLAA", SPATIAL_LABELS[0])
        self.assertEqual(spatial_scale("1.0 DLAA (native)"), 1.0)

    def test_legacy_schema_keeps_stage_order_at_its_existing_index(self) -> None:
        types = MyVideoEnhance.INPUT_TYPES()
        optional = list(types["optional"])
        self.assertEqual(optional[:3], ["vfi_precision", "vfi_ds_factor", "motion"])
        self.assertEqual(optional[8], "stage_order")
        options, widget = types["optional"]["stage_order"]
        self.assertEqual(options, list(STAGE_ORDERS))
        self.assertEqual(widget["default"], STAGE_ORDER_DLSS_THEN_VFI)
        self.assertTrue(widget["advanced"])
        self.assertEqual(
            list(types["required"]),
            [
                "images",
                "enable_super_resolution",
                "spatial_mode",
                "enable_neural_rendering",
                "nr_profile",
                "nr_intensity",
                "enable_frame_interpolation",
            ],
        )

    def test_all_off_returns_the_same_object_and_touches_nothing(self) -> None:
        images = _batch(_frame(), _frame(value=0.7))
        with mock.patch("my_nodes.nodes.video_enhance.run_frame_pipeline") as pipeline:
            output, multiplier, status = MyVideoEnhance().enhance(
                images, False, "2.0x", False, "standard", 1.0, False
            )
        self.assertIs(output, images)
        self.assertEqual(multiplier, 1)
        self.assertIn("pass-through", status)
        pipeline.assert_not_called()

    def test_an_unknown_stage_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MyVideoEnhance().enhance(
                _batch(_frame()), True, "2.0x", False, "standard", 1.0, True,
                stage_order="dlss_and_vfi_in_parallel",
            )

    def _pipeline_call(self, **enhance_kwargs):
        """Run the node with pipeline and folder_paths stubbed; return the recorded call."""
        images = _batch(_frame(), _frame(value=0.5))
        calls: list[dict] = []
        created: list[int] = []
        updates: list[int] = []

        class _Bar:
            def __init__(self, total: int) -> None:
                created.append(total)

            def update_absolute(self, value: int, total=None, preview=None) -> None:
                del total, preview
                updates.append(value)

        def fake_pipeline(source, source_spec, plan, write_frame, **kwargs):
            from my_nodes.core.video_enhance import frame_pipeline

            specs = frame_pipeline.pipeline_specs(source_spec, plan)
            total = frame_pipeline.pipeline_step_total(source_spec, plan)
            calls.append(
                {
                    "source": np.stack(list(source)),
                    "spec": source_spec,
                    "plan": plan,
                    "created": list(created),
                    "kwargs": kwargs,
                }
            )
            # The pipeline reports cumulative progress over all stages, then hands
            # the final frames to the node's writer.
            for step in range(1, total + 1):
                if kwargs["progress"] is not None:
                    kwargs["progress"](step, total)
            for index in range(specs.final.count):
                write_frame(
                    index,
                    np.full((specs.final.height, specs.final.width, 3), index, dtype=np.float32),
                )
            return PipelineResult(
                frame_count=specs.final.count,
                output_height=specs.final.height,
                output_width=specs.final.width,
                channel_order="RGBA",
                features=FEATURE_SR,
                stages=plan.stages,
            )

        node = MyVideoEnhance()
        node._progress_bar = _Bar
        temp_directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_directory, ignore_errors=True))
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = "/models"
        folder_paths.get_temp_directory = lambda: str(temp_directory)
        torch_module = types.ModuleType("torch")
        torch_module.from_numpy = lambda array: array
        import sys

        import my_nodes.nodes.video_enhance as node_module

        arguments = {
            "enable_super_resolution": True,
            "spatial_mode": "2.0x",
            "enable_neural_rendering": False,
            "nr_profile": "standard",
            "nr_intensity": 1.0,
            "enable_frame_interpolation": True,
        }
        arguments.update(enhance_kwargs)
        with mock.patch.object(node_module, "prepare_frames", return_value=images.copy()), \
             mock.patch.object(node_module, "resolve_runtime_dir", return_value="/runtime"), \
             mock.patch.object(node_module, "run_frame_pipeline", side_effect=fake_pipeline), \
             mock.patch("psutil.virtual_memory", return_value=types.SimpleNamespace(available=1 << 30)), \
             mock.patch.dict(sys.modules, {"folder_paths": folder_paths, "torch": torch_module}):
            output, multiplier, status = node.enhance(images, **arguments)
        return types.SimpleNamespace(
            call=calls[0],
            output=output,
            multiplier=multiplier,
            status=status,
            updates=updates,
            temp_directory=str(temp_directory),
        )

    def test_the_shared_pipeline_gets_the_plan_spec_temp_dir_and_options(self) -> None:
        result = self._pipeline_call(stage_order=STAGE_ORDER_DLSS_THEN_VFI)
        call = result.call
        self.assertEqual(call["spec"], FrameSpec(count=2, height=6, width=4))
        self.assertEqual(call["plan"].stages, ("dlss", "vfi"))
        self.assertEqual(np.asarray(call["source"]).shape, (2, 6, 4, 3))
        # DLSS first: 2 frames, then one VFI pair, so the bar totals 3 steps.
        self.assertEqual(call["created"], [3])
        self.assertEqual(result.updates, [1, 2, 3])
        # The two-stage run stages its intermediate frames on disk.
        self.assertEqual(call["kwargs"]["temp_directory"], result.temp_directory)
        options = call["kwargs"]
        self.assertEqual(options["vfi"].precision, "fp32")
        self.assertEqual(options["vfi"].models_dir, "/models")
        self.assertEqual(options["vfi"].ds_factor, 1.0)
        self.assertEqual(options["dlss"].runtime_dir, "/runtime")
        self.assertEqual(options["dlss"].motion_mode, "optical_flow")
        self.assertEqual(options["dlss"].channel_order, "auto")
        self.assertIsNotNone(options["dlss"].memory_hooks)
        from my_nodes.core.video_enhance.dlss_stage import comfy_interrupt

        self.assertIs(options["interrupt"], comfy_interrupt)
        # The output batch is the preallocated final IMAGE, written frame by frame.
        self.assertEqual(result.output.shape, (3, 12, 8, 3))
        self.assertEqual(result.multiplier, 2)
        self.assertIn("dlss+vfi", result.status)
        self.assertIn("SR 2.0x", result.status)
        self.assertIn("channels=RGBA", result.status)
        self.assertIn("frames=3", result.status)
        self.assertIn("input FPS x2", result.status)
        np.testing.assert_allclose(result.output[2], np.full((12, 8, 3), 2.0, dtype=np.float32))

    def test_vfi_first_order_and_the_single_stage_path_are_wired_too(self) -> None:
        result = self._pipeline_call(stage_order=STAGE_ORDER_VFI_THEN_DLSS)
        self.assertEqual(result.call["plan"].stages, ("vfi", "dlss"))
        # VFI first: 1 pair, then 3 DLSS frames, so the bar totals 4 steps.
        self.assertEqual(result.call["created"], [4])
        self.assertEqual(result.updates, [1, 2, 3, 4])
        self.assertEqual(result.call["kwargs"]["temp_directory"], result.temp_directory)
        self.assertEqual(result.output.shape, (3, 12, 8, 3))

        single = self._pipeline_call(enable_super_resolution=False)
        self.assertEqual(single.call["plan"].stages, ("vfi",))
        self.assertEqual(single.call["created"], [1])
        self.assertEqual(single.updates, [1])
        # A single stage needs no disk-backed intermediate.
        self.assertIsNone(single.call["kwargs"]["temp_directory"])
        self.assertEqual(single.output.shape, (3, 6, 4, 3))
        self.assertIn("frames=3", single.status)
        self.assertIn("vfi", single.status)

    def test_ram_preflight_rejects_an_output_that_would_exhaust_ram(self) -> None:
        images = _batch(_frame(), _frame())
        available = 4096
        with mock.patch(
            "psutil.virtual_memory", return_value=types.SimpleNamespace(available=available)
        ), mock.patch("my_nodes.nodes.video_enhance.run_frame_pipeline") as pipeline:
            with self.assertRaises(InsufficientRamError) as raised:
                MyVideoEnhance().enhance(images, True, "2.0x", False, "standard", 1.0, True)
        pipeline.assert_not_called()
        # The final IMAGE is 3 frames of 12x8 float32: 3456 bytes.
        message = str(raised.exception)
        self.assertIn("3456", message)
        self.assertIn(str(available), message)
        self.assertIn(f"{OUTPUT_RAM_FRACTION:.0%}", message)
        self.assertIn("MyVideoEnhanceStream", message)

    def test_ram_preflight_reports_a_measurement_failure(self) -> None:
        images = _batch(_frame(), _frame())
        with mock.patch(
            "psutil.virtual_memory", side_effect=OSError("no /proc/meminfo")
        ):
            with self.assertRaises(InsufficientRamError) as raised:
                MyVideoEnhance().enhance(images, True, "2.0x", False, "standard", 1.0, True)
        self.assertIn("no /proc/meminfo", str(raised.exception))

    def test_probe_requires_a_dlss_feature(self) -> None:
        with self.assertRaises(ValueError):
            MyDLSSRuntimeProbe().probe(False, "2.0x", False, "standard", 1.0)

    def test_probe_runs_the_same_pipeline(self) -> None:
        with mock.patch("my_nodes.nodes.video_enhance.run_dlss_stage") as dlss:
            from my_nodes.core.video_enhance.dlss_stage import DlssStageResult

            dlss.return_value = DlssStageResult(
                frames=np.zeros((1, 64, 64, 3), dtype=np.float32),
                output_width=64,
                output_height=64,
                channel_order="RGBA",
                features=FEATURE_SR,
            )
            status, = MyDLSSRuntimeProbe().probe(True, "2.0x", False, "standard", 1.0, runtime_dir="/runtime")
        self.assertIn("probe ok", status)
        self.assertEqual(dlss.call_args.kwargs["runtime_dir"], "/runtime")
        self.assertEqual(dlss.call_args.kwargs["motion_mode"], "none")
        self.assertEqual(dlss.call_args.args[1].shape, (1, 32, 32, 3))

    def test_the_advanced_controls_are_appended_after_the_legacy_widgets(self) -> None:
        optional = MyVideoEnhance.INPUT_TYPES()["optional"]
        # The names and their order are the contract the workflow depends on.
        self.assertEqual(list(ADVANCED_OPTIONAL), ADVANCED_NAMES)
        names = list(optional)
        self.assertEqual(
            names[-len(ADVANCED_NAMES) - 1 :], ["stage_order"] + ADVANCED_NAMES
        )
        self.assertEqual(optional["stage_order"][1]["default"], STAGE_ORDER_DLSS_THEN_VFI)
        # ComfyUI's default restore path reads widgets_values positionally. Keep
        # the complete old prefix stable so existing workflows cannot shift
        # stage_order into the first newly added field.
        self.assertEqual(names[: -len(ADVANCED_NAMES)], [
            "vfi_precision",
            "vfi_ds_factor",
            "motion",
            "scene_cut_threshold",
            "channel_order",
            "runtime_dir",
            "wine_prefix",
            "worker_timeout",
            "stage_order",
        ])
        # Every advanced control is optional, advanced, documented and defaults
        # to exactly the plan default it writes.
        plan_defaults = VideoEnhancePlan()
        for name, (field, bounds) in ADVANCED_CONTROLS.items():
            with self.subTest(control=name):
                options = optional[name][1]
                self.assertIs(options["advanced"], True)
                self.assertIn("tooltip", options)
                self.assertEqual(options["default"], getattr(plan_defaults, field))
                if bounds is not None:
                    self.assertEqual((options["min"], options["max"]), bounds)
        # The choice lists are the plan's, so a node can never offer a value the
        # plan rejects.
        self.assertEqual(list(optional["style"][0]), list(NR_STYLES))
        self.assertEqual(list(optional["preset"][0]), list(NR_PRESETS))
        self.assertEqual(list(optional["sr_preset"][0]), list(SR_PRESETS))
        self.assertEqual(optional["style"][1]["default"], "Cinematic")
        self.assertEqual(optional["preset"][1]["default"], "Default")
        self.assertEqual(optional["sr_preset"][1]["default"], "Default")
        self.assertEqual(optional["gpu_index"][0], "INT")
        self.assertEqual(optional["gpu_index"][1]["step"], 1)
        for name in ("ui_correction", "auto_mask"):
            self.assertEqual(optional[name][0], "BOOLEAN")

    def test_the_plan_builder_defaults_match_the_widget_defaults(self) -> None:
        # Both the widgets and `_plan` carry the defaults, so they are compared
        # instead of being restated in two places that can drift apart.
        import inspect

        parameters = inspect.signature(_plan).parameters
        for name, (field, _bounds) in ADVANCED_CONTROLS.items():
            with self.subTest(control=name):
                self.assertEqual(parameters[name].default, getattr(VideoEnhancePlan(), field))

    def test_the_model_tooltips_name_the_custom_profile(self) -> None:
        # The user has to learn from the widget itself that a model field only
        # applies to `custom`, while detail/color are post-NR composites.
        for name in (
            "style", "preset", "local_structure", "local_tone", "skin",
            "ui_correction", "auto_mask",
        ):
            with self.subTest(control=name):
                self.assertIn(CUSTOM_PROFILE_ONLY, _advanced_options(name)["tooltip"])
        for name in ("detail", "color"):
            with self.subTest(control=name):
                tooltip = _advanced_options(name)["tooltip"]
                self.assertIn("Post-NR", tooltip)
                self.assertIn("not only with nr_profile=custom", tooltip)
        self.assertIn("super resolution", _advanced_options("sr_preset")["tooltip"])
        self.assertIn("GPU", _advanced_options("gpu_index")["tooltip"])
        # The profile widget itself points at the advanced model fields.
        self.assertIn("custom", MyVideoEnhance.INPUT_TYPES()["required"]["nr_profile"][1]["tooltip"])

    def test_every_advanced_control_reaches_the_plan(self) -> None:
        result = self._pipeline_call(**ADVANCED_VALUES)
        plan = result.call["plan"]
        for name, (field, _bounds) in ADVANCED_CONTROLS.items():
            with self.subTest(control=name):
                self.assertEqual(getattr(plan, field), ADVANCED_VALUES[name])
        # A built-in profile ignores all of them, so nothing about it changes.
        self.assertIs(plan_neural_rendering_settings(plan).ui_correction, False)
        # `custom` is the profile that reads them: the header therefore shows the
        # mapped model selectors and the plan's own strengths.
        custom = self._pipeline_call(nr_profile=CUSTOM_NR_PROFILE, **ADVANCED_VALUES)
        custom_plan = custom.call["plan"]
        self.assertEqual(custom_plan.nr_profile, CUSTOM_NR_PROFILE)
        settings = plan_neural_rendering_settings(custom_plan)
        self.assertEqual(settings.profile, CUSTOM_NR_PROFILE)
        self.assertEqual(settings.style, STYLE_IDS[str(ADVANCED_VALUES["style"])])
        self.assertEqual(settings.preset, PRESET_IDS[str(ADVANCED_VALUES["preset"])])
        self.assertEqual(settings.intensity, custom_plan.nr_intensity)
        self.assertEqual(settings.structure, ADVANCED_VALUES["local_structure"])
        self.assertEqual(settings.tone, ADVANCED_VALUES["local_tone"])
        self.assertEqual(settings.skin, ADVANCED_VALUES["skin"])
        self.assertIs(settings.automask, True)
        self.assertIs(settings.ui_correction, True)
        header = build_header(custom_plan, _batch(_frame(), _frame(value=0.5)))
        self.assertEqual(header.style, settings.style)
        self.assertEqual(header.preset, settings.preset)
        self.assertEqual(header.ui_correction, 1)

    def test_the_v3_schema_exposes_the_same_advanced_widgets(self) -> None:
        try:
            from comfy_api.latest import io
        except ImportError:
            self.skipTest("comfy_api is not installed in this interpreter")
        schema = MyVideoEnhance.define_schema()
        ids = [item.id for item in schema.inputs]
        self.assertEqual(ids[-len(ADVANCED_NAMES) - 1 :], ["stage_order"] + ADVANCED_NAMES)
        # The v3 form is generated from the classic widgets, so one comparison
        # covers the names, defaults, ranges, choices, flags and tooltips.
        by_id = {item.id: item for item in schema.inputs}
        classic = MyVideoEnhance.INPUT_TYPES()["optional"]
        for name in ADVANCED_NAMES:
            with self.subTest(control=name):
                widget = by_id[name]
                _kind, options = ADVANCED_OPTIONAL[name]
                self.assertTrue(widget.advanced)
                self.assertEqual(widget.default, options["default"])
                self.assertEqual(widget.tooltip, options["tooltip"])
                for key in ("min", "max", "step"):
                    if key in options:
                        self.assertEqual(getattr(widget, key, None), options[key], key)
                if isinstance(_kind, list):
                    self.assertEqual(list(widget.options), _kind)
                self.assertEqual(classic[name][1]["default"], widget.default)
        self.assertIsNotNone(io)

    def test_define_schema_outputs_match_when_comfy_api_is_present(self) -> None:
        try:
            from comfy_api.latest import io
        except ImportError:
            self.skipTest("comfy_api is not installed in this interpreter")
        schema = MyVideoEnhance.define_schema()
        self.assertEqual(schema.node_id, "MyVideoEnhance")
        self.assertTrue(schema.is_experimental)
        ids = [item.id for item in schema.inputs]
        self.assertIn("enable_frame_interpolation", ids)
        advanced = [item.id for item in schema.inputs if getattr(item, "advanced", False)]
        self.assertIn("vfi_ds_factor", advanced)
        # New controls follow stage_order to preserve old positional workflows.
        self.assertEqual(ids[-len(ADVANCED_NAMES) - 1], "stage_order")
        self.assertIn("stage_order", advanced)
        self.assertEqual([output.display_name for output in schema.outputs], ["images", "fps_multiplier", "status"])
        self.assertIsNotNone(io)


class ChannelOrderTests(unittest.TestCase):
    def test_auto_picks_the_closer_order(self) -> None:
        source = _frame(value=0.2)
        self.assertEqual(select_channel_order("auto", source, source), "RGBA")
        self.assertEqual(select_channel_order("auto", source, swap_rb(source)), "BGRA")
