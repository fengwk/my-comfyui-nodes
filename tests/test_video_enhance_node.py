"""MyVideoEnhance and MyDLSSRuntimeProbe: planning, routing and backend boundaries.

The DLSS path drives the existing real-pipe fake DNR3 worker. GIMM is represented
by stand-in node classes and a real torch.nn.Module; the S-Lab implementation is
not imported. Comfy's model manager is stubbed so the test records ModelPatcher
lifecycle calls, including cleanup after a BaseException.
"""

from __future__ import annotations

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
    build_header,
    output_dimensions,
    prepare_frames,
    resolve_runtime_dir,
    run_dlss_stage,
)
from my_nodes.core.video_enhance.gimm_vfi import (
    GIMM_FLOW_NAME,
    GIMM_MODEL_NAME,
    GimmVfiError,
    _clear_cublas_workspaces,
    _clear_gimm_backwarp_cache,
    clear_patcher_cache,
    interpolate_offline,
    require_offline_weights,
    resolve_gimm_nodes,
)
from my_nodes.core.video_enhance.motion import MOTION_NONE, MotionGuideError, MotionGuides
from my_nodes.core.video_enhance.nr_profiles import neural_rendering_settings
from my_nodes.core.video_enhance.plan import NR_PROFILES, VideoEnhancePlan
from my_nodes.core.video_enhance.runtime import HostDriver
from my_nodes.nodes.video_enhance import (
    SPATIAL_LABELS,
    MyDLSSRuntimeProbe,
    MyVideoEnhance,
    spatial_scale,
)
from my_nodes.registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

from .video_enhance_fixtures import assert_process_gone, create_runtime_dir, fake_worker_command, read_report


def _batch(*frames: np.ndarray) -> np.ndarray:
    return np.stack(frames, axis=0).astype(np.float32)


def _frame(width: int = 4, height: int = 6, value: float = 0.2) -> np.ndarray:
    frame = np.full((height, width, 3), value, dtype=np.float32)
    frame[..., 0] = value
    frame[..., 2] = 1.0 - value
    return frame


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
        self.assertEqual(tuple(expected), NR_PROFILES)
        for profile, fields in expected.items():
            settings = neural_rendering_settings(profile, 0.5)
            for name, value in fields.items():
                self.assertEqual(getattr(settings, name), value, msg=f"{profile}.{name}")
            self.assertEqual(settings.intensity, 0.5)

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

    def _run(self, frames: np.ndarray, memory: _Memory):
        import sys
        import torch
        import comfy.model_management as mm
        from comfy.model_patcher import ModelPatcher

        class _RecordingLoader(_Loader):
            def loadmodel(self, model, precision="fp32", torch_compile=False):
                memory.events.append("model_loader")
                return super().loadmodel(model, precision, torch_compile)

        mappings = {
            "DownloadAndLoadGIMMVFIModel": _RecordingLoader,
            "GIMMVFI_interpolate": _Interpolator,
        }
        # The production function imports model_management locally, so patch the
        # module object it receives rather than the caller's name binding.
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
        with self.assertRaises(_Cancel):
            self._run(frames, memory)
        self.assertEqual(len(memory.unloaded), 1)
        self.assertEqual(memory.caches, 2)
        self.assertTrue(memory.devices_while_loaded[0].startswith("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        self.assertEqual(str(next(memory.unloaded[0].model.parameters()).device), "cpu")

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

    def test_all_off_returns_the_same_object_and_touches_nothing(self) -> None:
        images = _batch(_frame(), _frame(value=0.7))
        with mock.patch("my_nodes.core.video_enhance.dlss_stage.run_dlss_stage") as dlss, \
             mock.patch("my_nodes.core.video_enhance.gimm_vfi.interpolate_offline") as vfi:
            output, multiplier, status = MyVideoEnhance().enhance(
                images, False, "2.0x", False, "standard", 1.0, False
            )
        self.assertIs(output, images)
        self.assertEqual(multiplier, 1)
        self.assertIn("pass-through", status)
        dlss.assert_not_called()
        vfi.assert_not_called()

    def test_dlss_worker_is_closed_before_vfi_starts(self) -> None:
        # The combined node must not keep the Wine worker alive while GIMM loads.
        order: list[str] = []
        images = _batch(_frame(), _frame(value=0.5))

        def dlss(*_args, **kwargs):
            order.append("dlss-start")
            kwargs["progress"](1, 2)
            kwargs["progress"](2, 2)
            order.append("dlss-closed")
            from my_nodes.core.video_enhance.dlss_stage import DlssStageResult

            return DlssStageResult(frames=images, output_width=4, output_height=6, channel_order="RGBA", features=FEATURE_SR)

        def vfi(frames, **kwargs):
            kwargs["progress"](1, 1)
            order.append("vfi")
            self.assertEqual(order, ["dlss-start", "dlss-closed", "vfi"])
            middle = (frames[0] + frames[1]) / 2
            return np.stack((frames[0], middle, frames[1]), axis=0)

        import sys
        import types

        node = MyVideoEnhance()
        created: list[int] = []
        updates: list[int] = []

        class _Bar:
            def __init__(self, total: int) -> None:
                created.append(total)

            def update_absolute(self, value: int, total=None, preview=None) -> None:
                del total, preview
                updates.append(value)

        node._progress_bar = _Bar
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = "/models"
        torch_module = types.ModuleType("torch")
        torch_module.from_numpy = lambda array: array
        import my_nodes.nodes.video_enhance as node_module

        with mock.patch.object(node_module, "prepare_frames", return_value=images), \
             mock.patch.object(node_module, "resolve_runtime_dir", return_value="/runtime"), \
             mock.patch.object(node_module, "run_dlss_stage", side_effect=dlss), \
             mock.patch.object(node_module, "interpolate_offline", side_effect=vfi), \
             mock.patch.dict(sys.modules, {"folder_paths": folder_paths, "torch": torch_module}):
            output, multiplier, status = node.enhance(
                images, True, "2.0x", False, "standard", 1.0, True
            )
        self.assertEqual(output.shape[0], 3)
        self.assertEqual(multiplier, 2)
        self.assertIn("input FPS x2", status)
        self.assertEqual(created, [3])
        self.assertEqual(updates, [1, 2, 3])

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
        self.assertEqual([output.display_name for output in schema.outputs], ["images", "fps_multiplier", "status"])
        self.assertIsNotNone(io)


class ChannelOrderTests(unittest.TestCase):
    def test_auto_picks_the_closer_order(self) -> None:
        source = _frame(value=0.2)
        self.assertEqual(select_channel_order("auto", source, source), "RGBA")
        self.assertEqual(select_channel_order("auto", source, swap_rb(source)), "BGRA")
