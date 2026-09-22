"""MyVideoEnhanceStream: contracts, streaming wiring, routing and cleanup.

Everything outside the node is a recording stand-in: `video_io` never starts
FFmpeg, the shared pipeline never loads GIMM or Wine, and ComfyUI's temp path and
VIDEO wrapper are injected. What is pinned down here is the node's own behaviour:
which file it probes, that the frames walk from the reader through the pipeline
into the encoder one at a time instead of through a clip-sized list, which
encoder settings it asks for, and which temporary files it owns and deletes on
success, on failure and on cancellation.
"""

from __future__ import annotations

import io as io_module
import os
import shutil
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

from my_nodes.core.video_enhance import FEATURE_SR
from my_nodes.core.video_enhance.dlss_stage import RUNTIME_DIR_ENV, comfy_interrupt
from my_nodes.core.video_enhance.frame_pipeline import (
    FrameSpec,
    PipelineResult,
    pipeline_specs,
    pipeline_step_total,
)
from my_nodes.core.video_enhance.plan import (
    STAGE_ORDER_DLSS_THEN_VFI,
    STAGE_ORDER_VFI_THEN_DLSS,
    STAGE_ORDERS,
)
from my_nodes.core.video_enhance.video_io import VideoIOError, VideoSpec
from my_nodes.nodes.video_enhance import MyVideoEnhance
from my_nodes.nodes.video_enhance_stream import (
    FRAME_STORE_DIRECTORY,
    OUTPUT_CODECS,
    TEMP_PREFIX,
    MyVideoEnhanceStream,
)
from my_nodes.registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

import my_nodes.nodes.video_enhance_stream as stream_module

ALL_STAGES_OFF = {
    "enable_super_resolution": False,
    "spatial_mode": "2.0x",
    "enable_neural_rendering": False,
    "nr_profile": "standard",
    "nr_intensity": 1.0,
    "enable_frame_interpolation": False,
}


def _spec(
    root: Path,
    *,
    frames: int = 5,
    width: int = 8,
    height: int = 6,
    fps: Fraction = Fraction(30000, 1001),
    has_audio: bool = True,
) -> VideoSpec:
    """The probed description of a source file; FFmpeg never reads it."""
    path = root / "source.mkv"
    path.write_bytes(b"probe")
    return VideoSpec(
        path=path,
        width=width,
        height=height,
        frame_count=frames,
        fps=fps,
        has_audio=has_audio,
        pixel_format="yuv420p",
    )


class _FileVideo:
    """A native file-backed VIDEO: a local path and a trim window."""

    def __init__(self, path: Path, *, trim=(0.0, None)) -> None:
        self.path = Path(path)
        self._trim = trim
        self.stream_calls = 0

    def get_stream_source(self):
        self.stream_calls += 1
        return str(self.path)

    def get_active_trim_window(self):
        return self._trim


class _MemoryVideo:
    """A component or in-memory VIDEO: its stream source is a buffer."""

    def __init__(self) -> None:
        self.stream_calls = 0

    def get_stream_source(self):
        self.stream_calls += 1
        return io_module.BytesIO(b"not a file")

    def get_active_trim_window(self):
        return 0.0, None


class _UntouchableVideo:
    """A VIDEO that fails the test if the node inspects it at all."""

    def get_stream_source(self):
        raise AssertionError("a pass-through must not look at the source")

    def get_active_trim_window(self):
        raise AssertionError("a pass-through must not look at the source")


class _Reader:
    """Context-managed lazy frame source; every pull is logged when it happens."""

    def __init__(self, count: int, size: tuple[int, int], log, *, spec, ffmpeg_path, interrupt) -> None:
        self.count = count
        self.size = size
        self.log = log
        self.spec = spec
        self.ffmpeg_path = ffmpeg_path
        self.interrupt = interrupt
        self.entered = 0
        self.exited = 0
        self.index = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *_info):
        self.exited += 1
        return False

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.index == self.count:
            raise StopIteration
        value = np.float32((self.index + 1) / 10)
        self.index += 1
        self.log.append(("read", self.index - 1))
        return np.full((*self.size, 3), value, dtype=np.float32)


class _Writer:
    """Context-managed sink that keeps scalars, never the frames it is given."""

    def __init__(self, path, *, width, height, fps, expected_frames, codec, quality, interrupt, log) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.expected_frames = expected_frames
        self.codec = codec
        self.quality = quality
        self.interrupt = interrupt
        self.log = log
        self.entered = 0
        self.exited = 0
        self.written: list[tuple[int, tuple[int, ...], float]] = []

    def __enter__(self):
        self.entered += 1
        self.path.write_bytes(b"")
        return self

    def __exit__(self, *_info):
        # Deliberately keep the partial output: the real writer removes it itself,
        # so a leftover file here is exactly the residue the node must delete.
        self.exited += 1
        return False

    def write(self, frame: np.ndarray) -> None:
        index = len(self.written)
        self.written.append((index, frame.shape, float(frame[0, 0, 0])))
        self.log.append(("write", index))


def _remux_with_audio(source: VideoSpec, video_only: Path, output: Path) -> Path:
    """The audio remux: a new file at `output`, the video-only file stays put."""
    shutil.copyfile(video_only, output)
    return output


def _remux_without_audio(source: VideoSpec, video_only: Path, output: Path) -> Path:
    """The no-audio remux: the encoded file itself becomes the output."""
    os.replace(video_only, output)
    return output


def _failing_remux(source: VideoSpec, video_only: Path, output: Path) -> Path:
    """A remux failure: its own partial output goes, the video-only stays."""
    output.write_bytes(b"partial")
    output.unlink()
    raise VideoIOError(f"ffmpeg failed with exit code 1 while remuxing {output}")


class _FakeVideoIO:
    """The consumed `video_io` surface, recording every call it receives."""

    def __init__(self, spec: VideoSpec, log, *, remux=None, probe_error=None) -> None:
        self.spec = spec
        self.log = log
        self.remux = remux
        self.probe_error = probe_error
        self.probes: list[tuple] = []
        self.readers: list[_Reader] = []
        self.writers: list[_Writer] = []
        self.remuxes: list[tuple] = []

    def probe_cfr_video(self, source, *, ffprobe_path="ffprobe", interrupt=None) -> VideoSpec:
        self.probes.append((source, ffprobe_path, interrupt))
        if self.probe_error is not None:
            raise self.probe_error
        return self.spec

    def FFmpegFrameReader(self, spec, *, ffmpeg_path="ffmpeg", interrupt=None) -> _Reader:
        reader = _Reader(
            spec.frame_count,
            (spec.height, spec.width),
            self.log,
            spec=spec,
            ffmpeg_path=ffmpeg_path,
            interrupt=interrupt,
        )
        self.readers.append(reader)
        return reader

    def FFmpegFrameWriter(
        self, path, *, width, height, fps, expected_frames, codec="libx264", quality=18,
        ffmpeg_path="ffmpeg", interrupt=None,
    ) -> _Writer:
        writer = _Writer(
            path,
            width=width,
            height=height,
            fps=fps,
            expected_frames=expected_frames,
            codec=codec,
            quality=quality,
            interrupt=interrupt,
            log=self.log,
        )
        self.writers.append(writer)
        return writer

    def remux_audio(self, source, video_only_path, output_path, *, ffmpeg_path="ffmpeg", interrupt=None) -> Path:
        video_only = Path(video_only_path)
        output = Path(output_path)
        self.remuxes.append((source, video_only, output, interrupt))
        if self.remux is not None:
            return self.remux(source, video_only, output)
        strategy = _remux_with_audio if source.has_audio else _remux_without_audio
        return strategy(source, video_only, output)


class _Bar:
    """Records the progress the node reports."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.updates: list[int] = []

    def update_absolute(self, value: int, total=None, preview=None) -> None:
        del total, preview
        self.updates.append(value)


def _fake_pipeline(calls: list, *, failure=None):
    """Stand-in for the shared pipeline: lazy in, one `write_frame` per frame."""

    def pipeline(source, source_spec, plan, write_frame, **kwargs) -> PipelineResult:
        specs = pipeline_specs(source_spec, plan)
        total = pipeline_step_total(source_spec, plan)
        calls.append(
            types.SimpleNamespace(
                source=source, spec=source_spec, plan=plan, write_frame=write_frame, kwargs=kwargs
            )
        )
        if failure is not None:
            raise failure
        shape = (specs.final.height, specs.final.width, 3)
        index = 0
        for _frame in source:  # one source frame pulled, one final frame written
            write_frame(index, np.full(shape, float(index), dtype=np.float32))
            index += 1
        while index < specs.final.count:
            write_frame(index, np.full(shape, float(index), dtype=np.float32))
            index += 1
        if kwargs.get("progress") is not None:
            for step in range(1, total + 1):
                kwargs["progress"](step, total)
        return PipelineResult(
            frame_count=specs.final.count,
            output_height=specs.final.height,
            output_width=specs.final.width,
            channel_order="RGBA",
            features=FEATURE_SR,
            stages=plan.stages,
        )

    return pipeline


class _StreamTestCase(unittest.TestCase):
    """An isolated Comfy temp directory and the lazy ComfyUI stand-ins."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.comfy_temp = self.root / "comfy_temp"
        self.comfy_temp.mkdir()
        self.log: list[tuple[str, int]] = []

    def folder_paths(self, *, temp_directory: Path | None = None, models_dir: str = "/models"):
        module = types.ModuleType("folder_paths")
        module.models_dir = models_dir
        module.get_temp_directory = lambda: str(temp_directory or self.comfy_temp)
        return module

    def exploding_folder_paths(self):
        """A `folder_paths` whose temp directory must not be requested."""
        module = types.ModuleType("folder_paths")
        module.models_dir = "/models"

        def explode():
            raise AssertionError("the pass-through path must not ask for a temp directory")

        module.get_temp_directory = explode
        return module

    def run_node(self, video, video_io, *, folder_paths=None, video_factory=None, failure=None, **widgets):
        """Run `enhance` with every collaborator replaced; return what it did."""
        calls: list = []
        bars: list[_Bar] = []
        paths: list[str] = []

        def bar_factory(total: int) -> _Bar:
            bar = _Bar(total)
            bars.append(bar)
            return bar

        def default_factory(path: str):
            paths.append(path)
            return f"VIDEO:{path}"

        if video_io is None:
            # No `video_io` at all: proves the path never reaches the FFmpeg layer.
            io_patch = mock.patch.object(
                stream_module,
                "_video_io",
                side_effect=AssertionError("video_io must not be imported on this path"),
            )
        else:
            io_patch = mock.patch.object(stream_module, "_video_io", return_value=video_io)
        with io_patch, \
             mock.patch.object(
                 stream_module, "run_frame_pipeline", side_effect=_fake_pipeline(calls, failure=failure)
             ), \
             mock.patch.object(stream_module, "_progress_bar", side_effect=bar_factory), \
             mock.patch.object(
                 stream_module, "_video_from_file", side_effect=video_factory or default_factory
             ), \
             mock.patch.dict(sys.modules, {"folder_paths": folder_paths or self.folder_paths()}):
            output, multiplier, status = stream_module.MyVideoEnhanceStream().enhance(
                video, **dict(ALL_STAGES_OFF, **widgets)
            )
        return types.SimpleNamespace(
            output=output,
            multiplier=multiplier,
            status=status,
            calls=calls,
            bars=bars,
            paths=paths,
            io=video_io,
        )

    def assert_fails(self, exception_type, video, video_io, **kwargs):
        """Run the node and return the exception it raised."""
        with self.assertRaises(exception_type) as raised:
            self.run_node(video, video_io, **kwargs)
        return raised.exception

    def temp_residue(self) -> list[str]:
        return sorted(entry.name for entry in self.comfy_temp.iterdir())


class NodeContractTests(unittest.TestCase):
    def test_the_stream_node_is_registered_under_its_display_name(self) -> None:
        self.assertIs(NODE_CLASS_MAPPINGS["MyVideoEnhanceStream"], MyVideoEnhanceStream)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["MyVideoEnhanceStream"], "My Video Enhance Stream")
        self.assertEqual(MyVideoEnhanceStream.CATEGORY, "image/video")
        self.assertEqual(MyVideoEnhanceStream.FUNCTION, "enhance")

    def test_the_classic_contract_is_a_video_to_video_node(self) -> None:
        types_ = MyVideoEnhanceStream.INPUT_TYPES()
        self.assertEqual(types_["required"]["video"][0], "VIDEO")
        self.assertEqual(list(types_["required"]), [
            "video",
            "enable_super_resolution",
            "spatial_mode",
            "enable_neural_rendering",
            "nr_profile",
            "nr_intensity",
            "enable_frame_interpolation",
            "output_codec",
            "quality",
        ])
        self.assertEqual(MyVideoEnhanceStream.RETURN_TYPES, ("VIDEO", "INT", "STRING"))
        self.assertEqual(
            MyVideoEnhanceStream.RETURN_NAMES, ("video", "interpolation_multiplier", "status")
        )
        codec = types_["required"]["output_codec"]
        self.assertEqual(list(codec[0]), list(OUTPUT_CODECS))
        self.assertEqual(codec[1]["default"], "libx264")
        quality = types_["required"]["quality"][1]
        self.assertEqual((quality["default"], quality["min"], quality["max"], quality["step"]), (18, 0, 51, 1))

    def test_the_enhancement_controls_are_the_image_node_ones(self) -> None:
        stream_types = MyVideoEnhanceStream.INPUT_TYPES()
        image_types = MyVideoEnhance.INPUT_TYPES()
        # Same widgets and same defaults as My Video Enhance, minus its IMAGE input.
        for name, widget in image_types["required"].items():
            if name == "images":
                continue
            self.assertEqual(stream_types["required"][name], widget, name)
        for name, widget in image_types["optional"].items():
            if name == "stage_order":
                continue
            self.assertEqual(stream_types["optional"][name], widget, name)
        # The stream default is interpolation first, which the pipeline then enhances.
        stage_order = stream_types["optional"]["stage_order"]
        self.assertEqual(list(stage_order[0]), list(STAGE_ORDERS))
        self.assertEqual(stage_order[1]["default"], STAGE_ORDER_VFI_THEN_DLSS)
        self.assertTrue(stage_order[1]["advanced"])

    def test_the_v3_schema_exposes_native_video_sockets(self) -> None:
        try:
            from comfy_api.latest import io
        except ImportError:
            self.skipTest("comfy_api is not installed in this interpreter")
        schema = MyVideoEnhanceStream.define_schema()
        self.assertEqual(schema.node_id, "MyVideoEnhanceStream")
        self.assertEqual(schema.display_name, "My Video Enhance Stream")
        self.assertTrue(schema.is_experimental)
        self.assertIsInstance(schema.inputs[0], io.Video.Input)
        self.assertIsInstance(schema.outputs[0], io.Video.Output)
        self.assertEqual(
            [output.display_name for output in schema.outputs],
            ["video", "interpolation_multiplier", "status"],
        )
        classic = MyVideoEnhanceStream.INPUT_TYPES()
        self.assertEqual(
            [item.id for item in schema.inputs],
            list(classic["required"]) + list(classic["optional"]),
        )
        quality = next(item for item in schema.inputs if item.id == "quality")
        self.assertEqual((quality.default, quality.min, quality.max), (18, 0, 51))


class WidgetValidationTests(_StreamTestCase):
    def test_bad_widget_values_fail_before_any_tool_runs(self) -> None:
        for widgets, error in (
            ({"output_codec": "libvpx"}, ValueError),
            ({"quality": 52}, ValueError),
            ({"quality": 18.5}, TypeError),
            ({"stage_order": "dlss_and_vfi_in_parallel"}, ValueError),
            ({"vfi_precision": "fp8"}, ValueError),
            ({"motion": "warp"}, ValueError),
            ({"channel_order": "GBR"}, ValueError),
        ):
            with self.subTest(widgets=widgets):
                video_io = _FakeVideoIO(_spec(self.root), self.log)
                self.assert_fails(error, _FileVideo(self.root / "source.mkv"), video_io, **widgets)
                self.assertEqual(video_io.probes, [])
                self.assertEqual(video_io.readers, [])
                self.assertEqual(self.temp_residue(), [])


class PassThroughTests(_StreamTestCase):
    def test_no_stage_enabled_returns_the_input_untouched(self) -> None:
        video = _UntouchableVideo()
        video_io = mock.Mock(name="video_io")
        result = self.run_node(video, video_io, folder_paths=self.exploding_folder_paths())
        self.assertIs(result.output, video)
        self.assertEqual(result.multiplier, 1)
        self.assertIn("pass-through", result.status)
        # Nothing was probed, decoded, encoded or remuxed.
        self.assertEqual(video_io.mock_calls, [])
        self.assertEqual(result.calls, [])

    def test_a_pass_through_does_not_even_import_the_ffmpeg_layer(self) -> None:
        video = _UntouchableVideo()
        # `run_node` patches `_video_io` to raise, so reaching it fails the test.
        result = self.run_node(video, None, folder_paths=self.exploding_folder_paths())
        self.assertIs(result.output, video)


class SourceContractTests(_StreamTestCase):
    def test_an_active_trim_window_is_rejected_before_the_source_is_read(self) -> None:
        video = _FileVideo(self.root / "source.mkv", trim=(1.5, 2.0))
        error = self.assert_fails(ValueError, video, None, enable_frame_interpolation=True)
        self.assertIn("trim window", str(error))
        self.assertIn("1.5", str(error))
        self.assertIn("2.0", str(error))
        # Rejected on the cheap check alone: the source is neither read nor probed.
        self.assertEqual(video.stream_calls, 0)
        self.assertEqual(self.temp_residue(), [])

    def test_an_in_memory_video_is_rejected(self) -> None:
        video = _MemoryVideo()
        error = self.assert_fails(ValueError, video, None, enable_frame_interpolation=True)
        self.assertIn("local video file", str(error))
        self.assertIn("BytesIO", str(error))
        self.assertEqual(self.temp_residue(), [])

    def test_a_video_without_the_native_api_is_rejected(self) -> None:
        error = self.assert_fails(ValueError, object(), None, enable_frame_interpolation=True)
        self.assertIn("native file-backed VIDEO", str(error))
        self.assertIn("object", str(error))

    def test_a_probe_failure_propagates_before_any_file_is_owned(self) -> None:
        source = _spec(self.root)
        video_io = _FakeVideoIO(
            source, self.log, probe_error=VideoIOError(f"{source.path} is variable frame rate")
        )
        error = self.assert_fails(VideoIOError, _FileVideo(source.path), video_io, enable_frame_interpolation=True)
        self.assertIn("variable frame rate", str(error))
        self.assertEqual(len(video_io.probes), 1)
        self.assertEqual(video_io.readers, [])
        self.assertEqual(video_io.writers, [])
        self.assertEqual(self.temp_residue(), [])

    def test_a_rejected_source_does_not_create_the_temp_root(self) -> None:
        source = _spec(self.root)
        video_io = _FakeVideoIO(
            source, self.log, probe_error=VideoIOError(f"{source.path} is variable frame rate")
        )
        missing = self.root / "missing" / "nested"
        self.assert_fails(
            VideoIOError,
            _FileVideo(source.path),
            video_io,
            folder_paths=self.folder_paths(temp_directory=missing),
            enable_frame_interpolation=True,
        )
        # The temp root is only created once the source is accepted.
        self.assertFalse(missing.exists())


class StreamWiringTests(_StreamTestCase):
    def test_one_stage_streams_frame_by_frame_into_the_encoder(self) -> None:
        source = _spec(self.root, frames=5, width=8, height=6)
        video_io = _FakeVideoIO(source, self.log)
        result = self.run_node(_FileVideo(source.path), video_io, enable_frame_interpolation=True)

        self.assertEqual(video_io.probes, [(source.path, "ffprobe", comfy_interrupt)])
        call = result.calls[0]
        reader = video_io.readers[0]
        # The pipeline pulls the reader itself, so no decoded clip is ever held.
        self.assertIs(call.source, reader)
        self.assertEqual(call.spec, FrameSpec(count=5, height=6, width=8))
        self.assertEqual(call.plan.stages, ("vfi",))
        self.assertIsNone(call.kwargs["temp_directory"])  # one stage needs no spool
        self.assertEqual(call.kwargs["vfi"].precision, "fp32")
        self.assertEqual(call.kwargs["vfi"].models_dir, "/models")
        self.assertEqual(call.kwargs["vfi"].ds_factor, 1.0)
        self.assertEqual(call.kwargs["dlss"].motion_mode, "optical_flow")
        self.assertEqual(call.kwargs["dlss"].channel_order, "auto")
        self.assertIsNotNone(call.kwargs["dlss"].memory_hooks)
        self.assertIs(call.kwargs["interrupt"], comfy_interrupt)

        writer = video_io.writers[0]
        # The pipeline's frames reach the encoder in order, one at a time.
        self.assertEqual(
            [value for _index, _shape, value in writer.written], [float(index) for index in range(9)]
        )
        self.assertEqual((writer.width, writer.height), (8, 6))
        self.assertEqual(writer.fps, Fraction(60000, 1001))  # rational fps x2
        self.assertEqual(writer.expected_frames, 9)  # (5 - 1) * 2 + 1
        self.assertEqual((writer.codec, writer.quality), ("libx264", 18))
        self.assertIs(writer.interrupt, comfy_interrupt)
        self.assertEqual(writer.path.parent, self.comfy_temp)
        self.assertEqual([index for index, _shape, _value in writer.written], list(range(9)))
        self.assertEqual(writer.written[0][1], (6, 8, 3))
        # Every frame leaves the reader and reaches the encoder immediately: no
        # full read pass and no list of frames anywhere in between.
        self.assertEqual(self.log, [
            ("read", 0), ("write", 0), ("read", 1), ("write", 1), ("read", 2), ("write", 2),
            ("read", 3), ("write", 3), ("read", 4), ("write", 4),
            ("write", 5), ("write", 6), ("write", 7), ("write", 8),
        ])
        self.assertEqual((reader.entered, reader.exited), (1, 1))
        self.assertEqual((writer.entered, writer.exited), (1, 1))
        self.assertEqual(result.bars[0].total, 4)  # one VFI pair per gap
        self.assertEqual(result.bars[0].updates, [1, 2, 3, 4])

        spec_arg, video_only, output, interrupt = video_io.remuxes[0]
        self.assertIs(spec_arg, source)
        self.assertIs(interrupt, comfy_interrupt)
        self.assertEqual(video_only, writer.path)
        self.assertFalse(video_only.exists())  # the node deleted its residue
        self.assertTrue(output.exists())
        self.assertEqual(result.paths, [str(output)])
        self.assertEqual(result.output, f"VIDEO:{output}")
        self.assertEqual(result.multiplier, 2)
        self.assertIn("frames=9", result.status)
        self.assertIn("fps_multiplier=2", result.status)
        self.assertIn("output_fps=60000/1001", result.status)
        self.assertIn("codec=libx264 quality=18", result.status)
        self.assertIn("source audio remuxed", result.status)
        self.assertEqual(self.temp_residue(), [output.name])

    def test_two_stages_spool_the_intermediate_under_the_temp_root(self) -> None:
        source = _spec(self.root, frames=5, width=8, height=6)
        video_io = _FakeVideoIO(source, self.log)
        result = self.run_node(
            _FileVideo(source.path),
            video_io,
            enable_super_resolution=True,
            enable_frame_interpolation=True,
            stage_order=STAGE_ORDER_DLSS_THEN_VFI,
        )
        call = result.calls[0]
        self.assertEqual(call.plan.stages, ("dlss", "vfi"))
        self.assertEqual(
            call.kwargs["temp_directory"], str(self.comfy_temp / FRAME_STORE_DIRECTORY)
        )
        writer = video_io.writers[0]
        self.assertEqual((writer.width, writer.height), (16, 12))  # 2.0x DLAA/SR
        self.assertEqual(writer.expected_frames, 9)
        self.assertEqual(writer.fps, Fraction(30000, 1001) * 2)
        self.assertEqual(result.bars[0].total, 9)  # 5 DLSS frames + 4 VFI pairs
        self.assertEqual(result.multiplier, 2)
        self.assertEqual(self.temp_residue(), [video_io.remuxes[0][2].name])

    def test_the_default_order_interpolates_first_and_then_enhances(self) -> None:
        source = _spec(self.root, frames=5, width=8, height=6)
        video_io = _FakeVideoIO(source, self.log)
        result = self.run_node(
            _FileVideo(source.path),
            video_io,
            enable_super_resolution=True,
            enable_frame_interpolation=True,
        )
        call = result.calls[0]
        self.assertEqual(call.plan.stages, ("vfi", "dlss"))
        self.assertEqual(
            call.kwargs["temp_directory"], str(self.comfy_temp / FRAME_STORE_DIRECTORY)
        )
        writer = video_io.writers[0]
        self.assertEqual((writer.width, writer.height), (16, 12))
        self.assertEqual(writer.expected_frames, 9)
        self.assertEqual(result.bars[0].total, 13)  # 4 VFI pairs, then 9 DLSS frames
        self.assertEqual(result.paths, [str(video_io.remuxes[0][2])])

    def test_a_single_frame_clip_keeps_its_frame_rate(self) -> None:
        source = _spec(self.root, frames=1, width=8, height=6, fps=Fraction(25))
        video_io = _FakeVideoIO(source, self.log)
        result = self.run_node(_FileVideo(source.path), video_io, enable_frame_interpolation=True)
        writer = video_io.writers[0]
        # One frame has no pair to interpolate, so neither fps nor count is scaled.
        self.assertEqual(writer.fps, Fraction(25))
        self.assertEqual(writer.expected_frames, 1)
        self.assertEqual(result.multiplier, 1)
        self.assertIn("fps_multiplier=1", result.status)

    def test_encoder_and_runtime_widgets_are_routed(self) -> None:
        source = _spec(self.root, frames=3, width=8, height=6, has_audio=False)
        video_io = _FakeVideoIO(source, self.log)
        other_temp = self.root / "other_temp"
        other_temp.mkdir()
        result = self.run_node(
            _FileVideo(source.path),
            video_io,
            folder_paths=self.folder_paths(temp_directory=other_temp),
            enable_super_resolution=True,
            output_codec="h264_nvenc",
            quality=23,
            runtime_dir="/runtime",
            wine_prefix="Z:/wine",
            worker_timeout=42.0,
            vfi_ds_factor=0.5,
            motion="none",
            scene_cut_threshold=0.7,
            channel_order="BGRA",
        )
        writer = video_io.writers[0]
        self.assertEqual((writer.codec, writer.quality), ("h264_nvenc", 23))
        self.assertEqual(writer.path.parent, other_temp)
        self.assertEqual(writer.fps, source.fps)  # no VFI, so the fps is untouched
        self.assertEqual(writer.expected_frames, 3)
        self.assertEqual((writer.width, writer.height), (16, 12))
        call = result.calls[0]
        self.assertEqual(call.plan.stages, ("dlss",))
        self.assertIsNone(call.kwargs["temp_directory"])
        self.assertEqual(call.kwargs["vfi"].ds_factor, 0.5)
        self.assertEqual(call.kwargs["dlss"].runtime_dir, "/runtime")
        self.assertEqual(call.kwargs["dlss"].wine_prefix, "Z:/wine")
        self.assertEqual(call.kwargs["dlss"].timeout, 42.0)
        self.assertEqual(call.kwargs["dlss"].motion_mode, "none")
        self.assertEqual(call.kwargs["dlss"].scene_cut_threshold, 0.7)
        self.assertEqual(call.kwargs["dlss"].channel_order, "BGRA")
        # No audio: the encoded file is moved onto the output path.
        self.assertFalse(video_io.remuxes[0][1].exists())
        self.assertTrue(video_io.remuxes[0][2].exists())
        self.assertIn("no source audio", result.status)

    def test_the_runtime_directory_defaults_to_the_models_directory(self) -> None:
        source = _spec(self.root)
        video_io = _FakeVideoIO(source, self.log)
        # The shared resolver imports `folder_paths` lazily for this fallback.
        with mock.patch.dict(os.environ, {RUNTIME_DIR_ENV: ""}):
            result = self.run_node(_FileVideo(source.path), video_io, enable_super_resolution=True)
        self.assertEqual(result.calls[0].kwargs["dlss"].runtime_dir, "/models/dlss5")

    def test_a_missing_temp_root_is_created_before_anything_is_encoded(self) -> None:
        # Comfy's temp directory need not exist yet (fresh install, cleaned temp).
        # FFmpeg opens the output by path, so the node has to create the root
        # before the writer starts, or the first frame fails to open the file.
        source = _spec(self.root)
        video_io = _FakeVideoIO(source, self.log)
        missing = self.root / "missing" / "nested"
        self.assertFalse(missing.exists())
        result = self.run_node(
            _FileVideo(source.path),
            video_io,
            folder_paths=self.folder_paths(temp_directory=missing),
            enable_frame_interpolation=True,
        )
        self.assertTrue(missing.is_dir())
        writer = video_io.writers[0]
        self.assertEqual(writer.path.parent, missing)
        self.assertFalse(writer.path.exists())  # the video-only residue is gone
        output = video_io.remuxes[0][2]
        self.assertEqual(result.paths, [str(output)])
        self.assertEqual(sorted(entry.name for entry in missing.iterdir()), [output.name])

    def test_every_run_owns_unique_temporary_paths(self) -> None:
        source = _spec(self.root)
        first = self.run_node(_FileVideo(source.path), _FakeVideoIO(source, self.log), enable_frame_interpolation=True)
        second = self.run_node(_FileVideo(source.path), _FakeVideoIO(source, self.log), enable_frame_interpolation=True)
        owned = {
            first.io.writers[0].path,
            first.io.remuxes[0][2],
            second.io.writers[0].path,
            second.io.remuxes[0][2],
        }
        self.assertEqual(len(owned), 4)
        for path in owned:
            self.assertEqual(path.parent, self.comfy_temp)
            self.assertTrue(path.name.startswith(TEMP_PREFIX), path.name)
            self.assertEqual(path.suffix, ".mkv")


class CleanupTests(_StreamTestCase):
    def _assert_reaped(self, video_io) -> None:
        self.assertEqual((video_io.readers[0].entered, video_io.readers[0].exited), (1, 1))
        self.assertEqual((video_io.writers[0].entered, video_io.writers[0].exited), (1, 1))

    def test_a_failed_pipeline_reaps_both_processes_and_deletes_the_residue(self) -> None:
        source = _spec(self.root)
        video_io = _FakeVideoIO(source, self.log)
        error = self.assert_fails(
            RuntimeError,
            _FileVideo(source.path),
            video_io,
            enable_frame_interpolation=True,
            failure=RuntimeError("the VFI stage exploded"),
        )
        self.assertIn("the VFI stage exploded", str(error))
        self._assert_reaped(video_io)
        self.assertFalse(video_io.writers[0].path.exists())
        self.assertEqual(video_io.remuxes, [])
        self.assertEqual(self.temp_residue(), [])

    def test_a_failed_remux_keeps_nothing_behind(self) -> None:
        source = _spec(self.root)
        video_io = _FakeVideoIO(source, self.log, remux=_failing_remux)
        error = self.assert_fails(
            VideoIOError, _FileVideo(source.path), video_io, enable_frame_interpolation=True
        )
        self.assertIn("while remuxing", str(error))
        self._assert_reaped(video_io)
        # The video-only input the node owns is gone, and so is the remux's own
        # partial output (the real remux removes that one before it raises).
        self.assertFalse(video_io.remuxes[0][1].exists())
        self.assertFalse(video_io.remuxes[0][2].exists())
        self.assertEqual(self.temp_residue(), [])

    def test_a_failed_video_wrapper_deletes_the_encoded_output(self) -> None:
        source = _spec(self.root)
        video_io = _FakeVideoIO(source, self.log)

        def exploding_factory(path: str):
            raise RuntimeError("comfy_api is not available")

        error = self.assert_fails(
            RuntimeError,
            _FileVideo(source.path),
            video_io,
            enable_frame_interpolation=True,
            video_factory=exploding_factory,
        )
        self.assertIn("comfy_api is not available", str(error))
        self._assert_reaped(video_io)
        # The output became the node's own file once the remux handed it over.
        self.assertFalse(video_io.remuxes[0][2].exists())
        self.assertFalse(video_io.writers[0].path.exists())
        self.assertEqual(self.temp_residue(), [])

    def test_a_base_exception_cancel_cleans_up_like_any_other_failure(self) -> None:
        for cancel in (KeyboardInterrupt(), SystemExit("cancelled")):
            with self.subTest(cancel=type(cancel).__name__):
                self.log.clear()
                source = _spec(self.root)
                video_io = _FakeVideoIO(source, self.log)
                self.assert_fails(
                    type(cancel),
                    _FileVideo(source.path),
                    video_io,
                    enable_frame_interpolation=True,
                    failure=cancel,
                )
                self._assert_reaped(video_io)
                self.assertFalse(video_io.writers[0].path.exists())
                self.assertEqual(self.temp_residue(), [])


if __name__ == "__main__":  # pragma: no cover - convenience only
    unittest.main()
