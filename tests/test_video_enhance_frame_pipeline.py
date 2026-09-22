"""Shared sequential frame pipeline: specs, disk store, stage order, teardown.

The stage-level tests drive the production stages rather than stubs of them: the
real `iter_interpolate_offline` with GIMM stand-ins, and the real
`DlssStageStream` against the real-pipe fake DNR3 worker. That is what makes the
"drain stage 1, tear it down, then start stage 2" assertions meaningful.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np

from my_nodes.core.video_enhance import FEATURE_SR, gimm_vfi
from my_nodes.core.video_enhance import frame_pipeline as frame_pipeline_module
from my_nodes.core.video_enhance.dlss_stage import FrameValidationError, run_dlss_stage
from my_nodes.core.video_enhance.frame_pipeline import (
    FRAME_STORE_CHUNK_BYTES,
    DlssStageOptions,
    FramePipelineError,
    FrameSpec,
    FrameStore,
    FrameStoreError,
    VfiStageOptions,
    pipeline_specs,
    pipeline_step_total,
    run_frame_pipeline,
    stage_frame_spec,
    stage_step_counts,
    temp_frame_store,
)
from my_nodes.core.video_enhance.plan import STAGE_DLSS, STAGE_VFI, VideoEnhancePlan
from my_nodes.core.video_enhance.runtime import HostDriver

from .video_enhance_fixtures import (
    assert_process_gone,
    create_runtime_dir,
    fake_worker_command,
    read_report,
)


def _frame(height: int = 6, width: int = 4, value: float = 0.25) -> np.ndarray:
    frame = np.full((height, width, 3), value, dtype=np.float32)
    frame[..., 2] = 1.0 - value
    return frame


def _frame_bytes(spec: FrameSpec) -> int:
    """Exact float32 bytes of one frame of `spec`, to size tiny chunk caps."""
    return spec.height * spec.width * 3 * 4


def _stack(*frames: np.ndarray) -> np.ndarray:
    return np.stack(frames, axis=0).astype(np.float32)


def _average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left + right) / np.float32(2)


def _nearest(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """The fake DNR3 worker's deterministic nearest-neighbour enlargement."""
    in_h, in_w = rgb.shape[0], rgb.shape[1]
    rows = (np.arange(height) * in_h) // height
    cols = (np.arange(width) * in_w) // width
    return rgb[np.ix_(rows, cols)]


def _interpolate_frames(frames: np.ndarray) -> np.ndarray:
    """The pair assembly both the GIMM stage and its stand-ins produce."""
    produced = [frames[0]]
    for left, right in zip(frames, frames[1:]):
        produced.append(_average(left, right))
        produced.append(right)
    return np.stack(produced)


def _vfi_options() -> VfiStageOptions:
    return VfiStageOptions(precision="fp32", models_dir="/models")


def _dlss_options(**overrides) -> DlssStageOptions:
    settings = dict(
        runtime_dir="/runtime", motion_mode="none", scene_cut_threshold=0.2
    )
    settings.update(overrides)
    return DlssStageOptions(**settings)


class FrameSpecTests(unittest.TestCase):
    def test_shape_and_byte_size_are_the_float32_image_batch(self) -> None:
        spec = FrameSpec(count=3, height=4, width=6)
        self.assertEqual(spec.shape, (3, 4, 6, 3))
        self.assertEqual(spec.nbytes, 3 * 4 * 6 * 3 * 4)

    def test_invalid_values_are_rejected(self) -> None:
        for name in ("count", "height", "width"):
            with self.subTest(name=name, value=0):
                with self.assertRaises(ValueError):
                    FrameSpec(**{"count": 1, "height": 1, "width": 1, name: 0})
            with self.subTest(name=name, value=1.0):
                with self.assertRaises(TypeError):
                    FrameSpec(**{"count": 1, "height": 1, "width": 1, name: 1.0})
            with self.subTest(name=name, value=True):
                with self.assertRaises(TypeError):
                    FrameSpec(**{"count": 1, "height": 1, "width": 1, name: True})

    def test_spec_is_immutable_and_hashable(self) -> None:
        spec = FrameSpec(count=2, height=4, width=4)
        with self.assertRaises(Exception):
            spec.count = 3
        self.assertEqual(len({spec, FrameSpec(count=2, height=4, width=4)}), 1)


class PipelineSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = FrameSpec(count=3, height=6, width=4)

    def test_pass_through_plan_has_no_stage_and_no_store(self) -> None:
        specs = pipeline_specs(self.source, VideoEnhancePlan())
        self.assertEqual(specs.stages, ())
        self.assertEqual(specs.final, self.source)
        self.assertIsNone(specs.intermediate)
        self.assertFalse(specs.staged)
        self.assertEqual(pipeline_step_total(self.source, VideoEnhancePlan()), 0)

    def test_dlss_only_scales_the_dimensions_and_keeps_the_count(self) -> None:
        plan = VideoEnhancePlan(enable_super_resolution=True, sr_scale=2.0)
        specs = pipeline_specs(self.source, plan)
        self.assertEqual(specs.stages, (STAGE_DLSS,))
        self.assertEqual(specs.final, FrameSpec(count=3, height=12, width=8))
        self.assertIsNone(specs.intermediate)
        self.assertEqual(stage_step_counts(self.source, plan), ((STAGE_DLSS, 3),))
        self.assertEqual(pipeline_step_total(self.source, plan), 3)

    def test_native_dlss_keeps_the_frame_size(self) -> None:
        plan = VideoEnhancePlan(enable_neural_rendering=True)
        self.assertEqual(pipeline_specs(self.source, plan).final, self.source)

    def test_vfi_only_shares_both_boundaries(self) -> None:
        plan = VideoEnhancePlan(enable_frame_interpolation=True)
        specs = pipeline_specs(self.source, plan)
        self.assertEqual(specs.final, FrameSpec(count=5, height=6, width=4))
        self.assertEqual(pipeline_step_total(self.source, plan), 2)
        single = FrameSpec(count=1, height=6, width=4)
        self.assertEqual(stage_frame_spec(single, plan, STAGE_VFI), single)

    def test_two_stages_report_their_store_spec_in_both_orders(self) -> None:
        legacy = VideoEnhancePlan(
            enable_super_resolution=True, enable_frame_interpolation=True, sr_scale=2.0
        )
        specs = pipeline_specs(self.source, legacy)
        self.assertEqual(specs.stages, (STAGE_DLSS, STAGE_VFI))
        self.assertTrue(specs.staged)
        # DLSS first: the store holds 3 enhanced frames, VFI then shares boundaries.
        self.assertEqual(specs.intermediate, FrameSpec(count=3, height=12, width=8))
        self.assertEqual(specs.final, FrameSpec(count=5, height=12, width=8))
        self.assertEqual(pipeline_step_total(self.source, legacy), 3 + 2)

        reversed_plan = VideoEnhancePlan(
            enable_super_resolution=True,
            enable_frame_interpolation=True,
            sr_scale=2.0,
            stage_order="vfi_then_dlss",
        )
        specs = pipeline_specs(self.source, reversed_plan)
        self.assertEqual(specs.stages, (STAGE_VFI, STAGE_DLSS))
        # VFI first: the store holds 5 native frames, DLSS then scales each of them.
        self.assertEqual(specs.intermediate, FrameSpec(count=5, height=6, width=4))
        self.assertEqual(specs.final, FrameSpec(count=5, height=12, width=8))
        self.assertEqual(pipeline_step_total(self.source, reversed_plan), 2 + 5)

    def test_stage_order_does_not_change_a_single_stage(self) -> None:
        for order in ("dlss_then_vfi", "vfi_then_dlss"):
            with self.subTest(order=order):
                dlss = VideoEnhancePlan(enable_super_resolution=True, stage_order=order)
                self.assertEqual(dlss.stages, (STAGE_DLSS,))
                self.assertEqual(pipeline_step_total(self.source, dlss), 3)
                vfi = VideoEnhancePlan(enable_frame_interpolation=True, stage_order=order)
                self.assertEqual(vfi.stages, (STAGE_VFI,))
                self.assertEqual(pipeline_step_total(self.source, vfi), 2)

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(FramePipelineError):
            stage_frame_spec(self.source, VideoEnhancePlan(), "denoise")


class TempFrameStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def _write_frames(self, store: FrameStore, frames: np.ndarray) -> None:
        for index in range(frames.shape[0]):
            store.write(index, frames[index])

    def test_store_is_one_float32_file_deleted_when_the_block_exits(self) -> None:
        spec = FrameSpec(count=2, height=3, width=2)
        frames = np.arange(2 * 3 * 2 * 3, dtype=np.float32).reshape(spec.shape)
        with temp_frame_store(spec, self.directory) as store:
            self.assertIsInstance(store, FrameStore)
            self.assertEqual(store.spec, spec)
            self.assertEqual(store.nbytes, spec.nbytes)
            self._write_frames(store, frames)
            store.finish()
            # The intermediate really lives on disk: the raw file holds the frames.
            files = list(self.directory.iterdir())
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].stat().st_size, spec.nbytes)
            np.testing.assert_array_equal(
                np.fromfile(files[0], dtype=np.float32).reshape(spec.shape), frames
            )
            np.testing.assert_array_equal(np.stack(list(store)), frames)
            self.assertEqual(store.open_mappings, 0)
        self.assertEqual(os.listdir(self.directory), [])

    def test_every_mapping_is_released_when_the_block_exits(self) -> None:
        spec = FrameSpec(count=4, height=2, width=2)
        frame = _frame(height=2, width=2)
        with temp_frame_store(spec, self.directory, chunk_bytes=2 * _frame_bytes(spec)) as store:
            store.write(0, frame)
            self.assertEqual(store.open_mappings, 1)
        self.assertEqual(store.open_mappings, 0)
        self.assertEqual(store.mapped_bytes, 0)
        self.assertEqual(os.listdir(self.directory), [])

    def test_store_is_deleted_on_an_exception_inside_the_block(self) -> None:
        spec = FrameSpec(count=1, height=2, width=2)
        with self.assertRaisesRegex(RuntimeError, "stage exploded"):
            with temp_frame_store(spec, self.directory) as store:
                store.write(0, _frame(height=2, width=2))
                raise RuntimeError("stage exploded")
        # The failure path still unmapped the file before deleting it.
        self.assertEqual(store.open_mappings, 0)
        self.assertEqual(os.listdir(self.directory), [])

    def test_store_is_deleted_on_a_base_exception_cancel(self) -> None:
        class Cancel(BaseException):
            pass

        spec = FrameSpec(count=1, height=2, width=2)
        with self.assertRaises(Cancel):
            with temp_frame_store(spec, self.directory) as store:
                store.write(0, _frame(height=2, width=2))
                raise Cancel()
        self.assertEqual(store.open_mappings, 0)
        self.assertEqual(os.listdir(self.directory), [])

    def test_a_failed_preallocation_leaves_no_file(self) -> None:
        spec = FrameSpec(count=2, height=2, width=2)
        with mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.os.ftruncate",
            side_effect=OSError(27, "File too large"),
        ):
            with self.assertRaises(OSError):
                with temp_frame_store(spec, self.directory):
                    self.fail("a store without a preallocated file must not be yielded")
        self.assertEqual(os.listdir(self.directory), [])

    def test_a_failed_fd_close_leaves_no_file(self) -> None:
        spec = FrameSpec(count=2, height=2, width=2)
        with mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.os.close",
            side_effect=OSError(9, "Bad file descriptor"),
        ):
            with self.assertRaises(OSError):
                with temp_frame_store(spec, self.directory):
                    self.fail("a store whose fd cannot be closed must not be yielded")
        self.assertEqual(os.listdir(self.directory), [])

    def test_an_invalid_chunk_cap_leaves_no_file(self) -> None:
        spec = FrameSpec(count=2, height=2, width=2)
        with self.assertRaises(FrameStoreError) as raised:
            with temp_frame_store(spec, self.directory, chunk_bytes=0):
                self.fail("an invalid chunk cap must not yield a store")
        self.assertIn("chunk cap", str(raised.exception))
        self.assertEqual(os.listdir(self.directory), [])

    def test_a_base_exception_from_the_constructor_leaves_no_file(self) -> None:
        class Cancel(BaseException):
            pass

        spec = FrameSpec(count=2, height=2, width=2)
        with mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.FrameStore",
            side_effect=Cancel(),
        ):
            with self.assertRaises(Cancel):
                with temp_frame_store(spec, self.directory):
                    self.fail("a store that cannot be constructed must not be yielded")
        self.assertEqual(os.listdir(self.directory), [])

    def test_store_preflight_reports_required_and_available_bytes(self) -> None:
        spec = FrameSpec(count=1000, height=1000, width=1000)
        usage = mock.Mock(free=spec.nbytes - 1)
        with mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.shutil.disk_usage",
            return_value=usage,
        ):
            with self.assertRaises(FrameStoreError) as raised:
                with temp_frame_store(spec, self.directory):
                    self.fail("the store must not be created without free space")
        message = str(raised.exception)
        self.assertIn(str(spec.nbytes), message)
        self.assertIn(str(usage.free), message)
        self.assertEqual(os.listdir(self.directory), [])

    def test_store_requires_a_directory(self) -> None:
        with self.assertRaises(FrameStoreError):
            with temp_frame_store(FrameSpec(count=1, height=1, width=1), None):
                self.fail("a missing temporary directory must not yield a store")


class _StubRawMapping:
    """The `_mmap` half of a mapping: records the close, can fail it on demand."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False

    def close(self) -> None:
        self.closed = True
        if self.error is not None:
            raise self.error


class _StubMapping:
    """The two members `_close_mapping` uses, with injectable failures."""

    def __init__(
        self, *, flush_error: Exception | None = None, close_error: Exception | None = None
    ) -> None:
        self.flush_error = flush_error
        self.flushes = 0
        self.raw = _StubRawMapping(close_error)
        self._mmap = self.raw

    def flush(self) -> None:
        self.flushes += 1
        if self.flush_error is not None:
            raise self.flush_error


class _FlushControl:
    """Counts flush attempts of `_flush_failing_memmap` and fails them on demand."""

    def __init__(self) -> None:
        self.failing = False
        self.attempts = 0


def _flush_failing_memmap() -> tuple[type[np.memmap], _FlushControl]:
    """A `np.memmap` stand-in whose flush fails once the returned flag is set."""
    control = _FlushControl()

    class _Memmap(np.memmap):
        def flush(self) -> None:
            control.attempts += 1
            if control.failing:
                raise OSError(28, "No space left on device")
            super().flush()

    return _Memmap, control


def _close_failing_memmap() -> tuple[type[np.memmap], list[bool]]:
    """A `np.memmap` stand-in whose raw close fails once the returned flag is set."""
    failing = [False]

    class _Memmap(np.memmap):
        def __new__(cls, *args, **kwargs):
            array = super().__new__(cls, *args, **kwargs)
            if failing[0]:
                array._mmap = _StubRawMapping(OSError(5, "Input/output error"))
            return array

    return _Memmap, failing


class FrameStoreChunkTests(unittest.TestCase):
    """The chunked mapping: boundaries, values, mapping lifetime and misuse."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def _frames(self, count: int, height: int = 2, width: int = 2) -> np.ndarray:
        return np.stack(
            [_frame(height=height, width=width, value=index / 10.0) for index in range(count)]
        )

    def _open(self, spec: FrameSpec, *, frames_per_chunk: int):
        """A store whose cap covers exactly `frames_per_chunk` frames."""
        return temp_frame_store(
            spec, self.directory, chunk_bytes=frames_per_chunk * _frame_bytes(spec)
        )

    def test_write_and_read_across_many_chunks_keeps_order_and_values(self) -> None:
        spec = FrameSpec(count=7, height=2, width=2)
        frames = self._frames(7)
        with self._open(spec, frames_per_chunk=3) as store:
            self.assertEqual(store.frames_per_chunk, 3)
            self.assertEqual(store.chunk_bytes, 3 * _frame_bytes(spec))
            peak_mapped = 0
            for index in range(spec.count):
                store.write(index, frames[index])
                peak_mapped = max(peak_mapped, store.mapped_bytes)
                self.assertLessEqual(store.open_mappings, 1)
            store.finish()
            read: list[np.ndarray] = []
            for frame in store:
                read.append(frame)
                peak_mapped = max(peak_mapped, store.mapped_bytes)
                self.assertLessEqual(store.open_mappings, 1)
        self.assertEqual(len(read), spec.count)
        for index, frame in enumerate(read):
            np.testing.assert_array_equal(frame, frames[index])
        # Nothing ever mapped more than one chunk, and that chunk is capped.
        self.assertEqual(peak_mapped, 3 * _frame_bytes(spec))
        self.assertLessEqual(peak_mapped, store.chunk_bytes)
        self.assertEqual(store.open_mappings, 0)
        self.assertEqual(os.listdir(self.directory), [])

    def test_a_tiny_cap_still_maps_whole_frames(self) -> None:
        spec = FrameSpec(count=3, height=2, width=2)
        with self._open(spec, frames_per_chunk=1) as store:
            self.assertEqual(store.frames_per_chunk, 1)
        with temp_frame_store(spec, self.directory, chunk_bytes=1) as store:
            # A cap below one frame would never make progress, so one frame wins.
            self.assertEqual(store.frames_per_chunk, 1)
            self.assertEqual(store.chunk_bytes, _frame_bytes(spec))

    def test_the_exported_cap_is_used_when_no_cap_is_given(self) -> None:
        spec = FrameSpec(count=50, height=2, width=2)
        with mock.patch.object(
            frame_pipeline_module, "FRAME_STORE_CHUNK_BYTES", 5 * _frame_bytes(spec)
        ):
            with temp_frame_store(spec, self.directory) as store:
                self.assertEqual(store.frames_per_chunk, 5)
                self.assertEqual(store.chunk_bytes, 5 * _frame_bytes(spec))
        self.assertGreater(FRAME_STORE_CHUNK_BYTES, 0)
        self.assertIs(
            frame_pipeline_module.FRAME_STORE_CHUNK_BYTES,
            FRAME_STORE_CHUNK_BYTES,
        )

    def test_a_read_frame_stays_valid_after_its_chunk_is_unmapped(self) -> None:
        spec = FrameSpec(count=6, height=2, width=2)
        frames = self._frames(6)
        with self._open(spec, frames_per_chunk=2) as store:
            for index in range(spec.count):
                store.write(index, frames[index])
            store.finish()
            # Keep the first frame while the remaining chunks are mapped and
            # unmapped: it must not be a view into the vanished mapping.
            iterator = store.iter_frames()
            first = next(iterator)
            self.assertEqual(store.open_mappings, 1)
            for _ in range(spec.count - 1):
                next(iterator)
            np.testing.assert_array_equal(first, frames[0])
            self.assertIsNone(first.base)
            # A copy stays writable and independent after the chunks are gone.
            retained = first.copy()
            first[0, 0, 0] = -1.0
            np.testing.assert_array_equal(retained, frames[0])
            read = list(store)
            self.assertEqual(len(read), spec.count)
            self.assertEqual(store.open_mappings, 0)
            for index, frame in enumerate(read):
                np.testing.assert_array_equal(frame, frames[index])

    def test_each_iteration_maps_and_releases_one_chunk_at_a_time(self) -> None:
        spec = FrameSpec(count=5, height=2, width=2)
        frames = self._frames(5)
        chunk = 2 * _frame_bytes(spec)
        with self._open(spec, frames_per_chunk=2) as store:
            for index in range(spec.count):
                store.write(index, frames[index])
            store.finish()
            seen: list[int] = []
            for frame in store:
                seen.append(store.mapped_bytes)
                self.assertEqual(store.open_mappings, 1)
                self.assertLessEqual(store.mapped_bytes, store.chunk_bytes)
            # Two full chunks, then one mapping holding only the last frame.
            self.assertEqual(seen, [chunk, chunk, chunk, chunk, chunk // 2])
            self.assertEqual(store.open_mappings, 0)

    def test_an_abandoned_iteration_releases_its_chunk(self) -> None:
        spec = FrameSpec(count=4, height=2, width=2)
        with self._open(spec, frames_per_chunk=2) as store:
            for index in range(spec.count):
                store.write(index, _frame(height=2, width=2, value=index / 10.0))
            store.finish()
            # Closing the iterator releases the chunk it was holding.
            iterator = store.iter_frames()
            next(iterator)
            self.assertEqual(store.open_mappings, 1)
            iterator.close()
            self.assertEqual(store.open_mappings, 0)
            # So does the store cleanup when an iterator is simply dropped.
            abandoned = store.iter_frames()
            next(abandoned)
            self.assertEqual(store.open_mappings, 1)
        self.assertEqual(store.open_mappings, 0)
        self.assertEqual(store.mapped_bytes, 0)
        self.assertEqual(os.listdir(self.directory), [])

    def test_writes_are_sequential_and_bounded_by_the_frame_count(self) -> None:
        spec = FrameSpec(count=2, height=2, width=2)
        frame = _frame(height=2, width=2)
        with temp_frame_store(spec, self.directory) as store:
            store.write(0, frame)
            with self.assertRaises(FrameStoreError) as skipped:
                store.write(2, frame)
            self.assertIn("sequential", str(skipped.exception))
            self.assertEqual(store.frames_written, 1)
            with self.assertRaises(FrameStoreError) as repeated:
                store.write(0, frame)
            self.assertIn("sequential", str(repeated.exception))
            store.write(1, frame)
            with self.assertRaises(FrameStoreError) as extra:
                store.write(2, frame)
            self.assertIn("one too many", str(extra.exception))
            self.assertEqual(store.frames_written, 2)
            store.finish()
            with self.assertRaises(FrameStoreError) as sealed:
                store.write(0, frame)
            self.assertIn("finished", str(sealed.exception))

    def test_an_invalid_frame_is_rejected_before_it_reaches_disk(self) -> None:
        spec = FrameSpec(count=2, height=2, width=2)
        good = _frame(height=2, width=2)
        with temp_frame_store(spec, self.directory) as store:
            for bad, message in (
                (good.astype(np.float64), "float32"),
                (_frame(height=4, width=2), "(2, 2, 3)"),
                (np.full((2, 2), 1.0, dtype=np.float32), "(2, 2, 3)"),
            ):
                with self.assertRaises(FrameStoreError) as raised:
                    store.write(0, bad)
                self.assertIn(message, str(raised.exception))
            non_finite = good.copy()
            non_finite[0, 0, 0] = np.nan
            with self.assertRaises(FrameStoreError) as raised:
                store.write(0, non_finite)
            self.assertIn("non-finite", str(raised.exception))
            self.assertEqual(store.frames_written, 0)
            store.write(0, good)
            self.assertEqual(store.frames_written, 1)
            store.finish()

    def test_reading_before_finish_is_rejected(self) -> None:
        spec = FrameSpec(count=1, height=2, width=2)
        frame = _frame(height=2, width=2)
        with temp_frame_store(spec, self.directory) as store:
            store.write(0, frame)
            with self.assertRaises(FrameStoreError) as unfinished:
                list(store)
            self.assertIn("finished", str(unfinished.exception))
            self.assertEqual(store.open_mappings, 0)
            store.finish()
            self.assertEqual(len(list(store)), 1)

    def test_reading_a_store_with_a_missing_frame_is_rejected(self) -> None:
        spec = FrameSpec(count=2, height=2, width=2)
        frame = _frame(height=2, width=2)
        with temp_frame_store(spec, self.directory) as store:
            store.write(0, frame)
            store.finish()
            with self.assertRaises(FrameStoreError) as incomplete:
                list(store)
            self.assertIn("holds 1 of 2 frames", str(incomplete.exception))
            self.assertEqual(store.frames_written, 1)
            self.assertEqual(store.open_mappings, 0)

    def test_a_closed_store_refuses_more_work(self) -> None:
        spec = FrameSpec(count=1, height=2, width=2)
        frame = _frame(height=2, width=2)
        with temp_frame_store(spec, self.directory) as store:
            store.finish()
            store.close()
            self.assertTrue(store.closed)
            with self.assertRaises(FrameStoreError) as written:
                store.write(0, frame)
            self.assertIn("closed", str(written.exception))
            with self.assertRaises(FrameStoreError) as read:
                list(store)
            self.assertIn("closed", str(read.exception))
            store.close()

    def test_an_invalid_chunk_cap_is_rejected(self) -> None:
        spec = FrameSpec(count=1, height=2, width=2)
        for cap in (0, -1, 1.5, True, "big"):
            with self.subTest(cap=cap):
                with self.assertRaises(FrameStoreError):
                    FrameStore(self.directory / "unused.bin", spec, chunk_bytes=cap)
        self.assertEqual(os.listdir(self.directory), [])

    def test_mapped_bytes_never_grow_with_the_clip_length(self) -> None:
        # Ten times the frames at a fixed cap: the same bounded chunk is mapped.
        peaks: dict[int, int] = {}
        for count in (20, 200):
            spec = FrameSpec(count=count, height=2, width=2)
            with self._open(spec, frames_per_chunk=4) as store:
                peak = 0
                for index in range(count):
                    store.write(index, _frame(height=2, width=2, value=index / 100.0))
                    peak = max(peak, store.mapped_bytes)
                store.finish()
                for _ in store:
                    peak = max(peak, store.mapped_bytes)
                peaks[count] = peak
                self.assertEqual(store.open_mappings, 0)
        self.assertEqual(peaks[20], 4 * _frame_bytes(FrameSpec(count=20, height=2, width=2)))
        self.assertEqual(peaks[200], peaks[20])
        self.assertEqual(os.listdir(self.directory), [])

    def test_a_write_chunk_flush_failure_is_reported_and_still_releases(self) -> None:
        spec = FrameSpec(count=2, height=2, width=2)
        memmap, control = _flush_failing_memmap()
        with mock.patch.object(frame_pipeline_module.np, "memmap", memmap):
            with temp_frame_store(
                spec, self.directory, chunk_bytes=_frame_bytes(spec)
            ) as store:
                control.failing = True
                with self.assertRaises(FrameStoreError) as raised:
                    store.write(0, _frame(height=2, width=2))
                self.assertIn("No space left on device", str(raised.exception))
                self.assertEqual(control.attempts, 1)
                # The failing chunk was still closed, and the store knows it saw
                # the frame: a caller can report the run as failed deterministically.
                self.assertEqual(store.open_mappings, 0)
                self.assertEqual(store.mapped_bytes, 0)
                self.assertEqual(store.frames_written, 1)
        # And the file does not survive the failed run.
        self.assertEqual(os.listdir(self.directory), [])

    def test_a_finish_flush_failure_is_reported_and_still_releases(self) -> None:
        # Five of six frames: the live chunk is incomplete, which is the state
        # `finish()` has to release when a run is cut short.
        spec = FrameSpec(count=6, height=2, width=2)
        memmap, control = _flush_failing_memmap()
        with mock.patch.object(frame_pipeline_module.np, "memmap", memmap):
            with temp_frame_store(
                spec, self.directory, chunk_bytes=4 * _frame_bytes(spec)
            ) as store:
                for index in range(4):
                    store.write(index, _frame(height=2, width=2))  # full chunk
                store.write(4, _frame(height=2, width=2))  # live, incomplete chunk
                self.assertEqual(store.open_mappings, 1)
                control.failing = True
                with self.assertRaises(FrameStoreError) as raised:
                    store.finish()
                self.assertIn("No space left on device", str(raised.exception))
                self.assertEqual(store.open_mappings, 0)
                # The store is sealed, so the incomplete intermediate cannot be
                # read as if it were valid.
                with self.assertRaises(FrameStoreError):
                    list(store)
        self.assertEqual(os.listdir(self.directory), [])

    def test_read_chunks_are_mapped_read_only_and_release_strictly(self) -> None:
        spec = FrameSpec(count=3, height=2, width=2)
        modes: list[str] = []
        real_memmap = np.memmap

        def spy(path, dtype=None, mode="r+", **kwargs):
            modes.append(mode)
            return real_memmap(path, dtype=dtype, mode=mode, **kwargs)

        with mock.patch.object(frame_pipeline_module.np, "memmap", spy):
            with self._open(spec, frames_per_chunk=2) as store:
                for index in range(spec.count):
                    store.write(index, _frame(height=2, width=2))
                store.finish()
                self.assertEqual(len(list(store)), spec.count)
        self.assertEqual(modes[:2], ["r+", "r+"])
        self.assertEqual(set(modes[2:]), {"r"})

    def test_a_read_chunk_close_failure_is_reported(self) -> None:
        spec = FrameSpec(count=2, height=2, width=2)
        memmap, closing = _close_failing_memmap()
        with mock.patch.object(frame_pipeline_module.np, "memmap", memmap):
            with self._open(spec, frames_per_chunk=1) as store:
                for index in range(spec.count):
                    store.write(index, _frame(height=2, width=2))
                store.finish()
                closing[0] = True
                iterator = store.iter_frames()
                with self.assertRaises(FrameStoreError) as raised:
                    list(iterator)
                self.assertIn("Input/output error", str(raised.exception))
                self.assertEqual(store.open_mappings, 0)
        self.assertEqual(os.listdir(self.directory), [])

    def test_a_cancelled_read_does_not_report_a_close_failure(self) -> None:
        spec = FrameSpec(count=2, height=2, width=2)
        memmap, closing = _close_failing_memmap()
        with mock.patch.object(frame_pipeline_module.np, "memmap", memmap):
            with self._open(spec, frames_per_chunk=1) as store:
                for index in range(spec.count):
                    store.write(index, _frame(height=2, width=2))
                store.finish()
                closing[0] = True
                iterator = store.iter_frames()
                next(iterator)
                # Closing the iterator is the cancel path: GeneratorExit must win.
                iterator.close()
                self.assertEqual(store.open_mappings, 0)
        self.assertEqual(os.listdir(self.directory), [])

    def test_close_failures_of_a_read_chunk_are_reported_or_quiet(self) -> None:
        failing = _StubMapping(close_error=OSError(5, "Input/output error"))
        with self.assertRaises(FrameStoreError) as raised:
            frame_pipeline_module._close_mapping(failing, strict=True, flush=False)
        self.assertIn("Input/output error", str(raised.exception))
        self.assertTrue(failing.raw.closed)
        self.assertEqual(failing.flushes, 0)  # a read mapping needs no data flush

        quiet = _StubMapping(close_error=OSError(5, "Input/output error"))
        frame_pipeline_module._close_mapping(quiet, strict=False, flush=False)
        self.assertTrue(quiet.raw.closed)

    def test_a_flush_failure_is_reported_or_quiet_but_always_closes(self) -> None:
        failing = _StubMapping(flush_error=OSError(28, "No space left on device"))
        with self.assertRaises(FrameStoreError) as raised:
            frame_pipeline_module._close_mapping(failing, strict=True, flush=True)
        self.assertIn("No space left on device", str(raised.exception))
        self.assertTrue(failing.raw.closed)

        best_effort = _StubMapping(flush_error=OSError(28, "No space left on device"))
        frame_pipeline_module._close_mapping(best_effort, strict=False, flush=True)
        self.assertTrue(best_effort.raw.closed)
        self.assertEqual(best_effort.flushes, 1)

    def test_a_cleanup_failure_does_not_mask_the_caller_exception(self) -> None:
        class Cancel(BaseException):
            pass

        for error in (RuntimeError("stage exploded"), Cancel()):
            with self.subTest(error=type(error).__name__):
                spec = FrameSpec(count=6, height=2, width=2)
                memmap, control = _flush_failing_memmap()
                with mock.patch.object(frame_pipeline_module.np, "memmap", memmap):
                    with self.assertRaises(type(error)) as raised:
                        with temp_frame_store(
                            spec, self.directory, chunk_bytes=4 * _frame_bytes(spec)
                        ) as store:
                            for index in range(4):
                                store.write(index, _frame(height=2, width=2))
                            # A live, incomplete chunk is what cleanup has to
                            # release; from here on that release cannot flush.
                            store.write(4, _frame(height=2, width=2))
                            self.assertEqual(store.open_mappings, 1)
                            flushes_before = control.attempts
                            control.failing = True
                            raise error
                    # The caller's exception is the one that came out, and the
                    # failing cleanup flush really was attempted and swallowed.
                    self.assertIs(raised.exception, error)
                    self.assertEqual(control.attempts, flushes_before + 1)
                self.assertEqual(store.open_mappings, 0)
                self.assertEqual(os.listdir(self.directory), [])


RSS_CHILD = textwrap.dedent(
    """
    import json
    import resource
    import sys
    import tempfile

    import numpy as np

    from my_nodes.core.video_enhance.frame_pipeline import FrameSpec, temp_frame_store

    frames = int(sys.argv[1])
    cap = int(sys.argv[2])
    spec = FrameSpec(count=frames, height=32, width=32)
    frame = np.zeros((32, 32, 3), dtype=np.float32)
    with tempfile.TemporaryDirectory() as directory:
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mapped = 0
        with temp_frame_store(spec, directory, chunk_bytes=cap) as store:
            for index in range(frames):
                frame[0, 0, 0] = float(index % 7)
                store.write(index, frame)
                peak_mapped = max(peak_mapped, store.mapped_bytes)
            store.finish()
            read = 0
            for stored in store:
                read += 1
                peak_mapped = max(peak_mapped, store.mapped_bytes)
            open_at_exit = store.open_mappings
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(json.dumps({
            "raw": spec.nbytes,
            "delta_kib": after - before,
            "peak_mapped": peak_mapped,
            "read": read,
            "open_at_exit": open_at_exit,
        }))
    """
)


@unittest.skipUnless(sys.platform.startswith("linux"), "RSS accounting is Linux-specific")
class FrameStoreRssTests(unittest.TestCase):
    """Peak RSS must follow the chunk cap, not the preallocated file size.

    A whole-file mapping is correct but not bounded: Linux counts every touched
    page of it in RSS, so a long clip costs as much RAM as its raw file. Measured
    on this machine before the chunking fix: 24,576,000 bytes -> +23,708 KiB and
    245,760,000 bytes -> +239,596 KiB.
    """

    def _run_child(self, frames: int, cap: int) -> dict:
        """Measure one store in a fresh interpreter: no peak from a previous case."""
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "rss_child.py"
            script.write_text(RSS_CHILD, encoding="utf-8")
            root = str(Path(__file__).resolve().parents[1])
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [root, env.get("PYTHONPATH", "")]
            )
            done = subprocess.run(
                [sys.executable, str(script), str(frames), str(cap)],
                capture_output=True,
                text=True,
                env=env,
                cwd=directory,
                check=True,
            )
        return json.loads(done.stdout.strip().splitlines()[-1])

    def test_peak_rss_follows_the_chunk_cap_not_the_file_size(self) -> None:
        cap = 8 * 32 * 32 * 3 * 4  # eight frames: 98,304 bytes
        small = self._run_child(100, cap)
        large = self._run_child(3000, cap)
        self.assertEqual(small["read"], 100)
        self.assertEqual(large["read"], 3000)
        self.assertEqual(small["open_at_exit"], 0)
        self.assertEqual(large["open_at_exit"], 0)
        self.assertGreater(large["raw"], 100 * cap)
        # No mapping of either run ever covered more than the cap it was given.
        self.assertLessEqual(small["peak_mapped"], cap)
        self.assertLessEqual(large["peak_mapped"], cap)
        # A 30x longer clip costs the same peak RSS, within the cap plus slack,
        # instead of the ~36,000 KiB its raw file would have added when mapped whole.
        self.assertLess(large["delta_kib"] - small["delta_kib"], (cap + 4 * 1024 * 1024) // 1024)
        self.assertLess(large["delta_kib"], large["raw"] // 4 // 1024)


def _fake_vfi(events: list[str], on_open: Callable[[object], None] | None = None):
    """Interpolation stand-in with the real generator lifecycle."""

    def stage(
        source,
        count,
        *,
        precision,
        ds_factor,
        models_dir,
        node_mappings=None,
        load_device=None,
        memory_required=None,
        progress=None,
        interrupt=None,
    ):
        del precision, ds_factor, models_dir, node_mappings, load_device
        del memory_required, interrupt
        if on_open is not None:
            on_open(source)
        events.append("vfi-open")
        try:
            frames = [np.asarray(frame, dtype=np.float32) for frame in source]
            if len(frames) != count:
                raise AssertionError(f"vfi got {len(frames)} frames for count={count}")
            produced: list[np.ndarray] = [frames[0]]
            for left, right in zip(frames, frames[1:]):
                produced.append(_average(left, right))
                produced.append(right)
            for index, frame in enumerate(produced):
                if progress is not None and index % 2 == 1:
                    progress((index + 1) // 2, count - 1)
                yield frame
            events.append("vfi-drained")
        finally:
            events.append("vfi-closed")

    return stage


def _fake_dlss_stage(events: list[str], on_open: Callable[[object], None] | None = None):
    """DLSS stand-in with the real stream lifecycle (enter, iterate, exit)."""

    class _Stage:
        def __init__(self, plan, source, *, count, height, width, **kwargs):
            del plan
            self.source = source
            self.count = count
            self.height = height
            self.width = width
            self.channel_order = "RGBA"
            self.features = FEATURE_SR
            self.progress = kwargs.get("progress")

        def __enter__(self):
            if on_open is not None:
                on_open(self.source)
            events.append("dlss-open")
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append("dlss-closed")
            return False

        def __iter__(self):
            frames = [np.asarray(frame, dtype=np.float32) for frame in self.source]
            if len(frames) != self.count:
                raise AssertionError(f"dlss got {len(frames)} frames for count={self.count}")
            for index, frame in enumerate(frames):
                if self.progress is not None:
                    self.progress(index + 1, self.count)
                yield _nearest(frame, self.height * 2, self.width * 2)

    return _Stage


class PipelineRunTests(unittest.TestCase):
    """Stage order, disk staging, progress and the frame-count contract."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)
        self.events: list[str] = []
        self.written: dict[int, np.ndarray] = {}

    def _write(self, index: int, frame: np.ndarray) -> None:
        self.written[index] = np.array(frame)

    def _run(self, source, spec, plan, **kwargs):
        self.written = {}
        return run_frame_pipeline(
            source, spec, plan, self._write, temp_directory=self.directory, **kwargs
        )

    def test_pass_through_forwards_every_source_frame_in_order(self) -> None:
        frames = _stack(_frame(value=0.1), _frame(value=0.5))
        spec = FrameSpec(count=2, height=6, width=4)
        result = self._run(frames, spec, VideoEnhancePlan())
        self.assertEqual(result.stages, ())
        self.assertEqual(result.frame_count, 2)
        self.assertIsNone(result.channel_order)
        self.assertEqual(result.features, 0)
        self.assertEqual(sorted(self.written), [0, 1])
        np.testing.assert_array_equal(self.written[0], frames[0])
        np.testing.assert_array_equal(self.written[1], frames[1])

    def test_pass_through_rejects_a_short_and_a_long_source(self) -> None:
        spec = FrameSpec(count=2, height=6, width=4)
        with self.assertRaises(FramePipelineError) as short:
            self._run(_stack(_frame()), spec, VideoEnhancePlan())
        self.assertIn("expected 2 frames", str(short.exception))
        with self.assertRaises(FramePipelineError) as long:
            self._run(_stack(_frame(), _frame(), _frame()), spec, VideoEnhancePlan())
        self.assertIn("produced more", str(long.exception))

    def test_pass_through_rejects_a_wrong_frame_shape_or_dtype(self) -> None:
        spec = FrameSpec(count=1, height=6, width=4)
        with self.assertRaises(FramePipelineError):
            self._run(_stack(_frame(height=4, width=4)), spec, VideoEnhancePlan())
        with self.assertRaises(FramePipelineError) as raised:
            self._run(iter([_frame().astype(np.float64)]), spec, VideoEnhancePlan())
        self.assertIn("float32", str(raised.exception))

    def test_missing_stage_options_are_rejected(self) -> None:
        spec = FrameSpec(count=2, height=6, width=4)
        frames = _stack(_frame(), _frame())
        with self.assertRaises(FramePipelineError):
            self._run(frames, spec, VideoEnhancePlan(enable_frame_interpolation=True))
        with self.assertRaises(FramePipelineError):
            self._run(frames, spec, VideoEnhancePlan(enable_super_resolution=True))

    def test_two_stage_run_drains_and_closes_stage_one_before_stage_two(self) -> None:
        source = _stack(_frame(value=0.2), _frame(value=0.6))
        for stage_order, stages, second in (
            ("dlss_then_vfi", (STAGE_DLSS, STAGE_VFI), STAGE_VFI),
            ("vfi_then_dlss", (STAGE_VFI, STAGE_DLSS), STAGE_DLSS),
        ):
            with self.subTest(stage_order=stage_order):
                self.events.clear()
                observed: dict[str, object] = {}
                spec = FrameSpec(count=2, height=6, width=4)
                plan = VideoEnhancePlan(
                    enable_super_resolution=True,
                    enable_frame_interpolation=True,
                    stage_order=stage_order,
                )
                specs = pipeline_specs(spec, plan)
                assert specs.intermediate is not None

                def on_open(source) -> None:
                    observed["events"] = list(self.events)
                    observed["source"] = source
                    files = list(self.directory.iterdir())
                    observed["files"] = len(files)
                    observed["store"] = np.fromfile(files[0], dtype=np.float32).reshape(
                        specs.intermediate.shape
                    )

                vfi = _fake_vfi(self.events, on_open if second == STAGE_VFI else None)
                dlss = _fake_dlss_stage(self.events, on_open if second == STAGE_DLSS else None)
                with mock.patch(
                    "my_nodes.core.video_enhance.frame_pipeline.DlssStageStream", dlss
                ), mock.patch(
                    "my_nodes.core.video_enhance.frame_pipeline.iter_interpolate_offline", vfi
                ):
                    result = self._run(
                        source, spec, plan, dlss=_dlss_options(), vfi=_vfi_options()
                    )

                self.assertEqual(result.stages, stages)
                self.assertEqual(result.frame_count, specs.final.count)
                self.assertEqual(
                    (result.output_height, result.output_width),
                    (specs.final.height, specs.final.width),
                )
                self.assertEqual(
                    self.events,
                    ["dlss-open", "dlss-closed", "vfi-open", "vfi-drained", "vfi-closed"]
                    if stage_order == "dlss_then_vfi"
                    else ["vfi-open", "vfi-drained", "vfi-closed", "dlss-open", "dlss-closed"],
                )
                # Stage 2 opened with stage 1 already drained and closed, and the
                # intermediate it reads is that drained store on disk.
                self.assertEqual(
                    observed["events"],
                    ["dlss-open", "dlss-closed"]
                    if stage_order == "dlss_then_vfi"
                    else ["vfi-open", "vfi-drained", "vfi-closed"],
                )
                self.assertEqual(observed["files"], 1)
                # Stage 2 reads the disk store itself: the intermediate is not a
                # RAM-resident batch, it is streamed from the file in chunks.
                store = observed["source"]
                self.assertIsInstance(store, FrameStore)
                self.assertEqual(store.frames_written, specs.intermediate.count)
                # Stage 1 released every write mapping before stage 2 opened.
                self.assertEqual(store.open_mappings, 0)
                self.assertEqual(store.mapped_bytes, 0)
                np.testing.assert_allclose(
                    observed["store"], self._expected_store(source, specs, stage_order)
                )
                self.assertEqual(os.listdir(self.directory), [])

    def test_two_stage_run_holds_exact_values_across_many_tiny_chunks(self) -> None:
        source = _stack(_frame(value=0.1), _frame(value=0.5), _frame(value=0.9))
        for stage_order in ("dlss_then_vfi", "vfi_then_dlss"):
            with self.subTest(stage_order=stage_order):
                self.events.clear()
                spec = FrameSpec(count=3, height=6, width=4)
                plan = VideoEnhancePlan(
                    enable_super_resolution=True,
                    enable_frame_interpolation=True,
                    stage_order=stage_order,
                )
                specs = pipeline_specs(spec, plan)
                assert specs.intermediate is not None
                # Two frames per chunk: the store crosses several boundaries and
                # stage 2 keeps a frame across each one (GIMM keeps its previous
                # endpoint), which is exactly where a stale mapping would show.
                cap = 2 * _frame_bytes(specs.intermediate)
                vfi = _fake_vfi(self.events)
                dlss = _fake_dlss_stage(self.events)
                with mock.patch.object(
                    frame_pipeline_module, "FRAME_STORE_CHUNK_BYTES", cap
                ), mock.patch(
                    "my_nodes.core.video_enhance.frame_pipeline.DlssStageStream", dlss
                ), mock.patch(
                    "my_nodes.core.video_enhance.frame_pipeline.iter_interpolate_offline", vfi
                ):
                    result = self._run(
                        source, spec, plan, dlss=_dlss_options(), vfi=_vfi_options()
                    )

                self.assertEqual(result.frame_count, specs.final.count)
                intermediate = self._expected_store(source, specs, stage_order)
                # Two frames per chunk, so the store really crossed boundaries.
                self.assertGreater(-(-specs.intermediate.count // 2), 1)
                if stage_order == "dlss_then_vfi":
                    expected = _interpolate_frames(intermediate)
                else:
                    expected = np.stack(
                        [
                            _nearest(frame, specs.final.height, specs.final.width)
                            for frame in intermediate
                        ]
                    )
                self.assertEqual(len(expected), specs.final.count)
                for index, frame in enumerate(expected):
                    np.testing.assert_allclose(self.written[index], frame, rtol=0, atol=0)
                self.assertEqual(sorted(self.written), list(range(specs.final.count)))
                self.assertEqual(os.listdir(self.directory), [])

    def _expected_store(self, source, specs, stage_order: str) -> np.ndarray:
        """The expected store content, rebuilt with the same arithmetic as the stubs."""
        assert specs.intermediate is not None
        if stage_order == "dlss_then_vfi":
            return np.stack(
                [
                    _nearest(frame, specs.intermediate.height, specs.intermediate.width)
                    for frame in source
                ]
            )
        produced = [source[0]]
        for left, right in zip(source, source[1:]):
            produced.append(_average(left, right))
            produced.append(right)
        return np.stack(produced)

    def test_progress_is_cumulative_across_both_stages(self) -> None:
        seen: list[tuple[int, int]] = []
        plan = VideoEnhancePlan(
            enable_super_resolution=True, enable_frame_interpolation=True
        )
        spec = FrameSpec(count=3, height=6, width=4)
        total = pipeline_step_total(spec, plan)
        self.assertEqual(total, 5)
        with mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.DlssStageStream",
            _fake_dlss_stage(self.events),
        ), mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.iter_interpolate_offline",
            _fake_vfi(self.events),
        ):
            self._run(
                _stack(_frame(), _frame(value=0.5), _frame(value=0.9)),
                spec,
                plan,
                progress=lambda done, reported: seen.append((done, reported)),
                dlss=_dlss_options(),
                vfi=_vfi_options(),
            )
        self.assertEqual([done for done, _total in seen], list(range(1, total + 1)))
        self.assertEqual({reported for _done, reported in seen}, {total})

    def test_a_stage_that_produces_the_wrong_frame_count_is_rejected(self) -> None:
        spec = FrameSpec(count=2, height=6, width=4)
        plan = VideoEnhancePlan(enable_frame_interpolation=True)

        def short_vfi(source, count, **_kwargs):
            list(source)
            yield _frame()

        def long_vfi(source, count, **_kwargs):
            list(source)
            for _ in range(2 * count):
                yield _frame()

        with mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.iter_interpolate_offline", short_vfi
        ):
            with self.assertRaises(FramePipelineError) as raised:
                self._run(_stack(_frame(), _frame()), spec, plan, vfi=_vfi_options())
        self.assertIn("produced only", str(raised.exception))
        with mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.iter_interpolate_offline", long_vfi
        ):
            with self.assertRaises(FramePipelineError) as raised:
                self._run(_stack(_frame(), _frame()), spec, plan, vfi=_vfi_options())
        self.assertIn("produced more", str(raised.exception))

    def test_the_store_is_deleted_when_the_second_stage_fails(self) -> None:
        spec = FrameSpec(count=2, height=6, width=4)
        plan = VideoEnhancePlan(
            enable_super_resolution=True, enable_frame_interpolation=True
        )

        def exploding_vfi(source, count, **_kwargs):
            del source, count
            raise RuntimeError("vfi exploded")
            yield  # pragma: no cover - makes this a generator

        with mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.DlssStageStream",
            _fake_dlss_stage(self.events),
        ), mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.iter_interpolate_offline", exploding_vfi
        ):
            with self.assertRaisesRegex(RuntimeError, "vfi exploded"):
                self._run(
                    _stack(_frame(), _frame(value=0.5)),
                    spec,
                    plan,
                    dlss=_dlss_options(),
                    vfi=_vfi_options(),
                )
        self.assertEqual(os.listdir(self.directory), [])

    def test_the_store_is_deleted_on_a_cancel_inside_the_second_stage(self) -> None:
        class Cancel(BaseException):
            pass

        spec = FrameSpec(count=2, height=6, width=4)
        plan = VideoEnhancePlan(
            enable_super_resolution=True, enable_frame_interpolation=True
        )

        def cancelling_vfi(source, count, **_kwargs):
            del source, count
            raise Cancel()
            yield  # pragma: no cover - makes this a generator

        with mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.DlssStageStream",
            _fake_dlss_stage(self.events),
        ), mock.patch(
            "my_nodes.core.video_enhance.frame_pipeline.iter_interpolate_offline", cancelling_vfi
        ):
            with self.assertRaises(Cancel):
                self._run(
                    _stack(_frame(), _frame(value=0.5)),
                    spec,
                    plan,
                    dlss=_dlss_options(),
                    vfi=_vfi_options(),
                )
        self.assertEqual(os.listdir(self.directory), [])


class GimmStandIn:
    """GIMM-VFI stand-in that keeps the production iterator and its teardown.

    Only the external model plumbing is replaced: the checkpoint cache, the
    loader lookup and the model manager calls. Pair assembly, the frame-count
    contract and the teardown order stay production code.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.interpolations = 0
        # Called by the model manager while the stage runs, after the first frame.
        self.on_load: Callable[[], None] | None = None

    def install(self, stack: contextlib.ExitStack) -> None:
        import comfy.model_management as mm
        import torch

        class _Patcher:
            def __init__(self) -> None:
                self.model = object()

            def model_size(self) -> int:
                return 1

        owner = self

        class _Interpolator:
            def interpolate(
                self, module, images, ds_factor, factor, seed, output_flows=False
            ):
                del module, ds_factor, seed, output_flows
                if factor != 2:
                    raise AssertionError(f"unexpected interpolation factor {factor}")
                owner.interpolations += 1
                left, right = images[0], images[1]
                return (torch.stack((left, (left + right) / 2, right)), torch.zeros(1))

        def load_models_gpu(*_args, **_kwargs):
            owner.events.append("gimm-load-model")
            if owner.on_load is not None:
                owner.on_load()

        stack.enter_context(
            mock.patch.object(gimm_vfi, "cached_patcher", lambda *a, **k: _Patcher())
        )
        stack.enter_context(
            mock.patch.object(
                gimm_vfi, "resolve_gimm_nodes", lambda *a, **k: (object, _Interpolator)
            )
        )
        stack.enter_context(
            mock.patch.object(
                gimm_vfi,
                "_clear_gimm_backwarp_cache",
                lambda module: self.events.append("gimm-drop-backwarp-cache"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                gimm_vfi,
                "_clear_cublas_workspaces",
                lambda torch_module: self.events.append("gimm-clear-cublas"),
            )
        )
        stack.enter_context(
            mock.patch.object(mm, "get_torch_device", lambda: torch.device("cpu"))
        )
        stack.enter_context(
            mock.patch.object(
                mm, "free_memory", lambda *a, **k: self.events.append("gimm-free-memory")
            )
        )
        stack.enter_context(
            mock.patch.object(
                mm, "soft_empty_cache", lambda *a, **k: self.events.append("gimm-empty-cache")
            )
        )
        stack.enter_context(mock.patch.object(mm, "load_models_gpu", load_models_gpu))
        stack.enter_context(
            mock.patch.object(
                mm, "unload_model_and_clones", lambda patcher: self.events.append("gimm-unload")
            )
        )


class RealStageTestCase(unittest.TestCase):
    """Temporary disk space, worker tracking and the GIMM stand-in."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.directory = self.root / "frames"
        self.directory.mkdir()
        self._pids: list[int] = []
        self.addCleanup(self._assert_workers_gone)
        self.written: dict[int, np.ndarray] = {}
        self.gimm = GimmStandIn()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)

    def _assert_workers_gone(self) -> None:
        for pid in self._pids:
            assert_process_gone(pid)

    def _require_comfy(self) -> None:
        try:
            import comfy.model_management  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("the GIMM stage needs the ComfyUI interpreter")

    def _write(self, index: int, frame: np.ndarray) -> None:
        self.written[index] = np.array(frame)

    def _collect(self, source, spec, plan, **kwargs) -> object:
        self.written = {}
        return run_frame_pipeline(
            source, spec, plan, self._write, temp_directory=self.directory, **kwargs
        )

    def _record_worker(self, report: Path) -> int:
        wire = read_report(report)
        if "pid" not in wire:
            raise AssertionError(f"the fake worker never reported itself: {wire}")
        pid = int(wire["pid"])
        self._pids.append(pid)
        return pid

    def _dlss_options(self, features: int = FEATURE_SR, report: Path | None = None, on_start=None):
        report = report if report is not None else self.root / "report.json"
        runtime = create_runtime_dir(self.root / f"runtime-{features}", features)

        def factory(*, runtime_dir, features, wine_prefix):
            del runtime_dir, wine_prefix
            if on_start is not None:
                on_start()
            env = dict(os.environ)
            env["FAKE_DNR3_REPORT"] = str(report)
            return HostDriver.direct(
                fake_worker_command("ok"), runtime_dir=runtime, features=features, env=env
            )

        return _dlss_options(driver_factory=factory, memory_hooks=(lambda: None, lambda: None))


class IncrementalVfiTests(RealStageTestCase):
    """The incremental interpolation stage: order, count and teardown."""

    def setUp(self) -> None:
        super().setUp()
        self._require_comfy()
        self.gimm.install(self.stack)

    def _drain(self, frames: np.ndarray, count: int) -> list[np.ndarray]:
        stream = gimm_vfi.iter_interpolate_offline(
            frames,
            count,
            precision="fp32",
            ds_factor=1.0,
            models_dir="/models",
            node_mappings={},
        )
        try:
            return [np.array(frame) for frame in stream]
        finally:
            stream.close()

    def test_one_two_and_three_frames_keep_the_expected_order_and_count(self) -> None:
        for count in (1, 2, 3):
            with self.subTest(count=count):
                self.gimm.events.clear()
                self.gimm.interpolations = 0
                frames = _stack(*[_frame(value=0.1 * index) for index in range(count)])
                produced = self._drain(frames, count)
                self.assertEqual(len(produced), 2 * count - 1)
                self.assertEqual(produced[0].shape, frames.shape[1:])
                for index, frame in enumerate(frames):
                    np.testing.assert_array_equal(produced[index * 2], frame)
                for index in range(count - 1):
                    np.testing.assert_allclose(
                        produced[index * 2 + 1], _average(frames[index], frames[index + 1])
                    )
                if count == 1:
                    # One frame is passed through without touching the model.
                    self.assertEqual(self.gimm.interpolations, 0)
                    self.assertNotIn("gimm-load-model", self.gimm.events)
                else:
                    self.assertEqual(self.gimm.interpolations, count - 1)
                    self.assertIn("gimm-load-model", self.gimm.events)
                    self.assertIn("gimm-unload", self.gimm.events)

    def test_closing_the_stream_early_unloads_the_model(self) -> None:
        frames = _stack(_frame(value=0.1), _frame(value=0.4), _frame(value=0.8))
        stream = gimm_vfi.iter_interpolate_offline(
            frames, 3, precision="fp32", ds_factor=1.0, models_dir="/models", node_mappings={}
        )
        self.assertEqual(next(stream).shape, frames.shape[1:])
        self.assertNotIn("gimm-unload", self.gimm.events)
        stream.close()
        self.assertIn("gimm-drop-backwarp-cache", self.gimm.events)
        self.assertIn("gimm-unload", self.gimm.events)
        self.assertIn("gimm-clear-cublas", self.gimm.events)

    def test_a_short_or_long_source_is_rejected(self) -> None:
        for source, count, message in (
            (_stack(_frame()), 2, "ended early"),
            (_stack(_frame(), _frame(), _frame()), 2, "produced more"),
        ):
            with self.subTest(message=message):
                stream = gimm_vfi.iter_interpolate_offline(
                    source,
                    count,
                    precision="fp32",
                    ds_factor=1.0,
                    models_dir="/models",
                    node_mappings={},
                )
                with self.assertRaises(gimm_vfi.GimmVfiError) as raised:
                    list(stream)
                self.assertIn(message, str(raised.exception))

    def test_a_bad_frame_count_is_rejected_before_the_model(self) -> None:
        with self.assertRaises(gimm_vfi.GimmVfiError):
            next(
                gimm_vfi.iter_interpolate_offline(
                    _stack(_frame()),
                    0,
                    precision="fp32",
                    ds_factor=1.0,
                    models_dir="/models",
                    node_mappings={},
                )
            )
        self.assertNotIn("gimm-load-model", self.gimm.events)


class DlssStagePipelineTests(RealStageTestCase):
    """The incremental DLSS stage inside the pipeline, against the fake worker."""

    def test_single_stage_result_matches_the_compatibility_wrapper(self) -> None:
        frames = _stack(_frame(value=0.2), _frame(value=0.4))
        spec = FrameSpec(count=2, height=6, width=4)
        plan = VideoEnhancePlan(enable_super_resolution=True, sr_scale=2.0)

        report = self.root / "wrapper.json"
        wrapper = run_dlss_stage(
            plan,
            frames,
            runtime_dir="/runtime",
            wine_prefix="",
            channel_order="auto",
            motion_mode="none",
            scene_cut_threshold=0.2,
            driver_factory=self._dlss_options(report=report).driver_factory,
            memory_hooks=(lambda: None, lambda: None),
        )
        self._record_worker(report)
        self.assertTrue(read_report(report)["ended"])

        pipeline_report = self.root / "pipeline.json"
        result = self._collect(
            frames, spec, plan, dlss=self._dlss_options(report=pipeline_report)
        )
        self._record_worker(pipeline_report)

        self.assertEqual(result.frame_count, wrapper.frames.shape[0])
        self.assertEqual(result.output_height, wrapper.output_height)
        self.assertEqual(result.output_width, wrapper.output_width)
        self.assertEqual(result.channel_order, wrapper.channel_order)
        self.assertEqual(result.features, wrapper.features)
        np.testing.assert_array_equal(
            np.stack([self.written[index] for index in sorted(self.written)]), wrapper.frames
        )

    def test_a_short_source_fails_after_the_worker_started_and_still_reaps_it(self) -> None:
        frames = iter([_frame(value=0.2), _frame(value=0.4)])
        spec = FrameSpec(count=3, height=6, width=4)
        plan = VideoEnhancePlan(enable_super_resolution=True, sr_scale=2.0)
        report = self.root / "short.json"
        with self.assertRaises(FrameValidationError) as raised:
            self._collect(frames, spec, plan, dlss=self._dlss_options(report=report))
        self.assertIn("ended early", str(raised.exception))
        self._record_worker(report)
        self.assertEqual(sorted(self.written), [0, 1])


class TwoStagePipelineTests(RealStageTestCase):
    """Both stage orders with the real stages and the disk-backed intermediate."""

    def setUp(self) -> None:
        super().setUp()
        self._require_comfy()
        self.gimm.install(self.stack)

    def _expected_vfi_frames(self, frames: np.ndarray) -> np.ndarray:
        return _interpolate_frames(frames)

    def test_vfi_then_dlss_unloads_gimm_before_the_worker_starts(self) -> None:
        for count in (1, 2, 3):
            with self.subTest(count=count):
                self.gimm.events.clear()
                frames = _stack(*[_frame(value=0.1 * index) for index in range(count)])
                spec = FrameSpec(count=count, height=6, width=4)
                plan = VideoEnhancePlan(
                    enable_super_resolution=True,
                    enable_frame_interpolation=True,
                    sr_scale=2.0,
                    stage_order="vfi_then_dlss",
                )
                specs = pipeline_specs(spec, plan)
                assert specs.intermediate is not None
                observed: dict[str, object] = {}
                report = self.root / f"vfi-first-{count}.json"

                def on_start() -> None:
                    # The worker is about to start: stage 1 must have ended and its
                    # frames must be the ones sitting in the disk store.
                    self.gimm.events.append("dlss-worker-start")
                    files = list(self.directory.iterdir())
                    observed["files"] = len(files)
                    observed["store"] = np.fromfile(files[0], dtype=np.float32).reshape(
                        specs.intermediate.shape
                    )

                result = self._collect(
                    frames,
                    spec,
                    plan,
                    vfi=_vfi_options(),
                    dlss=self._dlss_options(report=report, on_start=on_start),
                )
                self._record_worker(report)

                self.assertEqual(result.stages, (STAGE_VFI, STAGE_DLSS))
                self.assertEqual(result.frame_count, 2 * count - 1)
                self.assertEqual(
                    (result.output_height, result.output_width),
                    (specs.final.height, specs.final.width),
                )
                if count == 1:
                    # A single frame loads no GIMM model at all, so there is no
                    # unload to order against the worker; nothing stays resident.
                    self.assertNotIn("gimm-load-model", self.gimm.events)
                else:
                    self.assertLess(
                        self.gimm.events.index("gimm-unload"),
                        self.gimm.events.index("dlss-worker-start"),
                    )
                self.assertEqual(observed["files"], 1)
                expected_store = self._expected_vfi_frames(frames)
                np.testing.assert_allclose(observed["store"], expected_store)
                # Stage 2 upscales every stored frame; the fake worker answers with
                # a nearest-neighbour enlargement of its input.
                for index, frame in enumerate(expected_store):
                    np.testing.assert_allclose(
                        self.written[index],
                        _nearest(frame, specs.final.height, specs.final.width),
                    )
                self.assertTrue(read_report(report)["ended"])
                self.assertEqual(os.listdir(self.directory), [])

    def test_dlss_then_vfi_reaps_the_worker_before_gimm_loads(self) -> None:
        frames = _stack(_frame(value=0.2), _frame(value=0.6), _frame(value=0.9))
        spec = FrameSpec(count=3, height=6, width=4)
        plan = VideoEnhancePlan(
            enable_super_resolution=True, enable_frame_interpolation=True, sr_scale=2.0
        )
        specs = pipeline_specs(spec, plan)
        assert specs.intermediate is not None
        report = self.root / "dlss-first.json"
        observed: dict[str, object] = {}

        def on_gimm_load() -> None:
            # The interpolation stage reached its model load: the worker must be
            # reaped already and the store must hold every enhanced frame.
            pid = self._record_worker(report)
            assert_process_gone(pid)
            files = list(self.directory.iterdir())
            observed["files"] = len(files)
            observed["store"] = np.fromfile(files[0], dtype=np.float32).reshape(
                specs.intermediate.shape
            )

        self.gimm.on_load = on_gimm_load
        result = self._collect(
            frames,
            spec,
            plan,
            vfi=_vfi_options(),
            dlss=self._dlss_options(report=report),
        )

        self.assertEqual(result.stages, (STAGE_DLSS, STAGE_VFI))
        self.assertEqual(result.frame_count, 2 * 3 - 1)
        self.assertEqual(
            (result.output_height, result.output_width),
            (specs.final.height, specs.final.width),
        )
        self.assertEqual(observed["files"], 1)
        expected_store = np.stack(
            [_nearest(frame, specs.intermediate.height, specs.intermediate.width) for frame in frames]
        )
        np.testing.assert_allclose(observed["store"], expected_store)
        np.testing.assert_allclose(self.written[0], expected_store[0])
        np.testing.assert_allclose(self.written[1], _average(expected_store[0], expected_store[1]))
        self.assertEqual(self.gimm.interpolations, 2)
        self.assertTrue(read_report(report)["ended"])
        self.assertEqual(os.listdir(self.directory), [])

    def test_both_orders_hold_exact_frames_across_tiny_store_chunks(self) -> None:
        """The real stages, with the store forced to cross several chunk boundaries.

        GIMM keeps the previous frame while it pulls the next one from the store,
        so a frame retained across a chunk boundary is the case that would break
        first if a read frame were a view into an unmapped chunk.
        """
        frames = _stack(_frame(value=0.2), _frame(value=0.5), _frame(value=0.7))
        for stage_order in ("dlss_then_vfi", "vfi_then_dlss"):
            with self.subTest(stage_order=stage_order):
                self.gimm.events.clear()
                self.gimm.on_load = None
                self.written.clear()
                spec = FrameSpec(count=3, height=6, width=4)
                plan = VideoEnhancePlan(
                    enable_super_resolution=True,
                    enable_frame_interpolation=True,
                    sr_scale=2.0,
                    stage_order=stage_order,
                )
                specs = pipeline_specs(spec, plan)
                assert specs.intermediate is not None
                report = self.root / f"tiny-chunks-{stage_order}.json"
                cap = 2 * _frame_bytes(specs.intermediate)
                with mock.patch.object(
                    frame_pipeline_module, "FRAME_STORE_CHUNK_BYTES", cap
                ):
                    result = self._collect(
                        frames,
                        spec,
                        plan,
                        vfi=_vfi_options(),
                        dlss=self._dlss_options(report=report),
                    )
                self._record_worker(report)

                if stage_order == "dlss_then_vfi":
                    intermediate = np.stack(
                        [
                            _nearest(frame, specs.intermediate.height, specs.intermediate.width)
                            for frame in frames
                        ]
                    )
                    expected = _interpolate_frames(intermediate)
                else:
                    intermediate = _interpolate_frames(frames)
                    expected = np.stack(
                        [
                            _nearest(frame, specs.final.height, specs.final.width)
                            for frame in intermediate
                        ]
                    )
                # Several chunks were crossed: two frames per chunk.
                self.assertGreater(-(-specs.intermediate.count // 2), 1)
                self.assertEqual(result.frame_count, len(expected))
                for index, frame in enumerate(expected):
                    np.testing.assert_allclose(self.written[index], frame, rtol=0, atol=0)
                self.assertEqual(sorted(self.written), list(range(len(expected))))
                self.assertTrue(read_report(report)["ended"])
                self.assertEqual(os.listdir(self.directory), [])
