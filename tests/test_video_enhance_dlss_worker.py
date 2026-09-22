from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from my_nodes.core.video_enhance import (
    FEATURE_NR,
    FEATURE_SR,
    NATIVE_PERF_QUALITY,
    Dnr3Error,
    Dnr3ProtocolError,
    Dnr3RemoteError,
    Dnr3ValidationError,
    Dnr3Worker,
    HostDriver,
    ProcessIOError,
    ProcessTimeout,
    dnr3,
    perf_quality_for_scale,
    runtime,
)

from .video_enhance_fake_dnr3_worker import REPORT_ENV
from .video_enhance_fixtures import (
    assert_process_gone,
    create_runtime_dir,
    fake_worker_command,
    read_report,
    wait_for_report,
)

SR_ONLY = FEATURE_SR
NR_ONLY = FEATURE_NR
SR_NR = FEATURE_SR | FEATURE_NR


def make_header(
    features: int,
    *,
    width: int = 4,
    height: int = 6,
    frame_count: int = 1,
    scale: float | None = None,
    **overrides,
) -> dnr3.Header:
    """A valid header for `features` (2x for SR, native size for NR-only)."""
    if features & FEATURE_SR:
        ratio = 2.0 if scale is None else scale
        values = {
            "output_width": round(width * ratio),
            "output_height": round(height * ratio),
            "perf_quality": perf_quality_for_scale(ratio),
        }
    else:
        values = {
            "output_width": width,
            "output_height": height,
            "perf_quality": NATIVE_PERF_QUALITY,
        }
    values.update(
        input_width=width,
        input_height=height,
        frame_count=frame_count,
        features=features,
    )
    for name, default in (
        ("warmup_frames", 0),
        ("preset", 0),
        ("style", 0),
        ("automask", True),
        ("ui_correction", False),
        ("intensity", 1.0),
        ("tone", 1.0),
        ("structure", 1.5),
        ("skin", -1.0),
        ("global_tone", -1.0),
    ):
        values.setdefault(name, default)
    values.update(overrides)
    return dnr3.Header(**values)


def frame(width: int = 4, height: int = 6, offset: float = 0.0) -> np.ndarray:
    """A small frame with a unique, easily verified content pattern."""
    rows = np.linspace(0.0, 0.4, height, dtype=np.float32)
    cols = np.linspace(0.0, 0.2, width, dtype=np.float32)
    plane = np.add.outer(rows, cols) + np.float32(offset)
    return np.stack([plane, plane / 2, 1.0 - plane], axis=-1).astype(np.float32)


def nearest(rgb: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Independent reference implementation of the fake worker's transform."""
    in_h, in_w = rgb.shape[:2]
    out = np.empty((out_height, out_width, 3), dtype=np.float32)
    for y in range(out_height):
        for x in range(out_width):
            out[y, x] = rgb[y * in_h // out_height, x * in_w // out_width]
    return out


class Dnr3WorkerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self._reports: list[Path] = []
        self.addCleanup(self._assert_workers_gone)

    def _assert_workers_gone(self) -> None:
        for path in self._reports:
            report = read_report(path)
            pid = report.get("pid")
            if isinstance(pid, int):
                assert_process_gone(pid)
            grandchild = report.get("grandchild")
            if isinstance(grandchild, int):
                assert_process_gone(grandchild)

    def report_path(self, name: str) -> Path:
        path = self.root / f"report-{name}.json"
        self._reports.append(path)
        return path

    def driver(
        self,
        features: int,
        mode: str = "ok",
        *,
        nr_name: str | None = None,
        report: Path | None = None,
    ) -> HostDriver:
        directory = create_runtime_dir(self.root / f"runtime-{features}-{mode}", features, nr_name=nr_name)
        env = dict(os.environ)
        if report is not None:
            env[REPORT_ENV] = str(report)
        return HostDriver.direct(
            fake_worker_command(mode), runtime_dir=directory, features=features, env=env
        )

    def worker(
        self,
        features: int,
        *,
        mode: str = "ok",
        header: dnr3.Header | None = None,
        nr_name: str | None = None,
        **kwargs,
    ) -> tuple[Dnr3Worker, Path]:
        report = self.report_path(f"{features}-{mode}-{len(self._reports)}")
        driver = self.driver(features, mode, nr_name=nr_name, report=report)
        worker = Dnr3Worker(header or make_header(features), driver=driver, **kwargs)
        return worker, report


class FeatureCombinationTests(Dnr3WorkerTestCase):
    def test_super_resolution_only_enlarges_and_needs_no_nr_runtime(self) -> None:
        worker, report = self.worker(SR_ONLY, timeout=10.0)
        self.assertIsNone(worker.files.nr)  # no nvngx_dlssnr*.dll was needed
        source = frame()
        with worker:
            output = worker.enhance(source, reset=True)
        self.assertEqual(output.shape, (12, 8, 3))
        np.testing.assert_allclose(output, nearest(source, 12, 8))
        wire = wait_for_report(report)
        self.assertEqual(wire["header"]["features"], FEATURE_SR)
        self.assertEqual(wire["header"]["perf_quality"], 0)
        self.assertTrue(wire["ended"])

    def test_neural_rendering_only_stays_at_native_resolution(self) -> None:
        worker, report = self.worker(NR_ONLY, timeout=10.0)
        self.assertIsNone(worker.files.sr)  # no nvngx_dlss.dll was needed
        self.assertEqual(worker.files.nr_name, runtime.NR_DLL)
        source = frame()
        with worker:
            output = worker.enhance(source, reset=True)
        self.assertEqual(output.shape, source.shape)
        np.testing.assert_allclose(output, source)
        wire = wait_for_report(report)
        self.assertEqual(wire["header"]["features"], FEATURE_NR)
        self.assertEqual(wire["header"]["perf_quality"], NATIVE_PERF_QUALITY)

    def test_both_features_use_the_carrier_ratio(self) -> None:
        worker, report = self.worker(SR_NR, header=make_header(SR_NR, scale=1.5), timeout=10.0)
        source = frame()
        with worker:
            output = worker.enhance(source, reset=True)
        self.assertEqual(output.shape, (9, 6, 3))
        np.testing.assert_allclose(output, nearest(source, 9, 6))
        wire = wait_for_report(report)
        self.assertEqual(wire["header"]["features"], SR_NR)
        self.assertEqual(wire["header"]["perf_quality"], 2)

    def test_rtx30_neural_runtime_wins_when_present(self) -> None:
        directory = self.root / "runtime-rtx30"
        create_runtime_dir(directory, NR_ONLY, nr_name=runtime.NR_DLL_RTX30)
        (directory / runtime.NR_DLL).write_bytes(b"universal build")
        files = runtime.resolve_runtime_files(directory, NR_ONLY)
        self.assertEqual(files.nr_name, runtime.NR_DLL_RTX30)
        env = runtime.build_environment(files=files, base_env={"DISPLAY": ":0"})
        self.assertEqual(env["DLSS5NR_SNR_FILENAME"], runtime.NR_DLL_RTX30)

    def test_driver_and_header_features_must_match(self) -> None:
        with self.assertRaises(Dnr3ValidationError) as raised:
            self.worker(NR_ONLY, header=make_header(SR_ONLY))
        self.assertIn("features", str(raised.exception))


class ExchangeTests(Dnr3WorkerTestCase):
    def test_multi_frame_session_keeps_order_and_reset_flags(self) -> None:
        worker, report = self.worker(SR_ONLY, header=make_header(SR_ONLY, frame_count=3), timeout=10.0)
        sources = [frame(offset=index / 10) for index in range(3)]
        with worker:
            outputs = [
                worker.enhance(source, reset=(index == 0))
                for index, source in enumerate(sources)
            ]
        for index, (source, output) in enumerate(zip(sources, outputs)):
            with self.subTest(frame=index):
                np.testing.assert_allclose(output, nearest(source, 12, 8))
        wire = wait_for_report(report)["frames"]
        self.assertEqual([entry["index"] for entry in wire], [0, 1, 2])
        self.assertEqual([entry["reset"] for entry in wire], [True, False, False])
        self.assertEqual(worker.frame_index, 3)

    def test_every_header_field_reaches_the_worker(self) -> None:
        header = make_header(
            SR_NR,
            frame_count=2,
            warmup_frames=1,
            preset=3,
            style=2,
            automask=False,
            ui_correction=True,
            intensity=0.5,
            tone=1.25,
            structure=1.75,
            skin=0.25,
            global_tone=0.75,
        )
        worker, report = self.worker(SR_NR, header=header, timeout=10.0)
        with worker:
            worker.enhance(frame(), reset=True)
            worker.enhance(frame(), reset=False)
        wire = wait_for_report(report)["header"]
        for field in ("input_width", "input_height", "output_width", "output_height",
                      "warmup_frames", "frame_count", "perf_quality", "features", "preset",
                      "style", "automask", "ui_correction", "intensity", "tone", "structure",
                      "skin", "global_tone"):
            with self.subTest(field=field):
                self.assertEqual(wire[field], getattr(header, field))

    def test_motion_vectors_are_transported_and_default_to_zero(self) -> None:
        worker, report = self.worker(NR_ONLY, header=make_header(NR_ONLY, frame_count=2), timeout=10.0)
        motion = np.zeros((6, 4, 2), dtype=np.float16)
        motion[1, 2, :] = np.float16(0.5)
        with worker:
            worker.enhance(frame(), motion, reset=True)
            worker.enhance(frame(), None, reset=False)
        frames = wait_for_report(report)["frames"]
        self.assertEqual(frames[0]["motion_nonzero"], 2)
        self.assertEqual(frames[1]["motion_nonzero"], 0)
        self.assertEqual(frames[0]["motion_bytes"], 6 * 4 * 2 * 2)

    def test_frame_budget_is_enforced(self) -> None:
        worker, _ = self.worker(NR_ONLY, header=make_header(NR_ONLY, frame_count=1), timeout=10.0)
        with worker:
            worker.enhance(frame(), reset=True)
            with self.assertRaises(Dnr3Error) as raised:
                worker.enhance(frame(), reset=False)
        self.assertIn("frame_count 1 is exhausted", str(raised.exception))
        # Nothing beyond the declared frame count reached the worker.
        self.assertEqual(len(wait_for_report(self._reports[0])["frames"]), 1)

    def test_enhance_after_finish_is_rejected(self) -> None:
        worker, _ = self.worker(NR_ONLY, header=make_header(NR_ONLY, frame_count=1), timeout=10.0)
        with worker:
            worker.enhance(frame(), reset=True)
            worker.finish()
            self.assertEqual(worker.frame_index, 1)
            with self.assertRaises(Dnr3Error) as raised:
                worker.enhance(frame())
        self.assertIn("no more frames", str(raised.exception))

    def test_finish_reports_a_missing_frame_count(self) -> None:
        worker, report = self.worker(SR_ONLY, header=make_header(SR_ONLY, frame_count=2), timeout=10.0)
        with self.assertRaises(Dnr3ProtocolError) as raised:
            with worker:
                worker.enhance(frame(), reset=True)
        self.assertIn("2 frames but 1 were sent", str(raised.exception))

    def test_input_validation_happens_before_anything_is_sent(self) -> None:
        worker, report = self.worker(SR_ONLY, timeout=10.0)
        with self.assertRaises(Dnr3ValidationError):
            with worker:
                wait_for_report(report)  # the worker is up and has our header
                worker.enhance(np.zeros((12, 8, 3), dtype=np.float32))
        wire = read_report(report)
        self.assertEqual(wire["frames"], [])
        self.assertNotIn("ended", wire)


class WorkerFailureTests(Dnr3WorkerTestCase):
    def test_remote_error_carries_the_worker_message_and_log(self) -> None:
        worker, _ = self.worker(NR_ONLY, mode="error", timeout=10.0)
        with self.assertRaises(Dnr3RemoteError) as raised:
            with worker:
                worker.enhance(frame(), reset=True)
        message = str(raised.exception)
        self.assertIn("DLSS worker failed on frame 0", message)
        self.assertIn("neural-rendering runtime unavailable", message)

    def test_wrong_reply_magic_is_a_protocol_error(self) -> None:
        worker, _ = self.worker(NR_ONLY, mode="bad-magic", timeout=10.0)
        with self.assertRaises(Dnr3ProtocolError) as raised:
            with worker:
                worker.enhance(frame(), reset=True)
        self.assertIn("reply magic", str(raised.exception))

    def test_wrong_frame_index_is_a_protocol_error(self) -> None:
        worker, _ = self.worker(NR_ONLY, mode="bad-index", timeout=10.0)
        with self.assertRaises(Dnr3ProtocolError) as raised:
            with worker:
                worker.enhance(frame(), reset=True)
        self.assertIn("reply is for frame 1 but frame 0", str(raised.exception))

    def test_wrong_float_count_is_a_protocol_error(self) -> None:
        worker, _ = self.worker(NR_ONLY, mode="bad-count", timeout=10.0)
        with self.assertRaises(Dnr3ProtocolError) as raised:
            with worker:
                worker.enhance(frame(), reset=True)
        self.assertIn("floats", str(raised.exception))

    def test_truncated_reply_is_an_io_error(self) -> None:
        worker, _ = self.worker(NR_ONLY, mode="short-reply", timeout=10.0)
        with self.assertRaises(ProcessIOError) as raised:
            with worker:
                worker.enhance(frame(), reset=True)
        self.assertIn("reply header", str(raised.exception))

    def test_child_crash_is_reported_with_its_stderr(self) -> None:
        worker, _ = self.worker(NR_ONLY, mode="crash", timeout=10.0)
        with self.assertRaises(ProcessIOError) as raised:
            with worker:
                worker.enhance(frame(), reset=True)
        message = str(raised.exception)
        self.assertIn("exit code 3", message)
        self.assertIn("fake DNR3 worker aborted", message)

    def test_missing_end_marker_fails_finish(self) -> None:
        worker, _ = self.worker(NR_ONLY, mode="no-end", timeout=10.0)
        with self.assertRaises(ProcessIOError) as raised:
            with worker:
                worker.enhance(frame(), reset=True)
        self.assertIn("END1 marker", str(raised.exception))

    def test_non_zero_exit_after_end_marker_is_reported(self) -> None:
        worker, _ = self.worker(NR_ONLY, mode="exit-after-end", timeout=10.0)
        with self.assertRaises(Dnr3ProtocolError) as raised:
            with worker:
                worker.enhance(frame(), reset=True)
        self.assertIn("code 3", str(raised.exception))

    def test_timeout_leaves_no_worker_behind(self) -> None:
        worker, report = self.worker(NR_ONLY, mode="hang", timeout=0.3, shutdown_grace=0.3)
        with self.assertRaises(ProcessTimeout) as raised:
            with worker:
                # Start the timeout only after Python startup and header parsing;
                # this test measures a hung reply, not interpreter launch speed.
                wait_for_report(report)
                worker.enhance(frame(), reset=True)
        self.assertIn("did not answer within 0.300s", str(raised.exception))
        pid = wait_for_report(report)["pid"]
        self.assertIsInstance(pid, int)
        assert_process_gone(pid)

    def test_interrupt_stops_a_hanging_worker_and_cleans_up(self) -> None:
        class ComfyInterrupt(BaseException):
            """Stands in for a ComfyUI interrupt raised by the callback."""

        def interrupt() -> bool:
            # Cancel only once the worker is really up, so the cleanup below is
            # what the assertion is about.
            if "pid" not in read_report(report_holder["path"]):
                return False
            raise ComfyInterrupt("user cancelled")

        report_holder: dict[str, Path] = {}
        worker, report = self.worker(
            NR_ONLY, mode="stubborn", timeout=10.0, interrupt=interrupt, shutdown_grace=0.3
        )
        report_holder["path"] = report
        with self.assertRaises(ComfyInterrupt):
            with worker:
                worker.enhance(frame(), reset=True)
        pid = wait_for_report(report)["pid"]
        assert_process_gone(pid)


class WorkerRuntimeRequirementTests(unittest.TestCase):
    def test_each_feature_only_requires_its_own_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sr_dir = create_runtime_dir(root / "sr", SR_ONLY)
            nr_dir = create_runtime_dir(root / "nr", NR_ONLY)
            self.assertFalse((sr_dir / runtime.NR_DLL).exists())
            self.assertFalse((nr_dir / runtime.SR_DLL).exists())
            driver = HostDriver.direct(fake_worker_command("ok"), runtime_dir=sr_dir, features=SR_ONLY)
            self.assertIsNone(driver.files.nr)
            driver = HostDriver.direct(fake_worker_command("ok"), runtime_dir=nr_dir, features=NR_ONLY)
            self.assertIsNone(driver.files.sr)
            self.assertEqual(driver.files.nr_name, runtime.NR_DLL)
            # Both together need both files.
            both = create_runtime_dir(root / "both", SR_NR)
            files = runtime.resolve_runtime_files(both, SR_NR)
            self.assertIsNotNone(files.sr)
            self.assertIsNotNone(files.nr)
