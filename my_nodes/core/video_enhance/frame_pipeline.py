"""Sequential frame pipeline shared by the IMAGE node and the streaming VIDEO node.

One execution walks the stages of its plan in the plan's fixed order, one frame
at a time, and hands every final frame to the caller's writer: the pipeline
itself never assembles the output batch. When both stages are active the first
one is drained completely into a float32 `np.memmap` store, closed and torn down
before the second one starts, so the GIMM-VFI module and the Wine/DLSS worker are
never GPU-resident at the same time and the full-precision intermediate never
lives in RAM.

The store maps at most `FRAME_STORE_CHUNK_BYTES` of the file at a time, one
chunk per write and one chunk per read, and unmaps each chunk before mapping the
next. Mapping the whole file instead would be correct but not bounded: Linux
counts every touched page of a mapping in the process RSS, so RAM would grow
with the clip length even though the pages are file-backed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from my_nodes.core.video_enhance.dlss_stage import DlssStageStream, output_dimensions
from my_nodes.core.video_enhance.dlss_worker import DEFAULT_TIMEOUT_SECONDS
from my_nodes.core.video_enhance.gimm_vfi import iter_interpolate_offline
from my_nodes.core.video_enhance.plan import STAGE_DLSS, STAGE_VFI, VideoEnhancePlan
from my_nodes.core.video_enhance.runtime import HostDriver

FRAME_DTYPE = np.float32
_ITEMSIZE = np.dtype(FRAME_DTYPE).itemsize

FRAME_STORE_CHUNK_BYTES = 256 * 1024 * 1024
"""Most bytes one frame-store mapping may cover; exported so tests can force boundaries."""


class FramePipelineError(RuntimeError):
    """The pipeline cannot honour its frame contract."""


class FrameStoreError(FramePipelineError):
    """The disk-backed intermediate frame store cannot be created or deleted."""


@dataclass(frozen=True)
class FrameSpec:
    """Immutable description of a frame sequence: count, height and width."""

    count: int
    height: int
    width: int

    def __post_init__(self) -> None:
        for name in ("count", "height", "width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of the equivalent float32 IMAGE batch `[N,H,W,3]`."""
        return (self.count, self.height, self.width, 3)

    @property
    def nbytes(self) -> int:
        """Exact float32 size of that batch."""
        return self.count * self.height * self.width * 3 * _ITEMSIZE


@dataclass(frozen=True)
class PipelineSpecs:
    """Pure plan of one run: the stage order and the specs it passes along.

    `intermediate` describes the disk-backed store between two active stages and
    stays None otherwise.
    """

    stages: tuple[str, ...]
    final: FrameSpec
    intermediate: FrameSpec | None = None

    @property
    def staged(self) -> bool:
        """True when the intermediate frames go to disk instead of RAM."""
        return self.intermediate is not None


def dlss_frame_spec(spec: FrameSpec, plan: VideoEnhancePlan) -> FrameSpec:
    """Spec of one DLSS stage: same frame count, the size follows `sr_scale`."""
    if not plan.enable_super_resolution:
        return spec
    width, height = output_dimensions(spec.width, spec.height, plan.sr_scale)
    return FrameSpec(count=spec.count, height=height, width=width)


def vfi_frame_spec(spec: FrameSpec, plan: VideoEnhancePlan) -> FrameSpec:
    """Spec of one VFI stage: the interpolated frames share both boundaries."""
    factor = plan.interpolation_factor
    return FrameSpec(
        count=factor * spec.count - (factor - 1), height=spec.height, width=spec.width
    )


def stage_frame_spec(spec: FrameSpec, plan: VideoEnhancePlan, stage: str) -> FrameSpec:
    """Spec after one named stage, without running it."""
    if stage == STAGE_DLSS:
        return dlss_frame_spec(spec, plan)
    if stage == STAGE_VFI:
        return vfi_frame_spec(spec, plan)
    raise FramePipelineError(f"unknown stage {stage!r}")


def stage_steps(spec: FrameSpec, stage: str) -> int:
    """Progress steps one stage reports: DLSS counts frames, VFI counts pairs."""
    if stage == STAGE_DLSS:
        return spec.count
    if stage == STAGE_VFI:
        return max(0, spec.count - 1)
    raise FramePipelineError(f"unknown stage {stage!r}")


def stage_step_counts(
    source_spec: FrameSpec, plan: VideoEnhancePlan
) -> tuple[tuple[str, int], ...]:
    """Per-stage progress steps in execution order."""
    spec = source_spec
    counts: list[tuple[str, int]] = []
    for stage in plan.stages:
        counts.append((stage, stage_steps(spec, stage)))
        spec = stage_frame_spec(spec, plan, stage)
    return tuple(counts)


def pipeline_step_total(source_spec: FrameSpec, plan: VideoEnhancePlan) -> int:
    """Total progress steps of one run, so a caller can size its progress bar."""
    return sum(count for _stage, count in stage_step_counts(source_spec, plan))


def pipeline_specs(source_spec: FrameSpec, plan: VideoEnhancePlan) -> PipelineSpecs:
    """Plan one run: the stage order, the final spec and the optional store spec."""
    stages = plan.stages
    spec = source_spec
    intermediate = stage_frame_spec(spec, plan, stages[0]) if len(stages) > 1 else None
    for stage in stages:
        spec = stage_frame_spec(spec, plan, stage)
    return PipelineSpecs(stages=stages, final=spec, intermediate=intermediate)


class FrameStore:
    """Sequential float32 frames in one preallocated file, mapped in bounded chunks.

    The clip never exists as a single mapping: `mapped_bytes` covers the live
    chunk only and is at most `FRAME_STORE_CHUNK_BYTES` (at least one whole
    frame), whatever the clip length. Every chunk is unmapped before the next one
    is mapped, in both directions.

    Writing is strictly sequential: `write(index, frame)` for index 0, 1, 2 ...
    `finish()` seals the store and releases the write mapping, `iter_frames()`
    then maps one chunk at a time and yields standalone float32 copies, so a
    frame a reader keeps - GIMM keeps its previous endpoint - stays valid after
    the chunk it came from is unmapped. `close()` releases the live mapping on
    any exit path and is idempotent.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        spec: FrameSpec,
        *,
        chunk_bytes: int | None = None,
    ) -> None:
        self._path = os.fspath(path)
        self._spec = spec
        self._frame_bytes = spec.height * spec.width * 3 * _ITEMSIZE
        cap = FRAME_STORE_CHUNK_BYTES if chunk_bytes is None else chunk_bytes
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise FrameStoreError(f"the frame-store chunk cap must be a positive int, got {cap!r}")
        # Whole frames only, at least one, so a tiny cap still makes progress.
        self._frames_per_chunk = max(1, min(spec.count, cap // self._frame_bytes))
        self._chunk_bytes = self._frames_per_chunk * self._frame_bytes
        self._written = 0
        self._mapping: np.memmap | None = None
        self._mapping_start = 0
        self._mapping_frames = 0
        self._mapping_writable = False
        self._sealed = False
        self._closed = False

    @property
    def spec(self) -> FrameSpec:
        return self._spec

    @property
    def path(self) -> str:
        """The raw file behind the store; it exists until the context exits."""
        return self._path

    @property
    def nbytes(self) -> int:
        """Size of the whole preallocated file."""
        return self._spec.nbytes

    @property
    def chunk_bytes(self) -> int:
        """Bytes one mapping of this store covers at most."""
        return self._chunk_bytes

    @property
    def frames_per_chunk(self) -> int:
        return self._frames_per_chunk

    @property
    def frames_written(self) -> int:
        return self._written

    @property
    def mapped_bytes(self) -> int:
        """Bytes of the one live mapping, or 0 while the file is not mapped."""
        return 0 if self._mapping is None else self._mapping_frames * self._frame_bytes

    @property
    def open_mappings(self) -> int:
        """Number of live mappings; never more than one, 0 when finished."""
        return 0 if self._mapping is None else 1

    @property
    def closed(self) -> bool:
        return self._closed

    def write(self, index: int, frame: np.ndarray) -> None:
        """Store one frame, in order; `frame` is validated before it reaches disk."""
        if self._closed:
            raise FrameStoreError("the frame store is closed")
        if self._sealed:
            raise FrameStoreError("the frame store was finished: it no longer accepts frames")
        if index != self._written:
            raise FrameStoreError(
                f"the frame store is sequential: expected frame {self._written}, got {index}"
            )
        if self._written >= self._spec.count:
            raise FrameStoreError(
                f"the frame store holds {self._spec.count} frames, so frame {index} is one too many"
            )
        array = self._checked_frame(index, frame)
        mapping = self._write_mapping(index)
        mapping[index - self._mapping_start] = array
        self._written += 1
        if self._written - self._mapping_start >= self._mapping_frames:
            # This chunk is complete: make it durable and release it before the
            # next one is mapped. A failing flush is the caller's to see.
            self._release_mapping(strict=True, flush=True)

    def finish(self) -> None:
        """Seal the store and release the write mapping; idempotent.

        A failing flush here is reported: stage 2 is about to read exactly these
        bytes, so an unreported writeback failure would feed it corrupt frames.
        """
        self._sealed = True
        self._release_mapping(strict=True, flush=True)

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield one standalone float32 copy per frame, one mapped chunk at a time."""
        if self._closed:
            raise FrameStoreError("the frame store is closed")
        if not self._sealed:
            raise FrameStoreError("the frame store must be finished before it is read")
        if self._written != self._spec.count:
            raise FrameStoreError(
                f"the frame store holds {self._written} of {self._spec.count} frames"
            )
        for start in range(0, self._spec.count, self._frames_per_chunk):
            frames = min(self._frames_per_chunk, self._spec.count - start)
            mapping = self._map(start, frames, writable=False)
            self._mapping, self._mapping_start, self._mapping_frames = mapping, start, frames
            self._mapping_writable = False
            try:
                for offset in range(frames):
                    if self._closed:
                        raise FrameStoreError("the frame store was closed while it was read")
                    # A copy, never a view: the reader may keep this frame after
                    # the chunk behind it is unmapped.
                    yield np.array(mapping[offset], dtype=FRAME_DTYPE, copy=True)
            except BaseException:
                # The consumer went away mid-chunk (a cancel closes the iterator
                # with GeneratorExit), or our own check above raised: release
                # quietly so it can never mask that.
                self._release_mapping(mapping, strict=False, flush=False)
                raise
            else:
                # A chunk that was read to its end releases strictly: a failing
                # close would otherwise be invisible.
                self._release_mapping(mapping, strict=True, flush=False)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_frames()

    def close(self, *, strict: bool = False) -> None:
        """Release the live mapping; idempotent, quiet unless `strict`.

        The quiet default is the cleanup form: on a failing run it must not mask
        the exception that is already propagating.
        """
        self._closed = True
        self._release_mapping(strict=strict, flush=self._mapping_writable)

    def _checked_frame(self, index: int, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        expected = self._spec.shape[1:]
        if array.dtype != FRAME_DTYPE or array.shape != expected:
            raise FrameStoreError(
                f"frame {index} must be float32 {expected}, got {array.dtype} {array.shape}"
            )
        if not np.isfinite(array).all():
            raise FrameStoreError(f"frame {index} contains non-finite values")
        return array

    def _write_mapping(self, index: int) -> np.memmap:
        start = (index // self._frames_per_chunk) * self._frames_per_chunk
        if self._mapping is not None and self._mapping_start == start:
            return self._mapping
        # Only reachable as a safety net: a chunk is released as soon as it is
        # full, so the next write always starts a new chunk.
        self._release_mapping(strict=True, flush=True)
        frames = min(self._frames_per_chunk, self._spec.count - start)
        self._mapping = self._map(start, frames, writable=True)
        self._mapping_start = start
        self._mapping_frames = frames
        self._mapping_writable = True
        return self._mapping

    def _map(self, start: int, frames: int, *, writable: bool) -> np.memmap:
        """Map `frames` frames starting at `start` inside the preallocated file."""
        return np.memmap(
            self._path,
            dtype=FRAME_DTYPE,
            # A read chunk is mapped read-only: it needs no data flush at all.
            mode="r+" if writable else "r",
            offset=start * self._frame_bytes,
            shape=(frames, self._spec.height, self._spec.width, 3),
        )

    def _release_mapping(
        self, mapping: np.memmap | None = None, *, strict: bool, flush: bool
    ) -> None:
        """Release the live chunk, or `mapping` when it is still the live one.

        `strict` reports a failing flush or close as a `FrameStoreError`; the
        quiet form is for cleanup while another exception is on its way out.
        """
        current = self._mapping
        if current is None or (mapping is not None and current is not mapping):
            return
        self._mapping = None
        self._mapping_start = 0
        self._mapping_frames = 0
        self._mapping_writable = False
        _close_mapping(current, strict=strict, flush=flush)


def _close_mapping(mapping: np.memmap, *, strict: bool, flush: bool) -> None:
    """Flush (when writable) and close one chunk mapping so its pages leave RSS.

    `np.memmap` has no public close, so this helper is the single owner of that
    private attribute: nothing else in this module touches `_mmap`. Both steps are
    attempted even when the first fails, so the mapping never stays open.

    `strict` marks a functional transition - a completed write chunk, `finish()`
    or a fully read chunk - where a failing flush or close means the intermediate
    may be incomplete or corrupt and must be reported. Cleanup while another
    exception is propagating stays quiet instead: masking the run's own failure
    would hide the reason the run failed.
    """
    failure: Exception | None = None
    if flush:
        try:
            mapping.flush()
        except (ValueError, OSError) as error:
            failure = error
    raw = getattr(mapping, "_mmap", None)
    if raw is not None:
        try:
            raw.close()
        except (BufferError, OSError, ValueError) as error:
            if failure is None:
                failure = error
    if failure is not None and strict:
        raise FrameStoreError(
            f"the disk-backed frame store could not be released cleanly: {failure}"
        ) from failure


@contextmanager
def temp_frame_store(
    spec: FrameSpec,
    directory: str | os.PathLike[str] | None,
    *,
    chunk_bytes: int | None = None,
) -> Iterator[FrameStore]:
    """Float32 frames on disk in `directory`, deleted on every exit path.

    The store replaces a RAM-resident intermediate: the free space of
    `directory` is checked against `spec.nbytes` before the file exists, the file
    is preallocated to that size, and the mapping is flushed, closed and unlinked
    on success, on an exception and on a BaseException cancel. The file is never
    mapped as a whole: at most `chunk_bytes` (by default
    `FRAME_STORE_CHUNK_BYTES`) worth of whole frames is mapped at any time.

    Nothing survives a failure on the way to a live store either: a failing
    preallocation, a failing fd close or an invalid `chunk_bytes` removes the file
    again, so only a yielded store can leave a `.float32` file behind.
    """
    if directory is None:
        raise FrameStoreError("the disk-backed intermediate frame store needs a temporary directory")
    directory = os.fspath(directory)
    os.makedirs(directory, exist_ok=True)
    required = spec.nbytes
    try:
        available = shutil.disk_usage(directory).free
    except OSError as error:
        raise FrameStoreError(f"cannot measure the free space of {directory}: {error}") from error
    if available < required:
        raise FrameStoreError(
            f"the {spec.count} frame intermediate needs {required} bytes as float32 "
            f"but only {available} bytes are free in {directory}"
        )
    handle, path = tempfile.mkstemp(prefix="my_nodes_frames_", suffix=".float32", dir=directory)
    try:
        try:
            os.ftruncate(handle, required)
        finally:
            # Close before the store exists, but inside this block: a failing
            # ftruncate or close must not leave the fd or the file behind.
            os.close(handle)
        store = FrameStore(path, spec, chunk_bytes=chunk_bytes)
    except BaseException:
        # An invalid spec or chunk cap fails here, before anything holds the
        # file: unlink it so only a live store ever leaves a .float32 behind.
        _unlink(path, strict=False)
        raise
    try:
        yield store
    except BaseException:
        # Never mask the run's own failure while cleaning up after it.
        _release_store(store, path, strict=False)
        raise
    _release_store(store, path, strict=True)


def _release_store(store: FrameStore, path: str, *, strict: bool) -> None:
    """Release every mapping of the store, then delete the file behind it.

    Both steps are attempted even when the first fails, so a broken mapping never
    leaves the intermediate on disk. The mapping failure is the one reported when
    both fail: it is what makes the intermediate untrustworthy.
    """
    failure: FrameStoreError | None = None
    try:
        store.close(strict=strict)
    except FrameStoreError as error:
        failure = error
    try:
        _unlink(path, strict=strict)
    except FrameStoreError:
        if failure is None:
            raise
    if failure is not None:
        raise failure


def _unlink(path: str, *, strict: bool) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as error:
        if strict:
            raise FrameStoreError(f"the temporary frame store {path} could not be deleted: {error}") from error


@dataclass(frozen=True)
class VfiStageOptions:
    """GIMM-VFI settings for one interpolation stage."""

    precision: str
    models_dir: str | os.PathLike[str]
    ds_factor: float = 1.0
    node_mappings: dict | None = None
    load_device: object | None = None
    memory_required: int | None = None


@dataclass(frozen=True)
class DlssStageOptions:
    """DNR3 settings for one DLSS stage."""

    runtime_dir: str
    motion_mode: str
    scene_cut_threshold: float
    channel_order: str = "auto"
    wine_prefix: str = ""
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    driver_factory: Callable[..., HostDriver] | None = None
    memory_hooks: tuple[Callable[[], None], Callable[[], None]] | None = None


@dataclass(frozen=True)
class PipelineResult:
    """What one pipeline run produced, without holding any of its frames."""

    frame_count: int
    output_height: int
    output_width: int
    channel_order: str | None
    features: int
    stages: tuple[str, ...]


def run_frame_pipeline(
    source: Iterable[np.ndarray],
    source_spec: FrameSpec,
    plan: VideoEnhancePlan,
    write_frame: Callable[[int, np.ndarray], None],
    *,
    temp_directory: str | os.PathLike[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    interrupt: Callable[[], object] | None = None,
    vfi: VfiStageOptions | None = None,
    dlss: DlssStageOptions | None = None,
) -> PipelineResult:
    """Run the enabled stages in order and hand each final frame to `write_frame`.

    `source` is an iterable of float32 `[H,W,3]` frames whose exact length is
    `source_spec.count`; a shorter or longer source is an error, never padded or
    truncated. `write_frame(index, frame)` receives the final frames in order, so
    no output batch is assembled here. Two active stages need `temp_directory`:
    stage 1 is drained into the disk-backed store, which maps one bounded chunk at
    a time, and closed before stage 2 starts. `progress(done, total)` is
    cumulative over the whole run.
    """
    specs = pipeline_specs(source_spec, plan)
    if STAGE_VFI in specs.stages and vfi is None:
        raise FramePipelineError("frame interpolation is enabled but no VfiStageOptions were given")
    if STAGE_DLSS in specs.stages and dlss is None:
        raise FramePipelineError("the DLSS stage is enabled but no DlssStageOptions were given")
    frames = _checked_source(source, source_spec)
    counts = stage_step_counts(source_spec, plan)
    total_steps = sum(count for _stage, count in counts)

    def stage_progress(stage: str) -> Callable[[int, int], None] | None:
        """Cumulative, deterministic progress for one stage of this run."""
        if progress is None:
            return None
        base = sum(count for name, count in counts[: specs.stages.index(stage)])

        def report(done: int, _stage_total: int) -> None:
            progress(base + done, total_steps)

        return report

    if not specs.stages:
        _forward(frames, source_spec.count, write_frame)
        return _result(specs, None, 0)

    if len(specs.stages) == 1:
        if specs.stages[0] == STAGE_DLSS:
            with _open_dlss_stage(
                plan, frames, source_spec, dlss, stage_progress(STAGE_DLSS), interrupt
            ) as stream:
                _forward(iter(stream), specs.final.count, write_frame)
                return _result(specs, stream.channel_order, stream.features)
        vfi_frames = _open_vfi_stage(frames, source_spec, vfi, stage_progress(STAGE_VFI), interrupt)
        try:
            _forward(vfi_frames, specs.final.count, write_frame)
        finally:
            vfi_frames.close()
        return _result(specs, None, 0)

    first, second = specs.stages
    intermediate = specs.intermediate
    if intermediate is None:  # pragma: no cover - two stages always define one
        raise FramePipelineError("two active stages need a disk-backed intermediate spec")
    channel_order: str | None = None
    features = 0
    with temp_frame_store(intermediate, temp_directory) as store:
        if first == STAGE_DLSS:
            with _open_dlss_stage(
                plan, frames, source_spec, dlss, stage_progress(STAGE_DLSS), interrupt
            ) as stream:
                _forward(iter(stream), intermediate.count, store.write)
            channel_order, features = stream.channel_order, stream.features
        else:
            vfi_frames = _open_vfi_stage(
                frames, source_spec, vfi, stage_progress(STAGE_VFI), interrupt
            )
            try:
                _forward(vfi_frames, intermediate.count, store.write)
            finally:
                vfi_frames.close()
        # Every write mapping is released before stage 2 maps its own chunks of
        # the same file, so one bounded chunk is mapped at a time either way.
        store.finish()
        # The first stage is fully drained and torn down before the second one
        # starts: GIMM and the Wine/DLSS worker never share the GPU.
        if second == STAGE_DLSS:
            with _open_dlss_stage(
                plan, store, intermediate, dlss, stage_progress(STAGE_DLSS), interrupt
            ) as stream:
                _forward(iter(stream), specs.final.count, write_frame)
                channel_order, features = stream.channel_order, stream.features
        else:
            vfi_frames = _open_vfi_stage(
                store, intermediate, vfi, stage_progress(STAGE_VFI), interrupt
            )
            try:
                _forward(vfi_frames, specs.final.count, write_frame)
            finally:
                vfi_frames.close()
    return _result(specs, channel_order, features)


def _result(specs: PipelineSpecs, channel_order: str | None, features: int) -> PipelineResult:
    return PipelineResult(
        frame_count=specs.final.count,
        output_height=specs.final.height,
        output_width=specs.final.width,
        channel_order=channel_order,
        features=features,
        stages=specs.stages,
    )


def _checked_source(source: Iterable[np.ndarray], spec: FrameSpec) -> Iterator[np.ndarray]:
    """Validate every input frame against the declared source spec."""
    for index, frame in enumerate(source):
        array = np.asarray(frame)
        if array.dtype != FRAME_DTYPE or array.shape != (spec.height, spec.width, 3):
            raise FramePipelineError(
                f"source frame {index} must be float32 {(spec.height, spec.width, 3)}, "
                f"got {array.dtype} {array.shape}"
            )
        yield array


def _forward(
    frames: Iterable[np.ndarray], expected_count: int, write_frame: Callable[[int, np.ndarray], None]
) -> None:
    """Write exactly `expected_count` frames; a short or long stage is an error."""
    iterator = iter(frames)
    for index in range(expected_count):
        try:
            frame = next(iterator)
        except StopIteration:
            raise FramePipelineError(
                f"expected {expected_count} frames but the stage produced only {index}"
            ) from None
        write_frame(index, frame)
    try:
        next(iterator)
    except StopIteration:
        return
    raise FramePipelineError(
        f"expected exactly {expected_count} frames but the stage produced more"
    )


def _open_dlss_stage(
    plan: VideoEnhancePlan,
    source: Iterable[np.ndarray],
    source_spec: FrameSpec,
    options: DlssStageOptions,
    progress: Callable[[int, int], None] | None,
    interrupt: Callable[[], object] | None,
) -> DlssStageStream:
    """One DLSS stage over `source`, sized by `source_spec`."""
    return DlssStageStream(
        plan,
        source,
        count=source_spec.count,
        height=source_spec.height,
        width=source_spec.width,
        runtime_dir=options.runtime_dir,
        wine_prefix=options.wine_prefix,
        channel_order=options.channel_order,
        motion_mode=options.motion_mode,
        scene_cut_threshold=options.scene_cut_threshold,
        timeout=options.timeout,
        progress=progress,
        interrupt=interrupt,
        driver_factory=options.driver_factory,
        memory_hooks=options.memory_hooks,
    )


def _open_vfi_stage(
    source: Iterable[np.ndarray],
    source_spec: FrameSpec,
    options: VfiStageOptions,
    progress: Callable[[int, int], None] | None,
    interrupt: Callable[[], object] | None,
) -> Iterator[np.ndarray]:
    """One GIMM-VFI stage over `source`, sized by `source_spec`."""
    return iter_interpolate_offline(
        source,
        source_spec.count,
        precision=options.precision,
        ds_factor=options.ds_factor,
        models_dir=options.models_dir,
        node_mappings=options.node_mappings,
        load_device=options.load_device,
        memory_required=options.memory_required,
        progress=progress,
        interrupt=interrupt,
    )
