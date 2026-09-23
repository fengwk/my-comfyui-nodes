"""One DLSS execution: validate frames, free Comfy memory, run one DNR3 worker.

Feature 1 (DLAA at 1.0x, or SR above it) and optional feature 18 share that
single worker. The worker context is closed before this function returns, on
every path including a ComfyUI BaseException interrupt.

The header carries the neural-rendering model fields and the child environment
carries the launch-time ones (UI correction, post-NR detail/color, the SR preset,
the GPU index and the requested channel order). Both are built from the same
validated plan; the environment values are written explicitly on every launch.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace

import numpy as np

from my_nodes.core.video_enhance import dnr3
from my_nodes.core.video_enhance.channel_order import (
    CHANNEL_ORDERS,
    apply_channel_order,
    select_channel_order,
)
from my_nodes.core.video_enhance.dlss_worker import DEFAULT_TIMEOUT_SECONDS, Dnr3Worker
from my_nodes.core.video_enhance import motion as motion_guides
from my_nodes.core.video_enhance.nr_profiles import (
    NeuralRenderingSettings,
    plan_neural_rendering_settings,
)
from my_nodes.core.video_enhance.plan import VideoEnhancePlan
from my_nodes.core.video_enhance.runtime import FEATURE_NR, FEATURE_SR, HostDriver

RUNTIME_DIR_ENV = "DLSS5_RUNTIME_DIR"

# Child environment the stage always writes explicitly for the DNR3 host. Every
# name is set on every launch, also when a feature is off, so a stale value left
# in the parent environment can never change what the worker does.
UI_CORRECTION_ENV = "DLSS5NR_UI_CORRECTION"
DETAIL_ENV = "DLSS5NR_DETAIL"
COLOR_ENV = "DLSS5NR_COLOR"
SR_PRESET_ENV = "DLSS5NR_SR_PRESET"
GPU_INDEX_ENV = "DLSS5NR_GPU_INDEX"
CHANNEL_ORDER_ENV = "DLSS5NR_CHANNEL_ORDER"
# User-facing SR preset -> the DLSS model selector the runtime is pinned to.
SR_PRESET_IDS: Mapping[str, int] = {
    "Default": 0,
    "E": 5,
    "F": 6,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
}


class FrameValidationError(ValueError):
    """The IMAGE batch cannot be sent to the DLSS worker."""


@dataclass(frozen=True)
class DlssStageResult:
    frames: np.ndarray
    output_width: int
    output_height: int
    channel_order: str
    features: int


def output_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    """Even-rounded target size. Scale 1.0 keeps the native dimensions."""
    if scale == 1.0:
        return width, height
    out_width = max(2, int(math.floor(width * scale + 0.5)))
    out_height = max(2, int(math.floor(height * scale + 0.5)))
    if out_width % 2:
        out_width += 1
    if out_height % 2:
        out_height += 1
    return out_width, out_height


def prepare_frames(images: object) -> np.ndarray:
    """Validate a nonempty finite `[N,H,W,3]` batch and stage contiguous CPU float32."""
    if hasattr(images, "detach"):
        batch = images.detach().cpu().numpy()
    else:
        batch = np.asarray(images)
    if batch.ndim != 4 or batch.shape[-1] != 3:
        raise FrameValidationError(
            f"DLSS input must be a nonempty IMAGE batch [N,H,W,3], got shape {getattr(batch, 'shape', None)}"
        )
    if batch.shape[0] < 1 or batch.shape[1] < 1 or batch.shape[2] < 1:
        raise FrameValidationError(
            f"DLSS input must be a nonempty IMAGE batch [N,H,W,3], got shape {batch.shape}"
        )
    frames = np.ascontiguousarray(batch, dtype=np.float32)
    if not np.isfinite(frames).all():
        raise FrameValidationError("DLSS input contains non-finite values")
    return frames


def build_header(
    plan: VideoEnhancePlan,
    frames: np.ndarray,
    settings: NeuralRenderingSettings | None = None,
) -> dnr3.Header:
    """Build the one DNR3 header for an in-memory batch; header validation checks the scale."""
    return build_header_for(
        plan,
        count=int(frames.shape[0]),
        height=int(frames.shape[1]),
        width=int(frames.shape[2]),
        settings=settings,
    )


def build_header_for(
    plan: VideoEnhancePlan,
    *,
    count: int,
    height: int,
    width: int,
    settings: NeuralRenderingSettings | None = None,
) -> dnr3.Header:
    """Build the header for one frame spec, without staging the frames themselves."""
    if not plan.uses_dlss:
        raise FrameValidationError("refusing to build a DLSS header when both DLSS features are off")
    features = 0
    if plan.enable_super_resolution:
        features |= FEATURE_SR
        out_width, out_height = output_dimensions(width, height, plan.sr_scale)
        perf_quality = dnr3.perf_quality_for_scale(plan.sr_scale)
    else:
        out_width, out_height = width, height
        perf_quality = dnr3.NATIVE_PERF_QUALITY
    if plan.enable_neural_rendering:
        features |= FEATURE_NR
    if settings is None:
        settings = plan_neural_rendering_settings(plan)
    return dnr3.Header(
        input_width=width,
        input_height=height,
        output_width=out_width,
        output_height=out_height,
        warmup_frames=0,
        frame_count=count,
        perf_quality=perf_quality,
        features=features,
        preset=settings.preset,
        style=settings.style,
        automask=settings.automask,
        ui_correction=settings.ui_correction,
        intensity=settings.intensity,
        tone=settings.tone,
        structure=settings.structure,
        skin=settings.skin,
        global_tone=settings.global_tone,
    )


def _env_number(value: float) -> str:
    """Deterministic decimal text for one plan value, so equal plans set equal env."""
    return repr(float(value))


def worker_environment(plan: VideoEnhancePlan, channel_order: str) -> dict[str, str]:
    """The child environment values one DLSS worker launch always sets itself.

    `channel_order` is the order the user asked for (`auto`, `RGBA` or `BGRA`),
    not the resolved one: the host needs the request, the stage decides the
    readback order. Every value is written explicitly, so an inherited variable
    of the same name is replaced instead of silently changing the run.
    """
    if channel_order not in CHANNEL_ORDERS:
        raise FrameValidationError(
            f"channel order must be one of {CHANNEL_ORDERS}, got {channel_order!r}"
        )
    settings = plan_neural_rendering_settings(plan)
    return {
        UI_CORRECTION_ENV: "1" if settings.ui_correction else "0",
        DETAIL_ENV: _env_number(plan.nr_detail),
        COLOR_ENV: _env_number(plan.nr_color),
        SR_PRESET_ENV: str(SR_PRESET_IDS[plan.sr_preset]),
        GPU_INDEX_ENV: str(plan.gpu_index),
        CHANNEL_ORDER_ENV: channel_order,
    }


def apply_worker_environment(
    driver: HostDriver, plan: VideoEnhancePlan, channel_order: str
) -> HostDriver:
    """Overlay the explicit worker values on a driver, replacing stale parents.

    The driver stays frozen and keeps everything else it carried: a host built
    from the real process environment or by a custom factory only has these six
    values replaced.
    """
    return replace(driver, env={**driver.env, **worker_environment(plan, channel_order)})


def resolve_runtime_dir(
    override: str,
    *,
    env: dict[str, str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Widget value, then DLSS5_RUNTIME_DIR, then `<models_dir>/dlss5`."""
    selected = override.strip()
    if not selected:
        environment = os.environ if env is None else env
        selected = environment.get(RUNTIME_DIR_ENV, "").strip()
    if not selected:
        if models_dir is None:
            import folder_paths

            models_dir = folder_paths.models_dir
        selected = os.path.join(os.fspath(models_dir), "dlss5")
    return selected


def _source_frame(source: Iterator[np.ndarray], index: int, header: dnr3.Header) -> np.ndarray:
    """Pull the next input frame; validate it before the worker sees any data."""
    try:
        frame = next(source)
    except StopIteration:
        raise FrameValidationError(
            f"DLSS stage expected {header.frame_count} frames but the input ended early at frame {index}"
        ) from None
    array = np.asarray(frame)
    if array.dtype != np.float32 or array.shape != (header.input_height, header.input_width, 3):
        raise FrameValidationError(
            f"DLSS stage frame {index} must be float32 "
            f"{(header.input_height, header.input_width, 3)}, got {array.dtype} {array.shape}"
        )
    if not np.isfinite(array).all():
        raise FrameValidationError(f"DLSS stage frame {index} contains non-finite values")
    return array


def _require_exhausted(source: Iterator[np.ndarray], count: int) -> None:
    try:
        next(source)
    except StopIteration:
        return
    raise FrameValidationError(
        f"DLSS stage expected exactly {count} frames but the input produced more"
    )


class DlssStageStream:
    """One DLSS stage over an iterable of frames: one worker, one guide chain.

    The stream is a context manager. `__enter__` validates the runtime, frees
    Comfy memory and starts exactly one `Dnr3Worker` for the whole stage, with
    the plan's launch environment layered over the driver's; `__exit__` always
    reaps that worker, including on a ComfyUI BaseException interrupt and on an
    abandoned iteration. Iterating yields one corrected float32 frame per input
    frame, in order, and refuses a source that is shorter or longer than `count`
    instead of padding or truncating it.

    `channel_order` is the resolved order: an explicit order is kept, `auto`
    is decided by the first input/output pair and then reused for the stage. The
    requested value also reaches the worker through `DLSS5NR_CHANNEL_ORDER`.
    """

    def __init__(
        self,
        plan: VideoEnhancePlan,
        source: Iterable[np.ndarray],
        *,
        count: int,
        height: int,
        width: int,
        runtime_dir: str,
        wine_prefix: str,
        channel_order: str,
        motion_mode: str,
        scene_cut_threshold: float,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        progress: Callable[[int, int], None] | None = None,
        interrupt: Callable[[], object] | None = None,
        driver_factory: Callable[..., HostDriver] | None = None,
        memory_hooks: tuple[Callable[[], None], Callable[[], None]] | None = None,
    ) -> None:
        self._header = build_header_for(plan, count=count, height=height, width=width)
        self._plan = plan
        self._source = source
        self._runtime_dir = runtime_dir
        self._wine_prefix = wine_prefix
        self._requested_channel_order = channel_order
        self._channel_order: str | None = None if channel_order == "auto" else channel_order
        self._motion_mode = motion_mode
        self._scene_cut_threshold = scene_cut_threshold
        self._timeout = timeout
        self._progress = progress
        self._interrupt = interrupt
        self._driver_factory = driver_factory
        self._memory_hooks = memory_hooks
        self._guides: motion_guides.MotionGuides | None = None
        self._worker: Dnr3Worker | None = None

    @property
    def header(self) -> dnr3.Header:
        """The validated header of this stage; it declares the output dimensions."""
        return self._header

    @property
    def features(self) -> int:
        return self._header.features

    @property
    def output_width(self) -> int:
        return self._header.output_width

    @property
    def output_height(self) -> int:
        return self._header.output_height

    @property
    def channel_order(self) -> str | None:
        """Resolved channel order, or None while `auto` has not seen a frame yet."""
        return self._channel_order

    def __enter__(self) -> DlssStageStream:
        guides = motion_guides.MotionGuides(self._motion_mode, self._scene_cut_threshold)
        if self._motion_mode == motion_guides.MOTION_OPTICAL_FLOW:
            # Import before Wine starts. A one-frame preflight would miss this,
            # because the first frame is a reset and does not call OpenCV.
            motion_guides._import_cv2()
        if self._memory_hooks is not None:
            free_memory, empty_cache = self._memory_hooks
            free_memory()
            empty_cache()
        factory = self._driver_factory if self._driver_factory is not None else HostDriver.wine
        prefix = self._wine_prefix.strip() or None
        if self._driver_factory is None:
            # The built-in driver takes the selected GPU, so its environment
            # already names the adapter before the explicit overlay below.
            driver = factory(
                runtime_dir=self._runtime_dir,
                features=self._header.features,
                wine_prefix=prefix,
                gpu_index=self._plan.gpu_index,
            )
        else:
            # A test or custom host factory only knows the original keywords;
            # its frozen driver is overlaid instead of being called differently.
            driver = factory(
                runtime_dir=self._runtime_dir, features=self._header.features, wine_prefix=prefix
            )
        worker = Dnr3Worker(
            self._header,
            driver=apply_worker_environment(driver, self._plan, self._requested_channel_order),
            timeout=self._timeout,
            interrupt=self._interrupt,
        )
        worker.__enter__()
        self._guides = guides
        self._worker = worker
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        worker, self._worker = self._worker, None
        if worker is None:
            return False
        return worker.__exit__(exc_type, exc, traceback)

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._worker is None or self._guides is None:
            raise FrameValidationError("the DLSS stage stream must be entered before it is iterated")
        worker = self._worker
        guides = self._guides
        header = self._header
        source = iter(self._source)
        for index in range(header.frame_count):
            frame = _source_frame(source, index, header)
            guide = guides.guide(frame)
            produced = worker.enhance(frame, guide.motion, reset=guide.reset)
            if self._channel_order is None:
                self._channel_order = select_channel_order("auto", frame, produced)
            if self._progress is not None:
                self._progress(index + 1, header.frame_count)
            yield apply_channel_order(produced, self._channel_order)
        _require_exhausted(source, header.frame_count)


def run_dlss_stage(
    plan: VideoEnhancePlan,
    frames: np.ndarray,
    *,
    runtime_dir: str,
    wine_prefix: str,
    channel_order: str,
    motion_mode: str,
    scene_cut_threshold: float,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    progress: Callable[[int, int], None] | None = None,
    interrupt: Callable[[], object] | None = None,
    driver_factory: Callable[..., HostDriver] | None = None,
    memory_hooks: tuple[Callable[[], None], Callable[[], None]] | None = None,
) -> DlssStageResult:
    """Collect `DlssStageStream` for one in-memory batch. Signature and result unchanged.

    `frames` must already be the contiguous CPU copy from `prepare_frames`.
    Memory hooks run after that copy exists and before the worker is spawned.
    """
    header = build_header(plan, frames)
    output = np.empty(
        (header.frame_count, header.output_height, header.output_width, 3), dtype=np.float32
    )
    stream = DlssStageStream(
        plan,
        frames,
        count=header.frame_count,
        height=header.input_height,
        width=header.input_width,
        runtime_dir=runtime_dir,
        wine_prefix=wine_prefix,
        channel_order=channel_order,
        motion_mode=motion_mode,
        scene_cut_threshold=scene_cut_threshold,
        timeout=timeout,
        progress=progress,
        interrupt=interrupt,
        driver_factory=driver_factory,
        memory_hooks=memory_hooks,
    )
    with stream:
        for index, frame in enumerate(stream):
            output[index] = frame
    assert stream.channel_order is not None
    return DlssStageResult(
        frames=output,
        output_width=header.output_width,
        output_height=header.output_height,
        channel_order=stream.channel_order,
        features=header.features,
    )


def default_memory_hooks() -> tuple[Callable[[], None], Callable[[], None]]:
    """Comfy hooks: free loaded models on the torch device, then drop the cache.

    `unload_all_models` is deliberately not used; it would also free every other
    device.
    """

    def free() -> None:
        import comfy.model_management as mm

        mm.free_memory(1e30, mm.get_torch_device())

    def empty() -> None:
        import comfy.model_management as mm

        mm.soft_empty_cache()

    return free, empty


def comfy_interrupt() -> None:
    """Raise Comfy's interrupt BaseException so ScopedProcess kills the group."""
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()


def probe_frame(width: int = 32, height: int = 32) -> np.ndarray:
    """One deterministic RGB frame for the runtime probe. Not a real image."""
    y = np.linspace(0.05, 0.95, height, dtype=np.float32)[:, None]
    x = np.linspace(0.05, 0.95, width, dtype=np.float32)[None, :]
    red = np.broadcast_to(x, (height, width))
    green = np.broadcast_to(y, (height, width))
    blue = np.full((height, width), 0.25, dtype=np.float32)
    frame = np.stack((red, green, blue), axis=-1)
    return frame.reshape(1, height, width, 3)
