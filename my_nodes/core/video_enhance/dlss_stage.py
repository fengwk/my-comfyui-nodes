"""One DLSS execution: validate frames, free Comfy memory, run one DNR3 worker.

Feature 1 (DLAA at 1.0x, or SR above it) and optional feature 18 share that
single worker. The worker context is closed before this function returns, on
every path including a ComfyUI BaseException interrupt.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from my_nodes.core.video_enhance import dnr3
from my_nodes.core.video_enhance.channel_order import apply_channel_order, select_channel_order
from my_nodes.core.video_enhance.dlss_worker import DEFAULT_TIMEOUT_SECONDS, Dnr3Worker
from my_nodes.core.video_enhance import motion as motion_guides
from my_nodes.core.video_enhance.nr_profiles import NeuralRenderingSettings, neural_rendering_settings
from my_nodes.core.video_enhance.plan import VideoEnhancePlan
from my_nodes.core.video_enhance.runtime import FEATURE_NR, FEATURE_SR, HostDriver

RUNTIME_DIR_ENV = "DLSS5_RUNTIME_DIR"


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
    """Build the one DNR3 header for this batch. Header validation checks the scale."""
    if not plan.uses_dlss:
        raise FrameValidationError("refusing to build a DLSS header when both DLSS features are off")
    count, height, width = (int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2]))
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
        settings = neural_rendering_settings(plan.nr_profile, plan.nr_intensity)
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
        ui_correction=False,
        intensity=settings.intensity,
        tone=settings.tone,
        structure=settings.structure,
        skin=settings.skin,
        global_tone=settings.global_tone,
    )


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
    """Run feature 1 and optional feature 18 in one scoped worker, then close it.

    `frames` must already be the contiguous CPU copy from `prepare_frames`.
    Memory hooks run after that copy exists and before the worker is spawned.
    """
    header = build_header(plan, frames)
    guides = motion_guides.MotionGuides(motion_mode, scene_cut_threshold)
    if motion_mode == motion_guides.MOTION_OPTICAL_FLOW:
        # Import before Wine starts. A one-frame preflight would miss this,
        # because the first frame is a reset and does not call OpenCV.
        motion_guides._import_cv2()
    if memory_hooks is not None:
        free_memory, empty_cache = memory_hooks
        free_memory()
        empty_cache()
    factory = driver_factory if driver_factory is not None else HostDriver.wine
    prefix = wine_prefix.strip() or None
    driver = factory(runtime_dir=runtime_dir, features=header.features, wine_prefix=prefix)
    output = np.empty(
        (header.frame_count, header.output_height, header.output_width, 3), dtype=np.float32
    )
    resolved_order: str | None = None if channel_order == "auto" else channel_order
    with Dnr3Worker(header, driver=driver, timeout=timeout, interrupt=interrupt) as worker:
        for index in range(header.frame_count):
            guide = guides.guide(frames[index])
            produced = worker.enhance(frames[index], guide.motion, reset=guide.reset)
            if resolved_order is None:
                resolved_order = select_channel_order("auto", frames[index], produced)
            output[index] = apply_channel_order(produced, resolved_order)
            if progress is not None:
                progress(index + 1, header.frame_count)
    assert resolved_order is not None
    return DlssStageResult(
        frames=output,
        output_width=header.output_width,
        output_height=header.output_height,
        channel_order=resolved_order,
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
