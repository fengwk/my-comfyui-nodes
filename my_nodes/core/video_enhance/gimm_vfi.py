"""Offline 2x interpolation through the installed ComfyUI-GIMM-VFI nodes.

This module does not vendor, copy or reimplement the S-Lab GIMM algorithm.
It looks up the already registered `DownloadAndLoadGIMMVFIModel` and
`GIMMVFI_interpolate` classes and calls their public methods. The loaded
module is moved back to CPU and wrapped in a real `ModelPatcher`; Comfy's
model manager loads it for the run and unloads it again on every exit path.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from functools import partial
from types import FunctionType, MethodType

import numpy as np

# Installed next to this pack, under ComfyUI/custom_nodes.
DEFAULT_PLUGIN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "ComfyUI-GIMM-VFI")
)
GIMM_MODEL_NAME = "gimmvfi_r_arb_lpips_fp32.safetensors"
GIMM_FLOW_NAME = "raft-things_fp32.safetensors"
GIMM_FACTOR = 2
GIMM_SEED = 0
VFI_PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16")

_PATCHERS: dict[tuple[str, str], object] = {}


class GimmVfiError(RuntimeError):
    """The installed GIMM plugin or its offline weights cannot be used."""


class _QuietProgressBar:
    """Call-local replacement for the two ProgressBar operations GIMM uses."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def update(self, *_args, **_kwargs) -> None:
        pass


def _quiet_tqdm(iterable, *_args, **_kwargs):
    """Fallback for GIMM's `tqdm(range(...))` when no tqdm global is present."""
    return iterable


def _quiet_interpolate_method(method) -> MethodType:
    """Clone a Python bound method with only its progress globals shadowed.

    The installed plugin exposes no progress switch, so a shallow globals copy
    keeps concurrent standalone calls untouched. This reuses the plugin's code
    object and binding; it does not copy or reimplement its interpolation.
    """
    if (
        not isinstance(method, MethodType)
        or method.__self__ is None
        or not isinstance(method.__func__, FunctionType)
    ):
        raise GimmVfiError(
            "GIMMVFI_interpolate.interpolate must be an ordinary Python bound instance method "
            "so its progress reporting can be isolated per call"
        )

    original = method.__func__
    globals_copy = original.__globals__.copy()
    globals_copy["ProgressBar"] = _QuietProgressBar
    original_tqdm = globals_copy.get("tqdm")
    if original_tqdm is None:
        globals_copy["tqdm"] = _quiet_tqdm
    elif callable(original_tqdm):
        globals_copy["tqdm"] = partial(original_tqdm, disable=True)
    else:
        raise GimmVfiError(
            "GIMMVFI_interpolate.interpolate has a non-callable tqdm global; "
            "its progress reporting cannot be isolated"
        )

    cloned = FunctionType(
        original.__code__,
        globals_copy,
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    cloned.__doc__ = original.__doc__
    cloned.__module__ = original.__module__
    cloned.__qualname__ = original.__qualname__
    cloned.__annotations__ = original.__annotations__
    cloned.__dict__.update(original.__dict__)
    if hasattr(original, "__type_params__"):
        cloned.__type_params__ = original.__type_params__
    return MethodType(cloned, method.__self__)


def gimm_model_dir(models_dir: str | os.PathLike[str]) -> str:
    return os.path.join(os.fspath(models_dir), "interpolation", "gimm-vfi")


def require_offline_weights(models_dir: str | os.PathLike[str]) -> tuple[str, str]:
    """Require both GIMM-VFI-R files. Never calls the plugin's downloader."""
    directory = gimm_model_dir(models_dir)
    missing = [
        os.path.join(directory, name)
        for name in (GIMM_MODEL_NAME, GIMM_FLOW_NAME)
        if not os.path.isfile(os.path.join(directory, name))
    ]
    if missing:
        listed = ", ".join(missing)
        raise GimmVfiError(
            "GIMM-VFI offline weights are missing: "
            f"{listed}. Install {GIMM_MODEL_NAME} and {GIMM_FLOW_NAME} under "
            f"{directory}. This node does not download them."
        )
    return os.path.join(directory, GIMM_MODEL_NAME), os.path.join(directory, GIMM_FLOW_NAME)


def resolve_gimm_nodes(
    node_mappings: dict | None = None, *, plugin_path: str | None = None
) -> tuple[type, type]:
    """Return the installed loader and interpolator classes. Never imports GIMM source."""
    if node_mappings is None:
        import nodes

        node_mappings = nodes.NODE_CLASS_MAPPINGS
    try:
        loader = node_mappings["DownloadAndLoadGIMMVFIModel"]
        interpolator = node_mappings["GIMMVFI_interpolate"]
    except KeyError as exc:
        installed = plugin_path or DEFAULT_PLUGIN_PATH
        raise GimmVfiError(
            "ComfyUI-GIMM-VFI is not registered (missing DownloadAndLoadGIMMVFIModel "
            f"or GIMMVFI_interpolate). Install the plugin at {installed} and restart "
            "ComfyUI. Its S-Lab license is non-commercial and this pack does not vendor that code."
        ) from exc
    return loader, interpolator


def _move_to_cpu(module, cpu):
    module.to(cpu)
    module.device = cpu
    flow = getattr(module, "flow_estimator", None)
    if flow is not None and hasattr(flow, "to"):
        flow.to(cpu)
        if hasattr(flow, "device"):
            flow.device = cpu
    return module


def _clear_gimm_backwarp_cache(module) -> None:
    """Drop the installed GIMM-R plugin's resolution-keyed CUDA grid cache."""
    method = getattr(module, "warp_w_mask", None)
    function = getattr(method, "__func__", method)
    warp = getattr(function, "__globals__", {}).get("warp")
    cache = getattr(warp, "__globals__", {}).get("backwarp_tenGrid")
    if isinstance(cache, dict):
        cache.clear()


def _clear_cublas_workspaces(torch_module) -> None:
    """Release PyTorch's version-specific cuBLAS workspace cache when available."""
    core = getattr(torch_module, "_C", None)
    clear = getattr(core, "_cuda_clearCublasWorkspaces", None)
    if callable(clear):
        clear()


def _checkpoint_key(precision: str, model_path: str, flow_path: str) -> tuple[str, str]:
    identity = f"{os.path.abspath(model_path)}:{os.path.getmtime(model_path)}:{os.path.getsize(model_path)}"
    identity += f"|{os.path.abspath(flow_path)}:{os.path.getmtime(flow_path)}:{os.path.getsize(flow_path)}"
    return precision, identity


def cached_patcher(
    precision: str,
    models_dir: str | os.PathLike[str],
    *,
    node_mappings: dict | None = None,
    load_device=None,
    offload_device=None,
):
    """Load through the external node, park the real module on CPU, and cache its patcher."""
    if precision not in VFI_PRECISIONS:
        raise GimmVfiError(f"vfi_precision must be one of {VFI_PRECISIONS}, got {precision!r}")
    model_path, flow_path = require_offline_weights(models_dir)
    key = _checkpoint_key(precision, model_path, flow_path)
    cached = _PATCHERS.get(key)
    if cached is not None:
        return cached

    import torch
    from comfy.model_patcher import ModelPatcher

    if offload_device is None:
        offload_device = torch.device("cpu")
    loader_cls, _interpolator_cls = resolve_gimm_nodes(node_mappings)
    loaded = loader_cls().loadmodel(GIMM_MODEL_NAME, precision=precision, torch_compile=False)
    module = loaded[0]
    if not isinstance(module, torch.nn.Module):
        raise GimmVfiError(
            "DownloadAndLoadGIMMVFIModel.loadmodel did not return a torch.nn.Module; "
            f"got {type(module).__name__}"
        )
    _move_to_cpu(module, offload_device)
    patcher = ModelPatcher(module, load_device=load_device, offload_device=offload_device)
    _PATCHERS[key] = patcher
    return patcher


def clear_patcher_cache() -> None:
    """Drop cached patchers. Tests use this; production keeps them on CPU."""
    _PATCHERS.clear()


def _pair_batch(left: np.ndarray, right: np.ndarray):
    import torch

    stacked = np.stack((left, right), axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(stacked))


def _as_frames(result) -> np.ndarray:
    images = result[0]
    if hasattr(images, "detach"):
        array = images.detach().cpu().float().numpy()
    else:
        array = np.asarray(images, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] != 3 or array.shape[-1] != 3:
        raise GimmVfiError(
            "GIMMVFI_interpolate did not return three RGB frames for factor 2, "
            f"got shape {getattr(array, 'shape', None)}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def interpolate_offline(
    frames: np.ndarray,
    *,
    precision: str,
    ds_factor: float,
    models_dir: str | os.PathLike[str],
    node_mappings: dict | None = None,
    load_device=None,
    memory_required: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    interrupt: Callable[[], None] | None = None,
) -> np.ndarray:
    """Collect `iter_interpolate_offline` for one in-memory batch. N frames become 2*N-1.

    The signature and the result are unchanged: this is the compatibility
    collector over the incremental iterator, which owns pair assembly, the
    model lifecycle and the frame-count contract.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 1:
        raise GimmVfiError(f"VFI input must be [N,H,W,3], got shape {frames.shape}")
    count = int(frames.shape[0])
    output = np.empty((2 * count - 1, frames.shape[1], frames.shape[2], 3), dtype=np.float32)
    stream = iter_interpolate_offline(
        frames,
        count,
        precision=precision,
        ds_factor=ds_factor,
        models_dir=models_dir,
        node_mappings=node_mappings,
        load_device=load_device,
        memory_required=memory_required,
        progress=progress,
        interrupt=interrupt,
    )
    try:
        for index, frame in enumerate(stream):
            output[index] = frame
    finally:
        # Exhausted already on the normal path; this unloads the model when the
        # collector is interrupted between two frames.
        stream.close()
    return output


def _require_frame_count(frame_count: object) -> int:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise GimmVfiError(f"frame_count must be an int, got {type(frame_count).__name__}")
    if frame_count < 1:
        raise GimmVfiError(f"frame_count must be at least 1, got {frame_count}")
    return frame_count


def _next_frame(source, index: int, count: int) -> np.ndarray:
    """Pull one staged `[H,W,3]` frame; an early end of the source is an error."""
    try:
        frame = next(source)
    except StopIteration:
        raise GimmVfiError(
            f"VFI expected {count} frames but the source ended early at frame {index}"
        ) from None
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise GimmVfiError(f"VFI frame {index} must be [H,W,3], got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _require_exhausted(source, count: int) -> None:
    try:
        next(source)
    except StopIteration:
        return
    raise GimmVfiError(f"VFI expected exactly {count} frames but the source produced more")


def iter_interpolate_offline(
    frames,
    frame_count: int,
    *,
    precision: str,
    ds_factor: float,
    models_dir: str | os.PathLike[str],
    node_mappings: dict | None = None,
    load_device=None,
    memory_required: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    interrupt: Callable[[], None] | None = None,
) -> Iterator[np.ndarray]:
    """Interpolate adjacent pairs one at a time, yielding F0,M01,F1,...,F_last.

    `frames` is an iterable of float32 `[H,W,3]` frames and `frame_count` its
    exact length: a shorter or longer source is an error, never padded or
    truncated. One frame is yielded unchanged and loads no model. The module and
    the cuBLAS workspace are released in `finally`, so an exhausted generator, a
    `GimmVfiError` and a `close()` from the caller all unload the same way.
    """
    count = _require_frame_count(frame_count)
    source = iter(frames)
    if count == 1:
        frame = _next_frame(source, 0, count)
        _require_exhausted(source, count)
        yield frame
        return
    ds_factor = float(ds_factor)
    if not np.isfinite(ds_factor) or not 0.01 <= ds_factor <= 1.0:
        raise GimmVfiError(f"vfi_ds_factor must be within [0.01, 1.0], got {ds_factor!r}")
    left = _next_frame(source, 0, count)

    import comfy.model_management as mm
    import torch

    pairs = count - 1
    device = load_device if load_device is not None else mm.get_torch_device()
    patcher = None
    try:
        # The external loader moves both GIMM and RAFT directly to `device`
        # before this module can wrap them in a ModelPatcher. Make room first.
        mm.free_memory(1e30, device)
        mm.soft_empty_cache()
        patcher = cached_patcher(
            precision,
            models_dir,
            node_mappings=node_mappings,
            load_device=device,
            offload_device=torch.device("cpu"),
        )
        _loader_cls, interpolator_cls = resolve_gimm_nodes(node_mappings)
        interpolator = interpolator_cls()
        interpolate = _quiet_interpolate_method(interpolator.interpolate)
        required = patcher.model_size() if memory_required is None else int(memory_required)
        mm.load_models_gpu([patcher], memory_required=required, force_full_load=True)
        module = patcher.model
        for index in range(pairs):
            if interrupt is not None:
                interrupt()
            right = _next_frame(source, index + 1, count)
            # Inference mode must not span a yield: the caller's own work runs
            # while this generator is suspended.
            with torch.inference_mode():
                produced = interpolate(
                    module,
                    _pair_batch(left, right),
                    ds_factor,
                    GIMM_FACTOR,
                    GIMM_SEED,
                    output_flows=False,
                )
            # The plugin's ProgressBar hook normally performs this cancellation
            # check. Its call-local quiet replacement must preserve that timing.
            if interrupt is not None:
                interrupt()
            triple = _as_frames(produced)
            yield triple[0]
            yield triple[1]
            left = right
            if progress is not None:
                progress(index + 1, pairs)
        _require_exhausted(source, count)
        yield left
    finally:
        try:
            if patcher is not None:
                try:
                    _clear_gimm_backwarp_cache(patcher.model)
                finally:
                    mm.unload_model_and_clones(patcher)
        finally:
            try:
                _clear_cublas_workspaces(torch)
            finally:
                mm.soft_empty_cache()
