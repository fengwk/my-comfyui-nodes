"""Offline 2x interpolation through the installed ComfyUI-GIMM-VFI nodes.

This module does not vendor, copy or reimplement the S-Lab GIMM algorithm.
It looks up the already registered `DownloadAndLoadGIMMVFIModel` and
`GIMMVFI_interpolate` classes and calls their public methods. The loaded
module is moved back to CPU and wrapped in a real `ModelPatcher`; Comfy's
model manager loads it for the run and unloads it again on every exit path.
"""

from __future__ import annotations

import os
from collections.abc import Callable

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
    """Interpolate adjacent pairs one at a time. N source frames become 2*N-1.

    N=1 is returned unchanged and does not load the model. The patcher is
    unloaded in `finally`, including when Comfy raises a BaseException.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 1:
        raise GimmVfiError(f"VFI input must be [N,H,W,3], got shape {frames.shape}")
    count = int(frames.shape[0])
    if count == 1:
        return np.ascontiguousarray(frames, dtype=np.float32)
    if not np.isfinite(ds_factor) or not 0.01 <= float(ds_factor) <= 1.0:
        raise GimmVfiError(f"vfi_ds_factor must be within [0.01, 1.0], got {ds_factor!r}")

    import comfy.model_management as mm
    import torch

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
        required = patcher.model_size() if memory_required is None else int(memory_required)
        output = np.empty((2 * count - 1, frames.shape[1], frames.shape[2], 3), dtype=np.float32)
        pairs = count - 1
        mm.load_models_gpu([patcher], memory_required=required, force_full_load=True)
        module = patcher.model
        with torch.inference_mode():
            for index in range(pairs):
                if interrupt is not None:
                    interrupt()
                produced = interpolator.interpolate(
                    module,
                    _pair_batch(frames[index], frames[index + 1]),
                    float(ds_factor),
                    GIMM_FACTOR,
                    GIMM_SEED,
                    output_flows=False,
                )
                triple = _as_frames(produced)
                output[index * 2] = triple[0]
                output[index * 2 + 1] = triple[1]
                if progress is not None:
                    progress(index + 1, pairs)
        output[-1] = np.ascontiguousarray(frames[-1], dtype=np.float32)
        return output
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
