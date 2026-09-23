"""Batched VOSR2 image/video-frame inference with bounded activation memory."""

from __future__ import annotations

import gc
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils

from .color import apply_color_alignment
from .settings import VOSR2Settings
from .tiled_vae import _gaussian_weights, _make_tile_grid

AE_FACTOR = 8
DIT_PATCH = 2
PAD_MULTIPLE = AE_FACTOR * DIT_PATCH
SAFE_VAE_TILE = 1024
MIN_VAE_TILE = 256
SPEED_DIT_TILE_BATCH = 4
BALANCED_DIT_TILE_BATCH = 2
SPEED_COLOR_DOWNSAMPLE = 4


def _pad_multiple(value: torch.Tensor, multiple: int = PAD_MULTIPLE) -> torch.Tensor:
    height, width = value.shape[-2:]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return value
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(value, (0, pad_w, 0, pad_h), mode=mode)


def _pad_square(value: torch.Tensor) -> torch.Tensor:
    height, width = value.shape[-2:]
    side = max(height, width)
    pad_h, pad_w = side - height, side - width
    if pad_h == 0 and pad_w == 0:
        return value
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(value, (0, pad_w, 0, pad_h), mode=mode)


def _noise(
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (1 << 64))
    return torch.randn(
        shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=device, dtype=dtype)


def _noise_batch(
    shape: tuple[int, ...],
    seed: int,
    offset: int,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack(
        [
            _noise(shape, seed + offset + index, device, dtype)
            for index in range(count)
        ]
    )


def _signature(frame_bhwc: torch.Tensor) -> torch.Tensor:
    frame = frame_bhwc.movedim(-1, 0).unsqueeze(0).float()
    return F.interpolate(frame, size=(32, 32), mode="area").cpu()


def _cache_hit(
    current: torch.Tensor,
    previous: torch.Tensor | None,
    threshold: float,
) -> bool:
    return (
        previous is not None
        and current.shape == previous.shape
        and float(F.mse_loss(current, previous)) < threshold
    )


def _latent_side(pixel_size: int) -> int:
    latent = math.ceil(max(0, int(pixel_size)) / AE_FACTOR)
    return math.ceil(latent / DIT_PATCH) * DIT_PATCH


def _tile_geometry(
    latent_h: int,
    latent_w: int,
    settings: VOSR2Settings,
) -> tuple[int, list[tuple[int, int]]]:
    side = max(DIT_PATCH, _latent_side(settings.tile_size))
    side = min(side, latent_h, latent_w)
    overlap = min(max(0, math.ceil(settings.tile_overlap / AE_FACTOR)), side - 1)
    heights = _make_tile_grid(latent_h, side, overlap)
    widths = _make_tile_grid(latent_w, side, overlap)
    return side, [(top, left) for top in heights for left in widths]


def _dit_tile_batch(settings: VOSR2Settings, frame_batch: int) -> int:
    profile_limit = (
        SPEED_DIT_TILE_BATCH
        if settings.quality_profile == "speed"
        else BALANCED_DIT_TILE_BATCH
    )
    return min(profile_limit, max(1, 8 // max(1, int(frame_batch))))


def _effective_vae_tile(
    height: int,
    width: int,
    settings: VOSR2Settings,
    auto_expand: bool,
) -> int:
    tile = int(settings.vae_tile_size)
    if auto_expand and tile > 0 and settings.quality_profile == "speed":
        tile = max(tile, min(height, width))
    return tile


def _next_vae_tile(current: int) -> int | None:
    if current <= 0:
        return SAFE_VAE_TILE
    if current <= MIN_VAE_TILE:
        return None
    reduced = min(SAFE_VAE_TILE, current // 2)
    reduced = max(MIN_VAE_TILE, (reduced // 64) * 64)
    return reduced if reduced < current else None


def _after_oom() -> None:
    gc.collect()
    comfy.model_management.soft_empty_cache()


def _encode_with_retry(
    model,
    image_pm1: torch.Tensor,
    tile: int,
    overlap: int,
):
    current = tile
    while True:
        try:
            return model.encode(image_pm1, current, min(overlap, max(0, current - 1))), current
        except torch.cuda.OutOfMemoryError:
            following = _next_vae_tile(current)
            if following is None:
                raise
            logging.warning(
                "[TE-Speed-VOSR2] VAE encode tile %s ran out of memory; retrying at %s",
                current,
                following,
            )
            _after_oom()
            current = following


def _decode_with_retry(
    model,
    latent: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    tile: int,
    overlap: int,
) -> torch.Tensor:
    current = tile
    while True:
        try:
            return model.decode(
                latent,
                mean,
                std,
                current,
                min(overlap, max(0, current - 1)),
            )
        except torch.cuda.OutOfMemoryError:
            following = _next_vae_tile(current)
            if following is None:
                raise
            logging.warning(
                "[TE-Speed-VOSR2] VAE decode tile %s ran out of memory; retrying at %s",
                current,
                following,
            )
            _after_oom()
            current = following


def _extract_dino_maps(
    model,
    padded: torch.Tensor,
    source_bhwc: torch.Tensor,
    locations: list[tuple[int, int]],
    tile_latent: int,
    dino_batch: int,
    cache_enabled: bool,
    cache: dict[str, Any],
) -> list[dict[tuple[int, int], list[torch.Tensor]]]:
    """Extract or reuse CPU-resident DINO features for every frame/tile."""
    batch = padded.shape[0]
    signatures = [_signature(source_bhwc[index]) for index in range(batch)]
    frame_sources: list[int | None] = []
    roots: list[int] = []

    previous_signature = cache.get("signature")
    previous_features = cache.get("features")
    age = int(cache.get("age", 0))
    previous_source: int | None = None

    for index, signature in enumerate(signatures):
        can_reuse = (
            cache_enabled
            and age < int(cache.get("refresh", 1))
            and _cache_hit(
                signature,
                previous_signature,
                float(cache.get("threshold", 0.0)),
            )
        )
        if can_reuse and (previous_source is not None or previous_features is not None):
            frame_sources.append(previous_source)
            age += 1
        else:
            frame_sources.append(index)
            roots.append(index)
            previous_source = index
            previous_features = None
            age = 0
        previous_signature = signature

    feature_maps: dict[int, dict[tuple[int, int], list[torch.Tensor]]] = {}
    external_features = cache.get("features")
    jobs = [(frame, location) for frame in roots for location in locations]

    start = 0
    active_batch = max(1, dino_batch)
    while start < len(jobs):
        group = jobs[start : start + active_batch]
        packed = None
        try:
            crops = []
            for frame, (top, left) in group:
                y0, x0 = top * AE_FACTOR, left * AE_FACTOR
                y1, x1 = (top + tile_latent) * AE_FACTOR, (left + tile_latent) * AE_FACTOR
                crops.append(padded[frame : frame + 1, :, y0:y1, x0:x1])
            packed = torch.cat(crops, dim=0)
            packed_features = model.dino_features(packed)
        except torch.cuda.OutOfMemoryError:
            if len(group) == 1:
                raise
            del packed
            active_batch = max(1, len(group) // 2)
            logging.warning(
                "[TE-Speed-VOSR2] DINO batch %s ran out of memory; retrying at %s",
                len(group),
                active_batch,
            )
            _after_oom()
            continue
        for item, (frame, location) in enumerate(group):
            feature_maps.setdefault(frame, {})[location] = [
                layer[item : item + 1].detach().cpu() for layer in packed_features
            ]
        del packed, packed_features
        start += len(group)
        comfy.model_management.throw_exception_if_processing_interrupted()

    resolved: list[dict[tuple[int, int], list[torch.Tensor]]] = []
    latest_map = external_features
    for index, source in enumerate(frame_sources):
        if source is None:
            if latest_map is None:
                raise RuntimeError("VOSR2 temporal cache has no reusable features")
            current_map = latest_map
        else:
            current_map = feature_maps[source]
            latest_map = current_map
        resolved.append(current_map)

    if cache_enabled:
        cache["signature"] = signatures[-1]
        cache["features"] = resolved[-1]
        cache["age"] = age
    else:
        cache.clear()
    return resolved


def _run_full_frame(
    model,
    resized: torch.Tensor,
    source_bhwc: torch.Tensor,
    seed: int,
    frame_offset: int,
    settings: VOSR2Settings,
    vae_tile: int,
    cache_enabled: bool,
    cache: dict[str, Any],
) -> torch.Tensor:
    batch, _, height, width = resized.shape
    padded = _pad_square(_pad_multiple(resized))
    encoded, vae_tile = _encode_with_retry(
        model,
        padded * 2.0 - 1.0,
        vae_tile,
        settings.vae_tile_overlap,
    )
    latent, mean, std = encoded
    locations = [(0, 0)]
    features = _extract_dino_maps(
        model,
        padded,
        source_bhwc,
        locations,
        min(latent.shape[-2:]),
        settings.dino_batch,
        cache_enabled,
        cache,
    )
    joined_features = [
        torch.cat([features[index][(0, 0)][layer] for index in range(batch)])
        for layer in range(len(features[0][(0, 0)]))
    ]
    noise = _noise_batch(
        tuple(latent.shape[1:]),
        seed,
        frame_offset,
        batch,
        latent.device,
        latent.dtype,
    )
    restored = model.one_step(latent, noise, joined_features)
    restored_cpu = restored.detach().cpu()
    del restored, noise, joined_features, features, latent, padded
    model.prepare_vae_decode()
    decoded = _decode_with_retry(
        model,
        restored_cpu,
        mean,
        std,
        vae_tile,
        settings.vae_tile_overlap,
    )
    return decoded[:, :, :height, :width]


def _run_tiled(
    model,
    resized: torch.Tensor,
    source_bhwc: torch.Tensor,
    seed: int,
    frame_offset: int,
    settings: VOSR2Settings,
    vae_tile: int,
    cache_enabled: bool,
    cache: dict[str, Any],
    progress: comfy.utils.ProgressBar,
) -> torch.Tensor:
    batch, _, height, width = resized.shape
    padded = _pad_multiple(resized)
    encoded, vae_tile = _encode_with_retry(
        model,
        padded * 2.0 - 1.0,
        vae_tile,
        settings.vae_tile_overlap,
    )
    latent, mean, std = encoded
    _, channels, latent_h, latent_w = latent.shape
    side, locations = _tile_geometry(latent_h, latent_w, settings)
    features = _extract_dino_maps(
        model,
        padded,
        source_bhwc,
        locations,
        side,
        settings.dino_batch,
        cache_enabled,
        cache,
    )
    noise = _noise_batch(
        tuple(latent.shape[1:]),
        seed,
        frame_offset,
        batch,
        latent.device,
        latent.dtype,
    )
    velocity_sum = torch.zeros_like(latent)
    weight_sum = torch.zeros_like(latent)
    weight = _gaussian_weights(side, side, channels, latent.device).to(latent.dtype)
    jobs = [(frame, top, left) for frame in range(batch) for top, left in locations]
    tile_batch = _dit_tile_batch(settings, batch)

    start = 0
    while start < len(jobs):
        group = jobs[start : start + tile_batch]
        low = z = feature_layers = None
        try:
            low = torch.cat(
                [
                    latent[frame : frame + 1, :, top : top + side, left : left + side]
                    for frame, top, left in group
                ]
            )
            z = torch.cat(
                [
                    noise[frame : frame + 1, :, top : top + side, left : left + side]
                    for frame, top, left in group
                ]
            )
            feature_layers = [
                torch.cat(
                    [
                        features[frame][(top, left)][layer]
                        for frame, top, left in group
                    ]
                )
                for layer in range(len(features[0][locations[0]]))
            ]
            velocity = model.velocity(torch.cat((low, z), dim=1), 1.0, 0.0, feature_layers)
        except torch.cuda.OutOfMemoryError:
            if len(group) == 1:
                raise
            del low, z, feature_layers
            tile_batch = max(1, len(group) // 2)
            logging.warning(
                "[TE-Speed-VOSR2] DiT tile batch %s ran out of memory; "
                "retrying at %s",
                len(group),
                tile_batch,
            )
            _after_oom()
            continue

        for item, (frame, top, left) in enumerate(group):
            velocity_sum[
                frame : frame + 1,
                :,
                top : top + side,
                left : left + side,
            ] += velocity[item : item + 1] * weight
            weight_sum[
                frame : frame + 1,
                :,
                top : top + side,
                left : left + side,
            ] += weight
        progress.update_absolute(min(progress.total, progress.current + len(group)))
        del low, z, feature_layers, velocity
        start += len(group)
        comfy.model_management.throw_exception_if_processing_interrupted()

    restored = noise - velocity_sum / weight_sum
    restored_cpu = restored.detach().cpu()
    del restored, noise, velocity_sum, weight_sum, weight, features, latent, padded
    model.prepare_vae_decode()
    decoded = _decode_with_retry(
        model,
        restored_cpu,
        mean,
        std,
        vae_tile,
        settings.vae_tile_overlap,
    )
    return decoded[:, :, :height, :width]


def _align_to_cpu(
    decoded_pm1: torch.Tensor,
    reference: torch.Tensor,
    settings: VOSR2Settings,
) -> list[torch.Tensor]:
    outputs = []
    decoded01 = ((decoded_pm1.float().clamp_(-1.0, 1.0) + 1.0) / 2.0).cpu()
    reference = reference.float().cpu()
    for index in range(decoded01.shape[0]):
        outputs.append(
            apply_color_alignment(
                decoded01[index : index + 1],
                reference[index : index + 1],
                settings.color_alignment,
                downsample=SPEED_COLOR_DOWNSAMPLE,
            ).movedim(1, -1)
        )
    return outputs


@torch.inference_mode()
def run_vosr2(
    model,
    images: torch.Tensor,
    scale: int,
    seed: int,
    settings: VOSR2Settings,
    batch_override: int | None = None,
    temporal_cache: bool = False,
    auto_expand_vae_tile: bool = False,
) -> tuple[torch.Tensor]:
    """Upscale a ComfyUI BHWC IMAGE batch and return a one-item result tuple."""
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("TE-Speed VOSR2 expects an IMAGE tensor with shape (B,H,W,C)")
    if images.shape[-1] != 3:
        raise ValueError(
            f"TE-Speed VOSR2 expects 3-channel RGB images, got {images.shape[-1]}"
        )
    scale = int(scale)
    if scale < 1:
        raise ValueError(f"TE-Speed VOSR2 scale must be at least 1, got {scale}")
    if images.shape[0] == 0:
        raise ValueError("TE-Speed VOSR2 received an empty IMAGE batch")

    settings = settings.normalized()
    if settings.color_alignment not in {"wavelet", "adain", "none"}:
        raise ValueError(f"Unknown color alignment: {settings.color_alignment!r}")

    batch_size = max(
        1,
        int(batch_override if batch_override is not None else settings.image_batch),
    )
    cache_enabled = bool(temporal_cache and settings.temporal_cache)
    cache: dict[str, Any] = {
        "threshold": settings.cache_threshold,
        "refresh": settings.cache_refresh,
    }

    source_cpu = images.detach().float().cpu()
    input_h, input_w = source_cpu.shape[1:3]
    target_h, target_w = input_h * scale, input_w * scale
    use_tiling = settings.tile_strategy == "tiled" or (
        settings.tile_strategy == "auto"
        and (target_h > settings.tile_size or target_w > settings.tile_size)
    )
    if settings.tile_strategy == "full_frame" and max(target_h, target_w) > 512:
        logging.warning(
            "[TE-Speed-VOSR2] full_frame target %sx%s exceeds the native 512px "
            "training size; tiled or auto is recommended",
            target_w,
            target_h,
        )

    if use_tiling:
        padded_h = target_h + (-target_h) % PAD_MULTIPLE
        padded_w = target_w + (-target_w) % PAD_MULTIPLE
        latent_h, latent_w = padded_h // AE_FACTOR, padded_w // AE_FACTOR
        _, locations = _tile_geometry(latent_h, latent_w, settings)
        progress = comfy.utils.ProgressBar(len(locations) * source_cpu.shape[0])
    else:
        progress = comfy.utils.ProgressBar(source_cpu.shape[0])

    original_policy = model.memory_policy
    if settings.memory_policy != "auto":
        model.set_memory_policy(settings.memory_policy)

    outputs: list[torch.Tensor] = []
    try:
        for start in range(0, source_cpu.shape[0], batch_size):
            end = min(source_cpu.shape[0], start + batch_size)
            chunk = source_cpu[start:end]
            nchw = chunk.movedim(-1, 1).to(model.device)
            resized = F.interpolate(
                nchw,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
            ).clamp_(0.0, 1.0)
            vae_tile = _effective_vae_tile(
                target_h,
                target_w,
                settings,
                auto_expand_vae_tile,
            )
            if use_tiling:
                decoded = _run_tiled(
                    model,
                    resized,
                    chunk,
                    int(seed),
                    start,
                    settings,
                    vae_tile,
                    cache_enabled,
                    cache,
                    progress,
                )
            else:
                decoded = _run_full_frame(
                    model,
                    resized,
                    chunk,
                    int(seed),
                    start,
                    settings,
                    vae_tile,
                    cache_enabled,
                    cache,
                )
                progress.update_absolute(end)
            outputs.extend(_align_to_cpu(decoded, resized, settings))
            del decoded, resized, nchw
            model.clear_staged()
            comfy.model_management.throw_exception_if_processing_interrupted()
    finally:
        model.clear_staged()
        if model.memory_policy != original_policy:
            model.set_memory_policy(original_policy)

    return (torch.cat(outputs, dim=0),)
