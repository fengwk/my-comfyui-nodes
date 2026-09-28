"""Aspect-preserving long-edge resize with one-frame interpolation buffers."""

from __future__ import annotations


def resize_long_edge(
    images, long_edge=640, interpolation="bicubic", check_interrupt=None,
    only_downscale=True,
):
    import torch
    import torch.nn.functional as F

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("Expected IMAGE tensor [N,H,W,C]")
    if not images.is_floating_point() or any(size <= 0 for size in images.shape):
        raise ValueError("IMAGE must be a non-empty floating-point tensor")
    if isinstance(long_edge, bool) or int(long_edge) != long_edge or long_edge < 1:
        raise ValueError("long_edge must be a positive integer")
    if interpolation not in {"bicubic", "bilinear", "nearest", "area"}:
        raise ValueError(f"Unsupported interpolation: {interpolation}")

    count, height, width, channels = images.shape
    long_edge = int(long_edge)
    if max(height, width) == long_edge or (
        only_downscale and max(height, width) < long_edge
    ):
        return images
    if width >= height:
        out_w = long_edge
        out_h = max(1, round(height * long_edge / width))
    else:
        out_h = long_edge
        out_w = max(1, round(width * long_edge / height))

    # Preallocate the IMAGE batch instead of retaining frames and concatenating.
    with torch.inference_mode():
        result = images.new_empty((count, out_h, out_w, channels))
        options = {}
        if interpolation in {"bicubic", "bilinear"}:
            options = {"align_corners": False, "antialias": True}
        for index in range(count):
            if check_interrupt is not None:
                check_interrupt()
            frame = images[index:index + 1].movedim(-1, 1).float()
            resized = F.interpolate(
                frame, size=(out_h, out_w), mode=interpolation, **options
            )
            result[index:index + 1].copy_(resized.clamp_(0, 1).movedim(1, -1))
    return result
