"""Deterministic blocky chroma-noise injection for a video tail.

Ported from minimax-h3-chained-character-swap `scripts/inject_tail_taper.py`.
This module has no ComfyUI dependency so it can be unit-tested in isolation.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from PIL import Image

DEFAULT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (185, 115, 215),
    (115, 195, 140),
    (150, 148, 162),
    (205, 150, 192),
    (138, 182, 148),
    (160, 120, 175),
)

# Validated POC grid. At 576x1024 this produces 16x16 pixel blocks.
DEFAULT_NOISE_GRID: tuple[int, int] = (36, 64)


def alpha_for(
    position: int,
    tail_frames: int,
    alpha: float,
    alpha_end: float,
    ramp_frames: int,
) -> float:
    """Return injection alpha for a zero-based position inside the tail."""
    if not 0 <= position < tail_frames:
        raise ValueError("position must be inside the injected tail")
    from_end = tail_frames - 1 - position
    if from_end >= ramp_frames:
        return alpha
    return alpha + (alpha_end - alpha) * (ramp_frames - from_end) / ramp_frames


def validate_taper(
    tail_frames: int,
    alpha: float,
    alpha_end: float,
    ramp_frames: int,
) -> None:
    if tail_frames < 1:
        raise ValueError("tail_frames must be positive")
    if ramp_frames < 1 or ramp_frames > tail_frames:
        raise ValueError("ramp_frames must be between 1 and tail_frames")
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= alpha_end <= 1.0:
        raise ValueError("alpha and alpha_end must be between 0 and 1")
    if alpha_end > alpha:
        raise ValueError("alpha_end must not exceed alpha for a taper")


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWxC RGB(A) frame, got shape {frame.shape}")
    rgb = frame[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def inject_tail_chroma_noise(
    frames: Sequence[np.ndarray],
    tail_frames: int,
    alpha: float,
    alpha_end: float,
    ramp_frames: int,
    seed: int,
    palette: Sequence[tuple[int, int, int]] = DEFAULT_PALETTE,
    noise_grid: tuple[int, int] = DEFAULT_NOISE_GRID,
) -> list[np.ndarray]:
    """Blend deterministic chroma blocks into the last `tail_frames` images.

    `frames` is a sequence of HxWxC arrays (uint8 0-255 or float 0-1).
    Returns a new list of uint8 RGB arrays. Frames before the tail are copied
    as RGB without visual change.
    """
    validate_taper(tail_frames, alpha, alpha_end, ramp_frames)
    if not frames:
        raise ValueError("frames must not be empty")
    if tail_frames > len(frames):
        raise ValueError(
            f"tail_frames={tail_frames} exceeds frame count {len(frames)}"
        )
    if noise_grid[0] < 1 or noise_grid[1] < 1:
        raise ValueError("noise_grid dimensions must be positive")
    if not palette:
        raise ValueError("palette must not be empty")

    rng = random.Random(seed)
    start = len(frames) - tail_frames
    out: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        rgb = _as_uint8_rgb(np.asarray(frame))
        if index < start:
            out.append(rgb)
            continue
        position = index - start
        amount = alpha_for(position, tail_frames, alpha, alpha_end, ramp_frames)
        small = Image.new("RGB", noise_grid)
        pixels = small.load()
        for y in range(noise_grid[1]):
            for x in range(noise_grid[0]):
                pixels[x, y] = rng.choice(tuple(palette))
        noisy = small.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
        blended = Image.blend(Image.fromarray(rgb, "RGB"), noisy, amount)
        out.append(np.asarray(blended, dtype=np.uint8))
    return out
