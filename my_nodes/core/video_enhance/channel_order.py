"""Channel-order selection for the DNR3 worker's raw RGBA readback.

The native bridge returns texture channels as stored. Some DLSS builds store
RGBA and some store BGRA. `auto` compares the first output against the source
and then applies that one order to the whole batch.
"""

from __future__ import annotations

import numpy as np

CHANNEL_ORDERS: tuple[str, ...] = ("auto", "RGBA", "BGRA")


def swap_rb(frame: np.ndarray) -> np.ndarray:
    """Return a copy with red and blue exchanged."""
    swapped = np.empty_like(frame)
    swapped[..., 0] = frame[..., 2]
    swapped[..., 1] = frame[..., 1]
    swapped[..., 2] = frame[..., 0]
    return swapped


def _downsampled(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = (np.arange(height) * frame.shape[0]) // height
    cols = (np.arange(width) * frame.shape[1]) // width
    return frame[np.ix_(rows, cols)]


def select_channel_order(order: str, source: np.ndarray, output: np.ndarray) -> str:
    """Resolve `auto` from the first pair, or validate an explicit order."""
    if order not in CHANNEL_ORDERS:
        raise ValueError(f"channel order must be one of {CHANNEL_ORDERS}, got {order!r}")
    if order != "auto":
        return order
    height = min(source.shape[0], output.shape[0])
    width = min(source.shape[1], output.shape[1])
    reference = _downsampled(source, height, width)
    candidate = _downsampled(output, height, width)
    direct = float(np.mean(np.abs(candidate - reference)))
    swapped = float(np.mean(np.abs(swap_rb(candidate) - reference)))
    return "BGRA" if swapped + 1e-6 < direct else "RGBA"


def apply_channel_order(frame: np.ndarray, order: str) -> np.ndarray:
    """Apply a resolved (non-auto) order. RGBA is returned unchanged."""
    if order == "RGBA":
        return frame
    if order == "BGRA":
        return swap_rb(frame)
    raise ValueError(f"channel order must already be resolved, got {order!r}")
