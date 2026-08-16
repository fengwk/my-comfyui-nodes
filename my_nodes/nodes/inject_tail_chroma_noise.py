"""ComfyUI node: inject tapered chroma blocks into the tail of an IMAGE batch."""

from __future__ import annotations

import numpy as np

from my_nodes.core.inject_tail_chroma_noise import inject_tail_chroma_noise

try:
    from comfy_api.latest import io
except ImportError:  # pragma: no cover - unit tests run outside ComfyUI
    io = None


def images_to_numpy_list(images) -> list[np.ndarray]:
    """Convert a Comfy IMAGE batch (`[N,H,W,C]` tensor/array) to numpy frames."""
    if images is None:
        raise ValueError("images is required")
    if hasattr(images, "detach"):
        batch = images.detach().cpu().numpy()
    else:
        batch = np.asarray(images)
    if batch.ndim != 4:
        raise ValueError(f"expected IMAGE batch [N,H,W,C], got shape {batch.shape}")
    return [batch[i] for i in range(batch.shape[0])]


def numpy_list_to_image_batch(frames: list[np.ndarray], like=None):
    """Stack uint8 RGB frames back into a float IMAGE batch in [0, 1]."""
    stacked = np.stack(frames, axis=0).astype(np.float32) / 255.0
    if like is not None and hasattr(like, "new_tensor"):
        return like.new_tensor(stacked)
    try:
        import torch

        return torch.from_numpy(stacked)
    except ImportError:
        return stacked


class InjectTailChromaNoise:
    """Blend T3 chroma-block noise into the last N frames of an IMAGE batch.

    Wire this after `GetVideoComponents` and before `CreateVideo` /
    `MiniMaxH3ChainExternalVideo`. Do not recurse the already-injected output
    into the next chain's clean context extraction.
    """

    NODE_ID = "MiniMaxH3InjectTailNoise"
    DISPLAY_NAME = "MiniMax H3 Inject Tail Noise"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames as a Comfy IMAGE batch [N,H,W,C]. "
                               "Use GetVideoComponents.images, not a Python list.",
                }),
                "tail_frames": ("INT", {
                    "default": 22,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "How many trailing frames to inject. Validated recipe: 22.",
                }),
                "alpha": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend strength for the early tail frames.",
                }),
                "alpha_end": ("FLOAT", {
                    "default": 0.10,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend strength of the final tail frame. Must be <= alpha.",
                }),
                "ramp_frames": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "How many last tail frames linearly taper to alpha_end.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Deterministic purple-green block-noise seed.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "inject"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Inject deterministic purple-green color blocks into the last tail_frames. "
        "Early tail uses alpha; the final ramp_frames taper to alpha_end. "
        "Use only as a temporary Motion Context pixel input (context_frames)."
    )

    def inject(self, images, tail_frames, alpha, alpha_end, ramp_frames, seed):
        frames = images_to_numpy_list(images)
        injected = inject_tail_chroma_noise(
            frames,
            int(tail_frames),
            float(alpha),
            float(alpha_end),
            int(ramp_frames),
            int(seed),
        )
        return (numpy_list_to_image_batch(injected, like=images),)

    if io is not None:
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=cls.NODE_ID,
                display_name=cls.DISPLAY_NAME,
                category="conditioning/minimax",
                description=cls.DESCRIPTION,
                inputs=[
                    io.Image.Input(
                        "images",
                        tooltip="Video frames as a Comfy IMAGE batch. Use GetVideoComponents.",
                    ),
                    io.Int.Input("tail_frames", default=22, min=1, max=4096),
                    io.Float.Input("alpha", default=0.45, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("alpha_end", default=0.10, min=0.0, max=1.0, step=0.01),
                    io.Int.Input("ramp_frames", default=3, min=1, max=4096),
                    io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                ],
                outputs=[io.Image.Output(display_name="images")],
            )

        @classmethod
        def execute(cls, images, tail_frames, alpha, alpha_end, ramp_frames, seed):
            result = cls().inject(images, tail_frames, alpha, alpha_end, ramp_frames, seed)
            return io.NodeOutput(result[0])
