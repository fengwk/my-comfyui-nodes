"""ComfyUI wrappers for the Linux-native TE-Speed VOSR2 backend."""

from __future__ import annotations

from dataclasses import replace

from my_nodes.core.vosr2.settings import VOSR2Settings

NODE_CATEGORY = "TE-Speed/VOSR2"
MODEL_TYPE = "TE_SPEED_VOSR2_MODEL"
SETTINGS_TYPE = "TE_SPEED_VOSR2_SETTINGS"


class TESpeedVOSR2Loader:
    NODE_ID = "TESpeedVOSR2Loader"
    DISPLAY_NAME = "TE-Speed VOSR2 Loader"

    @classmethod
    def INPUT_TYPES(cls):
        try:
            from my_nodes.core.vosr2.model_store import (
                DEFAULT_BUNDLE,
                bundle_names,
            )

            bundles = bundle_names()
        except ImportError:
            DEFAULT_BUNDLE = "VOSR2"
            bundles = [DEFAULT_BUNDLE]
        return {
            "required": {
                "model_bundle": (bundles, {"default": DEFAULT_BUNDLE}),
                "precision": (
                    ["auto", "fp16", "bf16"],
                    {"default": "auto"},
                ),
                "memory_policy": (
                    ["auto", "resident", "staged"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "torch_compile": ("BOOLEAN", {"default": False}),
                "vae_encode_amp": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = NODE_CATEGORY

    def load(
        self,
        model_bundle,
        precision,
        memory_policy,
        torch_compile=False,
        vae_encode_amp=False,
    ):
        from my_nodes.core.vosr2.model_store import _bundle_path, load_model

        model = load_model(model_bundle, precision)
        model.set_memory_policy(memory_policy)
        model.vae_encode_amp = bool(vae_encode_amp)
        model.set_torch_compile(bool(torch_compile), _bundle_path(model_bundle))
        return (model,)


class TESpeedVOSR2Settings:
    NODE_ID = "TESpeedVOSR2Settings"
    DISPLAY_NAME = "TE-Speed VOSR2 Settings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quality_profile": (
                    ["manual", "speed"],
                    {"default": "speed"},
                ),
                "tile_strategy": (
                    ["auto", "full_frame", "tiled"],
                    {"default": "auto"},
                ),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 128, "max": 4096, "step": 64},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 32, "min": 0, "max": 512, "step": 8},
                ),
                "vae_tile_size": (
                    "INT",
                    {"default": 1024, "min": 0, "max": 8192, "step": 64},
                ),
                "vae_tile_overlap": (
                    "INT",
                    {"default": 32, "min": 0, "max": 512, "step": 8},
                ),
                "image_batch": (
                    "INT",
                    {"default": 1, "min": 1, "max": 32},
                ),
                "frame_batch": (
                    "INT",
                    {"default": 2, "min": 1, "max": 8},
                ),
                "dino_batch": (
                    "INT",
                    {"default": 2, "min": 1, "max": 8},
                ),
                "temporal_cache": ("BOOLEAN", {"default": False}),
                "cache_threshold": (
                    "FLOAT",
                    {
                        "default": 0.003,
                        "min": 0.0,
                        "max": 0.25,
                        "step": 0.0005,
                    },
                ),
                "cache_refresh": (
                    "INT",
                    {"default": 4, "min": 1, "max": 32},
                ),
                "memory_policy": (
                    ["auto", "resident", "staged"],
                    {"default": "auto"},
                ),
                "color_alignment": (
                    ["wavelet", "adain", "none"],
                    {"default": "wavelet"},
                ),
            }
        }

    RETURN_TYPES = (SETTINGS_TYPE,)
    RETURN_NAMES = ("settings",)
    FUNCTION = "make"
    CATEGORY = NODE_CATEGORY

    def make(self, **kwargs):
        return (VOSR2Settings(**kwargs).normalized(),)


class _VOSR2Base:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "images": ("IMAGE",),
                "scale": (
                    "INT",
                    {"default": 2, "min": 1, "max": 16},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            },
            "optional": {
                "settings": (SETTINGS_TYPE,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = NODE_CATEGORY

    def upscale(self, model, images, scale, seed, settings=None):
        from my_nodes.core.vosr2.inference import run_vosr2

        effective = (settings or VOSR2Settings()).normalized()
        return run_vosr2(model, images, scale, seed, effective)


class TESpeedVOSR2Image(_VOSR2Base):
    NODE_ID = "TESpeedVOSR2Image"
    DISPLAY_NAME = "TE-Speed VOSR2 Image"


class TESpeedVOSR2Video(_VOSR2Base):
    NODE_ID = "TESpeedVOSR2Video"
    DISPLAY_NAME = "TE-Speed VOSR2 Video Frames"

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["optional"].update(
            {
                "frame_batch": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8},
                ),
                "temporal_cache": ("BOOLEAN", {"default": False}),
                "cache_threshold": (
                    "FLOAT",
                    {
                        "default": 0.003,
                        "min": 0.0,
                        "max": 0.25,
                        "step": 0.0005,
                    },
                ),
                "cache_refresh": (
                    "INT",
                    {"default": 4, "min": 1, "max": 32},
                ),
            }
        )
        return result

    def upscale(
        self,
        model,
        images,
        scale,
        seed,
        settings=None,
        frame_batch=0,
        temporal_cache=False,
        cache_threshold=0.003,
        cache_refresh=4,
    ):
        from my_nodes.core.vosr2.inference import run_vosr2

        defaults = VOSR2Settings()
        effective = (settings or defaults).normalized()
        effective_batch = int(frame_batch) or effective.frame_batch
        # An unchanged Video widget inherits the linked Settings value; with
        # boolean widgets, disabling a linked cache requires changing Settings.
        inherit = settings is not None
        effective = replace(
            effective,
            frame_batch=effective_batch,
            temporal_cache=(
                effective.temporal_cache
                if inherit and temporal_cache == defaults.temporal_cache
                else bool(temporal_cache)
            ),
            cache_threshold=(
                effective.cache_threshold
                if inherit and cache_threshold == defaults.cache_threshold
                else float(cache_threshold)
            ),
            cache_refresh=(
                effective.cache_refresh
                if inherit and cache_refresh == defaults.cache_refresh
                else int(cache_refresh)
            ),
        ).normalized()
        return run_vosr2(
            model,
            images,
            scale,
            seed,
            effective,
            batch_override=effective_batch,
            temporal_cache=effective.temporal_cache,
            auto_expand_vae_tile=True,
        )


VOSR2_NODE_CLASSES = (
    TESpeedVOSR2Loader,
    TESpeedVOSR2Settings,
    TESpeedVOSR2Image,
    TESpeedVOSR2Video,
)

__all__ = [
    "TESpeedVOSR2Image",
    "TESpeedVOSR2Loader",
    "TESpeedVOSR2Settings",
    "TESpeedVOSR2Video",
    "VOSR2_NODE_CLASSES",
]
