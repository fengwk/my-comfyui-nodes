"""ComfyUI nodes for the combined video-enhance pipeline.

Importing this module does not import torch, ComfyUI or the GIMM plugin. Those
are loaded only when an enabled stage actually runs, so the lightweight unit
test environment can still import the registry.
"""

from __future__ import annotations

import numpy as np

from my_nodes.core.video_enhance.dlss_stage import (
    comfy_interrupt,
    default_memory_hooks,
    prepare_frames,
    probe_frame,
    resolve_runtime_dir,
    run_dlss_stage,
)
from my_nodes.core.video_enhance.gimm_vfi import interpolate_offline
from my_nodes.core.video_enhance.motion import MOTION_MODES, MOTION_OPTICAL_FLOW
from my_nodes.core.video_enhance.nr_profiles import neural_rendering_settings
from my_nodes.core.video_enhance.plan import (
    NR_PROFILES,
    VideoEnhancePlan,
)

try:
    from comfy_api.latest import io
except ImportError:  # pragma: no cover - unit tests run outside ComfyUI
    io = None

NODE_CATEGORY = "image/video"
SPATIAL_MODES: tuple[tuple[str, float], ...] = (
    ("1.0 DLAA (native)", 1.0),
    ("1.5x", 1.5),
    ("2.0x", 2.0),
    ("3.0x", 3.0),
)
SPATIAL_LABELS: tuple[str, ...] = tuple(label for label, _scale in SPATIAL_MODES)
_SCALE_BY_LABEL = dict(SPATIAL_MODES)
CHANNEL_ORDERS = ("auto", "RGBA", "BGRA")
VFI_PRECISIONS = ("fp32", "fp16", "bf16")

DESCRIPTION = (
    "DLSS feature 1 (DLAA at 1.0, or super resolution above it), then optional "
    "feature 18 neural rendering in the same worker, then optional offline "
    "GIMM-VFI 2x after that worker exits. Disabled stages are not touched. "
    "1.0 is native DLAA, not an upscale. Neural-rendering profiles are local "
    "UX presets, not NVIDIA official presets. Frame interpolation is offline "
    "GIMM-VFI, not DLSS Frame Generation."
)


def spatial_scale(label: str) -> float:
    try:
        return _SCALE_BY_LABEL[label]
    except KeyError:
        raise ValueError(
            f"spatial mode must be one of {SPATIAL_LABELS}, got {label!r}"
        ) from None


def _plan(enable_sr: bool, spatial_mode: str, enable_nr: bool, nr_profile: str, nr_intensity: float, enable_vfi: bool) -> VideoEnhancePlan:
    return VideoEnhancePlan(
        enable_super_resolution=bool(enable_sr),
        sr_scale=spatial_scale(spatial_mode),
        enable_neural_rendering=bool(enable_nr),
        nr_profile=str(nr_profile),
        nr_intensity=float(nr_intensity),
        enable_frame_interpolation=bool(enable_vfi),
        interpolation_factor=2,
    )


def _check_choice(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


def _progress_bar(total: int):
    from comfy.utils import ProgressBar

    return ProgressBar(total)


def _batch_from_numpy(frames):
    import torch

    return torch.from_numpy(np.ascontiguousarray(frames))


def _status(plan: VideoEnhancePlan, frame_count: int, channel_order: str | None) -> tuple[int, str]:
    multiplier = 2 if plan.uses_frame_interpolation and frame_count > 1 else 1
    parts = ["pass-through" if plan.is_pass_through else "+".join(plan.stages)]
    if plan.enable_super_resolution:
        label = "DLAA 1.0x" if plan.sr_scale == 1.0 else f"SR {plan.sr_scale:.1f}x"
        parts.append(label)
    if plan.enable_neural_rendering:
        settings = neural_rendering_settings(plan.nr_profile, plan.nr_intensity)
        parts.append(
            f"NR {settings.profile} intensity={settings.intensity:.2f} "
            f"style={settings.style} preset={settings.preset}"
        )
    if channel_order is not None:
        parts.append(f"channels={channel_order}")
    parts.append(f"frames={frame_count}")
    parts.append(f"fps_multiplier={multiplier}")
    if multiplier == 2:
        parts.append("encode at input FPS x2")
    return multiplier, "; ".join(parts)


class MyVideoEnhance:
    """IMAGE in, IMAGE plus fps multiplier and a short status string out."""

    NODE_ID = "MyVideoEnhance"
    DISPLAY_NAME = "My Video Enhance"
    DESCRIPTION = DESCRIPTION

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Source frames as an IMAGE batch [N,H,W,3]."}),
                "enable_super_resolution": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "DLSS feature 1. 1.0 is native DLAA, not an upscale.",
                }),
                "spatial_mode": (list(SPATIAL_LABELS), {"default": "2.0x"}),
                "enable_neural_rendering": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "DLSS feature 18 after feature 1, in the same worker. Experimental.",
                }),
                "nr_profile": (list(NR_PROFILES), {
                    "default": "standard",
                    "tooltip": "Local UX profile, not an NVIDIA official preset.",
                }),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "enable_frame_interpolation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offline GIMM-VFI 2x after the DLSS worker exits. Not DLSS Frame Generation.",
                }),
            },
            "optional": {
                "vfi_precision": (list(VFI_PRECISIONS), {"default": "fp32", "advanced": True}),
                "vfi_ds_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "advanced": True,
                }),
                "motion": (list(MOTION_MODES), {"default": MOTION_OPTICAL_FLOW, "advanced": True}),
                "scene_cut_threshold": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True,
                    "tooltip": "Mean absolute grayscale difference that resets DLSS history. 0 disables the cut check.",
                }),
                "channel_order": (list(CHANNEL_ORDERS), {"default": "auto", "advanced": True}),
                "runtime_dir": ("STRING", {
                    "default": "", "advanced": True,
                    "tooltip": "NVIDIA runtime directory. Empty uses DLSS5_RUNTIME_DIR, then models/dlss5.",
                }),
                "wine_prefix": ("STRING", {
                    "default": "", "advanced": True,
                    "tooltip": "Wine prefix. Empty uses DLSS5_WINEPREFIX, then WINEPREFIX, then ~/.wine.",
                }),
                "worker_timeout": ("FLOAT", {
                    "default": 600.0, "min": 1.0, "max": 86400.0, "step": 1.0, "advanced": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "fps_multiplier", "status")
    FUNCTION = "enhance"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = DESCRIPTION

    def enhance(
        self,
        images,
        enable_super_resolution,
        spatial_mode,
        enable_neural_rendering,
        nr_profile,
        nr_intensity,
        enable_frame_interpolation,
        vfi_precision="fp32",
        vfi_ds_factor=1.0,
        motion=MOTION_OPTICAL_FLOW,
        scene_cut_threshold=0.2,
        channel_order="auto",
        runtime_dir="",
        wine_prefix="",
        worker_timeout=600.0,
    ):
        plan = _plan(
            enable_super_resolution,
            spatial_mode,
            enable_neural_rendering,
            nr_profile,
            nr_intensity,
            enable_frame_interpolation,
        )
        _check_choice(str(vfi_precision), VFI_PRECISIONS, "vfi_precision")
        _check_choice(str(motion), MOTION_MODES, "motion")
        _check_choice(str(channel_order), CHANNEL_ORDERS, "channel_order")
        if plan.is_pass_through:
            count = int(images.shape[0]) if hasattr(images, "shape") else len(images)
            _multiplier, status = _status(plan, count, None)
            return (images, 1, status)

        frames = prepare_frames(images) if plan.uses_dlss else _numpy_without_backend(images)
        resolved_order = None
        source_count = int(frames.shape[0])
        progress_state = {"bar": None, "completed": 0}

        def progress(done: int, _total: int) -> None:
            # `done` restarts at 1 for VFI, so add the frames the DLSS stage already reported.
            progress_state["bar"].update_absolute(progress_state["completed"] + done)

        if plan.uses_dlss:
            # VFI interpolates the DLSS output, whose count is still the source count.
            vfi_steps = max(0, source_count - 1) if plan.uses_frame_interpolation else 0
            progress_state["bar"] = self._progress_bar(source_count + vfi_steps)
            result = run_dlss_stage(
                plan,
                frames,
                runtime_dir=resolve_runtime_dir(str(runtime_dir)),
                wine_prefix=str(wine_prefix),
                channel_order=str(channel_order),
                motion_mode=str(motion),
                scene_cut_threshold=float(scene_cut_threshold),
                timeout=float(worker_timeout),
                progress=progress,
                interrupt=comfy_interrupt,
                memory_hooks=default_memory_hooks(),
            )
            frames = result.frames
            resolved_order = result.channel_order
            progress_state["completed"] = source_count
        if plan.uses_frame_interpolation:
            import folder_paths

            if progress_state["bar"] is None:
                progress_state["bar"] = self._progress_bar(max(0, int(frames.shape[0]) - 1))
            frames = interpolate_offline(
                frames,
                precision=str(vfi_precision),
                ds_factor=float(vfi_ds_factor),
                models_dir=folder_paths.models_dir,
                progress=progress,
                interrupt=comfy_interrupt,
            )
        multiplier, status = _status(plan, int(frames.shape[0]), resolved_order)
        return (_batch_from_numpy(frames), multiplier, status)

    @staticmethod
    def _progress_bar(total: int):
        return _progress_bar(total)

    if io is not None:
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=cls.NODE_ID,
                display_name=cls.DISPLAY_NAME,
                category=NODE_CATEGORY,
                description=cls.DESCRIPTION,
                is_experimental=True,
                inputs=[
                    io.Image.Input("images", tooltip="Source frames as an IMAGE batch [N,H,W,3]."),
                    io.Boolean.Input("enable_super_resolution", default=False, tooltip="DLSS feature 1. 1.0 is native DLAA."),
                    io.Combo.Input("spatial_mode", options=list(SPATIAL_LABELS), default="2.0x"),
                    io.Boolean.Input("enable_neural_rendering", default=False, tooltip="Experimental DLSS feature 18."),
                    io.Combo.Input("nr_profile", options=list(NR_PROFILES), default="standard", tooltip="Local UX profile, not an NVIDIA preset."),
                    io.Float.Input("nr_intensity", default=1.0, min=0.0, max=2.0, step=0.05),
                    io.Boolean.Input("enable_frame_interpolation", default=False, tooltip="Offline GIMM-VFI 2x, not DLSS Frame Generation."),
                    io.Combo.Input("vfi_precision", options=list(VFI_PRECISIONS), default="fp32", advanced=True),
                    io.Float.Input("vfi_ds_factor", default=1.0, min=0.01, max=1.0, step=0.01, advanced=True),
                    io.Combo.Input("motion", options=list(MOTION_MODES), default=MOTION_OPTICAL_FLOW, advanced=True),
                    io.Float.Input("scene_cut_threshold", default=0.2, min=0.0, max=1.0, step=0.01, advanced=True),
                    io.Combo.Input("channel_order", options=list(CHANNEL_ORDERS), default="auto", advanced=True),
                    io.String.Input("runtime_dir", default="", advanced=True),
                    io.String.Input("wine_prefix", default="", advanced=True),
                    io.Float.Input("worker_timeout", default=600.0, min=1.0, max=86400.0, step=1.0, advanced=True),
                ],
                outputs=[
                    io.Image.Output(display_name="images"),
                    io.Int.Output(display_name="fps_multiplier"),
                    io.String.Output(display_name="status"),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            images, multiplier, status = cls().enhance(**kwargs)
            return io.NodeOutput(images, multiplier, status)


class MyDLSSRuntimeProbe:
    """Run one small deterministic frame through the real DNR3 pipeline."""

    NODE_ID = "MyDLSSRuntimeProbe"
    DISPLAY_NAME = "My DLSS Runtime Probe"
    DESCRIPTION = (
        "Runs one 32x32 deterministic RGB frame through the same DNR3 worker as "
        "My Video Enhance. At least DLAA/SR or neural rendering must be enabled. "
        "Missing user DLLs or Wine prefix files fail with the path that is missing."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_super_resolution": ("BOOLEAN", {"default": True}),
                "spatial_mode": (list(SPATIAL_LABELS), {"default": "2.0x"}),
                "enable_neural_rendering": ("BOOLEAN", {"default": False}),
                "nr_profile": (list(NR_PROFILES), {"default": "standard"}),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "runtime_dir": ("STRING", {"default": "", "advanced": True}),
                "wine_prefix": ("STRING", {"default": "", "advanced": True}),
                "worker_timeout": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 86400.0, "step": 1.0, "advanced": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "probe"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = DESCRIPTION

    def probe(
        self,
        enable_super_resolution,
        spatial_mode,
        enable_neural_rendering,
        nr_profile,
        nr_intensity,
        runtime_dir="",
        wine_prefix="",
        worker_timeout=120.0,
    ):
        plan = _plan(
            enable_super_resolution,
            spatial_mode,
            enable_neural_rendering,
            nr_profile,
            nr_intensity,
            False,
        )
        if not plan.uses_dlss:
            raise ValueError("MyDLSSRuntimeProbe requires super resolution (DLAA/SR) or neural rendering")
        result = run_dlss_stage(
            plan,
            probe_frame(),
            runtime_dir=resolve_runtime_dir(str(runtime_dir)),
            wine_prefix=str(wine_prefix),
            channel_order="RGBA",
            motion_mode="none",
            scene_cut_threshold=0.2,
            timeout=float(worker_timeout),
            interrupt=comfy_interrupt,
            memory_hooks=default_memory_hooks(),
        )
        label = "DLAA 1.0x" if plan.sr_scale == 1.0 and plan.enable_super_resolution else (
            f"SR {plan.sr_scale:.1f}x" if plan.enable_super_resolution else "native"
        )
        nr = f" + NR {plan.nr_profile}" if plan.enable_neural_rendering else ""
        return (
            f"probe ok: {label}{nr}; features={result.features:#x}; "
            f"output={result.output_width}x{result.output_height}; channels={result.channel_order}",
        )

    if io is not None:
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=cls.NODE_ID,
                display_name=cls.DISPLAY_NAME,
                category=NODE_CATEGORY,
                description=cls.DESCRIPTION,
                is_experimental=True,
                inputs=[
                    io.Boolean.Input("enable_super_resolution", default=True),
                    io.Combo.Input("spatial_mode", options=list(SPATIAL_LABELS), default="2.0x"),
                    io.Boolean.Input("enable_neural_rendering", default=False),
                    io.Combo.Input("nr_profile", options=list(NR_PROFILES), default="standard"),
                    io.Float.Input("nr_intensity", default=1.0, min=0.0, max=2.0, step=0.05),
                    io.String.Input("runtime_dir", default="", advanced=True),
                    io.String.Input("wine_prefix", default="", advanced=True),
                    io.Float.Input("worker_timeout", default=120.0, min=1.0, max=86400.0, step=1.0, advanced=True),
                ],
                outputs=[io.String.Output(display_name="status")],
            )

        @classmethod
        def execute(cls, **kwargs):
            status, = cls().probe(**kwargs)
            return io.NodeOutput(status)


def _numpy_without_backend(images):
    """CPU float32 copy for the VFI-only path. Does not start DLSS or GIMM."""
    if hasattr(images, "detach"):
        batch = images.detach().cpu().numpy()
    else:
        batch = np.asarray(images)
    if batch.ndim != 4 or batch.shape[-1] != 3 or batch.shape[0] < 1:
        raise ValueError(f"expected IMAGE batch [N,H,W,3], got shape {getattr(batch, 'shape', None)}")
    frames = np.ascontiguousarray(batch, dtype=np.float32)
    if not np.isfinite(frames).all():
        raise ValueError("IMAGE batch contains non-finite values")
    return frames
