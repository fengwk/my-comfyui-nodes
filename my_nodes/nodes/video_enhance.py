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
from my_nodes.core.video_enhance.frame_pipeline import (
    DlssStageOptions,
    FrameSpec,
    VfiStageOptions,
    pipeline_specs,
    pipeline_step_total,
    run_frame_pipeline,
)
from my_nodes.core.video_enhance.motion import MOTION_MODES, MOTION_OPTICAL_FLOW
from my_nodes.core.video_enhance.nr_profiles import plan_neural_rendering_settings
from my_nodes.core.video_enhance.plan import (
    NR_PRESETS,
    NR_PROFILES,
    NR_STYLES,
    SR_PRESETS,
    STAGE_ORDER_DLSS_THEN_VFI,
    STAGE_ORDERS,
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

# The final IMAGE batch has to be RAM-resident, so refuse one that would claim
# more than this share of the RAM that is available right now.
OUTPUT_RAM_FRACTION = 0.8

STAGE_ORDER_TOOLTIP = (
    "Order of the two stages when both are enabled: dlss_then_vfi is the legacy "
    "default, vfi_then_dlss interpolates first and enhances the interpolated frames."
)

CUSTOM_PROFILE_ONLY = "Used when nr_profile=custom."

# The advanced DLSS controls both nodes append, in this order, after
# `stage_order`. Every model field is read only by the `custom` profile; the
# post-NR composite controls and the SR preset apply whenever their feature runs.
ADVANCED_OPTIONAL: dict[str, tuple] = {
    "style": (list(NR_STYLES), {
        "default": "Cinematic", "advanced": True,
        "tooltip": f"DLSSNR.Style, the neural-rendering model style. {CUSTOM_PROFILE_ONLY}",
    }),
    "preset": (list(NR_PRESETS), {
        "default": "Default", "advanced": True,
        "tooltip": f"DLSSNR.Hint.Render.Preset, the neural-rendering model preset. {CUSTOM_PROFILE_ONLY}",
    }),
    "local_structure": ("FLOAT", {
        "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "advanced": True,
        "tooltip": f"DLSSNR.LocalStructureStrength. {CUSTOM_PROFILE_ONLY}",
    }),
    "local_tone": ("FLOAT", {
        "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "advanced": True,
        "tooltip": f"DLSSNR.LocalToneStrength. {CUSTOM_PROFILE_ONLY}",
    }),
    "skin": ("FLOAT", {
        "default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05, "advanced": True,
        "tooltip": (
            "DLSSNR.SkinStructureStrength; -1 leaves the model default in place. "
            f"{CUSTOM_PROFILE_ONLY}"
        ),
    }),
    "detail": ("FLOAT", {
        "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "advanced": True,
        "tooltip": (
            "Post-NR composite detail. Applies whenever neural rendering is enabled, "
            "not only with nr_profile=custom; 1.0 keeps the raw model output."
        ),
    }),
    "color": ("FLOAT", {
        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True,
        "tooltip": (
            "Post-NR composite color. Applies whenever neural rendering is enabled, "
            "not only with nr_profile=custom; 1.0 keeps the raw model output."
        ),
    }),
    "ui_correction": ("BOOLEAN", {
        "default": False, "advanced": True,
        "tooltip": f"DLSSNR.UICorrection, the UI/high-frequency correction pass. {CUSTOM_PROFILE_ONLY}",
    }),
    "auto_mask": ("BOOLEAN", {
        "default": False, "advanced": True,
        "tooltip": f"DLSSNR.UseAutoMask. {CUSTOM_PROFILE_ONLY}",
    }),
    "sr_preset": (list(SR_PRESETS), {
        "default": "Default", "advanced": True,
        "tooltip": (
            "DLSS super-resolution model preset; Default keeps the runtime default. "
            "Applies whenever super resolution runs."
        ),
    }),
    "gpu_index": ("INT", {
        "default": 0, "min": 0, "max": 15, "step": 1, "advanced": True,
        "tooltip": "Adapter index the DLSS worker uses; 0 is the first visible GPU.",
    }),
}

DESCRIPTION = (
    "DLSS feature 1 (DLAA at 1.0, or super resolution above it), then optional "
    "feature 18 neural rendering in the same worker, then optional offline "
    "GIMM-VFI 2x after that worker exits; stage_order can interpolate first and "
    "enhance the interpolated frames instead. Disabled stages are not touched. "
    "A two-stage run stages the intermediate frames on disk, not in RAM. "
    "1.0 is native DLAA, not an upscale. Neural-rendering profiles are local "
    "UX presets, not NVIDIA official presets; the advanced model fields are read "
    "by nr_profile=custom, while detail and color are post-NR composite controls. "
    "Frame interpolation is offline GIMM-VFI, not DLSS Frame Generation."
)


class InsufficientRamError(RuntimeError):
    """The final IMAGE batch cannot be allocated without exhausting RAM."""


def spatial_scale(label: str) -> float:
    try:
        return _SCALE_BY_LABEL[label]
    except KeyError:
        raise ValueError(
            f"spatial mode must be one of {SPATIAL_LABELS}, got {label!r}"
        ) from None


def _plan(
    enable_sr: bool,
    spatial_mode: str,
    enable_nr: bool,
    nr_profile: str,
    nr_intensity: float,
    enable_vfi: bool,
    stage_order: str = STAGE_ORDER_DLSS_THEN_VFI,
    *,
    style: str = "Cinematic",
    preset: str = "Default",
    local_structure: float = 1.0,
    local_tone: float = 1.0,
    skin: float = -1.0,
    detail: float = 1.0,
    color: float = 1.0,
    ui_correction: bool = False,
    auto_mask: bool = False,
    sr_preset: str = "Default",
    gpu_index: int = 0,
) -> VideoEnhancePlan:
    """Build the plan from the node settings; unknown values fail here, before a backend.

    The widget names differ from the plan field names only where a plan field is
    neural-rendering specific (`nr_style`, `nr_local_tone`, `nr_ui_correction`).
    """
    return VideoEnhancePlan(
        enable_super_resolution=bool(enable_sr),
        sr_scale=spatial_scale(spatial_mode),
        enable_neural_rendering=bool(enable_nr),
        nr_profile=str(nr_profile),
        nr_intensity=float(nr_intensity),
        nr_style=str(style),
        nr_preset=str(preset),
        nr_local_structure=local_structure,
        nr_local_tone=local_tone,
        nr_skin=skin,
        nr_detail=detail,
        nr_color=color,
        nr_ui_correction=ui_correction,
        nr_auto_mask=auto_mask,
        sr_preset=str(sr_preset),
        gpu_index=gpu_index,
        enable_frame_interpolation=bool(enable_vfi),
        interpolation_factor=2,
        stage_order=stage_order,
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


def require_output_ram(required_bytes: int) -> None:
    """Reject a final IMAGE batch that would not fit safely in the free RAM.

    The returned IMAGE must be one resident float32 batch, so its exact size is
    known before the first frame is produced. `MyVideoEnhanceStream` keeps the
    frames on disk instead of one batch and is the answer when this fails.
    """
    import psutil

    try:
        available = int(psutil.virtual_memory().available)
    except (psutil.Error, OSError) as error:
        raise InsufficientRamError(
            f"cannot read the available RAM to size a {required_bytes} byte IMAGE output: {error}"
        ) from error
    if required_bytes > OUTPUT_RAM_FRACTION * available:
        raise InsufficientRamError(
            f"the final IMAGE needs {required_bytes} bytes as float32 but only {available} bytes "
            f"of RAM are available, which is more than {OUTPUT_RAM_FRACTION:.0%} of the free RAM. "
            "Lower the resolution or frame count, or use MyVideoEnhanceStream, which keeps the "
            "frames on disk instead of one resident IMAGE batch."
        )


def _status(plan: VideoEnhancePlan, frame_count: int, channel_order: str | None) -> tuple[int, str]:
    multiplier = 2 if plan.uses_frame_interpolation and frame_count > 1 else 1
    parts = ["pass-through" if plan.is_pass_through else "+".join(plan.stages)]
    if plan.enable_super_resolution:
        label = "DLAA 1.0x" if plan.sr_scale == 1.0 else f"SR {plan.sr_scale:.1f}x"
        parts.append(label)
    if plan.enable_neural_rendering:
        settings = plan_neural_rendering_settings(plan)
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


def _advanced_inputs() -> list:
    """The v3 `io` inputs of `ADVANCED_OPTIONAL`, derived from the classic widgets.

    Both schemas therefore expose the same names, choices, defaults, ranges and
    tooltips in the same order; only the widget representation differs.
    """
    kinds = {"FLOAT": io.Float.Input, "INT": io.Int.Input, "BOOLEAN": io.Boolean.Input}
    inputs: list = []
    for name, (kind, options) in ADVANCED_OPTIONAL.items():
        if isinstance(kind, list):
            inputs.append(io.Combo.Input(name, options=list(kind), **options))
        else:
            inputs.append(kinds[kind](name, **options))
    return inputs


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
                    "tooltip": (
                        "Local UX profile, not an NVIDIA official preset. custom "
                        "reads the advanced model fields instead."
                    ),
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
                "stage_order": (list(STAGE_ORDERS), {
                    "default": STAGE_ORDER_DLSS_THEN_VFI, "advanced": True,
                    "tooltip": STAGE_ORDER_TOOLTIP,
                }),
                # Keep every pre-existing widget at its serialized array index.
                # ComfyUI restores workflows positionally unless its experimental
                # named-value restore setting is enabled.
                **ADVANCED_OPTIONAL,
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
        stage_order=STAGE_ORDER_DLSS_THEN_VFI,
        style="Cinematic",
        preset="Default",
        local_structure=1.0,
        local_tone=1.0,
        skin=-1.0,
        detail=1.0,
        color=1.0,
        ui_correction=False,
        auto_mask=False,
        sr_preset="Default",
        gpu_index=0,
    ):
        plan = _plan(
            enable_super_resolution,
            spatial_mode,
            enable_neural_rendering,
            nr_profile,
            nr_intensity,
            enable_frame_interpolation,
            stage_order,
            style=style,
            preset=preset,
            local_structure=local_structure,
            local_tone=local_tone,
            skin=skin,
            detail=detail,
            color=color,
            ui_correction=ui_correction,
            auto_mask=auto_mask,
            sr_preset=sr_preset,
            gpu_index=gpu_index,
        )
        _check_choice(str(vfi_precision), VFI_PRECISIONS, "vfi_precision")
        _check_choice(str(motion), MOTION_MODES, "motion")
        _check_choice(str(channel_order), CHANNEL_ORDERS, "channel_order")
        if plan.is_pass_through:
            count = int(images.shape[0]) if hasattr(images, "shape") else len(images)
            _multiplier, status = _status(plan, count, None)
            return (images, 1, status)

        frames = prepare_frames(images)
        source_spec = FrameSpec(
            count=int(frames.shape[0]), height=int(frames.shape[1]), width=int(frames.shape[2])
        )
        specs = pipeline_specs(source_spec, plan)
        # Every check that can reject the run happens before ComfyUI's paths and
        # the output allocation are touched, so a refusal leaves nothing behind.
        require_output_ram(specs.final.nbytes)
        # Only the final IMAGE is held in RAM; an intermediate goes to disk.
        output = np.empty(specs.final.shape, dtype=np.float32)

        import folder_paths

        def write_frame(index: int, frame) -> None:
            output[index] = frame

        bar = self._progress_bar(pipeline_step_total(source_spec, plan))

        def progress(done: int, _total: int) -> None:
            bar.update_absolute(done)

        result = run_frame_pipeline(
            frames,
            source_spec,
            plan,
            write_frame,
            temp_directory=folder_paths.get_temp_directory() if specs.staged else None,
            progress=progress,
            interrupt=comfy_interrupt,
            vfi=VfiStageOptions(
                precision=str(vfi_precision),
                models_dir=folder_paths.models_dir,
                ds_factor=float(vfi_ds_factor),
            ),
            dlss=DlssStageOptions(
                runtime_dir=resolve_runtime_dir(str(runtime_dir)),
                motion_mode=str(motion),
                scene_cut_threshold=float(scene_cut_threshold),
                channel_order=str(channel_order),
                wine_prefix=str(wine_prefix),
                timeout=float(worker_timeout),
                memory_hooks=default_memory_hooks(),
            ),
        )
        multiplier, status = _status(plan, result.frame_count, result.channel_order)
        return (_batch_from_numpy(output), multiplier, status)

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
                    io.Combo.Input("nr_profile", options=list(NR_PROFILES), default="standard", tooltip="Local UX profile, not an NVIDIA preset. custom reads the advanced model fields instead."),
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
                    io.Combo.Input("stage_order", options=list(STAGE_ORDERS), default=STAGE_ORDER_DLSS_THEN_VFI, advanced=True, tooltip=STAGE_ORDER_TOOLTIP),
                    *_advanced_inputs(),
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
