"""Native VIDEO in, enhanced VIDEO out, streamed one frame at a time.

`MyVideoEnhanceStream` decodes the source with FFmpeg, runs the same shared
frame pipeline as `MyVideoEnhance`, encodes every finished frame immediately and
finally remuxes the source audio. RAM therefore stays at one frame per stage: no
clip-sized IMAGE batch or frame list is ever built here, and a two-stage run only
adds the disk-backed float32 spool between its stages.

Importing this module imports neither torch, ComfyUI, the FFmpeg layer
(`my_nodes.core.video_enhance.video_io`) nor the GIMM plugin. They are loaded
when a run actually reaches them, so the lightweight unit test environment can
still import the registry.
"""

from __future__ import annotations

import os
import uuid
from fractions import Fraction
from pathlib import Path

from my_nodes.core.video_enhance.dlss_stage import (
    comfy_interrupt,
    default_memory_hooks,
    resolve_runtime_dir,
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
from my_nodes.core.video_enhance.plan import (
    NR_PROFILES,
    STAGE_ORDER_VFI_THEN_DLSS,
    STAGE_ORDERS,
    VideoEnhancePlan,
)
from my_nodes.nodes.video_enhance import (
    CHANNEL_ORDERS,
    NODE_CATEGORY,
    SPATIAL_LABELS,
    STAGE_ORDER_TOOLTIP,
    VFI_PRECISIONS,
    _check_choice,
    _plan,
    _progress_bar,
    _status,
)

try:
    from comfy_api.latest import io
except ImportError:  # pragma: no cover - unit tests run outside ComfyUI
    io = None

OUTPUT_CODECS: tuple[str, ...] = ("libx264", "h264_nvenc")
DEFAULT_OUTPUT_CODEC: str = OUTPUT_CODECS[0]
# CRF for libx264 and CQ for h264_nvenc; both encoders use the same 0..51 scale.
QUALITY_RANGE: tuple[int, int] = (0, 51)
DEFAULT_QUALITY: int = 18
# Both temporary files live in Comfy's temp directory and are unique per run.
TEMP_PREFIX = "my_nodes_enhance_stream_"
# The float32 spool between two active stages; the core pipeline deletes the file
# it creates there on every exit path.
FRAME_STORE_DIRECTORY = "my_nodes_stream_frames"
PASS_THROUGH_STATUS = (
    "pass-through: no stage is enabled, the input VIDEO is returned unchanged and "
    "nothing is probed, decoded or encoded"
)
CODEC_TOOLTIP = (
    "Encoder for the enhanced video. libx264 is the CPU default; h264_nvenc is the "
    "NVENC preset and needs a supported GPU."
)
DESCRIPTION = (
    "Streams a native local CFR VIDEO through the same DLSS and GIMM-VFI stages as "
    "My Video Enhance, one frame at a time, and returns a new file-backed VIDEO with "
    "the source audio remuxed. Memory stays at one frame per stage, so a long clip "
    "needs no RAM-resident IMAGE batch. Stage order defaults to vfi_then_dlss. v1 "
    "requires a local seekable CFR SDR file and processes all of it: variable frame "
    "rate, live sources and active trim windows are rejected. 1.0 is native DLAA, "
    "not an upscale; neural-rendering profiles are local UX presets, not NVIDIA "
    "official presets; frame interpolation is offline GIMM-VFI, not DLSS Frame "
    "Generation."
)


def _video_io():
    """Import the FFmpeg I/O layer on first use (stdlib plus NumPy, no torch)."""
    from my_nodes.core.video_enhance import video_io

    return video_io


def _video_from_file(path: str):
    """Wrap the encoded file as ComfyUI's native file-backed VIDEO."""
    from comfy_api.latest import InputImpl

    return InputImpl.VideoFromFile(path)


def _crf(value) -> int:
    """Validate the encoder quality knob before any file or process is created."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"quality must be an int, got {type(value).__name__}")
    if not QUALITY_RANGE[0] <= value <= QUALITY_RANGE[1]:
        raise ValueError(f"quality must be within {QUALITY_RANGE}, got {value}")
    return value


def _fps_multiplier(plan: VideoEnhancePlan, final_frame_count: int) -> int:
    """The fps factor this run really applies; `_status` uses the same rule.

    A single source frame has no pair to interpolate, so the VFI stage stays at
    one frame and must not claim a doubled frame rate either.
    """
    if plan.uses_frame_interpolation and final_frame_count > 1:
        return plan.interpolation_factor
    return 1


def _local_video_path(video) -> Path:
    """The local file behind a native VIDEO, or an actionable error.

    `get_stream_source()` is the file path for a file-backed VIDEO. Component or
    in-memory videos answer with a BytesIO instead, which would have to be
    encoded in RAM first and cannot be streamed.
    """
    getter = getattr(video, "get_stream_source", None)
    if getter is None:
        raise ValueError(
            "MyVideoEnhanceStream needs a native file-backed VIDEO input, got "
            f"{type(video).__name__}"
        )
    source = getter()
    if not isinstance(source, (str, os.PathLike)):
        raise ValueError(
            "MyVideoEnhanceStream streams a local video file, but this VIDEO has no file "
            f"behind it (get_stream_source returned {type(source).__name__}). Save it to "
            "a file first, for example with a video output node."
        )
    return Path(os.fspath(source))


def _reject_trim_window(video) -> None:
    """v1 enhances whole files, so an active trim window would be ignored."""
    window = getattr(video, "get_active_trim_window", None)
    start, duration = window() if window is not None else (0.0, None)
    if not start and not duration:
        return
    raise ValueError(
        f"this VIDEO carries an active trim window (start={start!r}, duration={duration!r}) "
        "and MyVideoEnhanceStream v1 processes the whole file, so the trim would be "
        "silently dropped. Trim it before this node, or use My Video Enhance for the "
        "trimmed range."
    )


def _temp_output_path(directory: str | os.PathLike[str], kind: str) -> Path:
    """A unique not-yet-existing `.mkv` path this run owns in `directory`."""
    return Path(directory) / f"{TEMP_PREFIX}{kind}_{uuid.uuid4().hex}.mkv"


def _remove_quietly(path) -> None:
    """Delete one temporary file this node owns; a missing file is not an error."""
    try:
        os.unlink(path)
    except OSError:
        pass


class MyVideoEnhanceStream:
    """VIDEO in, enhanced VIDEO plus fps multiplier and a short status string out."""

    NODE_ID = "MyVideoEnhanceStream"
    DISPLAY_NAME = "My Video Enhance Stream"
    DESCRIPTION = DESCRIPTION

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "Local CFR video file to enhance."}),
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
                "output_codec": (list(OUTPUT_CODECS), {
                    "default": DEFAULT_OUTPUT_CODEC, "tooltip": CODEC_TOOLTIP,
                }),
                "quality": ("INT", {
                    "default": DEFAULT_QUALITY,
                    "min": QUALITY_RANGE[0],
                    "max": QUALITY_RANGE[1],
                    "step": 1,
                    "tooltip": "CRF for libx264 and CQ for h264_nvenc; lower is better and larger.",
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
                    "default": STAGE_ORDER_VFI_THEN_DLSS, "advanced": True,
                    "tooltip": STAGE_ORDER_TOOLTIP,
                }),
            },
        }

    RETURN_TYPES = ("VIDEO", "INT", "STRING")
    RETURN_NAMES = ("video", "interpolation_multiplier", "status")
    FUNCTION = "enhance"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = DESCRIPTION

    def enhance(
        self,
        video,
        enable_super_resolution,
        spatial_mode,
        enable_neural_rendering,
        nr_profile,
        nr_intensity,
        enable_frame_interpolation,
        output_codec=DEFAULT_OUTPUT_CODEC,
        quality=DEFAULT_QUALITY,
        vfi_precision="fp32",
        vfi_ds_factor=1.0,
        motion=MOTION_OPTICAL_FLOW,
        scene_cut_threshold=0.2,
        channel_order="auto",
        runtime_dir="",
        wine_prefix="",
        worker_timeout=600.0,
        stage_order=STAGE_ORDER_VFI_THEN_DLSS,
    ):
        plan = _plan(
            enable_super_resolution,
            spatial_mode,
            enable_neural_rendering,
            nr_profile,
            nr_intensity,
            enable_frame_interpolation,
            stage_order,
        )
        _check_choice(str(vfi_precision), VFI_PRECISIONS, "vfi_precision")
        _check_choice(str(motion), MOTION_MODES, "motion")
        _check_choice(str(channel_order), CHANNEL_ORDERS, "channel_order")
        codec = _check_choice(str(output_codec), OUTPUT_CODECS, "output_codec")
        crf = _crf(quality)
        if plan.is_pass_through:
            return (video, 1, PASS_THROUGH_STATUS)

        _reject_trim_window(video)
        source_path = _local_video_path(video)

        import folder_paths

        video_io = _video_io()
        temp_directory = folder_paths.get_temp_directory()
        spec = video_io.probe_cfr_video(source_path, interrupt=comfy_interrupt)
        source_spec = FrameSpec(count=spec.frame_count, height=spec.height, width=spec.width)
        specs = pipeline_specs(source_spec, plan)
        multiplier = _fps_multiplier(plan, specs.final.count)
        output_fps = Fraction(spec.fps) * multiplier
        # Comfy's temp root is not guaranteed to exist yet (a fresh install or a
        # cleaned temp directory), and every output below is written into it.
        # Nothing is owned before this point, so a failure here leaves no residue.
        os.makedirs(temp_directory, exist_ok=True)
        video_only_path = _temp_output_path(temp_directory, "video_only")
        output_path = _temp_output_path(temp_directory, "enhanced")
        frame_store_directory = os.path.join(temp_directory, FRAME_STORE_DIRECTORY)

        bar = _progress_bar(pipeline_step_total(source_spec, plan))

        def progress(done: int, _total: int) -> None:
            bar.update_absolute(done)

        # Everything this node has to delete again if the run fails. The encoded
        # result joins the list only once the remux has handed it over.
        owned_paths: list[Path] = [video_only_path]
        try:
            with video_io.FFmpegFrameReader(spec, interrupt=comfy_interrupt) as reader:
                with video_io.FFmpegFrameWriter(
                    video_only_path,
                    width=specs.final.width,
                    height=specs.final.height,
                    fps=output_fps,
                    expected_frames=specs.final.count,
                    codec=codec,
                    quality=crf,
                    interrupt=comfy_interrupt,
                ) as writer:
                    def write_frame(_index: int, frame) -> None:
                        """The pipeline owns the frame order; the encoder takes the frame."""
                        writer.write(frame)

                    result = run_frame_pipeline(
                        reader,
                        source_spec,
                        plan,
                        write_frame,
                        # Only a two-stage run needs the disk-backed spool; a single
                        # stage keeps the whole run at one frame of RAM.
                        temp_directory=frame_store_directory if specs.staged else None,
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
            encoded_path = Path(
                video_io.remux_audio(spec, video_only_path, output_path, interrupt=comfy_interrupt)
            )
            # The encoded file is this node's result from here on: a later failure
            # has to delete it again instead of leaving it in the temp directory.
            owned_paths.append(encoded_path)
            output = _video_from_file(str(encoded_path))
        except BaseException:
            for owned in owned_paths:
                _remove_quietly(owned)
            raise
        # The remux consumed the video-only file; delete whatever is left of it.
        _remove_quietly(video_only_path)

        # The status text reuses the IMAGE node's wording; its own multiplier is
        # the one `_fps_multiplier` already applied above.
        _, plan_status = _status(plan, result.frame_count, result.channel_order)
        audio = "source audio remuxed" if spec.has_audio else "no source audio"
        status = (
            f"{plan_status}; output_fps={output_fps}; codec={codec} quality={crf}; {audio}"
        )
        return (output, multiplier, status)

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
                    io.Video.Input("video", tooltip="Local CFR video file to enhance."),
                    io.Boolean.Input("enable_super_resolution", default=False, tooltip="DLSS feature 1. 1.0 is native DLAA."),
                    io.Combo.Input("spatial_mode", options=list(SPATIAL_LABELS), default="2.0x"),
                    io.Boolean.Input("enable_neural_rendering", default=False, tooltip="Experimental DLSS feature 18."),
                    io.Combo.Input("nr_profile", options=list(NR_PROFILES), default="standard", tooltip="Local UX profile, not an NVIDIA preset."),
                    io.Float.Input("nr_intensity", default=1.0, min=0.0, max=2.0, step=0.05),
                    io.Boolean.Input("enable_frame_interpolation", default=False, tooltip="Offline GIMM-VFI 2x, not DLSS Frame Generation."),
                    io.Combo.Input("output_codec", options=list(OUTPUT_CODECS), default=DEFAULT_OUTPUT_CODEC, tooltip=CODEC_TOOLTIP),
                    io.Int.Input("quality", default=DEFAULT_QUALITY, min=QUALITY_RANGE[0], max=QUALITY_RANGE[1], step=1),
                    io.Combo.Input("vfi_precision", options=list(VFI_PRECISIONS), default="fp32", advanced=True),
                    io.Float.Input("vfi_ds_factor", default=1.0, min=0.01, max=1.0, step=0.01, advanced=True),
                    io.Combo.Input("motion", options=list(MOTION_MODES), default=MOTION_OPTICAL_FLOW, advanced=True),
                    io.Float.Input("scene_cut_threshold", default=0.2, min=0.0, max=1.0, step=0.01, advanced=True),
                    io.Combo.Input("channel_order", options=list(CHANNEL_ORDERS), default="auto", advanced=True),
                    io.String.Input("runtime_dir", default="", advanced=True),
                    io.String.Input("wine_prefix", default="", advanced=True),
                    io.Float.Input("worker_timeout", default=600.0, min=1.0, max=86400.0, step=1.0, advanced=True),
                    io.Combo.Input("stage_order", options=list(STAGE_ORDERS), default=STAGE_ORDER_VFI_THEN_DLSS, advanced=True, tooltip=STAGE_ORDER_TOOLTIP),
                ],
                outputs=[
                    io.Video.Output(display_name="video"),
                    io.Int.Output(display_name="interpolation_multiplier"),
                    io.String.Output(display_name="status"),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            video, multiplier, status = cls().enhance(**kwargs)
            return io.NodeOutput(video, multiplier, status)
