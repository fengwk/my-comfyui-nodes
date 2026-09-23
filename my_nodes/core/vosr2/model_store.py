"""VOSR 2.0 bundle discovery, loading, and ComfyUI-aware memory scheduling."""

from __future__ import annotations

import gc
import json
import logging
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

from . import tiled_vae
from .models.dinov2 import build_dinov2_vitl14
from .models.lightningdit import LightningDiT
from .models.qwenimage_vae2d import AutoencoderKLQwenImage2D

DEFAULT_BUNDLE = "VOSR2"
VOSR2_FOLDER_KEY = "vosr2"
VAE_DECODE_RESERVE = 6 * 1024**3

_ROOT = Path(folder_paths.models_dir) / VOSR2_FOLDER_KEY
folder_paths.add_model_folder_path(VOSR2_FOLDER_KEY, str(_ROOT))

_VAE_SUBDIR = "Qwen-Image-vae-2d"
_VAE_FILES = ("config.json", "diffusion_pytorch_model.safetensors")
_DINO_FILES = ("dinov2_vitl14.safetensors", "dinov2_vitl14_pretrain.pth")
_DIT_CANDIDATES = (
    Path("clean_weights") / "ema_model.safetensors",
    Path("checkpoints") / "ema_model.safetensors",
    Path("ema_model.safetensors"),
)

_REQUIRED_ARGS: dict[str, Any] = {
    "ae_type": "qwen",
    "dim": 1536,
    "depth": 36,
    "num_heads": 24,
    "patch_size": 2,
    "enc_type": "dinov2l",
    "enc_dim": 1024,
    "layer_dinov2b_list": [17],
    "auxiliary_time_cond": False,
    "distill_type": "onestep",
}
_DIT_ARG_KEYS = (
    "mlp_ratio",
    "use_qknorm",
    "use_swiglu",
    "use_rope",
    "use_rmsnorm",
    "encdim_ratio",
    "resolution",
)
_STRIPPABLE_PREFIXES = ("module.", "_orig_mod.", "ema_model.")
_TRAINING_ONLY_KEYS = (
    re.compile(r"^n_averaged$"),
    re.compile(r"^step_count$"),
    re.compile(r"^decay$"),
)
_HF_DINO_LAYER = re.compile(r"^encoder\.layer\.(\d+)\.(.+)$")
_HF_DINO_DIRECT = {
    "embeddings.cls_token": "cls_token",
    "embeddings.mask_token": "mask_token",
    "embeddings.position_embeddings": "pos_embed",
    "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
    "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
    "layernorm.weight": "norm.weight",
    "layernorm.bias": "norm.bias",
}
_HF_DINO_LAYER_SUFFIXES = {
    "attention.output.dense.weight": "attn.proj.weight",
    "attention.output.dense.bias": "attn.proj.bias",
    "layer_scale1.lambda1": "ls1.gamma",
    "layer_scale2.lambda1": "ls2.gamma",
    "mlp.fc1.weight": "mlp.fc1.weight",
    "mlp.fc1.bias": "mlp.fc1.bias",
    "mlp.fc2.weight": "mlp.fc2.weight",
    "mlp.fc2.bias": "mlp.fc2.bias",
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
}


class VOSR2LoadError(RuntimeError):
    """A VOSR2 bundle is absent, malformed, or incompatible."""


def _bundle_path(name: str) -> Path:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise VOSR2LoadError(f"Invalid VOSR2 bundle name: {name!r}")
    root = _ROOT.resolve()
    candidate = (root / name).resolve()
    if candidate.parent != root:
        raise VOSR2LoadError(f"Invalid VOSR2 bundle name: {name!r}")
    return candidate


def _find_dit(bundle: Path) -> Path | None:
    return next((bundle / path for path in _DIT_CANDIDATES if (bundle / path).is_file()), None)


def _find_dino(bundle: Path) -> Path | None:
    return next((bundle / name for name in _DINO_FILES if (bundle / name).is_file()), None)


def _is_bundle(bundle: Path) -> bool:
    vae = bundle / _VAE_SUBDIR
    return (
        bundle.is_dir()
        and (bundle / "args.json").is_file()
        and _find_dit(bundle) is not None
        and _find_dino(bundle) is not None
        and all((vae / name).is_file() for name in _VAE_FILES)
    )


def bundle_names() -> list[str]:
    """Return complete local bundles; keep the standard name selectable if absent."""
    found = (
        sorted(path.name for path in _ROOT.iterdir() if _is_bundle(path))
        if _ROOT.is_dir()
        else []
    )
    return found if DEFAULT_BUNDLE in found else [DEFAULT_BUNDLE, *found]


def _load_args(bundle: Path) -> dict[str, Any]:
    path = bundle / "args.json"
    if not path.is_file():
        raise VOSR2LoadError(f"VOSR2 bundle {bundle} is missing args.json")
    try:
        args = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VOSR2LoadError(f"Could not read VOSR2 config {path}: {exc}") from exc
    for key, expected in _REQUIRED_ARGS.items():
        if args.get(key) != expected:
            raise VOSR2LoadError(
                f"Incompatible VOSR2 config {path}: expected {key}={expected!r}, "
                f"got {args.get(key)!r}"
            )
    missing = [key for key in _DIT_ARG_KEYS if key not in args]
    if missing:
        raise VOSR2LoadError(f"VOSR2 config {path} is missing fields: {missing}")
    return args


def _clean_key(key: str) -> str | None:
    cleaned = key
    for prefix in _STRIPPABLE_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    if any(pattern.match(cleaned) for pattern in _TRAINING_ONLY_KEYS):
        return None
    return cleaned


def _strict_assign(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    source: Path,
) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise VOSR2LoadError(
            f"VOSR2 checkpoint {source} does not match {type(module).__name__} "
            f"(missing={missing}, unexpected={unexpected})"
        )


def _load_safetensors_lean(
    module: torch.nn.Module,
    path: Path,
    dtype: torch.dtype,
) -> None:
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            cleaned = _clean_key(key)
            if cleaned is not None:
                state[cleaned] = checkpoint.get_tensor(key).to(dtype)
    _strict_assign(module, state, path)
    del state
    gc.collect()


def _hf_dino_direct_key(key: str) -> str | None:
    direct = _HF_DINO_DIRECT.get(key)
    if direct is not None:
        return direct
    match = _HF_DINO_LAYER.match(key)
    if match is None:
        return None
    suffix = _HF_DINO_LAYER_SUFFIXES.get(match.group(2))
    return None if suffix is None else f"blocks.{match.group(1)}.{suffix}"


def _load_hf_dino_safetensors(
    module: torch.nn.Module,
    path: Path,
    dtype: torch.dtype,
) -> None:
    """Strictly convert Hugging Face DINOv2-L keys to the local Meta layout."""
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        qkv_keys: set[str] = set()
        for key in keys:
            direct = _hf_dino_direct_key(key)
            if direct is not None:
                state[direct] = checkpoint.get_tensor(key).to(dtype)
                continue
            if ".attention.attention." in key:
                qkv_keys.add(key)
                continue
            raise VOSR2LoadError(f"Unsupported Hugging Face DINOv2 key in {path}: {key}")

        expected_qkv = {
            (
                f"encoder.layer.{layer}.attention.attention."
                f"{projection}.{parameter}"
            )
            for layer in range(24)
            for projection in ("query", "key", "value")
            for parameter in ("weight", "bias")
        }
        if qkv_keys != expected_qkv:
            raise VOSR2LoadError(
                f"Hugging Face DINOv2 checkpoint {path} has incompatible attention "
                f"keys (missing={sorted(expected_qkv - qkv_keys)}, "
                f"unexpected={sorted(qkv_keys - expected_qkv)})"
            )

        for layer in range(24):
            for parameter in ("weight", "bias"):
                parts = []
                for projection in ("query", "key", "value"):
                    key = (
                        f"encoder.layer.{layer}.attention.attention."
                        f"{projection}.{parameter}"
                    )
                    parts.append(checkpoint.get_tensor(key).to(dtype))
                state[f"blocks.{layer}.attn.qkv.{parameter}"] = torch.cat(parts)

    _strict_assign(module, state, path)
    del state
    gc.collect()


def _load_dino(
    module: torch.nn.Module,
    path: Path,
    dtype: torch.dtype,
) -> None:
    if path.suffix.lower() == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            hugging_face_layout = "embeddings.cls_token" in checkpoint.keys()
        if hugging_face_layout:
            _load_hf_dino_safetensors(module, path, dtype)
        else:
            _load_safetensors_lean(module, path, dtype)
        return
    loaded = comfy.utils.load_torch_file(str(path), safe_load=True)
    if not isinstance(loaded, dict):
        raise VOSR2LoadError(f"DINOv2 checkpoint {path} is not a state dictionary")
    state = {
        cleaned: value.to(dtype)
        for key, value in loaded.items()
        if isinstance(value, torch.Tensor)
        and (cleaned := _clean_key(str(key))) is not None
    }
    _strict_assign(module, state, path)
    del loaded, state
    gc.collect()


def _dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    if value == "auto":
        return comfy.model_management.unet_dtype(device=device)
    raise VOSR2LoadError(f"Unknown VOSR2 precision: {value!r}")


class TESpeedVOSR2Model:
    """Matched DiT/VAE/DINO bundle with explicit phase-level scheduling."""

    def __init__(
        self,
        dit_patcher,
        vae_patcher,
        dino_patcher,
        args: dict[str, Any],
        dtype_name: str,
        compute_dtype: torch.dtype,
    ):
        self.dit_patcher = dit_patcher
        self.vae_patcher = vae_patcher
        self.dino_patcher = dino_patcher
        self.args = args
        self.dtype_name = dtype_name
        self.compute_dtype = compute_dtype
        self.vae_encode_amp = False
        self.memory_policy = "auto"
        self._active_patcher = None
        self._loaded_once = False
        self._resident_ready = False
        self._vae_decode_only = False
        self._compile_requested = False
        self._compile_enabled = False
        self._compile_fallback = False
        self._eager_forward = None
        self.dino_size = int(args.get("dinov2_size", 448))
        self.dino_layer = int(args["layer_dinov2b_list"][0])

    @property
    def device(self) -> torch.device:
        return self.dit_patcher.load_device

    @property
    def _patchers(self) -> tuple:
        return (self.dit_patcher, self.vae_patcher, self.dino_patcher)

    def set_memory_policy(self, policy: str) -> None:
        if policy not in {"auto", "resident", "staged"}:
            raise ValueError(f"Unknown VOSR2 memory policy: {policy!r}")
        if policy != self.memory_policy:
            if self._loaded_once:
                for patcher in self._patchers:
                    comfy.model_management.unload_model_and_clones(patcher)
            self._active_patcher = None
            self._loaded_once = False
            self._resident_ready = False
            self._vae_decode_only = False
        self.memory_policy = policy

    def _load_stage(self, patcher):
        if self._active_patcher is patcher:
            return patcher.model, patcher.load_device
        if self._active_patcher is not None and self._active_patcher is not patcher:
            comfy.model_management.unload_model_and_clones(self._active_patcher)
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)
        self._active_patcher = patcher
        self._loaded_once = True
        return patcher.model, patcher.load_device

    def pin_resident(self) -> None:
        if not self._resident_ready:
            comfy.model_management.load_models_gpu(
                list(self._patchers),
                force_full_load=True,
            )
            self._resident_ready = True
            self._vae_decode_only = False
            self._active_patcher = None
            self._loaded_once = True

    def _load(self, patcher):
        if self.memory_policy == "resident":
            if self._vae_decode_only:
                return self._load_stage(patcher)
            else:
                self.pin_resident()
        elif self.memory_policy == "staged":
            return self._load_stage(patcher)
        else:
            comfy.model_management.load_models_gpu([patcher], force_full_load=True)
            self._loaded_once = True
        return patcher.model, patcher.load_device

    def clear_staged(self) -> None:
        if self._active_patcher is not None:
            comfy.model_management.unload_model_and_clones(self._active_patcher)
            self._active_patcher = None
            if self.memory_policy == "staged" or self._vae_decode_only:
                self._loaded_once = False
        self._vae_decode_only = False

    def prepare_vae_decode(self) -> None:
        """Make room for fp32 VAE decode without evicting unrelated models."""
        device = self.vae_patcher.load_device
        if self.memory_policy == "staged":
            self._load_stage(self.vae_patcher)
            return
        if self.memory_policy == "resident" and device.type == "cuda":
            free = comfy.model_management.get_free_memory(device)
            if free < VAE_DECODE_RESERVE:
                logging.info(
                    "[TE-Speed-VOSR2] releasing DiT/DINO for VAE decode "
                    "(free %.2f GiB, reserve %.2f GiB)",
                    free / 1024**3,
                    VAE_DECODE_RESERVE / 1024**3,
                )
                comfy.model_management.unload_model_and_clones(self.dit_patcher)
                comfy.model_management.unload_model_and_clones(self.dino_patcher)
                self._resident_ready = False
                self._vae_decode_only = True
                self._load_stage(self.vae_patcher)
                return
        self._load(self.vae_patcher)

    def set_torch_compile(self, enabled: bool, bundle_dir: Path | None = None) -> None:
        """Compile DiT lazily; restore eager execution after any runtime failure."""
        self._compile_requested = bool(enabled)
        model = self.dit_patcher.model
        if not enabled:
            if self._eager_forward is not None:
                model.forward_flexible = self._eager_forward
            self._compile_enabled = False
            self._compile_fallback = False
            return
        if self._compile_enabled:
            return
        try:
            import triton  # noqa: F401
        except ImportError:
            logging.warning(
                "[TE-Speed-VOSR2] torch_compile requested but Triton is unavailable; "
                "using eager mode"
            )
            self._compile_fallback = True
            return

        eager = self._eager_forward or model.forward_flexible
        try:
            compiled = torch.compile(eager, dynamic=False)
        except Exception as exc:
            logging.warning("[TE-Speed-VOSR2] torch.compile setup failed: %s", exc)
            self._compile_fallback = True
            return

        def compiled_with_fallback(*args, **kwargs):
            try:
                return compiled(*args, **kwargs)
            except Exception as exc:
                if not self._compile_fallback:
                    logging.warning(
                        "[TE-Speed-VOSR2] compiled DiT failed; using eager mode: %s",
                        exc,
                    )
                self._compile_fallback = True
                self._compile_enabled = False
                model.forward_flexible = eager
                return eager(*args, **kwargs)

        self._eager_forward = eager
        model.forward_flexible = compiled_with_fallback
        self._compile_enabled = True
        self._compile_fallback = False
        logging.info("[TE-Speed-VOSR2] DiT torch.compile wrapper installed")

    def _vae_autocast(self, device: torch.device):
        if not self.vae_encode_amp or device.type != "cuda":
            return nullcontext()
        # BF16 avoids fp16's narrow exponent range while retaining tensor-core
        # acceleration. Decode remains strict fp32.
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def dino_features(self, image01: torch.Tensor) -> list[torch.Tensor]:
        model, device = self._load(self.dino_patcher)
        image = torch.nn.functional.interpolate(
            image01.to(device),
            size=(self.dino_size, self.dino_size),
            mode="bicubic",
        ).clamp_(0.0, 1.0)
        dtype = model.pos_embed.dtype
        mean = image.new_tensor((0.485, 0.456, 0.406), dtype=dtype).view(
            1, 3, 1, 1
        )
        std = image.new_tensor((0.229, 0.224, 0.225), dtype=dtype).view(
            1, 3, 1, 1
        )
        normalized = (image.to(dtype) - mean) / std
        return [model.forward_intermediate_layer(normalized, self.dino_layer)]

    def encode(
        self,
        image_pm1: torch.Tensor,
        tile_size: int,
        overlap: int,
    ):
        model, device = self._load(self.vae_patcher)
        image = image_pm1.to(device=device, dtype=torch.float32)
        with self._vae_autocast(device):
            return tiled_vae.encode_dispatch(model, image, tile_size, overlap)

    def velocity(
        self,
        inp: torch.Tensor,
        t_cur: float,
        t_next: float,
        features: list[torch.Tensor],
    ) -> torch.Tensor:
        model, device = self._load(self.dit_patcher)
        dtype = model.t_embedder.mlp[0].weight.dtype
        inp = inp.to(device=device, dtype=dtype)
        features = [feature.to(device=device, dtype=dtype) for feature in features]
        batch = inp.shape[0]
        current = torch.full((batch,), t_cur, device=device, dtype=dtype)
        following = torch.full((batch,), t_next, device=device, dtype=dtype)
        return model.forward_flexible(inp, current, following, features)

    def one_step(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        features: list[torch.Tensor],
    ) -> torch.Tensor:
        device = self.dit_patcher.load_device
        z = noise.to(device)
        velocity = self.velocity(
            torch.cat((latent.to(device), z), dim=1),
            1.0,
            0.0,
            features,
        )
        return z - velocity

    def decode(
        self,
        latent: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        tile_size: int,
        overlap: int,
    ) -> torch.Tensor:
        model, device = self._load(self.vae_patcher)
        # Decode is intentionally strict fp32 even if a caller has an outer
        # autocast context. Only VAE encode may opt into BF16 autocast.
        with torch.autocast(device_type=device.type, enabled=False):
            return tiled_vae.decode_dispatch(
                model,
                latent.to(device=device, dtype=torch.float32),
                mean.to(device=device, dtype=torch.float32),
                std.to(device=device, dtype=torch.float32),
                tile_size,
                overlap,
            )


def load_model(name: str, dtype_name: str) -> TESpeedVOSR2Model:
    """Load one complete local VOSR2 bundle without network access."""
    bundle = _bundle_path(name)
    if not bundle.is_dir():
        raise VOSR2LoadError(
            f"VOSR2 bundle not found: {bundle}. Place the official bundle under "
            f"{_ROOT}/<name>/"
        )
    args = _load_args(bundle)
    dit_path = _find_dit(bundle)
    if dit_path is None:
        raise VOSR2LoadError(f"Missing VOSR2 DiT weight under {bundle}")
    dino_path = _find_dino(bundle)
    if dino_path is None:
        raise VOSR2LoadError(
            f"Missing DINOv2 weight under {bundle}; expected one of {_DINO_FILES}"
        )
    vae_dir = bundle / _VAE_SUBDIR
    missing_vae = [name for name in _VAE_FILES if not (vae_dir / name).is_file()]
    if missing_vae:
        raise VOSR2LoadError(f"VOSR2 bundle {bundle} is missing VAE files: {missing_vae}")

    load_device = comfy.model_management.get_torch_device()
    # Staged scheduling is a node-level promise independent of ComfyUI's
    # --highvram/--gpu-only policy, so every component has an explicit CPU
    # offload target.
    offload_device = torch.device("cpu")
    compute_dtype = _dtype(dtype_name, load_device)

    with torch.inference_mode(False):
        latent_channels = 16
        dit = LightningDiT(
            input_size=args["resolution"] // 8,
            patch_size=args["patch_size"],
            in_channels=latent_channels * 2,
            out_channels=latent_channels,
            hidden_size=args["dim"],
            depth=args["depth"],
            num_heads=args["num_heads"],
            mlp_ratio=args["mlp_ratio"],
            z_dims=args["enc_dim"],
            encdim_ratio=args["encdim_ratio"],
            auxiliary_time_cond=args["auxiliary_time_cond"],
            use_qknorm=args["use_qknorm"],
            use_swiglu=args["use_swiglu"],
            use_rope=args["use_rope"],
            use_rmsnorm=args["use_rmsnorm"],
            num_fused_layers=len(args["layer_dinov2b_list"]),
        )
        _load_safetensors_lean(dit, dit_path, compute_dtype)
        dit.eval().to(compute_dtype).requires_grad_(False)

        vae = AutoencoderKLQwenImage2D.from_pretrained(str(vae_dir))
        vae.eval().float().requires_grad_(False)

        dino = build_dinov2_vitl14()
        _load_dino(dino, dino_path, compute_dtype)
        dino.eval().to(compute_dtype).requires_grad_(False)

    return TESpeedVOSR2Model(
        comfy.model_patcher.ModelPatcher(
            dit,
            load_device=load_device,
            offload_device=offload_device,
        ),
        comfy.model_patcher.ModelPatcher(
            vae,
            load_device=load_device,
            offload_device=offload_device,
        ),
        comfy.model_patcher.ModelPatcher(
            dino,
            load_device=load_device,
            offload_device=offload_device,
        ),
        args,
        dtype_name,
        compute_dtype,
    )
