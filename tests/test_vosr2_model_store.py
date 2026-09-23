"""Model-store tests based on scheduler event traces and synthetic checkpoints."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

try:
    import torch
    import comfy.model_management
except ImportError:
    torch = None

if torch is not None:
    from safetensors.torch import save_file

    from my_nodes.core.vosr2.model_store import (
        TESpeedVOSR2Model,
        _load_hf_dino_safetensors,
    )


@unittest.skipIf(torch is None, "requires the ComfyUI Python environment")
class VOSR2ModelStoreTests(unittest.TestCase):
    """Verify exact component transitions without allocating production models."""

    class _Scale(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.gamma = torch.nn.Parameter(torch.empty(1))

    class _Attention(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(1, 3)
            self.proj = torch.nn.Linear(1, 1)

    class _Mlp(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(1, 2)
            self.fc2 = torch.nn.Linear(2, 1)

    class _Block(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(1)
            self.attn = VOSR2ModelStoreTests._Attention()
            self.ls1 = VOSR2ModelStoreTests._Scale()
            self.norm2 = torch.nn.LayerNorm(1)
            self.mlp = VOSR2ModelStoreTests._Mlp()
            self.ls2 = VOSR2ModelStoreTests._Scale()

    class _TinyDino(torch.nn.Module if torch is not None else object):
        def __init__(self):
            super().__init__()
            self.cls_token = torch.nn.Parameter(torch.empty(1, 1, 1))
            self.mask_token = torch.nn.Parameter(torch.empty(1, 1))
            self.pos_embed = torch.nn.Parameter(torch.empty(1, 2, 1))
            self.patch_embed = torch.nn.Module()
            self.patch_embed.proj = torch.nn.Conv2d(3, 1, 1)
            self.blocks = torch.nn.ModuleList(
                [VOSR2ModelStoreTests._Block() for _ in range(24)]
            )
            self.norm = torch.nn.LayerNorm(1)

    def make_model(self):
        def patcher(name):
            return SimpleNamespace(
                name=name,
                model=SimpleNamespace(),
                load_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
            )

        return TESpeedVOSR2Model(
            patcher("dit"),
            patcher("vae"),
            patcher("dino"),
            {"dinov2_size": 448, "layer_dinov2b_list": [17]},
            "fp16",
            torch.float16,
        )

    def test_staged_policy_unloads_the_previous_component(self):
        model = self.make_model()
        model.set_memory_policy("staged")
        events = []
        with (
            mock.patch(
                "my_nodes.core.vosr2.model_store.comfy.model_management.load_models_gpu",
                side_effect=lambda patchers, **kwargs: events.append(
                    ("load", patchers[0].name)
                ),
            ),
            mock.patch(
                "my_nodes.core.vosr2.model_store.comfy.model_management.unload_model_and_clones",
                side_effect=lambda patcher: events.append(("unload", patcher.name)),
            ),
        ):
            model._load(model.vae_patcher)
            model._load(model.vae_patcher)
            model._load(model.dino_patcher)
            model._load(model.dino_patcher)
            model._load(model.dit_patcher)
            model.clear_staged()
        self.assertEqual(
            events,
            [
                ("load", "vae"),
                ("unload", "vae"),
                ("load", "dino"),
                ("unload", "dino"),
                ("load", "dit"),
                ("unload", "dit"),
            ],
        )

    def test_resident_decode_pressure_does_not_reload_transformers(self):
        model = self.make_model()
        model.set_memory_policy("resident")
        events = []
        with (
            mock.patch(
                "my_nodes.core.vosr2.model_store.comfy.model_management.get_free_memory",
                return_value=1,
            ),
            mock.patch(
                "my_nodes.core.vosr2.model_store.comfy.model_management.load_models_gpu",
                side_effect=lambda patchers, **kwargs: events.append(
                    ("load", tuple(item.name for item in patchers))
                ),
            ),
            mock.patch(
                "my_nodes.core.vosr2.model_store.comfy.model_management.unload_model_and_clones",
                side_effect=lambda patcher: events.append(("unload", patcher.name)),
            ),
        ):
            model._resident_ready = True
            model.prepare_vae_decode()
            model._load(model.vae_patcher)
        self.assertEqual(
            events,
            [
                ("unload", "dit"),
                ("unload", "dino"),
                ("load", ("vae",)),
            ],
        )
        self.assertTrue(model._vae_decode_only)

    def test_leaving_resident_policy_unloads_only_bundle_components(self):
        model = self.make_model()
        model.memory_policy = "resident"
        model._resident_ready = True
        model._loaded_once = True
        with mock.patch(
            "my_nodes.core.vosr2.model_store.comfy.model_management.unload_model_and_clones"
        ) as unload:
            model.set_memory_policy("staged")
        self.assertEqual(
            [call.args[0].name for call in unload.call_args_list],
            ["dit", "vae", "dino"],
        )
        self.assertEqual(model.memory_policy, "staged")
        self.assertFalse(model._resident_ready)

    def test_switching_from_used_auto_to_staged_starts_clean(self):
        model = self.make_model()
        with (
            mock.patch(
                "my_nodes.core.vosr2.model_store.comfy.model_management.load_models_gpu"
            ),
            mock.patch(
                "my_nodes.core.vosr2.model_store.comfy.model_management.unload_model_and_clones"
            ) as unload,
        ):
            model._load(model.dino_patcher)
            model.set_memory_policy("staged")
        self.assertEqual(
            [call.args[0].name for call in unload.call_args_list],
            ["dit", "vae", "dino"],
        )
        self.assertFalse(model._loaded_once)
        self.assertIsNone(model._active_patcher)

    def test_hugging_face_dino_checkpoint_is_strictly_converted(self):
        state = {
            "embeddings.cls_token": torch.full((1, 1, 1), 10.0),
            "embeddings.mask_token": torch.full((1, 1), 11.0),
            "embeddings.position_embeddings": torch.full((1, 2, 1), 12.0),
            "embeddings.patch_embeddings.projection.weight": torch.full(
                (1, 3, 1, 1), 13.0
            ),
            "embeddings.patch_embeddings.projection.bias": torch.full((1,), 14.0),
            "layernorm.weight": torch.full((1,), 15.0),
            "layernorm.bias": torch.full((1,), 16.0),
        }
        for layer in range(24):
            prefix = f"encoder.layer.{layer}"
            for index, projection in enumerate(("query", "key", "value"), 1):
                state[f"{prefix}.attention.attention.{projection}.weight"] = (
                    torch.full((1, 1), float(index))
                )
                state[f"{prefix}.attention.attention.{projection}.bias"] = (
                    torch.full((1,), float(index + 3))
                )
            for name, shape, value in (
                ("attention.output.dense.weight", (1, 1), 7.0),
                ("attention.output.dense.bias", (1,), 8.0),
                ("layer_scale1.lambda1", (1,), 9.0),
                ("layer_scale2.lambda1", (1,), 10.0),
                ("mlp.fc1.weight", (2, 1), 11.0),
                ("mlp.fc1.bias", (2,), 12.0),
                ("mlp.fc2.weight", (1, 2), 13.0),
                ("mlp.fc2.bias", (1,), 14.0),
                ("norm1.weight", (1,), 15.0),
                ("norm1.bias", (1,), 16.0),
                ("norm2.weight", (1,), 17.0),
                ("norm2.bias", (1,), 18.0),
            ):
                state[f"{prefix}.{name}"] = torch.full(shape, value)

        module = self._TinyDino()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dinov2.safetensors"
            save_file(state, path)
            _load_hf_dino_safetensors(module, path, torch.float32)

        torch.testing.assert_close(
            module.blocks[0].attn.qkv.weight[:, 0],
            torch.tensor([1.0, 2.0, 3.0]),
        )
        torch.testing.assert_close(
            module.blocks[23].attn.qkv.bias,
            torch.tensor([4.0, 5.0, 6.0]),
        )
        self.assertEqual(float(module.cls_token.item()), 10.0)

    def test_vae_decode_is_strict_fp32(self):
        model = self.make_model()
        model.vae_patcher.load_device = torch.device("cpu")
        model.set_memory_policy("staged")
        observed = {}

        def decode(_vae, latent, mean, std, tile_size, overlap):
            observed["dtypes"] = (latent.dtype, mean.dtype, std.dtype)
            observed["autocast"] = torch.is_autocast_enabled("cpu")
            return latent

        with (
            mock.patch(
                "my_nodes.core.vosr2.model_store.comfy.model_management.load_models_gpu"
            ),
            mock.patch(
                "my_nodes.core.vosr2.model_store.comfy.model_management.unload_model_and_clones"
            ),
            mock.patch(
                "my_nodes.core.vosr2.model_store.tiled_vae.decode_dispatch",
                side_effect=decode,
            ),
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            result = model.decode(
                torch.ones((1, 1, 1, 1), dtype=torch.float16),
                torch.ones((1, 1, 1, 1), dtype=torch.float16),
                torch.ones((1, 1, 1, 1), dtype=torch.float16),
                0,
                0,
            )
        self.assertEqual(
            observed["dtypes"],
            (torch.float32, torch.float32, torch.float32),
        )
        self.assertFalse(observed["autocast"])
        self.assertEqual(result.dtype, torch.float32)

    def test_compiled_dit_failure_restores_eager_execution(self):
        # A failing compiled callable must only penalize its first invocation;
        # replacing the method proves all later tiles use the eager path.
        model = self.make_model()
        eager = mock.Mock(return_value="eager-result")
        model.dit_patcher.model.forward_flexible = eager
        compiled = mock.Mock(side_effect=RuntimeError("synthetic compile failure"))
        with (
            mock.patch.dict(sys.modules, {"triton": ModuleType("triton")}),
            mock.patch(
                "my_nodes.core.vosr2.model_store.torch.compile",
                return_value=compiled,
            ),
        ):
            model.set_torch_compile(True)
            result = model.dit_patcher.model.forward_flexible("input")
        self.assertEqual(result, "eager-result")
        self.assertTrue(model._compile_fallback)
        self.assertFalse(model._compile_enabled)
        self.assertIs(model.dit_patcher.model.forward_flexible, eager)
        eager.assert_called_once_with("input")


if __name__ == "__main__":
    unittest.main()
