"""Windows-contract and lazy-dispatch tests for the four ComfyUI nodes."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from my_nodes.core.vosr2.settings import VOSR2Settings
from my_nodes.nodes.te_speed_vosr2 import (
    TESpeedVOSR2Image,
    TESpeedVOSR2Loader,
    TESpeedVOSR2Settings,
    TESpeedVOSR2Video,
)
from my_nodes.registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


class VOSR2SettingsTests(unittest.TestCase):
    """Keep dataclass defaults and normalization stable for serialized workflows."""

    def test_defaults_match_te_speed_v1_contract(self):
        self.assertEqual(
            VOSR2Settings().__dict__,
            {
                "quality_profile": "speed",
                "tile_strategy": "auto",
                "tile_size": 512,
                "tile_overlap": 32,
                "vae_tile_size": 1024,
                "vae_tile_overlap": 32,
                "image_batch": 1,
                "frame_batch": 2,
                "dino_batch": 2,
                "temporal_cache": False,
                "cache_threshold": 0.003,
                "cache_refresh": 4,
                "memory_policy": "auto",
                "color_alignment": "wavelet",
            },
        )

    def test_normalized_returns_a_sanitized_copy(self):
        original = VOSR2Settings(
            quality_profile="invalid",
            tile_strategy="invalid",
            tile_size=1,
            tile_overlap=999,
            vae_tile_size=0,
            vae_tile_overlap=999,
            image_batch=0,
            frame_batch=0,
            dino_batch=0,
            cache_threshold=-1.0,
            cache_refresh=0,
            memory_policy="invalid",
        )
        normalized = original.normalized()
        self.assertIsNot(original, normalized)
        self.assertEqual(original.tile_size, 1)
        self.assertEqual(normalized.quality_profile, "manual")
        self.assertEqual(normalized.tile_strategy, "auto")
        self.assertEqual(normalized.tile_size, 128)
        self.assertEqual(normalized.tile_overlap, 127)
        self.assertEqual(normalized.vae_tile_size, 0)
        self.assertEqual(normalized.vae_tile_overlap, 0)
        self.assertEqual(normalized.image_batch, 1)
        self.assertEqual(normalized.frame_batch, 1)
        self.assertEqual(normalized.dino_batch, 1)
        self.assertEqual(normalized.cache_threshold, 0.0)
        self.assertEqual(normalized.cache_refresh, 1)
        self.assertEqual(normalized.memory_policy, "auto")


class VOSR2NodeContractTests(unittest.TestCase):
    """Compare positional widget schemas and forwarded calls with the v1 contract."""

    def test_all_nodes_are_registered_with_windows_compatible_ids(self):
        expected = {
            "TESpeedVOSR2Loader": TESpeedVOSR2Loader,
            "TESpeedVOSR2Settings": TESpeedVOSR2Settings,
            "TESpeedVOSR2Image": TESpeedVOSR2Image,
            "TESpeedVOSR2Video": TESpeedVOSR2Video,
        }
        for node_id, node_class in expected.items():
            self.assertIs(NODE_CLASS_MAPPINGS[node_id], node_class)
            self.assertEqual(
                NODE_DISPLAY_NAME_MAPPINGS[node_id],
                node_class.DISPLAY_NAME,
            )

    def test_loader_contract_matches_windows_workflows(self):
        fake_store = ModuleType("my_nodes.core.vosr2.model_store")
        fake_store.DEFAULT_BUNDLE = "VOSR2"
        fake_store.bundle_names = lambda: ["VOSR2"]
        with mock.patch.dict(
            sys.modules,
            {"my_nodes.core.vosr2.model_store": fake_store},
        ):
            inputs = TESpeedVOSR2Loader.INPUT_TYPES()
        self.assertEqual(
            list(inputs["required"]),
            ["model_bundle", "precision", "memory_policy"],
        )
        self.assertEqual(
            list(inputs["optional"]),
            ["torch_compile", "vae_encode_amp"],
        )
        self.assertEqual(inputs["required"]["precision"][1]["default"], "auto")
        self.assertEqual(
            inputs["required"]["memory_policy"][1]["default"],
            "auto",
        )

    def test_settings_widget_order_and_defaults_are_stable(self):
        required = TESpeedVOSR2Settings.INPUT_TYPES()["required"]
        self.assertEqual(
            list(required),
            [
                "quality_profile",
                "tile_strategy",
                "tile_size",
                "tile_overlap",
                "vae_tile_size",
                "vae_tile_overlap",
                "image_batch",
                "frame_batch",
                "dino_batch",
                "temporal_cache",
                "cache_threshold",
                "cache_refresh",
                "memory_policy",
                "color_alignment",
            ],
        )
        values = {
            name: options[1]["default"]
            for name, options in required.items()
        }
        self.assertEqual(values, VOSR2Settings().__dict__)

    def test_image_and_video_contracts(self):
        image = TESpeedVOSR2Image.INPUT_TYPES()
        video = TESpeedVOSR2Video.INPUT_TYPES()
        self.assertEqual(
            list(image["required"]),
            ["model", "images", "scale", "seed"],
        )
        self.assertEqual(list(image["optional"]), ["settings"])
        self.assertEqual(image["required"]["scale"][0], "INT")
        self.assertEqual(image["required"]["scale"][1]["default"], 2)
        self.assertEqual(image["required"]["seed"][1]["default"], 42)
        self.assertEqual(
            list(video["optional"]),
            [
                "settings",
                "frame_batch",
                "temporal_cache",
                "cache_threshold",
                "cache_refresh",
            ],
        )

    def test_loader_configures_the_loaded_model(self):
        configured = SimpleNamespace()
        configured.set_memory_policy = mock.Mock()
        configured.set_torch_compile = mock.Mock()
        fake_store = ModuleType("my_nodes.core.vosr2.model_store")
        fake_store.load_model = mock.Mock(return_value=configured)
        fake_store._bundle_path = mock.Mock(return_value="/models/vosr2/VOSR2")
        with mock.patch.dict(
            sys.modules,
            {"my_nodes.core.vosr2.model_store": fake_store},
        ):
            result = TESpeedVOSR2Loader().load(
                "VOSR2",
                "fp16",
                "staged",
                True,
                True,
            )
        self.assertEqual(result, (configured,))
        fake_store.load_model.assert_called_once_with("VOSR2", "fp16")
        configured.set_memory_policy.assert_called_once_with("staged")
        self.assertTrue(configured.vae_encode_amp)
        configured.set_torch_compile.assert_called_once_with(
            True,
            "/models/vosr2/VOSR2",
        )

    def test_image_and_video_forward_normalized_settings(self):
        calls = []
        fake_inference = ModuleType("my_nodes.core.vosr2.inference")

        def run(*args, **kwargs):
            calls.append((args, kwargs))
            return ("output",)

        fake_inference.run_vosr2 = run
        with mock.patch.dict(
            sys.modules,
            {"my_nodes.core.vosr2.inference": fake_inference},
        ):
            image_result = TESpeedVOSR2Image().upscale(
                "model",
                "images",
                2,
                7,
            )
            video_result = TESpeedVOSR2Video().upscale(
                "model",
                "images",
                4,
                9,
                VOSR2Settings(frame_batch=3),
                frame_batch=5,
                temporal_cache=True,
                cache_threshold=0.02,
                cache_refresh=8,
            )
        self.assertEqual(image_result, ("output",))
        self.assertEqual(video_result, ("output",))
        self.assertIsInstance(calls[0][0][4], VOSR2Settings)
        video_settings = calls[1][0][4]
        self.assertEqual(video_settings.frame_batch, 5)
        self.assertTrue(video_settings.temporal_cache)
        self.assertEqual(video_settings.cache_threshold, 0.02)
        self.assertEqual(video_settings.cache_refresh, 8)
        self.assertEqual(
            calls[1][1],
            {
                "batch_override": 5,
                "temporal_cache": True,
                "auto_expand_vae_tile": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
