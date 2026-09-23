"""Tensor-only inference tests that traverse production orchestration cheaply."""

from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import torch
    import torch.nn.functional as F
    import comfy.model_management
except ImportError:
    torch = None

if torch is not None:
    from my_nodes.core.vosr2 import inference
    from my_nodes.core.vosr2.color import adain_color_fix
    from my_nodes.core.vosr2.settings import VOSR2Settings


@unittest.skipIf(torch is None, "requires the ComfyUI Python environment")
class VOSR2InferenceTests(unittest.TestCase):
    """Use a deterministic fake backend to expose batching, caching, and retries."""

    class FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.memory_policy = "auto"
            self.dino_batches = []
            self.velocity_batches = []
            self.prepared_decode = 0
            self.clear_count = 0

        def set_memory_policy(self, policy):
            self.memory_policy = policy

        def encode(self, image, tile_size, overlap):
            latent = F.avg_pool2d(image, kernel_size=8, stride=8)
            channels = latent.shape[1]
            mean = torch.zeros((1, channels, 1, 1), device=latent.device)
            std = torch.ones_like(mean)
            return latent, mean, std

        def dino_features(self, image):
            self.dino_batches.append(image.shape[0])
            feature = image.mean(dim=(2, 3)).unsqueeze(1)
            return [feature]

        def velocity(self, value, t_cur, t_next, features):
            self.velocity_batches.append(value.shape[0])
            low, noise = value.chunk(2, dim=1)
            return noise - low

        def one_step(self, latent, noise, features):
            return latent

        def prepare_vae_decode(self):
            self.prepared_decode += 1

        def decode(self, latent, mean, std, tile_size, overlap):
            return F.interpolate(latent, scale_factor=8, mode="nearest")

        def clear_staged(self):
            self.clear_count += 1

    def test_noise_is_deterministic_without_touching_global_rng(self):
        torch.manual_seed(123)
        expected_next = torch.rand(1)
        torch.manual_seed(123)
        first = inference._noise(
            (3, 4, 4),
            99,
            torch.device("cpu"),
            torch.float32,
        )
        actual_next = torch.rand(1)
        second = inference._noise(
            (3, 4, 4),
            99,
            torch.device("cpu"),
            torch.float32,
        )
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(actual_next, expected_next)

    def test_frame_seed_wraps_at_uint64_boundary(self):
        # The Comfy widget permits UINT64_MAX; the following frame must wrap
        # rather than passing UINT64_MAX + 1 to torch.Generator.manual_seed.
        batch = inference._noise_batch(
            (1, 2, 2),
            (1 << 64) - 1,
            0,
            2,
            torch.device("cpu"),
            torch.float32,
        )
        wrapped = inference._noise(
            (1, 2, 2),
            0,
            torch.device("cpu"),
            torch.float32,
        )
        torch.testing.assert_close(batch[1], wrapped)

    def test_tile_geometry_covers_the_full_latent(self):
        settings = VOSR2Settings(tile_size=512, tile_overlap=32)
        side, locations = inference._tile_geometry(128, 128, settings)
        self.assertEqual(side, 64)
        coverage = torch.zeros((128, 128), dtype=torch.bool)
        for top, left in locations:
            coverage[top : top + side, left : left + side] = True
        self.assertTrue(bool(coverage.all()))

    def test_adain_is_finite_for_a_single_pixel(self):
        # Population variance matches NumPy/OpenCV semantics and avoids the
        # one-sample NaN produced by PyTorch's default unbiased estimator.
        target = torch.full((1, 3, 1, 1), 0.8)
        source = torch.full((1, 3, 1, 1), 0.2)
        result = adain_color_fix(target, source)
        self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(result, source)

    def test_identical_adjacent_frames_reuse_dino_features(self):
        model = self.FakeModel()
        frame = torch.rand((1, 3, 64, 64))
        padded = frame.expand(2, -1, -1, -1).clone()
        source = padded.movedim(1, -1).clone()
        cache = {"threshold": 0.003, "refresh": 4}
        maps = inference._extract_dino_maps(
            model,
            padded,
            source,
            [(0, 0)],
            8,
            4,
            True,
            cache,
        )
        self.assertEqual(model.dino_batches, [1])
        self.assertEqual(len(maps), 2)
        torch.testing.assert_close(
            maps[0][(0, 0)][0],
            maps[1][(0, 0)][0],
        )

    def test_tiled_pipeline_preserves_shape_and_batches_tiles(self):
        model = self.FakeModel()
        images = torch.rand((2, 40, 48, 3))
        settings = VOSR2Settings(
            tile_strategy="tiled",
            tile_size=128,
            tile_overlap=16,
            vae_tile_size=128,
            vae_tile_overlap=16,
            image_batch=2,
            dino_batch=4,
            color_alignment="none",
        )
        (result,) = inference.run_vosr2(
            model,
            images,
            scale=2,
            seed=7,
            settings=settings,
        )
        self.assertEqual(result.shape, (2, 80, 96, 3))
        self.assertEqual(result.device.type, "cpu")
        self.assertGreater(max(model.velocity_batches), 1)
        self.assertGreaterEqual(model.prepared_decode, 1)
        self.assertGreaterEqual(model.clear_count, 1)

    def test_full_frame_pipeline_preserves_shape(self):
        model = self.FakeModel()
        images = torch.rand((1, 32, 48, 3))
        settings = VOSR2Settings(
            tile_strategy="full_frame",
            vae_tile_size=0,
            image_batch=1,
            color_alignment="none",
        )
        (result,) = inference.run_vosr2(
            model,
            images,
            scale=1,
            seed=42,
            settings=settings,
        )
        self.assertEqual(result.shape, (1, 32, 48, 3))

    def test_tiled_oom_keeps_the_smaller_dit_batch(self):
        model = self.FakeModel()
        original_velocity = model.velocity
        attempts = []

        def oom_once(value, t_cur, t_next, features):
            attempts.append(value.shape[0])
            if value.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("synthetic")
            return original_velocity(value, t_cur, t_next, features)

        model.velocity = oom_once
        images = torch.rand((1, 64, 128, 3))
        settings = VOSR2Settings(
            quality_profile="speed",
            tile_strategy="tiled",
            tile_size=256,
            tile_overlap=32,
            vae_tile_size=0,
            color_alignment="none",
        )
        (result,) = inference.run_vosr2(
            model,
            images,
            scale=1,
            seed=42,
            settings=settings,
        )
        self.assertEqual(result.shape, images.shape)
        self.assertEqual(attempts, [3, 1, 1, 1])

    def test_dino_oom_keeps_the_smaller_batch(self):
        # A failed batch of three must not be retried for each later tile.
        model = self.FakeModel()
        attempts = []
        original = model.dino_features

        def oom_large(image):
            attempts.append(image.shape[0])
            if image.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("synthetic")
            return original(image)

        model.dino_features = oom_large
        frame = torch.rand((1, 3, 64, 64))
        inference._extract_dino_maps(
            model,
            frame,
            frame.movedim(1, -1),
            [(0, 0), (0, 1), (1, 0)],
            4,
            3,
            False,
            {"threshold": 0.003, "refresh": 4},
        )
        self.assertEqual(attempts, [3, 1, 1, 1])

    def test_dino_batch_assembly_oom_retries_at_smaller_batch(self):
        # Fail the crop concatenation, before DINO runs, and check persistent backoff.
        model = self.FakeModel()
        frame = torch.rand((1, 3, 64, 64))
        original_cat = torch.cat
        attempts = []

        def oom_on_packed_crops(tensors, *args, **kwargs):
            if tensors and tensors[0].ndim == 4 and tensors[0].shape[-1] == 32:
                attempts.append(len(tensors))
                if len(tensors) > 1:
                    raise torch.cuda.OutOfMemoryError("synthetic")
            return original_cat(tensors, *args, **kwargs)

        with patch.object(torch, "cat", side_effect=oom_on_packed_crops):
            inference._extract_dino_maps(
                model,
                frame,
                frame.movedim(1, -1),
                [(0, 0), (0, 1), (1, 0)],
                4,
                3,
                False,
                {"threshold": 0.003, "refresh": 4},
            )
        self.assertEqual(attempts, [3, 1, 1, 1])
        self.assertEqual(model.dino_batches, [1, 1, 1])

    def test_dit_batch_assembly_oom_retries_at_smaller_batch(self):
        # Fail latent tile concatenation, before DiT runs, and finish all tiles.
        model = self.FakeModel()
        images = torch.rand((1, 64, 128, 3))
        settings = VOSR2Settings(
            quality_profile="speed",
            tile_strategy="tiled",
            tile_size=256,
            tile_overlap=32,
            vae_tile_size=0,
            color_alignment="none",
        )
        original_cat = torch.cat
        attempts = []

        def oom_on_latent_tiles(tensors, *args, **kwargs):
            if (
                tensors
                and tensors[0].ndim == 4
                and tensors[0].shape[-1] == 8
                and kwargs.get("dim", 0) == 0
            ):
                attempts.append(len(tensors))
                if len(tensors) > 1:
                    raise torch.cuda.OutOfMemoryError("synthetic")
            return original_cat(tensors, *args, **kwargs)

        with patch.object(torch, "cat", side_effect=oom_on_latent_tiles):
            (result,) = inference.run_vosr2(
                model, images, scale=1, seed=42, settings=settings
            )
        self.assertEqual(result.shape, images.shape)
        self.assertEqual(attempts.count(3), 1)
        self.assertEqual(model.velocity_batches, [1, 1, 1])

    def test_cache_refresh_across_frame_chunks(self):
        # With refresh=2, frame 3 must re-run DINO even when every frame is
        # identical and each invocation holds just one frame.
        model = self.FakeModel()
        frame = torch.rand((1, 3, 64, 64))
        cache = {"threshold": 0.003, "refresh": 2}
        for _ in range(4):
            inference._extract_dino_maps(
                model,
                frame,
                frame.movedim(1, -1),
                [(0, 0)],
                8,
                2,
                True,
                cache,
            )
        self.assertEqual(model.dino_batches, [1, 1])
        self.assertEqual(cache["age"], 0)


if __name__ == "__main__":
    unittest.main()
