from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
    import comfy.model_management
except ImportError:
    torch = None

if torch is not None:
    from my_nodes.nodes.selflift import h3_upscaler


@unittest.skipIf(torch is None, "requires the ComfyUI Python environment")
class UpscalerOffloadTests(unittest.TestCase):
    class FakeModel:
        def __init__(self, events, error=None):
            self.events = events
            self.error = error
            self.conv_in = SimpleNamespace(
                out_channels=24,
                weight=torch.empty(1, dtype=torch.float32),
            )

        def temporal_window_budget(self, length):
            return length

        def temporal_chunk_settings(self):
            return 32, 0

        def __call__(self, x, scale, target_size):
            self.events.append("infer")
            if self.error is not None:
                raise self.error
            return x

    def run_lift(self, model):
        events = model.events
        patcher = SimpleNamespace(model=model, offload_device=torch.device("cpu"))
        z0_low = torch.zeros((1, 24, 2, 2, 2), dtype=torch.float32)

        with (
            mock.patch.object(h3_upscaler, "_load_model", return_value=patcher),
            mock.patch.object(
                h3_upscaler.comfy.model_management,
                "load_models_gpu",
                side_effect=lambda *args, **kwargs: events.append("load"),
            ),
            mock.patch.object(
                h3_upscaler.comfy.model_management,
                "unload_model_and_clones",
                side_effect=lambda *args, **kwargs: events.append("unload"),
            ) as unload,
            mock.patch.object(
                h3_upscaler.comfy.model_management,
                "intermediate_device",
                return_value=torch.device("cpu"),
            ),
        ):
            output = h3_upscaler.learned_latent_lift(
                z0_low,
                (2, 2),
                "fake.pth",
                device=torch.device("cpu"),
            )
        return output, patcher, unload

    def test_unloads_after_result_is_materialized(self):
        events = []
        output, patcher, unload = self.run_lift(self.FakeModel(events))

        self.assertEqual(events, ["load", "infer", "unload"])
        unload.assert_called_once_with(patcher)
        torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-6, rtol=0)

    def test_unloads_when_upscaler_forward_fails(self):
        events = []
        model = self.FakeModel(events, RuntimeError("inference failed"))

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_lift(model)

        self.assertEqual(events, ["load", "infer", "unload"])


if __name__ == "__main__":
    unittest.main()
