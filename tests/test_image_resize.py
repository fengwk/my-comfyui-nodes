"""Long-edge geometry, frame preservation and bounded interpolation tests."""

import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from my_nodes.core.image_resize import resize_long_edge
from my_nodes.nodes.image_resize import MyImageResizeLongEdge


class ImageResizeTests(unittest.TestCase):
    def test_landscape_portrait_square_and_thin_images(self):
        for h, w, edge, expected in [
            (928, 1664, 640, (357, 640)),
            (1664, 928, 640, (640, 357)),
            (12, 12, 24, (24, 24)),
            (1, 100, 2, (1, 2)),
        ]:
            with self.subTest(h=h, w=w):
                result = resize_long_edge(
                    torch.zeros(1, h, w, 3), edge, only_downscale=False
                )
                self.assertEqual(tuple(result.shape), (1, *expected, 3))

    def test_modes_preserve_frames_and_match_interpolation(self):
        images = torch.rand(3, 9, 16, 3)
        original = images.clone()
        for mode in ("bicubic", "bilinear", "nearest", "area"):
            check = Mock()
            actual = resize_long_edge(images, 8, mode, check)
            options = (
                {"align_corners": False, "antialias": True}
                if mode in {"bicubic", "bilinear"} else {}
            )
            expected = F.interpolate(
                images.movedim(-1, 1), size=(4, 8), mode=mode, **options
            ).clamp(0, 1).movedim(1, -1)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(check.call_count, 3)
        torch.testing.assert_close(images, original)

    def test_identity_and_dtype(self):
        images = torch.rand(2, 8, 16, 4).half()
        self.assertIs(resize_long_edge(images, 16), images)
        result = resize_long_edge(images, 32, only_downscale=False)
        self.assertEqual(result.dtype, images.dtype)
        self.assertEqual(result.device, images.device)
        self.assertEqual(tuple(result.shape), (2, 16, 32, 4))

    def test_invalid_inputs(self):
        for images, edge, mode in [
            (torch.zeros(2, 3), 8, "bicubic"),
            (torch.zeros(0, 8, 8, 3), 8, "bicubic"),
            (torch.zeros(1, 8, 8, 3, dtype=torch.uint8), 8, "bicubic"),
            (torch.zeros(1, 8, 8, 3), 0, "bicubic"),
            (torch.zeros(1, 8, 8, 3), 1.5, "bicubic"),
            (torch.zeros(1, 8, 8, 3), 4, "invalid"),
        ]:
            with self.assertRaises(ValueError):
                resize_long_edge(images, edge, mode)

    def test_only_downscale_returns_original_without_interpolation(self):
        # Identity return guarantees no resampling, copy or pixel changes.
        for h, w in [(8, 16), (16, 8), (16, 16), (32, 16)]:
            images = torch.rand(2, h, w, 3)
            with patch("torch.nn.functional.interpolate") as interpolate:
                self.assertIs(resize_long_edge(images, 32), images)
                interpolate.assert_not_called()

    def test_only_downscale_does_not_change_large_image_results(self):
        images = torch.rand(2, 16, 32, 3)
        torch.testing.assert_close(
            resize_long_edge(images, 8),
            resize_long_edge(images, 8, only_downscale=False),
        )

    def test_node_forwards_flag_and_accepts_old_inputs(self):
        import sys
        from types import ModuleType

        management = ModuleType("comfy.model_management")
        management.throw_exception_if_processing_interrupted = Mock()
        images = torch.rand(2, 8, 16, 3)
        with patch.dict(sys.modules, {"comfy.model_management": management}):
            node = MyImageResizeLongEdge()
            self.assertIs(node.resize(images, 32, "bicubic")[0], images)
            self.assertEqual(
                tuple(node.resize(images, 32, "bicubic", False)[0].shape),
                (2, 16, 32, 3),
            )

    def test_interrupt_propagates(self):
        check = Mock(side_effect=RuntimeError("interrupted"))
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            resize_long_edge(torch.zeros(2, 8, 8, 3), 4, check_interrupt=check)
        check.assert_called_once()

    def test_registration_and_defaults(self):
        from my_nodes.registry import NODE_CLASS_MAPPINGS

        self.assertIs(NODE_CLASS_MAPPINGS["MyImageResizeLongEdge"], MyImageResizeLongEdge)
        inputs = MyImageResizeLongEdge.INPUT_TYPES()["required"]
        self.assertEqual(inputs["long_edge"][1]["default"], 640)
        self.assertEqual(inputs["interpolation"][1]["default"], "bicubic")
        optional = MyImageResizeLongEdge.INPUT_TYPES()["optional"]
        self.assertTrue(optional["only_downscale"][1]["default"])


if __name__ == "__main__":
    unittest.main()
