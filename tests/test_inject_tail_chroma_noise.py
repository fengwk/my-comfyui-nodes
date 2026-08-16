from __future__ import annotations

import unittest

import numpy as np

from my_nodes.core.inject_tail_chroma_noise import (
    alpha_for,
    inject_tail_chroma_noise,
    validate_taper,
)
from my_nodes.nodes.inject_tail_chroma_noise import (
    InjectTailChromaNoise,
    images_to_numpy_list,
    numpy_list_to_image_batch,
)


class TaperTests(unittest.TestCase):
    def test_validated_three_frame_ramp(self) -> None:
        values = [alpha_for(position, 22, 0.45, 0.10, 3) for position in range(19, 22)]
        self.assertAlmostEqual(values[0], 0.45 + (0.10 - 0.45) / 3)
        self.assertAlmostEqual(values[1], 0.45 + 2 * (0.10 - 0.45) / 3)
        self.assertAlmostEqual(values[2], 0.10)

    def test_early_tail_stays_at_alpha(self) -> None:
        self.assertEqual(alpha_for(0, 22, 0.45, 0.10, 3), 0.45)
        self.assertEqual(alpha_for(18, 22, 0.45, 0.10, 3), 0.45)

    def test_invalid_ramp_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_taper(22, 0.45, 0.10, 23)
        with self.assertRaises(ValueError):
            validate_taper(22, 1.1, 0.10, 3)
        with self.assertRaises(ValueError):
            validate_taper(22, 0.10, 0.45, 3)


class InjectionTests(unittest.TestCase):
    def _gray_frames(self, count: int = 24, height: int = 32, width: int = 48) -> list[np.ndarray]:
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        return [frame.copy() for _ in range(count)]

    def test_injection_is_deterministic(self) -> None:
        frames = self._gray_frames()
        first = inject_tail_chroma_noise(frames, 22, 0.45, 0.10, 3, seed=42)
        second = inject_tail_chroma_noise(frames, 22, 0.45, 0.10, 3, seed=42)
        self.assertEqual(len(first), 24)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_head_frames_are_unchanged(self) -> None:
        frames = self._gray_frames()
        out = inject_tail_chroma_noise(frames, 22, 0.45, 0.10, 3, seed=7)
        np.testing.assert_array_equal(out[0], frames[0])
        np.testing.assert_array_equal(out[1], frames[1])

    def test_tail_frames_are_changed(self) -> None:
        frames = self._gray_frames()
        out = inject_tail_chroma_noise(frames, 22, 0.45, 0.10, 3, seed=7)
        self.assertFalse(np.array_equal(out[-1], frames[-1]))
        self.assertFalse(np.array_equal(out[2], frames[2]))

    def test_rejects_tail_longer_than_clip(self) -> None:
        with self.assertRaises(ValueError):
            inject_tail_chroma_noise(self._gray_frames(4), 22, 0.45, 0.10, 3, seed=0)


class NodeAdapterTests(unittest.TestCase):
    def test_image_batch_roundtrip_and_inject(self) -> None:
        batch = np.full((8, 16, 24, 3), 0.5, dtype=np.float32)
        frames = images_to_numpy_list(batch)
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames[0].shape, (16, 24, 3))

        node = InjectTailChromaNoise()
        (injected,) = node.inject(batch, tail_frames=4, alpha=0.45, alpha_end=0.10, ramp_frames=3, seed=1)
        self.assertEqual(tuple(injected.shape), (8, 16, 24, 3))
        np.testing.assert_allclose(np.asarray(injected[:4]), batch[:4], atol=1 / 255)
        self.assertGreater(
            np.mean(np.abs(np.asarray(injected[-1]) - batch[-1])),
            0.01,
        )

    def test_numpy_list_to_batch_is_unit_interval(self) -> None:
        frames = [np.full((4, 4, 3), 128, dtype=np.uint8)]
        batch = numpy_list_to_image_batch(frames)
        self.assertAlmostEqual(float(np.asarray(batch).max()), 128 / 255.0, places=5)


if __name__ == "__main__":
    unittest.main()
