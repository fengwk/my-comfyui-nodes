"""The custom-node pack exposes one registration route to ComfyUI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


class RegistryEntrypointTests(unittest.TestCase):
    def test_root_exposes_v1_mappings_without_partial_v3_extension(self):
        # ComfyUI selects V1 first; the old V3 list contained V1-only nodes.
        root = Path(__file__).resolve().parents[1] / "__init__.py"
        spec = importlib.util.spec_from_file_location("reviewed_node_pack", root)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertNotIn("comfy_entrypoint", vars(module))
        self.assertIn("TESpeedVOSR2Video", module.NODE_CLASS_MAPPINGS)
        self.assertNotIn("MySelfLiftH3Sampler", module.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
