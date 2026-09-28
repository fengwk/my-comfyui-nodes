"""The custom-node pack exposes one registration route to ComfyUI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


class RegistryEntrypointTests(unittest.TestCase):
    def test_all_registered_display_names_have_one_my_prefix(self):
        # Check both the public mappings and V3 schemas so UI paths agree.
        from my_nodes.registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        for node_id, cls in NODE_CLASS_MAPPINGS.items():
            with self.subTest(node_id=node_id):
                name = NODE_DISPLAY_NAME_MAPPINGS[node_id]
                self.assertTrue(name.startswith("My "), name)
                self.assertFalse(name.startswith("My My "), name)
                if hasattr(cls, "define_schema"):
                    self.assertEqual(cls.define_schema().display_name, name)

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
        # ComfyUI serves the presentation-only extension without changing V1 inputs.
        self.assertEqual(module.WEB_DIRECTORY, "./web")
        self.assertTrue((root.parent / module.WEB_DIRECTORY / "video_enhance.js").is_file())


if __name__ == "__main__":
    unittest.main()
