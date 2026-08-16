"""ComfyUI custom-node pack entrypoint.

Clone or symlink this repository into `ComfyUI/custom_nodes/`.
Comfy loads this file via spec_from_file_location, so the pack root is
inserted onto sys.path before importing the internal package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from my_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from my_nodes.registry import NODE_CLASSES

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "comfy_entrypoint"]


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension, io
    from typing_extensions import override

    class MyNodesExtension(ComfyExtension):
        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return list(NODE_CLASSES)

    return MyNodesExtension()
