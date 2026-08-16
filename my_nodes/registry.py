"""Collect node classes from feature modules.

Add a new node by:
1. putting a `io.ComfyNode` subclass in `my_nodes/nodes/`
2. appending it to `NODE_CLASSES` below
"""

from __future__ import annotations

from my_nodes.nodes.inject_tail_chroma_noise import InjectTailChromaNoise

NODE_CLASSES = (
    InjectTailChromaNoise,
)

NODE_CLASS_MAPPINGS = {cls.NODE_ID: cls for cls in NODE_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {cls.NODE_ID: cls.DISPLAY_NAME for cls in NODE_CLASSES}
