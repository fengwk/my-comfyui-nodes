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


def _node_id(cls):
    if hasattr(cls, "NODE_ID"):
        return cls.NODE_ID
    return cls.define_schema().node_id


def _display_name(cls):
    if hasattr(cls, "DISPLAY_NAME"):
        return cls.DISPLAY_NAME
    return cls.define_schema().display_name


NODE_CLASS_MAPPINGS = {_node_id(cls): cls for cls in NODE_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {_node_id(cls): _display_name(cls) for cls in NODE_CLASSES}
