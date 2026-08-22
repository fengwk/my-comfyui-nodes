"""Collect node classes from feature modules.

Add a new node by:
1. putting a `io.ComfyNode` subclass in `my_nodes/nodes/`
2. appending it to `NODE_CLASSES` below
"""

from __future__ import annotations

import logging

from my_nodes.nodes.inject_tail_chroma_noise import InjectTailChromaNoise

NODE_CLASSES = (
    InjectTailChromaNoise,
)

# Third-party Sol-Attn node (Kijai / t8star), vendored verbatim from
# https://huggingface.co/t8star/Sol-Attn-v2-wheels. It imports comfy_kitchen
# and comfy_api at module scope, so import it lazily here: unit tests run
# outside ComfyUI, where comfy_api does not exist.
try:
    from my_nodes.vendor.sol_attn_minimax_v2 import SolAttnMiniMax
except ImportError as exc:
    logging.warning("SolAttnMiniMax not registered (missing dependency): %s", exc)
    SolAttnMiniMax = None

if SolAttnMiniMax is not None:
    NODE_CLASSES = NODE_CLASSES + (SolAttnMiniMax,)


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
