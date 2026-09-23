"""Collect node classes from feature modules.

Add a new node by:
1. putting a node class with the V1 ComfyUI contract in `my_nodes/nodes/`
2. appending it to `NODE_CLASSES` below
"""

from __future__ import annotations

import logging

from my_nodes.nodes.inject_tail_chroma_noise import InjectTailChromaNoise

# Video enhance stays importable without torch/Comfy/GIMM. Registration itself
# must not start a backend; the nodes import those only when a stage is enabled,
# so these are plain top-level imports rather than guarded ones.
from my_nodes.nodes.video_enhance import MyDLSSRuntimeProbe, MyVideoEnhance
from my_nodes.nodes.video_enhance_stream import MyVideoEnhanceStream

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

try:
    from my_nodes.nodes.te_speed_vosr2 import VOSR2_NODE_CLASSES
except ImportError as exc:
    logging.warning("TE-Speed VOSR2 nodes not registered (missing dependency): %s", exc)
    VOSR2_NODE_CLASSES = ()

NODE_CLASSES = NODE_CLASSES + VOSR2_NODE_CLASSES
NODE_CLASSES = NODE_CLASSES + (MyVideoEnhance, MyDLSSRuntimeProbe, MyVideoEnhanceStream)


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
