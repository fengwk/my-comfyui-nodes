# VOSR2 third-party notices

The model definitions, tiled VAE implementation, and color-alignment baseline
in this directory are modified from
[ComfyUI-VOSR2](https://github.com/ylchen333/ComfyUI-VOSR2), whose source is
licensed under Apache License 2.0. A copy is included as
`LICENSE-APACHE-2.0.txt`.

Those files in turn contain code adapted from:

- [VOSR](https://github.com/cswry/VOSR)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [DiT](https://github.com/facebookresearch/DiT)
- [SiT](https://github.com/willisma/SiT)
- Qwen-Image / Wan / Hugging Face model implementations

Changes in this repository include ComfyUI node-pack integration, Linux-native
loading, SageAttention dispatch, bounded positional caches, staged/resident
model scheduling, frame-local temporal feature reuse, batched tile execution,
and accelerated low-frequency color alignment.
