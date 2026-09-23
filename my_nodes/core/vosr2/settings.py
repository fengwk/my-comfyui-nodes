"""Validated inference settings compatible with TE-Speed-VOSR2 workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace

QUALITY_PROFILES = ("manual", "speed")
TILE_STRATEGIES = ("auto", "full_frame", "tiled")
MEMORY_POLICIES = ("auto", "resident", "staged")
COLOR_ALIGNMENTS = ("wavelet", "adain", "none")


@dataclass
class VOSR2Settings:
    """User-facing VOSR2 scheduling controls.

    Defaults and field order match TE-Speed-VOSR2 1.0's Windows extension so
    existing serialized settings links remain compatible.
    """

    quality_profile: str = "speed"
    tile_strategy: str = "auto"
    tile_size: int = 512
    tile_overlap: int = 32
    vae_tile_size: int = 1024
    vae_tile_overlap: int = 32
    image_batch: int = 1
    frame_batch: int = 2
    dino_batch: int = 2
    temporal_cache: bool = False
    cache_threshold: float = 0.003
    cache_refresh: int = 4
    memory_policy: str = "auto"
    color_alignment: str = "wavelet"

    def normalized(self) -> "VOSR2Settings":
        """Return a sanitized copy without mutating a linked settings object."""
        quality_profile = (
            self.quality_profile if self.quality_profile in QUALITY_PROFILES else "manual"
        )
        tile_strategy = (
            self.tile_strategy if self.tile_strategy in TILE_STRATEGIES else "auto"
        )
        memory_policy = (
            self.memory_policy if self.memory_policy in MEMORY_POLICIES else "auto"
        )

        tile_size = max(128, int(self.tile_size))
        tile_overlap = min(max(0, int(self.tile_overlap)), tile_size - 1)

        vae_tile_size = max(0, int(self.vae_tile_size))
        if vae_tile_size == 0:
            vae_tile_overlap = 0
        else:
            vae_tile_overlap = min(max(0, int(self.vae_tile_overlap)), vae_tile_size)

        return replace(
            self,
            quality_profile=quality_profile,
            tile_strategy=tile_strategy,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            vae_tile_size=vae_tile_size,
            vae_tile_overlap=vae_tile_overlap,
            image_batch=max(1, int(self.image_batch)),
            frame_batch=max(1, int(self.frame_batch)),
            dino_batch=max(1, int(self.dino_batch)),
            temporal_cache=bool(self.temporal_cache),
            cache_threshold=max(0.0, float(self.cache_threshold)),
            cache_refresh=max(1, int(self.cache_refresh)),
            memory_policy=memory_policy,
            color_alignment=str(self.color_alignment),
        )
