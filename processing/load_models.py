from __future__ import annotations

"""
Model loader utilities.

- SigLIP embedder for all images (fast, light).
- LLaVA embedder for meme/screenshot routing (heavier, optional).

Both loaders pick device automatically; override via args if needed.
"""

from typing import Optional

import torch

from processing.no_meme_image_embeddings import SiglipImageEmbedder
from processing.meme_image_embeddings import LlavaImageEmbedder


def load_siglip_embedder(
    model_name: str = "google/siglip-base-patch16-224",
    device: Optional[str] = None,
    cache_size: int = 256,
    dtype: Optional[torch.dtype] = None,
) -> SiglipImageEmbedder:
    return SiglipImageEmbedder(
        model_name=model_name,
        device=device,
        cache_size=cache_size,
        dtype=dtype,
    )


def load_llava_embedder(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    device: Optional[str] = None,
    cache_size: int = 128,
    dtype: Optional[torch.dtype] = None,
) -> LlavaImageEmbedder:
    return LlavaImageEmbedder(
        model_name=model_name,
        device=device,
        cache_size=cache_size,
        dtype=dtype,
    )


def load_all(
    device: Optional[str] = None,
    siglip_model: str = "google/siglip-base-patch16-224",
    llava_model: str = "llava-hf/llava-1.5-7b-hf",
) -> dict:
    """
    Load and return embedders in a dict.

    Returns:
        {
            "siglip": SiglipImageEmbedder,
            "llava": LlavaImageEmbedder,
        }
    """
    return {
        "siglip": load_siglip_embedder(model_name=siglip_model, device=device),
        "llava": load_llava_embedder(model_name=llava_model, device=device),
    }


__all__ = ["load_siglip_embedder", "load_llava_embedder", "load_all"]
