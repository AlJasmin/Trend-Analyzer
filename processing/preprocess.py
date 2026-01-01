from __future__ import annotations

"""Minimal preprocessing utilities (RGB, resize, normalize)."""

from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def resize_square(img: Image.Image, size: int = 224) -> Image.Image:
    return img.resize((size, size), Image.BICUBIC)


def siglip_preprocess(
    images: Iterable[Image.Image],
    size: int = 224,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Simple SigLIP-style preprocessing: RGB -> resize -> tensor -> normalize (mean=std=0.5).
    Returns a stacked tensor [B, 3, size, size].
    """
    tensor_list: List[torch.Tensor] = []
    for img in images:
        rgb = ensure_rgb(img)
        resized = resize_square(rgb, size=size)
        arr = np.array(resized, dtype=np.float32) / 255.0  # HWC, 0-1
        t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        t = (t - 0.5) / 0.5
        tensor_list.append(t)

    batch = torch.stack(tensor_list, dim=0)
    if device:
        batch = batch.to(device=device, dtype=dtype or batch.dtype)
    elif dtype:
        batch = batch.to(dtype=dtype)
    return batch


__all__ = ["ensure_rgb", "resize_square", "siglip_preprocess"]
