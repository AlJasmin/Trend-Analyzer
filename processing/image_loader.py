#should load the image/s with a generater using next() and create a suitable input  (PIL/np) 

from __future__ import annotations

"""Utility helpers to load images from disk or URL once and return RGB PIL images."""

import io
from pathlib import Path
from typing import Iterable, Optional, Union

import requests
from PIL import Image


def is_url(path_like: Union[str, Path]) -> bool:
    return str(path_like).lower().startswith(("http://", "https://"))


def load_image(source: Union[str, Path], timeout: int = 10, max_size: Optional[int] = None) -> Image.Image:
    """
    Load an image from a local path or URL and return an RGB PIL Image.

    Args:
        source: file path or URL.
        timeout: request timeout for URLs.
        max_size: if set, resize so the longest side is at most this many pixels (keeps aspect ratio).
    """
    if is_url(source):
        resp = requests.get(str(source), timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {path}")
        img = Image.open(path).convert("RGB")

    if max_size:
        w, h = img.size
        scale = max(w, h) / float(max_size)
        if scale > 1.0:
            new_w, new_h = int(w / scale), int(h / scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
    return img


def iter_images(sources: Iterable[Union[str, Path]], timeout: int = 10, max_size: Optional[int] = None):
    """
    Generator over (source, image) pairs to avoid loading all images into RAM at once.

    Args:
        sources: iterable of file paths or URLs.
        timeout: request timeout for URLs.
        max_size: optional longest side resize.
    """
    for src in sources:
        yield src, load_image(src, timeout=timeout, max_size=max_size)


__all__ = ["load_image", "is_url", "iter_images"]
