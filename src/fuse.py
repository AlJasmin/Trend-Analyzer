import numpy as np
from .normalize import l2_normalize

def fuse_concat(text_a: np.ndarray, img_a: np.ndarray) -> np.ndarray:
    return np.concatenate([text_a, img_a], axis=1)

def fuse_weighted_sum(text_a: np.ndarray, img_a: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    if text_a.shape[1] != img_a.shape[1]:
        raise ValueError("Weighted sum needs same dimension for text and image.")
    return alpha * text_a + (1.0 - alpha) * img_a

def append_has_image_flag(fused: np.ndarray, has_image: np.ndarray) -> np.ndarray:
    """
    has_image: (N,) with 0/1
    """
    has_image = has_image.reshape(-1, 1).astype(np.float32)
    return np.concatenate([fused, has_image], axis=1)

def postprocess(fused: np.ndarray, normalize_after: bool = True) -> np.ndarray:
    return l2_normalize(fused) if normalize_after else fused
