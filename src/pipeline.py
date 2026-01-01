import numpy as np
from pathlib import Path

from .config import FusionConfig
from .io_utils import load_npy, save_npy
from .align import PCAAligner
from .fuse import (
    fuse_concat,
    fuse_weighted_sum,
    append_has_image_flag,
    postprocess,
)

def build_fused_vectors(
    text_path: str,
    image_path: str,
    out_fused_path: str,
    cfg: FusionConfig,
    pca_text_path: str = "models/pca_text.joblib",
    pca_img_path: str = "models/pca_image.joblib",
    has_image_path: str | None = None,
    fit_pca: bool = True,
):
    text_embs = load_npy(text_path)   # (N, Dt)
    img_embs = load_npy(image_path)   # (N, Di)

    if text_embs.shape[0] != img_embs.shape[0]:
        raise ValueError("Text and image embeddings must have same N (same number of posts).")

    # PCA align
    aligner = PCAAligner(target_dim=cfg.target_dim)
    pca_text_path = str(Path(pca_text_path))
    pca_img_path = str(Path(pca_img_path))
    Path(pca_text_path).parent.mkdir(parents=True, exist_ok=True)

    if fit_pca:
        aligner.fit(text_embs, img_embs)
        aligner.save(pca_text_path, pca_img_path)
    else:
        aligner.load(pca_text_path, pca_img_path)

    text_a, img_a = aligner.transform(text_embs, img_embs)

    # Fusion
    if cfg.method == "concat":
        fused = fuse_concat(text_a, img_a)
    elif cfg.method == "wsum":
        fused = fuse_weighted_sum(text_a, img_a, alpha=cfg.alpha)
    else:
        raise ValueError(f"Unknown fusion method: {cfg.method}")

    # Optional: has_image flag anhängen
    if cfg.use_has_image_flag:
        if not has_image_path:
            raise ValueError("use_has_image_flag=True needs has_image_path.")
        has_image = load_npy(has_image_path)
        if has_image.shape[0] != fused.shape[0]:
            raise ValueError("has_image must have same N as embeddings.")
        fused = append_has_image_flag(fused, has_image)

    fused = postprocess(fused, normalize_after=cfg.normalize_after)
    save_npy(out_fused_path, fused)

    return fused
