import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # advances/
sys.path.insert(0, str(ROOT))

import argparse
from src.config import FusionConfig
from src.pipeline import build_fused_vectors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Path to text embeddings .npy")
    ap.add_argument("--image", required=True, help="Path to image embeddings .npy")
    ap.add_argument("--out", required=True, help="Output path for fused .npy")

    ap.add_argument("--target-dim", type=int, default=256)
    ap.add_argument("--method", choices=["concat", "wsum"], default="concat")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--no-normalize-after", action="store_true")

    ap.add_argument("--use-has-image-flag", action="store_true")
    ap.add_argument("--has-image", default=None, help="Path to has_image.npy (0/1)")

    ap.add_argument("--pca-text", default="models/pca_text.joblib")
    ap.add_argument("--pca-image", default="models/pca_image.joblib")
    ap.add_argument("--no-fit", action="store_true", help="Use saved PCA instead of fitting new PCA")

    args = ap.parse_args()

    cfg = FusionConfig(
        target_dim=args.target_dim,
        method=args.method,
        alpha=args.alpha,
        normalize_after=not args.no_normalize_after,
        use_has_image_flag=args.use_has_image_flag,
    )

    fused = build_fused_vectors(
        text_path=args.text,
        image_path=args.image,
        out_fused_path=args.out,
        cfg=cfg,
        pca_text_path=args.pca_text,
        pca_img_path=args.pca_image,
        has_image_path=args.has_image,
        fit_pca=not args.no_fit,
    )

    print("Done.")
    print("Fused shape:", fused.shape)

if __name__ == "__main__":
    main()
