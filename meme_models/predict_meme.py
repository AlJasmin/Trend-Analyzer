# predict_images.py
import argparse
from pathlib import Path
from types import SimpleNamespace

import open_clip
import torch
import yaml
from PIL import Image

from meme_models.meme_image_CLIP_Lightning_module import ImageOnlyMemeCLIP

torch.set_float32_matmul_precision("high")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
cfg = SimpleNamespace()


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _ensure_drop_probs(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("predict.model.drop_probs must be a list of at least three floats.")
    if len(values) < 3:
        raise ValueError("predict.model.drop_probs must provide at least three values.")
    return list(values[:3])


def load_predict_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        settings = yaml.safe_load(fp) or {}

    predict_cfg = settings.get("predict")
    if not isinstance(predict_cfg, dict):
        raise ValueError("Missing 'predict' section in the YAML config.")

    base_defaults = {
        "clip_variant": "ViT-L-14",
        "class_names": ["not_meme", "is_meme"],
    }
    merged = {**base_defaults, **predict_cfg}

    class_names = merged.get("class_names") or base_defaults["class_names"]

    model_defaults = {
        "feature_dim": 768,
        "map_dim": 768,
        "num_mapping_layers": 1,
        "num_pre_output_layers": 1,
        "drop_probs": [0.0, 0.0, 0.0],
        "ratio": 0.5,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "pos_label": 1,
        "label_smoothing": 0.0,
        "scale": 30,
        "num_classes": len(class_names),
    }
    model_cfg = predict_cfg.get("model") or {}
    drop_probs = model_cfg.get("drop_probs", model_defaults["drop_probs"])
    model_cfg["drop_probs"] = _ensure_drop_probs(drop_probs)
    merged["model"] = {**model_defaults, **model_cfg}

    if not merged.get("checkpoint_file"):
        raise ValueError("The 'predict.checkpoint_file' value must be set in the config.")

    merged["class_names"] = class_names

    return _to_namespace(merged)


def collect_images(inputs, recursive):
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
        elif p.is_dir():
            iterator = p.rglob("*") if recursive else p.glob("*")
            for f in iterator:
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    paths.append(f)
    if not paths:
        raise ValueError("No image files found.") 
    return sorted(paths)


def load_clip_encoder(device):
    variant = getattr(cfg, "clip_variant", "ViT-L-14")
    model, _, preprocess = open_clip.create_model_and_transforms(
        variant, pretrained="laion2b_s32b_b82k"
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


def encode_image(path, clip_model, preprocess, device):
    image = Image.open(path).convert("RGB")
    with torch.no_grad():
        tensor = preprocess(image).unsqueeze(0).to(device)
        feat = clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0)


def load_classifier(device):
    checkpoint = Path(cfg.checkpoint_file).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    model = ImageOnlyMemeCLIP.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        cfg=cfg.model,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    return model


def predict(model, clip_model, preprocess, device, image_paths):
    class_names = getattr(cfg, "class_names", ["not_meme", "is_meme"])
    results = []
    with torch.no_grad():
        for path in image_paths:
            try:
                feat = encode_image(path, clip_model, preprocess, device)
            except (OSError, ValueError) as exc:
                print(f"[WARN] Skipping {path}: {exc}")
                continue

            logits = model(feat.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
            pred_idx = int(probs.argmax().item())
            results.append(
                {
                    "path": str(path),
                    "label": class_names[pred_idx],
                    "score": float(probs[pred_idx]),
                    "probs": {name: float(probs[i]) for i, name in enumerate(class_names)},
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict meme / non-meme for new images.")
    parser.add_argument("inputs", nargs="+", help="Image files or directories")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively")
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Path to the YAML settings file (default: config/settings.yaml)",
    )
    args = parser.parse_args()

    global cfg
    cfg = load_predict_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = collect_images(args.inputs, args.recursive)
    clip_model, preprocess = load_clip_encoder(device)
    classifier = load_classifier(device)

    predictions = predict(classifier, clip_model, preprocess, device, image_paths)
    if not predictions:
        print("No predictions generated.")
        return

    for pred in predictions:
        probs_fmt = ", ".join(f"{k}={v:.4f}" for k, v in pred["probs"].items())
        print(f"{pred['path']} -> {pred['label']} (confidence {pred['score']:.4f}) [{probs_fmt}]")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


#python models/predict_meme.py "Z:\CODING\UNI\MemeCLIP\PREDICT"
