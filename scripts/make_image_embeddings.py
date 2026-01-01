import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # advances/
sys.path.insert(0, str(ROOT))


import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from processing.image_embeddings import LlavaImageEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("make_image_embeddings")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def iter_posts_from_json_file(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(data, dict):
        if "posts" in data and isinstance(data["posts"], list):
            for item in data["posts"]:
                if isinstance(item, dict):
                    yield item
            return

        if "items" in data and isinstance(data["items"], list):
            for item in data["items"]:
                if isinstance(item, dict):
                    yield item
            return

        if "data" in data and isinstance(data["data"], dict) and "children" in data["data"]:
            children = data["data"]["children"]
            if isinstance(children, list):
                for c in children:
                    if isinstance(c, dict) and "data" in c and isinstance(c["data"], dict):
                        yield c["data"]
                return

    raise ValueError(f"Unrecognized JSON structure in {path}")


def _first_str(*vals: Any) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def extract_image_source(post: Dict[str, Any]) -> Optional[str]:
    # 1) direkte URL-Felder
    url = _first_str(
        post.get("image_url"),
        post.get("image"),
        post.get("url_overridden_by_dest"),
        post.get("url"),
    )
    if url and url.lower().startswith(("http://", "https://")):
        if url.lower().endswith(IMG_EXTS) or ("i.redd.it" in url) or ("i.imgur.com" in url):
            return url

    # 2) preview -> images -> source -> url
    preview = post.get("preview")
    if isinstance(preview, dict):
        images = preview.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict):
                source = first.get("source")
                if isinstance(source, dict):
                    purl = _first_str(source.get("url"))
                    if purl:
                        return purl.replace("&amp;", "&")

    # 3) media_metadata (gallery)
    media_meta = post.get("media_metadata")
    if isinstance(media_meta, dict) and media_meta:
        for _, meta in media_meta.items():
            if isinstance(meta, dict):
                s = meta.get("s")
                if isinstance(s, dict):
                    u = _first_str(s.get("u"))
                    if u:
                        return u.replace("&amp;", "&")

    # 4) lokale Bilder (falls du welche hast)
    local_path = _first_str(post.get("local_image_path"), post.get("image_path"), post.get("file_path"))
    if local_path:
        p = Path(local_path)
        if p.exists():
            return str(p)

    return None


def main():
    raw_files = sorted(RAW_DIR.glob("*.json"))
    if not raw_files:
        raise FileNotFoundError(f"No JSON files found in {RAW_DIR.resolve()}")

    posts: List[Dict[str, Any]] = []
    for rf in raw_files:
        try:
            before = len(posts)
            posts.extend(list(iter_posts_from_json_file(rf)))
            logger.info("Loaded %d posts from %s", len(posts) - before, rf.name)
        except Exception as e:
            logger.warning("Skipping %s: %s", rf.name, e)

    if not posts:
        raise RuntimeError("No posts loaded. Check data/raw JSON format.")

    sources = [extract_image_source(p) for p in posts]
    has_image = np.array([1 if s else 0 for s in sources], dtype=np.int8)

    logger.info("Total posts: %d", len(posts))
    logger.info("With image source: %d", int(has_image.sum()))

    embedder = LlavaImageEmbedder(
        model_name="llava-hf/llava-1.5-7b-hf",
        cache_size=128,
        timeout=15,
    )

    image_vecs: List[Optional[np.ndarray]] = [None] * len(posts)
    dim: Optional[int] = None

    for i, src in enumerate(sources):
        if not src:
            continue
        try:
            vec = np.asarray(embedder.get_embedding(src), dtype=np.float32)
            if dim is None:
                dim = int(vec.shape[0])
                logger.info("Detected embedding dim: %d", dim)
            if vec.shape[0] != dim:
                logger.warning("Dim mismatch at %d (%d != %d), skipping.", i, vec.shape[0], dim)
                continue
            image_vecs[i] = vec
            if (i + 1) % 25 == 0:
                logger.info("Processed %d/%d", i + 1, len(posts))
        except Exception as e:
            logger.warning("Failed on %d: %s", i, e)

    if dim is None:
        raise RuntimeError("No image embeddings computed. Check URLs/model.")

    zero = np.zeros((dim,), dtype=np.float32)
    image_matrix = np.vstack([v if v is not None else zero for v in image_vecs]).astype(np.float32)

    np.save(OUT_DIR / "image.npy", image_matrix, allow_pickle=False)
    np.save(OUT_DIR / "has_image.npy", has_image, allow_pickle=False)

    logger.info("Saved image.npy shape: %s", image_matrix.shape)
    logger.info("Saved has_image.npy shape: %s", has_image.shape)


if __name__ == "__main__":
    main()
