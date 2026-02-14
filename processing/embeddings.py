from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from db.store import CONFIG_PATH, connect_from_config, load_settings
    from reddit.reddit_cleaner import build_topic_text
except ImportError:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from db.store import CONFIG_PATH, connect_from_config, load_settings
    from reddit.reddit_cleaner import build_topic_text

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress

    class _NullTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable or []

        def __iter__(self):
            return iter(self._iterable)

        def update(self, n=1):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, **kwargs):
        return _NullTqdm(iterable, **kwargs)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed topic_text and store embeddings in MongoDB."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument("--model", default=None, help="SentenceTransformer model name")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for embedding"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per chunk (model limit)",
    )
    parser.add_argument(
        "--overlap", type=int, default=None, help="Token overlap for chunking"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of posts to process"
    )
    parser.add_argument("--skip", type=int, default=0, help="Skip N posts")
    parser.add_argument(
        "--force", action="store_true", help="Recompute embeddings even if present"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable L2 normalization"
    )
    parser.add_argument(
        "--test-run",
        type=int,
        default=0,
        help="Embed N posts and plot without storing to MongoDB",
    )
    parser.add_argument(
        "--plot-method",
        choices=["pca", "umap", "tsne"],
        default="pca",
        help="Dimensionality reduction method for plotting",
    )
    parser.add_argument(
        "--plot-seed", type=int, default=42, help="Random seed for plot projection"
    )
    parser.add_argument(
        "--umap-n-neighbors", type=int, default=30, help="UMAP n_neighbors"
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=0.1, help="UMAP min_dist"
    )
    parser.add_argument("--umap-metric", default="cosine", help="UMAP distance metric")
    parser.add_argument(
        "--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity"
    )
    parser.add_argument("--tsne-metric", default="cosine", help="t-SNE distance metric")
    parser.add_argument(
        "--plot-output", default=None, help="Optional path to save plot image"
    )
    return parser.parse_args()


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def chunk_tokens(
    tokens: List[int],
    chunk_size: int,
    overlap: int,
) -> List[List[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def get_topic_text(doc: Dict[str, Any]) -> str:
    topic_text = (doc.get("topic_text") or "").strip()
    if topic_text:
        return topic_text
    title = doc.get("title", "")
    selftext = doc.get("selftext", "")
    return build_topic_text(title, selftext, bool(selftext))


def embed_short_texts(
    model,
    texts: List[str],
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    if normalize:
        vectors = np.vstack([l2_normalize(v) for v in vectors])
    return vectors


def embed_long_text(
    model,
    tokenizer,
    tokens: List[int],
    max_tokens: int,
    overlap: int,
    batch_size: int,
    normalize: bool,
) -> Optional[np.ndarray]:
    chunk_size = max_tokens - 2
    if chunk_size <= 0:
        raise ValueError("max_tokens must be >= 3")
    chunks = chunk_tokens(tokens, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return None

    chunk_texts = [tokenizer.decode(chunk) for chunk in chunks]
    chunk_lengths = np.array([len(chunk) for chunk in chunks], dtype=np.float32)
    vectors = embed_short_texts(
        model, chunk_texts, batch_size=batch_size, normalize=False
    )
    weights = chunk_lengths / max(chunk_lengths.sum(), 1.0)
    weighted = (vectors * weights[:, None]).sum(axis=0)
    return l2_normalize(weighted) if normalize else weighted


def iter_posts(
    collection,
    query: Dict[str, Any],
    limit: Optional[int],
    skip: int,
) -> Iterable[Dict[str, Any]]:
    projection = {
        "_id": 0,
        "post_id": 1,
        "subreddit": 1,
        "title": 1,
        "selftext": 1,
        "topic_text": 1,
        "cleaned_selftext": 1,
        "embedding": 1,
    }
    cursor = collection.find(query, projection).skip(int(skip))
    if limit:
        cursor = cursor.limit(int(limit))
    for doc in cursor:
        yield doc


def plot_embeddings(
    embeddings: List[np.ndarray],
    labels: List[str],
    output_path: Optional[str],
    method: str,
    seed: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    tsne_perplexity: float,
    tsne_metric: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it first."
        ) from exc

    if not embeddings:
        logger.info("No embeddings to plot.")
        return

    matrix = np.vstack(embeddings)
    if matrix.shape[0] < 2:
        coords = np.zeros((matrix.shape[0], 2))
    else:
        if method == "pca":
            try:
                from sklearn.decomposition import PCA
            except ImportError as exc:
                raise SystemExit(
                    "scikit-learn is required for plotting. Install it first."
                ) from exc
            coords = PCA(n_components=2).fit_transform(matrix)
        elif method == "umap":
            try:
                import umap
            except ImportError as exc:
                raise SystemExit(
                    "umap-learn is required for plotting. Install it first."
                ) from exc
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                random_state=seed,
            )
            coords = reducer.fit_transform(matrix)
        else:
            try:
                from sklearn.manifold import TSNE
            except ImportError as exc:
                raise SystemExit(
                    "scikit-learn is required for plotting. Install it first."
                ) from exc
            perplexity = min(tsne_perplexity, max(2.0, matrix.shape[0] - 1))
            coords = TSNE(
                n_components=2,
                perplexity=perplexity,
                metric=tsne_metric,
                init="random",
                random_state=seed,
            ).fit_transform(matrix)

    unique_labels = sorted({label or "unknown" for label in labels})
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
    color_map = {label: cmap(idx) for idx, label in enumerate(unique_labels)}
    colors = [color_map.get(label or "unknown") for label in labels]

    plt.figure(figsize=(9, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.7, s=18)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=color_map[label],
            label=label,
            markersize=6,
        )
        for label in unique_labels
    ]
    plt.legend(
        handles=handles, title="subreddit", bbox_to_anchor=(1.02, 1), loc="upper left"
    )
    plt.title(f"Topic text embeddings ({method.upper()})")
    plt.tight_layout()

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        logger.info("Plot saved to %s", out_path)
    else:
        plt.show()


def run_test_plot(
    store,
    model,
    tokenizer,
    *,
    query: Dict[str, Any],
    limit: int,
    skip: int,
    max_tokens: int,
    overlap: int,
    batch_size: int,
    normalize: bool,
    plot_method: str,
    plot_seed: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    tsne_perplexity: float,
    tsne_metric: str,
    output_path: Optional[str],
) -> None:
    docs = list(iter_posts(store.posts, query, limit, skip))
    if not docs:
        logger.info("No posts found for test run.")
        return

    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    short_texts: List[str] = []
    short_labels: List[str] = []
    long_items: List[Tuple[str, str, List[int]]] = []

    progress = tqdm(total=len(docs), desc="Embed (test)", unit="post")
    for doc in docs:
        post_id = doc.get("post_id")
        if not post_id:
            progress.update(1)
            continue
        text = get_topic_text(doc)
        if not text.strip():
            progress.update(1)
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        subreddit = doc.get("subreddit", "unknown")
        if len(tokens) <= max_tokens - 2:
            short_texts.append(text)
            short_labels.append(subreddit)
        else:
            long_items.append((str(post_id), subreddit, tokens))
        progress.update(1)

    if short_texts:
        vectors = embed_short_texts(
            model, short_texts, batch_size=batch_size, normalize=normalize
        )
        for label, vec in zip(short_labels, vectors):
            embeddings.append(vec)
            labels.append(label)

    for _, label, tokens in long_items:
        vec = embed_long_text(
            model,
            tokenizer,
            tokens,
            max_tokens=max_tokens,
            overlap=overlap,
            batch_size=batch_size,
            normalize=normalize,
        )
        if vec is None:
            continue
        embeddings.append(vec)
        labels.append(label)

    progress.close()
    plot_embeddings(
        embeddings,
        labels,
        output_path,
        method=plot_method,
        seed=plot_seed,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        tsne_perplexity=tsne_perplexity,
        tsne_metric=tsne_metric,
    )


def main() -> None:
    args = parse_args()
    settings = load_settings(Path(args.config))
    emb_cfg = settings.get("embeddings") or {}

    model_name = (
        args.model or emb_cfg.get("model") or "sentence-transformers/all-mpnet-base-v2"
    )
    batch_size = int(args.batch_size or emb_cfg.get("batch_size") or 32)
    max_tokens = int(args.max_tokens or emb_cfg.get("max_tokens") or 512)
    overlap = int(args.overlap or emb_cfg.get("chunk_overlap") or 50)
    normalize = not args.no_normalize

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required. Install it first."
        ) from exc

    logger.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    store = connect_from_config(Path(args.config))
    try:
        if args.force:
            query: Dict[str, Any] = {}
        else:
            query = {"$or": [{"embedding": {"$exists": False}}, {"embedding": None}]}

        if args.test_run:
            run_test_plot(
                store,
                model,
                tokenizer,
                query=query,
                limit=int(args.test_run),
                skip=args.skip,
                max_tokens=max_tokens,
                overlap=overlap,
                batch_size=batch_size,
                normalize=normalize,
                plot_method=args.plot_method,
                plot_seed=args.plot_seed,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_min_dist=args.umap_min_dist,
                umap_metric=args.umap_metric,
                tsne_perplexity=args.tsne_perplexity,
                tsne_metric=args.tsne_metric,
                output_path=args.plot_output,
            )
            return

        total = store.posts.count_documents(query)
        if args.limit:
            total = min(total, int(args.limit))

        processed = 0
        updated = 0
        skipped = 0
        short_count = 0
        long_count = 0

        batch: List[Dict[str, Any]] = []
        progress = tqdm(total=total, desc="Embed posts", unit="post")

        for doc in iter_posts(store.posts, query, args.limit, args.skip):
            batch.append(doc)
            if len(batch) < batch_size:
                continue

            processed, updated, skipped, short_count, long_count = process_batch(
                batch,
                model,
                tokenizer,
                model_name=model_name,
                max_tokens=max_tokens,
                overlap=overlap,
                batch_size=batch_size,
                normalize=normalize,
                collection=store.posts,
                processed=processed,
                updated=updated,
                skipped=skipped,
                short_count=short_count,
                long_count=long_count,
                progress=progress,
            )
            batch = []

        if batch:
            processed, updated, skipped, short_count, long_count = process_batch(
                batch,
                model,
                tokenizer,
                model_name=model_name,
                max_tokens=max_tokens,
                overlap=overlap,
                batch_size=batch_size,
                normalize=normalize,
                collection=store.posts,
                processed=processed,
                updated=updated,
                skipped=skipped,
                short_count=short_count,
                long_count=long_count,
                progress=progress,
            )

        progress.close()
        logger.info(
            "Embedding complete. processed=%s updated=%s skipped=%s short=%s long=%s",
            processed,
            updated,
            skipped,
            short_count,
            long_count,
        )
    finally:
        store.close()


def process_batch(
    batch: List[Dict[str, Any]],
    model,
    tokenizer,
    *,
    model_name: str,
    max_tokens: int,
    overlap: int,
    batch_size: int,
    normalize: bool,
    collection,
    processed: int,
    updated: int,
    skipped: int,
    short_count: int,
    long_count: int,
    progress,
) -> Tuple[int, int, int, int, int]:
    texts: List[str] = []
    ids: List[str] = []
    long_items: List[Tuple[str, List[int]]] = []

    for doc in batch:
        processed += 1
        post_id = doc.get("post_id")
        if not post_id:
            skipped += 1
            continue
        text = get_topic_text(doc)
        if not text.strip():
            skipped += 1
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens - 2:
            texts.append(text)
            ids.append(str(post_id))
        else:
            long_items.append((str(post_id), tokens))

    if texts:
        vectors = embed_short_texts(
            model, texts, batch_size=batch_size, normalize=normalize
        )
        for post_id, vec in zip(ids, vectors):
            collection.update_one(
                {"post_id": post_id},
                {
                    "$set": {
                        "embedding": vec.tolist(),
                        "embedding_model": model_name,
                        "embedding_dim": int(vec.shape[0]),
                    }
                },
            )
            updated += 1
        short_count += len(texts)

    for post_id, tokens in long_items:
        vec = embed_long_text(
            model,
            tokenizer,
            tokens,
            max_tokens=max_tokens,
            overlap=overlap,
            batch_size=batch_size,
            normalize=normalize,
        )
        if vec is None:
            skipped += 1
            continue
        collection.update_one(
            {"post_id": post_id},
            {
                "$set": {
                    "embedding": vec.tolist(),
                    "embedding_model": model_name,
                    "embedding_dim": int(vec.shape[0]),
                }
            },
        )
        updated += 1
        long_count += 1

    if progress is not None:
        progress.update(len(batch))

    return processed, updated, skipped, short_count, long_count


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
