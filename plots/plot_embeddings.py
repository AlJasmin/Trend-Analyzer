from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from pymongo import UpdateOne

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from db.store import CONFIG_PATH, connect_from_config, load_settings
except ImportError:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from db.store import CONFIG_PATH, connect_from_config, load_settings

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
    parser = argparse.ArgumentParser(description="Plot stored embeddings with UMAP.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to settings.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Max number of posts to plot")
    parser.add_argument("--skip", type=int, default=0, help="Skip N posts")
    parser.add_argument("--sample", type=int, default=0, help="Random sample size (overrides limit)")
    parser.add_argument("--plot-output", default="plots/embeddings.png", help="Output image path")
    parser.add_argument("--plot-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--umap-n-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--umap-metric", default="cosine", help="UMAP distance metric")
    parser.add_argument(
        "--umap-cluster-dim",
        type=int,
        default=50,
        help="UMAP n_components for clustering coordinates",
    )
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Store UMAP coordinates (2D + cluster dim) in MongoDB",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Bulk write batch size")
    return parser.parse_args()


def iter_embeddings(
    collection,
    query: Dict[str, Any],
    *,
    limit: Optional[int],
    skip: int,
    sample: int,
) -> Iterable[Dict[str, Any]]:
    projection = {
        "_id": 0,
        "post_id": 1,
        "subreddit": 1,
        "embedding": 1,
    }

    if sample and sample > 0:
        pipeline = [
            {"$match": query},
            {"$sample": {"size": int(sample)}},
            {"$project": projection},
        ]
        for doc in collection.aggregate(pipeline):
            yield doc
        return

    cursor = collection.find(query, projection).skip(int(skip))
    if limit:
        cursor = cursor.limit(int(limit))
    for doc in cursor:
        yield doc


def compute_umap(
    embeddings: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    n_components: int,
) -> None:
    try:
        import umap
    except ImportError as exc:
        raise SystemExit("umap-learn is required for plotting. Install it first.") from exc

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def plot_umap(
    coords: np.ndarray,
    labels: List[str],
    *,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it first.") from exc

    unique_labels = sorted({label or "unknown" for label in labels})
    cmap = plt.get_cmap("tab10", max(len(unique_labels), 1))
    color_map = {label: cmap(idx) for idx, label in enumerate(unique_labels)}
    colors = [color_map.get(label or "unknown") for label in labels]

    plt.figure(figsize=(9, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.7, s=18)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color_map[label], label=label, markersize=6)
        for label in unique_labels
    ]
    plt.legend(handles=handles, title="subreddit", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title("Topic text embeddings (UMAP)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    logger.info("Plot saved to %s", output_path)


def main() -> None:
    args = parse_args()
    settings = load_settings(Path(args.config))
    _ = settings  # reserved for future plot config
    batch_size = int(args.batch_size or 0) or 1000

    store = connect_from_config(Path(args.config))
    query: Dict[str, Any] = {"embedding": {"$exists": True, "$ne": None}}
    docs = list(
        iter_embeddings(
            store.posts,
            query,
            limit=args.limit,
            skip=args.skip,
            sample=args.sample,
        )
    )

    if not docs:
        store.close()
        logger.info("No embeddings found to plot.")
        return

    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    post_ids: List[str] = []
    expected_dim: Optional[int] = None
    progress = tqdm(docs, desc="Load embeddings", unit="post")
    for doc in progress:
        vec = doc.get("embedding")
        if not vec:
            continue
        arr = np.asarray(vec, dtype=np.float32)
        if expected_dim is None:
            expected_dim = int(arr.shape[0])
        if arr.shape[0] != expected_dim:
            continue
        embeddings.append(arr)
        labels.append(doc.get("subreddit", "unknown"))
        post_ids.append(str(doc.get("post_id") or ""))
    progress.close()

    if not embeddings:
        store.close()
        logger.info("No embeddings found to plot.")
        return

    matrix = np.vstack(embeddings)
    coords_2d = compute_umap(
        matrix,
        seed=args.plot_seed,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        n_components=2,
    )
    plot_umap(
        coords_2d,
        labels,
        output_path=Path(args.plot_output),
    )

    if args.save_db:
        coords_cluster = compute_umap(
            matrix,
            seed=args.plot_seed,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            n_components=args.umap_cluster_dim,
        )
        updates: List[Any] = []
        matched = 0
        modified = 0
        for idx, post_id in enumerate(post_ids):
            if not post_id:
                continue
            updates.append(
                UpdateOne(
                    {"post_id": post_id},
                    {
                        "$set": {
                            "umap_x": float(coords_2d[idx, 0]),
                            "umap_y": float(coords_2d[idx, 1]),
                            "umap_50d": [float(val) for val in coords_cluster[idx]],
                        }
                    },
                )
            )
            if len(updates) >= batch_size:
                result = store.posts.bulk_write(updates, ordered=False)
                matched += result.matched_count
                modified += result.modified_count
                updates = []
        if updates:
            result = store.posts.bulk_write(updates, ordered=False)
            matched += result.matched_count
            modified += result.modified_count
        logger.info("Stored UMAP coordinates for %s posts (matched=%s, modified=%s).", len(post_ids), matched, modified)

    store.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
