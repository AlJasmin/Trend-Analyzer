from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

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


def plot_umap(
    embeddings: np.ndarray,
    labels: List[str],
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it first.") from exc

    try:
        import umap
    except ImportError as exc:
        raise SystemExit("umap-learn is required for plotting. Install it first.") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    coords = reducer.fit_transform(embeddings)

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

    store = connect_from_config(Path(args.config))
    try:
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
    finally:
        store.close()

    if not docs:
        logger.info("No embeddings found to plot.")
        return

    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    progress = tqdm(docs, desc="Load embeddings", unit="post")
    for doc in progress:
        vec = doc.get("embedding")
        if not vec:
            continue
        embeddings.append(np.asarray(vec, dtype=np.float32))
        labels.append(doc.get("subreddit", "unknown"))
    progress.close()

    if not embeddings:
        logger.info("No embeddings found to plot.")
        return

    matrix = np.vstack(embeddings)
    plot_umap(
        matrix,
        labels,
        seed=args.plot_seed,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        output_path=Path(args.plot_output),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
