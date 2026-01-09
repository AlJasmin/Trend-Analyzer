from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pymongo import UpdateOne

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402

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

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable=None, **kwargs):
        return _NullTqdm(iterable, **kwargs)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster stored embeddings with UMAP + DBSCAN/HDBSCAN.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to settings.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Max number of posts to cluster")
    parser.add_argument("--skip", type=int, default=0, help="Skip N posts")
    parser.add_argument("--sample", type=int, default=0, help="Random sample size (overrides limit)")
    parser.add_argument("--plot-output", default="plots/clusters.png", help="Output plot path")
    parser.add_argument("--plot-seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--umap-cluster-dim",
        type=int,
        default=50,
        help="UMAP n_components for clustering",
    )
    parser.add_argument("--umap-n-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--umap-metric", default="cosine", help="UMAP distance metric")
    parser.add_argument(
        "--clusterer",
        choices=("dbscan", "hdbscan"),
        default="dbscan",
        help="Clustering algorithm",
    )
    parser.add_argument("--dbscan-eps", type=float, default=0.5, help="DBSCAN eps")
    parser.add_argument("--dbscan-min-samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--dbscan-metric", default="euclidean", help="DBSCAN distance metric")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=8, help="HDBSCAN min_cluster_size")
    parser.add_argument("--hdbscan-min-samples", type=int, default=None, help="HDBSCAN min_samples")
    parser.add_argument("--hdbscan-metric", default="euclidean", help="HDBSCAN distance metric")
    parser.add_argument("--only-missing", action="store_true", help="Only assign topic_id if missing")
    parser.add_argument("--noise-topic-id", default="noise", help="Topic_id label for noise points")
    parser.add_argument("--save-db", action="store_true", help="Persist topic_id assignments to DB")
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


def load_embeddings(
    collection,
    query: Dict[str, Any],
    *,
    limit: Optional[int],
    skip: int,
    sample: int,
) -> Tuple[List[str], np.ndarray]:
    docs = list(iter_embeddings(collection, query, limit=limit, skip=skip, sample=sample))
    if not docs:
        return [], np.empty((0, 0), dtype=np.float32)

    post_ids: List[str] = []
    vectors: List[np.ndarray] = []
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
        post_ids.append(str(doc.get("post_id")))
        vectors.append(arr)
    progress.close()

    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32)
    return post_ids, np.vstack(vectors)


def fit_umap(
    matrix: np.ndarray,
    *,
    seed: int,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise SystemExit("umap-learn is required for clustering. Install it first.") from exc

    n_samples = matrix.shape[0]
    if n_samples <= 1:
        return np.zeros((n_samples, 2), dtype=np.float32)
    adjusted_neighbors = max(2, min(int(n_neighbors), n_samples - 1))

    reducer = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=adjusted_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(matrix)


def fit_dbscan(
    coords: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    metric: str,
) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for clustering. Install it first.") from exc

    if coords.shape[0] == 0:
        return np.array([], dtype=int)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    return model.fit_predict(coords)


def fit_hdbscan(
    coords: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: Optional[int],
    metric: str,
) -> np.ndarray:
    try:
        import hdbscan
    except ImportError as exc:
        raise SystemExit("hdbscan is required for HDBSCAN clustering. Install it first.") from exc

    if coords.shape[0] == 0:
        return np.array([], dtype=int)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    return clusterer.fit_predict(coords)


def compute_centroids(coords: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    centroids: Dict[int, np.ndarray] = {}
    for label in sorted({int(value) for value in labels}):
        mask = labels == label
        if not np.any(mask):
            continue
        centroids[int(label)] = coords[mask].mean(axis=0)
    return centroids


def compute_distance(vec: np.ndarray, centroid: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        denom = np.linalg.norm(vec) * np.linalg.norm(centroid)
        if denom == 0:
            return 1.0
        return float(1.0 - float(np.dot(vec, centroid) / denom))
    if metric == "manhattan":
        return float(np.sum(np.abs(vec - centroid)))
    return float(np.linalg.norm(vec - centroid))


def compute_distances(coords: np.ndarray, labels: np.ndarray, metric: str) -> np.ndarray:
    centroids = compute_centroids(coords, labels)
    distances = np.full(len(labels), np.nan, dtype=np.float64)
    for idx, (label, vec) in enumerate(zip(labels, coords)):
        centroid = centroids.get(int(label))
        if centroid is None:
            continue
        distances[idx] = compute_distance(vec, centroid, metric)
    return distances


def update_center_distances(
    collection,
    post_ids: List[str],
    distances: np.ndarray,
) -> None:
    updates: List[UpdateOne] = []
    for post_id, distance in zip(post_ids, distances):
        if not np.isfinite(distance):
            continue
        updates.append(
            UpdateOne(
                {"post_id": post_id},
                {"$set": {"center_distance": float(distance)}},
            )
        )
    if not updates:
        return
    collection.bulk_write(updates, ordered=False)
    logger.info("Stored center_distance for %s posts.", len(updates))


def log_cluster_medians(
    labels: np.ndarray,
    distances: np.ndarray,
    topic_map: Dict[int, str],
    noise_topic_id: str,
) -> None:
    unique_labels = sorted({int(value) for value in labels})
    for label in unique_labels:
        mask = (labels == label) & np.isfinite(distances)
        if not np.any(mask):
            continue
        median_distance = float(np.median(distances[mask]))
        if int(label) == -1:
            topic_id = noise_topic_id
        else:
            topic_id = topic_map.get(int(label), str(label))
        logger.info("Median center_distance for %s: %.6f", topic_id, median_distance)


def build_topic_id_map(labels: np.ndarray) -> Dict[int, str]:
    unique_labels = sorted({int(label) for label in labels if int(label) != -1})
    return {label: f"topic_{idx + 1}" for idx, label in enumerate(unique_labels)}


def update_topic_ids(
    collection,
    post_ids: List[str],
    labels: np.ndarray,
    *,
    topic_map: Dict[int, str],
    noise_topic_id: str,
) -> None:
    cluster_posts: Dict[Any, List[str]] = {}
    for post_id, label in zip(post_ids, labels):
        label_int = int(label)
        if label_int == -1:
            topic_id = noise_topic_id
        else:
            topic_id = topic_map.get(label_int)
        if topic_id is None:
            continue
        cluster_posts.setdefault(topic_id, []).append(post_id)

    for topic_id, ids in cluster_posts.items():
        if not ids:
            continue
        collection.update_many({"post_id": {"$in": ids}}, {"$set": {"topic_id": topic_id}})
        logger.info("Assigned topic_id %s to %s posts.", topic_id, len(ids))


def plot_clusters(
    coords: np.ndarray,
    labels: List[str],
    *,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it first.") from exc

    if coords.shape[0] == 0:
        logger.info("No coordinates to plot.")
        return

    unique_labels = sorted({label for label in labels}, key=lambda value: str(value))
    cluster_labels = [label for label in unique_labels if label != "noise"]
    cmap = plt.get_cmap("tab20", max(len(cluster_labels), 1))
    color_map = {label: cmap(idx) for idx, label in enumerate(cluster_labels)}
    colors = []
    for label in labels:
        if label == "noise":
            colors.append((0.5, 0.5, 0.5, 0.5))
        else:
            colors.append(color_map.get(label, (0.2, 0.2, 0.2, 0.8)))

    plt.figure(figsize=(9, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.8, s=18)
    plt.title("Topic clusters (UMAP + DBSCAN)")
    plt.tight_layout()

    if 0 < len(cluster_labels) <= 20:
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=color_map[label], label=str(label), markersize=6)
            for label in cluster_labels
        ]
        if "noise" in unique_labels:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=(0.5, 0.5, 0.5, 0.6),
                    label="noise",
                    markersize=6,
                )
            )
        plt.legend(handles=handles, title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    logger.info("Plot saved to %s", output_path)


def build_query(only_missing: bool) -> Dict[str, Any]:
    query: Dict[str, Any] = {"embedding": {"$exists": True, "$ne": None}}
    if only_missing:
        query["$or"] = [
            {"topic_id": {"$exists": False}},
            {"topic_id": None},
            {"topic_id": ""},
        ]
    return query


def main() -> None:
    args = parse_args()
    np.random.seed(args.plot_seed)
    store = connect_from_config(Path(args.config))
    try:
        query = build_query(args.only_missing)
        post_ids, matrix = load_embeddings(
            store.posts,
            query,
            limit=args.limit,
            skip=args.skip,
            sample=args.sample,
        )
    finally:
        store.close()

    if not post_ids or matrix.size == 0:
        logger.info("No embeddings found to cluster.")
        return

    cluster_coords = fit_umap(
        matrix,
        seed=args.plot_seed,
        n_components=args.umap_cluster_dim,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
    )
    labels = fit_dbscan(
        cluster_coords,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
        metric=args.dbscan_metric,
    ) if args.clusterer == "dbscan" else fit_hdbscan(
        cluster_coords,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        metric=args.hdbscan_metric,
    )

    topic_map = build_topic_id_map(labels)
    noise_count = int(np.sum(labels == -1))
    logger.info("Found %s clusters (noise=%s).", len(topic_map), noise_count)

    mapped_labels = [
        topic_map.get(int(label), args.noise_topic_id) if int(label) != -1 else args.noise_topic_id
        for label in labels
    ]

    metric = args.dbscan_metric if args.clusterer == "dbscan" else args.hdbscan_metric
    distances = compute_distances(cluster_coords, labels, metric)

    if args.save_db:
        store = connect_from_config(Path(args.config))
        try:
            update_topic_ids(
                store.posts,
                post_ids,
                labels,
                topic_map=topic_map,
                noise_topic_id=args.noise_topic_id,
            )
            update_center_distances(store.posts, post_ids, distances)
        finally:
            store.close()
    else:
        logger.info("Skipping DB updates (use --save-db to persist).")

    log_cluster_medians(labels, distances, topic_map, args.noise_topic_id)

    plot_coords = fit_umap(
        matrix,
        seed=args.plot_seed,
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
    )
    plot_clusters(plot_coords, mapped_labels, output_path=Path(args.plot_output))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
