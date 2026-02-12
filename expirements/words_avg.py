from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

    def tqdm(iterable=None, **kwargs):
        return _NullTqdm(iterable, **kwargs)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot word count distribution with percentile trimming."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--plot-output", default="plots/words_avg.png", help="Output plot path"
    )
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins")
    parser.add_argument("--limit", type=int, default=None, help="Max number of posts")
    parser.add_argument("--skip", type=int, default=0, help="Skip N posts")
    parser.add_argument(
        "--sample", type=int, default=0, help="Random sample size (overrides limit)"
    )
    parser.add_argument(
        "--trim-percent", type=float, default=10.0, help="Drop longest X%% of posts"
    )
    parser.add_argument(
        "--untere-perzentil", type=float, default=0.0, help="Show lower percentile line"
    )
    return parser.parse_args()


def iter_posts(
    collection,
    *,
    limit: Optional[int],
    skip: int,
    sample: int,
) -> Iterable[Dict[str, Any]]:
    projection = {"_id": 0, "selftext": 1}
    if sample and sample > 0:
        pipeline = [
            {"$match": {}},
            {"$sample": {"size": int(sample)}},
            {"$project": projection},
        ]
        for doc in collection.aggregate(pipeline):
            yield doc
        return

    cursor = collection.find({}, projection).skip(int(skip))
    if limit:
        cursor = cursor.limit(int(limit))
    for doc in cursor:
        yield doc


def word_count(text: str) -> int:
    return len(text.split())


def percentile(values: List[int], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values_sorted[int(k)])
    lower = values_sorted[f]
    upper = values_sorted[c]
    return float(lower + (upper - lower) * (k - f))


def main() -> None:
    args = parse_args()
    output_path = Path(args.plot_output)

    store = connect_from_config(Path(args.config))
    try:
        docs = list(
            iter_posts(
                store.posts, limit=args.limit, skip=args.skip, sample=args.sample
            )
        )
    finally:
        store.close()

    if not docs:
        logger.info("No posts found.")
        return

    word_counts: List[int] = []
    progress = tqdm(docs, desc="Count words", unit="post")
    for doc in progress:
        text = doc.get("selftext") or ""
        word_counts.append(word_count(text))
    progress.close()

    if not word_counts:
        logger.info("No word counts to plot.")
        return

    trim_percent = min(max(args.trim_percent, 0.0), 99.0)
    cutoff_pct = 100.0 - trim_percent
    cutoff = percentile(word_counts, cutoff_pct)
    lower_pct = min(max(args.untere_perzentil, 0.0), 99.0)
    lower_cutoff = percentile(word_counts, lower_pct) if lower_pct > 0 else None
    trimmed = [count for count in word_counts if count <= cutoff]
    removed = len(word_counts) - len(trimmed)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it first."
        ) from exc

    plt.figure(figsize=(9, 6))
    plt.hist(
        trimmed, bins=args.bins, color="#4c72b0", alpha=0.8, orientation="horizontal"
    )
    plt.axhline(
        cutoff,
        color="#c0392b",
        linestyle="--",
        label=f"p{int(cutoff_pct)}={cutoff:.1f}",
    )
    if lower_cutoff is not None:
        plt.axhline(
            lower_cutoff,
            color="#27ae60",
            linestyle="--",
            label=f"p{int(lower_pct)}={lower_cutoff:.1f}",
        )
    plt.xlabel("Post count")
    plt.ylabel("Words per post (selftext)")
    plt.title("Selftext word count distribution (trimmed)")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    logger.info(
        "Trimmed %s of %s posts (%.1f%%).",
        removed,
        len(word_counts),
        (removed / len(word_counts)) * 100,
    )
    logger.info("Percentile cutoff p%s: %.2f words", int(cutoff_pct), cutoff)
    if lower_cutoff is not None:
        logger.info("Lower percentile p%s: %.2f words", int(lower_pct), lower_cutoff)
    logger.info("Plot saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
