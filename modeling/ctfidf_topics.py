from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402
from reddit.reddit_cleaner import build_topic_text  # noqa: E402

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
        description="Compute c-TF-IDF keywords per topic_id."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--output", default="reports/ctfidf_topics.csv", help="Output CSV path"
    )
    parser.add_argument("--top-n", type=int, default=12, help="Top terms per topic")
    parser.add_argument(
        "--min-posts", type=int, default=5, help="Minimum posts per topic"
    )
    parser.add_argument(
        "--include-noise", action="store_true", help="Include noise topic_id"
    )
    parser.add_argument("--min-df", type=int, default=2, help="Min document frequency")
    parser.add_argument(
        "--max-df", type=float, default=0.95, help="Max document frequency"
    )
    parser.add_argument("--ngram-max", type=int, default=2, help="Max n-gram size")
    parser.add_argument(
        "--stopwords-file",
        default=None,
        help="Optional stopwords file (one term per line)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of posts to read"
    )
    parser.add_argument("--skip", type=int, default=0, help="Skip N posts")
    return parser.parse_args()


def iter_posts(
    collection,
    query: Dict[str, Any],
    *,
    limit: Optional[int],
    skip: int,
) -> Iterable[Dict[str, Any]]:
    projection = {
        "_id": 0,
        "post_id": 1,
        "topic_id": 1,
        "topic_text": 1,
        "title": 1,
        "selftext": 1,
    }
    cursor = collection.find(query, projection).skip(int(skip))
    if limit:
        cursor = cursor.limit(int(limit))
    for doc in cursor:
        yield doc


def get_topic_text(doc: Dict[str, Any]) -> str:
    text = (doc.get("topic_text") or "").strip()
    if text:
        return text
    title = doc.get("title", "")
    selftext = doc.get("selftext", "")
    return build_topic_text(title, selftext, bool(selftext))


def collect_documents(
    collection,
    *,
    include_noise: bool,
    min_posts: int,
    limit: Optional[int],
    skip: int,
) -> Tuple[List[str], List[str], List[int]]:
    query: Dict[str, Any] = {"topic_id": {"$exists": True, "$ne": None}}
    docs = list(iter_posts(collection, query, limit=limit, skip=skip))
    if not docs:
        return [], [], []

    by_topic: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}
    progress = tqdm(docs, desc="Collect posts", unit="post")
    for doc in progress:
        topic_id = str(doc.get("topic_id") or "")
        if not topic_id:
            continue
        if not include_noise and topic_id == "noise":
            continue
        text = get_topic_text(doc)
        if not text:
            continue
        by_topic.setdefault(topic_id, []).append(text)
        counts[topic_id] = counts.get(topic_id, 0) + 1
    progress.close()

    topic_ids: List[str] = []
    documents: List[str] = []
    post_counts: List[int] = []
    for topic_id, texts in sorted(by_topic.items()):
        count = counts.get(topic_id, len(texts))
        if count < min_posts:
            continue
        topic_ids.append(topic_id)
        documents.append("\n".join(texts))
        post_counts.append(count)

    return topic_ids, documents, post_counts


def compute_ctfidf(
    documents: List[str],
    *,
    min_df: int,
    max_df: float,
    ngram_max: int,
    stop_words: Optional[List[str]],
) -> Tuple[np.ndarray, List[str]]:
    try:
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise SystemExit("scikit-learn is required. Install it first.") from exc

    vectorizer = CountVectorizer(
        stop_words=stop_words or "english",
        ngram_range=(1, max(1, int(ngram_max))),
        min_df=int(min_df),
        max_df=float(max_df),
    )
    counts = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64), []

    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    tf = counts.multiply(1.0 / row_sums[:, None])

    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    n_docs = counts.shape[0]
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    ctfidf = tf.multiply(idf)
    return ctfidf, list(terms)


def top_terms_for_topic(
    row, terms: List[str], top_n: int
) -> Tuple[List[str], List[float]]:
    if row.nnz == 0:
        return [], []
    indices = row.indices
    data = row.data
    if len(data) <= top_n:
        order = np.argsort(-data)
    else:
        order = np.argsort(-data)[:top_n]
    top_terms = [terms[indices[idx]] for idx in order]
    top_scores = [float(data[idx]) for idx in order]
    return top_terms, top_scores


def load_stopwords(path: Path) -> List[str]:
    if not path.exists():
        raise SystemExit(f"Stopwords file not found: {path}")
    words: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip().lower()
        if not item or item.startswith("#"):
            continue
        words.append(item)
    return words


def write_csv(
    path: Path,
    topic_ids: List[str],
    post_counts: List[int],
    ctfidf,
    terms: List[str],
    *,
    top_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["topic_id", "post_count", "top_terms", "top_scores"])
        for idx, topic_id in enumerate(topic_ids):
            row = ctfidf.getrow(idx)
            top_terms, top_scores = top_terms_for_topic(row, terms, top_n)
            writer.writerow(
                [
                    topic_id,
                    post_counts[idx],
                    "|".join(top_terms),
                    "|".join(f"{score:.6f}" for score in top_scores),
                ]
            )


def main() -> None:
    args = parse_args()
    store = connect_from_config(Path(args.config))
    try:
        topic_ids, documents, post_counts = collect_documents(
            store.posts,
            include_noise=args.include_noise,
            min_posts=args.min_posts,
            limit=args.limit,
            skip=args.skip,
        )
    finally:
        store.close()

    if not topic_ids:
        logger.info("No topics found to compute c-TF-IDF.")
        return

    stop_words: Optional[List[str]] = None
    if args.stopwords_file:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        except ImportError as exc:
            raise SystemExit(
                "scikit-learn is required for stopwords. Install it first."
            ) from exc
        custom = load_stopwords(Path(args.stopwords_file))
        merged = set(ENGLISH_STOP_WORDS)
        merged.update(custom)
        stop_words = sorted(merged)

    ctfidf, terms = compute_ctfidf(
        documents,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        stop_words=stop_words,
    )
    if ctfidf.size == 0 or not terms:
        logger.info("No terms found after vectorization.")
        return

    output_path = Path(args.output)
    write_csv(
        output_path,
        topic_ids,
        post_counts,
        ctfidf,
        terms,
        top_n=args.top_n,
    )
    logger.info("Wrote c-TF-IDF topics to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
