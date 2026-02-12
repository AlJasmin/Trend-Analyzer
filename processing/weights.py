from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

    def tqdm(iterable=None, **kwargs):
        return _NullTqdm(iterable, **kwargs)


logger = logging.getLogger(__name__)

STANCE_LABELS = ("agree", "disagree", "neutral")
SENTIMENT_LABELS = ("positive", "negative", "neutral")

STANCE_ALIASES = {
    "favor": "agree",
    "support": "agree",
    "pro": "agree",
    "yes": "agree",
    "for": "agree",
    "against": "disagree",
    "oppose": "disagree",
    "con": "disagree",
    "no": "disagree",
    "anti": "disagree",
    "none": "neutral",
}

SENTIMENT_ALIASES = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "negative": "negative",
    "neutral": "neutral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute comment weights and weighted stance/sentiment distributions."
    )
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to settings.yaml")
    parser.add_argument(
        "--save-db",
        action="store_true",
        help="Persist computed weights and post metrics to MongoDB.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute without writing to MongoDB.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Bulk write batch size")
    parser.add_argument("--limit", type=int, default=None, help="Max number of comments to process")
    parser.add_argument("--skip", type=int, default=0, help="Skip N comments")
    parser.add_argument(
        "--unknown-mode",
        choices=("drop", "neutral"),
        default="drop",
        help="How to handle non-core labels",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=5,
        help="Min stance-labeled comments required for polarization score",
    )
    return parser.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_label(
    raw: Any,
    *,
    labels: Iterable[str],
    aliases: Dict[str, str],
    unknown_mode: str,
) -> Optional[str]:
    if raw is None:
        return None
    label = str(raw).strip().lower()
    if not label:
        return None
    if label in aliases:
        label = aliases[label]
    if label in labels:
        return label
    if unknown_mode == "neutral" and "neutral" in labels:
        return "neutral"
    return None


def get_confidence(doc: Dict[str, Any]) -> float:
    conf = parse_float(doc.get("llm_confidence"))
    if conf is None:
        conf = parse_float(doc.get("confidence"))
    if conf is None:
        conf = 1.0
    return clamp(conf, 0.0, 1.0)


def compute_weight(upvote_score: Any, confidence: float) -> float:
    score = parse_float(upvote_score)
    if score is None:
        score = 0.0
    score = max(0.0, score)
    upvote_w = 1.0 + math.log1p(score)
    conf_w = 0.5 + 0.5 * confidence
    return upvote_w * conf_w


def iter_comments(
    collection,
    *,
    limit: Optional[int],
    skip: int,
) -> Iterable[Dict[str, Any]]:
    projection = {
        "_id": 0,
        "comment_id": 1,
        "post_id": 1,
        "upvote_score": 1,
        "score": 1,
        "llm_confidence": 1,
        "confidence": 1,
        "weight": 1,
        "stance_label": 1,
        "sentiment_label": 1,
    }
    cursor = collection.find({}, projection).skip(int(skip))
    if limit:
        cursor = cursor.limit(int(limit))
    for doc in cursor:
        yield doc


def build_distribution(
    sums: Dict[str, float],
    total: float,
    labels: Iterable[str],
) -> Dict[str, float]:
    if total <= 0:
        return {label: 0.0 for label in labels}
    return {label: sums.get(label, 0.0) / total for label in labels}


def compute_polarization(dist: Dict[str, float]) -> float:
    agree = dist.get("agree", 0.0)
    disagree = dist.get("disagree", 0.0)
    neutral = dist.get("neutral", 0.0)
    denom = max(agree + disagree, 1e-9)
    value = (1.0 - neutral) * (1.0 - abs(agree - disagree) / denom)
    return clamp(value, 0.0, 1.0)


def flush_updates(collection, updates: List[UpdateOne], *, save_db: bool) -> tuple[int, int]:
    if not updates:
        return 0, 0
    if not save_db:
        return len(updates), 0
    result = collection.bulk_write(updates, ordered=False)
    return result.matched_count, result.modified_count


def main() -> None:
    args = parse_args()
    save_db = bool(args.save_db) and not bool(args.dry_run)
    batch_size = int(args.batch_size or 0) or 1000
    min_comments = int(args.min_comments or 0)

    store = connect_from_config(Path(args.config))
    updates: List[UpdateOne] = []
    comment_updates = 0
    comment_matched = 0
    comment_modified = 0
    scanned = 0

    post_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "stance_sums": {label: 0.0 for label in STANCE_LABELS},
            "sentiment_sums": {label: 0.0 for label in SENTIMENT_LABELS},
            "stance_count": 0,
            "sentiment_count": 0,
            "stance_weight_sum": 0.0,
            "sentiment_weight_sum": 0.0,
        }
    )

    try:
        total = store.comments.count_documents({})
        if args.limit:
            total = min(total, int(args.limit))
        progress = tqdm(total=total, desc="Scan comments", unit="comment")

        for doc in iter_comments(store.comments, limit=args.limit, skip=args.skip):
            scanned += 1
            comment_id = str(doc.get("comment_id") or "")
            post_id = str(doc.get("post_id") or "")
            if not comment_id or not post_id:
                progress.update(1)
                continue

            confidence = get_confidence(doc)
            upvote_score = doc.get("upvote_score")
            if upvote_score is None:
                upvote_score = doc.get("score")
            weight = compute_weight(upvote_score, confidence)

            existing_weight = parse_float(doc.get("weight"))
            if existing_weight is None or abs(existing_weight - weight) > 1e-6:
                updates.append(
                    UpdateOne(
                        {"comment_id": comment_id},
                        {"$set": {"weight": float(weight)}},
                    )
                )
                comment_updates += 1

            stance_label = normalize_label(
                doc.get("stance_label"),
                labels=STANCE_LABELS,
                aliases=STANCE_ALIASES,
                unknown_mode=args.unknown_mode,
            )
            if stance_label:
                stats = post_stats[post_id]
                stats["stance_sums"][stance_label] += weight
                stats["stance_weight_sum"] += weight
                stats["stance_count"] += 1

            sentiment_label = normalize_label(
                doc.get("sentiment_label"),
                labels=SENTIMENT_LABELS,
                aliases=SENTIMENT_ALIASES,
                unknown_mode=args.unknown_mode,
            )
            if sentiment_label:
                stats = post_stats[post_id]
                stats["sentiment_sums"][sentiment_label] += weight
                stats["sentiment_weight_sum"] += weight
                stats["sentiment_count"] += 1

            if len(updates) >= batch_size:
                matched, modified = flush_updates(store.comments, updates, save_db=save_db)
                comment_matched += matched
                comment_modified += modified
                updates = []

            progress.update(1)

        if updates:
            matched, modified = flush_updates(store.comments, updates, save_db=save_db)
            comment_matched += matched
            comment_modified += modified
        progress.close()

        post_updates: List[UpdateOne] = []
        post_update_count = 0
        post_matched = 0
        post_modified = 0

        for post_id, stats in post_stats.items():
            update: Dict[str, Any] = {}

            stance_total = stats["stance_weight_sum"]
            if stance_total > 0:
                stance_dist = build_distribution(stats["stance_sums"], stance_total, STANCE_LABELS)
                update["stance_dist_weighted"] = stance_dist
                if stats["stance_count"] >= min_comments:
                    update["polarization_score"] = compute_polarization(stance_dist)

            sentiment_total = stats["sentiment_weight_sum"]
            if sentiment_total > 0:
                sentiment_dist = build_distribution(
                    stats["sentiment_sums"],
                    sentiment_total,
                    SENTIMENT_LABELS,
                )
                update["sentiment_dist_weighted"] = sentiment_dist

            if not update:
                continue

            post_updates.append(
                UpdateOne(
                    {"post_id": post_id},
                    {"$set": update},
                )
            )
            post_update_count += 1

            if len(post_updates) >= batch_size:
                matched, modified = flush_updates(store.posts, post_updates, save_db=save_db)
                post_matched += matched
                post_modified += modified
                post_updates = []

        if post_updates:
            matched, modified = flush_updates(store.posts, post_updates, save_db=save_db)
            post_matched += matched
            post_modified += modified

    finally:
        store.close()

    logger.info(
        "Comments scanned=%s, updates=%s, matched=%s, modified=%s.",
        scanned,
        comment_updates,
        comment_matched,
        comment_modified,
    )
    logger.info(
        "Posts updated=%s, matched=%s, modified=%s.",
        post_update_count,
        post_matched,
        post_modified,
    )
    if not save_db:
        logger.info("Dry run complete (no DB writes). Use --save-db to persist.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
