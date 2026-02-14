from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from db.store import CONFIG_PATH, connect_from_config
except ImportError:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from db.store import CONFIG_PATH, connect_from_config

logger = logging.getLogger(__name__)

POST_FIELDS = [
    "post_id",
    "subreddit",
    "title",
    "selftext",
    "cleaned_selftext",
    "topic_text",
    "embedding",
    "embedding_model",
    "embedding_dim",
    "topic_id",
    "topic_name",
    "topic_description",
    "stance_dist_weighted",
    "sentiment_dist_weighted",
    "polarization_score",
    "snapshot_week",
    "created_utc",
]

COMMENT_FIELDS = [
    "comment_id",
    "post_id",
    "comment_text",
    "comment_text_clean",
    "sentiment_label",
    "stance_label",
    "weight",
    "llm_confidence",
    "llm_comment_exp",
    "snapshot_week",
    "created_utc",
]

ARRAY_FIELDS = {
    "posts": {"embedding"},
    "comments": set(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report missing/null/empty field counts."
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--posts-only", action="store_true", help="Report only posts collection."
    )
    parser.add_argument(
        "--comments-only", action="store_true", help="Report only comments collection."
    )
    parser.add_argument(
        "--fields-posts",
        default="",
        help="Comma-separated override for posts fields.",
    )
    parser.add_argument(
        "--fields-comments",
        default="",
        help="Comma-separated override for comments fields.",
    )
    return parser.parse_args()


def _parse_fields(raw: str, fallback: List[str]) -> List[str]:
    if not raw:
        return fallback
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_group(fields: Iterable[str], array_fields: set[str]) -> Dict[str, object]:
    group: Dict[str, object] = {"_id": None, "total": {"$sum": 1}}
    for field in fields:
        field_ref = f"${field}"
        group[f"{field}__missing"] = {
            "$sum": {"$cond": [{"$eq": [{"$type": field_ref}, "missing"]}, 1, 0]}
        }
        group[f"{field}__null"] = {
            "$sum": {"$cond": [{"$eq": [field_ref, None]}, 1, 0]}
        }
        group[f"{field}__empty_str"] = {
            "$sum": {"$cond": [{"$eq": [field_ref, ""]}, 1, 0]}
        }
        if field in array_fields:
            group[f"{field}__empty_arr"] = {
                "$sum": {"$cond": [{"$eq": [field_ref, []]}, 1, 0]}
            }
    return group


def _print_report(
    name: str,
    fields: List[str],
    stats: Dict[str, int],
    array_fields: set[str],
) -> None:
    total = stats.get("total", 0)
    print(f"\n{name} collection")
    print(f"total_docs: {total}")
    for field in fields:
        missing = stats.get(f"{field}__missing", 0)
        nulls = stats.get(f"{field}__null", 0)
        empty_str = stats.get(f"{field}__empty_str", 0)
        empty_arr = stats.get(f"{field}__empty_arr", 0) if field in array_fields else 0
        empty_total = missing + nulls + empty_str + empty_arr
        print(
            f"{field}: missing={missing}, null={nulls}, empty_str={empty_str}, "
            f"empty_arr={empty_arr}, total_empty={empty_total}"
        )


def _run_report(
    store, collection_name: str, fields: List[str]
) -> Optional[Dict[str, int]]:
    if not fields:
        return None
    array_fields = ARRAY_FIELDS.get(collection_name, set())
    pipeline = [{"$group": _build_group(fields, array_fields)}]
    rows = list(store.db[collection_name].aggregate(pipeline))
    if not rows:
        return {"total": 0}
    return rows[0]


def main() -> None:
    args = parse_args()
    posts_fields = _parse_fields(args.fields_posts, POST_FIELDS)
    comments_fields = _parse_fields(args.fields_comments, COMMENT_FIELDS)

    if args.posts_only and args.comments_only:
        logger.warning("Both --posts-only and --comments-only set; defaulting to both.")
        args.posts_only = False
        args.comments_only = False

    store = connect_from_config(Path(args.config))
    try:
        if not args.comments_only:
            stats = _run_report(store, "posts", posts_fields)
            if stats:
                _print_report("posts", posts_fields, stats, ARRAY_FIELDS["posts"])
        if not args.posts_only:
            stats = _run_report(store, "comments", comments_fields)
            if stats:
                _print_report(
                    "comments", comments_fields, stats, ARRAY_FIELDS["comments"]
                )
    finally:
        store.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
