from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from db.store import CONFIG_PATH, connect_from_config  # noqa: E402
from reddit.reddit_cleaner import build_topic_text  # noqa: E402

logger = logging.getLogger(__name__)

DELETED_MARKERS = {"[deleted]", "[removed]"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export posts with comments missing stance/sentiment labels "
            "into JSONL for stance_sentiment_batch.py."
        )
    )
    parser.add_argument(
        "--config", default=str(CONFIG_PATH), help="Path to settings.yaml"
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "llm" / "stance_sentiment_missing.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--missing-mode",
        choices=["both", "either"],
        default="both",
        help="Match comments missing both labels or either label",
    )
    parser.add_argument(
        "--limit-posts",
        type=int,
        default=0,
        help="Max posts to export (0 = no limit)",
    )
    parser.add_argument(
        "--limit-comments",
        type=int,
        default=0,
        help="Max comments to export (0 = no limit)",
    )
    parser.add_argument(
        "--max-comments-per-post",
        type=int,
        default=0,
        help="Max comments per post (0 = no limit)",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include [deleted]/[removed] comments",
    )
    return parser.parse_args()


def build_missing_filter(mode: str) -> Dict[str, Any]:
    missing_sentiment = {
        "$or": [
            {"sentiment_label": {"$exists": False}},
            {"sentiment_label": None},
            {"sentiment_label": ""},
        ]
    }
    missing_stance = {
        "$or": [
            {"stance_label": {"$exists": False}},
            {"stance_label": None},
            {"stance_label": ""},
        ]
    }
    if mode == "either":
        return {"$or": [missing_sentiment, missing_stance]}
    return {"$and": [missing_sentiment, missing_stance]}


def get_comment_text(doc: Dict[str, Any]) -> str:
    for key in ("comment_text_clean", "comment_text", "body_clean", "body"):
        value = doc.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def is_deleted_comment(text: str) -> bool:
    return text.strip().lower() in DELETED_MARKERS


def get_post_text(doc: Dict[str, Any]) -> str:
    topic_text = str(doc.get("topic_text") or "").strip()
    if topic_text:
        return topic_text
    title = str(doc.get("title") or "")
    selftext = str(doc.get("selftext") or "")
    return build_topic_text(title, selftext, bool(selftext)).strip()


def iter_missing_comments(
    collection,
    *,
    missing_mode: str,
) -> Iterable[Dict[str, Any]]:
    missing_filter = build_missing_filter(missing_mode)
    text_or = [
        {"comment_text_clean": {"$exists": True, "$ne": ""}},
        {"comment_text": {"$exists": True, "$ne": ""}},
        {"body_clean": {"$exists": True, "$ne": ""}},
        {"body": {"$exists": True, "$ne": ""}},
    ]
    query = {
        "$and": [
            {"post_id": {"$exists": True, "$nin": [None, ""]}},
            missing_filter,
            {"$or": text_or},
        ]
    }
    projection = {
        "_id": 0,
        "comment_id": 1,
        "post_id": 1,
        "comment_text_clean": 1,
        "comment_text": 1,
        "body_clean": 1,
        "body": 1,
        "created_utc": 1,
    }
    cursor = collection.find(query, projection).sort(
        [("post_id", 1), ("created_utc", 1), ("comment_id", 1)]
    )
    return cursor


def fetch_post(posts, post_id: str) -> Optional[Dict[str, Any]]:
    if not post_id:
        return None
    return posts.find_one(
        {"post_id": post_id},
        {"_id": 0, "post_id": 1, "topic_text": 1, "title": 1, "selftext": 1},
    )


def export_missing() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limit_posts = int(args.limit_posts or 0)
    limit_comments = int(args.limit_comments or 0)
    max_comments_per_post = int(args.max_comments_per_post or 0)

    store = connect_from_config(Path(args.config))
    scanned_comments = 0
    exported_posts = 0
    exported_comments = 0
    accepted_comments = 0
    skipped_deleted = 0
    skipped_empty = 0
    skipped_missing_post = 0
    skipped_overflow = 0
    skipped_comment_id = 0

    current_post_id = ""
    current_post_text = ""
    current_comments: List[Dict[str, str]] = []
    skip_post = False

    def flush_current() -> None:
        nonlocal exported_posts, exported_comments, current_comments
        if not current_post_id or skip_post or not current_comments:
            return
        payload = {
            "post_id": current_post_id,
            "post_text": current_post_text,
            "comments": current_comments,
        }
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        exported_posts += 1
        exported_comments += len(current_comments)

    try:
        with output_path.open("w", encoding="utf-8") as fp:
            cursor = iter_missing_comments(
                store.comments,
                missing_mode=args.missing_mode,
            )
            for doc in cursor:
                scanned_comments += 1
                post_id = str(doc.get("post_id") or "").strip()
                if not post_id:
                    continue

                if post_id != current_post_id:
                    flush_current()
                    if limit_posts and exported_posts >= limit_posts:
                        current_post_id = ""
                        current_comments = []
                        break

                    current_post_id = post_id
                    current_comments = []
                    skip_post = False
                    post_doc = fetch_post(store.posts, post_id)
                    if not post_doc:
                        skip_post = True
                        skipped_missing_post += 1
                        continue
                    current_post_text = get_post_text(post_doc)
                    if not current_post_text:
                        skip_post = True
                        skipped_missing_post += 1
                        continue

                if skip_post:
                    continue
                if (
                    max_comments_per_post
                    and len(current_comments) >= max_comments_per_post
                ):
                    skipped_overflow += 1
                    continue

                comment_id = str(doc.get("comment_id") or "").strip()
                if not comment_id:
                    skipped_comment_id += 1
                    continue

                comment_text = get_comment_text(doc)
                if not comment_text:
                    skipped_empty += 1
                    continue
                if not args.include_deleted and is_deleted_comment(comment_text):
                    skipped_deleted += 1
                    continue

                current_comments.append(
                    {
                        "comment_id": comment_id,
                        "comment_text": comment_text,
                    }
                )
                accepted_comments += 1

                if limit_comments and accepted_comments >= limit_comments:
                    break

            flush_current()
    finally:
        store.close()

    logger.info(
        "Scanned %s comments, exported %s posts (%s comments).",
        scanned_comments,
        exported_posts,
        exported_comments,
    )
    logger.info(
        "Skipped: deleted=%s, empty=%s, missing_post=%s, overflow=%s, missing_comment_id=%s.",
        skipped_deleted,
        skipped_empty,
        skipped_missing_post,
        skipped_overflow,
        skipped_comment_id,
    )
    logger.info("Output written to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    export_missing()
